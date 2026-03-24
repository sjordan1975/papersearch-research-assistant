"""LLM-as-Judge — score generated answers on quality dimensions.

Uses an LLM to evaluate QAResponse quality across four dimensions:
- Relevance: Does the answer address the question?
- Accuracy: Is the information factually correct given the sources?
- Completeness: Does it cover all relevant aspects?
- Citation quality: Are sources cited appropriately?

Each dimension is scored 1–5. The LLM outputs JSON that we parse
into JudgeScores.
"""

import json
import re

from src.base.interfaces import BaseLLM
from src.models import JudgeScores, QAResponse


_SYSTEM_PROMPT = """You are an expert evaluator of question-answering systems.

You will evaluate an answer based on a question and source documents.
Score each dimension from 1 (poor) to 5 (excellent).

Output ONLY valid JSON with these exact keys:
{"relevance": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "citation_quality": <1-5>}"""


def build_judge_prompt(qa_response: QAResponse) -> str:
    """Build a prompt for the LLM judge.

    Includes the query, generated answer, and source chunks for context.
    """
    sources_text = "\n\n".join(
        f"[{i+1}] {chunk.content}" for i, chunk in enumerate(qa_response.chunks_used)
    )

    return f"""Evaluate the following answer.

## Question
{qa_response.query}

## Sources Provided
{sources_text}

## Generated Answer
{qa_response.answer}

## Evaluation Criteria

Score each dimension from 1 (poor) to 5 (excellent):

1. **Relevance**: Does the answer directly address the question asked?
2. **Accuracy**: Is the information factually correct based on the sources?
3. **Completeness**: Does the answer cover all relevant aspects from the sources?
4. **Citation Quality**: Are citations used appropriately to support claims?

Output your scores as JSON:
{{"relevance": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "citation_quality": <1-5>}}"""


def parse_judge_response(response: str) -> JudgeScores:
    """Parse LLM response into JudgeScores.

    Handles JSON wrapped in markdown code blocks or surrounding text.
    Raises ValueError if parsing fails or scores are invalid.
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\}", response)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"Could not parse JSON from response: {response[:100]}")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}")

    # Validate with Pydantic (raises ValidationError if invalid)
    try:
        return JudgeScores(**data)
    except Exception as e:
        raise ValueError(f"Invalid scores: {e}")


class AnswerJudge:
    """Evaluate QA responses using an LLM judge.

    Args:
        llm: An LLM client implementing BaseLLM. Should use LiteLLMClient
            in production for Langfuse tracing.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def judge(self, qa_response: QAResponse) -> JudgeScores:
        """Score a single QA response."""
        prompt = build_judge_prompt(qa_response)
        response = self.llm.generate(
            prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return parse_judge_response(response)

    def judge_batch(self, qa_responses: list[QAResponse]) -> list[JudgeScores]:
        """Score multiple QA responses."""
        return [self.judge(qa) for qa in qa_responses]


def aggregate_judge_scores(scores: list[JudgeScores]) -> JudgeScores:
    """Compute average scores across multiple JudgeScores.

    Returns a JudgeScores with averaged values for each dimension.
    """
    if not scores:
        raise ValueError("Cannot aggregate empty list of scores")

    n = len(scores)
    return JudgeScores(
        relevance=sum(s.relevance for s in scores) / n,
        accuracy=sum(s.accuracy for s in scores) / n,
        completeness=sum(s.completeness for s in scores) / n,
        citation_quality=sum(s.citation_quality for s in scores) / n,
    )
