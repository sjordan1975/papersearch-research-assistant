"""Answer generation — query + retrieved chunks → cited answer.

Takes a user query and retrieved chunks, builds a citation-aware prompt,
sends it to an LLM, and parses the response into a QAResponse with
structured citations.
"""

import re

from src.base.interfaces import BaseLLM
from src.models import Citation, Chunk, QAResponse, RetrievalResult


_SYSTEM_PROMPT = """You are a research assistant that answers questions based on provided source documents.

Rules:
- Only use information from the provided sources
- Cite sources using bracket notation [1], [2], etc.
- If the sources don't contain enough information, say so
- Be concise and accurate"""


def build_prompt(query: str, results: list[RetrievalResult]) -> str:
    """Build a citation-aware prompt from a query and retrieval results.

    Each chunk is numbered [1], [2], etc. so the LLM can reference them.
    """
    if not results:
        return (
            f"Question: {query}\n\n"
            "No source documents were provided. "
            "Answer based on your knowledge or state that you cannot answer."
        )

    sources = []
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        source = chunk.metadata.source
        sources.append(f"[{i}] (Source: {source})\n{chunk.content}")

    sources_text = "\n\n".join(sources)

    return (
        f"Sources:\n\n{sources_text}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using the sources above. "
        f"Cite your sources using [1], [2], etc."
    )


def _parse_citation_refs(answer: str) -> list[int]:
    """Extract citation reference numbers from an answer string.

    Matches [1], [2], etc. Returns unique, sorted 1-based indices.
    """
    refs = re.findall(r"\[(\d+)\]", answer)
    return sorted(set(int(r) for r in refs))


class AnswerGenerator:
    """Generate cited answers from retrieved chunks.

    Args:
        llm: An LLM client implementing BaseLLM.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, query: str, results: list[RetrievalResult]) -> QAResponse:
        """Generate a QAResponse with citations from retrieval results."""
        prompt = build_prompt(query, results)
        raw_answer = self.llm.generate(prompt, system_prompt=_SYSTEM_PROMPT)

        # Parse citation references from the answer
        cited_indices = _parse_citation_refs(raw_answer)

        # Build citation objects for referenced chunks
        citations = []
        for idx in cited_indices:
            if 1 <= idx <= len(results):
                chunk = results[idx - 1].chunk
                citations.append(Citation(
                    chunk_id=chunk.id,
                    source=chunk.metadata.source,
                    page_number=chunk.metadata.page_number,
                    text_snippet=chunk.content[:200],
                    relevance_score=results[idx - 1].score,
                ))

        return QAResponse(
            query=query,
            answer=raw_answer,
            citations=citations,
            chunks_used=[r.chunk for r in results],
        )
