"""Tests for LLM generation and answer pipeline.

TDD (partial): tests cover prompt construction, output parsing, and the
LLM client interface. Actual LLM calls are tested via a stub.

Testable:
  - Prompt template renders with chunks and query
  - Citations are built from chunks correctly
  - QAResponse is assembled with proper structure
  - LLM client implements BaseLLM

Not testable without API:
  - Quality of generated answers
  - Whether the LLM actually follows citation instructions
"""

import pytest

from src.base.interfaces import BaseLLM
from src.models import (
    Chunk,
    ChunkMetadata,
    ChunkingStrategy,
    Citation,
    QAResponse,
    RetrievalResult,
    RetrieverType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(content: str, index: int, source: str = "paper_001.pdf") -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id="00000000-0000-0000-0000-000000000001",
            source=source,
            chunk_index=index,
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
        ),
    )


def make_retrieval_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=make_chunk("Transformers use self-attention mechanisms.", 0),
            score=0.95,
            retriever_type=RetrieverType.DENSE,
        ),
        RetrievalResult(
            chunk=make_chunk("BERT is a bidirectional encoder.", 1),
            score=0.85,
            retriever_type=RetrieverType.DENSE,
        ),
        RetrievalResult(
            chunk=make_chunk("GPT uses autoregressive decoding.", 2, source="paper_002.pdf"),
            score=0.75,
            retriever_type=RetrieverType.DENSE,
        ),
    ]


class StubLLM(BaseLLM):
    """Returns a canned response that mimics citation format."""

    def __init__(self, response: str = "Transformers use attention [1]. BERT extends this [2]."):
        self._response = response
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0, metadata: dict | None = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_metadata = metadata
        return self._response


# ---------------------------------------------------------------------------
# LiteLLM client
# ---------------------------------------------------------------------------

class TestLiteLLMClient:

    def test_is_base_llm(self):
        from src.generation.llm import LiteLLMClient
        client = LiteLLMClient(model="gpt-4o-mini")
        assert isinstance(client, BaseLLM)

    def test_stores_model_name(self):
        from src.generation.llm import LiteLLMClient
        client = LiteLLMClient(model="claude-sonnet-4-20250514")
        assert client.model == "claude-sonnet-4-20250514"

    def test_passes_metadata_to_litellm(self, monkeypatch):
        """Metadata dict (trace context) should flow through to litellm.completion()."""
        import litellm as _litellm
        from src.generation.llm import LiteLLMClient

        captured_kwargs = {}

        def fake_completion(**kwargs):
            captured_kwargs.update(kwargs)
            # Return a minimal response object
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "fake answer"
            return resp

        monkeypatch.setattr(_litellm, "completion", fake_completion)
        client = LiteLLMClient(model="gpt-4o-mini")
        meta = {"trace_name": "qa_generation", "trace_id": "abc-123", "session_id": "sess-1"}
        client.generate("hello", metadata=meta)

        assert captured_kwargs["metadata"] == meta

    def test_no_metadata_omits_key(self, monkeypatch):
        """When metadata is None, don't send the key to litellm at all."""
        import litellm as _litellm
        from src.generation.llm import LiteLLMClient

        captured_kwargs = {}

        def fake_completion(**kwargs):
            captured_kwargs.update(kwargs)
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "fake answer"
            return resp

        monkeypatch.setattr(_litellm, "completion", fake_completion)
        client = LiteLLMClient(model="gpt-4o-mini")
        client.generate("hello")

        assert "metadata" not in captured_kwargs


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPromptBuilder:

    def test_builds_prompt_with_query(self):
        from src.generation.answer import build_prompt
        results = make_retrieval_results()
        prompt = build_prompt("What is attention?", results)
        assert "What is attention?" in prompt

    def test_includes_all_chunks_numbered(self):
        from src.generation.answer import build_prompt
        results = make_retrieval_results()
        prompt = build_prompt("test query", results)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt
        assert "Transformers use self-attention" in prompt
        assert "BERT is a bidirectional" in prompt
        assert "GPT uses autoregressive" in prompt

    def test_includes_source_info(self):
        from src.generation.answer import build_prompt
        results = make_retrieval_results()
        prompt = build_prompt("test query", results)
        assert "paper_001.pdf" in prompt
        assert "paper_002.pdf" in prompt

    def test_instructs_citation(self):
        """Prompt should tell the LLM to cite sources."""
        from src.generation.answer import build_prompt
        results = make_retrieval_results()
        prompt = build_prompt("test query", results)
        prompt_lower = prompt.lower()
        assert "cite" in prompt_lower or "citation" in prompt_lower or "[1]" in prompt_lower

    def test_empty_results_still_produces_prompt(self):
        from src.generation.answer import build_prompt
        prompt = build_prompt("test query", [])
        assert "test query" in prompt


# ---------------------------------------------------------------------------
# Answer generation (with stub LLM)
# ---------------------------------------------------------------------------

class TestAnswerGenerator:

    @pytest.fixture
    def generator(self):
        from src.generation.answer import AnswerGenerator
        return AnswerGenerator(llm=StubLLM())

    def test_returns_qa_response(self, generator):
        results = make_retrieval_results()
        response = generator.generate("What is attention?", results)
        assert isinstance(response, QAResponse)

    def test_response_has_query(self, generator):
        results = make_retrieval_results()
        response = generator.generate("What is attention?", results)
        assert response.query == "What is attention?"

    def test_response_has_answer(self, generator):
        results = make_retrieval_results()
        response = generator.generate("What is attention?", results)
        assert len(response.answer) > 0

    def test_response_has_chunks_used(self, generator):
        results = make_retrieval_results()
        response = generator.generate("What is attention?", results)
        assert len(response.chunks_used) == len(results)

    def test_response_has_citations(self, generator):
        results = make_retrieval_results()
        response = generator.generate("What is attention?", results)
        assert len(response.citations) > 0
        for citation in response.citations:
            assert isinstance(citation, Citation)
            assert citation.source  # non-empty

    def test_citations_reference_valid_chunks(self, generator):
        results = make_retrieval_results()
        response = generator.generate("What is attention?", results)
        chunk_ids = {c.id for r in results for c in [r.chunk]}
        for citation in response.citations:
            assert citation.chunk_id in chunk_ids

    def test_no_results_produces_answer(self):
        """Even with no retrieval results, should produce an answer (possibly declining)."""
        from src.generation.answer import AnswerGenerator
        llm = StubLLM(response="I don't have enough information to answer this question.")
        generator = AnswerGenerator(llm=llm)
        response = generator.generate("What is attention?", [])
        assert isinstance(response, QAResponse)
        assert response.citations == []
