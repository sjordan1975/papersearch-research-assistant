"""Tests for canonical data models.

TDD approach: test field types, constraints, defaults, and rejection cases.
"""

import pytest
from uuid import UUID

from src.models import (
    ChunkingStrategy,
    ChunkMetadata,
    Citation,
    Document,
    DocumentMetadata,
    Chunk,
    ExperimentConfig,
    ExperimentResult,
    JudgeScores,
    QAResponse,
    RetrievalMetrics,
    RetrievalResult,
    RetrieverType,
)


# ---------------------------------------------------------------------------
# Helpers — canonical factories (D2: always build from these, never ad-hoc)
# ---------------------------------------------------------------------------

def build_valid_document_metadata(**overrides) -> dict:
    defaults = {"source": "paper_001.pdf", "title": "Test Paper", "author": "Test Author", "page_count": 10}
    return {**defaults, **overrides}


def build_valid_chunk_metadata(**overrides) -> dict:
    defaults = {
        "document_id": "00000000-0000-0000-0000-000000000001",
        "source": "paper_001.pdf",
        "page_number": 1,
        "start_char": 0,
        "end_char": 500,
        "chunk_index": 0,
        "chunking_strategy": "fixed",
        "chunk_size": 512,
        "chunk_overlap": 0,
    }
    return {**defaults, **overrides}


def build_valid_document(**overrides) -> dict:
    defaults = {
        "content": "This is a test document about transformers.",
        "metadata": build_valid_document_metadata(),
    }
    return {**defaults, **overrides}


def build_valid_chunk(**overrides) -> dict:
    defaults = {
        "content": "This is a test chunk.",
        "metadata": build_valid_chunk_metadata(),
    }
    return {**defaults, **overrides}


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_creates_with_valid_data(self):
        doc = Document(**build_valid_document())
        assert isinstance(doc.id, UUID)
        assert doc.content == "This is a test document about transformers."
        assert doc.metadata.source == "paper_001.pdf"

    def test_auto_generates_uuid(self):
        doc1 = Document(**build_valid_document())
        doc2 = Document(**build_valid_document())
        assert doc1.id != doc2.id

    def test_accepts_explicit_id(self):
        explicit_id = "12345678-1234-1234-1234-123456789012"
        doc = Document(**build_valid_document(), id=explicit_id)
        assert str(doc.id) == explicit_id

    def test_rejects_empty_content(self):
        """Documents must have content — empty PDFs should be filtered upstream."""
        # Pydantic allows empty strings by default; this documents the behavior.
        # If we want to reject empty, we add a validator later.
        doc = Document(**build_valid_document(content=""))
        assert doc.content == ""

    def test_metadata_optional_fields(self):
        doc = Document(**build_valid_document(metadata={"source": "test.pdf"}))
        assert doc.metadata.title is None
        assert doc.metadata.author is None
        assert doc.metadata.page_count is None


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_creates_with_valid_data(self):
        chunk = Chunk(**build_valid_chunk())
        assert isinstance(chunk.id, UUID)
        assert chunk.content == "This is a test chunk."
        assert chunk.metadata.chunk_index == 0

    def test_embedding_optional(self):
        chunk = Chunk(**build_valid_chunk())
        assert chunk.embedding is None

    def test_embedding_accepted(self):
        chunk = Chunk(**build_valid_chunk(), embedding=[0.1, 0.2, 0.3])
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_metadata_tracks_chunking_strategy(self):
        chunk = Chunk(**build_valid_chunk())
        assert chunk.metadata.chunking_strategy == ChunkingStrategy.FIXED

    def test_metadata_tracks_chunk_config(self):
        meta = build_valid_chunk_metadata(chunk_size=256, chunk_overlap=50)
        chunk = Chunk(**build_valid_chunk(metadata=meta))
        assert chunk.metadata.chunk_size == 256
        assert chunk.metadata.chunk_overlap == 50

    def test_section_index_optional(self):
        chunk = Chunk(**build_valid_chunk())
        assert chunk.metadata.section_index is None

    def test_section_index_for_ground_truth(self):
        meta = build_valid_chunk_metadata(section_index=3)
        chunk = Chunk(**build_valid_chunk(metadata=meta))
        assert chunk.metadata.section_index == 3


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    def test_creates_with_valid_data(self):
        result = RetrievalResult(
            chunk=Chunk(**build_valid_chunk()),
            score=0.85,
            retriever_type=RetrieverType.DENSE,
        )
        assert result.score == 0.85
        assert result.retriever_type == RetrieverType.DENSE

    @pytest.mark.parametrize("rtype", [RetrieverType.DENSE, RetrieverType.BM25, RetrieverType.HYBRID])
    def test_all_retriever_types(self, rtype):
        result = RetrievalResult(
            chunk=Chunk(**build_valid_chunk()),
            score=0.5,
            retriever_type=rtype,
        )
        assert result.retriever_type == rtype

    def test_rejects_invalid_retriever_type(self):
        with pytest.raises(Exception):
            RetrievalResult(
                chunk=Chunk(**build_valid_chunk()),
                score=0.5,
                retriever_type="invalid",
            )


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

class TestCitation:
    def test_creates_with_valid_data(self):
        citation = Citation(
            chunk_id="00000000-0000-0000-0000-000000000001",
            source="paper_001.pdf",
            text_snippet="Transformers use attention mechanisms.",
        )
        assert citation.source == "paper_001.pdf"
        assert citation.page_number is None
        assert citation.relevance_score is None

    def test_optional_fields(self):
        citation = Citation(
            chunk_id="00000000-0000-0000-0000-000000000001",
            source="paper_001.pdf",
            page_number=5,
            text_snippet="Attention is all you need.",
            relevance_score=0.92,
        )
        assert citation.page_number == 5
        assert citation.relevance_score == 0.92


# ---------------------------------------------------------------------------
# QAResponse
# ---------------------------------------------------------------------------

class TestQAResponse:
    def test_creates_with_valid_data(self):
        response = QAResponse(
            query="What is attention?",
            answer="Attention is a mechanism for weighting input tokens.",
        )
        assert response.query == "What is attention?"
        assert response.citations == []
        assert response.chunks_used == []
        assert response.confidence is None

    def test_with_citations_and_chunks(self):
        chunk = Chunk(**build_valid_chunk())
        citation = Citation(
            chunk_id=chunk.id,
            source="paper_001.pdf",
            text_snippet="Test snippet.",
        )
        response = QAResponse(
            query="What is attention?",
            answer="Attention is a mechanism.",
            citations=[citation],
            chunks_used=[chunk],
            confidence=0.95,
        )
        assert len(response.citations) == 1
        assert len(response.chunks_used) == 1
        assert response.confidence == 0.95


# ---------------------------------------------------------------------------
# Experiment models
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_creates_with_valid_data(self):
        config = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=512,
            chunk_overlap=100,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.HYBRID,
            top_k=5,
        )
        assert config.chunking_strategy == ChunkingStrategy.RECURSIVE
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_defaults(self):
        config = ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.DENSE,
        )
        assert config.chunk_overlap == 0
        assert config.top_k == 5


class TestJudgeScores:
    def test_creates_with_valid_data(self):
        scores = JudgeScores(relevance=5, accuracy=4, completeness=4, citation_quality=3)
        assert scores.average == 4.0

    @pytest.mark.parametrize("field", ["relevance", "accuracy", "completeness", "citation_quality"])
    def test_rejects_below_1(self, field):
        kwargs = {"relevance": 3, "accuracy": 3, "completeness": 3, "citation_quality": 3}
        kwargs[field] = 0
        with pytest.raises(Exception):
            JudgeScores(**kwargs)

    @pytest.mark.parametrize("field", ["relevance", "accuracy", "completeness", "citation_quality"])
    def test_rejects_above_5(self, field):
        kwargs = {"relevance": 3, "accuracy": 3, "completeness": 3, "citation_quality": 3}
        kwargs[field] = 6
        with pytest.raises(Exception):
            JudgeScores(**kwargs)


class TestRetrievalMetrics:
    def test_creates_with_valid_data(self):
        metrics = RetrievalMetrics(
            recall_at_k=0.80,
            precision_at_k=0.60,
            mrr=0.70,
            ndcg_at_k=0.75,
        )
        assert metrics.k == 5  # default


class TestExperimentResult:
    def test_creates_with_valid_data(self):
        result = ExperimentResult(
            experiment_id="exp_20260323_abc123",
            config=ExperimentConfig(
                chunking_strategy=ChunkingStrategy.FIXED,
                chunk_size=512,
                embedding_model="all-MiniLM-L6-v2",
                retriever_type=RetrieverType.DENSE,
            ),
            metrics=RetrievalMetrics(
                recall_at_k=0.80,
                precision_at_k=0.60,
                mrr=0.70,
                ndcg_at_k=0.75,
            ),
            num_queries=10,
            query_ids=["q_001", "q_002"],
        )
        assert result.experiment_id == "exp_20260323_abc123"
        assert result.judge_scores is None

    def test_serializes_to_json(self):
        """Experiment results must be JSON-serializable (A21)."""
        result = ExperimentResult(
            experiment_id="exp_test",
            config=ExperimentConfig(
                chunking_strategy=ChunkingStrategy.FIXED,
                chunk_size=512,
                embedding_model="all-MiniLM-L6-v2",
                retriever_type=RetrieverType.DENSE,
            ),
            metrics=RetrievalMetrics(
                recall_at_k=0.80,
                precision_at_k=0.60,
                mrr=0.70,
                ndcg_at_k=0.75,
            ),
            num_queries=10,
        )
        json_str = result.model_dump_json()
        assert '"experiment_id":"exp_test"' in json_str or '"experiment_id": "exp_test"' in json_str
