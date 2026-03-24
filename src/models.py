"""Canonical data models for the PaperSearch RAG pipeline.

This is the single source of truth for all domain objects.
Every module imports from here — no ad-hoc dicts or dataclasses.
"""

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RetrieverType(str, Enum):
    """Which retrieval method produced this result."""
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


# ---------------------------------------------------------------------------
# Metadata models (nested inside Document / Chunk)
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Metadata extracted from a PDF during loading."""
    source: str  # filename or path
    title: str | None = None
    author: str | None = None
    page_count: int | None = None


class ChunkMetadata(BaseModel):
    """Provenance metadata for a chunk — where it came from."""
    document_id: UUID
    source: str  # filename
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    chunk_index: int  # position within the document's chunk list
    chunking_strategy: ChunkingStrategy
    chunk_size: int  # configured chunk size (tokens or chars)
    chunk_overlap: int = 0  # configured overlap
    section_index: int | None = None  # for ground truth alignment (D2 cache key)


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """A loaded PDF document before chunking."""
    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    """A chunk of text from a document, optionally with its embedding."""
    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class RetrievalResult(BaseModel):
    """A single retrieval hit: chunk + score + which retriever found it."""
    chunk: Chunk
    score: float
    retriever_type: RetrieverType


class Citation(BaseModel):
    """A citation extracted from an LLM-generated answer."""
    chunk_id: UUID
    source: str  # filename
    page_number: int | None = None
    text_snippet: str
    relevance_score: float | None = None


class QAResponse(BaseModel):
    """A complete question-answer response with citations."""
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks_used: list[Chunk] = Field(default_factory=list)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Experiment / evaluation models
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run."""
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int = 0
    embedding_model: str  # e.g. "all-MiniLM-L6-v2"
    retriever_type: RetrieverType
    top_k: int = 5
    hybrid_alpha: float = 0.5  # dense vs BM25 weight for hybrid retrieval
    reranker: str | None = None  # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrievalMetrics(BaseModel):
    """IR metrics for a single experiment."""
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int = 5


class JudgeScores(BaseModel):
    """LLM-as-Judge scores for a single answer."""
    relevance: float = Field(ge=1, le=5)
    accuracy: float = Field(ge=1, le=5)
    completeness: float = Field(ge=1, le=5)
    citation_quality: float = Field(ge=1, le=5)

    @property
    def average(self) -> float:
        return (self.relevance + self.accuracy + self.completeness + self.citation_quality) / 4


class ExperimentResult(BaseModel):
    """Full result of a single experiment run, saved as JSON."""
    experiment_id: str
    config: ExperimentConfig
    metrics: RetrievalMetrics
    judge_scores: JudgeScores | None = None
    num_queries: int
    query_ids: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None
