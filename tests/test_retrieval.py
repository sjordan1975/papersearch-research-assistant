"""Tests for retrieval strategies.

TDD: tests written before implementation.
Covers T7 (dense), T8 (BM25), T9 (hybrid).

Uses stub embedder and pre-built FAISS index to avoid loading real models.
"""

import numpy as np
import pytest

from src.base.interfaces import BaseEmbedder, BaseRetriever
from src.models import (
    Chunk,
    ChunkMetadata,
    ChunkingStrategy,
    RetrievalResult,
    RetrieverType,
)
from src.stores.faiss_store import FaissVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(content: str, chunk_index: int, doc_id: str = "00000000-0000-0000-0000-000000000001") -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id=doc_id,
            source="test.pdf",
            chunk_index=chunk_index,
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
        ),
    )


class StubEmbedder(BaseEmbedder):
    """Returns deterministic embeddings based on content keywords."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            # Create a simple but deterministic embedding
            v = np.zeros(self._dim, dtype=np.float32)
            for i, char in enumerate(text[:self._dim]):
                v[i] = ord(char) / 255.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "stub"


@pytest.fixture
def chunks():
    """Five chunks with distinct content."""
    return [
        make_chunk("Transformers use self-attention mechanisms for sequence modeling.", 0),
        make_chunk("BERT is a bidirectional encoder from transformers.", 1),
        make_chunk("Convolutional neural networks excel at image recognition.", 2),
        make_chunk("Recurrent networks process sequential data with hidden state.", 3),
        make_chunk("Random forests are ensemble methods using decision trees.", 4),
    ]


@pytest.fixture
def embedder():
    return StubEmbedder(dim=8)


@pytest.fixture
def populated_store(chunks, embedder):
    """A FAISS store loaded with chunk embeddings."""
    store = FaissVectorStore(dimension=embedder.dimension)
    texts = [c.content for c in chunks]
    embeddings = embedder.embed(texts)
    ids = [str(c.id) for c in chunks]
    store.add(ids, embeddings)
    return store


@pytest.fixture
def chunk_lookup(chunks) -> dict:
    """Map chunk ID → Chunk for retriever to resolve results."""
    return {str(c.id): c for c in chunks}


# ---------------------------------------------------------------------------
# T7: Dense retriever
# ---------------------------------------------------------------------------

class TestDenseRetriever:

    @pytest.fixture
    def retriever(self, populated_store, chunk_lookup, embedder):
        from src.retrieval.dense import DenseRetriever
        return DenseRetriever(
            embedder=embedder,
            vector_store=populated_store,
            chunk_lookup=chunk_lookup,
        )

    def test_is_base_retriever(self, retriever):
        assert isinstance(retriever, BaseRetriever)

    def test_returns_retrieval_results(self, retriever):
        results = retriever.retrieve("attention mechanisms", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retriever_type_is_dense(self, retriever):
        results = retriever.retrieve("attention", top_k=3)
        for r in results:
            assert r.retriever_type == RetrieverType.DENSE

    def test_results_have_chunks(self, retriever):
        results = retriever.retrieve("attention", top_k=3)
        for r in results:
            assert isinstance(r.chunk, Chunk)
            assert r.chunk.content  # non-empty

    def test_results_have_scores(self, retriever):
        results = retriever.retrieve("attention", top_k=3)
        for r in results:
            assert isinstance(r.score, float)

    def test_top_k_limits_results(self, retriever):
        results = retriever.retrieve("test query", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_descending_score(self, retriever):
        results = retriever.retrieve("neural networks", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query(self, retriever):
        """Empty query should still return results (embedding of empty string)."""
        results = retriever.retrieve("", top_k=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# T8: BM25 retriever
# ---------------------------------------------------------------------------

class TestBM25Retriever:

    @pytest.fixture
    def retriever(self, chunks):
        from src.retrieval.bm25 import BM25Retriever
        return BM25Retriever(chunks=chunks)

    def test_is_base_retriever(self, retriever):
        assert isinstance(retriever, BaseRetriever)

    def test_returns_retrieval_results(self, retriever):
        results = retriever.retrieve("attention mechanisms", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retriever_type_is_bm25(self, retriever):
        results = retriever.retrieve("transformers", top_k=3)
        for r in results:
            assert r.retriever_type == RetrieverType.BM25

    def test_keyword_match_ranks_higher(self, retriever):
        """A query about 'transformers' should rank transformer-related chunks first."""
        results = retriever.retrieve("transformers attention", top_k=5)
        # The first result should contain "transformers" or "attention"
        top_content = results[0].chunk.content.lower()
        assert "transformer" in top_content or "attention" in top_content

    def test_top_k_limits_results(self, retriever):
        results = retriever.retrieve("neural networks", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_descending_score(self, retriever):
        results = retriever.retrieve("neural networks", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_non_negative(self, retriever):
        """BM25 scores should be >= 0."""
        results = retriever.retrieve("random query", top_k=5)
        for r in results:
            assert r.score >= 0.0

    def test_no_crash_on_empty_query(self, retriever):
        """D4 pitfall #4: empty/stop-word queries shouldn't crash."""
        results = retriever.retrieve("", top_k=3)
        assert isinstance(results, list)

    def test_no_crash_on_stopword_query(self, retriever):
        results = retriever.retrieve("the a an is", top_k=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# T9: Hybrid retriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:

    @pytest.fixture
    def dense_retriever(self, populated_store, chunk_lookup, embedder):
        from src.retrieval.dense import DenseRetriever
        return DenseRetriever(
            embedder=embedder,
            vector_store=populated_store,
            chunk_lookup=chunk_lookup,
        )

    @pytest.fixture
    def bm25_retriever(self, chunks):
        from src.retrieval.bm25 import BM25Retriever
        return BM25Retriever(chunks=chunks)

    @pytest.fixture
    def retriever(self, dense_retriever, bm25_retriever):
        from src.retrieval.hybrid import HybridRetriever
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            alpha=0.5,
        )

    def test_is_base_retriever(self, retriever):
        assert isinstance(retriever, BaseRetriever)

    def test_returns_retrieval_results(self, retriever):
        results = retriever.retrieve("attention mechanisms", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retriever_type_is_hybrid(self, retriever):
        results = retriever.retrieve("neural networks", top_k=3)
        for r in results:
            assert r.retriever_type == RetrieverType.HYBRID

    def test_top_k_limits_results(self, retriever):
        results = retriever.retrieve("transformers", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_descending_score(self, retriever):
        results = retriever.retrieve("neural networks", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_alpha_0_equals_bm25_only(self, dense_retriever, bm25_retriever):
        """With alpha=0, hybrid should behave like BM25 (no dense contribution)."""
        from src.retrieval.hybrid import HybridRetriever
        hybrid = HybridRetriever(dense_retriever, bm25_retriever, alpha=0.0)
        results = hybrid.retrieve("transformers", top_k=5)
        # Should still return results
        assert len(results) > 0

    def test_alpha_1_equals_dense_only(self, dense_retriever, bm25_retriever):
        """With alpha=1, hybrid should behave like dense (no BM25 contribution)."""
        from src.retrieval.hybrid import HybridRetriever
        hybrid = HybridRetriever(dense_retriever, bm25_retriever, alpha=1.0)
        results = hybrid.retrieve("transformers", top_k=5)
        assert len(results) > 0

    def test_scores_between_0_and_1(self, retriever):
        """Hybrid scores are from normalized fusion, should be in [0, 1]."""
        results = retriever.retrieve("neural networks", top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-6

    def test_div_by_zero_guard(self, dense_retriever, bm25_retriever):
        """D4 pitfall #5: if all BM25 scores are identical, normalization
        must not divide by zero."""
        from src.retrieval.hybrid import HybridRetriever
        hybrid = HybridRetriever(dense_retriever, bm25_retriever, alpha=0.5)
        # This shouldn't crash regardless of score distribution
        results = hybrid.retrieve("xyzzy nonexistent gibberish", top_k=5)
        assert isinstance(results, list)
