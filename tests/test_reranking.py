"""Tests for cross-encoder reranker."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models import Chunk, ChunkMetadata, ChunkingStrategy, RetrievalResult, RetrieverType
from src.reranking.cross_encoder import CrossEncoderReranker


def _make_result(content: str, score: float) -> RetrievalResult:
    """Helper to build a RetrievalResult."""
    return RetrievalResult(
        chunk=Chunk(
            content=content,
            metadata=ChunkMetadata(
                document_id=uuid4(),
                source="test.pdf",
                chunk_index=0,
                chunking_strategy=ChunkingStrategy.FIXED,
                chunk_size=512,
            ),
        ),
        score=score,
        retriever_type=RetrieverType.DENSE,
    )


class TestCrossEncoderReranker:
    """Test reranker with mocked cross-encoder model."""

    @patch("src.reranking.cross_encoder.CrossEncoder")
    def test_rerank_reorders_by_score(self, mock_ce_class):
        """Reranker should reorder results by cross-encoder scores."""
        mock_model = MagicMock()
        # Cross-encoder scores: third result is most relevant
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker(model_name="test-model")

        results = [
            _make_result("low relevance", 0.8),
            _make_result("high relevance", 0.3),
            _make_result("medium relevance", 0.5),
        ]

        reranked = reranker.rerank("test query", results, top_k=3)

        assert len(reranked) == 3
        assert reranked[0].chunk.content == "high relevance"
        assert reranked[0].score == 0.9
        assert reranked[1].chunk.content == "medium relevance"
        assert reranked[2].chunk.content == "low relevance"

    @patch("src.reranking.cross_encoder.CrossEncoder")
    def test_rerank_truncates_to_top_k(self, mock_ce_class):
        """Reranker should return only top_k results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.7]
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker(model_name="test-model")

        results = [_make_result(f"doc{i}", 0.5) for i in range(4)]
        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].score == 0.9

    @patch("src.reranking.cross_encoder.CrossEncoder")
    def test_rerank_empty_results(self, mock_ce_class):
        """Reranker should handle empty input."""
        mock_ce_class.return_value = MagicMock()
        reranker = CrossEncoderReranker(model_name="test-model")

        reranked = reranker.rerank("query", [], top_k=5)
        assert reranked == []

    @patch("src.reranking.cross_encoder.CrossEncoder")
    def test_rerank_passes_query_passage_pairs(self, mock_ce_class):
        """Reranker should pass (query, passage) pairs to the model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.8]
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker(model_name="test-model")

        results = [
            _make_result("passage one", 0.5),
            _make_result("passage two", 0.5),
        ]

        reranker.rerank("my query", results, top_k=2)

        # Verify the pairs passed to predict
        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [("my query", "passage one"), ("my query", "passage two")]
