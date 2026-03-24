"""Cross-encoder reranker using SentenceTransformers.

A cross-encoder scores (query, passage) pairs jointly — unlike a bi-encoder
which embeds them independently. This is more accurate but much slower,
so it's applied as a second stage on already-retrieved candidates.

Analogy for the JS/TS dev: a bi-encoder is like indexing documents once
and comparing hashes at query time (fast, approximate). A cross-encoder
is like running a full comparison function on each candidate (slow, precise).
"""

import logging

from sentence_transformers import CrossEncoder

from src.base.interfaces import BaseReranker
from src.models import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Rerank retrieval results using a cross-encoder model.

    Args:
        model_name: HuggingFace model ID. Default is a lightweight
            MS MARCO model (~80MB) fine-tuned for passage reranking.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.info("Loaded cross-encoder: %s", model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank results by cross-encoder relevance score.

        Args:
            query: The search query.
            results: Retrieval results from first-stage retriever.
            top_k: Number of results to return after reranking.

        Returns:
            Top-k results reordered by cross-encoder score.
        """
        if not results:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, r.chunk.content) for r in results]

        # Score all pairs at once (batched inference)
        scores = self.model.predict(pairs)

        # Replace scores and sort
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                retriever_type=result.retriever_type,
            ))

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
