"""Hybrid retriever — fuse dense and BM25 scores.

Combines dense (semantic) and BM25 (keyword) retrieval using alpha-weighted
score fusion:

    hybrid_score = alpha * dense_normalized + (1 - alpha) * bm25_normalized

Both score sets are min-max normalized to [0, 1] before fusion, so they're
on comparable scales.

D4 pitfall #5: if all scores from one retriever are identical (e.g., all
BM25 scores are 0 for a gibberish query), min-max normalization would
divide by zero. We guard against this by returning 0.0 in that case.
"""

from src.base.interfaces import BaseRetriever
from src.models import Chunk, RetrievalResult, RetrieverType


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] range. Returns 0.0 if all scores are equal."""
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s:
        return {k: 0.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}


class HybridRetriever(BaseRetriever):
    """Combine dense and BM25 retrieval with score fusion.

    Args:
        dense_retriever: A dense (embedding-based) retriever.
        bm25_retriever: A BM25 (keyword-based) retriever.
        alpha: Weight for dense scores. 1.0 = pure dense, 0.0 = pure BM25.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        alpha: float = 0.5,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        # Fetch more candidates than needed — fusion may reorder
        fetch_k = top_k * 2

        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_k)

        # Collect all unique chunks and their raw scores
        chunk_map: dict[str, Chunk] = {}
        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        for r in dense_results:
            cid = str(r.chunk.id)
            chunk_map[cid] = r.chunk
            dense_scores[cid] = r.score

        for r in bm25_results:
            cid = str(r.chunk.id)
            chunk_map[cid] = r.chunk
            bm25_scores[cid] = r.score

        # Normalize both score sets to [0, 1]
        dense_norm = _min_max_normalize(dense_scores)
        bm25_norm = _min_max_normalize(bm25_scores)

        # Fuse scores for all candidate chunks
        all_ids = set(dense_norm.keys()) | set(bm25_norm.keys())
        fused: dict[str, float] = {}
        for cid in all_ids:
            d = dense_norm.get(cid, 0.0)
            b = bm25_norm.get(cid, 0.0)
            fused[cid] = self.alpha * d + (1 - self.alpha) * b

        # Sort by fused score, take top_k
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=score,
                retriever_type=RetrieverType.HYBRID,
            )
            for cid, score in ranked
        ]
