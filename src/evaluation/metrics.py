"""IR metrics for retrieval evaluation.

Pure functions — no I/O, no state. Each takes a list[bool] where
retrieved_relevant[i] is True if the chunk at rank i is relevant.

Metrics:
- Precision@K: fraction of top-K results that are relevant
- Recall@K: fraction of relevant items found in top-K
- MRR: reciprocal rank of first relevant result
- NDCG@K: position-weighted relevance (rewards hits at higher ranks)
"""

import math

from src.models import RetrievalMetrics


def precision_at_k(retrieved_relevant: list[bool], k: int) -> float:
    """Precision@K = (relevant in top-K) / K."""
    top_k = retrieved_relevant[:k]
    return sum(top_k) / k


def recall_at_k(
    retrieved_relevant: list[bool], k: int, total_relevant: int = 1
) -> float:
    """Recall@K = (relevant found in top-K) / total_relevant.

    In this dataset total_relevant=1 per query (one qrel section),
    so recall is binary: 0.0 or 1.0.
    """
    if total_relevant == 0:
        return 0.0
    top_k = retrieved_relevant[:k]
    return sum(top_k) / total_relevant


def mrr(retrieved_relevant: list[bool]) -> float:
    """Mean Reciprocal Rank = 1 / rank_of_first_relevant (0 if none)."""
    for i, is_rel in enumerate(retrieved_relevant):
        if is_rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_relevant: list[bool], k: int) -> float:
    """NDCG@K with binary relevance.

    DCG@K  = sum( rel_i / log2(i + 2) ) for i in [0, k)
    IDCG@K = sum( 1 / log2(i + 2) )     for the top-n_relevant positions

    With binary relevance, rel_i is 0 or 1.
    """
    top_k = retrieved_relevant[:k]
    n_relevant = sum(top_k)
    if n_relevant == 0:
        return 0.0

    # Actual DCG
    dcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(top_k) if rel
    )

    # Ideal DCG: all relevant items at top ranks
    total_relevant = sum(retrieved_relevant)  # across full list
    n_ideal = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))

    return dcg / idcg


def compute_retrieval_metrics(
    query_results: dict[str, list[bool]],
    k: int = 5,
) -> RetrievalMetrics:
    """Macro-average all metrics across queries.

    Args:
        query_results: {query_id: [is_relevant_at_rank_0, ...]}
        k: cutoff for @K metrics.

    Returns:
        RetrievalMetrics with averaged scores.
    """
    if not query_results:
        raise ValueError("No queries to evaluate")

    n = len(query_results)
    total_p = 0.0
    total_r = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0

    for labels in query_results.values():
        total_p += precision_at_k(labels, k)
        total_r += recall_at_k(labels, k, total_relevant=1)
        total_mrr += mrr(labels)
        total_ndcg += ndcg_at_k(labels, k)

    return RetrievalMetrics(
        precision_at_k=total_p / n,
        recall_at_k=total_r / n,
        mrr=total_mrr / n,
        ndcg_at_k=total_ndcg / n,
        k=k,
    )
