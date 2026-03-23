"""BM25 retriever — sparse keyword-based retrieval.

BM25 (Best Matching 25) ranks documents by term frequency and inverse
document frequency. Unlike dense retrieval, it works purely on keyword
overlap — no embeddings needed.

Useful for queries with specific technical terms that dense models might
not capture well (e.g., model names, acronyms, chemical formulas).
"""

from rank_bm25 import BM25Okapi

from src.base.interfaces import BaseRetriever
from src.models import Chunk, RetrievalResult, RetrieverType


class BM25Retriever(BaseRetriever):
    """Retrieve chunks using BM25 keyword matching.

    Args:
        chunks: The corpus of chunks to search over.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Tokenize: simple whitespace + lowercase
        tokenized = [c.content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        tokens = query.lower().split()

        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Rank by score, take top_k
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            # D4 pitfall #4: skip zero-score results (no keyword match)
            if score <= 0:
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=score,
                retriever_type=RetrieverType.BM25,
            ))

        return results
