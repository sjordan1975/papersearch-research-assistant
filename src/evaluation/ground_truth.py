"""Ground truth alignment for retrieval evaluation.

Maps retrieved chunks back to qrels (query relevance labels) to determine
whether each retrieval hit is correct.

The alignment problem: qrels label relevance at the section level
(doc_id + section_id), but the retriever returns chunks. A chunk is a
"hit" if it came from the relevant section — determined by matching
chunk.metadata.section_index against the qrel's section_id.

D4 pitfall #6: chunks with section_index=None are logged and treated
as non-relevant (they can't be aligned).
"""

import json
import logging
from pathlib import Path

from src.models import Chunk, RetrievalMetrics, RetrievalResult
from src.evaluation.metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)


class GroundTruthAligner:
    """Maps retrieval results to ground truth relevance labels.

    Args:
        qrels: {query_id: {"doc_id": str, "section_id": int}}
        queries: {query_id: {"query": str, "type": str, "source": str}}
    """

    def __init__(
        self,
        qrels: dict[str, dict],
        queries: dict[str, dict],
    ) -> None:
        self.qrels = qrels
        self.queries = queries

    @classmethod
    def from_files(cls, qrels_path: str, queries_path: str) -> "GroundTruthAligner":
        """Load qrels and queries from JSON files."""
        with open(qrels_path) as f:
            qrels = json.load(f)
        with open(queries_path) as f:
            queries = json.load(f)
        return cls(qrels=qrels, queries=queries)

    @staticmethod
    def _doc_id_from_source(source: str) -> str:
        """Extract doc_id from a chunk's source filename.

        source is stored as "{doc_id}.pdf" by the loader.
        """
        return Path(source).stem

    def is_relevant(self, query_id: str, chunk: Chunk) -> bool:
        """Check if a chunk is relevant to the given query.

        A chunk is relevant if:
        1. The query exists in qrels
        2. The chunk's doc matches the qrel's doc_id
        3. The chunk's section_index matches the qrel's section_id
        """
        qrel = self.qrels.get(query_id)
        if qrel is None:
            return False

        if chunk.metadata.section_index is None:
            logger.warning(
                "Chunk %s has no section_index — cannot align to ground truth",
                chunk.id,
            )
            return False

        doc_id = self._doc_id_from_source(chunk.metadata.source)
        return (
            doc_id == qrel["doc_id"]
            and chunk.metadata.section_index == qrel["section_id"]
        )

    def build_relevance_labels(
        self,
        query_id: str,
        results: list[RetrievalResult],
    ) -> list[bool]:
        """Convert a ranked result list to a bool list for metric computation."""
        return [self.is_relevant(query_id, r.chunk) for r in results]

    def evaluate(
        self,
        query_results: dict[str, list[RetrievalResult]],
        k: int = 5,
    ) -> RetrievalMetrics:
        """End-to-end: from query->results, compute aggregate IR metrics.

        Skips queries not in qrels (no ground truth available).
        """
        labels_by_query: dict[str, list[bool]] = {}
        for query_id, results in query_results.items():
            if query_id not in self.qrels:
                logger.debug("Skipping query %s — not in qrels", query_id)
                continue
            labels_by_query[query_id] = self.build_relevance_labels(
                query_id, results
            )

        return compute_retrieval_metrics(labels_by_query, k=k)

    @property
    def corpus_query_ids(self) -> set[str]:
        """Query IDs that exist in both qrels and queries."""
        return set(self.qrels.keys()) & set(self.queries.keys())
