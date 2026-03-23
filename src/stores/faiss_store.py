"""FAISS vector store — local similarity search.

Wraps FAISS IndexFlatIP (inner product) for exact nearest-neighbor search.
With L2-normalized vectors, inner product equals cosine similarity.

D1: We chose FAISS over cloud vector stores (Pinecone, Turbopuffer) because
the core skills are backend-agnostic and FAISS has zero setup cost.
The BaseVectorStore ABC proves we can swap backends.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from src.base.interfaces import BaseVectorStore


class FaissVectorStore(BaseVectorStore):
    """FAISS-backed vector store using IndexFlatIP.

    Args:
        dimension: Embedding dimension (must match the embedder).
    """

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []

    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        if len(ids) != len(embeddings):
            raise ValueError(
                f"IDs length ({len(ids)}) must match embeddings length ({len(embeddings)})"
            )
        if embeddings.ndim == 2 and embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match "
                f"store dimension ({self._dimension})"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._ids.extend(ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []

        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self._ids[idx], float(score)))

        return results

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "ids.json", "w") as f:
            json.dump(self._ids, f)

    def load(self, path: str) -> None:
        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "ids.json") as f:
            self._ids = json.load(f)

    def __len__(self) -> int:
        return self._index.ntotal
