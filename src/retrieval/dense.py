"""Dense retriever — embed query, search FAISS, return ranked chunks.

The simplest retrieval strategy: embed the query with the same model used
to embed the chunks, find the nearest neighbors in the vector store, and
wrap the results as RetrievalResult objects.
"""

import numpy as np

from src.base.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from src.models import Chunk, RetrievalResult, RetrieverType


class DenseRetriever(BaseRetriever):
    """Retrieve chunks by dense vector similarity.

    Args:
        embedder: The embedding model (must match the one used to build the index).
        vector_store: A populated FAISS (or compatible) vector store.
        chunk_lookup: Mapping from chunk ID string to Chunk object.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        chunk_lookup: dict[str, Chunk],
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_lookup = chunk_lookup

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed([query])[0]
        raw_results = self.vector_store.search(query_embedding, top_k)

        results = []
        for chunk_id, score in raw_results:
            chunk = self.chunk_lookup.get(chunk_id)
            if chunk is not None:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    retriever_type=RetrieverType.DENSE,
                ))

        return results
