"""Abstract base classes for all swappable pipeline components.

Every concrete implementation inherits from these. The experiment runner
and CLI tools work with these interfaces, not specific implementations.

Python ABCs vs TS interfaces:
- TS: `interface IChunker { chunk(doc: Document): Chunk[] }`
- Python: class with @abstractmethod — enforced at instantiation time
- If you forget to implement an abstract method, Python raises TypeError
  when you try to create an instance (not when you call the method)
"""

from abc import ABC, abstractmethod

import numpy as np

from src.models import Chunk, Document, RetrievalResult


class BaseChunker(ABC):
    """Split a document into chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Break a document into a list of chunks with metadata."""
        ...


class BaseEmbedder(ABC):
    """Convert text to vector embeddings."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (n_texts, embedding_dim).

        Embeddings MUST be L2-normalized (D3 pitfall #3).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The embedding dimension (e.g. 384 for MiniLM, 768 for mpnet).

        Used to validate FAISS index compatibility (D3 pitfall #3).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier, used as cache key (D2)."""
        ...


class BaseVectorStore(ABC):
    """Store and search vector embeddings."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Add embeddings with their IDs to the store."""
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Search for the top_k most similar embeddings.

        Returns list of (id, score) pairs, sorted by descending score.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""
        ...


class BaseRetriever(ABC):
    """Retrieve relevant chunks for a query."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Find the top_k most relevant chunks for a query."""
        ...


class BaseReranker(ABC):
    """Rerank retrieval results for better precision in top positions."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank results and return the top_k."""
        ...


class BaseLLM(ABC):
    """Generate text from a prompt."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        metadata: dict | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user message.
            system_prompt: Optional system message.
            temperature: Sampling temperature (0.0 = deterministic).
            metadata: Optional dict forwarded to the LLM provider for
                observability. LiteLLM passes this to Langfuse as trace
                context. Useful keys: trace_name, trace_id, session_id.
        """
        ...
