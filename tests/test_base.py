"""Tests for abstract base classes.

Verifies that ABCs enforce their contracts:
- Can't instantiate an ABC directly
- Can't instantiate a subclass that's missing an abstract method
- A complete subclass works fine
"""

import numpy as np
import pytest

from src.base.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLM,
    BaseReranker,
    BaseRetriever,
    BaseVectorStore,
)
from src.models import Chunk, Document, RetrievalResult


class TestABCEnforcement:
    """ABCs should raise TypeError if you try to instantiate them directly
    or if a subclass doesn't implement all abstract methods."""

    def test_cannot_instantiate_base_chunker(self):
        with pytest.raises(TypeError):
            BaseChunker()

    def test_cannot_instantiate_base_embedder(self):
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_cannot_instantiate_base_vector_store(self):
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_cannot_instantiate_base_retriever(self):
        with pytest.raises(TypeError):
            BaseRetriever()

    def test_cannot_instantiate_base_reranker(self):
        with pytest.raises(TypeError):
            BaseReranker()

    def test_cannot_instantiate_base_llm(self):
        with pytest.raises(TypeError):
            BaseLLM()

    def test_incomplete_subclass_raises(self):
        """A subclass that forgets to implement an abstract method
        raises TypeError at instantiation — not at call time."""

        class IncompleteChunker(BaseChunker):
            pass  # forgot to implement chunk()

        with pytest.raises(TypeError):
            IncompleteChunker()


class TestConcreteSubclass:
    """A complete subclass should instantiate and work normally."""

    def test_complete_chunker_works(self):
        class StubChunker(BaseChunker):
            def chunk(self, document: Document) -> list[Chunk]:
                return []

        chunker = StubChunker()
        assert chunker.chunk(None) == []

    def test_complete_embedder_works(self):
        class StubEmbedder(BaseEmbedder):
            def embed(self, texts: list[str]) -> np.ndarray:
                return np.zeros((len(texts), 3))

            @property
            def dimension(self) -> int:
                return 3

            @property
            def model_name(self) -> str:
                return "stub"

        embedder = StubEmbedder()
        result = embedder.embed(["hello"])
        assert result.shape == (1, 3)
        assert embedder.dimension == 3
        assert embedder.model_name == "stub"

    def test_complete_retriever_works(self):
        class StubRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                return []

        retriever = StubRetriever()
        assert retriever.retrieve("test") == []
