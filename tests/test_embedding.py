"""Tests for SentenceTransformer embedding.

TDD: tests written before implementation.
Tests verify the BaseEmbedder contract — shapes, normalization, properties.
Uses the real model (MiniLM is small enough for CI).

Key requirements:
  - A10: L2-normalized embeddings for cosine similarity
  - D4 pitfall #3: dimension must match model (384 for MiniLM, 768 for mpnet)
  - D2: model_name used as cache key
"""

import numpy as np
import pytest

from src.base.interfaces import BaseEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Load MiniLM once for all tests in this module (it's ~80MB)."""
    from src.embedding.sentence_transformer import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")


class TestSentenceTransformerEmbedder:

    def test_is_base_embedder(self, embedder):
        assert isinstance(embedder, BaseEmbedder)

    def test_dimension_matches_model(self, embedder):
        """MiniLM-L6-v2 produces 384-dimensional embeddings."""
        assert embedder.dimension == 384

    def test_model_name_stored(self, embedder):
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_embed_returns_ndarray(self, embedder):
        result = embedder.embed(["hello world"])
        assert isinstance(result, np.ndarray)

    def test_embed_shape_single(self, embedder):
        result = embedder.embed(["hello world"])
        assert result.shape == (1, 384)

    def test_embed_shape_batch(self, embedder):
        result = embedder.embed(["hello", "world", "foo"])
        assert result.shape == (3, 384)

    def test_l2_normalized(self, embedder):
        """A10: embeddings must be L2-normalized (unit vectors) for FAISS IndexFlatIP."""
        result = embedder.embed(["The cat sat on the mat.", "Dogs are great pets."])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_different_texts_produce_different_embeddings(self, embedder):
        result = embedder.embed(["quantum physics", "chocolate cake recipe"])
        # Cosine similarity should be well below 1.0 for unrelated texts
        sim = float(np.dot(result[0], result[1]))
        assert sim < 0.8

    def test_similar_texts_produce_similar_embeddings(self, embedder):
        result = embedder.embed(["the dog ran fast", "the dog sprinted quickly"])
        sim = float(np.dot(result[0], result[1]))
        assert sim > 0.5

    def test_empty_list_returns_empty_array(self, embedder):
        result = embedder.embed([])
        assert result.shape == (0, 384)

    def test_dtype_is_float(self, embedder):
        result = embedder.embed(["test"])
        assert result.dtype in (np.float32, np.float64)


class TestMpnetEmbedder:
    """Verify mpnet model produces correct dimension.

    Separate class because loading a second model is expensive.
    Only test the things that differ from MiniLM.
    """

    @pytest.fixture(scope="class")
    def mpnet(self):
        from src.embedding.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name="all-mpnet-base-v2")

    def test_dimension_is_768(self, mpnet):
        assert mpnet.dimension == 768

    def test_model_name_stored(self, mpnet):
        assert mpnet.model_name == "all-mpnet-base-v2"

    def test_embed_shape(self, mpnet):
        result = mpnet.embed(["hello world"])
        assert result.shape == (1, 768)

    def test_l2_normalized(self, mpnet):
        result = mpnet.embed(["test sentence"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
