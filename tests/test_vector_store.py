"""Tests for FAISS vector store.

TDD: tests written before implementation.

The store wraps FAISS IndexFlatIP (inner product = cosine similarity
when vectors are L2-normalized). Tests verify:
  - add/search contract
  - save/load persistence
  - D4 pitfall #3: dimension validation
  - Edge cases: empty index, top_k > stored, duplicate IDs
"""

import numpy as np
import pytest

from src.base.interfaces import BaseVectorStore


def _random_vectors(n: int, dim: int = 384) -> np.ndarray:
    """Generate random L2-normalized vectors."""
    v = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


@pytest.fixture
def store():
    from src.stores.faiss_store import FaissVectorStore
    return FaissVectorStore(dimension=384)


class TestFaissVectorStore:

    def test_is_base_vector_store(self, store):
        assert isinstance(store, BaseVectorStore)

    def test_add_and_search(self, store):
        ids = ["a", "b", "c"]
        embeddings = _random_vectors(3)
        store.add(ids, embeddings)

        results = store.search(embeddings[0], top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        # First result should be the query itself (exact match, score ≈ 1.0)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_returns_tuples_of_id_and_score(self, store):
        store.add(["x"], _random_vectors(1))
        results = store.search(_random_vectors(1)[0], top_k=1)
        assert len(results) == 1
        id_, score = results[0]
        assert isinstance(id_, str)
        assert isinstance(score, float)

    def test_search_sorted_by_descending_score(self, store):
        ids = [f"doc_{i}" for i in range(10)]
        store.add(ids, _random_vectors(10))
        results = store.search(_random_vectors(1)[0], top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, store):
        store.add([f"doc_{i}" for i in range(20)], _random_vectors(20))
        results = store.search(_random_vectors(1)[0], top_k=5)
        assert len(results) == 5

    def test_top_k_exceeds_stored(self, store):
        """Requesting more results than stored vectors should return all available."""
        store.add(["a", "b"], _random_vectors(2))
        results = store.search(_random_vectors(1)[0], top_k=10)
        assert len(results) == 2

    def test_search_empty_index(self, store):
        results = store.search(_random_vectors(1)[0], top_k=5)
        assert results == []

    def test_add_multiple_batches(self, store):
        """Adding in multiple batches should accumulate."""
        store.add(["a", "b"], _random_vectors(2))
        store.add(["c", "d"], _random_vectors(2))
        results = store.search(_random_vectors(1)[0], top_k=10)
        assert len(results) == 4

    def test_save_and_load(self, store, tmp_path):
        embeddings = _random_vectors(5)
        ids = [f"doc_{i}" for i in range(5)]
        store.add(ids, embeddings)

        save_path = str(tmp_path / "test_index")
        store.save(save_path)

        # Load into a fresh store
        from src.stores.faiss_store import FaissVectorStore
        loaded = FaissVectorStore(dimension=384)
        loaded.load(save_path)

        # Should produce same results
        results_original = store.search(embeddings[0], top_k=5)
        results_loaded = loaded.search(embeddings[0], top_k=5)
        assert [r[0] for r in results_original] == [r[0] for r in results_loaded]

    def test_dimension_mismatch_on_add(self, store):
        """D4 pitfall #3: adding wrong-dimension vectors should fail."""
        wrong_dim = _random_vectors(3, dim=768)  # store expects 384
        with pytest.raises(ValueError, match="dimension"):
            store.add(["a", "b", "c"], wrong_dim)

    def test_ids_count_must_match_embeddings(self, store):
        with pytest.raises(ValueError, match="length"):
            store.add(["a", "b"], _random_vectors(3))

    def test_len(self, store):
        """Store should report how many vectors it holds."""
        assert len(store) == 0
        store.add(["a", "b"], _random_vectors(2))
        assert len(store) == 2
        store.add(["c"], _random_vectors(1))
        assert len(store) == 3
