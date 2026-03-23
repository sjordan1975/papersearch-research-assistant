"""SentenceTransformers embedder — local embedding models.

Wraps the sentence-transformers library to provide BaseEmbedder-compatible
embeddings. Unlike Project 3's API-based OpenAI embeddings, these run locally:
no API cost, no rate limits, but the model lives in RAM.

Supported models:
  - all-MiniLM-L6-v2: 384d, fast, ~80MB
  - all-mpnet-base-v2: 768d, higher quality, ~420MB

Embeddings are L2-normalized so FAISS IndexFlatIP computes cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.base.interfaces import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embed text using a local SentenceTransformers model.

    Args:
        model_name: HuggingFace model identifier (e.g., "all-MiniLM-L6-v2").
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (n_texts, embedding_dim).

        Embeddings are L2-normalized (A10).
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
