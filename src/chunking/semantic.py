"""Semantic chunking — split where topic similarity drops.

Unlike fixed and recursive chunking which use structural cues (characters,
paragraphs, sentences), semantic chunking uses meaning. It:

1. Splits text into sentences
2. Embeds each sentence
3. Computes cosine similarity between adjacent sentences
4. Splits where similarity drops below a threshold (topic boundary)

This is the only chunking strategy that uses the embedding model during
chunking (not just during indexing). That means it's slower but potentially
produces more coherent chunks.

D4 pitfall #2: if all similarities are high (no clear boundaries), a max
chunk size fallback prevents runaway chunk sizes.
"""

import re

import numpy as np

from src.base.interfaces import BaseChunker, BaseEmbedder
from src.models import Chunk, ChunkMetadata, ChunkingStrategy, Document


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex.

    Handles common abbreviations and decimal numbers to avoid false splits.
    Good enough for academic papers — no heavy NLP dependency needed.
    """
    # Split on sentence-ending punctuation followed by space + capital letter or end
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(BaseChunker):
    """Split a document at semantic boundaries using embedding similarity.

    Args:
        embedder: An embedding model implementing BaseEmbedder.
        chunk_size: Soft max characters per chunk (hard cap = 2x this).
        chunk_overlap: Not used for semantic chunking (boundaries are semantic,
            not positional), but stored in metadata for experiment config tracking.
        similarity_threshold: Split when adjacent sentence similarity drops
            below this value. If None, uses the mean - 1 stddev of all
            similarities as an adaptive threshold.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        similarity_threshold: float | None = None,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [self._make_chunk(sentences[0], 0, document)]

        # Embed all sentences
        embeddings = self.embedder.embed(sentences)

        # Compute cosine similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)

        # Determine threshold
        if self.similarity_threshold is not None:
            threshold = self.similarity_threshold
        else:
            # Adaptive: mean - 1 stddev
            mean = np.mean(similarities)
            std = np.std(similarities)
            threshold = mean - std

        # Find split points: where similarity drops below threshold
        split_indices = self._find_splits(sentences, similarities, threshold)

        # Build chunks from sentence groups
        chunks = []
        chunk_index = 0
        for start_idx, end_idx in split_indices:
            content = " ".join(sentences[start_idx:end_idx])
            if content.strip():
                chunks.append(self._make_chunk(content, chunk_index, document))
                chunk_index += 1

        return chunks

    def _find_splits(
        self,
        sentences: list[str],
        similarities: list[float],
        threshold: float,
    ) -> list[tuple[int, int]]:
        """Find (start, end) sentence index ranges for each chunk.

        Splits at low-similarity boundaries, with a hard cap on chunk size
        to handle the degenerate case where all similarities are high.
        """
        hard_cap = self.chunk_size * 2
        groups: list[tuple[int, int]] = []
        start = 0
        current_len = len(sentences[0])

        for i in range(len(similarities)):
            next_len = len(sentences[i + 1])

            # Split if: similarity is low OR chunk would exceed hard cap
            would_exceed = (current_len + next_len + 1) > hard_cap
            low_similarity = similarities[i] < threshold

            if low_similarity or would_exceed:
                groups.append((start, i + 1))
                start = i + 1
                current_len = next_len
            else:
                current_len += next_len + 1  # +1 for space

        # Final group
        if start < len(sentences):
            groups.append((start, len(sentences)))

        return groups

    def _make_chunk(self, content: str, chunk_index: int, document: Document) -> Chunk:
        return Chunk(
            content=content,
            metadata=ChunkMetadata(
                document_id=document.id,
                source=document.metadata.source,
                chunk_index=chunk_index,
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ),
        )
