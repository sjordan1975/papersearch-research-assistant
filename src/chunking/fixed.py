"""Fixed-size chunking — split text at character boundaries.

The simplest strategy: slice the document into chunks of a fixed character
count, with optional overlap. No awareness of document structure.

This is the baseline. If recursive or semantic chunking doesn't beat this
by a meaningful margin, the added complexity isn't worth it.
"""

from src.base.interfaces import BaseChunker
from src.models import Chunk, ChunkMetadata, ChunkingStrategy, Document


class FixedChunker(BaseChunker):
    """Split a document into fixed-size character chunks.

    Args:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        # D4 pitfall #1: guarantee forward progress
        step = max(self.chunk_size - self.chunk_overlap, 1)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end]

            # Skip chunks that are only whitespace
            if segment.strip():
                chunks.append(Chunk(
                    content=segment,
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        source=document.metadata.source,
                        start_char=start,
                        end_char=min(end, len(text)),
                        chunk_index=chunk_index,
                        chunking_strategy=ChunkingStrategy.FIXED,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    ),
                ))
                chunk_index += 1

            start += step

        return chunks
