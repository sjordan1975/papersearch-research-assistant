"""Recursive chunking — split on natural boundaries, falling back gracefully.

Tries separators in order of structural significance:
  1. Double newline (paragraph break)
  2. Single newline
  3. Sentence-ending punctuation (". ", "! ", "? ")
  4. Space (word boundary)
  5. Character-level (last resort)

This is the same approach as LangChain's RecursiveCharacterTextSplitter.
The key idea: respect document structure when possible, but always guarantee
chunks fit within the size limit.
"""

import re

from src.base.interfaces import BaseChunker
from src.models import Chunk, ChunkMetadata, ChunkingStrategy, Document

# Ordered from most to least structural
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Split a document by recursively trying natural text boundaries.

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

        raw_chunks = self._split_text(text)

        chunks = []
        for i, content in enumerate(raw_chunks):
            # Calculate approximate start_char (for traceability, not exact reconstruction)
            chunks.append(Chunk(
                content=content,
                metadata=ChunkMetadata(
                    document_id=document.id,
                    source=document.metadata.source,
                    chunk_index=i,
                    chunking_strategy=ChunkingStrategy.RECURSIVE,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
            ))

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text, then merge small pieces with overlap."""
        pieces = self._recursive_split(text, 0)
        return self._merge_with_overlap(pieces)

    def _recursive_split(self, text: str, sep_index: int) -> list[str]:
        """Split text using separators[sep_index], recurse on oversized pieces."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(_SEPARATORS):
            # Last resort: hard character split
            step = max(self.chunk_size, 1)
            return [text[i:i + step] for i in range(0, len(text), step)]

        sep = _SEPARATORS[sep_index]

        if sep == "":
            # Character-level split
            step = max(self.chunk_size, 1)
            return [text[i:i + step] for i in range(0, len(text), step)]

        # Split on this separator
        if sep in (". ", "! ", "? "):
            # Keep the punctuation with the preceding text
            parts = re.split(r"(?<=[.!?])\s", text)
        else:
            parts = text.split(sep)

        # Reassemble parts into chunks that fit within chunk_size
        result = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # This part is still too big — recurse with next separator
                result.extend(self._recursive_split(part, sep_index + 1))

        return result

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Merge small pieces into chunks up to chunk_size, with overlap."""
        if not pieces:
            return []

        merged = []
        current = pieces[0]

        for piece in pieces[1:]:
            combined = current + " " + piece
            if len(combined) <= self.chunk_size:
                current = combined
            else:
                merged.append(current)
                # Apply overlap: carry some text from the end of current
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    overlap_text = current[-self.chunk_overlap:]
                    # Try to start overlap at a word boundary
                    space_idx = overlap_text.find(" ")
                    if space_idx != -1:
                        overlap_text = overlap_text[space_idx + 1:]
                    current = overlap_text + " " + piece
                else:
                    current = piece

        if current.strip():
            merged.append(current)

        return merged
