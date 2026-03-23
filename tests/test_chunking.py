"""Tests for chunking strategies.

TDD: tests define the contract before implementation.
All three strategies implement BaseChunker and return list[Chunk].

Adapted from project 3's test_chunking.py for the ABC-based design.
"""

import numpy as np
import pytest

from src.base.interfaces import BaseChunker, BaseEmbedder
from src.models import Chunk, ChunkingStrategy, ChunkMetadata, Document, DocumentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_words(n: int) -> str:
    """Generate a string of n words, each ~6 chars, for predictable test input."""
    return " ".join(f"word{i:02d}" for i in range(n))


def make_document(content: str, source: str = "test.pdf") -> Document:
    """Build a Document from content string."""
    return Document(
        content=content,
        metadata=DocumentMetadata(source=source, page_count=1),
    )


# Two clearly different topics for semantic chunking tests
MULTI_TOPIC_TEXT = (
    "The company reported strong quarterly earnings. Revenue increased by 15% "
    "compared to the previous year. Operating margins expanded to 22%. "
    "The board of directors approved a quarterly dividend of $0.50 per share. "
    "Net income rose to $45 million, exceeding analyst expectations. "
    # Topic shift
    "Hurricane season brought unprecedented storms to the Gulf Coast. "
    "Rainfall totals exceeded 30 inches in some areas. Emergency services "
    "were deployed across three states. Flooding damaged thousands of homes "
    "and displaced many families. The National Weather Service issued warnings."
)


SAMPLE_SENTENCES = (
    "The company reported strong earnings. Revenue grew by 15% year-over-year. "
    "Operating margins improved significantly. The board approved a dividend increase. "
    "International markets showed mixed results. The Asia-Pacific region outperformed. "
    "European operations faced headwinds. Management remains cautiously optimistic. "
    "New product launches are planned for Q3. R&D spending increased by 20%. "
    "The workforce expanded to 5,000 employees. Customer satisfaction scores improved."
)


class StubEmbedder(BaseEmbedder):
    """Minimal embedder for semantic chunker tests.

    Returns embeddings that make the first 5 sentences similar to each other
    and the last 5 sentences similar to each other, with a gap in between.
    This lets us test boundary detection without loading a real model.
    """

    def __init__(self):
        self._call_count = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        # Return 2D embeddings: topic A near [1, 0], topic B near [0, 1]
        embeddings = []
        for text in texts:
            self._call_count += 1
            if any(w in text.lower() for w in ["hurricane", "storm", "rain", "flood", "weather", "emergency"]):
                embeddings.append([0.1, 0.9])
            else:
                embeddings.append([0.9, 0.1])
        result = np.array(embeddings, dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / norms

    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "stub"


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------

class TestFixedChunker:
    @pytest.fixture
    def chunker(self):
        from src.chunking.fixed import FixedChunker
        return FixedChunker(chunk_size=100, chunk_overlap=0)

    def test_is_base_chunker(self, chunker):
        assert isinstance(chunker, BaseChunker)

    def test_returns_list_of_chunks(self, chunker):
        doc = make_document(make_words(50))
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunks_cover_all_text(self):
        """With overlap=0, concatenating all chunks should reproduce the full text."""
        from src.chunking.fixed import FixedChunker
        chunker = FixedChunker(chunk_size=100, chunk_overlap=0)
        text = make_words(50)
        doc = make_document(text)
        chunks = chunker.chunk(doc)
        reconstructed = "".join(c.content for c in chunks)
        assert reconstructed == text

    def test_overlap_produces_shared_text(self):
        from src.chunking.fixed import FixedChunker
        chunker = FixedChunker(chunk_size=100, chunk_overlap=30)
        doc = make_document(make_words(100))
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        # Tail of chunk[0] should appear in head of chunk[1]
        for i in range(min(3, len(chunks) - 1)):
            tail = chunks[i].content[-30:]
            assert tail in chunks[i + 1].content

    def test_overlap_zero_no_shared_text(self):
        from src.chunking.fixed import FixedChunker
        chunker = FixedChunker(chunk_size=80, chunk_overlap=0)
        doc = make_document(make_words(50))
        chunks = chunker.chunk(doc)
        if len(chunks) > 1:
            # End of chunk 0 should not overlap with start of chunk 1
            assert chunks[0].content[-1:] != chunks[1].content[:1] or \
                chunks[0].content != chunks[1].content

    def test_metadata_populated(self, chunker):
        doc = make_document(make_words(50))
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.document_id == doc.id
            assert chunk.metadata.source == "test.pdf"
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.chunking_strategy == ChunkingStrategy.FIXED
            assert chunk.metadata.chunk_size == 100
            assert chunk.metadata.chunk_overlap == 0
            assert chunk.id  # UUID auto-generated

    def test_start_end_char_offsets(self, chunker):
        text = make_words(50)
        doc = make_document(text)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert doc.content[chunk.metadata.start_char:chunk.metadata.end_char] == chunk.content

    def test_short_text_returns_one_chunk(self, chunker):
        doc = make_document("Hello world")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"

    def test_empty_text_returns_no_chunks(self, chunker):
        doc = make_document("")
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self, chunker):
        doc = make_document("   \n\n   ")
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_chunk_size_respected(self, chunker):
        doc = make_document(make_words(200))
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert len(chunk.content) <= 100

    def test_no_infinite_loop_overlap_exceeds_size(self):
        """D4 pitfall #1: overlap >= size must not loop forever."""
        from src.chunking.fixed import FixedChunker
        chunker = FixedChunker(chunk_size=50, chunk_overlap=100)
        doc = make_document(make_words(50))
        chunks = chunker.chunk(doc)  # should terminate
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Recursive chunking
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    @pytest.fixture
    def chunker(self):
        from src.chunking.recursive import RecursiveChunker
        return RecursiveChunker(chunk_size=200, chunk_overlap=0)

    def test_is_base_chunker(self, chunker):
        assert isinstance(chunker, BaseChunker)

    def test_returns_list_of_chunks(self, chunker):
        doc = make_document(SAMPLE_SENTENCES)
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_prefers_paragraph_boundaries(self):
        """Given text with paragraph breaks, chunks should split there first."""
        from src.chunking.recursive import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three content here."
        doc = make_document(text)
        chunks = chunker.chunk(doc)
        # All text fits in one chunk at size 200, so expect 1 chunk
        assert len(chunks) == 1
        # With a smaller size, should split at paragraph boundaries
        small_chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = small_chunker.chunk(doc)
        assert len(chunks) >= 2
        # Each chunk should start cleanly (not mid-word)
        for chunk in chunks:
            assert chunk.content.strip()

    def test_falls_back_to_sentence_then_word(self):
        """A long paragraph with no sub-paragraph breaks should split at sentences."""
        from src.chunking.recursive import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=0)
        doc = make_document(SAMPLE_SENTENCES)
        chunks = chunker.chunk(doc)
        # Should split at sentence boundaries where possible
        for chunk in chunks[:-1]:
            stripped = chunk.content.rstrip()
            assert stripped.endswith((".", "!", "?", '"')), (
                f"Chunk doesn't end at sentence boundary: ...{stripped[-30:]!r}"
            )

    def test_metadata_populated(self, chunker):
        doc = make_document(SAMPLE_SENTENCES)
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.document_id == doc.id
            assert chunk.metadata.source == "test.pdf"
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.chunking_strategy == ChunkingStrategy.RECURSIVE
            assert chunk.metadata.chunk_size == 200

    def test_overlap_works(self):
        from src.chunking.recursive import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=50)
        doc = make_document(SAMPLE_SENTENCES)
        chunks = chunker.chunk(doc)
        if len(chunks) > 1:
            # Some overlap should exist between consecutive chunks
            for i in range(min(3, len(chunks) - 1)):
                words_a = set(chunks[i].content.split())
                words_b = set(chunks[i + 1].content.split())
                assert words_a & words_b, f"No word overlap between chunk {i} and {i+1}"

    def test_empty_text_returns_no_chunks(self, chunker):
        doc = make_document("")
        assert chunker.chunk(doc) == []

    def test_short_text_returns_one_chunk(self, chunker):
        doc = make_document("Short text.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_chunk_size_approximately_respected(self, chunker):
        doc = make_document(SAMPLE_SENTENCES * 3)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            # Allow some overshoot since we don't split mid-sentence
            assert len(chunk.content) <= 200 * 1.5, (
                f"Chunk too long: {len(chunk.content)} chars"
            )

    def test_no_infinite_loop_overlap_exceeds_size(self):
        from src.chunking.recursive import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=100)
        doc = make_document(SAMPLE_SENTENCES)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    @pytest.fixture
    def chunker(self):
        from src.chunking.semantic import SemanticChunker
        return SemanticChunker(
            embedder=StubEmbedder(),
            chunk_size=500,
            chunk_overlap=0,
        )

    def test_is_base_chunker(self, chunker):
        assert isinstance(chunker, BaseChunker)

    def test_returns_list_of_chunks(self, chunker):
        doc = make_document(MULTI_TOPIC_TEXT)
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_detects_topic_shift(self, chunker):
        """With two distinct topics, should produce at least 2 chunks."""
        doc = make_document(MULTI_TOPIC_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2, (
            f"Expected ≥2 chunks for multi-topic text, got {len(chunks)}"
        )

    def test_no_mid_sentence_splits(self, chunker):
        doc = make_document(MULTI_TOPIC_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks[:-1]:
            stripped = chunk.content.rstrip()
            assert stripped.endswith((".", "!", "?")), (
                f"Chunk doesn't end at sentence boundary: ...{stripped[-30:]!r}"
            )

    def test_metadata_populated(self, chunker):
        doc = make_document(MULTI_TOPIC_TEXT)
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.document_id == doc.id
            assert chunk.metadata.chunking_strategy == ChunkingStrategy.SEMANTIC
            assert chunk.metadata.chunk_index == i

    def test_empty_text_returns_no_chunks(self, chunker):
        doc = make_document("")
        assert chunker.chunk(doc) == []

    def test_single_sentence_returns_one_chunk(self, chunker):
        doc = make_document("Just one sentence here.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_max_chunk_size_fallback(self):
        """D4 pitfall #2: even if all similarities are high, chunks should not
        exceed the max size — there must be a hard cap fallback."""
        from src.chunking.semantic import SemanticChunker

        # Embedder that returns identical vectors for everything (all similarities = 1.0)
        class IdenticalEmbedder(BaseEmbedder):
            def embed(self, texts: list[str]) -> np.ndarray:
                v = np.ones((len(texts), 3), dtype=np.float32)
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                return v / norms

            @property
            def dimension(self) -> int:
                return 3

            @property
            def model_name(self) -> str:
                return "identical"

        chunker = SemanticChunker(
            embedder=IdenticalEmbedder(),
            chunk_size=200,
            chunk_overlap=0,
        )
        doc = make_document(SAMPLE_SENTENCES * 3)
        chunks = chunker.chunk(doc)
        # Despite all similarities being identical, chunks shouldn't be huge
        for chunk in chunks:
            assert len(chunk.content) <= 200 * 2, (
                f"Chunk exceeded max size fallback: {len(chunk.content)} chars"
            )
