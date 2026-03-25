"""Integration test: full pipeline from document → answer.

T20: Verify the RAG pipeline works end-to-end.

This is a smoke test, not a load test. It uses:
- Synthetic document (PDF loading tested separately)
- Real chunking, embedding, retrieval
- Mocked LLM (LLM response parsing tested separately)

Should complete in <10 seconds.
"""

from uuid import uuid4

import numpy as np
import pytest

from src.chunking.fixed import FixedChunker
from src.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.generation.answer import AnswerGenerator, build_prompt, _parse_citation_refs
from src.models import (
    Chunk,
    ChunkMetadata,
    ChunkingStrategy,
    Document,
    DocumentMetadata,
    RetrievalResult,
    RetrieverType,
)
from src.retrieval.dense import DenseRetriever
from src.stores.faiss_store import FaissVectorStore


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = Document(
    content="""
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    The field has seen remarkable advances in recent years.

    Supervised Learning

    Supervised learning uses labeled training data to learn a mapping from
    inputs to outputs. Common algorithms include linear regression, decision
    trees, and neural networks. The model learns by minimizing the error
    between its predictions and the true labels.

    Unsupervised Learning

    Unsupervised learning finds patterns in unlabeled data. Clustering
    algorithms like K-means group similar data points together. Dimensionality
    reduction techniques like PCA help visualize high-dimensional data.

    Deep Learning

    Deep learning uses neural networks with many layers. Convolutional neural
    networks excel at image recognition. Recurrent neural networks handle
    sequential data like text and time series. Transformers have revolutionized
    natural language processing.

    Conclusion

    Machine learning continues to transform industries from healthcare to
    finance. Understanding these fundamental concepts is essential for
    practitioners in the field.
    """.strip(),
    metadata=DocumentMetadata(source="test_ml_intro.pdf", page_count=1),
)


class MockLLM:
    """Mock LLM that returns a canned response with citations."""

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        return (
            "Machine learning is a subset of artificial intelligence [1]. "
            "It includes supervised learning which uses labeled data [2], "
            "and unsupervised learning which finds patterns in unlabeled data [3]."
        )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Integration tests for the complete RAG pipeline."""

    @pytest.fixture
    def chunker(self):
        """Fixed chunker with small chunk size for testing."""
        return FixedChunker(chunk_size=200, chunk_overlap=20)

    @pytest.fixture
    def embedder(self):
        """Real embedder - tests actual embedding."""
        return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    def test_document_to_chunks(self, chunker):
        """Verify document → chunks works."""
        chunks = chunker.chunk(SAMPLE_DOCUMENT)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata.source == "test_ml_intro.pdf" for c in chunks)
        # Content should be preserved across chunks
        combined = " ".join(c.content for c in chunks)
        assert "Machine learning" in combined
        assert "supervised" in combined.lower()

    def test_chunks_to_embeddings(self, chunker, embedder):
        """Verify chunks → embeddings works."""
        chunks = chunker.chunk(SAMPLE_DOCUMENT)
        texts = [c.content for c in chunks]

        embeddings = embedder.embed(texts)

        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == 384  # MiniLM dimension
        # Embeddings should be L2 normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=5)

    def test_embeddings_to_retrieval(self, chunker, embedder):
        """Verify embeddings → FAISS → retrieval works."""
        chunks = chunker.chunk(SAMPLE_DOCUMENT)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)

        # Build chunk lookup and IDs
        chunk_lookup = {str(c.id): c for c in chunks}
        ids = [str(c.id) for c in chunks]

        # Index chunks
        store = FaissVectorStore(dimension=384)
        store.add(ids, embeddings)

        # Create retriever
        retriever = DenseRetriever(
            embedder=embedder, vector_store=store, chunk_lookup=chunk_lookup
        )

        # Query about supervised learning
        results = retriever.retrieve("What is supervised learning?", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Top result should be about supervised learning
        top_chunk = results[0].chunk.content.lower()
        assert "supervised" in top_chunk or "labeled" in top_chunk

    def test_retrieval_to_answer(self, chunker, embedder):
        """Verify retrieval → answer generation works (with mocked LLM)."""
        chunks = chunker.chunk(SAMPLE_DOCUMENT)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)

        # Build chunk lookup and IDs
        chunk_lookup = {str(c.id): c for c in chunks}
        ids = [str(c.id) for c in chunks]

        store = FaissVectorStore(dimension=384)
        store.add(ids, embeddings)
        retriever = DenseRetriever(
            embedder=embedder, vector_store=store, chunk_lookup=chunk_lookup
        )

        # Retrieve
        query = "What is machine learning?"
        results = retriever.retrieve(query, top_k=3)

        # Generate answer with mock LLM
        generator = AnswerGenerator(llm=MockLLM())
        response = generator.generate(query, results)

        # Verify response structure
        assert response.query == query
        assert response.answer is not None
        assert len(response.answer) > 0
        assert len(response.citations) > 0
        # Citations should have valid structure
        for citation in response.citations:
            assert citation.chunk_id is not None
            assert citation.source is not None

    def test_full_pipeline_end_to_end(self, chunker, embedder):
        """Full integration: document → chunks → embed → index → retrieve → answer."""
        # 1. Chunk document
        chunks = chunker.chunk(SAMPLE_DOCUMENT)
        assert len(chunks) > 0

        # 2. Embed chunks
        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)
        assert embeddings.shape[0] == len(chunks)

        # Build chunk lookup and IDs
        chunk_lookup = {str(c.id): c for c in chunks}
        ids = [str(c.id) for c in chunks]

        # 3. Index in FAISS
        store = FaissVectorStore(dimension=384)
        store.add(ids, embeddings)
        assert len(store) == len(chunks)

        # 4. Retrieve for a query
        retriever = DenseRetriever(
            embedder=embedder, vector_store=store, chunk_lookup=chunk_lookup
        )
        query = "Explain deep learning and neural networks"
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3

        # Verify relevance - should retrieve deep learning content
        retrieved_text = " ".join(r.chunk.content.lower() for r in results)
        assert "deep learning" in retrieved_text or "neural" in retrieved_text

        # 5. Generate answer (mocked)
        generator = AnswerGenerator(llm=MockLLM())
        response = generator.generate(query, results)

        # Final verification
        assert response.query == query
        assert response.answer is not None
        assert "[1]" in response.answer  # Has citations


class TestPromptConstruction:
    """Test prompt building for answer generation."""

    def test_build_prompt_with_results(self):
        """Verify prompt includes sources and question."""
        doc_id = uuid4()
        chunk = Chunk(
            content="Neural networks process data in layers.",
            metadata=ChunkMetadata(
                document_id=doc_id,
                source="paper.pdf",
                chunk_index=0,
                chunking_strategy=ChunkingStrategy.FIXED,
                chunk_size=512,
            ),
        )
        results = [
            RetrievalResult(chunk=chunk, score=0.9, retriever_type=RetrieverType.DENSE)
        ]

        prompt = build_prompt("How do neural networks work?", results)

        assert "[1]" in prompt
        assert "paper.pdf" in prompt
        assert "Neural networks process data" in prompt
        assert "How do neural networks work?" in prompt

    def test_build_prompt_empty_results(self):
        """Verify graceful handling of no results."""
        prompt = build_prompt("What is AI?", [])

        assert "What is AI?" in prompt
        assert "No source documents" in prompt

    def test_parse_citations(self):
        """Verify citation parsing from answer text."""
        answer = "AI is powerful [1]. It learns from data [2] and [3]."
        refs = _parse_citation_refs(answer)

        assert refs == [1, 2, 3]

    def test_parse_citations_duplicates(self):
        """Verify duplicate citations are deduplicated."""
        answer = "Point A [1]. Related to point B [1] and [2]."
        refs = _parse_citation_refs(answer)

        assert refs == [1, 2]
