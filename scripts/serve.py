#!/usr/bin/env python3
"""Interactive Q&A with the RAG pipeline.

Usage:
    python scripts/serve.py
    python scripts/serve.py --config fixed_512_0_all-MiniLM-L6-v2
    python scripts/serve.py --retriever hybrid --top-k 10

Loads a pre-built index (from ingest.py) and provides an interactive
prompt for asking questions. Retrieves relevant chunks, generates an
answer with citations, and displays the result.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.generation.answer import AnswerGenerator
from src.generation.llm import LiteLLMClient
from src.models import Chunk, ChunkMetadata, ChunkingStrategy, RetrieverType
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.stores.faiss_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_chunks(chunks_path: Path) -> list[Chunk]:
    """Load chunks from JSON file."""
    data = json.loads(chunks_path.read_text())
    chunks = []
    for item in data:
        metadata = ChunkMetadata(**item["metadata"])
        chunk = Chunk(
            id=item["id"],
            content=item["content"],
            metadata=metadata,
        )
        chunks.append(chunk)
    return chunks


def build_retriever(
    retriever_type: str,
    chunks: list[Chunk],
    embedder: SentenceTransformerEmbedder,
    index_path: Path,
):
    """Build the appropriate retriever."""
    # Load FAISS index
    vector_store = FaissVectorStore(dimension=embedder.dimension)
    vector_store.load(str(index_path))

    # Build chunk lookup
    chunk_lookup = {str(c.id): c for c in chunks}

    if retriever_type == "dense":
        return DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
    elif retriever_type == "bm25":
        return BM25Retriever(chunks=chunks)
    elif retriever_type == "hybrid":
        dense = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
        bm25 = BM25Retriever(chunks=chunks)
        return HybridRetriever(dense=dense, bm25=bm25)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def format_response(response, show_sources: bool = True) -> str:
    """Format a QAResponse for display."""
    output = []
    output.append("\n" + "=" * 60)
    output.append("ANSWER:")
    output.append("=" * 60)
    output.append(response.answer)

    if show_sources and response.citations:
        output.append("\n" + "-" * 60)
        output.append("CITATIONS:")
        output.append("-" * 60)
        for i, citation in enumerate(response.citations, 1):
            output.append(f"\n[{i}] {citation.source}")
            if citation.page_number:
                output.append(f"    Page: {citation.page_number}")
            output.append(f"    Snippet: {citation.text_snippet[:100]}...")

    output.append("\n" + "=" * 60)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Q&A with the RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory with cached artifacts from ingest.py",
    )
    parser.add_argument(
        "--config",
        default="fixed_512_0_all-MiniLM-L6-v2",
        help="Config key (chunking_size_overlap_embedding)",
    )
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="Retriever type (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for answer generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source citations in output",
    )

    args = parser.parse_args()

    # Check for cached artifacts
    chunks_path = args.cache_dir / "chunks" / f"{args.config}.json"
    index_path = args.cache_dir / "indices" / f"{args.config}.faiss"

    if not chunks_path.exists():
        logger.error("Chunks not found: %s", chunks_path)
        logger.error("Run 'python scripts/ingest.py' first with matching config")
        sys.exit(1)

    if not index_path.exists():
        logger.error("Index not found: %s", index_path)
        logger.error("Run 'python scripts/ingest.py' first with matching config")
        sys.exit(1)

    # Extract embedding model from config
    # Config format: chunking_size_overlap_embedding
    config_parts = args.config.rsplit("_", 1)
    embedding_model = config_parts[-1] if len(config_parts) > 1 else "all-MiniLM-L6-v2"

    logger.info("Loading chunks from: %s", chunks_path)
    chunks = load_chunks(chunks_path)
    logger.info("Loaded %d chunks", len(chunks))

    logger.info("Loading embedding model: %s", embedding_model)
    embedder = SentenceTransformerEmbedder(model_name=embedding_model)

    logger.info("Building %s retriever...", args.retriever)
    retriever = build_retriever(args.retriever, chunks, embedder, index_path)

    logger.info("Initializing LLM: %s", args.model)
    llm = LiteLLMClient(model=args.model)
    generator = AnswerGenerator(llm=llm)

    print("\n" + "=" * 60)
    print("PaperSearch Research Assistant")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Retriever: {args.retriever} (top-k={args.top_k})")
    print(f"Model: {args.model}")
    print(f"Chunks loaded: {len(chunks)}")
    print("=" * 60)
    print("\nEnter your question (or 'quit' to exit):\n")

    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Retrieve
        logger.info("Retrieving top-%d chunks...", args.top_k)
        results = retriever.retrieve(query, top_k=args.top_k)

        if not results:
            print("\nNo relevant chunks found for this query.\n")
            continue

        # Generate answer
        logger.info("Generating answer...")
        response = generator.generate(query, results)

        # Display
        print(format_response(response, show_sources=not args.no_sources))
        print()


if __name__ == "__main__":
    main()
