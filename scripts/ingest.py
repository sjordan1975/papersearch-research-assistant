#!/usr/bin/env python3
"""Ingest corpus into chunks, embeddings, and searchable index.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --chunking fixed --embedding all-MiniLM-L6-v2
    python scripts/ingest.py --chunking semantic --chunk-size 1024

This processes the corpus JSON files, chunks them, embeds them, and builds
FAISS + BM25 indices. Results are cached in data/cache/ for experiment runs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.fixed import FixedChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.experiment.runner import chunk_section, load_corpus_sections
from src.models import ChunkingStrategy
from src.stores.faiss_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int, embedder=None):
    """Instantiate chunker based on strategy name."""
    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "semantic":
        if embedder is None:
            raise ValueError("Semantic chunking requires an embedder")
        return SemanticChunker(
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest corpus into searchable index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/pdf/arxiv/corpus"),
        help="Directory containing corpus JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory to save cached artifacts",
    )
    parser.add_argument(
        "--chunking",
        choices=["fixed", "recursive", "semantic"],
        default="fixed",
        help="Chunking strategy (default: fixed)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in characters (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Overlap between chunks in characters (default: 0)",
    )
    parser.add_argument(
        "--embedding",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )

    args = parser.parse_args()

    if not args.corpus_dir.exists():
        logger.error("Corpus directory not found: %s", args.corpus_dir)
        logger.error("Run 'python scripts/download_data.py' first")
        sys.exit(1)

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build cache key for this config
    config_key = f"{args.chunking}_{args.chunk_size}_{args.chunk_overlap}_{args.embedding}"
    logger.info("Ingesting with config: %s", config_key)

    # Initialize embedder
    logger.info("Loading embedding model: %s", args.embedding)
    embedder = SentenceTransformerEmbedder(model_name=args.embedding)

    # Initialize chunker
    chunker = get_chunker(
        args.chunking,
        args.chunk_size,
        args.chunk_overlap,
        embedder if args.chunking == "semantic" else None,
    )

    # Process corpus
    logger.info("Loading corpus from: %s", args.corpus_dir)
    all_chunks = []
    section_count = 0

    for section in load_corpus_sections(args.corpus_dir):
        chunks = chunk_section(chunker, section)
        all_chunks.extend(chunks)
        section_count += 1
        if section_count % 100 == 0:
            logger.info("Processed %d sections, %d chunks so far", section_count, len(all_chunks))

    logger.info("Total: %d sections -> %d chunks", section_count, len(all_chunks))

    if not all_chunks:
        logger.error("No chunks produced. Check corpus data.")
        sys.exit(1)

    # Embed chunks
    logger.info("Embedding %d chunks...", len(all_chunks))
    chunk_texts = [c.content for c in all_chunks]
    embeddings = embedder.embed(chunk_texts)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Build FAISS index
    logger.info("Building FAISS index...")
    vector_store = FaissVectorStore(dimension=embedder.dimension)
    chunk_ids = [str(c.id) for c in all_chunks]
    vector_store.add(chunk_ids, embeddings)

    # Save artifacts
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = chunks_dir / f"{config_key}.json"

    # Save chunks as JSON (for BM25 and lookup)
    chunks_data = [
        {
            "id": str(c.id),
            "content": c.content,
            "metadata": c.metadata.model_dump(),
        }
        for c in all_chunks
    ]
    chunks_path.write_text(json.dumps(chunks_data, indent=2))
    logger.info("Saved chunks to: %s", chunks_path)

    # Save FAISS index
    indices_dir = args.output_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    index_path = indices_dir / f"{config_key}.faiss"
    vector_store.save(str(index_path))
    logger.info("Saved FAISS index to: %s", index_path)

    logger.info("Ingestion complete!")
    logger.info("  Chunks: %d", len(all_chunks))
    logger.info("  Index dimension: %d", embedder.dimension)
    logger.info("  Config key: %s", config_key)


if __name__ == "__main__":
    main()
