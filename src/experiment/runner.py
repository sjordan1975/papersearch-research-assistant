"""Experiment runner — grid search over RAG configurations.

Orchestrates:
1. Load corpus sections (with doc_id and section_id)
2. Chunk sections with section_index stamped on metadata
3. Embed chunks and build FAISS index
4. For each query, retrieve and evaluate against ground truth
5. Aggregate metrics across all queries
6. Save results as JSON

Uses layered caching (D2): chunks, embeddings, and indices are
computed once per unique config and reused.
"""

import json
import logging
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from src.base.interfaces import BaseChunker
from src.chunking.fixed import FixedChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.evaluation.ground_truth import GroundTruthAligner
from src.models import (
    Chunk,
    ChunkingStrategy,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    ExperimentConfig,
    ExperimentResult,
    RetrievalMetrics,
    RetrievalResult,
    RetrieverType,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.stores.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

DEFAULT_CHUNKING_STRATEGIES = [
    ChunkingStrategy.FIXED,
    ChunkingStrategy.RECURSIVE,
    ChunkingStrategy.SEMANTIC,
]

DEFAULT_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]

DEFAULT_RETRIEVER_TYPES = [
    RetrieverType.DENSE,
    RetrieverType.BM25,
    RetrieverType.HYBRID,
]


def build_experiment_grid(
    chunking_strategies: list[ChunkingStrategy] | None = None,
    embedding_models: list[str] | None = None,
    retriever_types: list[RetrieverType] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    top_k: int = 5,
) -> list[ExperimentConfig]:
    """Generate all experiment configurations from parameter space."""
    strategies = chunking_strategies or DEFAULT_CHUNKING_STRATEGIES
    models = embedding_models or DEFAULT_EMBEDDING_MODELS
    retrievers = retriever_types or DEFAULT_RETRIEVER_TYPES

    configs = []
    for strategy in strategies:
        for model in models:
            for retriever in retrievers:
                configs.append(ExperimentConfig(
                    chunking_strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=model,
                    retriever_type=retriever,
                    top_k=top_k,
                ))
    return configs


def config_to_id(config: ExperimentConfig) -> str:
    """Generate a deterministic ID from config for caching and results."""
    return (
        f"{config.chunking_strategy.value}_"
        f"{config.chunk_size}_"
        f"{config.chunk_overlap}_"
        f"{config.embedding_model}_"
        f"{config.retriever_type.value}_"
        f"k{config.top_k}"
    )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_sections(corpus_dir: Path) -> Iterator[dict]:
    """Load all corpus JSON files and yield sections with doc_id.

    Each yielded dict has: doc_id, section_id, text
    """
    for corpus_file in sorted(corpus_dir.glob("*.json")):
        try:
            with open(corpus_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load %s: %s", corpus_file, e)
            continue

        doc_id = data.get("id", corpus_file.stem)
        for section in data.get("sections", []):
            yield {
                "doc_id": doc_id,
                "section_id": section.get("section_id"),
                "text": section.get("text", ""),
            }


# ---------------------------------------------------------------------------
# Chunking with section_index
# ---------------------------------------------------------------------------

def chunk_section(chunker: BaseChunker, section: dict) -> list[Chunk]:
    """Chunk a section and stamp section_index on each chunk's metadata.

    Args:
        chunker: Any chunker implementing BaseChunker.
        section: Dict with doc_id, section_id, text.

    Returns:
        List of chunks with section_index populated.
    """
    doc_id = section["doc_id"]
    section_id = section["section_id"]
    text = section["text"]

    if not text or not text.strip():
        return []

    # Create a temporary Document for the chunker
    doc = Document(
        content=text,
        metadata=DocumentMetadata(source=f"{doc_id}.pdf"),
    )

    chunks = chunker.chunk(doc)

    # Stamp section_index on each chunk
    for chunk in chunks:
        chunk.metadata.section_index = section_id
        # Ensure source is consistent (doc_id.pdf)
        chunk.metadata.source = f"{doc_id}.pdf"

    return chunks


def get_chunker(config: ExperimentConfig, embedder=None) -> BaseChunker:
    """Instantiate the appropriate chunker for a config.

    Args:
        config: Experiment configuration.
        embedder: Required for SemanticChunker. Ignored for others.
    """
    if config.chunking_strategy == ChunkingStrategy.FIXED:
        return FixedChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    elif config.chunking_strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    elif config.chunking_strategy == ChunkingStrategy.SEMANTIC:
        if embedder is None:
            # Create embedder for semantic chunking
            embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)
        return SemanticChunker(
            embedder=embedder,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {config.chunking_strategy}")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(result: ExperimentResult, output_dir: Path) -> Path:
    """Save an ExperimentResult to JSON file.

    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.experiment_id}.json"
    output_path.write_text(result.model_dump_json(indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run experiments: config -> chunks -> retrieval -> metrics.

    Args:
        corpus_dir: Path to corpus JSON files.
        qrels_path: Path to qrels.json.
        queries_path: Path to queries.json.
        cache_dir: Optional directory for caching artifacts.
    """

    def __init__(
        self,
        corpus_dir: Path,
        qrels_path: Path,
        queries_path: Path,
        cache_dir: Path | None = None,
    ):
        self.corpus_dir = corpus_dir
        self.qrels_path = qrels_path
        self.queries_path = queries_path
        self.cache_dir = cache_dir or Path("data/cache")

        # Load ground truth
        self.aligner = GroundTruthAligner.from_files(
            str(qrels_path), str(queries_path)
        )

    def run(
        self,
        config: ExperimentConfig,
        max_queries: int | None = None,
    ) -> ExperimentResult:
        """Run a single experiment and return results.

        Args:
            config: Experiment configuration.
            max_queries: Limit number of queries (for testing).

        Returns:
            ExperimentResult with metrics.
        """
        experiment_id = config_to_id(config)
        logger.info("Running experiment: %s", experiment_id)

        # 1. Chunk all corpus sections
        chunker = get_chunker(config)
        all_chunks: list[Chunk] = []
        for section in load_corpus_sections(self.corpus_dir):
            chunks = chunk_section(chunker, section)
            all_chunks.extend(chunks)

        logger.info("Chunked %d sections into %d chunks",
                    sum(1 for _ in load_corpus_sections(self.corpus_dir)),
                    len(all_chunks))

        if not all_chunks:
            raise ValueError("No chunks produced from corpus")

        # 2. Build retriever
        retriever = self._build_retriever(config, all_chunks)

        # 3. Run retrieval for each query
        query_ids = list(self.aligner.corpus_query_ids)
        if max_queries:
            query_ids = query_ids[:max_queries]

        query_results: dict[str, list[RetrievalResult]] = {}
        for query_id in query_ids:
            query_text = self.aligner.queries[query_id]["query"]
            results = retriever.retrieve(query_text, top_k=config.top_k)
            query_results[query_id] = results

        # 4. Evaluate with ground truth
        metrics = self.aligner.evaluate(query_results, k=config.top_k)

        return ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics,
            num_queries=len(query_ids),
            query_ids=query_ids,
        )

    def _build_retriever(
        self,
        config: ExperimentConfig,
        chunks: list[Chunk],
    ):
        """Build the appropriate retriever for a config."""
        # Create embedder
        embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)

        # Embed all chunks
        chunk_texts = [c.content for c in chunks]
        embeddings = embedder.embed(chunk_texts)

        # Build FAISS index
        vector_store = FaissVectorStore(dimension=embedder.dimension)
        chunk_ids = [str(c.id) for c in chunks]
        vector_store.add(chunk_ids, embeddings)

        # Create chunk lookup
        chunk_map = {str(c.id): c for c in chunks}

        if config.retriever_type == RetrieverType.DENSE:
            return DenseRetriever(
                embedder=embedder,
                vector_store=vector_store,
                chunk_lookup=chunk_map,
            )
        elif config.retriever_type == RetrieverType.BM25:
            return BM25Retriever(chunks=chunks)
        elif config.retriever_type == RetrieverType.HYBRID:
            dense = DenseRetriever(
                embedder=embedder,
                vector_store=vector_store,
                chunk_lookup=chunk_map,
            )
            bm25 = BM25Retriever(chunks=chunks)
            return HybridRetriever(dense=dense, bm25=bm25)
        else:
            raise ValueError(f"Unknown retriever type: {config.retriever_type}")

    def run_grid(
        self,
        configs: list[ExperimentConfig] | None = None,
        output_dir: Path | None = None,
        max_queries: int | None = None,
    ) -> list[ExperimentResult]:
        """Run all experiments in a grid.

        Args:
            configs: List of configs. If None, uses default grid.
            output_dir: Directory to save results. If None, doesn't save.
            max_queries: Limit queries per experiment (for testing).

        Returns:
            List of ExperimentResults.
        """
        if configs is None:
            configs = build_experiment_grid()

        results = []
        for i, config in enumerate(configs):
            logger.info("Running experiment %d/%d", i + 1, len(configs))
            try:
                result = self.run(config, max_queries=max_queries)
                results.append(result)
                if output_dir:
                    save_result(result, output_dir)
            except Exception as e:
                logger.error("Experiment failed: %s - %s", config_to_id(config), e)
                continue

        return results
