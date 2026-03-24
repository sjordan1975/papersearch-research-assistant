#!/usr/bin/env python3
"""Run RAG experiments and output evaluation metrics.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --max-queries 50
    python scripts/evaluate.py --chunking fixed --embedding all-MiniLM-L6-v2 --retriever dense
    python scripts/evaluate.py --grid

Runs the experiment grid (or a single config) and outputs IR metrics
(Precision@K, Recall@K, MRR, NDCG@K) as JSON.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment.runner import (
    ExperimentRunner,
    build_experiment_grid,
    config_to_id,
    save_result,
)
from src.models import ChunkingStrategy, ExperimentConfig, RetrieverType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_metrics(result) -> str:
    """Format experiment result for display."""
    lines = [
        f"\nExperiment: {result.experiment_id}",
        "-" * 50,
        f"  Chunking:  {result.config.chunking_strategy.value}",
        f"  Chunk size: {result.config.chunk_size}",
        f"  Embedding: {result.config.embedding_model}",
        f"  Retriever: {result.config.retriever_type.value}",
        f"  Top-K:     {result.config.top_k}",
        "-" * 50,
        f"  Queries evaluated: {result.num_queries}",
        "-" * 50,
        "  IR Metrics:",
        f"    Precision@{result.metrics.k}: {result.metrics.precision_at_k:.4f}",
        f"    Recall@{result.metrics.k}:    {result.metrics.recall_at_k:.4f}",
        f"    MRR:              {result.metrics.mrr:.4f}",
        f"    NDCG@{result.metrics.k}:       {result.metrics.ndcg_at_k:.4f}",
    ]
    if result.judge_scores:
        lines.extend([
            "-" * 50,
            "  Judge Scores:",
            f"    Relevance:    {result.judge_scores.relevance:.2f}",
            f"    Accuracy:     {result.judge_scores.accuracy:.2f}",
            f"    Completeness: {result.judge_scores.completeness:.2f}",
            f"    Citation:     {result.judge_scores.citation_quality:.2f}",
            f"    Average:      {result.judge_scores.average:.2f}",
        ])
    if result.duration_seconds is not None:
        lines.extend([
            "-" * 50,
            f"  Duration: {result.duration_seconds:.1f}s",
        ])
    lines.append("-" * 50)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG experiments and evaluate",
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
        "--qrels",
        type=Path,
        default=Path("data/pdf/arxiv/qrels.json"),
        help="Path to qrels.json",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/pdf/arxiv/queries.json"),
        help="Path to queries.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save result JSON files (default: outputs/)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries (for testing)",
    )

    # Single config options
    parser.add_argument(
        "--chunking",
        choices=["fixed", "recursive", "semantic"],
        default=None,
        help="Chunking strategy (default: run grid)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Chunk overlap (default: 0)",
    )
    parser.add_argument(
        "--embedding",
        default=None,
        help="Embedding model (default: run grid)",
    )
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default=None,
        help="Retriever type (default: run grid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K for retrieval (default: 5)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid fusion alpha (default: 0.5). 1.0 = pure dense, 0.0 = pure BM25.",
    )
    parser.add_argument(
        "--reranker",
        default=None,
        help="Reranker model (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2). None = no reranking.",
    )

    # Grid option
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run full experiment grid (overrides single config options)",
    )

    args = parser.parse_args()

    # Validate data exists
    if not args.corpus_dir.exists():
        logger.error("Corpus directory not found: %s", args.corpus_dir)
        logger.error("Run 'python scripts/download_data.py' first")
        sys.exit(1)

    if not args.qrels.exists() or not args.queries.exists():
        logger.error("qrels.json or queries.json not found")
        logger.error("Run 'python scripts/download_data.py' first")
        sys.exit(1)

    # Initialize runner
    runner = ExperimentRunner(
        corpus_dir=args.corpus_dir,
        qrels_path=args.qrels,
        queries_path=args.queries,
    )

    # Determine configs to run
    if args.grid:
        configs = build_experiment_grid(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
        )
        logger.info("Running full grid: %d configurations", len(configs))
    elif args.chunking or args.embedding or args.retriever:
        # Single config
        chunking = ChunkingStrategy(args.chunking) if args.chunking else ChunkingStrategy.FIXED
        embedding = args.embedding or "all-MiniLM-L6-v2"
        retriever = RetrieverType(args.retriever) if args.retriever else RetrieverType.HYBRID

        configs = [ExperimentConfig(
            chunking_strategy=chunking,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=embedding,
            retriever_type=retriever,
            top_k=args.top_k,
            hybrid_alpha=args.alpha,
            reranker=args.reranker,
        )]
        logger.info("Running single config: %s", config_to_id(configs[0]))
    else:
        # Default: single config with defaults
        configs = [ExperimentConfig(
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model="all-MiniLM-L6-v2",
            retriever_type=RetrieverType.HYBRID,
            top_k=args.top_k,
            hybrid_alpha=args.alpha,
            reranker=args.reranker,
        )]
        logger.info("Running default config: %s", config_to_id(configs[0]))

    # Run experiments
    results = []
    for i, config in enumerate(configs, 1):
        logger.info("Running experiment %d/%d: %s", i, len(configs), config_to_id(config))
        try:
            result = runner.run(config, max_queries=args.max_queries)
            results.append(result)
            print(format_metrics(result))

            if args.output_dir:
                save_result(result, args.output_dir)
                logger.info("Saved to: %s/%s.json", args.output_dir, result.experiment_id)

        except Exception as e:
            logger.error("Experiment failed: %s", e)
            continue

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Config':<50} {'MRR':>8} {'NDCG':>8}")
        print("-" * 68)
        for r in sorted(results, key=lambda x: x.metrics.mrr, reverse=True):
            print(f"{r.experiment_id:<50} {r.metrics.mrr:>8.4f} {r.metrics.ndcg_at_k:>8.4f}")
        print("=" * 60)

    # Output JSON summary if output dir specified
    if args.output_dir:
        summary_path = args.output_dir / "summary.json"
        summary = [
            {
                "experiment_id": r.experiment_id,
                "mrr": r.metrics.mrr,
                "ndcg": r.metrics.ndcg_at_k,
                "precision": r.metrics.precision_at_k,
                "recall": r.metrics.recall_at_k,
            }
            for r in results
        ]
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
