#!/usr/bin/env python3
"""Generate visualizations from experiment results.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --output-dir figures/
    python scripts/visualize.py --show

Reads experiment results from outputs/ and generates T17 visualizations:
1. Heatmap: MRR by chunking × retriever
2. Dimension impact: bar chart comparing retriever types
3. Radar chart: multi-metric comparison for top configs
4. Metric comparison: grouped bar chart across all configs
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(output_dir: Path) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    results = []
    for result_file in output_dir.glob("*.json"):
        if result_file.name == "summary.json":
            continue
        try:
            data = json.loads(result_file.read_text())
            results.append({
                "experiment_id": data["experiment_id"],
                "chunking": data["config"]["chunking_strategy"],
                "chunk_size": data["config"]["chunk_size"],
                "embedding": data["config"]["embedding_model"],
                "retriever": data["config"]["retriever_type"],
                "top_k": data["config"]["top_k"],
                "hybrid_alpha": data["config"].get("hybrid_alpha", 0.5),
                "precision": data["metrics"]["precision_at_k"],
                "recall": data["metrics"]["recall_at_k"],
                "mrr": data["metrics"]["mrr"],
                "ndcg": data["metrics"]["ndcg_at_k"],
                "num_queries": data["num_queries"],
                "duration_seconds": data.get("duration_seconds"),
                "reranker": data["config"].get("reranker"),
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {result_file}: {e}")
            continue

    if not results:
        raise ValueError(f"No valid results found in {output_dir}")

    return pd.DataFrame(results)


def plot_heatmap(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Heatmap: MRR by chunking strategy × retriever type."""
    # Pivot for heatmap
    pivot = df.pivot_table(
        values="mrr",
        index="chunking",
        columns="retriever",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "MRR"},
        ax=ax,
    )
    ax.set_title("MRR by Chunking Strategy × Retriever Type")
    ax.set_xlabel("Retriever Type")
    ax.set_ylabel("Chunking Strategy")

    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_mrr.png", dpi=150)
    print(f"Saved: {output_dir / 'heatmap_mrr.png'}")

    if show:
        plt.show()
    plt.close(fig)


def plot_retriever_impact(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Bar chart: impact of retriever type on MRR."""
    fig, ax = plt.subplots(figsize=(8, 5))

    retriever_means = df.groupby("retriever")["mrr"].mean().sort_values(ascending=False)

    colors = ["#2ecc71" if r == retriever_means.index[0] else "#3498db" for r in retriever_means.index]
    bars = ax.bar(retriever_means.index, retriever_means.values, color=colors, edgecolor="black")

    ax.set_xlabel("Retriever Type")
    ax.set_ylabel("Mean MRR")
    ax.set_title("Retriever Type Impact on MRR")
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, val in zip(bars, retriever_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "retriever_impact.png", dpi=150)
    print(f"Saved: {output_dir / 'retriever_impact.png'}")

    if show:
        plt.show()
    plt.close(fig)


def plot_radar(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Radar chart: multi-metric comparison for top configs."""
    # Select top 3 configs by MRR
    top_configs = df.nlargest(3, "mrr")

    metrics = ["precision", "recall", "mrr", "ndcg"]
    metric_labels = ["Precision@K", "Recall@K", "MRR", "NDCG@K"]

    # Normalize recall (can be > 1) for radar chart
    df_norm = top_configs.copy()
    df_norm["recall"] = df_norm["recall"] / df_norm["recall"].max()

    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Close polygon

        label = f"{row['chunking']}/{row['retriever']}"
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Top 3 Configs: Multi-Metric Comparison", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    fig.savefig(output_dir / "radar_top_configs.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'radar_top_configs.png'}")

    if show:
        plt.show()
    plt.close(fig)


def plot_grouped_bars(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Grouped bar chart: all metrics across all configs."""
    # Create short labels
    df = df.copy()
    df["label"] = df["chunking"].str[:3] + "/" + df["retriever"].str[:3]

    # Sort by MRR
    df = df.sort_values("mrr", ascending=False)

    metrics = ["mrr", "ndcg", "precision"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric.upper())

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("IR Metrics Across Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "grouped_metrics.png", dpi=150)
    print(f"Saved: {output_dir / 'grouped_metrics.png'}")

    if show:
        plt.show()
    plt.close(fig)


def plot_chunking_comparison(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Bar chart: chunking strategy impact on MRR."""
    fig, ax = plt.subplots(figsize=(8, 5))

    chunking_means = df.groupby("chunking")["mrr"].mean().sort_values(ascending=False)

    colors = ["#9b59b6" if c == chunking_means.index[0] else "#95a5a6" for c in chunking_means.index]
    bars = ax.bar(chunking_means.index, chunking_means.values, color=colors, edgecolor="black")

    ax.set_xlabel("Chunking Strategy")
    ax.set_ylabel("Mean MRR")
    ax.set_title("Chunking Strategy Impact on MRR")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, chunking_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "chunking_impact.png", dpi=150)
    print(f"Saved: {output_dir / 'chunking_impact.png'}")

    if show:
        plt.show()
    plt.close(fig)


def plot_latency(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Box plot: query latency across configurations."""
    if df["duration_seconds"].isna().all():
        print("Skipping latency plot: no duration data in results")
        return

    df = df.dropna(subset=["duration_seconds"]).copy()
    df["label"] = df["chunking"].str[:3] + "/" + df["retriever"].str[:3]
    df = df.sort_values("duration_seconds")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = sns.color_palette("Set2", len(df))
    bars = ax.bar(df["label"], df["duration_seconds"], color=colors, edgecolor="black")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Experiment Duration by Configuration")

    for bar, val in zip(bars, df["duration_seconds"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "latency.png", dpi=150)
    print(f"Saved: {output_dir / 'latency.png'}")

    if show:
        plt.show()
    plt.close(fig)


def plot_before_after(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Before/after chart: compare best config with and without reranking."""
    has_reranker = df[df["reranker"].notna()]
    no_reranker = df[df["reranker"].isna()]

    if has_reranker.empty or no_reranker.empty:
        print("Skipping before/after plot: need results both with and without reranker")
        return

    # Find matching pairs: same config except reranker
    pairs = []
    for _, rr_row in has_reranker.iterrows():
        match = no_reranker[
            (no_reranker["chunking"] == rr_row["chunking"])
            & (no_reranker["embedding"] == rr_row["embedding"])
            & (no_reranker["retriever"] == rr_row["retriever"])
        ]
        if not match.empty:
            pairs.append((match.iloc[0], rr_row))

    if not pairs:
        print("Skipping before/after plot: no matching config pairs found")
        return

    metrics = ["mrr", "ndcg", "precision"]
    metric_labels = ["MRR", "NDCG@5", "Precision@5"]

    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6), squeeze=False)

    for col, (before, after) in enumerate(pairs):
        ax = axes[0, col]
        label = f"{before['chunking'][:3]}/{before['retriever'][:3]}"

        before_vals = [before[m] for m in metrics]
        after_vals = [after[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width / 2, before_vals, width, label="Before (no reranking)",
                        color="#3498db", edgecolor="black")
        bars2 = ax.bar(x + width / 2, after_vals, width, label="After (reranking)",
                        color="#2ecc71", edgecolor="black")

        # Add delta labels
        for i, (b, a) in enumerate(zip(before_vals, after_vals)):
            delta = a - b
            sign = "+" if delta >= 0 else ""
            ax.text(x[i] + width / 2, max(a, b) + 0.02,
                    f"{sign}{delta:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#2ecc71" if delta >= 0 else "#e74c3c")

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title(f"Before/After Reranking: {label}")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "before_after.png", dpi=150)
    print(f"Saved: {output_dir / 'before_after.png'}")

    if show:
        plt.show()
    plt.close(fig)


def plot_fusion_sweep(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Line chart: NDCG@5 as a function of hybrid alpha."""
    hybrid_df = df[df["retriever"] == "hybrid"].copy()

    if hybrid_df["hybrid_alpha"].nunique() < 2:
        print("Skipping fusion sweep plot: need multiple alpha values (run with --alpha sweep)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for chunking, group in hybrid_df.groupby("chunking"):
        group = group.sort_values("hybrid_alpha")
        ax.plot(group["hybrid_alpha"], group["ndcg"], "o-", label=chunking, linewidth=2, markersize=8)

    ax.set_xlabel("Fusion Weight α (1.0 = pure dense, 0.0 = pure BM25)")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Hybrid Fusion Weight Sweep")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fusion_sweep.png", dpi=150)
    print(f"Saved: {output_dir / 'fusion_sweep.png'}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing experiment result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save visualization images (use docs/assets for final README)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        print("Run 'python scripts/evaluate.py' first")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} experiment results")
    print(f"\nConfigs: {df['experiment_id'].tolist()}")

    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_heatmap(df, args.output_dir, args.show)
    plot_retriever_impact(df, args.output_dir, args.show)
    plot_chunking_comparison(df, args.output_dir, args.show)
    plot_radar(df, args.output_dir, args.show)
    plot_grouped_bars(df, args.output_dir, args.show)
    plot_latency(df, args.output_dir, args.show)
    plot_fusion_sweep(df, args.output_dir, args.show)
    plot_before_after(df, args.output_dir, args.show)

    print(f"\nAll visualizations saved to: {args.output_dir}/")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    summary = df[["chunking", "retriever", "mrr", "ndcg", "precision"]].sort_values("mrr", ascending=False)
    print(summary.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
