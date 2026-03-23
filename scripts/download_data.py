"""Download Open RAG Benchmark data for PaperSearch.

Usage:
    python scripts/download_data.py              # 50 papers (dev, default)
    python scripts/download_data.py --n-papers 1000  # full dataset

Steps:
    1. Download benchmark JSONs (queries, answers, qrels, pdf_urls) from HuggingFace
    2. Select N papers — relevant docs first (sorted by query diversity), then
       distractors to fill remaining slots
    3. Download corpus JSONs for selected papers (parsed sections with ground truth)
    4. Download PDFs from arXiv for selected papers

All downloads are idempotent — existing files are skipped.
"""

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "pdf" / "arxiv"
CORPUS_DIR = DATA_DIR / "corpus"
PDF_DIR = DATA_DIR / "pdfs"
SUBSET_FILE = PROJECT_ROOT / "data" / "subset.json"

HF_BASE = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
BENCHMARK_FILES = ["queries.json", "answers.json", "qrels.json", "pdf_urls.json"]

TOTAL_PAPERS = 1000


# ---------------------------------------------------------------------------
# Step 1: Benchmark JSONs
# ---------------------------------------------------------------------------

def download_benchmark_files() -> None:
    """Download the 4 benchmark JSON files if not already present."""
    print("Step 1: Benchmark files")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in BENCHMARK_FILES:
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"  {filename} — already exists")
            continue
        url = f"{HF_BASE}/{filename}"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  {filename} — downloaded")
        except Exception as e:
            raise RuntimeError(f"Failed to download {filename}: {e}") from e


# ---------------------------------------------------------------------------
# Step 2: Select papers
# ---------------------------------------------------------------------------

def select_papers(n: int) -> list[str]:
    """Select N papers: relevant docs first (by query diversity), then distractors."""
    print(f"\nStep 2: Select {n} papers")

    with open(DATA_DIR / "qrels.json") as f:
        qrels = json.load(f)
    with open(DATA_DIR / "queries.json") as f:
        queries = json.load(f)
    with open(DATA_DIR / "pdf_urls.json") as f:
        all_doc_ids = list(json.load(f).keys())

    # Group queries by document
    doc_to_queries: dict[str, list[tuple[str, dict]]] = {}
    for qid, qrel in qrels.items():
        did = qrel["doc_id"]
        if did not in doc_to_queries:
            doc_to_queries[did] = []
        doc_to_queries[did].append((qid, queries[qid]))

    # Rank relevant docs by query diversity
    relevant_scored = []
    for did, qs in doc_to_queries.items():
        types = set(q["type"] for _, q in qs)
        sources = set(q["source"] for _, q in qs)
        relevant_scored.append((did, len(types) + len(sources), len(qs)))
    relevant_scored.sort(key=lambda x: (-x[1], -x[2]))

    relevant_ids = [d[0] for d in relevant_scored]
    n_relevant = min(n, len(relevant_ids))
    selected = relevant_ids[:n_relevant]

    # Fill remaining slots with distractors
    if n > n_relevant:
        relevant_set = set(relevant_ids)
        distractors = [d for d in all_doc_ids if d not in relevant_set]
        n_distractors = min(n - n_relevant, len(distractors))
        selected.extend(distractors[:n_distractors])

    # Compute stats
    selected_set = set(selected)
    selected_qids = [qid for qid, qrel in qrels.items() if qrel["doc_id"] in selected_set]
    type_counts = Counter(queries[qid]["type"] for qid in selected_qids)
    source_counts = Counter(queries[qid]["source"] for qid in selected_qids)

    result = {
        "n_requested": n,
        "doc_ids": selected,
        "stats": {
            "num_docs": len(selected),
            "num_relevant": n_relevant,
            "num_distractors": len(selected) - n_relevant,
            "num_queries": len(selected_qids),
            "query_types": dict(type_counts),
            "query_sources": dict(source_counts),
        },
    }

    SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {n_relevant} relevant docs, {len(selected) - n_relevant} distractors")
    print(f"  {len(selected_qids)} queries across {n_relevant} relevant docs")
    print(f"  Saved to {SUBSET_FILE.relative_to(PROJECT_ROOT)}")

    return selected


# ---------------------------------------------------------------------------
# Step 3: Corpus JSONs
# ---------------------------------------------------------------------------

def download_corpus(doc_ids: list[str]) -> None:
    """Download corpus JSON files from HuggingFace."""
    print(f"\nStep 3: Corpus files ({len(doc_ids)} papers)")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    already = {p.stem for p in CORPUS_DIR.glob("*.json")}
    to_download = [d for d in doc_ids if d not in already]
    print(f"  {len(already)} cached, {len(to_download)} to fetch")

    for i, doc_id in enumerate(to_download):
        url = f"{HF_BASE}/corpus/{doc_id}.json"
        dest = CORPUS_DIR / f"{doc_id}.json"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [{i+1}/{len(to_download)}] {doc_id}.json")
        except Exception as e:
            print(f"  [{i+1}/{len(to_download)}] {doc_id}.json FAILED: {e}")
        if i < len(to_download) - 1:
            time.sleep(0.5)

    print(f"  {len(list(CORPUS_DIR.glob('*.json')))} total corpus files")


# ---------------------------------------------------------------------------
# Step 4: PDFs from arXiv
# ---------------------------------------------------------------------------

def download_pdfs(doc_ids: list[str]) -> None:
    """Download PDFs from arXiv. 1-second delay per request."""
    print(f"\nStep 4: PDFs from arXiv ({len(doc_ids)} papers)")
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "pdf_urls.json") as f:
        pdf_urls = json.load(f)

    already = {p.stem for p in PDF_DIR.glob("*.pdf")}
    to_download = [d for d in doc_ids if d not in already]
    print(f"  {len(already)} cached, {len(to_download)} to fetch")

    failed = []
    for i, doc_id in enumerate(to_download):
        url = pdf_urls.get(doc_id)
        if not url:
            print(f"  [{i+1}/{len(to_download)}] {doc_id} — no URL")
            failed.append(doc_id)
            continue

        dest = PDF_DIR / f"{doc_id}.pdf"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PaperSearch-RAG-Research/1.0"})
            with urllib.request.urlopen(req) as resp:
                dest.write_bytes(resp.read())
            size_kb = dest.stat().st_size / 1024
            print(f"  [{i+1}/{len(to_download)}] {doc_id}.pdf ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  [{i+1}/{len(to_download)}] {doc_id}.pdf FAILED: {e}")
            failed.append(doc_id)

        if i < len(to_download) - 1:
            time.sleep(1)

    downloaded = len(list(PDF_DIR.glob("*.pdf")))
    print(f"  {downloaded} total PDFs")
    if failed:
        print(f"  Failed ({len(failed)}): {failed}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download Open RAG Benchmark data")
    parser.add_argument(
        "--n-papers", type=int, default=50,
        help=f"Number of papers to download (default: 50 for dev, max: {TOTAL_PAPERS})",
    )
    args = parser.parse_args()

    n = min(args.n_papers, TOTAL_PAPERS)
    if n < 1:
        parser.error("--n-papers must be at least 1")

    print(f"=== Downloading data for {n} papers ===\n")

    download_benchmark_files()
    doc_ids = select_papers(n)
    download_corpus(doc_ids)
    download_pdfs(doc_ids)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
