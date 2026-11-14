"""
run_heldout_eval.py: Evaluate BM25, dense, and hybrid retrievers on held-out table questions.

For each question in eval/heldout_tables.json, chunks the question's PDF document
using the project's existing chunker, retrieves the top 5 chunks with each retriever
(BM25, dense bge-small, hybrid RRF k=60), and checks whether the retrieved chunks
contain all must_contain strings (case-insensitive substring match).

Prints per-retriever Recall@1, Recall@3, Recall@5 (counts out of 15) and per-question
hit ranks. Writes the same output to eval/results/heldout_baseline_<timestamp>.txt.

Usage:
    python -m scripts.run_heldout_eval
"""

import json
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

# Project root is one level above scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extractor import extract_text_from_pdf
from src.chunker import chunk_document
from src.retriever import retrieve as bm25_retrieve
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever


def is_hit(chunk_text: str, must_contain: list[str]) -> bool:
    """Check if a chunk contains all must_contain strings (case-insensitive)."""
    if not must_contain:
        return True
    text_lower = chunk_text.lower()
    return all(s.lower() in text_lower for s in must_contain)


def find_first_hit_rank(results, must_contain: list[str], max_k: int = 5) -> str:
    """Return the 1-based rank of the first hit, or 'miss'."""
    for i, result in enumerate(results[:max_k]):
        if is_hit(result.passage, must_contain):
            return str(i + 1)
    return "miss"


def count_hits_at_k(results, must_contain: list[str], k: int) -> bool:
    """True if any chunk in the top k is a hit."""
    for result in results[:k]:
        if is_hit(result.passage, must_contain):
            return True
    return False


def main():
    # Load held-out questions
    heldout_path = PROJECT_ROOT / "eval" / "heldout_tables.json"
    with open(heldout_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)

    # Pre-extract and chunk each unique document once
    papers_dir = PROJECT_ROOT / "papers"
    doc_chunks: dict[str, list] = {}
    for q in questions:
        doc_name = q["doc"]
        if doc_name not in doc_chunks:
            pdf_path = papers_dir / doc_name
            doc_result = extract_text_from_pdf(pdf_path)
            chunks = chunk_document(doc_result)
            doc_chunks[doc_name] = chunks

    # Initialize retrievers
    dense_retriever = DenseRetriever()
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retrieve,
        dense_retriever=dense_retriever,
        k=60,
    )

    # Collect results per retriever
    retriever_names = ["BM25", "Dense", "Hybrid"]
    # Per-question results: {retriever_name: [(item_id, hit_rank_str), ...]}
    per_question: dict[str, list[tuple[str, str]]] = {name: [] for name in retriever_names}
    # Recall counters: {retriever_name: {k: count}}
    recall_counts: dict[str, dict[int, int]] = {
        name: {1: 0, 3: 0, 5: 0} for name in retriever_names
    }

    for q in questions:
        item_id = q["id"]
        query = q["question"]
        must_contain = q.get("must_contain", [])
        chunks = doc_chunks[q["doc"]]

        # BM25
        bm25_results = bm25_retrieve(chunks, query, top_k=5)
        bm25_rank = find_first_hit_rank(bm25_results, must_contain)
        per_question["BM25"].append((item_id, bm25_rank))
        for k in [1, 3, 5]:
            if count_hits_at_k(bm25_results, must_contain, k):
                recall_counts["BM25"][k] += 1

        # Dense
        dense_results = dense_retriever.retrieve(chunks, query, top_k=5)
        dense_rank = find_first_hit_rank(dense_results, must_contain)
        per_question["Dense"].append((item_id, dense_rank))
        for k in [1, 3, 5]:
            if count_hits_at_k(dense_results, must_contain, k):
                recall_counts["Dense"][k] += 1

        # Hybrid
        hybrid_results = hybrid_retriever.retrieve(chunks, query, top_k=5)
        hybrid_rank = find_first_hit_rank(hybrid_results, must_contain)
        per_question["Hybrid"].append((item_id, hybrid_rank))
        for k in [1, 3, 5]:
            if count_hits_at_k(hybrid_results, must_contain, k):
                recall_counts["Hybrid"][k] += 1

    # Build output
    buf = StringIO()

    buf.write("=== Held-Out Table Retrieval Baseline ===\n\n")

    # Per-retriever recall
    for name in retriever_names:
        counts = recall_counts[name]
        buf.write(
            f"{name}: R@1 {counts[1]}/{total}, "
            f"R@3 {counts[3]}/{total}, "
            f"R@5 {counts[5]}/{total}\n"
        )

    buf.write("\n=== Per-Question Hit Ranks ===\n\n")

    # Header
    buf.write(f"{'ID':<20} {'BM25':>6} {'Dense':>6} {'Hybrid':>6}\n")
    buf.write("-" * 42 + "\n")

    for i in range(total):
        item_id = per_question["BM25"][i][0]
        bm25_rank = per_question["BM25"][i][1]
        dense_rank = per_question["Dense"][i][1]
        hybrid_rank = per_question["Hybrid"][i][1]
        buf.write(f"{item_id:<20} {bm25_rank:>6} {dense_rank:>6} {hybrid_rank:>6}\n")

    output = buf.getvalue()

    # Print to stdout
    print(output, end="")

    # Write to file
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"heldout_baseline_{timestamp}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()

