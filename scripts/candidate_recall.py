"""
candidate_recall.py: BM25 passage recall analysis at extended rank cutoffs.

For every answerable item in eval/dataset_all.json, retrieves the top 50
BM25 passages and reports passage Recall@k for k = 1, 3, 5, 10, 20, 30, 50.
Uses the same chunking, retrieval, and evidence-matching logic as run_eval.py.

No network, no LLM, no dense or hybrid retriever.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_document
from src.extractor import extract_text_from_pdf
from src.metrics import passage_contains_evidence
from src.retriever import retrieve


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    papers_dir = PROJECT_ROOT / "papers"
    dataset_path = PROJECT_ROOT / "eval" / "dataset_all.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    answerable = [it for it in items if it.get("answerable")]
    n = len(answerable)

    # Cache chunks per document (same as run_eval.py)
    doc_chunk_cache: dict = {}

    K_VALUES = [1, 3, 5, 10, 20, 30, 50]

    # Per-item results: (item_id, paper, best_rank_or_None)
    item_ranks: list = []
    # Per-paper accumulators
    per_paper_ranks: dict = defaultdict(list)

    for it in answerable:
        doc_name = it["document"]
        question = it["question"]
        evidence_phrase = it["evidence_phrase"]
        item_id = it["id"]
        paper = Path(doc_name).stem

        if doc_name not in doc_chunk_cache:
            pdf_path = papers_dir / doc_name
            if not pdf_path.exists():
                pdf_path = PROJECT_ROOT / doc_name
            doc_result = extract_text_from_pdf(pdf_path)
            doc_chunk_cache[doc_name] = chunk_document(doc_result)

        chunks = doc_chunk_cache[doc_name]
        retrieved = retrieve(chunks, question, top_k=50)

        # Find best 1-based rank where evidence is found
        best_rank = None
        for r in retrieved:
            if passage_contains_evidence(r.passage, evidence_phrase):
                best_rank = r.rank
                break

        item_ranks.append((item_id, paper, best_rank))
        per_paper_ranks[paper].append((item_id, best_rank))

    # ---------- Compute recall counts ----------
    def recall_count(ranks_list, k):
        return sum(1 for _, _, r in ranks_list if r is not None and r <= k)

    lines: list = []
    lines.append("=" * 72)
    lines.append("       CANDIDATE RECALL ANALYSIS  (BM25 top-50)")
    lines.append("=" * 72)
    lines.append(f"  Dataset:       {dataset_path}")
    lines.append(f"  Answerable:    {n}")
    lines.append("")

    # Overall recall table
    lines.append("-" * 72)
    lines.append("  OVERALL PASSAGE RECALL")
    lines.append("-" * 72)
    for k in K_VALUES:
        c = recall_count(item_ranks, k)
        pct = c / n * 100
        lines.append(f"    R@{k:<3d}  {c:>2}/{n}   ({pct:5.1f}%)")

    # Self-check
    r1 = recall_count(item_ranks, 1)
    r3 = recall_count(item_ranks, 3)
    r5 = recall_count(item_ranks, 5)
    expected = (20, 32, 33)
    actual = (r1, r3, r5)
    check = "PASS" if actual == expected else "FAIL"
    lines.append("")
    lines.append(f"  SELF-CHECK: R@1={r1}, R@3={r3}, R@5={r5}  "
                 f"expected 20/45, 32/45, 33/45  => {check}")

    # Per-paper breakdown
    lines.append("")
    lines.append("-" * 72)
    lines.append("  PER-PAPER BREAKDOWN")
    lines.append("-" * 72)
    for paper in sorted(per_paper_ranks.keys()):
        pranks = per_paper_ranks[paper]
        pn = len(pranks)
        lines.append(f"  [{paper}]  (n={pn})")
        for k in K_VALUES:
            c = sum(1 for _, r in pranks if r is not None and r <= k)
            pct = c / pn * 100
            lines.append(f"      R@{k:<3d}  {c:>2}/{pn}   ({pct:5.1f}%)")

    # Items with best rank NOT in 1..5
    lines.append("")
    lines.append("-" * 72)
    lines.append("  ITEMS WITH BEST RANK > 5 (or not retrieved)")
    lines.append("-" * 72)
    lines.append(f"  {'ID':<12s} {'Paper':<14s} {'Best Rank':>10s}")
    lines.append(f"  {'---':<12s} {'---':<14s} {'---':>10s}")
    outside_top5 = [(iid, p, r) for iid, p, r in item_ranks
                    if r is None or r > 5]
    for iid, p, r in outside_top5:
        rank_str = "not retrieved" if r is None else str(r)
        lines.append(f"  {iid:<12s} {p:<14s} {rank_str:>13s}")
    lines.append(f"  Total: {len(outside_top5)}")

    # Rescuable items (rank 6..20)
    rescuable = [(iid, p, r) for iid, p, r in item_ranks
                 if r is not None and 6 <= r <= 20]
    lines.append("")
    lines.append(f"  Rescuable items (rank 6-20): {len(rescuable)}")

    lines.append("")
    lines.append("=" * 72)

    output = "\n".join(lines) + "\n"

    # Print to console
    print(output, end="")

    # Save to file
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"candidate_recall_{timestamp}.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

