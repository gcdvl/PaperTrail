"""
compare_runs.py: Side-by-side comparison of PaperTrail retrieval evaluation runs.

Loads two or more evaluation results JSON files, prints a side-by-side comparison table
of page-level and passage-level Recall@1/3/5 and MRR, and lists the specific items each
retriever gets right that the other gets wrong.

Usage:
    python scripts/compare_runs.py eval/results/run1.json eval/results/run2.json [run3.json ...]
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


def _load_run(path: Path) -> Dict[str, Any]:
    """Loads and validates an evaluation results JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" not in data or "details" not in data:
        raise ValueError(f"File {path.name} is missing 'summary' or 'details' keys.")
    return data


def _get_run_label(data: Dict[str, Any], path: Path) -> str:
    """Derives a concise, descriptive label for a run."""
    summary = data.get("summary", {})
    retriever = summary.get("retriever")
    model = summary.get("model_name")

    if retriever:
        retriever_name = retriever.upper()
        if model:
            short_model = model.split("/")[-1]
            return f"{retriever_name} ({short_model})"
        return retriever_name
    return path.stem


def format_table(runs: List[Tuple[str, Dict[str, Any]]]) -> str:
    """Formats a side-by-side metric comparison table for the given runs."""
    lines: List[str] = []
    max_label_len = max(len(label) for label, _ in runs) if runs else 20
    col_width = max(26, max_label_len + 4)
    label_width = 28

    header = f"{'Metric':<{label_width}}" + "".join(
        f"{label:>{col_width}}" for label, _ in runs
    )
    separator = "-" * len(header)

    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    # Page-level rows
    lines.append("PAGE-LEVEL (Answerable)")
    metric_keys_page = [
        ("Page Recall@1", "mean_recall@1"),
        ("Page Recall@3", "mean_recall@3"),
        ("Page Recall@5", "mean_recall@5"),
        ("Page MRR", "mean_reciprocal_rank"),
    ]
    for row_name, key in metric_keys_page:
        row_str = f"  {row_name:<{label_width - 2}}"
        for _, run_data in runs:
            pl = run_data.get("summary", {}).get("page_level", {})
            val = pl.get(key)
            if val is not None:
                if "recall" in key:
                    row_str += f"{val * 100:>20.1f}%  "
                else:
                    row_str += f"{val:>22.4f}  "
            else:
                row_str += f"{'N/A':>{col_width}}"
        lines.append(row_str)

    lines.append(separator)

    # Passage-level rows
    lines.append("PASSAGE-LEVEL (Evidence Grounding)")
    metric_keys_passage = [
        ("Passage Recall@1", "mean_recall@1"),
        ("Passage Recall@3", "mean_recall@3"),
        ("Passage Recall@5", "mean_recall@5"),
        ("Passage MRR", "mean_reciprocal_rank"),
    ]
    for row_name, key in metric_keys_passage:
        row_str = f"  {row_name:<{label_width - 2}}"
        for _, run_data in runs:
            psl = run_data.get("summary", {}).get("passage_level", {})
            val = psl.get(key)
            if val is not None:
                if "recall" in key:
                    row_str += f"{val * 100:>20.1f}%  "
                else:
                    row_str += f"{val:>22.4f}  "
            else:
                row_str += f"{'N/A':>{col_width}}"
        lines.append(row_str)

    lines.append(separator)
    return "\n".join(lines)


def compare_item_outcomes(
    run_a: Tuple[str, Dict[str, Any]],
    run_b: Tuple[str, Dict[str, Any]],
    k: int = 1,
) -> str:
    """
    Finds items that run_a got right and run_b got wrong, and vice versa.
    Prioritizes passage-level recall at cutoff k, falling back to page-level.
    """
    label_a, data_a = run_a
    label_b, data_b = run_b

    details_a = {item["id"]: item for item in data_a.get("details", []) if item.get("answerable", True)}
    details_b = {item["id"]: item for item in data_b.get("details", []) if item.get("answerable", True)}

    common_ids = sorted(set(details_a.keys()) & set(details_b.keys()))

    def is_correct(item: Dict[str, Any]) -> bool:
        # Check passage-level hit at k first
        if f"passage_recall@{k}" in item:
            return float(item[f"passage_recall@{k}"]) > 0.0
        # Fall back to page-level hit at k
        if f"page_recall@{k}" in item:
            return float(item[f"page_recall@{k}"]) > 0.0
        return False

    a_right_b_wrong = []
    b_right_a_wrong = []

    for item_id in common_ids:
        item_a = details_a[item_id]
        item_b = details_b[item_id]

        correct_a = is_correct(item_a)
        correct_b = is_correct(item_b)

        if correct_a and not correct_b:
            a_right_b_wrong.append((item_id, item_a["question"], item_a.get("evidence_phrase")))
        elif correct_b and not correct_a:
            b_right_a_wrong.append((item_id, item_b["question"], item_b.get("evidence_phrase")))

    lines: List[str] = []
    lines.append(f"\nPairwise Comparison (Passage Recall@{k}): [{label_a}] vs. [{label_b}]")
    lines.append("=" * 72)

    lines.append(f"\n  Items RIGHT in [{label_a}] but WRONG in [{label_b}] ({len(a_right_b_wrong)} items):")
    if a_right_b_wrong:
        for item_id, q, ev in a_right_b_wrong:
            lines.append(f"    - [{item_id}] {q}")
            if ev:
                lines.append(f"      Evidence: \"{ev}\"")
    else:
        lines.append("    (None)")

    lines.append(f"\n  Items RIGHT in [{label_b}] but WRONG in [{label_a}] ({len(b_right_a_wrong)} items):")
    if b_right_a_wrong:
        for item_id, q, ev in b_right_a_wrong:
            lines.append(f"    - [{item_id}] {q}")
            if ev:
                lines.append(f"      Evidence: \"{ev}\"")
    else:
        lines.append("    (None)")

    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of PaperTrail retrieval evaluation results."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to two or more evaluation results JSON files.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Recall cutoff rank for item-by-item win/loss comparison (default: 1).",
    )
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Error: Please provide at least two results JSON files to compare.", file=sys.stderr)
        return 1

    runs: List[Tuple[str, Dict[str, Any]]] = []
    for fpath_str in args.files:
        path = Path(fpath_str)
        try:
            data = _load_run(path)
            label = _get_run_label(data, path)
            runs.append((label, data))
        except Exception as exc:
            print(f"Error loading {path}: {exc}", file=sys.stderr)
            return 1

    print("\n" + "=" * 72)
    print("         PAPERTRAIL MULTI-RETRIEVER BENCHMARK COMPARISON")
    print("=" * 72)
    print(format_table(runs))

    # Pairwise comparison: Run 0 vs all others, and between each pair
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            print(compare_item_outcomes(runs[i], runs[j], k=args.k))

    print("\n" + "=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
