"""
run_eval.py: Evaluation script for PaperTrail BM25 ranked retrieval.

Supports both v1 (dataset.json) and v2 (dataset_v2.json) dataset formats.

V2 features:
- Multiple documents (PDFs resolved from papers/ folder or project root).
- Page-level AND passage-level metrics reported separately.
- Per-document metric breakdown.
- Unanswerable items: top-1 BM25 score distribution vs. answerable items.
- Failure listing: for every miss, prints the question, expected evidence
  phrase, and top-3 retrieved passage IDs.

Prints a formatted summary table and saves detailed JSON results to
eval/results/<timestamp>.json.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_document
from src.extractor import extract_text_from_pdf
from src.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    passage_recall_at_k,
    passage_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    score_distribution_stats,
)
from src.dense_retriever import DEFAULT_DENSE_MODEL, DenseRetriever
from src.hybrid_retriever import DEFAULT_RRF_K, HybridRetriever
from src.retriever import retrieve


def _resolve_pdf(doc_name: str, papers_dir: Path, project_root: Path) -> Path:
    """Resolves a document name to a PDF path, checking papers_dir first."""
    pdf_path = papers_dir / doc_name
    if pdf_path.exists():
        return pdf_path
    pdf_path = project_root / doc_name
    if pdf_path.exists():
        return pdf_path
    raise FileNotFoundError(
        f"PDF document '{doc_name}' not found in '{papers_dir}' or '{project_root}'"
    )


def _is_v2_item(item: Dict[str, Any]) -> bool:
    """Checks if a dataset item follows the v2 format."""
    return "answerable" in item


def evaluate_dataset(
    dataset_path: Path,
    output_dir: Path,
    papers_dir: Optional[Path] = None,
    retriever_type: str = "bm25",
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Optional[Path] = None,
    rrf_k: int = DEFAULT_RRF_K,
    weight_bm25: float = 1.0,
    weight_dense: float = 1.0,
) -> Dict[str, Any]:
    """
    Runs retrieval evaluation over a dataset and saves timestamped results.

    Supports both v1 and v2 dataset formats. V2 items include answerable flag,
    evidence_phrase for passage-level metrics, and unanswerable score analysis.

    Args:
        dataset_path: Path to the JSON evaluation dataset.
        output_dir: Directory where results JSON will be written.
        papers_dir: Directory containing PDF documents. Defaults to PROJECT_ROOT/papers.
        retriever_type: 'bm25', 'dense', or 'hybrid'.
        model_name: Embedding model identifier.
        cache_dir: Embedding cache directory.
        rrf_k: RRF smoothing constant.
        weight_bm25: RRF BM25 weight.
        weight_dense: RRF dense weight.

    Returns:
        Dictionary containing aggregate summary metrics and per-item details.
    """
    if papers_dir is None:
        papers_dir = PROJECT_ROOT / "papers"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    is_v2 = any(_is_v2_item(item) for item in data)

    # Initialize retriever engine
    if retriever_type == "dense":
        retriever_obj = DenseRetriever(model_name=model_name, cache_dir=cache_dir)
    elif retriever_type == "hybrid":
        dense_inst = DenseRetriever(model_name=model_name, cache_dir=cache_dir)
        retriever_obj = HybridRetriever(
            dense_retriever=dense_inst,
            k=rrf_k,
            weight_bm25=weight_bm25,
            weight_dense=weight_dense,
        )
    elif retriever_type == "bm25":
        retriever_obj = None
    else:
        raise ValueError(f"Unknown retriever type: '{retriever_type}'. Expected 'bm25', 'dense', or 'hybrid'.")

    # Document chunk cache to avoid re-extracting/chunking for each question
    doc_chunk_cache: Dict[str, Any] = {}

    results_per_item: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    # Page-level accumulators
    page_r1_list: List[float] = []
    page_r3_list: List[float] = []
    page_r5_list: List[float] = []
    page_rr_list: List[float] = []

    # Passage-level accumulators (v2 answerable items only)
    passage_r1_list: List[float] = []
    passage_r3_list: List[float] = []
    passage_r5_list: List[float] = []
    passage_rr_list: List[float] = []

    # Per-document accumulators
    per_doc_page_r1: Dict[str, List[float]] = defaultdict(list)
    per_doc_page_rr: Dict[str, List[float]] = defaultdict(list)
    per_doc_passage_r1: Dict[str, List[float]] = defaultdict(list)
    per_doc_passage_rr: Dict[str, List[float]] = defaultdict(list)

    # Score distributions for unanswerable threshold analysis
    answerable_top1_scores: List[float] = []
    unanswerable_top1_scores: List[float] = []

    for item in data:
        doc_name = item["document"]
        question = item["question"]
        expected_pages = item.get("expected_page_numbers", [])
        answerable = item.get("answerable", True)  # v1 items default to answerable
        evidence_phrase = item.get("evidence_phrase")

        if doc_name not in doc_chunk_cache:
            pdf_path = _resolve_pdf(doc_name, papers_dir, PROJECT_ROOT)
            doc_result = extract_text_from_pdf(pdf_path)
            doc_chunk_cache[doc_name] = chunk_document(doc_result)

        chunks = doc_chunk_cache[doc_name]
        if retriever_obj is not None:
            retrieved = retriever_obj.retrieve(chunks, question, top_k=10)
        else:
            retrieved = retrieve(chunks, question, top_k=10)

        retrieved_pages = [r.page_number for r in retrieved]
        retrieved_chunk_ids = [r.passage_id for r in retrieved]
        retrieved_passages = [r.passage for r in retrieved]
        retrieved_scores = [r.score for r in retrieved]

        # Top-1 score for threshold analysis
        top1_score = retrieved_scores[0] if retrieved_scores else 0.0

        item_result: Dict[str, Any] = {
            "id": item.get("id", ""),
            "document": doc_name,
            "question": question,
            "answerable": answerable,
            "expected_page_numbers": expected_pages,
            "retrieved_page_numbers": retrieved_pages[:5],
            "retrieved_chunk_ids": retrieved_chunk_ids[:5],
            "scores": [round(s, 4) for s in retrieved_scores[:5]],
            "top1_score": round(top1_score, 4),
        }

        if answerable:
            # Page-level metrics (for answerable items)
            r1 = recall_at_k(retrieved_pages, expected_pages, k=1)
            r3 = recall_at_k(retrieved_pages, expected_pages, k=3)
            r5 = recall_at_k(retrieved_pages, expected_pages, k=5)
            rr = reciprocal_rank(retrieved_pages, expected_pages)

            page_r1_list.append(r1)
            page_r3_list.append(r3)
            page_r5_list.append(r5)
            page_rr_list.append(rr)
            per_doc_page_r1[doc_name].append(r1)
            per_doc_page_rr[doc_name].append(rr)

            item_result.update({
                "page_recall@1": r1,
                "page_recall@3": r3,
                "page_recall@5": r5,
                "page_reciprocal_rank": round(rr, 4),
            })

            # Passage-level metrics (v2 with evidence_phrase)
            if evidence_phrase:
                pr1 = passage_recall_at_k(retrieved_passages, evidence_phrase, k=1)
                pr3 = passage_recall_at_k(retrieved_passages, evidence_phrase, k=3)
                pr5 = passage_recall_at_k(retrieved_passages, evidence_phrase, k=5)
                prr = passage_reciprocal_rank(retrieved_passages, evidence_phrase)

                passage_r1_list.append(pr1)
                passage_r3_list.append(pr3)
                passage_r5_list.append(pr5)
                passage_rr_list.append(prr)
                per_doc_passage_r1[doc_name].append(pr1)
                per_doc_passage_rr[doc_name].append(prr)

                item_result.update({
                    "evidence_phrase": evidence_phrase,
                    "passage_recall@1": pr1,
                    "passage_recall@3": pr3,
                    "passage_recall@5": pr5,
                    "passage_reciprocal_rank": round(prr, 4),
                })

                # Failure listing: page-level OR passage-level miss at k=5
                is_page_miss = r5 < 1.0
                is_passage_miss = pr5 < 1.0
                if is_page_miss or is_passage_miss:
                    failures.append({
                        "id": item.get("id", ""),
                        "question": question,
                        "evidence_phrase": evidence_phrase,
                        "page_miss": is_page_miss,
                        "passage_miss": is_passage_miss,
                        "top3_passage_ids": retrieved_chunk_ids[:3],
                    })
            else:
                # v1 style: page-level failure listing
                if r5 < 1.0:
                    failures.append({
                        "id": item.get("id", ""),
                        "question": question,
                        "evidence_phrase": None,
                        "page_miss": True,
                        "passage_miss": None,
                        "top3_passage_ids": retrieved_chunk_ids[:3],
                    })

            answerable_top1_scores.append(top1_score)
        else:
            # Unanswerable items: record score only
            unanswerable_top1_scores.append(top1_score)
            item_result["unanswerable_top1_score"] = round(top1_score, 4)

        results_per_item.append(item_result)

    # Build summary
    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retriever": retriever_type,
        "dataset_format": "v2" if is_v2 else "v1",
        "total_queries": len(data),
        "answerable_count": len(answerable_top1_scores),
        "unanswerable_count": len(unanswerable_top1_scores),
    }

    if retriever_type in ("dense", "hybrid"):
        summary["model_name"] = model_name
    if retriever_type == "hybrid":
        summary["rrf_k"] = rrf_k
        summary["weight_bm25"] = weight_bm25
        summary["weight_dense"] = weight_dense

    # Page-level summary
    if page_r1_list:
        summary["page_level"] = {
            "mean_recall@1": round(mean_recall_at_k(page_r1_list), 4),
            "mean_recall@3": round(mean_recall_at_k(page_r3_list), 4),
            "mean_recall@5": round(mean_recall_at_k(page_r5_list), 4),
            "mean_reciprocal_rank": round(mean_reciprocal_rank(page_rr_list), 4),
        }

    # Passage-level summary (v2 only)
    if passage_r1_list:
        summary["passage_level"] = {
            "mean_recall@1": round(mean_recall_at_k(passage_r1_list), 4),
            "mean_recall@3": round(mean_recall_at_k(passage_r3_list), 4),
            "mean_recall@5": round(mean_recall_at_k(passage_r5_list), 4),
            "mean_reciprocal_rank": round(mean_reciprocal_rank(passage_rr_list), 4),
        }

    # Per-document breakdown
    all_docs = sorted(set(
        list(per_doc_page_r1.keys()) + list(per_doc_passage_r1.keys())
    ))
    if len(all_docs) > 0:
        per_doc_summary: Dict[str, Any] = {}
        for doc in all_docs:
            doc_metrics: Dict[str, Any] = {}
            if doc in per_doc_page_r1:
                doc_metrics["page_level"] = {
                    "mean_recall@1": round(mean_recall_at_k(per_doc_page_r1[doc]), 4),
                    "mean_reciprocal_rank": round(mean_reciprocal_rank(per_doc_page_rr[doc]), 4),
                    "query_count": len(per_doc_page_r1[doc]),
                }
            if doc in per_doc_passage_r1:
                doc_metrics["passage_level"] = {
                    "mean_recall@1": round(mean_recall_at_k(per_doc_passage_r1[doc]), 4),
                    "mean_reciprocal_rank": round(mean_reciprocal_rank(per_doc_passage_rr[doc]), 4),
                    "query_count": len(per_doc_passage_r1[doc]),
                }
            per_doc_summary[doc] = doc_metrics
        summary["per_document"] = per_doc_summary

    # Score distribution for threshold analysis
    if answerable_top1_scores or unanswerable_top1_scores:
        summary["score_distributions"] = {
            "answerable_top1": score_distribution_stats(answerable_top1_scores),
            "unanswerable_top1": score_distribution_stats(unanswerable_top1_scores),
        }

    full_output: Dict[str, Any] = {
        "summary": summary,
        "details": results_per_item,
    }

    if failures:
        full_output["failures"] = failures

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{timestamp_slug}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # Print summary table
    print("\n" + "=" * 72)
    print("             PAPERTRAIL RETRIEVAL EVALUATION REPORT")
    print("=" * 72)
    print(f" Dataset:           {dataset_path}")
    print(f" Format:            {'v2' if is_v2 else 'v1'}")
    print(f" Retriever:         {summary['retriever']}")
    if "model_name" in summary:
        print(f" Model:             {summary['model_name']}")
    if summary.get("retriever") == "hybrid":
        print(f" RRF Parameters:    k={summary.get('rrf_k')}, w_bm25={summary.get('weight_bm25')}, w_dense={summary.get('weight_dense')}")
    print(f" Total Queries:     {summary['total_queries']}")
    print(f"   Answerable:      {summary['answerable_count']}")
    print(f"   Unanswerable:    {summary['unanswerable_count']}")
    print(f" Results Saved To:  {out_file}")

    # Page-level metrics
    if "page_level" in summary:
        pl = summary["page_level"]
        print("-" * 72)
        print(" PAGE-LEVEL METRICS (answerable items only)")
        print("-" * 72)
        print(f"   Recall@1             | {pl['mean_recall@1'] * 100:.1f}%  ({pl['mean_recall@1']:.4f})")
        print(f"   Recall@3             | {pl['mean_recall@3'] * 100:.1f}%  ({pl['mean_recall@3']:.4f})")
        print(f"   Recall@5             | {pl['mean_recall@5'] * 100:.1f}%  ({pl['mean_recall@5']:.4f})")
        print(f"   MRR                  | {pl['mean_reciprocal_rank']:.4f}")

    # Passage-level metrics
    if "passage_level" in summary:
        psl = summary["passage_level"]
        print("-" * 72)
        print(" PASSAGE-LEVEL METRICS (answerable items with evidence_phrase)")
        print("-" * 72)
        print(f"   Recall@1             | {psl['mean_recall@1'] * 100:.1f}%  ({psl['mean_recall@1']:.4f})")
        print(f"   Recall@3             | {psl['mean_recall@3'] * 100:.1f}%  ({psl['mean_recall@3']:.4f})")
        print(f"   Recall@5             | {psl['mean_recall@5'] * 100:.1f}%  ({psl['mean_recall@5']:.4f})")
        print(f"   MRR                  | {psl['mean_reciprocal_rank']:.4f}")

    # Per-document breakdown
    if "per_document" in summary:
        print("-" * 72)
        print(" PER-DOCUMENT BREAKDOWN")
        print("-" * 72)
        for doc, doc_m in summary["per_document"].items():
            print(f"\n   [{doc}]")
            if "page_level" in doc_m:
                dpl = doc_m["page_level"]
                print(f"     Page R@1: {dpl['mean_recall@1']*100:.1f}%  MRR: {dpl['mean_reciprocal_rank']:.4f}  (n={dpl['query_count']})")
            if "passage_level" in doc_m:
                dpsl = doc_m["passage_level"]
                print(f"     Pass R@1: {dpsl['mean_recall@1']*100:.1f}%  MRR: {dpsl['mean_reciprocal_rank']:.4f}  (n={dpsl['query_count']})")

    # Score distributions
    if "score_distributions" in summary:
        sd = summary["score_distributions"]
        print("-" * 72)
        print(" TOP-1 BM25 SCORE DISTRIBUTIONS (for refusal threshold analysis)")
        print("-" * 72)
        ans = sd["answerable_top1"]
        unans = sd["unanswerable_top1"]
        print(f"   Answerable   (n={ans['count']:>3}):  min={ans['min']:.4f}  max={ans['max']:.4f}  mean={ans['mean']:.4f}  median={ans['median']:.4f}")
        print(f"   Unanswerable (n={unans['count']:>3}):  min={unans['min']:.4f}  max={unans['max']:.4f}  mean={unans['mean']:.4f}  median={unans['median']:.4f}")

    # Failure listing
    if failures:
        print("-" * 72)
        print(f" FAILURE LISTING ({len(failures)} misses)")
        print("-" * 72)
        for f_item in failures:
            miss_types = []
            if f_item.get("page_miss"):
                miss_types.append("page")
            if f_item.get("passage_miss"):
                miss_types.append("passage")
            miss_label = "+".join(miss_types) + " miss"
            print(f"\n   [{f_item['id']}] ({miss_label})")
            print(f"     Q: {f_item['question']}")
            if f_item.get("evidence_phrase"):
                print(f"     Evidence: \"{f_item['evidence_phrase']}\"")
            print(f"     Top-3 IDs: {f_item['top3_passage_ids']}")

    print("\n" + "=" * 72)

    return full_output


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

    parser = argparse.ArgumentParser(description="Run retrieval evaluation for PaperTrail.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "eval" / "dataset.json"),
        help="Path to the JSON evaluation dataset (v1 or v2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval" / "results"),
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=str(PROJECT_ROOT / "papers"),
        help="Directory containing PDF documents referenced in the dataset.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Retrieval engine to evaluate ('bm25', 'dense', or 'hybrid').",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_DENSE_MODEL,
        help="Embedding model identifier for dense/hybrid retrieval.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache embeddings on disk.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF smoothing constant k (default: 60).",
    )
    parser.add_argument(
        "--weight-bm25",
        type=float,
        default=1.0,
        help="Weight for BM25 ranking in hybrid retrieval.",
    )
    parser.add_argument(
        "--weight-dense",
        type=float,
        default=1.0,
        help="Weight for dense ranking in hybrid retrieval.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    evaluate_dataset(
        Path(args.dataset),
        Path(args.output_dir),
        papers_dir=Path(args.papers_dir),
        retriever_type=args.retriever,
        model_name=args.model_name,
        cache_dir=cache_dir,
        rrf_k=args.rrf_k,
        weight_bm25=args.weight_bm25,
        weight_dense=args.weight_dense,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
