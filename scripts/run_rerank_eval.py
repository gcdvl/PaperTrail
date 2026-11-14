"""
run_rerank_eval.py: Cross-encoder reranking evaluation on frozen and held-out sets.

Compares BM25-only baseline against BM25+reranker on passage recall, reports
gains/losses, timing, and a pre-registered adoption rule.

No Gemini, no dense or hybrid retriever, no network calls (model loads from cache).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_document
from src.extractor import extract_text_from_pdf
from src.metrics import passage_contains_evidence
from src.reranker import CrossEncoderReranker, DEFAULT_POOL, DEFAULT_RERANK_MODEL
from src.retriever import RetrievalResult, retrieve

# ---------------------------------------------------------------------------
# Fixed settings (no CLI options)
# ---------------------------------------------------------------------------
MODEL = DEFAULT_RERANK_MODEL
POOL = DEFAULT_POOL      # BM25 top-20 fed into reranker
TOP_K = 5                # final top-k evaluated


# ---------------------------------------------------------------------------
# Held-out matching rule (from run_heldout_eval.py)
# ---------------------------------------------------------------------------

def is_hit(chunk_text: str, must_contain: List[str]) -> bool:
    """Check if a chunk contains all must_contain strings (case-insensitive)."""
    if not must_contain:
        return True
    text_lower = chunk_text.lower()
    return all(s.lower() in text_lower for s in must_contain)


def find_first_hit_rank(results, must_contain: List[str], max_k: int = 5) -> Optional[int]:
    """Return the 1-based rank of the first hit within max_k, or None."""
    for i, result in enumerate(results[:max_k]):
        if is_hit(result.passage, must_contain):
            return i + 1
    return None


def count_hits_at_k(results, must_contain: List[str], k: int) -> bool:
    """True if any chunk in the top k is a hit."""
    for result in results[:k]:
        if is_hit(result.passage, must_contain):
            return True
    return False


# ---------------------------------------------------------------------------
# Evidence matching (from candidate_recall.py / run_eval.py)
# ---------------------------------------------------------------------------

def best_evidence_rank(results, evidence_phrase: str, max_k: int) -> Optional[int]:
    """Return 1-based rank of first passage containing evidence within max_k, or None."""
    for r in results[:max_k]:
        if passage_contains_evidence(r.passage, evidence_phrase):
            return r.rank
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_pdf(doc_name: str) -> Path:
    papers_dir = PROJECT_ROOT / "papers"
    pdf_path = papers_dir / doc_name
    if pdf_path.exists():
        return pdf_path
    pdf_path = PROJECT_ROOT / doc_name
    if pdf_path.exists():
        return pdf_path
    raise FileNotFoundError(f"PDF '{doc_name}' not found")


def get_chunks(doc_name: str, cache: dict) -> list:
    if doc_name not in cache:
        pdf_path = resolve_pdf(doc_name)
        doc_result = extract_text_from_pdf(pdf_path)
        cache[doc_name] = chunk_document(doc_result)
    return cache[doc_name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Rerank evaluation")
    parser.add_argument("--out-dir", type=str,
                        default=str(PROJECT_ROOT / "eval" / "results"),
                        help="Output directory for results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model and warm up
    # ------------------------------------------------------------------
    print("Loading cross-encoder model...", flush=True)
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker(model_name=MODEL)
    dummy = RetrievalResult("warmup::p1::c1", "dummy.pdf", 1, "warmup passage text", 0.0, 1)
    reranker.rerank("warmup query", [dummy], top_k=1)
    model_load_s = time.perf_counter() - t0
    print(f"Model loaded and warmed up in {model_load_s:.2f}s\n", flush=True)

    doc_chunk_cache: dict = {}
    rerank_times_ms: List[float] = []

    # ------------------------------------------------------------------
    # FROZEN SET (eval/dataset_all.json, answerable only, n=45)
    # ------------------------------------------------------------------
    dataset_path = PROJECT_ROOT / "eval" / "dataset_all.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)
    answerable = [it for it in all_items if it.get("answerable")]
    n_frozen = len(answerable)

    # Per-item JSON records
    per_item_json: List[dict] = []

    # Per-item tuple: (id, paper, bm25_rank_in_20, rerank_rank_in_20)
    frozen_item_results: List[Tuple[str, str, Optional[int], Optional[int]]] = []
    per_paper_results: Dict[str, List[Tuple[Optional[int], Optional[int]]]] = defaultdict(list)

    for it in answerable:
        doc_name = it["document"]
        question = it["question"]
        evidence = it["evidence_phrase"]
        item_id = it["id"]
        paper = Path(doc_name).stem

        chunks = get_chunks(doc_name, doc_chunk_cache)

        # BM25 top-20 (pool)
        bm25_pool = retrieve(chunks, question, top_k=POOL)

        # Best rank of evidence inside BM25 top-20 (None if not in top 20)
        bm25_best_rank = best_evidence_rank(bm25_pool, evidence, max_k=POOL)

        # CrossEncoder rerank of the top-20 pool
        t_start = time.perf_counter()
        reranked_20 = reranker.rerank(question, bm25_pool, top_k=POOL)
        t_end = time.perf_counter()
        rerank_times_ms.append((t_end - t_start) * 1000)

        # Best rank of evidence inside reranked 20 (None if absent)
        rerank_best_rank = best_evidence_rank(reranked_20, evidence, max_k=POOL)

        frozen_item_results.append((item_id, paper, bm25_best_rank, rerank_best_rank))
        per_paper_results[paper].append((bm25_best_rank, rerank_best_rank))

        per_item_json.append({
            "id": item_id,
            "paper": paper,
            "question": question,
            "evidence_phrase": evidence,
            "bm25_best_rank_in_20": bm25_best_rank,
            "rerank_best_rank_in_20": rerank_best_rank,
            "baseline_top5": [
                {"passage_id": r.passage_id, "score": round(r.score, 4), "rank": r.rank}
                for r in bm25_pool[:TOP_K]
            ],
            "reranked_top5": [
                {"passage_id": r.passage_id, "score": round(r.score, 4), "rank": r.rank}
                for r in reranked_20[:TOP_K]
            ],
            "bm25_pool_20": [
                {"passage_id": r.passage_id, "score": round(r.score, 4), "rank": r.rank}
                for r in bm25_pool
            ],
            "reranked_pool_20": [
                {"passage_id": r.passage_id, "score": round(r.score, 4), "rank": r.rank}
                for r in reranked_20
            ],
        })

    # Passage recall counts: evidence rank <= k
    def count_recall(ranks: List[Optional[int]], k: int) -> int:
        return sum(1 for r in ranks if r is not None and r <= k)

    bm25_ranks = [r[2] for r in frozen_item_results]
    rerank_ranks = [r[3] for r in frozen_item_results]

    bl_r1 = count_recall(bm25_ranks, 1)
    bl_r3 = count_recall(bm25_ranks, 3)
    bl_r5 = count_recall(bm25_ranks, 5)

    rr_r1 = count_recall(rerank_ranks, 1)
    rr_r3 = count_recall(rerank_ranks, 3)
    rr_r5 = count_recall(rerank_ranks, 5)

    # Items changed at R@5
    gained = []  # baseline miss (rank > 5 or None), reranked hit (rank <= 5)
    lost = []    # baseline hit (rank <= 5), reranked miss (rank > 5 or None)

    for item_id, paper, bl_rank, rr_rank in frozen_item_results:
        bl_hit = bl_rank is not None and bl_rank <= TOP_K
        rr_hit = rr_rank is not None and rr_rank <= TOP_K
        if not bl_hit and rr_hit:
            gained.append((item_id, paper, bl_rank, rr_rank))
        elif bl_hit and not rr_hit:
            lost.append((item_id, paper, bl_rank, rr_rank))

    net = len(gained) - len(lost)

    # ------------------------------------------------------------------
    # HELD-OUT SET (eval/heldout_tables.json, 15 items)
    # ------------------------------------------------------------------
    heldout_path = PROJECT_ROOT / "eval" / "heldout_tables.json"
    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout_items = json.load(f)
    n_heldout = len(heldout_items)

    heldout_per_item: List[dict] = []
    ho_bl_hits = {1: 0, 3: 0, 5: 0}
    ho_rr_hits = {1: 0, 3: 0, 5: 0}

    for q in heldout_items:
        item_id = q["id"]
        query = q["question"]
        must_contain = q.get("must_contain", [])
        doc_name = q["doc"]

        chunks = get_chunks(doc_name, doc_chunk_cache)

        bm25_pool = retrieve(chunks, query, top_k=POOL)
        for k in [1, 3, 5]:
            if count_hits_at_k(bm25_pool, must_contain, k):
                ho_bl_hits[k] += 1

        t_start = time.perf_counter()
        reranked_20 = reranker.rerank(query, bm25_pool, top_k=POOL)
        t_end = time.perf_counter()
        rerank_times_ms.append((t_end - t_start) * 1000)

        for k in [1, 3, 5]:
            if count_hits_at_k(reranked_20, must_contain, k):
                ho_rr_hits[k] += 1

        bl_first_rank = find_first_hit_rank(bm25_pool, must_contain, max_k=5)
        rr_first_rank = find_first_hit_rank(reranked_20, must_contain, max_k=5)

        heldout_per_item.append({
            "id": item_id,
            "bm25_rank": bl_first_rank,
            "rerank_rank": rr_first_rank,
        })

    # ------------------------------------------------------------------
    # Timing (over all 60 queries)
    # ------------------------------------------------------------------
    mean_ms = sum(rerank_times_ms) / len(rerank_times_ms) if rerank_times_ms else 0.0
    max_ms = max(rerank_times_ms) if rerank_times_ms else 0.0

    # ------------------------------------------------------------------
    # Per-paper breakdown and net R@5
    # ------------------------------------------------------------------
    per_paper_net: Dict[str, int] = {}
    for paper, pairs in per_paper_results.items():
        bl_cnt = sum(1 for bl, _ in pairs if bl is not None and bl <= TOP_K)
        rr_cnt = sum(1 for _, rr in pairs if rr is not None and rr <= TOP_K)
        per_paper_net[paper] = rr_cnt - bl_cnt

    # ------------------------------------------------------------------
    # Build text report
    # ------------------------------------------------------------------
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("       RERANK EVALUATION  (BM25 + cross-encoder)")
    lines.append("=" * 72)
    lines.append(f"  Model:           {MODEL}")
    lines.append(f"  Pool (BM25):     {POOL}")
    lines.append(f"  Top-k (final):   {TOP_K}")
    lines.append(f"  Model load:      {model_load_s:.2f}s (including warm-up)")
    lines.append(f"  Rerank timing:   mean={mean_ms:.1f} ms/query, max={max_ms:.1f} ms/query (n={len(rerank_times_ms)})")
    lines.append("")

    # --- Frozen set ---
    lines.append("=" * 72)
    lines.append(f"  FROZEN SET  (eval/dataset_all.json, answerable, n={n_frozen})")
    lines.append("=" * 72)
    lines.append("")
    lines.append("-" * 72)
    lines.append("  OVERALL PASSAGE RECALL")
    lines.append("-" * 72)
    lines.append(f"  {'':10s} {'Baseline':>12s} {'Reranked':>12s}")
    for k in [1, 3, 5]:
        b = count_recall(bm25_ranks, k)
        r = count_recall(rerank_ranks, k)
        lines.append(f"    R@{k}     {b:>2}/{n_frozen} ({b/n_frozen*100:5.1f}%)  "
                     f"{r:>2}/{n_frozen} ({r/n_frozen*100:5.1f}%)")

    # Per-paper breakdown
    lines.append("")
    lines.append("-" * 72)
    lines.append("  PER-PAPER BREAKDOWN")
    lines.append("-" * 72)
    for paper in sorted(per_paper_results.keys()):
        pairs = per_paper_results[paper]
        pn = len(pairs)
        lines.append(f"  [{paper}]  (n={pn})")
        lines.append(f"    {'':8s} {'Baseline':>12s} {'Reranked':>12s}")
        for k in [1, 3, 5]:
            b = sum(1 for bl, _ in pairs if bl is not None and bl <= k)
            r = sum(1 for _, rr in pairs if rr is not None and rr <= k)
            lines.append(f"      R@{k}   {b:>2}/{pn} ({b/pn*100:5.1f}%)  "
                         f"{r:>2}/{pn} ({r/pn*100:5.1f}%)")

    # Changes at R@5
    lines.append("")
    lines.append("-" * 72)
    lines.append("  ITEMS CHANGED AT R@5")
    lines.append("-" * 72)
    if gained:
        lines.append("  GAINED (baseline miss -> reranked hit):")
        for iid, p, bl, rr in gained:
            bl_str = str(bl) if bl is not None else "not in pool"
            rr_str = str(rr) if rr is not None else "not in pool"
            lines.append(f"    {iid:<12s} {p:<14s} BM25 top-20 rank: {bl_str:>11s} -> reranked rank: {rr_str}")
    else:
        lines.append("  GAINED: (none)")
    if lost:
        lines.append("  LOST (baseline hit -> reranked miss):")
        for iid, p, bl, rr in lost:
            bl_str = str(bl) if bl is not None else "not in pool"
            rr_str = str(rr) if rr is not None else "not in pool"
            lines.append(f"    {iid:<12s} {p:<14s} BM25 top-20 rank: {bl_str:>11s} -> reranked rank: {rr_str}")
    else:
        lines.append("  LOST: (none)")
    lines.append(f"  Counts: gained={len(gained)}, lost={len(lost)}, net={net:+d}")

    # Self-checks 1 and 2
    lines.append("")
    lines.append("-" * 72)
    lines.append("  SELF-CHECKS (FROZEN SET)")
    lines.append("-" * 72)
    sc1 = "PASS" if (bl_r1, bl_r3, bl_r5) == (20, 32, 33) else "FAIL"
    lines.append(f"  SELF-CHECK 1: baseline R@1={bl_r1}, R@3={bl_r3}, R@5={bl_r5} "
                 f"(expected 20/32/33) => {sc1}")

    sc2_expected = rr_r5 - bl_r5
    sc2 = "PASS" if net == sc2_expected else "FAIL"
    lines.append(f"  SELF-CHECK 2: net={net:+d}, reranked R@5 - baseline R@5 = "
                 f"{rr_r5} - {bl_r5} = {sc2_expected:+d} => {sc2}")

    # --- Held-out set ---
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  HELD-OUT SET  (eval/heldout_tables.json, n={n_heldout})")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {'':10s} {'Baseline':>12s} {'Reranked':>12s}")
    for k in [1, 3, 5]:
        b = ho_bl_hits[k]
        r = ho_rr_hits[k]
        lines.append(f"    R@{k}     {b:>2}/{n_heldout} ({b/n_heldout*100:5.1f}%)  "
                     f"{r:>2}/{n_heldout} ({r/n_heldout*100:5.1f}%)")

    lines.append("")
    lines.append(f"  Per-question hit ranks:")
    lines.append(f"    {'ID':<20s} {'BM25':>6s} {'Reranked':>8s}")
    lines.append(f"    {'-'*20:s} {'-'*6:s} {'-'*8:s}")
    for h in heldout_per_item:
        bl_s = str(h["bm25_rank"]) if h["bm25_rank"] is not None else "miss"
        rr_s = str(h["rerank_rank"]) if h["rerank_rank"] is not None else "miss"
        lines.append(f"    {h['id']:<20s} {bl_s:>6s} {rr_s:>8s}")

    sc3 = "PASS" if ho_bl_hits[5] == 8 else "FAIL"
    lines.append("")
    lines.append(f"  SELF-CHECK 3: held-out baseline R@5 = {ho_bl_hits[5]}/15 (expected 8) => {sc3}")

    # --- Adoption rule ---
    lines.append("")
    lines.append("=" * 72)
    lines.append("  ADOPTION RULE")
    lines.append("=" * 72)

    rule_a = net >= 3
    rule_b = rr_r1 >= bl_r1
    rule_c = all(v >= -1 for v in per_paper_net.values())
    rule_d = ho_rr_hits[5] >= 8

    lines.append(f"  (a) net R@5 gain >= 3:        net={net:+d}  => {'PASS' if rule_a else 'FAIL'}")
    lines.append(f"  (b) reranked R@1 >= baseline:  {rr_r1} >= {bl_r1}  => {'PASS' if rule_b else 'FAIL'}")
    per_paper_str = ", ".join(f"{p}: {v:+d}" for p, v in sorted(per_paper_net.items()))
    lines.append(f"  (c) no paper net R@5 < -1:    {per_paper_str}  => {'PASS' if rule_c else 'FAIL'}")
    lines.append(f"  (d) held-out reranked R@5 >= 8: {ho_rr_hits[5]}/15  => {'PASS' if rule_d else 'FAIL'}")

    adopt = rule_a and rule_b and rule_c and rule_d
    lines.append("")
    lines.append(f"  ADOPT: {'YES' if adopt else 'NO'}")

    lines.append("")
    lines.append("=" * 72)

    output = "\n".join(lines) + "\n"

    # Print to console
    print(output, end="")

    # Save text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = out_dir / f"rerank_eval_{timestamp}.txt"
    txt_path.write_text(output, encoding="utf-8")
    print(f"Saved text:  {txt_path}")

    # Save JSON
    json_path = out_dir / f"rerank_eval_{timestamp}.json"
    json_data = {
        "model": MODEL,
        "pool": POOL,
        "top_k": TOP_K,
        "model_load_s": round(model_load_s, 2),
        "rerank_mean_ms": round(mean_ms, 1),
        "rerank_max_ms": round(max_ms, 1),
        "frozen_set": {
            "n": n_frozen,
            "baseline": {"R@1": bl_r1, "R@3": bl_r3, "R@5": bl_r5},
            "reranked": {"R@1": rr_r1, "R@3": rr_r3, "R@5": rr_r5},
            "gained": [{"id": iid, "paper": p, "bm25_rank": bl, "rerank_rank": rr} for iid, p, bl, rr in gained],
            "lost": [{"id": iid, "paper": p, "bm25_rank": bl, "rerank_rank": rr} for iid, p, bl, rr in lost],
            "net": net,
            "per_item": per_item_json,
        },
        "heldout_set": {
            "n": n_heldout,
            "baseline": ho_bl_hits,
            "reranked": ho_rr_hits,
            "per_item": heldout_per_item,
        },
        "adoption": {
            "rule_a": rule_a,
            "rule_b": rule_b,
            "rule_c": rule_c,
            "rule_d": rule_d,
            "adopt": adopt,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON:  {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
