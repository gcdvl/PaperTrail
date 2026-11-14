"""
scripts/inspect_items.py: Read-only inspection script for flagged benchmark items.

Loads specified items from eval/dataset_all.json and eval/results/answers_20261024_140625.json,
chunks each item's document using PaperTrail's chunker with evaluation settings, and outputs
raw texts and exact evidence-phrase match results to eval/inspect_items.txt.

Additionally inspects the 7 false-refusal items by running the project's BM25 retriever
(same chunker and settings as the eval run) and writes results to eval/inspect_refusals.txt.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import Chunk, chunk_document
from src.extractor import extract_text_from_pdf
from src.metrics import passage_contains_evidence
from src.retriever import retrieve, tokenize

# Original 9 flagged items → eval/inspect_items.txt (cited-ID based)
FLAGGED_ITEM_IDS = [
    "drs-01",
    "drs-03",
    "attn-07",
    "bert-08",
    "bert-03",
    "bert-14",
    "drs-13",
    "attn-u07",
    "bert-u03",
]

# 7 false-refusal items → eval/inspect_refusals.txt (BM25-based)
REFUSAL_ITEM_IDS = [
    "attn-14",
    "bert-02",
    "bert-11",
    "drs-05",
    "drs-09",
    "drs-10",
    "drs-11",
]


def resolve_pdf(doc_name: str, papers_dir: Path, project_root: Path) -> Path:
    """Resolves document name to a PDF path."""
    pdf_path = papers_dir / doc_name
    if pdf_path.exists():
        return pdf_path
    pdf_path = project_root / doc_name
    if pdf_path.exists():
        return pdf_path
    raise FileNotFoundError(f"PDF document '{doc_name}' not found.")


def _find_boundary_span(
    doc_chunks: List[Chunk],
    evidence_phrase: str,
) -> Optional[Tuple[str, str]]:
    """
    If no single chunk contains the evidence phrase, check pairs of adjacent
    chunks (in document order) to find a boundary span.

    Returns (chunk_id_a, chunk_id_b) or None.
    """
    for i in range(len(doc_chunks) - 1):
        combined = doc_chunks[i].text + " " + doc_chunks[i + 1].text
        if passage_contains_evidence(combined, evidence_phrase):
            return (doc_chunks[i].passage_id, doc_chunks[i + 1].passage_id)
    return None


def inspect_flagged_items(
    dataset_by_id: dict,
    results_by_id: dict,
    papers_dir: Path,
    doc_chunks_cache: Dict[str, List[Chunk]],
) -> None:
    """Writes eval/inspect_items.txt for the original 9 flagged items (cited-ID based)."""
    output_path = PROJECT_ROOT / "eval" / "inspect_items.txt"
    output_lines: List[str] = []

    for item_id in FLAGGED_ITEM_IDS:
        dataset_item = dataset_by_id.get(item_id, {})
        result_item = results_by_id.get(item_id, {})

        doc_name = dataset_item.get("document") or result_item.get("document", "")
        question = dataset_item.get("question") or result_item.get("question", "")
        model_answer = result_item.get("answer", "")
        evidence_phrase = dataset_item.get("evidence_phrase")
        cited_ids = result_item.get("citations", [])

        # Chunk document
        if doc_name not in doc_chunks_cache:
            pdf_path = resolve_pdf(doc_name, papers_dir, PROJECT_ROOT)
            doc_result = extract_text_from_pdf(pdf_path)
            doc_chunks_cache[doc_name] = chunk_document(doc_result)

        doc_chunks = doc_chunks_cache[doc_name]
        chunk_map = {c.passage_id: c for c in doc_chunks}

        output_lines.append("=" * 80)
        output_lines.append(f"ITEM ID: {item_id}")
        output_lines.append(f"QUESTION: {question}")
        output_lines.append(f"MODEL ANSWER: {model_answer}")
        ev_display = evidence_phrase if evidence_phrase else "none"
        output_lines.append(f"EVIDENCE PHRASE: {ev_display}")
        output_lines.append("")

        output_lines.append("CITED CHUNKS:")
        if not cited_ids:
            output_lines.append("None")
        else:
            for cid in cited_ids:
                output_lines.append(f"- Cited Chunk ID: {cid}")
                chunk_obj = chunk_map.get(cid)
                if chunk_obj:
                    has_ev = (
                        passage_contains_evidence(chunk_obj.text, evidence_phrase)
                        if evidence_phrase
                        else False
                    )
                    output_lines.append(f"  Contains evidence phrase: {has_ev}")
                    output_lines.append("  Full text:")
                    output_lines.append(chunk_obj.text)
                else:
                    output_lines.append("  Contains evidence phrase: False")
                    output_lines.append("  Full text: [Chunk ID not found in document]")
                output_lines.append("")

        # All chunks in document containing evidence phrase
        if evidence_phrase:
            matching_chunks = [
                c.passage_id for c in doc_chunks
                if passage_contains_evidence(c.text, evidence_phrase)
            ]
        else:
            matching_chunks = []

        output_lines.append("ALL CHUNKS IN DOCUMENT CONTAINING EVIDENCE PHRASE:")
        if matching_chunks:
            for mc in matching_chunks:
                output_lines.append(f"- {mc}")
        elif not evidence_phrase:
            output_lines.append("none")
        else:
            output_lines.append("none: phrase spans a chunk boundary")

        output_lines.append("")

    output_content = "\n".join(output_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"Flagged-items inspection written to {output_path}")


NO_OVERLAP_LABEL = "no query-token overlap (cannot be retrieved at any top_k)"

# Specific terms to show tokenizer behavior for, when a no-overlap chunk is found
TOKENIZER_PROBE_TERMS = ["SST-2", "110-layer", "CIFAR-10", "34-layer"]


def inspect_refusal_items(
    dataset_by_id: dict,
    papers_dir: Path,
    doc_chunks_cache: Dict[str, List[Chunk]],
) -> None:
    """
    Inspects the 7 false-refusal items using the project's BM25 retriever.

    For each item:
    - Computes the full BM25 ranking of all chunks in the document.
    - Reports the top 5 retrieved chunk IDs with full text.
    - Reports which chunk(s) contain the evidence phrase and their BM25 ranks.
    - For chunks with no query-token overlap, prints token-level diagnostics.
    - Writes detailed output to eval/inspect_refusals.txt.
    - Prints a compact console summary.
    """
    output_path = PROJECT_ROOT / "eval" / "inspect_refusals.txt"
    output_lines: List[str] = []
    # Track whether we've already printed the tokenizer probe (once per run)
    tokenizer_probe_printed = False

    for item_id in REFUSAL_ITEM_IDS:
        dataset_item = dataset_by_id.get(item_id, {})

        doc_name = dataset_item.get("document", "")
        question = dataset_item.get("question", "")
        evidence_phrase = dataset_item.get("evidence_phrase")
        ev_display = evidence_phrase if evidence_phrase else "none"

        # Chunk document (same settings as eval run: default min_chars=60, max_chars=1000)
        if doc_name not in doc_chunks_cache:
            pdf_path = resolve_pdf(doc_name, papers_dir, PROJECT_ROOT)
            doc_result = extract_text_from_pdf(pdf_path)
            doc_chunks_cache[doc_name] = chunk_document(doc_result)

        doc_chunks = doc_chunks_cache[doc_name]
        chunk_map = {c.passage_id: c for c in doc_chunks}

        # Run BM25 retriever with full ranking (top_k = total chunks)
        full_ranking = retrieve(doc_chunks, question, top_k=len(doc_chunks))
        # Build rank lookup: passage_id → 1-based rank
        rank_by_id = {r.passage_id: r.rank for r in full_ranking}

        # Compute query tokens (same tokenizer as the retriever)
        query_tokens = tokenize(question, remove_stopwords=True)
        query_token_set = set(query_tokens)

        top5 = full_ranking[:5]
        top5_ids = [r.passage_id for r in top5]

        # Find chunks containing the evidence phrase
        matching_chunk_ids: List[str] = []
        boundary_span: Optional[Tuple[str, str]] = None
        if evidence_phrase:
            matching_chunk_ids = [
                c.passage_id for c in doc_chunks
                if passage_contains_evidence(c.text, evidence_phrase)
            ]
            if not matching_chunk_ids:
                boundary_span = _find_boundary_span(doc_chunks, evidence_phrase)

        # Determine if any phrase-containing chunk is in top 5
        if matching_chunk_ids:
            any_in_top5 = any(cid in top5_ids for cid in matching_chunk_ids)
        elif boundary_span:
            any_in_top5 = boundary_span[0] in top5_ids or boundary_span[1] in top5_ids
        else:
            any_in_top5 = False

        # Helper to format rank for a chunk ID
        def _rank_label(cid: str) -> str:
            r = rank_by_id.get(cid, None)
            if r is not None:
                return str(r)
            return NO_OVERLAP_LABEL

        # ---- FILE OUTPUT ----
        output_lines.append("=" * 80)
        output_lines.append(f"ITEM ID: {item_id}")
        output_lines.append(f"QUESTION: {question}")
        output_lines.append(f"EVIDENCE PHRASE: {ev_display}")
        output_lines.append("")

        output_lines.append("TOP-5 RETRIEVED CHUNKS (rank order):")
        for r in top5:
            output_lines.append(f"  Rank {r.rank}: {r.passage_id} (score {r.score:.4f})")
            output_lines.append("  Full text:")
            output_lines.append(r.passage)
            output_lines.append("")

        output_lines.append("EVIDENCE PHRASE CHUNK LOCATIONS:")
        if matching_chunk_ids:
            for cid in matching_chunk_ids:
                output_lines.append(f"  {cid} — BM25 rank: {_rank_label(cid)}")
        elif boundary_span:
            cid_a, cid_b = boundary_span
            output_lines.append(
                f"  spans a chunk boundary: {cid_a} | {cid_b} — ranks: {_rank_label(cid_a)}, {_rank_label(cid_b)}"
            )
        else:
            output_lines.append("  not found in any chunk")

        output_lines.append("")

        # ---- CONSOLE OUTPUT ----
        print(f"Item: {item_id}")
        print(f"  Evidence Phrase: {ev_display}")
        print(f"  Top-5 IDs: {', '.join(top5_ids)}")
        if matching_chunk_ids:
            parts = []
            for cid in matching_chunk_ids:
                parts.append(f"{cid} (rank {_rank_label(cid)})")
            print(f"  Phrase in: {', '.join(parts)}")
        elif boundary_span:
            cid_a, cid_b = boundary_span
            print(f"  Phrase in: spans a chunk boundary: {cid_a} | {cid_b} (ranks {_rank_label(cid_a)}, {_rank_label(cid_b)})")
        else:
            print("  Phrase in: not found")
        in_top5_str = "yes" if any_in_top5 else "no"
        print(f"  IN TOP 5: {in_top5_str}")

        # ---- TOKEN DIAGNOSTICS for no-overlap chunks ----
        no_overlap_cids = []
        if matching_chunk_ids:
            no_overlap_cids = [cid for cid in matching_chunk_ids if cid not in rank_by_id]
        elif boundary_span:
            no_overlap_cids = [cid for cid in boundary_span if cid not in rank_by_id]

        if no_overlap_cids:
            print(f"  --- Token diagnostics (no query-token overlap) ---")
            print(f"  Question tokens: {query_tokens}")
            for cid in no_overlap_cids:
                chunk_obj = chunk_map[cid]
                chunk_tokens = tokenize(chunk_obj.text, remove_stopwords=True)
                chunk_token_set = set(chunk_tokens)
                overlap = sorted(query_token_set & chunk_token_set)
                print(f"  Chunk {cid} tokens: {chunk_tokens}")
                print(f"  Overlap: {overlap}")

            # Print tokenizer probe for specific terms (once per run)
            if not tokenizer_probe_printed:
                print(f"  --- Tokenizer probe ---")
                for term in TOKENIZER_PROBE_TERMS:
                    result = tokenize(term, remove_stopwords=True)
                    print(f"    tokenize({term!r}) -> {result}")
                tokenizer_probe_printed = True

        print()

    output_content = "\n".join(output_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"Refusal inspection written to {output_path}")


def main() -> None:
    dataset_path = PROJECT_ROOT / "eval" / "dataset_all.json"
    results_path = PROJECT_ROOT / "eval" / "results" / "answers_20261024_140625.json"
    papers_dir = PROJECT_ROOT / "papers"

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)
    dataset_by_id = {item["id"]: item for item in dataset_list}

    with open(results_path, "r", encoding="utf-8") as f:
        results_obj = json.load(f)
    results_by_id = {r["id"]: r for r in results_obj.get("results", [])}

    # Cache chunked documents to avoid redundant extraction
    doc_chunks_cache: Dict[str, List[Chunk]] = {}

    # 1. Original flagged-item inspection → eval/inspect_items.txt
    inspect_flagged_items(dataset_by_id, results_by_id, papers_dir, doc_chunks_cache)

    # 2. False-refusal inspection → eval/inspect_refusals.txt
    inspect_refusal_items(dataset_by_id, papers_dir, doc_chunks_cache)


if __name__ == "__main__":
    main()
