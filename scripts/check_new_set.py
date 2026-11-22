"""
check_new_set.py: Validate a new evaluation dataset against PDF documents.

Usage:
    python scripts/check_new_set.py --dataset PATH --papers-dir PATH
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import Chunk, chunk_document
from src.extractor import DocumentExtractionResult, extract_text_from_pdf
from src.metrics import passage_contains_evidence
from src.retriever import DEFAULT_STOPWORDS

REQUIRED_FIELDS = [
    "id",
    "document",
    "question",
    "answerable",
    "expected_page_numbers",
    "evidence_phrase",
]


def check_dataset(
    dataset_path: Path,
    papers_dir: Path,
    allowed_pages_path: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """
    Validates a dataset JSON against PDF documents in papers_dir.
    Optionally checks that expected_page_numbers fall within allowed page ranges.

    Returns:
        (errors, warnings) lists of string messages.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not dataset_path.exists():
        errors.append(f"Dataset file does not exist: {dataset_path}")
        return errors, warnings

    allowed_pages: Optional[Dict[str, List[int]]] = None
    if allowed_pages_path is not None:
        allowed_pages_path = Path(allowed_pages_path)
        if not allowed_pages_path.exists():
            errors.append(f"Allowed-pages file does not exist: {allowed_pages_path}")
            return errors, warnings
        try:
            with open(allowed_pages_path, "r", encoding="utf-8") as f:
                allowed_pages = json.load(f)
        except Exception as e:
            errors.append(f"Failed to load allowed-pages JSON: {e}")
            return errors, warnings

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            items: List[Dict[str, Any]] = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load JSON dataset: {e}")
        return errors, warnings

    if not isinstance(items, list):
        errors.append("Dataset root must be a JSON array of items.")
        return errors, warnings

    # Print dataset path and item count first
    print(f"Dataset: {dataset_path}")
    print(f"Item count: {len(items)}")
    # Print chunker function and settings reused from pipeline
    print(f"Chunker: src.chunker.chunk_document (min_chars=60, max_chars=1000)")

    # Cache extracted PDFs and chunks
    pdf_cache: Dict[str, DocumentExtractionResult] = {}
    chunk_cache: Dict[str, List[Chunk]] = {}

    def get_pdf(doc_name: str) -> Optional[DocumentExtractionResult]:
        if doc_name not in pdf_cache:
            pdf_path = papers_dir / doc_name
            if not pdf_path.exists():
                return None
            try:
                doc_result = extract_text_from_pdf(pdf_path)
                pdf_cache[doc_name] = doc_result
                chunk_cache[doc_name] = chunk_document(doc_result, min_chars=60, max_chars=1000)
            except Exception as exc:
                errors.append(f"Failed to extract PDF '{doc_name}': {exc}")
                return None
        return pdf_cache[doc_name]

    seen_ids: Set[str] = set()
    items_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for idx, item in enumerate(items, start=1):
        item_id = item.get("id", f"<item {idx}>")

        # 1. Missing required fields
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        if missing:
            errors.append(f"[{item_id}] Missing required field(s): {', '.join(missing)}")

        # 2. Duplicate id
        if "id" in item:
            raw_id = item["id"]
            if raw_id in seen_ids:
                errors.append(f"[{raw_id}] Duplicate id")
            seen_ids.add(raw_id)

        doc_name = item.get("document")
        if not doc_name:
            continue

        items_by_doc[doc_name].append(item)
        doc_result = get_pdf(doc_name)
        if doc_result is None:
            errors.append(f"[{item_id}] Document not found in papers dir: {doc_name}")
            continue

        chunks = chunk_cache.get(doc_name, [])
        answerable = item.get("answerable")

        # --- Answerable checks ---
        if answerable is True:
            phrase = item.get("evidence_phrase")
            if not phrase or not isinstance(phrase, str):
                errors.append(f"[{item_id}] evidence_phrase must be a non-empty string for answerable items")
            else:
                words = phrase.strip().split()
                word_count = len(words)
                if word_count < 4 or word_count > 25:
                    errors.append(
                        f"[{item_id}] evidence_phrase has {word_count} words (must be 4 to 25 words): '{phrase}'"
                    )

                # ERROR if no single chunk contains it
                has_chunk = any(passage_contains_evidence(c.text, phrase) for c in chunks)
                if not has_chunk:
                    errors.append(f"[{item_id}] No single chunk contains evidence_phrase: '{phrase}'")

                # Pages containing the phrase
                pages_with_phrase = [
                    p.page_number for p in doc_result.pages
                    if passage_contains_evidence(p.text, phrase)
                ]

                expected_pages = item.get("expected_page_numbers")
                if not isinstance(expected_pages, list):
                    errors.append(f"[{item_id}] expected_page_numbers must be a list of 1-based page numbers")
                    expected_pages = []

                # Allowed-pages range check
                if allowed_pages is not None:
                    if doc_name not in allowed_pages:
                        errors.append(f"[{item_id}] Document '{doc_name}' is missing from allowed-pages mapping")
                    else:
                        page_range = allowed_pages[doc_name]
                        first_page, last_page = page_range[0], page_range[1]
                        for exp_page in expected_pages:
                            if not (first_page <= exp_page <= last_page):
                                errors.append(
                                    f"[{item_id}] expected_page_number {exp_page} is outside allowed range "
                                    f"[{first_page}, {last_page}] for '{doc_name}'"
                                )

                # Missing from a listed page
                for exp_page in expected_pages:
                    if exp_page not in pages_with_phrase:
                        errors.append(f"[{item_id}] evidence_phrase is missing from listed page {exp_page}")

                # Appears on a page NOT listed
                for actual_page in pages_with_phrase:
                    if actual_page not in expected_pages:
                        errors.append(
                            f"[{item_id}] evidence_phrase appears on unlisted page {actual_page} (listed: {expected_pages})"
                        )

                # WARN if more than 50% of phrase non-stopwords appear in question
                question = item.get("question", "")
                phrase_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", phrase.lower())
                non_stopwords = [w for w in phrase_tokens if w not in DEFAULT_STOPWORDS]
                if non_stopwords:
                    question_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower()))
                    overlap = [w for w in non_stopwords if w in question_tokens]
                    overlap_ratio = len(overlap) / len(non_stopwords)
                    if overlap_ratio > 0.5:
                        warnings.append(
                            f"[{item_id}] Over 50% of non-stopword words ({len(overlap)}/{len(non_stopwords)} = "
                            f"{overlap_ratio:.1%}) in evidence_phrase appear in question"
                        )

        # --- Unanswerable checks ---
        elif answerable is False:
            search_terms = item.get("search_terms")
            if search_terms is None or not isinstance(search_terms, list) or len(search_terms) < 3:
                errors.append(f"[{item_id}] search_terms is missing or has fewer than 3 entries")
            else:
                # Print each term with the pages where it appears
                for term in search_terms:
                    term_pages = [
                        p.page_number for p in doc_result.pages
                        if passage_contains_evidence(p.text, term)
                    ]
                    pages_repr = term_pages if term_pages else "none"
                    print(f"  [{item_id}] term '{term}': pages {pages_repr}")

                # ERROR if all terms appear together in one chunk
                all_in_chunk = any(
                    all(passage_contains_evidence(c.text, term) for term in search_terms)
                    for c in chunks
                )
                if all_in_chunk:
                    errors.append(f"[{item_id}] All search terms appear together in one chunk: review by hand")

            subtype = item.get("subtype")
            if subtype is not None and subtype not in ("a", "b", "c"):
                errors.append(f"[{item_id}] Invalid subtype '{subtype}'; must be one of 'a', 'b', 'c'")

    # --- Check table items limit per paper (at most 20% of answerable items) ---
    for doc_name, doc_items in items_by_doc.items():
        ans_items = [it for it in doc_items if it.get("answerable") is True]
        if ans_items:
            table_items = [
                it for it in ans_items
                if it.get("table") is True or it.get("qtype") == "table"
            ]
            ratio = len(table_items) / len(ans_items)
            if ratio > 0.20:
                errors.append(
                    f"[{doc_name}] Table items exceed 20% of answerable items: "
                    f"{len(table_items)}/{len(ans_items)} ({ratio:.1%})"
                )

    # --- Print counts per document ---
    print("\n" + "=" * 60)
    print("COUNTS PER DOCUMENT")
    print("=" * 60)
    for doc_name, doc_items in sorted(items_by_doc.items()):
        ans_count = sum(1 for it in doc_items if it.get("answerable") is True)
        unans_count = sum(1 for it in doc_items if it.get("answerable") is False)
        table_count = sum(1 for it in doc_items if it.get("table") is True or it.get("qtype") == "table")

        print(f"\nDocument: {doc_name}")
        print(f"  Answerable:   {ans_count}")
        print(f"  Unanswerable: {unans_count}")
        print(f"  Total:        {len(doc_items)}")
        print(f"  Table:        {table_count}")

        # per qtype
        qtypes = Counter(it.get("qtype", "<none>") for it in doc_items)
        print("  By qtype:")
        for qt, cnt in sorted(qtypes.items()):
            print(f"    {qt}: {cnt}")

        # per subtype
        unans_items = [it for it in doc_items if it.get("answerable") is False]
        if unans_items:
            subtypes = Counter(it.get("subtype", "<none>") for it in unans_items)
            print("  By subtype (unanswerable):")
            for st, cnt in sorted(subtypes.items()):
                print(f"    {st}: {cnt}")

    return errors, warnings


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Validate a new evaluation dataset.")
    parser.add_argument("--dataset", required=True, type=str, help="Path to evaluation dataset JSON")
    parser.add_argument("--papers-dir", required=True, type=str, help="Path to directory containing PDFs")
    parser.add_argument(
        "--allowed-pages",
        required=False,
        type=str,
        default=None,
        help="Path to JSON mapping filename -> [first_page, last_page] (1-based)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    papers_dir = Path(args.papers_dir)
    allowed_pages_path = Path(args.allowed_pages) if args.allowed_pages else None

    errors, warnings = check_dataset(dataset_path, papers_dir, allowed_pages_path=allowed_pages_path)

    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {len(errors)} error(s), {len(warnings)} warning(s)")
    print("=" * 60)

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  WARN: {w}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
