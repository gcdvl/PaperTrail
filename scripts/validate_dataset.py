"""
validate_dataset.py: Validates a PaperTrail v2 evaluation dataset.

Checks:
1. JSON structure: every item has required fields with correct types.
2. Consistency: answerable items must have evidence_phrase (non-null) and non-empty
   expected_page_numbers; unanswerable items must have null evidence_phrase and empty
   expected_page_numbers.
3. Evidence grounding: for every answerable item, verifies the evidence_phrase actually
   appears (whitespace-normalized, case-insensitive) on at least one of the expected
   pages in the referenced PDF.

Usage:
    python scripts/validate_dataset.py [--dataset eval/dataset_v2.json] [--papers-dir papers/]
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractor import extract_text_from_pdf


def _normalize_whitespace(text: str) -> str:
    """Collapses all whitespace sequences to single spaces and strips."""
    return re.sub(r"\s+", " ", text).strip()


def validate_schema(item: Dict[str, Any], idx: int) -> List[str]:
    """Validates the schema of a single dataset item. Returns a list of error strings."""
    errors: List[str] = []
    item_id = item.get("id", f"<index {idx}>")

    required_fields = {
        "id": str,
        "document": str,
        "question": str,
        "answerable": bool,
        "expected_page_numbers": list,
    }

    for field, expected_type in required_fields.items():
        if field not in item:
            errors.append(f"[{item_id}] Missing required field: '{field}'")
        elif not isinstance(item[field], expected_type):
            errors.append(
                f"[{item_id}] Field '{field}' has wrong type: "
                f"expected {expected_type.__name__}, got {type(item[field]).__name__}"
            )

    # evidence_phrase: must exist, must be str or None
    if "evidence_phrase" not in item:
        errors.append(f"[{item_id}] Missing required field: 'evidence_phrase'")
    else:
        ep = item["evidence_phrase"]
        if ep is not None and not isinstance(ep, str):
            errors.append(
                f"[{item_id}] Field 'evidence_phrase' must be a string or null, "
                f"got {type(ep).__name__}"
            )

    # Consistency checks
    if item.get("answerable") is True:
        if not item.get("expected_page_numbers"):
            errors.append(
                f"[{item_id}] Answerable item must have non-empty 'expected_page_numbers'"
            )
        if item.get("evidence_phrase") is None:
            errors.append(
                f"[{item_id}] Answerable item must have non-null 'evidence_phrase'"
            )
        elif isinstance(item.get("evidence_phrase"), str) and not item["evidence_phrase"].strip():
            errors.append(
                f"[{item_id}] Answerable item 'evidence_phrase' must not be empty"
            )
    elif item.get("answerable") is False:
        if item.get("expected_page_numbers"):
            errors.append(
                f"[{item_id}] Unanswerable item must have empty 'expected_page_numbers'"
            )
        if item.get("evidence_phrase") is not None:
            errors.append(
                f"[{item_id}] Unanswerable item must have null 'evidence_phrase'"
            )

    # Validate page numbers are positive ints
    for pn in item.get("expected_page_numbers", []):
        if not isinstance(pn, int) or pn < 1:
            errors.append(
                f"[{item_id}] Invalid page number {pn!r} in 'expected_page_numbers'"
            )

    return errors


def _tokenize_words(text: str) -> Set[str]:
    """Tokenizes text into lowercase word tokens."""
    return set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))


def _extract_candidates(text: str) -> List[str]:
    """Extracts non-empty candidate lines and sentences from page text."""
    if not text or not text.strip():
        return []
    candidates: List[str] = []
    # 1. Non-empty physical lines
    for line in text.splitlines():
        cleaned = _normalize_whitespace(line)
        if len(cleaned) > 2 and not re.match(r"^\d+[\.\)]?$", cleaned):
            candidates.append(cleaned)
    # 2. Sentences from paragraphs
    for para in re.split(r"\n\s*\n+", text.strip()):
        flat = _normalize_whitespace(para)
        for s in re.split(r"(?<=[a-zA-Z0-9][.!?])\s+", flat):
            s = s.strip()
            if len(s) > 2 and not re.match(r"^\d+[\.\)]?$", s):
                candidates.append(s)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def find_suggestions(
    item: Dict[str, Any],
    project_root: Path,
    papers_dir: Path,
    page_text_cache: Optional[Dict[Tuple[str, int], str]] = None,
) -> Dict[str, Any]:
    """
    Finds suggestions for an item whose evidence_phrase was not found on expected pages.

    Returns:
        Dict containing item_id, expected_pages, other_pages, and top_candidates.
    """
    if page_text_cache is None:
        page_text_cache = {}

    item_id = item.get("id", "?")
    doc_name = item.get("document", "")
    question = item.get("question", "")
    evidence = item.get("evidence_phrase", "")
    expected_pages = item.get("expected_page_numbers", [])

    pdf_path = papers_dir / doc_name
    if not pdf_path.exists():
        pdf_path = project_root / doc_name
    if not pdf_path.exists():
        return {
            "item_id": item_id,
            "expected_pages": expected_pages,
            "other_pages": [],
            "top_candidates": [],
        }

    # Ensure all pages of PDF are extracted and cached
    try:
        doc_result = extract_text_from_pdf(pdf_path)
        for page in doc_result.pages:
            k = (str(pdf_path), page.page_number)
            page_text_cache[k] = page.text
    except Exception:
        return {
            "item_id": item_id,
            "expected_pages": expected_pages,
            "other_pages": [],
            "top_candidates": [],
        }

    norm_evidence = _normalize_whitespace(unicodedata.normalize("NFKC", evidence)).lower() if evidence else ""

    # 1. Check all other pages where the phrase does exist
    other_pages: List[int] = []
    if norm_evidence:
        for page in doc_result.pages:
            if page.page_number not in expected_pages:
                page_text = page_text_cache.get((str(pdf_path), page.page_number), "")
                norm_page = _normalize_whitespace(unicodedata.normalize("NFKC", page_text)).lower()
                if norm_evidence in norm_page:
                    other_pages.append(page.page_number)

    # 2. Extract candidate lines/sentences from expected pages
    target_words = _tokenize_words(question) | _tokenize_words(evidence)
    phrase_words = _tokenize_words(evidence)

    all_candidates: List[str] = []
    for page_num in expected_pages:
        page_text = page_text_cache.get((str(pdf_path), page_num), "")
        all_candidates.extend(_extract_candidates(page_text))

    # Deduplicate candidates across pages
    seen_cands: Set[str] = set()
    unique_candidates: List[str] = []
    for c in all_candidates:
        if c not in seen_cands:
            seen_cands.add(c)
            unique_candidates.append(c)

    # Score and rank candidates by word overlap with question and phrase
    scored = []
    for idx, cand in enumerate(unique_candidates):
        c_words = _tokenize_words(cand)
        overlap = len(c_words & target_words)
        phrase_overlap = len(c_words & phrase_words)
        scored.append((overlap, phrase_overlap, -idx, cand))

    scored.sort(reverse=True)
    top_candidates = [(cand, overlap) for overlap, _, _, cand in scored[:3]]

    return {
        "item_id": item_id,
        "expected_pages": expected_pages,
        "other_pages": other_pages,
        "top_candidates": top_candidates,
    }


def format_suggestions(suggestions: Dict[str, Any]) -> str:
    """Formats suggestion dictionary into human-readable string lines."""
    lines: List[str] = []
    expected_pages = suggestions.get("expected_pages", [])
    pages_str = ", ".join(str(p) for p in expected_pages) if expected_pages else "None"
    top_candidates = suggestions.get("top_candidates", [])
    other_pages = suggestions.get("other_pages", [])

    lines.append(
        f"    --> Suggested line(s)/sentence(s) on expected page(s) [{pages_str}] by word overlap:"
    )
    if top_candidates:
        for idx, (cand_text, score) in enumerate(top_candidates, start=1):
            lines.append(f'        {idx}. [overlap={score}] "{cand_text}"')
    else:
        lines.append("        (No extractable text or candidate lines found)")

    if other_pages:
        other_str = ", ".join(str(p) for p in other_pages)
        lines.append(f"    --> Other page(s) where evidence phrase exists: [{other_str}]")
    else:
        lines.append("    --> Other page(s) where evidence phrase exists: None")

    return "\n".join(lines)


def validate_evidence_on_page(
    item: Dict[str, Any],
    project_root: Path,
    papers_dir: Path,
    page_text_cache: Dict[Tuple[str, int], str],
    suggest: bool = False,
) -> List[str]:
    """
    For answerable items, checks that the evidence_phrase appears (whitespace-normalized,
    case-insensitive) in at least one of the expected pages of the document.

    Returns a list of error strings.
    """
    errors: List[str] = []
    item_id = item.get("id", "?")

    if not item.get("answerable", False):
        return errors

    evidence = item.get("evidence_phrase")
    if not evidence or not isinstance(evidence, str):
        return errors  # Schema validation already catches this

    doc_name = item["document"]
    expected_pages = item.get("expected_page_numbers", [])

    # Resolve PDF path: try papers_dir first, then project root
    pdf_path = papers_dir / doc_name
    if not pdf_path.exists():
        pdf_path = project_root / doc_name
    if not pdf_path.exists():
        errors.append(
            f"[{item_id}] Document '{doc_name}' not found in "
            f"'{papers_dir}' or '{project_root}'"
        )
        return errors

    # Extract and cache page text
    for page_num in expected_pages:
        cache_key = (str(pdf_path), page_num)
        if cache_key not in page_text_cache:
            try:
                doc_result = extract_text_from_pdf(pdf_path)
                for page in doc_result.pages:
                    k = (str(pdf_path), page.page_number)
                    page_text_cache[k] = page.text
            except Exception as exc:
                errors.append(f"[{item_id}] Failed to extract text from '{doc_name}': {exc}")
                return errors

    # Check evidence phrase on expected pages
    norm_evidence = _normalize_whitespace(unicodedata.normalize("NFKC", evidence)).lower()
    found_on_page = False
    for page_num in expected_pages:
        cache_key = (str(pdf_path), page_num)
        page_text = page_text_cache.get(cache_key, "")
        norm_page = _normalize_whitespace(unicodedata.normalize("NFKC", page_text)).lower()
        if norm_evidence in norm_page:
            found_on_page = True
            break

    if not found_on_page:
        pages_str = ", ".join(str(p) for p in expected_pages)
        err_msg = (
            f"[{item_id}] Evidence phrase not found on expected page(s) [{pages_str}] "
            f"of '{doc_name}': \"{evidence}\""
        )
        if suggest:
            suggestions = find_suggestions(
                item,
                project_root,
                papers_dir,
                page_text_cache=page_text_cache,
            )
            formatted = format_suggestions(suggestions)
            if formatted:
                err_msg = f"{err_msg}\n{formatted}"
        errors.append(err_msg)

    return errors


def validate_dataset(
    dataset_path: Path,
    papers_dir: Path,
    project_root: Path,
    check_evidence: bool = True,
    suggest: bool = False,
) -> Tuple[List[str], int]:
    """
    Validates an entire dataset file.

    Args:
        dataset_path: Path to the JSON dataset file.
        papers_dir: Directory containing PDF documents.
        project_root: Project root for fallback PDF resolution.
        check_evidence: If True, verify evidence phrases in PDFs.
        suggest: If True, include line/sentence suggestions and other matching pages for missing evidence.

    Returns:
        Tuple of (list of error strings, total item count).
    """
    if not dataset_path.exists():
        return ([f"Dataset file not found: {dataset_path}"], 0)

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return ([f"Invalid JSON in {dataset_path}: {exc}"], 0)

    if not isinstance(data, list):
        return ([f"Dataset must be a JSON array, got {type(data).__name__}"], 0)

    all_errors: List[str] = []
    page_text_cache: Dict[Tuple[str, int], str] = {}

    # Check for duplicate IDs
    seen_ids: Dict[str, int] = {}
    for idx, item in enumerate(data):
        item_id = item.get("id", "")
        if item_id in seen_ids:
            all_errors.append(
                f"[{item_id}] Duplicate ID at indices {seen_ids[item_id]} and {idx}"
            )
        seen_ids[item_id] = idx

    for idx, item in enumerate(data):
        # Schema validation
        all_errors.extend(validate_schema(item, idx))

        # Evidence grounding validation
        if check_evidence:
            all_errors.extend(
                validate_evidence_on_page(
                    item,
                    project_root,
                    papers_dir,
                    page_text_cache,
                    suggest=suggest,
                )
            )

    return (all_errors, len(data))


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
        description="Validate a PaperTrail v2 evaluation dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "eval" / "dataset_v2.json"),
        help="Path to the JSON evaluation dataset.",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=str(PROJECT_ROOT / "papers"),
        help="Directory containing PDF documents referenced in the dataset.",
    )
    parser.add_argument(
        "--no-evidence-check",
        action="store_true",
        help="Skip evidence phrase verification against PDFs.",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="For items with missing evidence, suggest candidate lines/sentences on expected pages and report other pages where evidence exists.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    papers_dir = Path(args.papers_dir)

    print(f"\nValidating dataset: {dataset_path}")
    print(f"Papers directory:   {papers_dir}")
    print("-" * 65)

    errors, total = validate_dataset(
        dataset_path,
        papers_dir,
        PROJECT_ROOT,
        check_evidence=not args.no_evidence_check,
        suggest=args.suggest,
    )

    if errors:
        print(f"\n  VALIDATION FAILED -- {len(errors)} error(s) in {total} items:\n")
        for err in errors:
            print(f"  X {err}")
        print()
        return 1
    else:
        print(f"\n  OK: VALIDATION PASSED -- {total} items, 0 errors.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
