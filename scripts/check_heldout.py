"""Check heldout tables against PDF extracted text.

Extracts the raw text of the stated page from the PDF using the project's
existing PDF extraction logic on that single page, checks if all must_contain
strings appear, and finds other pages where the main numeric value appears.
"""

import argparse
import json
from pathlib import Path
import re
import unicodedata
import pymupdf


def extract_single_page_text(pdf_path: Path, page_num_1based: int) -> str:
    """Extract raw text from a single page using the project's extractor logic."""
    doc = pymupdf.open(str(pdf_path))
    with doc:
        page_idx = page_num_1based - 1
        page = doc[page_idx]
        raw_text = page.get_text()
        raw_text = unicodedata.normalize("NFKC", raw_text)
        return raw_text.strip()


def extract_all_pages_text(pdf_path: Path) -> dict[int, str]:
    """Extract raw text for all pages in a PDF document."""
    doc = pymupdf.open(str(pdf_path))
    pages_text = {}
    with doc:
        for idx in range(len(doc)):
            page = doc[idx]
            raw_text = page.get_text()
            raw_text = unicodedata.normalize("NFKC", raw_text)
            pages_text[idx + 1] = raw_text.strip()
    return pages_text


def is_numeric_string(s: str) -> bool:
    """Check if string looks numeric (integer, decimal, or percentage number)."""
    cleaned = s.strip().rstrip("%")
    return bool(re.match(r"^\d+(\.\d+)?$", cleaned))


def get_snippet_window(text: str, target: str, window_size: int = 150) -> str:
    """Extract a window_size character window around the first occurrence of target."""
    idx = text.lower().find(target.lower())
    if idx == -1:
        return "<not found>"
    half = (window_size - len(target)) // 2
    start = max(0, idx - half)
    end = min(len(text), start + window_size)
    if end - start < window_size and start > 0:
        start = max(0, end - window_size)
    snippet = text[start:end]
    return snippet.replace("\r\n", " | ").replace("\n", " | ").replace("\r", " | ")


def main():
    parser = argparse.ArgumentParser(description="Check heldout tables against PDF extracted text.")
    parser.add_argument("--show", type=str, default=None, help="Question ID to show detailed text windows for.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    eval_file = root_dir / "eval" / "heldout_tables.json"
    papers_dir = root_dir / "papers"

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Cache full PDF page texts for searching other pages
    pdf_docs: dict[str, dict[int, str]] = {}
    for item in questions:
        doc_name = item["doc"]
        if doc_name not in pdf_docs:
            pdf_path = papers_dir / doc_name
            pdf_docs[doc_name] = extract_all_pages_text(pdf_path)

    all_found_count = 0

    for item in questions:
        q_id = item["id"]
        doc_name = item["doc"]
        page_num = item["page"]
        must_contain = item.get("must_contain", [])

        pdf_path = papers_dir / doc_name
        page_text = extract_single_page_text(pdf_path, page_num)
        page_text_lower = page_text.lower()

        if not must_contain:
            print(f"{q_id} | {doc_name} | page {page_num} | no must_contain")
            continue

        missing = [s for s in must_contain if s.lower() not in page_text_lower]
        all_found = len(missing) == 0
        if all_found:
            all_found_count += 1

        # First must_contain string that looks numeric
        numeric_val = next((s for s in must_contain if is_numeric_string(s)), None)

        other_pages = []
        if numeric_val:
            num_lower = numeric_val.lower()
            doc_pages = pdf_docs[doc_name]
            for p_num, p_text in doc_pages.items():
                if p_num != page_num and num_lower in p_text.lower():
                    other_pages.append(p_num)

        other_pages_str = str(other_pages) if numeric_val else "none"

        # Check for pages where all must_contain strings appear together (excluding stated page)
        doc_pages = pdf_docs[doc_name]
        all_strings_pages = []
        for p_num, p_text in doc_pages.items():
            if p_num != page_num:
                p_text_lower = p_text.lower()
                if all(s.lower() in p_text_lower for s in must_contain):
                    all_strings_pages.append(p_num)

        print(
            f"ID: {q_id} | Doc: {doc_name} | Page: {page_num} | "
            f"All Found: {all_found} | Missing: {missing} | "
            f"Numeric Value: {numeric_val} | Other Pages: {other_pages_str} | "
            f"All-strings pages: {all_strings_pages}"
        )

        if args.show and args.show == q_id:
            show_pages = [page_num] + [p for p in all_strings_pages if p != page_num]
            for p in show_pages:
                p_text = doc_pages[p]
                print(f"--- Page {p} ---")
                for s in must_contain:
                    window = get_snippet_window(p_text, s, window_size=150)
                    print(f"  [{s}]: {window}")

    print(f"Total questions where all strings were found: {all_found_count} / {len(questions)}")


if __name__ == "__main__":
    main()

