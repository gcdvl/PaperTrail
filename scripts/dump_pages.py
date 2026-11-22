"""
dump_pages.py: Dump extracted text from a PDF file page by page.

Usage:
    python scripts/dump_pages.py --pdf PATH --out PATH
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractor import extract_text_from_pdf


def dump_pages(pdf_path: Path, out_path: Path) -> int:
    """
    Extracts text from a PDF and writes each page to out_path with a header.
    Returns the total number of pages.
    """
    doc_result = extract_text_from_pdf(pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for page in doc_result.pages:
            f.write(f"=== PAGE {page.page_number} ===\n")
            if page.text:
                f.write(page.text)
                f.write("\n")

    return doc_result.total_pages


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Dump extracted text from a PDF page by page.")
    parser.add_argument("--pdf", required=True, type=str, help="Path to input PDF file")
    parser.add_argument("--out", required=True, type=str, help="Path to output text file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)

    total_pages = dump_pages(pdf_path, out_path)
    print(f"Page count: {total_pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
