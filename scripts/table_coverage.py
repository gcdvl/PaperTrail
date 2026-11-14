"""
table_coverage.py: Run page.find_tables() with default and text strategies
on specified pages and print a summary block per page.
"""

from pathlib import Path
import pymupdf

PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers"

PAGES = [
    ("attention.pdf", 6),
    ("attention.pdf", 8),
    ("attention.pdf", 9),
    ("attention.pdf", 10),
    ("drs.pdf", 5),
    ("drs.pdf", 6),
    ("drs.pdf", 7),
    ("drs.pdf", 8),
    ("drs.pdf", 11),
    ("drs.pdf", 12),
    ("bert.pdf", 6),
    ("bert.pdf", 7),
    ("bert.pdf", 8),
    ("bert.pdf", 9),
    ("bert.pdf", 16),
]


def render_table_preview(table, max_chars: int = 150) -> str:
    """Render a table's rows as 'cell | cell | cell', joined with ' // ', truncated."""
    rows = table.extract()
    parts = []
    total = 0
    for row in rows:
        cells = [str(c).replace("\n", " ") if c is not None else "" for c in row]
        row_str = " | ".join(cells)
        if total + len(row_str) > max_chars:
            row_str = row_str[: max_chars - total]
            parts.append(row_str)
            break
        parts.append(row_str)
        total += len(row_str) + 4  # 4 for ' // '
    return " // ".join(parts)[:max_chars]


def main():
    open_docs: dict[str, pymupdf.Document] = {}
    try:
        for pdf_name, page_num in PAGES:
            if pdf_name not in open_docs:
                open_docs[pdf_name] = pymupdf.open(str(PAPERS_DIR / pdf_name))
            doc = open_docs[pdf_name]
            page = doc[page_num - 1]  # 0-based index

            # Run both strategies
            tf_default = page.find_tables()
            tf_text = page.find_tables(strategy="text")

            n_default = len(tf_default.tables)
            n_text = len(tf_text.tables)

            print(f"{pdf_name} p{page_num}: default={n_default}, text={n_text}")

            # Preview first table from default strategy if any
            if n_default > 0:
                preview = render_table_preview(tf_default.tables[0])
                print(f"  default[0]: {preview}")

            # Preview first table from text strategy if any
            if n_text > 0:
                preview = render_table_preview(tf_text.tables[0])
                print(f"  text[0]:    {preview}")

    finally:
        for doc in open_docs.values():
            doc.close()


if __name__ == "__main__":
    main()

