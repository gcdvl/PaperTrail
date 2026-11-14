"""
cli.py: Command-line entry point for extracting, searching, and retrieving text from PDF documents.

Usage:
    python -m src.cli path/to/document.pdf
    python -m src.cli path/to/document.pdf --full-text
    python -m src.cli path/to/document.pdf --search "keyword"
    python -m src.cli path/to/document.pdf --retrieve "query" --top-k 5
"""

import argparse
import sys
from pathlib import Path
from src.extractor import (
    extract_text_from_pdf,
    PDFExtractionError,
)
from src.search import search_document
from src.chunker import chunk_document
from src.retriever import retrieve


def build_parser() -> argparse.ArgumentParser:
    """Builds and configures the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="papertrail-extract",
        description="Extract, search, and rank 1-based page metadata from a PDF research paper.",
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the local PDF file to inspect and extract.",
    )
    parser.add_argument(
        "--retrieve",
        type=str,
        default=None,
        nargs="?",
        const="",
        metavar="QUERY",
        help="Perform BM25 ranked retrieval to return top matching passages.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of top ranked passages to return during retrieval (default: 5).",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        nargs="?",
        const="",
        metavar="QUERY",
        help="Search extracted PDF text for passages matching the given keyword or phrase.",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Display the entire extracted text for each page instead of a snippet.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="Number of characters to show in the page text snippet preview (default: 200).",
    )
    return parser


def main(args=None) -> int:
    """Main CLI execution flow."""
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

    parser = build_parser()
    parsed_args = parser.parse_args(args)

    pdf_path = parsed_args.pdf_path

    try:
        result = extract_text_from_pdf(pdf_path)
    except PDFExtractionError as err:
        print(f"\n[Error] PDF Extraction Failed: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\n[Unexpected Error] An unhandled error occurred: {err}", file=sys.stderr)
        return 1

    # Branch 1: Ranked BM25 Retrieval Mode
    if parsed_args.retrieve is not None:
        query = parsed_args.retrieve.strip()
        top_k = parsed_args.top_k
        print("=" * 60)
        print(f" Document:        {result.document_name}")
        print(f" File Path:       {result.file_path}")
        print(f" Retrieval Query: '{parsed_args.retrieve}'")
        print(f" Top-K Requested: {top_k}")
        print("=" * 60)

        if not query:
            print("\n[Retrieval] Query is empty. Please provide a non-empty query term.")
            return 0

        chunks = chunk_document(result)
        retrieval_results = retrieve(chunks, query, top_k=top_k)

        if not retrieval_results:
            print(f"\nNo relevant passages retrieved for query: '{parsed_args.retrieve}'")
            print("=" * 60)
            return 0

        print(f"\nRetrieved {len(retrieval_results)} ranked passage(s):\n")
        for res in retrieval_results:
            print(f"--- [Rank {res.rank} | Score: {res.score:.4f} | Page {res.page_number} of {result.total_pages}] ---")
            print(f"Passage ID: {res.passage_id}")
            print(f"Document:   {res.document_name}")
            print(f"Page:       {res.page_number}")
            print(f"Score:      {res.score:.4f}")
            print("Passage:")
            print(res.passage)
            print()

        print("=" * 60)
        print("Retrieval completed successfully.")
        print("=" * 60)
        return 0

    # If --search is specified, execute keyword search workflow
    if parsed_args.search is not None:
        query = parsed_args.search.strip()
        print("=" * 60)
        print(f" Document:     {result.document_name}")
        print(f" File Path:    {result.file_path}")
        print(f" Search Query: '{parsed_args.search}'")
        print("=" * 60)

        if not query:
            print("\n[Search] Query is empty. Please provide a non-empty search term.")
            return 0

        search_results = search_document(result, query)

        if not search_results:
            print(f"\nNo matching passages found for query: '{parsed_args.search}'")
            print("=" * 60)
            return 0

        print(f"\nFound {len(search_results)} matching passage(s):\n")
        for idx, res in enumerate(search_results, start=1):
            print(f"--- [Result {idx} | Page {res.page_number} of {result.total_pages}] ---")
            print(f"Document: {res.document_name}")
            print(f"Page:     {res.page_number}")
            print("Passage:")
            print(res.passage)
            print()

        print("=" * 60)
        print("Search completed successfully.")
        print("=" * 60)
        return 0

    # Print summary header
    print("=" * 60)
    print(f" Document:    {result.document_name}")
    print(f" File Path:   {result.file_path}")
    print(f" Total Pages: {result.total_pages}")
    print(f" Has Text:    {'Yes' if result.has_text else 'No (Warning: Document has no extractable text)'}")
    print("=" * 60)

    if result.total_pages == 0:
        print("\nNote: The document contains 0 pages.")
        return 0

    if not result.has_text:
        print("\n[Warning] No extractable text was found on any page.")
        print("          This PDF might contain scanned images without OCR, or blank pages.")

    # Print page-by-page breakdown
    for page in result.pages:
        print(f"\n--- [Page {page.page_number} of {result.total_pages}] ---")
        print(f"Characters extracted: {page.char_count}")

        if not page.has_text:
            print("Status: [EMPTY / NO EXTRACTABLE TEXT]")
            continue

        if parsed_args.full_text:
            print("Extracted Text:")
            print(page.text)
        else:
            preview = page.text[: parsed_args.preview_chars]
            if len(page.text) > parsed_args.preview_chars:
                preview += " ... [truncated]"
            print(f"Preview snippet:\n{preview}")

    print("\n" + "=" * 60)
    print("Extraction completed successfully.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
