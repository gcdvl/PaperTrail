"""
search.py: Core keyword-search module for PaperTrail.

This module provides simple, case-insensitive keyword search across extracted
PDF documents and pages. It preserves 1-based page metadata and document names
so downstream citation and question-answering components can trace every match
directly to its source evidence.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Union

from src.extractor import (
    DocumentExtractionResult,
    PageContent,
    extract_text_from_pdf,
)


@dataclass(frozen=True)
class SearchResult:
    """
    Represents a matching passage found within a document page.

    Attributes:
        document_name: The filename of the source document (e.g., 'sample_paper.pdf').
        page_number: The 1-based physical page number where the match was found.
        passage: The text of the matching passage or paragraph.
        query: The search term that produced this match.
    """
    document_name: str
    page_number: int
    passage: str
    query: str = ""


def split_into_passages(text: str) -> List[str]:
    """
    Splits page text into candidate passages (paragraphs or major sections).

    In PDF extraction, paragraphs and distinct sections are typically separated
    by blank lines (two or more newlines), or numbered section headers (e.g.
    '1. Introduction', '3.2 Attention', 'Abstract', 'References'). Lines within
    a paragraph are wrapped with single newlines.

    Args:
        text: The extracted text of a page.

    Returns:
        List of non-empty passage strings.
    """
    if not text or not text.strip():
        return []

    # Normalize carriage returns
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split by two or more newlines, or section header boundaries
    section_pattern = r"(?:\n\s*\n+|\n(?=[0-9]+(?:\.[0-9]+)*\.?\s+[A-Z]|Abstract\b|References\b))"
    raw_passages = re.split(section_pattern, normalized)
    passages = [p.strip() for p in raw_passages if p.strip()]
    return passages if passages else [normalized]


def search_pages(
    pages: List[PageContent],
    query: str,
) -> List[SearchResult]:
    """
    Searches a collection of PageContent objects for passages matching a query.

    Performs case-insensitive keyword matching. Words in multi-word queries are
    matched flexibly across whitespace and line breaks.

    Args:
        pages: List of PageContent objects (typically from DocumentExtractionResult.pages).
        query: The search term or phrase.

    Returns:
        List of SearchResult objects preserving 1-based page numbers and document names.
        Returns an empty list if query is empty or no matches are found.
    """
    if not query or not query.strip():
        return []

    query_clean = query.strip()
    words = query_clean.split()
    if not words:
        return []

    # Build regex pattern matching words separated by any whitespace (including newlines)
    # Using re.escape on each word ensures special characters in queries don't break regex
    pattern_str = r"\s+".join(re.escape(w) for w in words)
    pattern = re.compile(pattern_str, re.IGNORECASE)

    results: List[SearchResult] = []

    for page in pages:
        # Skip blank pages or pages with no extractable text
        if not page.has_text or not page.text.strip():
            continue

        passages = split_into_passages(page.text)
        for passage in passages:
            if pattern.search(passage):
                results.append(
                    SearchResult(
                        document_name=page.document_name,
                        page_number=page.page_number,
                        passage=passage,
                        query=query_clean,
                    )
                )

    return results


def search_document(
    document: DocumentExtractionResult,
    query: str,
) -> List[SearchResult]:
    """
    Searches an extracted document for passages matching a query.

    Args:
        document: DocumentExtractionResult from extract_text_from_pdf.
        query: The search term or phrase.

    Returns:
        List of SearchResult objects.
    """
    return search_pages(document.pages, query)


def search_pdf(
    file_path: Union[str, Path],
    query: str,
) -> List[SearchResult]:
    """
    Convenience function that extracts text from a PDF file and searches it.

    Args:
        file_path: Local path to the PDF document.
        query: The search term or phrase.

    Returns:
        List of SearchResult objects.
    """
    document = extract_text_from_pdf(file_path)
    return search_document(document, query)
