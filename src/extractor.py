"""
extractor.py: Core module for extracting text and metadata from PDF files.

This module provides data models and functions to load local PDF documents,
iterate through each page, and extract text while preserving 1-based page numbers
and document names for reliable downstream citation and evidence grounding.
"""

from dataclasses import dataclass
from pathlib import Path
import unicodedata
from typing import List, Union
import pymupdf


class PDFExtractionError(Exception):
    """Base exception for all PDF extraction errors."""
    pass


class PDFFileNotFoundError(PDFExtractionError):
    """Raised when the specified PDF file path does not exist."""
    pass


class InvalidPDFError(PDFExtractionError):
    """Raised when the file is not a valid or readable PDF document."""
    pass


@dataclass(frozen=True)
class PageContent:
    """
    Represents the extracted text and metadata for a single PDF page.

    Attributes:
        document_name: The filename of the source document (e.g., 'attention.pdf').
        page_number: The 1-based physical page number (matching what a reader sees).
        text: The cleaned, extracted plain text of the page.
        char_count: Total number of characters in the extracted text.
        has_text: True if the page contains extractable text, False if blank/scanned image.
    """
    document_name: str
    page_number: int
    text: str
    char_count: int
    has_text: bool


@dataclass(frozen=True)
class DocumentExtractionResult:
    """
    Represents the extraction results for an entire PDF document.

    Attributes:
        document_name: The filename of the document.
        file_path: The absolute path to the PDF file.
        total_pages: Total number of pages in the PDF.
        pages: List of PageContent objects, one for each page in order.
        has_text: True if at least one page contained extractable text.
    """
    document_name: str
    file_path: str
    total_pages: int
    pages: List[PageContent]
    has_text: bool


def extract_text_from_pdf(file_path: Union[str, Path]) -> DocumentExtractionResult:
    """
    Extracts text page-by-page from a local PDF file, preserving page numbers.

    Design rationale:
    - 1-based page indexing: PyMuPDF internally uses 0-based indexing (0, 1, 2...),
      but academic citations and PDF reader interfaces use 1-based page numbers.
      We convert each page index to (page_idx + 1) to ensure citations point to
      the exact page a human reader expects.
    - Graceful textless page handling: Scanned papers or image-only pages return empty
      strings. Rather than crashing, we record has_text=False so the caller can warn
      the user or trigger OCR in later milestones if needed.

    Args:
        file_path: Local path to the PDF file (string or pathlib.Path).

    Returns:
        DocumentExtractionResult containing metadata and list of PageContent objects.

    Raises:
        PDFFileNotFoundError: If the file path does not exist or is a directory.
        InvalidPDFError: If the file is not a PDF or PyMuPDF fails to parse it.
    """
    path = Path(file_path).resolve()

    # 1. Validate that the path exists and is a file
    if not path.exists():
        raise PDFFileNotFoundError(f"File not found at path: '{path}'")
    if not path.is_file():
        raise PDFFileNotFoundError(f"Expected a file, but path is a directory: '{path}'")

    # 2. Validate file extension
    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError(
            f"Invalid file extension '{path.suffix}'. Only .pdf files are supported."
        )

    # 3. Attempt to open and parse the PDF document with PyMuPDF
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise InvalidPDFError(
            f"Failed to open '{path.name}' as a PDF document. File may be corrupted or malformed. "
            f"Details: {exc}"
        ) from exc

    pages: List[PageContent] = []

    # 4. Extract text page-by-page while preserving 1-based page numbers
    try:
        with doc:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                raw_text = page.get_text()
                raw_text = unicodedata.normalize("NFKC", raw_text)
                cleaned_text = raw_text.strip()
                has_text = len(cleaned_text) > 0

                page_content = PageContent(
                    document_name=path.name,
                    page_number=page_idx + 1,  # Preserve 1-based page numbering
                    text=cleaned_text,
                    char_count=len(cleaned_text),
                    has_text=has_text,
                )
                pages.append(page_content)
    except Exception as exc:
        raise InvalidPDFError(
            f"Error occurred while reading pages from '{path.name}': {exc}"
        ) from exc

    doc_has_text = any(p.has_text for p in pages)

    return DocumentExtractionResult(
        document_name=path.name,
        file_path=str(path),
        total_pages=len(pages),
        pages=pages,
        has_text=doc_has_text,
    )
