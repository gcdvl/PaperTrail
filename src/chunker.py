"""
chunker.py: Chunking engine for PDF documents with stable passage IDs and page metadata.

Splits extracted pages into retrieval passages, merges very short passages, and splits
very long ones while strictly preserving 1-based page numbers and never crossing page boundaries.
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
from src.search import split_into_passages


@dataclass(frozen=True)
class Chunk:
    """
    Represents an independently retrievable text passage with stable metadata.

    Attributes:
        passage_id: Deterministic identifier formatted as '<doc>::p<page>::c<index>'.
        document_name: Filename of the source document.
        page_number: 1-based physical page number.
        text: Plain text content of the chunk.
        char_count: Number of characters in the chunk.
    """
    passage_id: str
    document_name: str
    page_number: int
    text: str
    char_count: int


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """
    Splits text exceeding max_chars into smaller chunks.

    Prefers splitting along sentence boundaries; falls back to word boundaries
    if sentences exceed max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current_chunk = ""

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If a single sentence is longer than max_chars, split by words
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            words = sentence.split()
            sub_chunk = ""
            for word in words:
                if len(sub_chunk) + len(word) + 1 > max_chars:
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                    sub_chunk = word
                else:
                    sub_chunk = f"{sub_chunk} {word}" if sub_chunk else word
            if sub_chunk:
                chunks.append(sub_chunk.strip())
            continue

        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text[:max_chars]]


def _merge_and_split_passages(
    raw_passages: List[str],
    min_chars: int,
    max_chars: int,
) -> List[str]:
    """
    Merges short passages and splits long passages within a single page.
    """
    if not raw_passages:
        return []

    # Step 1: Split any passage exceeding max_chars
    normalized: List[str] = []
    for passage in raw_passages:
        if len(passage) > max_chars:
            normalized.extend(_split_long_text(passage, max_chars))
        else:
            normalized.append(passage)

    if not normalized:
        return []

    # Step 2: Merge short passages with neighbors
    merged: List[str] = []
    buffer = ""

    for passage in normalized:
        if not buffer:
            buffer = passage
        else:
            # Check if previous buffer was too short
            if len(buffer) < min_chars and (len(buffer) + len(passage) + 2) <= max_chars:
                buffer = f"{buffer}\n\n{passage}"
            else:
                merged.append(buffer)
                buffer = passage

    if buffer:
        # If the trailing buffer is short and there is a preceding chunk that can absorb it
        if len(buffer) < min_chars and merged and (len(merged[-1]) + len(buffer) + 2) <= max_chars:
            merged[-1] = f"{merged[-1]}\n\n{buffer}"
        else:
            merged.append(buffer)

    return merged


def chunk_page(
    page: PageContent,
    min_chars: int = 60,
    max_chars: int = 1000,
) -> List[Chunk]:
    """
    Splits a single PageContent into a list of Chunk objects.

    Guarantees:
    - Never crosses page boundaries.
    - Stable 1-based passage_id: '<doc>::p<page>::c<index>'.
    - Merges passages shorter than min_chars when possible without exceeding max_chars.
    - Splits passages longer than max_chars.

    Args:
        page: PageContent instance to chunk.
        min_chars: Minimum target character count for merging short passages.
        max_chars: Maximum allowable character count before splitting.

    Returns:
        List of Chunk objects for this page.
    """
    if not page.has_text or not page.text.strip():
        return []

    raw_passages = split_into_passages(page.text)
    processed_texts = _merge_and_split_passages(raw_passages, min_chars, max_chars)

    chunks: List[Chunk] = []
    for idx, text in enumerate(processed_texts, start=1):
        chunk_id = f"{page.document_name}::p{page.page_number}::c{idx}"
        chunks.append(
            Chunk(
                passage_id=chunk_id,
                document_name=page.document_name,
                page_number=page.page_number,
                text=text,
                char_count=len(text),
            )
        )

    return chunks


def chunk_pages(
    pages: List[PageContent],
    min_chars: int = 60,
    max_chars: int = 1000,
) -> List[Chunk]:
    """
    Chunks a collection of PageContent objects across a document.

    Processes each page independently to ensure page boundaries are never crossed.

    Args:
        pages: List of PageContent instances.
        min_chars: Minimum target length for merging short passages.
        max_chars: Maximum allowable length before splitting.

    Returns:
        List of Chunk objects across all pages in document order.
    """
    all_chunks: List[Chunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page, min_chars=min_chars, max_chars=max_chars))
    return all_chunks


def chunk_document(
    document: DocumentExtractionResult,
    min_chars: int = 60,
    max_chars: int = 1000,
) -> List[Chunk]:
    """
    Convenience function to chunk an entire DocumentExtractionResult.

    Args:
        document: DocumentExtractionResult from extractor.py.
        min_chars: Minimum character length for merging.
        max_chars: Maximum character length for splitting.

    Returns:
        List of Chunk objects for the document.
    """
    return chunk_pages(document.pages, min_chars=min_chars, max_chars=max_chars)


def chunk_pdf(
    file_path: Union[str, Path],
    min_chars: int = 60,
    max_chars: int = 1000,
) -> List[Chunk]:
    """
    Convenience function that extracts text from a PDF file and chunks it.

    Args:
        file_path: Local path to the PDF document.
        min_chars: Minimum character length for merging.
        max_chars: Maximum character length for splitting.

    Returns:
        List of Chunk objects.
    """
    document = extract_text_from_pdf(file_path)
    return chunk_document(document, min_chars=min_chars, max_chars=max_chars)
