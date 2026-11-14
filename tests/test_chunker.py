"""
test_chunker.py: Unit tests for passage chunking and stable identifier generation.

Verifies:
1. Stable, deterministic passage_ids formatted as '<doc>::p<page>::c<index>'.
2. Strict isolation per page (chunks never cross page boundaries).
3. Merging short passages on the same page.
4. Splitting long passages exceeding max_chars.
5. Safe handling of blank/textless pages.
6. Integration with DocumentExtractionResult.
"""

from pathlib import Path
import pytest

from src.chunker import (
    Chunk,
    chunk_page,
    chunk_pages,
    chunk_document,
    chunk_pdf,
)
from src.extractor import PageContent, extract_text_from_pdf


class TestChunker:
    """Test suite for chunker algorithms, invariants, and data models."""

    def test_chunk_ids_stable_and_deterministic(self, sample_three_page_pdf: Path):
        """Passage IDs follow '<doc>::p<page>::c<index>' format and are deterministic across calls."""
        doc = extract_text_from_pdf(sample_three_page_pdf)
        chunks_run_1 = chunk_document(doc)
        chunks_run_2 = chunk_document(doc)

        assert len(chunks_run_1) > 0
        assert len(chunks_run_1) == len(chunks_run_2)

        for c1, c2 in zip(chunks_run_1, chunks_run_2):
            assert c1.passage_id == c2.passage_id
            assert f"{c1.document_name}::p{c1.page_number}::c" in c1.passage_id
            assert isinstance(c1, Chunk)
            assert c1.char_count == len(c1.text)

    def test_no_page_crossing(self, sample_three_page_pdf: Path):
        """
        Critical invariant: Chunks must NEVER cross page boundaries.
        Each chunk must originate purely from a single physical page.
        """
        doc = extract_text_from_pdf(sample_three_page_pdf)
        chunks = chunk_document(doc)

        for chunk in chunks:
            # Match the chunk back to its source page in doc
            source_page = doc.pages[chunk.page_number - 1]
            assert chunk.page_number == source_page.page_number
            assert chunk.document_name == source_page.document_name
            # Chunk text must be contained within the source page text
            for line in chunk.text.split("\n\n"):
                clean_line = line.strip()
                if clean_line:
                    assert clean_line in source_page.text

    def test_merge_short_passages_within_same_page(self):
        """Short passages below min_chars on the same page are merged together."""
        page = PageContent(
            document_name="test.pdf",
            page_number=1,
            text="Heading\n\nShort intro sentence.\n\nAnother short sentence here.",
            char_count=60,
            has_text=True,
        )
        chunks = chunk_page(page, min_chars=50, max_chars=500)

        # Because raw passages are very short (< 50 chars), they should merge into one chunk
        assert len(chunks) == 1
        assert "Heading" in chunks[0].text
        assert "Short intro" in chunks[0].text
        assert "Another short" in chunks[0].text
        assert chunks[0].passage_id == "test.pdf::p1::c1"

    def test_split_long_passages(self):
        """A passage longer than max_chars is split into smaller chunks without exceeding max_chars."""
        sentence = "This is a long sentence meant to test splitting. "
        long_text = sentence * 20  # ~980 characters
        page = PageContent(
            document_name="long_doc.pdf",
            page_number=2,
            text=long_text,
            char_count=len(long_text),
            has_text=True,
        )

        chunks = chunk_page(page, min_chars=50, max_chars=300)

        assert len(chunks) > 1
        for c in chunks:
            assert c.char_count <= 300
            assert c.page_number == 2
            assert c.document_name == "long_doc.pdf"

        # IDs should be sequential on page 2: c1, c2, c3...
        expected_ids = [f"long_doc.pdf::p2::c{i}" for i in range(1, len(chunks) + 1)]
        assert [c.passage_id for c in chunks] == expected_ids

    def test_blank_page_yields_no_chunks(self, mixed_blank_and_text_pdf: Path):
        """Blank pages yield zero chunks and do not shift subsequent page numbering."""
        doc = extract_text_from_pdf(mixed_blank_and_text_pdf)
        chunks = chunk_document(doc)

        page_numbers_in_chunks = {c.page_number for c in chunks}
        # Page 2 is blank in mixed_blank_and_text_pdf
        assert 1 in page_numbers_in_chunks
        assert 2 not in page_numbers_in_chunks
        assert 3 in page_numbers_in_chunks

        # Chunk on page 3 still has page_number == 3
        page_3_chunks = [c for c in chunks if c.page_number == 3]
        assert len(page_3_chunks) > 0
        assert page_3_chunks[0].passage_id == "mixed_document.pdf::p3::c1"

    def test_chunk_pdf_convenience_function(self, sample_three_page_pdf: Path):
        """chunk_pdf() reads the file from disk and chunks end-to-end."""
        chunks = chunk_pdf(sample_three_page_pdf)
        assert len(chunks) > 0
        assert chunks[0].document_name == "deep_residual_learning.pdf"
