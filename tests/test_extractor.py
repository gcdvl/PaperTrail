"""
test_extractor.py: Comprehensive unit and integration tests for PDF text extraction.

Tests verify:
1. Normal multi-page text extraction.
2. Strict 1-based page numbering preservation.
3. Accurate handling of documents with mixed blank and non-blank pages.
4. Graceful handling of textless/blank PDFs.
5. Robust error handling for missing files, directories, non-PDF extensions, and corrupted files.
6. CLI entry point functionality and exit codes.
"""

import io
from pathlib import Path
import sys
import pytest
from src.extractor import (
    extract_text_from_pdf,
    PDFFileNotFoundError,
    InvalidPDFError,
)
from src.cli import main


class TestPDFExtraction:
    """Tests for core extraction function and data structures."""

    def test_normal_extraction_basic_metadata(self, sample_three_page_pdf: Path):
        """Verifies total page count, document name, and file path are captured correctly."""
        result = extract_text_from_pdf(sample_three_page_pdf)

        assert result.document_name == "deep_residual_learning.pdf"
        assert result.file_path == str(sample_three_page_pdf.resolve())
        assert result.total_pages == 3
        assert result.has_text is True
        assert len(result.pages) == 3

    def test_one_based_page_numbering_preserved(self, sample_three_page_pdf: Path):
        """
        Critical test: Verifies that page numbers are strictly 1-based (1, 2, 3...)
        and match the exact physical page reader view for reliable citation grounding.
        """
        result = extract_text_from_pdf(sample_three_page_pdf)

        page_numbers = [page.page_number for page in result.pages]
        assert page_numbers == [1, 2, 3], "Page numbers must be 1-based and sequential"
        assert 0 not in page_numbers, "Page numbers must not use 0-based indexing"

    def test_page_content_accuracy(self, sample_three_page_pdf: Path):
        """Verifies that the extracted text matches each page's actual content."""
        result = extract_text_from_pdf(sample_three_page_pdf)

        p1, p2, p3 = result.pages

        assert "Abstract: Deeper neural networks" in p1.text
        assert "1. Introduction" in p1.text
        assert p1.has_text is True
        assert p1.char_count > 0

        assert "2. Deep Residual Learning" in p2.text
        assert "F(x) := H(x) - x" in p2.text
        assert p2.has_text is True

        assert "3. Experiments on ImageNet" in p3.text
        assert "34-layer ResNet" in p3.text
        assert p3.has_text is True

    def test_mixed_blank_and_text_pages_preserves_numbering(self, mixed_blank_and_text_pdf: Path):
        """
        Verifies that an intermediate blank page does not distort or shift
        subsequent page numbers.
        """
        result = extract_text_from_pdf(mixed_blank_and_text_pdf)

        assert result.total_pages == 3
        assert [p.page_number for p in result.pages] == [1, 2, 3]

        # Page 1 has text
        assert result.pages[0].has_text is True
        assert "Content on page one" in result.pages[0].text

        # Page 2 is blank but still present as Page 2
        assert result.pages[1].has_text is False
        assert result.pages[1].text == ""
        assert result.pages[1].char_count == 0

        # Page 3 has text and is labeled Page 3
        assert result.pages[2].has_text is True
        assert "Content on page three" in result.pages[2].text

    def test_blank_pdf_handled_gracefully(self, blank_pdf: Path):
        """
        Verifies that a completely blank/scanned PDF does not throw an unhandled exception,
        returns has_text=False, and records empty pages properly.
        """
        result = extract_text_from_pdf(blank_pdf)

        assert result.total_pages == 1
        assert result.has_text is False
        assert len(result.pages) == 1
        assert result.pages[0].page_number == 1
        assert result.pages[0].has_text is False
        assert result.pages[0].text == ""
        assert result.pages[0].char_count == 0

    def test_extraction_normalizes_ligatures(self, tmp_path: Path):
        """Verifies that PDF ligatures (U+FB01 'ﬁ', U+FB02 'ﬂ') are normalized to 'fi' and 'fl'."""
        import pymupdf
        pdf_path = tmp_path / "ligatures.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(
            pymupdf.Rect(50, 50, 500, 500),
            "<p>System con\ufb01guration and optical \ufb02ow.</p>",
        )
        doc.save(str(pdf_path))
        doc.close()

        result = extract_text_from_pdf(pdf_path)
        assert result.total_pages == 1
        assert "configuration" in result.pages[0].text
        assert "flow" in result.pages[0].text
        assert "\ufb01" not in result.pages[0].text
        assert "\ufb02" not in result.pages[0].text


class TestPDFExtractionErrors:
    """Tests for edge cases and graceful error handling."""

    def test_file_not_found(self, tmp_path: Path):
        """Non-existent path raises PDFFileNotFoundError with clear message."""
        missing_path = tmp_path / "does_not_exist.pdf"
        with pytest.raises(PDFFileNotFoundError) as exc_info:
            extract_text_from_pdf(missing_path)

        assert "File not found" in str(exc_info.value)

    def test_directory_path_raises_error(self, tmp_path: Path):
        """Passing a directory instead of a file raises PDFFileNotFoundError."""
        sub_dir = tmp_path / "some_folder"
        sub_dir.mkdir()
        with pytest.raises(PDFFileNotFoundError) as exc_info:
            extract_text_from_pdf(sub_dir)

        assert "Expected a file, but path is a directory" in str(exc_info.value)

    def test_non_pdf_file_raises_invalid_pdf_error(self, non_pdf_file: Path):
        """Non-.pdf extension raises InvalidPDFError."""
        with pytest.raises(InvalidPDFError) as exc_info:
            extract_text_from_pdf(non_pdf_file)

        assert "Invalid file extension" in str(exc_info.value)
        assert ".txt" in str(exc_info.value)

    def test_corrupted_pdf_raises_invalid_pdf_error(self, corrupted_pdf: Path):
        """Corrupted or invalid PDF content raises InvalidPDFError."""
        with pytest.raises(InvalidPDFError) as exc_info:
            extract_text_from_pdf(corrupted_pdf)

        assert "Failed to open" in str(exc_info.value) or "corrupted" in str(exc_info.value)


class TestCommandLineInterface:
    """Tests for cli.py entry point execution and output."""

    def test_cli_success_output(self, sample_three_page_pdf: Path, capsys):
        """CLI runs successfully on a valid PDF and outputs page numbers and snippets."""
        exit_code = main([str(sample_three_page_pdf)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "deep_residual_learning.pdf" in captured.out
        assert "Total Pages: 3" in captured.out
        assert "--- [Page 1 of 3] ---" in captured.out
        assert "--- [Page 2 of 3] ---" in captured.out
        assert "--- [Page 3 of 3] ---" in captured.out
        assert "Extraction completed successfully." in captured.out

    def test_cli_full_text_flag(self, sample_three_page_pdf: Path, capsys):
        """CLI with --full-text displays complete page content."""
        exit_code = main([str(sample_three_page_pdf), "--full-text"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Extracted Text:" in captured.out
        assert "Formally, denoting the desired underlying mapping as H(x)" in captured.out

    def test_cli_blank_pdf_warning(self, blank_pdf: Path, capsys):
        """CLI outputs an explicit warning when a PDF has no extractable text."""
        exit_code = main([str(blank_pdf)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No extractable text was found on any page" in captured.out
        assert "Status: [EMPTY / NO EXTRACTABLE TEXT]" in captured.out

    def test_cli_missing_file_returns_error(self, tmp_path: Path, capsys):
        """CLI prints error to stderr and returns exit code 1 for missing files."""
        missing_file = tmp_path / "ghost.pdf"
        exit_code = main([str(missing_file)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[Error] PDF Extraction Failed" in captured.err

    def test_cli_non_pdf_returns_error(self, non_pdf_file: Path, capsys):
        """CLI prints error to stderr and returns exit code 1 for non-PDF files."""
        exit_code = main([str(non_pdf_file)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid file extension" in captured.err

    def test_cli_prints_unicode_page_without_raising(self, unicode_characters_pdf: Path, monkeypatch):
        """CLI prints a page containing \u2217 and \u03b1 through the CLI path without raising."""
        buffer = io.BytesIO()
        cp1252_stream = io.TextIOWrapper(buffer, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", cp1252_stream)

        exit_code = main([str(unicode_characters_pdf), "--full-text"])

        assert exit_code == 0
        cp1252_stream.flush()
        output = buffer.getvalue().decode("utf-8")
        assert "\u2217" in output
        assert "\u03b1" in output
        assert "Extracted Text:" in output
        assert "Extraction completed successfully." in output

