"""
test_search.py: Comprehensive unit and integration tests for keyword search.

Tests verify:
1. Matching terms and passage extraction accuracy.
2. Case-insensitive search across varied casings (lowercase, uppercase, mixed case).
3. Queries matching across multiple distinct pages with preserved page numbers.
4. Clean handling of queries with no matches.
5. Handling of empty and whitespace-only queries.
6. Preservation of 1-based page numbers even with intermediate blank pages.
7. Flexible multi-word phrase matching across line wraps.
8. Safe execution on blank/textless documents.
9. Convenience function `search_pdf` integration.
10. CLI search functionality, formatted output, and exit codes.
"""

from pathlib import Path
import pytest

from src.extractor import extract_text_from_pdf
from src.search import (
    SearchResult,
    split_into_passages,
    search_pages,
    search_document,
    search_pdf,
)
from src.cli import main


class TestSplitIntoPassages:
    """Tests for splitting page text into clean candidate passages."""

    def test_split_double_newline(self):
        text = "Paragraph One.\n\nParagraph Two is here.\n\nParagraph Three."
        passages = split_into_passages(text)
        assert len(passages) == 3
        assert passages[0] == "Paragraph One."
        assert passages[1] == "Paragraph Two is here."
        assert passages[2] == "Paragraph Three."

    def test_split_single_paragraph_with_linebreaks(self):
        text = "This is line one of a paragraph.\nThis is line two of the same paragraph."
        passages = split_into_passages(text)
        assert len(passages) == 1
        assert "line one" in passages[0]
        assert "line two" in passages[0]

    def test_split_empty_and_whitespace(self):
        assert split_into_passages("") == []
        assert split_into_passages("   \n\n   ") == []


class TestKeywordSearchCore:
    """Tests for core keyword search functions and data models."""

    def test_matching_single_term(self, sample_three_page_pdf: Path):
        """Verifies searching for an existing term returns matching passage and correct page."""
        doc = extract_text_from_pdf(sample_three_page_pdf)
        results = search_document(doc, "ImageNet")

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.document_name == "deep_residual_learning.pdf"
        assert result.page_number == 3
        assert "Experiments on ImageNet" in result.passage
        assert "ImageNet" in result.passage

    def test_case_insensitive_matching(self, sample_three_page_pdf: Path):
        """Verifies search is case-insensitive (lowercase, uppercase, mixed case)."""
        doc = extract_text_from_pdf(sample_three_page_pdf)

        results_lower = search_document(doc, "resnet")
        results_upper = search_document(doc, "RESNET")
        results_mixed = search_document(doc, "ResNet")

        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert len(results_mixed) == 1

        assert results_lower[0].page_number == 3
        assert results_lower[0].passage == results_upper[0].passage == results_mixed[0].passage

    def test_multiple_matching_pages(self, sample_three_page_pdf: Path):
        """
        Verifies that a keyword appearing on multiple distinct pages returns results
        from all matching pages with their respective 1-based page numbers.
        """
        doc = extract_text_from_pdf(sample_three_page_pdf)

        # "neural networks" / "residual learning" appears on Page 1 and Page 2
        results = search_document(doc, "residual learning")

        assert len(results) == 2
        pages_found = [r.page_number for r in results]
        assert pages_found == [1, 2], "Match must be found on Page 1 and Page 2"

        # Page 1 match
        assert results[0].page_number == 1
        assert "residual learning framework" in results[0].passage

        # Page 2 match
        assert results[1].page_number == 2
        assert "2. Deep Residual Learning" in results[1].passage

    def test_no_matches_returns_empty_list(self, sample_three_page_pdf: Path):
        """Verifies searching for a non-existent term returns an empty list without error."""
        doc = extract_text_from_pdf(sample_three_page_pdf)
        results = search_document(doc, "quantum annealing hyperparameter")

        assert results == []
        assert isinstance(results, list)

    def test_empty_and_whitespace_queries_return_empty_list(self, sample_three_page_pdf: Path):
        """Verifies empty or whitespace-only queries clearly return empty lists."""
        doc = extract_text_from_pdf(sample_three_page_pdf)

        assert search_document(doc, "") == []
        assert search_document(doc, "   ") == []
        assert search_document(doc, "\t\n") == []

    def test_page_number_preservation_with_intermediate_blank_page(
        self, mixed_blank_and_text_pdf: Path
    ):
        """
        Critical test: Verifies that even if Page 2 is completely blank, a match on
        Page 3 retains page_number == 3 (no 0-based shift or blank-page collapse).
        """
        doc = extract_text_from_pdf(mixed_blank_and_text_pdf)
        results = search_document(doc, "page three")

        assert len(results) == 1
        assert results[0].page_number == 3
        assert results[0].document_name == "mixed_document.pdf"
        assert "Content on page three" in results[0].passage

    def test_blank_pdf_returns_no_matches(self, blank_pdf: Path):
        """Verifies searching in a document with no extractable text returns an empty list."""
        doc = extract_text_from_pdf(blank_pdf)
        results = search_document(doc, "anything")

        assert results == []

    def test_multi_word_phrase_matching(self, sample_three_page_pdf: Path):
        """Verifies matching multi-word phrases, including flexible whitespace."""
        doc = extract_text_from_pdf(sample_three_page_pdf)
        results = search_document(doc, "convolutional   neural    networks")

        assert len(results) == 1
        assert results[0].page_number == 1
        assert "Deep convolutional neural networks" in results[0].passage

    def test_search_pdf_convenience_function(self, sample_three_page_pdf: Path):
        """Verifies the direct search_pdf function extracts and searches end-to-end."""
        results = search_pdf(sample_three_page_pdf, "baseline 34-layer")

        assert len(results) == 1
        assert results[0].page_number == 3
        assert "baseline 34-layer plain net" in results[0].passage


class TestCLISearchIntegration:
    """Tests for CLI --search flag execution, output formatting, and exit codes."""

    def test_cli_search_matching_results(self, sample_three_page_pdf: Path, capsys):
        """CLI with --search displays document name, page numbers, matching passages, and exit 0."""
        exit_code = main([str(sample_three_page_pdf), "--search", "residual learning"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Document:     deep_residual_learning.pdf" in captured.out
        assert "Search Query: 'residual learning'" in captured.out
        assert "Found 2 matching passage(s):" in captured.out
        assert "--- [Result 1 | Page 1 of 3] ---" in captured.out
        assert "--- [Result 2 | Page 2 of 3] ---" in captured.out
        assert "Search completed successfully." in captured.out

    def test_cli_search_no_matches(self, sample_three_page_pdf: Path, capsys):
        """CLI with --search for non-existent keyword displays clear message and exit 0."""
        exit_code = main([str(sample_three_page_pdf), "--search", "unsupervised pretraining"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No matching passages found for query: 'unsupervised pretraining'" in captured.out

    def test_cli_search_empty_query(self, sample_three_page_pdf: Path, capsys):
        """CLI with empty --search query displays a clear guidance message and exit 0."""
        exit_code = main([str(sample_three_page_pdf), "--search", "   "])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "[Search] Query is empty. Please provide a non-empty search term." in captured.out

    def test_cli_search_on_missing_file_reports_error(self, tmp_path: Path, capsys):
        """CLI search on non-existent file returns exit code 1 with clean error."""
        missing = tmp_path / "missing_paper.pdf"
        exit_code = main([str(missing), "--search", "transformer"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[Error] PDF Extraction Failed" in captured.err
