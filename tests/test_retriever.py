"""
test_retriever.py: Unit and CLI integration tests for BM25 ranked retrieval.

Verifies:
1. BM25 ranking order (most relevant chunks score highest).
2. top_k parameter bounds and 1-based ranks.
3. Clean handling of empty queries and empty corpora.
4. Clean handling of queries with zero matches.
5. Tokenization and stopword removal behavior.
6. CLI --retrieve and --top-k integration.
"""

from pathlib import Path
import pymupdf
import pytest

from src.chunker import Chunk, chunk_pdf
from src.retriever import (
    RetrievalResult,
    retrieve,
    retrieve_from_pdf,
    tokenize,
)
from src.cli import main


@pytest.fixture
def sample_corpus() -> list[Chunk]:
    """Provides a synthetic 4-chunk corpus for deterministic retrieval testing."""
    return [
        Chunk(
            passage_id="doc.pdf::p1::c1",
            document_name="doc.pdf",
            page_number=1,
            text="The Transformer model relies on self-attention mechanisms to process sequential tokens.",
            char_count=87,
        ),
        Chunk(
            passage_id="doc.pdf::p1::c2",
            document_name="doc.pdf",
            page_number=1,
            text="Convolutional neural networks apply kernel filters over spatial image grids for computer vision.",
            char_count=98,
        ),
        Chunk(
            passage_id="doc.pdf::p2::c1",
            document_name="doc.pdf",
            page_number=2,
            text="Self-attention allows tokens to attend to all other tokens in a sequence simultaneously.",
            char_count=88,
        ),
        Chunk(
            passage_id="doc.pdf::p2::c2",
            document_name="doc.pdf",
            page_number=2,
            text="Recurrent neural networks process sequential data step by step with hidden state vectors.",
            char_count=91,
        ),
    ]


class TestRetriever:
    """Tests for core BM25 retrieval functionality."""

    def test_ranking_order(self, sample_corpus: list[Chunk]):
        """Chunks with higher term overlap score highest and appear at Rank 1."""
        query = "computer vision convolutional filters"
        results = retrieve(sample_corpus, query, top_k=4)

        assert len(results) > 0
        assert results[0].passage_id == "doc.pdf::p1::c2"
        assert results[0].rank == 1
        assert "Convolutional neural networks" in results[0].passage
        # Verify scores are strictly descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_bounds(self, sample_corpus: list[Chunk]):
        """top_k restricts the maximum number of retrieved passages."""
        query = "neural networks"
        results_top1 = retrieve(sample_corpus, query, top_k=1)
        results_top2 = retrieve(sample_corpus, query, top_k=2)

        assert len(results_top1) == 1
        assert len(results_top2) == 2
        assert results_top1[0].rank == 1
        assert results_top2[0].rank == 1
        assert results_top2[1].rank == 2

    def test_empty_query_returns_empty_list(self, sample_corpus: list[Chunk]):
        """Empty or whitespace queries safely return an empty list."""
        assert retrieve(sample_corpus, "") == []
        assert retrieve(sample_corpus, "   ") == []
        assert retrieve(sample_corpus, "\t\n") == []

    def test_empty_corpus_returns_empty_list(self):
        """Retrieving against an empty corpus safely returns an empty list."""
        assert retrieve([], "transformer") == []

    def test_no_matches_returns_empty_list(self, sample_corpus: list[Chunk]):
        """Queries with no matching terms in the corpus return an empty list."""
        results = retrieve(sample_corpus, "quantum superposition qubit")
        assert results == []

    def test_stopword_removal_preserves_salient_tokens(self):
        """Stopwords are removed while preserving key content tokens."""
        text = "What is the Transformer architecture and how does it work?"
        tokens = tokenize(text, remove_stopwords=True)

        assert "transformer" in tokens
        assert "architecture" in tokens
        assert "what" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens

    def test_all_stopwords_fallback(self):
        """If a query consists solely of stopwords, original tokens are retained."""
        query = "what is it"
        tokens = tokenize(query, remove_stopwords=True)
        assert len(tokens) > 0
        assert tokens == ["what", "is", "it"]

    def test_retrieve_from_pdf_convenience(self, sample_three_page_pdf: Path):
        """retrieve_from_pdf() extracts, chunks, and ranks end-to-end."""
        results = retrieve_from_pdf(sample_three_page_pdf, "ImageNet residual layer", top_k=2)
        assert len(results) > 0
        assert results[0].document_name == "deep_residual_learning.pdf"
        assert results[0].page_number == 3
        assert "Experiments on ImageNet" in results[0].passage

    def test_retriever_finds_passage_with_ligatures(self, tmp_path: Path):
        """
        The retriever must find that passage for the query 'configuration' when
        a PDF contains 'con\ufb01guration'.
        """
        pdf_path = tmp_path / "ligature_paper.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(
            pymupdf.Rect(50, 50, 500, 500),
            "<p>The network con\ufb01guration consists of twelve transformer layers.</p>",
        )
        doc.save(str(pdf_path))
        doc.close()

        results = retrieve_from_pdf(pdf_path, query="configuration", top_k=1)
        assert len(results) >= 1
        assert results[0].rank == 1
        assert "configuration" in results[0].passage


class TestCLIRetrievalIntegration:
    """Tests for CLI --retrieve and --top-k command-line flags."""

    def test_cli_retrieve_success(self, sample_three_page_pdf: Path, capsys):
        """CLI with --retrieve outputs rank, score, document, page, and passage."""
        exit_code = main([str(sample_three_page_pdf), "--retrieve", "ImageNet baseline", "--top-k", "2"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Document:        deep_residual_learning.pdf" in captured.out
        assert "Retrieval Query: 'ImageNet baseline'" in captured.out
        assert "Top-K Requested: 2" in captured.out
        assert "--- [Rank 1" in captured.out
        assert "Passage ID:" in captured.out
        assert "Retrieval completed successfully." in captured.out

    def test_cli_retrieve_no_matches(self, sample_three_page_pdf: Path, capsys):
        """CLI with --retrieve on non-matching query displays clear message."""
        exit_code = main([str(sample_three_page_pdf), "--retrieve", "quantum chromodynamics"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No relevant passages retrieved for query: 'quantum chromodynamics'" in captured.out

    def test_cli_retrieve_empty_query(self, sample_three_page_pdf: Path, capsys):
        """CLI with empty --retrieve displays clear empty query guidance."""
        exit_code = main([str(sample_three_page_pdf), "--retrieve", "   "])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "[Retrieval] Query is empty. Please provide a non-empty query term." in captured.out
