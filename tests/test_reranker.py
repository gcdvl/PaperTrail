"""
tests/test_reranker.py: Offline tests for CrossEncoderReranker and
retrieve_with_rerank. Uses a stub scorer — no model download required.
"""

import copy

import pytest

from src.chunker import Chunk
from src.reranker import (
    CrossEncoderReranker,
    DEFAULT_POOL,
    DEFAULT_RERANK_MODEL,
    retrieve_with_rerank,
)
from src.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(rank: int, passage_id: str = "", passage: str = "",
                 score: float = 0.0, page_number: int = 1,
                 document_name: str = "test.pdf") -> RetrievalResult:
    """Build a RetrievalResult with sensible defaults."""
    return RetrievalResult(
        passage_id=passage_id or f"test.pdf::p{page_number}::c{rank}",
        document_name=document_name,
        page_number=page_number,
        passage=passage or f"passage {rank}",
        score=score,
        rank=rank,
    )


def _stub_scorer(score_map: dict):
    """
    Returns a scorer callable that maps passage text -> rerank score.
    Any passage not in score_map gets 0.0.
    """
    def scorer(pairs):
        return [score_map.get(passage, 0.0) for _query, passage in pairs]
    return scorer


# ---------------------------------------------------------------------------
# Tests: rerank ordering
# ---------------------------------------------------------------------------

class TestRerankOrdering:
    """Reranked order follows stub scores."""

    def test_order_follows_scores(self):
        results = [_make_result(1, passage="A"), _make_result(2, passage="B"),
                   _make_result(3, passage="C")]
        scorer = _stub_scorer({"A": 1.0, "B": 5.0, "C": 3.0})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", results, top_k=3)

        assert [r.passage for r in out] == ["B", "C", "A"]

    def test_ties_keep_original_bm25_order(self):
        results = [_make_result(1, passage="X"), _make_result(2, passage="Y"),
                   _make_result(3, passage="Z")]
        scorer = _stub_scorer({"X": 2.0, "Y": 2.0, "Z": 2.0})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", results, top_k=3)

        # All tied at 2.0; original order X, Y, Z should be preserved
        assert [r.passage for r in out] == ["X", "Y", "Z"]

    def test_partial_ties(self):
        """When only some items tie, tied items keep BM25 order among themselves."""
        results = [_make_result(1, passage="A"), _make_result(2, passage="B"),
                   _make_result(3, passage="C"), _make_result(4, passage="D")]
        # B and C tie at 3.0; A wins at 5.0; D loses at 1.0
        scorer = _stub_scorer({"A": 5.0, "B": 3.0, "C": 3.0, "D": 1.0})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", results, top_k=4)

        assert [r.passage for r in out] == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Tests: top_k truncation
# ---------------------------------------------------------------------------

class TestTopKTruncation:
    """top_k limits the number of returned results."""

    def test_top_k_truncates(self):
        results = [_make_result(i, passage=f"p{i}") for i in range(1, 6)]
        scorer = _stub_scorer({f"p{i}": float(6 - i) for i in range(1, 6)})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", results, top_k=2)

        assert len(out) == 2
        assert [r.passage for r in out] == ["p1", "p2"]

    def test_pool_smaller_than_top_k(self):
        """When fewer results exist than top_k, return all that exist."""
        results = [_make_result(1, passage="only"), _make_result(2, passage="two")]
        scorer = _stub_scorer({"only": 1.0, "two": 2.0})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", results, top_k=10)

        assert len(out) == 2


# ---------------------------------------------------------------------------
# Tests: rank and field preservation
# ---------------------------------------------------------------------------

class TestRankAndFieldPreservation:
    """Ranks are 1-based consecutive; metadata fields are preserved; score is stub."""

    def test_ranks_are_consecutive_1based(self):
        results = [_make_result(1, passage="A"), _make_result(2, passage="B"),
                   _make_result(3, passage="C")]
        scorer = _stub_scorer({"A": 0.5, "B": 1.5, "C": 1.0})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", results, top_k=3)

        assert [r.rank for r in out] == [1, 2, 3]

    def test_metadata_preserved(self):
        orig = RetrievalResult(
            passage_id="doc.pdf::p3::c2",
            document_name="doc.pdf",
            page_number=3,
            passage="important text",
            score=9.99,
            rank=7,
        )
        scorer = _stub_scorer({"important text": 42.0})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", [orig], top_k=1)

        assert len(out) == 1
        r = out[0]
        assert r.passage_id == "doc.pdf::p3::c2"
        assert r.document_name == "doc.pdf"
        assert r.page_number == 3
        assert r.passage == "important text"

    def test_score_is_stub_score(self):
        results = [_make_result(1, passage="A")]
        scorer = _stub_scorer({"A": 77.7})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", results, top_k=1)

        assert out[0].score == pytest.approx(77.7)


# ---------------------------------------------------------------------------
# Tests: empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Empty inputs return []."""

    def test_empty_results_returns_empty(self):
        rr = CrossEncoderReranker(scorer=lambda pairs: [])
        assert rr.rerank("q", [], top_k=5) == []

    def test_top_k_zero_returns_empty(self):
        results = [_make_result(1, passage="A")]
        scorer = _stub_scorer({"A": 1.0})
        rr = CrossEncoderReranker(scorer=scorer)
        assert rr.rerank("q", results, top_k=0) == []


# ---------------------------------------------------------------------------
# Tests: immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    """Input list and its objects are not mutated by rerank."""

    def test_input_list_not_mutated(self):
        results = [_make_result(1, passage="A"), _make_result(2, passage="B")]
        original_list = list(results)  # shallow copy of the list
        scorer = _stub_scorer({"A": 2.0, "B": 1.0})
        rr = CrossEncoderReranker(scorer=scorer)
        rr.rerank("q", results, top_k=2)

        assert results == original_list

    def test_input_objects_not_mutated(self):
        orig = _make_result(1, passage="X", score=5.0)
        orig_copy = copy.deepcopy(orig)
        scorer = _stub_scorer({"X": 99.0})
        rr = CrossEncoderReranker(scorer=scorer)
        out = rr.rerank("q", [orig], top_k=1)

        # Original object unchanged
        assert orig.score == orig_copy.score
        assert orig.rank == orig_copy.rank
        # Output is different from input
        assert out[0].score == pytest.approx(99.0)
        assert out[0].rank == 1


# ---------------------------------------------------------------------------
# Tests: retrieve_with_rerank
# ---------------------------------------------------------------------------

class TestRetrieveWithRerank:
    """retrieve_with_rerank chains BM25 and reranking correctly."""

    def test_uses_pool_for_bm25_and_returns_top_k(self):
        # Build enough chunks so BM25 returns several candidates
        chunks = [
            Chunk(
                passage_id=f"doc.pdf::p1::c{i}",
                document_name="doc.pdf",
                page_number=1,
                text=f"unique_term_{i} common information passage",
                char_count=40,
            )
            for i in range(1, 11)
        ]

        # Stub scorer: higher index -> higher rerank score
        def scorer(pairs):
            scores = []
            for _q, passage in pairs:
                # Extract the index from "unique_term_<i>"
                for token in passage.split():
                    if token.startswith("unique_term_"):
                        scores.append(float(token.split("_")[-1]))
                        break
                else:
                    scores.append(0.0)
            return scores

        rr = CrossEncoderReranker(scorer=scorer)
        # Query that matches all chunks via "common"
        out = retrieve_with_rerank(chunks, "common information", rr,
                                   top_k=3, pool=8)

        # Should return at most 3 results
        assert len(out) <= 3
        # Ranks are 1-based consecutive
        assert [r.rank for r in out] == list(range(1, len(out) + 1))
        # Scores should be descending
        scores = [r.score for r in out]
        assert scores == sorted(scores, reverse=True)
        # Metadata preserved
        for r in out:
            assert r.document_name == "doc.pdf"
            assert r.page_number == 1

    def test_empty_chunks_returns_empty(self):
        rr = CrossEncoderReranker(scorer=lambda pairs: [])
        out = retrieve_with_rerank([], "query", rr, top_k=5, pool=20)
        assert out == []


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Module constants have expected values."""

    def test_default_model(self):
        assert DEFAULT_RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_default_pool(self):
        assert DEFAULT_POOL == 20

