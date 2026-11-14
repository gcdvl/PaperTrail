"""
test_metrics_v2.py: Unit tests for Milestone 3.5 passage-level metrics, evidence phrase
matching, score distribution statistics, validator, and unanswerable-item handling.

All expected values are hand-computed.
"""

import json
from pathlib import Path
import sys

import pymupdf
import pytest

from src.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    normalize_whitespace,
    passage_contains_evidence,
    passage_recall_at_k,
    passage_reciprocal_rank,
    score_distribution_stats,
)


# ---------------------------------------------------------------------------
# Tests for normalize_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    """Tests for whitespace normalization utility."""

    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_whitespace("hello\t\n  world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_single_word(self):
        assert normalize_whitespace("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# Tests for passage_contains_evidence
# ---------------------------------------------------------------------------

class TestPassageContainsEvidence:
    """Tests for evidence phrase matching in passages."""

    def test_exact_match(self):
        passage = "The Transformer model uses an attention mechanism."
        assert passage_contains_evidence(passage, "attention mechanism") is True

    def test_case_insensitive(self):
        passage = "The Transformer model uses an Attention Mechanism."
        assert passage_contains_evidence(passage, "attention mechanism") is True

    def test_whitespace_normalized(self):
        passage = "uses  an   attention\n\nmechanism for"
        assert passage_contains_evidence(passage, "attention mechanism") is True

    def test_evidence_with_extra_whitespace(self):
        passage = "uses an attention mechanism for"
        assert passage_contains_evidence(passage, "attention   mechanism") is True

    def test_no_match(self):
        passage = "The model uses convolution layers."
        assert passage_contains_evidence(passage, "attention mechanism") is False

    def test_empty_passage(self):
        assert passage_contains_evidence("", "attention") is False

    def test_empty_evidence(self):
        assert passage_contains_evidence("some passage", "") is False

    def test_both_empty(self):
        assert passage_contains_evidence("", "") is False

    def test_ligature_passage_matches_plain_evidence(self):
        """A passage containing 'con\ufb01guration' must match the phrase 'configuration'."""
        passage = "The network con\ufb01guration uses twelve layers."
        assert passage_contains_evidence(passage, "configuration") is True

    def test_plain_passage_matches_ligature_evidence(self):
        """A passage containing 'configuration' matches an evidence phrase with ligature 'con\ufb01guration'."""
        passage = "The network configuration uses twelve layers."
        assert passage_contains_evidence(passage, "con\ufb01guration") is True

    def test_fl_ligature_matching(self):
        """Ligature \ufb02 (fl) in passage or evidence is normalized via NFKC."""
        passage = "Optical \ufb02ow estimation in video processing."
        assert passage_contains_evidence(passage, "flow") is True


# ---------------------------------------------------------------------------
# Tests for passage_recall_at_k (hand-computed)
# ---------------------------------------------------------------------------

class TestPassageRecallAtK:
    """Hand-computed tests for passage-level Recall@k."""

    def test_hit_at_rank_1(self):
        """Evidence is in the first passage → Recall@1 = 1.0."""
        passages = [
            "The attention mechanism maps queries to outputs.",
            "Convolution layers are used in CNNs.",
            "Residual connections help gradient flow.",
        ]
        # "attention mechanism" is in passages[0]
        assert passage_recall_at_k(passages, "attention mechanism", k=1) == 1.0
        assert passage_recall_at_k(passages, "attention mechanism", k=3) == 1.0

    def test_hit_at_rank_2(self):
        """Evidence is in the second passage → Recall@1 = 0.0, Recall@2 = 1.0."""
        passages = [
            "Residual connections help gradient flow.",
            "The attention mechanism maps queries to outputs.",
            "Convolution layers are used in CNNs.",
        ]
        assert passage_recall_at_k(passages, "attention mechanism", k=1) == 0.0
        assert passage_recall_at_k(passages, "attention mechanism", k=2) == 1.0

    def test_no_hit(self):
        """Evidence not in any passage → Recall@k = 0.0 for all k."""
        passages = [
            "Residual connections help gradient flow.",
            "Convolution layers are used in CNNs.",
        ]
        assert passage_recall_at_k(passages, "attention mechanism", k=1) == 0.0
        assert passage_recall_at_k(passages, "attention mechanism", k=5) == 0.0

    def test_empty_evidence(self):
        """Empty evidence phrase → 0.0."""
        assert passage_recall_at_k(["some passage"], "", k=5) == 0.0

    def test_k_zero_or_negative(self):
        """k <= 0 → 0.0."""
        assert passage_recall_at_k(["attention mechanism"], "attention", k=0) == 0.0
        assert passage_recall_at_k(["attention mechanism"], "attention", k=-1) == 0.0

    def test_empty_passages(self):
        """No passages → 0.0."""
        assert passage_recall_at_k([], "attention", k=5) == 0.0

    def test_whitespace_normalized_matching(self):
        """Evidence with internal whitespace differences should still match."""
        passages = ["The  attention\nmechanism  works well."]
        assert passage_recall_at_k(passages, "attention mechanism", k=1) == 1.0


# ---------------------------------------------------------------------------
# Tests for passage_reciprocal_rank (hand-computed)
# ---------------------------------------------------------------------------

class TestPassageReciprocalRank:
    """Hand-computed tests for passage-level Reciprocal Rank."""

    def test_rank_1(self):
        """Evidence in first passage → RR = 1/1 = 1.0."""
        passages = [
            "The attention mechanism maps queries.",
            "Convolution layers handle images.",
        ]
        assert passage_reciprocal_rank(passages, "attention mechanism") == 1.0

    def test_rank_2(self):
        """Evidence in second passage → RR = 1/2 = 0.5."""
        passages = [
            "Convolution layers handle images.",
            "The attention mechanism maps queries.",
        ]
        assert passage_reciprocal_rank(passages, "attention mechanism") == 0.5

    def test_rank_3(self):
        """Evidence in third passage → RR = 1/3 ≈ 0.333."""
        passages = [
            "Convolution layers handle images.",
            "Residual connections help gradient flow.",
            "The attention mechanism maps queries.",
        ]
        assert pytest.approx(
            passage_reciprocal_rank(passages, "attention mechanism"), abs=0.0001
        ) == 1.0 / 3.0

    def test_no_match(self):
        """Evidence not found → RR = 0.0."""
        passages = [
            "Convolution layers handle images.",
            "Residual connections help gradient flow.",
        ]
        assert passage_reciprocal_rank(passages, "attention mechanism") == 0.0

    def test_empty_evidence(self):
        assert passage_reciprocal_rank(["some passage"], "") == 0.0

    def test_empty_passages(self):
        assert passage_reciprocal_rank([], "attention") == 0.0


# ---------------------------------------------------------------------------
# Tests for passage-level MRR (composing with mean_reciprocal_rank)
# ---------------------------------------------------------------------------

class TestPassageMRR:
    """Hand-computed passage-level MRR from individual RR values."""

    def test_passage_mrr_hand_computed(self):
        """
        Three queries:
          Q1: evidence at rank 1 → RR = 1.0
          Q2: evidence at rank 2 → RR = 0.5
          Q3: evidence not found → RR = 0.0
        MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
        """
        passages_q1 = ["The attention mechanism works.", "Something else."]
        passages_q2 = ["Something else.", "Residual learning framework."]
        passages_q3 = ["Something else.", "Another thing."]

        rr1 = passage_reciprocal_rank(passages_q1, "attention mechanism")
        rr2 = passage_reciprocal_rank(passages_q2, "residual learning")
        rr3 = passage_reciprocal_rank(passages_q3, "transformer architecture")

        assert rr1 == 1.0
        assert rr2 == 0.5
        assert rr3 == 0.0

        mrr = mean_reciprocal_rank([rr1, rr2, rr3])
        assert mrr == 0.5


# ---------------------------------------------------------------------------
# Tests for score_distribution_stats
# ---------------------------------------------------------------------------

class TestScoreDistributionStats:
    """Tests for score distribution statistics."""

    def test_basic_distribution(self):
        """Hand-computed stats for [1.0, 2.0, 3.0, 4.0, 5.0]."""
        stats = score_distribution_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0

    def test_even_count_median(self):
        """Even count: median = average of two middle values."""
        # [1.0, 2.0, 3.0, 4.0] → median = (2.0 + 3.0) / 2 = 2.5
        stats = score_distribution_stats([1.0, 2.0, 3.0, 4.0])
        assert stats["count"] == 4
        assert stats["median"] == 2.5

    def test_single_value(self):
        stats = score_distribution_stats([42.0])
        assert stats["count"] == 1
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["mean"] == 42.0
        assert stats["median"] == 42.0

    def test_empty_list(self):
        stats = score_distribution_stats([])
        assert stats["count"] == 0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0

    def test_unsorted_input(self):
        """Input doesn't need to be sorted; function sorts internally."""
        stats = score_distribution_stats([5.0, 1.0, 3.0])
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0


# ---------------------------------------------------------------------------
# Tests for unanswerable-item handling in metrics
# ---------------------------------------------------------------------------

class TestUnanswerableHandling:
    """
    Verifies that unanswerable items are properly excluded from passage-level
    metrics. In the eval harness, unanswerable items do NOT contribute to
    passage_recall or passage_rr lists. These tests verify the exclusion logic
    at the metric level: the metric functions return 0.0 / are not called
    for unanswerable items (evidence_phrase is None/empty).
    """

    def test_passage_recall_with_none_evidence(self):
        """Passage recall with empty/None-equivalent evidence → 0.0."""
        passages = ["Some passage text with information."]
        assert passage_recall_at_k(passages, "", k=5) == 0.0

    def test_passage_rr_with_none_evidence(self):
        """Passage reciprocal rank with empty evidence → 0.0."""
        passages = ["Some passage text with information."]
        assert passage_reciprocal_rank(passages, "") == 0.0

    def test_unanswerable_excluded_from_mrr(self):
        """
        Simulate eval loop: 2 answerable (RR=1.0, RR=0.5) + 1 unanswerable (excluded).
        Passage MRR should be computed over answerable only: (1.0 + 0.5) / 2 = 0.75.
        """
        answerable_rrs = []
        # Answerable Q1: evidence at rank 1
        answerable_rrs.append(
            passage_reciprocal_rank(["attention mechanism here"], "attention mechanism")
        )
        # Answerable Q2: evidence at rank 2
        answerable_rrs.append(
            passage_reciprocal_rank(["no match", "residual learning here"], "residual learning")
        )
        # Unanswerable Q3: NOT added to the list (excluded by eval loop)

        assert answerable_rrs == [1.0, 0.5]
        mrr = mean_reciprocal_rank(answerable_rrs)
        assert mrr == 0.75

    def test_score_distribution_separation(self):
        """
        Simulate score separation: answerable items have high scores,
        unanswerable items have low scores.
        """
        answerable_scores = [5.0, 4.5, 6.0]
        unanswerable_scores = [0.5, 0.8, 0.3]

        ans_stats = score_distribution_stats(answerable_scores)
        unans_stats = score_distribution_stats(unanswerable_scores)

        assert ans_stats["min"] == 4.5
        assert ans_stats["max"] == 6.0
        assert pytest.approx(ans_stats["mean"], abs=0.01) == (5.0 + 4.5 + 6.0) / 3

        assert unans_stats["min"] == 0.3
        assert unans_stats["max"] == 0.8
        assert pytest.approx(unans_stats["mean"], abs=0.01) == (0.5 + 0.8 + 0.3) / 3


# ---------------------------------------------------------------------------
# Tests for the dataset validator
# ---------------------------------------------------------------------------

class TestDatasetValidator:
    """Tests for scripts/validate_dataset.py validation logic."""

    @pytest.fixture
    def validator_module(self):
        """Import the validator module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "validate_dataset",
            str(Path(__file__).resolve().parent.parent / "scripts" / "validate_dataset.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_valid_answerable_item(self, validator_module):
        """A well-formed answerable item passes schema validation."""
        item = {
            "id": "q01",
            "document": "paper.pdf",
            "question": "What is X?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "X is defined as",
        }
        errors = validator_module.validate_schema(item, 0)
        assert errors == []

    def test_valid_unanswerable_item(self, validator_module):
        """A well-formed unanswerable item passes schema validation."""
        item = {
            "id": "q02",
            "document": "paper.pdf",
            "question": "What is Y?",
            "answerable": False,
            "expected_page_numbers": [],
            "evidence_phrase": None,
        }
        errors = validator_module.validate_schema(item, 0)
        assert errors == []

    def test_missing_field(self, validator_module):
        """Missing required field produces an error."""
        item = {
            "id": "q03",
            "document": "paper.pdf",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "something",
        }
        errors = validator_module.validate_schema(item, 0)
        assert any("question" in e for e in errors)

    def test_answerable_missing_evidence(self, validator_module):
        """Answerable item with null evidence_phrase is an error."""
        item = {
            "id": "q04",
            "document": "paper.pdf",
            "question": "What?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": None,
        }
        errors = validator_module.validate_schema(item, 0)
        assert any("evidence_phrase" in e for e in errors)

    def test_answerable_empty_pages(self, validator_module):
        """Answerable item with empty pages is an error."""
        item = {
            "id": "q05",
            "document": "paper.pdf",
            "question": "What?",
            "answerable": True,
            "expected_page_numbers": [],
            "evidence_phrase": "something",
        }
        errors = validator_module.validate_schema(item, 0)
        assert any("expected_page_numbers" in e for e in errors)

    def test_unanswerable_with_evidence(self, validator_module):
        """Unanswerable item with non-null evidence_phrase is an error."""
        item = {
            "id": "q06",
            "document": "paper.pdf",
            "question": "What?",
            "answerable": False,
            "expected_page_numbers": [],
            "evidence_phrase": "should not be here",
        }
        errors = validator_module.validate_schema(item, 0)
        assert any("evidence_phrase" in e for e in errors)

    def test_unanswerable_with_pages(self, validator_module):
        """Unanswerable item with non-empty pages is an error."""
        item = {
            "id": "q07",
            "document": "paper.pdf",
            "question": "What?",
            "answerable": False,
            "expected_page_numbers": [1],
            "evidence_phrase": None,
        }
        errors = validator_module.validate_schema(item, 0)
        assert any("expected_page_numbers" in e for e in errors)

    def test_invalid_page_number(self, validator_module):
        """Page number < 1 produces an error."""
        item = {
            "id": "q08",
            "document": "paper.pdf",
            "question": "What?",
            "answerable": True,
            "expected_page_numbers": [0],
            "evidence_phrase": "something",
        }
        errors = validator_module.validate_schema(item, 0)
        assert any("page number" in e.lower() for e in errors)

    def test_evidence_grounding_passes(self, validator_module, tmp_path):
        """Evidence phrase found on expected page passes grounding check."""
        # Create a test PDF with known text
        pdf_path = tmp_path / "test_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), "The attention mechanism is fundamental.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        item = {
            "id": "q_grounding",
            "document": "test_doc.pdf",
            "question": "What is fundamental?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "attention mechanism",
        }
        cache = {}
        errors = validator_module.validate_evidence_on_page(
            item, tmp_path, tmp_path, cache
        )
        assert errors == []

    def test_evidence_grounding_fails(self, validator_module, tmp_path):
        """Evidence phrase NOT on expected page fails grounding check."""
        pdf_path = tmp_path / "test_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), "Convolution layers are used.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        item = {
            "id": "q_missing",
            "document": "test_doc.pdf",
            "question": "What mechanism?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "attention mechanism",
        }
        cache = {}
        errors = validator_module.validate_evidence_on_page(
            item, tmp_path, tmp_path, cache
        )
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_evidence_grounding_passes_with_ligatures(self, validator_module, tmp_path):
        """Grounding succeeds when PDF text contains ligatures and evidence phrase uses plain ASCII."""
        pdf_path = tmp_path / "ligature_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(
            pymupdf.Rect(50, 50, 500, 500),
            "<p>The network con\ufb01guration consists of attention layers.</p>",
        )
        doc.save(str(pdf_path))
        doc.close()

        item = {
            "id": "q_lig",
            "document": "ligature_doc.pdf",
            "question": "What configuration?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "network configuration",
        }
        cache = {}
        errors = validator_module.validate_evidence_on_page(
            item, tmp_path, tmp_path, cache
        )
        assert errors == []

    def test_full_dataset_validation(self, validator_module, tmp_path):
        """Full dataset validation with mixed valid and invalid items."""
        # Create a test PDF
        pdf_path = tmp_path / "papers" / "test.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), "The evidence phrase here.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        dataset = [
            {
                "id": "ok1",
                "document": "test.pdf",
                "question": "What?",
                "answerable": True,
                "expected_page_numbers": [1],
                "evidence_phrase": "evidence phrase",
            },
            {
                "id": "ok2",
                "document": "test.pdf",
                "question": "What is missing?",
                "answerable": False,
                "expected_page_numbers": [],
                "evidence_phrase": None,
            },
        ]

        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        errors, total = validator_module.validate_dataset(
            dataset_path,
            papers_dir=pdf_path.parent,
            project_root=tmp_path,
        )
        assert total == 2
        assert errors == []

    def test_duplicate_ids_detected(self, validator_module, tmp_path):
        """Duplicate IDs are flagged."""
        dataset = [
            {
                "id": "dup",
                "document": "test.pdf",
                "question": "Q1",
                "answerable": False,
                "expected_page_numbers": [],
                "evidence_phrase": None,
            },
            {
                "id": "dup",
                "document": "test.pdf",
                "question": "Q2",
                "answerable": False,
                "expected_page_numbers": [],
                "evidence_phrase": None,
            },
        ]
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        errors, total = validator_module.validate_dataset(
            dataset_path,
            papers_dir=tmp_path,
            project_root=tmp_path,
            check_evidence=False,
        )
        assert total == 2
        assert any("duplicate" in e.lower() for e in errors)

    def test_find_suggestions_reports_other_pages_and_top_candidates(
        self, validator_module, tmp_path
    ):
        """find_suggestions identifies other pages containing the phrase and top-3 overlap candidates on expected pages."""
        pdf_path = tmp_path / "multi_page.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text(
            (50, 72),
            "1. Introduction\n"
            "Deep convolutional neural networks are widely used in computer vision.\n"
            "We propose a novel residual learning framework for easier optimization.\n"
            "Our baseline model achieves high training accuracy on ImageNet.",
            fontsize=11,
        )
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text(
            (50, 72),
            "2. Attention Mechanisms\n"
            "The multi-head attention mechanism computes scaled dot-product attention.",
            fontsize=11,
        )
        doc.save(str(pdf_path))
        doc.close()

        item = {
            "id": "q_suggest",
            "document": "multi_page.pdf",
            "question": "What residual learning framework is proposed for deep neural networks?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "multi-head attention mechanism",  # On page 2, not page 1
        }
        suggestions = validator_module.find_suggestions(item, tmp_path, tmp_path)

        assert suggestions["other_pages"] == [2]
        assert len(suggestions["top_candidates"]) <= 3
        assert len(suggestions["top_candidates"]) > 0
        top_cand, top_score = suggestions["top_candidates"][0]
        assert "residual learning framework" in top_cand
        assert top_score > 0

    def test_find_suggestions_when_phrase_nowhere_in_doc(
        self, validator_module, tmp_path
    ):
        """When evidence phrase is not anywhere in the document, other_pages is empty."""
        pdf_path = tmp_path / "single_page.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 72), "Only general statements here.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        item = {
            "id": "q_nowhere",
            "document": "single_page.pdf",
            "question": "Where is the phantom phrase?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "completely non-existent phrase",
        }
        suggestions = validator_module.find_suggestions(item, tmp_path, tmp_path)

        assert suggestions["other_pages"] == []
        assert len(suggestions["top_candidates"]) >= 1

    def test_validate_dataset_with_suggest_flag(
        self, validator_module, tmp_path
    ):
        """validate_dataset includes suggestions when suggest=True."""
        pdf_path = tmp_path / "doc.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 72), "Page one discusses model training dynamics.", fontsize=11)
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((50, 72), "Page two contains target evidence phrase here.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        dataset = [
            {
                "id": "q_test",
                "document": "doc.pdf",
                "question": "What is on page two?",
                "answerable": True,
                "expected_page_numbers": [1],  # Wrong page
                "evidence_phrase": "target evidence phrase",  # Actually on page 2
            }
        ]
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        # Without suggest
        errors_no_sug, total = validator_module.validate_dataset(
            dataset_path,
            papers_dir=tmp_path,
            project_root=tmp_path,
            suggest=False,
        )
        assert len(errors_no_sug) == 1
        assert "Suggested line(s)/sentence(s)" not in errors_no_sug[0]

        # With suggest
        errors_sug, total = validator_module.validate_dataset(
            dataset_path,
            papers_dir=tmp_path,
            project_root=tmp_path,
            suggest=True,
        )
        assert len(errors_sug) == 1
        assert "Suggested line(s)/sentence(s)" in errors_sug[0]
        assert "Other page(s) where evidence phrase exists: [2]" in errors_sug[0]

    def test_validate_dataset_does_not_modify_file(
        self, validator_module, tmp_path
    ):
        """validate_dataset with suggest=True does not modify the dataset file on disk."""
        dataset = [
            {
                "id": "q_immutable",
                "document": "doc.pdf",
                "question": "Question text?",
                "answerable": True,
                "expected_page_numbers": [1],
                "evidence_phrase": "phrase",
            }
        ]
        dataset_path = tmp_path / "dataset.json"
        original_content = json.dumps(dataset, indent=2)
        dataset_path.write_text(original_content, encoding="utf-8")

        pdf_path = tmp_path / "doc.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 72), "Different text on page one.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        validator_module.validate_dataset(
            dataset_path,
            papers_dir=tmp_path,
            project_root=tmp_path,
            suggest=True,
        )

        assert dataset_path.read_text(encoding="utf-8") == original_content

    def test_cli_suggest_flag_output(
        self, validator_module, tmp_path, monkeypatch, capsys
    ):
        """CLI invocation with --suggest outputs suggestions to stdout."""
        pdf_path = tmp_path / "sample.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 72), "First page content about learning rates.", fontsize=11)
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((50, 72), "Second page holds secret finding.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        dataset = [
            {
                "id": "cli_item",
                "document": "sample.pdf",
                "question": "What is the secret finding?",
                "answerable": True,
                "expected_page_numbers": [1],
                "evidence_phrase": "secret finding",
            }
        ]
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        test_args = [
            "validate_dataset.py",
            "--dataset", str(dataset_path),
            "--papers-dir", str(tmp_path),
            "--suggest",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        exit_code = validator_module.main()
        assert exit_code == 1  # Validation failed
        captured = capsys.readouterr()
        assert "Suggested line(s)/sentence(s)" in captured.out
        assert "secret finding" in captured.out
        assert "Other page(s) where evidence phrase exists: [2]" in captured.out
