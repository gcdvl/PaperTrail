"""
metrics.py: Evaluation metrics for information retrieval in PaperTrail.

Provides decoupled, pure functions to calculate:
- Page-level Recall@k and Mean Reciprocal Rank (MRR).
- Passage-level Recall@k and MRR using evidence phrase matching.
- Score distribution statistics for answerable vs. unanswerable items.

No print statements or terminal dependencies are included in this module.
"""

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Whitespace normalization for evidence phrase matching
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapses all whitespace sequences to single spaces and strips."""
    return re.sub(r"\s+", " ", text).strip()


def passage_contains_evidence(passage: str, evidence_phrase: str) -> bool:
    """
    Checks if a passage contains the evidence phrase.

    Matching is whitespace-normalized and case-insensitive: both strings
    have all whitespace runs collapsed to a single space, then compared
    in lowercase.

    Args:
        passage: The retrieved passage text.
        evidence_phrase: The expected evidence substring.

    Returns:
        True if the normalized evidence phrase is found in the normalized passage.
    """
    if not passage or not evidence_phrase:
        return False
    norm_passage = normalize_whitespace(unicodedata.normalize("NFKC", passage)).lower()
    norm_evidence = normalize_whitespace(unicodedata.normalize("NFKC", evidence_phrase)).lower()
    return norm_evidence in norm_passage


# ---------------------------------------------------------------------------
# Page-level metrics (existing, unchanged API)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_pages: Sequence[int], expected_pages: Sequence[int], k: int) -> float:
    """
    Computes Recall@k (Hit@k) for a single query.

    Checks whether at least one relevant expected page is present in the top-k
    retrieved page results.

    Args:
        retrieved_pages: Ordered sequence of retrieved 1-based page numbers.
        expected_pages: Sequence of ground-truth 1-based page numbers.
        k: The rank cutoff threshold (e.g., 1, 3, 5).

    Returns:
        1.0 if at least one expected page is in the top-k retrieved pages, 0.0 otherwise.
        Returns 0.0 if expected_pages is empty or k <= 0.
    """
    if not expected_pages or k <= 0:
        return 0.0

    top_k_pages = retrieved_pages[:k]
    expected_set = set(expected_pages)

    for page in top_k_pages:
        if page in expected_set:
            return 1.0

    return 0.0


def reciprocal_rank(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """
    Computes Reciprocal Rank (RR) for a single query.

    Finds the 1-based rank position of the first relevant page in the retrieved list.
    Reciprocal rank is 1 / rank. If no relevant page is found, returns 0.0.

    Args:
        retrieved_pages: Ordered sequence of retrieved 1-based page numbers.
        expected_pages: Sequence of ground-truth 1-based page numbers.

    Returns:
        1.0 / rank of first relevant page, or 0.0 if no relevant page was retrieved.
    """
    if not expected_pages:
        return 0.0

    expected_set = set(expected_pages)

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(rr_scores: Sequence[float]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across multiple queries.

    Args:
        rr_scores: List of reciprocal rank scores for each query.

    Returns:
        Average reciprocal rank, or 0.0 if rr_scores is empty.
    """
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)


def mean_recall_at_k(recall_scores: Sequence[float]) -> float:
    """
    Computes average Recall@k across multiple queries.

    Args:
        recall_scores: List of Recall@k scores for each query.

    Returns:
        Average Recall@k, or 0.0 if recall_scores is empty.
    """
    if not recall_scores:
        return 0.0
    return sum(recall_scores) / len(recall_scores)


# ---------------------------------------------------------------------------
# Passage-level metrics (new in Milestone 3.5)
# ---------------------------------------------------------------------------

def passage_recall_at_k(
    retrieved_passages: Sequence[str],
    evidence_phrase: str,
    k: int,
) -> float:
    """
    Computes passage-level Recall@k for a single query.

    A retrieved chunk counts as a hit only if it contains the evidence_phrase
    (whitespace-normalized, case-insensitive).

    Args:
        retrieved_passages: Ordered sequence of retrieved passage texts.
        evidence_phrase: The expected evidence substring.
        k: The rank cutoff threshold.

    Returns:
        1.0 if at least one passage in the top-k contains the evidence phrase,
        0.0 otherwise. Returns 0.0 if evidence_phrase is empty or k <= 0.
    """
    if not evidence_phrase or k <= 0:
        return 0.0

    for passage in retrieved_passages[:k]:
        if passage_contains_evidence(passage, evidence_phrase):
            return 1.0

    return 0.0


def passage_reciprocal_rank(
    retrieved_passages: Sequence[str],
    evidence_phrase: str,
) -> float:
    """
    Computes passage-level Reciprocal Rank for a single query.

    The rank of the first passage containing the evidence phrase (whitespace-
    normalized, case-insensitive) determines the reciprocal rank.

    Args:
        retrieved_passages: Ordered sequence of retrieved passage texts.
        evidence_phrase: The expected evidence substring.

    Returns:
        1.0 / rank of the first passage containing the evidence phrase,
        or 0.0 if no passage contains it. Returns 0.0 if evidence_phrase is empty.
    """
    if not evidence_phrase:
        return 0.0

    for rank, passage in enumerate(retrieved_passages, start=1):
        if passage_contains_evidence(passage, evidence_phrase):
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------------------------
# Score distribution for unanswerable threshold analysis
# ---------------------------------------------------------------------------

def score_distribution_stats(scores: Sequence[float]) -> dict:
    """
    Computes basic distribution statistics for a sequence of scores.

    Args:
        scores: Sequence of float scores.

    Returns:
        Dictionary with keys: count, min, max, mean, median.
        Returns all zeros/0 count if scores is empty.
    """
    if not scores:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    mean = sum(sorted_scores) / n

    if n % 2 == 1:
        median = sorted_scores[n // 2]
    else:
        median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0

    return {
        "count": n,
        "min": sorted_scores[0],
        "max": sorted_scores[-1],
        "mean": mean,
        "median": median,
    }


# ---------------------------------------------------------------------------
# Answer evaluation metrics (overall & hit/miss breakdown)
# ---------------------------------------------------------------------------

def summarize_answer_metrics(records: Sequence[dict]) -> dict:
    """
    Computes summary metrics for answer evaluation over a sequence of item records.

    Expected record keys:
      - expected_answerable (bool or None): Ground-truth answerable status.
      - model_answerable (bool): Model predicted answerable status.
      - citations (list): List of cited passage IDs.
      - citations_valid (bool): Whether all cited IDs are in the retrieved pool.
      - grounded (bool): Whether at least one cited passage contains evidence_phrase.
      - parse_ok (bool): Whether LLM JSON response was parsed successfully.
    """
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "unanswerable_count": 0,
            "refusal_count_unanswerable": 0,
            "refusal_rate_unanswerable": 0.0,
            "answerable_count": 0,
            "false_refusal_count": 0,
            "false_refusal_rate": 0.0,
            "citation_validity_count": 0,
            "citation_validity_total": 0,
            "citation_validity_rate": 0.0,
            "citation_grounding_count": 0,
            "citation_grounding_total": 0,
            "citation_grounding_rate": 0.0,
            "parse_failure_count": 0,
            "parse_failure_rate": 0.0,
        }

    parse_failures = sum(1 for r in records if not r.get("parse_ok", True))

    unanswerable = [r for r in records if r.get("expected_answerable") is False]
    refusals = sum(1 for r in unanswerable if not r.get("model_answerable", False))
    refusal_rate = (refusals / len(unanswerable)) if unanswerable else 0.0

    answerable = [r for r in records if r.get("expected_answerable") is True]
    false_refusals = sum(1 for r in answerable if not r.get("model_answerable", False))
    false_refusal_rate = (false_refusals / len(answerable)) if answerable else 0.0

    # Citation validity: evaluated on answered items that produced citations
    answered_with_cits = [r for r in records if r.get("model_answerable") and r.get("citations")]
    valid_cits = sum(1 for r in answered_with_cits if r.get("citations_valid", False))
    validity_rate = (valid_cits / len(answered_with_cits)) if answered_with_cits else 0.0

    # Citation grounding: evaluated on answerable items where model attempted an answer
    answerable_answered = [r for r in records if r.get("expected_answerable") is True and r.get("model_answerable")]
    grounded = sum(1 for r in answerable_answered if r.get("grounded", False))
    grounding_rate = (grounded / len(answerable_answered)) if answerable_answered else 0.0

    return {
        "total": total,
        "unanswerable_count": len(unanswerable),
        "refusal_count_unanswerable": refusals,
        "refusal_rate_unanswerable": refusal_rate,
        "answerable_count": len(answerable),
        "false_refusal_count": false_refusals,
        "false_refusal_rate": false_refusal_rate,
        "citation_validity_count": valid_cits,
        "citation_validity_total": len(answered_with_cits),
        "citation_validity_rate": validity_rate,
        "citation_grounding_count": grounded,
        "citation_grounding_total": len(answerable_answered),
        "citation_grounding_rate": grounding_rate,
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / total,
    }


def compute_answer_eval_metrics(records: Sequence[dict]) -> dict:
    """
    Computes answer evaluation metrics overall and partitioned by retrieval hit vs miss.
    """
    hits = [r for r in records if r.get("retrieval_hit", False)]
    misses = [r for r in records if not r.get("retrieval_hit", False)]
    return {
        "overall": summarize_answer_metrics(records),
        "retrieval_hit": summarize_answer_metrics(hits),
        "retrieval_miss": summarize_answer_metrics(misses),
    }

