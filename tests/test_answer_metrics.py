import json
import pytest

from src.answerer import answer_question, _strip_code_fences
from src.llm_client import StubLLMClient
from src.metrics import (
    compute_answer_eval_metrics,
    summarize_answer_metrics,
    passage_contains_evidence,
)


def _eval_single_item(item: dict, passages: list, llm: StubLLMClient) -> dict:
    """Helper that runs answer_question and produces an evaluation record."""
    passage_dicts = [{"id": p["id"], "text": p["text"]} for p in passages]
    res = answer_question(item["question"], passage_dicts, llm)

    passage_map = {p["id"]: p["text"] for p in passages}
    retrieved_ids = set(passage_map.keys())
    ev = item.get("evidence_phrase")
    expected_ans = item.get("answerable")

    # Retrieval hit: does any retrieved passage contain evidence phrase?
    retrieval_hit = (
        any(passage_contains_evidence(p["text"], ev) for p in passages)
        if (ev and expected_ans is not False)
        else False
    )

    # Raw citations validity
    raw_citations = []
    raw_answerable = res.answerable
    if res.parse_ok and res.raw_response:
        try:
            raw_data = json.loads(_strip_code_fences(res.raw_response))
            raw_citations = list(raw_data.get("citations") or [])
            raw_answerable = bool(raw_data.get("answerable", False))
        except Exception:
            raw_citations = list(res.citations)
    else:
        raw_citations = list(res.citations)

    if raw_citations:
        citations_valid = all(cid in retrieved_ids for cid in raw_citations)
    elif raw_answerable:
        citations_valid = False
    else:
        citations_valid = True

    # Grounding check
    if expected_ans is True and res.answerable and res.citations:
        grounded = (
            any(passage_contains_evidence(passage_map[cid], ev)
                for cid in res.citations if cid in passage_map)
            if ev else False
        )
    else:
        grounded = False

    return {
        "id": item.get("id", "test-item"),
        "expected_answerable": expected_ans,
        "model_answerable": res.answerable,
        "evidence_phrase": ev,
        "citations": res.citations,
        "raw_citations": raw_citations,
        "retrieval_hit": retrieval_hit,
        "grounded": grounded,
        "citations_valid": citations_valid,
        "parse_ok": res.parse_ok,
    }


def test_unanswerable_refusal_metric():
    """Ensure refusal rate on unanswerable items is computed accurately."""
    item = {
        "id": "unans-01",
        "question": "What is the secret key?",
        "answerable": False,
        "evidence_phrase": None,
    }
    passages = [{"id": "doc::p1::c1", "text": "This paper discusses deep learning."}]

    # Model correctly refuses
    llm = StubLLMClient(json.dumps({"answerable": False, "answer": "", "citations": []}))
    rec = _eval_single_item(item, passages, llm)
    metrics = summarize_answer_metrics([rec])

    assert metrics["unanswerable_count"] == 1
    assert metrics["refusal_count_unanswerable"] == 1
    assert metrics["refusal_rate_unanswerable"] == 1.0


def test_wrongly_answered_unanswerable_metric():
    """Ensure wrongly answered unanswerable items are tracked."""
    item = {
        "id": "unans-02",
        "question": "What is the secret key?",
        "answerable": False,
        "evidence_phrase": None,
    }
    passages = [{"id": "doc::p1::c1", "text": "This paper discusses deep learning."}]

    # Model incorrectly answers an unanswerable item
    llm = StubLLMClient(json.dumps({
        "answerable": True,
        "answer": "The secret key is 12345.",
        "citations": ["doc::p1::c1"],
    }))
    rec = _eval_single_item(item, passages, llm)
    metrics = summarize_answer_metrics([rec])

    assert metrics["unanswerable_count"] == 1
    assert metrics["refusal_count_unanswerable"] == 0
    assert metrics["refusal_rate_unanswerable"] == 0.0


def test_false_refusal_metric():
    """Ensure false-refusal rate on answerable items is computed accurately."""
    item = {
        "id": "ans-01",
        "question": "What architecture is used?",
        "answerable": True,
        "evidence_phrase": "Transformer architecture",
    }
    passages = [{"id": "doc::p1::c1", "text": "We propose the Transformer architecture."}]

    # Model refuses an answerable item
    llm = StubLLMClient(json.dumps({"answerable": False, "answer": "", "citations": []}))
    rec = _eval_single_item(item, passages, llm)
    metrics = summarize_answer_metrics([rec])

    assert metrics["answerable_count"] == 1
    assert metrics["false_refusal_count"] == 1
    assert metrics["false_refusal_rate"] == 1.0


def test_retrieval_hit_vs_miss_split():
    """Ensure summary metrics split cleanly between retrieval hit and miss."""
    item_hit = {
        "id": "item-hit",
        "question": "What is the learning rate?",
        "answerable": True,
        "evidence_phrase": "learning rate of 0.001",
    }
    passages_hit = [{"id": "doc::p1::c1", "text": "We used a learning rate of 0.001 with Adam."}]
    llm_hit = StubLLMClient(json.dumps({
        "answerable": True,
        "answer": "The learning rate is 0.001.",
        "citations": ["doc::p1::c1"],
    }))
    rec_hit = _eval_single_item(item_hit, passages_hit, llm_hit)

    item_miss = {
        "id": "item-miss",
        "question": "What is the batch size?",
        "answerable": True,
        "evidence_phrase": "batch size was 64",
    }
    # Passages do NOT contain the evidence phrase
    passages_miss = [{"id": "doc::p2::c1", "text": "We evaluated on ImageNet validation set."}]
    llm_miss = StubLLMClient(json.dumps({
        "answerable": True,
        "answer": "Batch size was 64.",
        "citations": ["doc::p2::c1"],
    }))
    rec_miss = _eval_single_item(item_miss, passages_miss, llm_miss)

    split_metrics = compute_answer_eval_metrics([rec_hit, rec_miss])

    # Overall: 2 items, 1 grounded out of 2 answered
    assert split_metrics["overall"]["total"] == 2
    assert split_metrics["overall"]["citation_grounding_count"] == 1
    assert split_metrics["overall"]["citation_grounding_total"] == 2
    assert split_metrics["overall"]["citation_grounding_rate"] == 0.5

    # Retrieval Hit: 1 item, 100% grounded, 0 false refusals
    assert split_metrics["retrieval_hit"]["total"] == 1
    assert split_metrics["retrieval_hit"]["citation_grounding_count"] == 1
    assert split_metrics["retrieval_hit"]["citation_grounding_rate"] == 1.0
    assert split_metrics["retrieval_hit"]["false_refusal_count"] == 0

    # Retrieval Miss: 1 item, 0% grounded (evidence was not in retrieved set)
    assert split_metrics["retrieval_miss"]["total"] == 1
    assert split_metrics["retrieval_miss"]["citation_grounding_count"] == 0
    assert split_metrics["retrieval_miss"]["citation_grounding_rate"] == 0.0


def test_citation_validity_metric():
    """Ensure invalid citations (IDs not in retrieved pool) are flagged."""
    item = {
        "id": "cit-01",
        "question": "What optimizer was used?",
        "answerable": True,
        "evidence_phrase": "Adam optimizer",
    }
    passages = [{"id": "doc::p1::c1", "text": "We used Adam optimizer."}]

    # Model hallucinates a non-existent citation ID
    llm = StubLLMClient(json.dumps({
        "answerable": True,
        "answer": "Adam.",
        "citations": ["doc::p99::c99"],
    }))
    rec = _eval_single_item(item, passages, llm)
    metrics = summarize_answer_metrics([rec])

    assert rec["citations_valid"] is False
    assert metrics["citation_validity_count"] == 0


def test_parse_failure_metric():
    """Ensure malformed JSON outputs are recorded as parse failures."""
    item = {
        "id": "parse-01",
        "question": "What is the answer?",
        "answerable": True,
        "evidence_phrase": "answer",
    }
    passages = [{"id": "doc::p1::c1", "text": "Here is the answer text."}]

    # Model outputs plain text instead of JSON
    llm = StubLLMClient("This is not valid JSON at all")
    rec = _eval_single_item(item, passages, llm)
    metrics = summarize_answer_metrics([rec])

    assert rec["parse_ok"] is False
    assert metrics["parse_failure_count"] == 1
    assert metrics["parse_failure_rate"] == 1.0
