"""
Tests for src/verifier.py using stubbed LLM clients.

Verifies:
- Valid verdict mappings for verify_citation and verify_addresses_question
- Parse failure handling (invalid JSON, missing keys, invalid verdicts) - never silently default
- Code fence stripping
- Multi-citation handling
- No network access during tests
"""

import json
import pytest

from src.llm_client import StubLLMClient
from src.verifier import (
    verify_citation,
    verify_addresses_question,
    CitationVerificationResult,
    QuestionAddressingResult,
    VALID_CITATION_VERDICTS,
    VALID_ADDRESSES_VERDICTS,
)


def make_stub(response: str) -> StubLLMClient:
    return StubLLMClient(response=response)


# ---------------------------------------------------------------------------
# Citation Verification Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["supported", "partially_supported", "not_supported"])
def test_verify_citation_valid_verdicts(verdict: str):
    payload = {
        "verdict": verdict,
        "reason": f"This is an explanation for {verdict}."
    }
    llm = make_stub(json.dumps(payload))
    res: CitationVerificationResult = verify_citation(
        question="How many layers?",
        answer="The model has 6 layers.",
        cited_chunk_text="The encoder has N = 6 identical layers.",
        llm=llm,
    )
    assert res.parse_ok is True
    assert res.verdict == verdict
    assert res.reason == f"This is an explanation for {verdict}."


def test_verify_citation_markdown_code_fences():
    payload = {
        "verdict": "supported",
        "reason": "Direct match in passage."
    }
    raw = f"```json\n{json.dumps(payload)}\n```"
    llm = make_stub(raw)
    res = verify_citation("Q", "A", "Passage text", llm)
    assert res.parse_ok is True
    assert res.verdict == "supported"


def test_verify_citation_invalid_json():
    llm = make_stub("{not valid json at all")
    res = verify_citation("Q", "A", "Passage text", llm)
    assert res.parse_ok is False
    assert res.verdict is None
    assert "JSON decoding failed" in res.reason


def test_verify_citation_invalid_verdict_value():
    payload = {"verdict": "completely_true", "reason": "Looks good"}
    llm = make_stub(json.dumps(payload))
    res = verify_citation("Q", "A", "Passage text", llm)
    assert res.parse_ok is False
    assert res.verdict is None
    assert "Invalid verdict" in res.reason


def test_verify_citation_missing_verdict_key():
    payload = {"reason": "Looks good but no verdict"}
    llm = make_stub(json.dumps(payload))
    res = verify_citation("Q", "A", "Passage text", llm)
    assert res.parse_ok is False
    assert res.verdict is None


# ---------------------------------------------------------------------------
# Addresses Question Verification Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["answers_question", "states_not_in_paper", "off_topic_or_wrong_figure"])
def test_verify_addresses_question_valid_verdicts(verdict: str):
    payload = {
        "verdict": verdict,
        "reason": f"Reason for {verdict}."
    }
    llm = make_stub(json.dumps(payload))
    res: QuestionAddressingResult = verify_addresses_question(
        question="What BLEU score?",
        answer="The BLEU score was 28.4.",
        llm=llm,
    )
    assert res.parse_ok is True
    assert res.verdict == verdict
    assert res.reason == f"Reason for {verdict}."


def test_verify_addresses_invalid_json():
    llm = make_stub("not json")
    res = verify_addresses_question("Q", "A", llm)
    assert res.parse_ok is False
    assert res.verdict is None
    assert "JSON decoding failed" in res.reason


def test_verify_addresses_invalid_verdict():
    payload = {"verdict": "irrelevant", "reason": "Not a valid category"}
    llm = make_stub(json.dumps(payload))
    res = verify_addresses_question("Q", "A", llm)
    assert res.parse_ok is False
    assert res.verdict is None
    assert "Invalid verdict" in res.reason


# ---------------------------------------------------------------------------
# Multi-Citation Answer Simulation
# ---------------------------------------------------------------------------

def test_multi_citation_verification_flow():
    """
    Simulates evaluating an answer with 2 cited chunks:
    one is supported, one is partially_supported.
    """
    answers = [
        json.dumps({"verdict": "supported", "reason": "Chunk 1 contains the parameter count."}),
        json.dumps({"verdict": "partially_supported", "reason": "Chunk 2 gives context on training steps."}),
    ]
    call_count = 0

    class SequentialStub(StubLLMClient):
        def generate(self, system: str, user: str) -> str:
            nonlocal call_count
            resp = answers[call_count]
            call_count += 1
            return resp

    seq_llm = SequentialStub("")
    citations = [
        ("attention.pdf::p5::c1", "Text of chunk 1"),
        ("attention.pdf::p7::c6", "Text of chunk 2"),
    ]
    results = []
    for cid, text in citations:
        r = verify_citation(
            question="What are the parameters and training duration?",
            answer="There are 65M parameters and trained for 100k steps.",
            cited_chunk_text=text,
            llm=seq_llm,
        )
        results.append((cid, r))

    assert len(results) == 2
    assert results[0][1].verdict == "supported"
    assert results[1][1].verdict == "partially_supported"
    assert all(r[1].parse_ok for r in results)

