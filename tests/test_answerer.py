import json
import pytest
from src.answerer import answer_question, AnswerResult
from src.llm_client import StubLLMClient

# Helper to create a stub LLM client with given raw response string

def make_stub(response: str) -> StubLLMClient:
    return StubLLMClient(response=response)


def test_answer_question_success():
    raw = json.dumps({
        "answerable": True,
        "answer": "Transformer architecture enables parallel processing.",
        "citations": ["doc.pdf::p1::c1", "doc.pdf::p2::c2"]
    })
    llm = make_stub(raw)
    passages = [
        {"id": "doc.pdf::p1::c1", "text": "..."},
        {"id": "doc.pdf::p2::c2", "text": "..."},
    ]
    result: AnswerResult = answer_question("What does Transformer do?", passages, llm)
    assert result.parse_ok
    assert result.answerable
    assert result.answer == "Transformer architecture enables parallel processing."
    assert set(result.citations) == {"doc.pdf::p1::c1", "doc.pdf::p2::c2"}
    assert result.raw_response == raw


def test_answer_question_refuse():
    raw = json.dumps({"answerable": False, "answer": "", "citations": []})
    llm = make_stub(raw)
    passages = [{"id": "doc.pdf::p1::c1", "text": "..."}]
    result = answer_question("When was the moon made?", passages, llm)
    assert result.parse_ok
    assert not result.answerable
    assert result.answer == ""
    assert result.citations == []


def test_answer_question_code_fence():
    raw = "```json\n" + json.dumps({"answerable": True, "answer": "Yes", "citations": ["doc.pdf::p1::c1"]}) + "\n```"
    llm = make_stub(raw)
    passages = [{"id": "doc.pdf::p1::c1", "text": "..."}]
    result = answer_question("Is this valid?", passages, llm)
    assert result.parse_ok
    assert result.answerable
    assert result.answer == "Yes"
    assert result.citations == ["doc.pdf::p1::c1"]


def test_answer_question_invalid_json():
    raw = "{invalid json"
    llm = make_stub(raw)
    passages = [{"id": "doc.pdf::p1::c1", "text": "..."}]
    result = answer_question("Anything?", passages, llm)
    assert not result.parse_ok
    assert not result.answerable
    assert result.citations == []
    assert result.raw_response == raw


def test_citation_filtering_invalid_ids():
    raw = json.dumps({
        "answerable": True,
        "answer": "Answer with mixed citations",
        "citations": ["doc.pdf::p1::c1", "invalid::id"]
    })
    llm = make_stub(raw)
    passages = [{"id": "doc.pdf::p1::c1", "text": "..."}]
    result = answer_question("Test filtering", passages, llm)
    assert result.parse_ok
    assert result.answerable
    assert result.citations == ["doc.pdf::p1::c1"]
    raw2 = json.dumps({"answerable": True, "answer": "Answer", "citations": ["invalid::id"]})
    llm2 = make_stub(raw2)
    result2 = answer_question("Test filtering", passages, llm2)
    assert not result2.answerable
    assert result2.citations == []
