"""
test_service.py: Unit and integration tests for service.py layer.

Verifies:
1. ingest_pdf returns a record with correct total_pages, >0 chunks, and store.get works.
2. Ingest of a PDF with no text raises NoTextError.
3. Ingest of a non-PDF file raises a PDFExtractionError subclass.
4. ask answered: stub returns JSON citing a real passage_id; citation page and text match that chunk.
5. ask not_found: stub returns answerable=False.
6. ask error: stub returns invalid JSON string; status is error, never not_found.
7. ask with non-existent citation ID does not return status answered.
8. ask propagates LLM exceptions.
9. 'retrieved' field is present in all three statuses (answered, not_found, error).
"""

import json
from pathlib import Path
import pytest

from src.extractor import PDFExtractionError
from src.llm_client import LLMClient, StubLLMClient
from src.service import (
    DocumentRecord,
    DocumentStore,
    NoTextError,
    ask,
    ingest_pdf,
)


def test_ingest_pdf_success(sample_three_page_pdf: Path):
    """ingest_pdf returns a record with the right total_pages and >0 chunks; store.get works."""
    store = DocumentStore()
    record = ingest_pdf(sample_three_page_pdf, store)

    assert isinstance(record, DocumentRecord)
    assert record.total_pages == 3
    assert len(record.chunks) > 0
    assert isinstance(record.chunks, tuple)
    assert record.filename == sample_three_page_pdf.name
    assert store.get(record.doc_id) is record
    assert record in store.list()

    # Also test original_filename override
    custom_record = ingest_pdf(sample_three_page_pdf, store, original_filename="my_paper.pdf")
    assert custom_record.filename == "my_paper.pdf"
    assert store.get(custom_record.doc_id) is custom_record


def test_ingest_pdf_no_text(blank_pdf: Path):
    """ingest of a PDF with no text raises NoTextError (blank-page PDF)."""
    store = DocumentStore()
    with pytest.raises(NoTextError):
        ingest_pdf(blank_pdf, store)
    assert len(store.list()) == 0


def test_ingest_pdf_non_pdf(non_pdf_file: Path):
    """ingest of a non-PDF file raises a PDFExtractionError subclass."""
    store = DocumentStore()
    with pytest.raises(PDFExtractionError):
        ingest_pdf(non_pdf_file, store)
    assert len(store.list()) == 0


def test_ask_answered(sample_three_page_pdf: Path):
    """
    ask answered: stub returns JSON citing a real passage_id (build the stub AFTER ingest,
    use record.chunks[0].passage_id); citation page and text match that chunk.
    """
    store = DocumentStore()
    record = ingest_pdf(sample_three_page_pdf, store)
    chunk0 = record.chunks[0]

    llm_payload = json.dumps({
        "answerable": True,
        "answer": "Deep convolutional neural networks led to breakthroughs.",
        "citations": [chunk0.passage_id],
    })
    stub = StubLLMClient(llm_payload)

    query = "breakthroughs in convolutional neural networks"
    res = ask(record, query, stub, top_k=5)

    assert res["status"] == "answered"
    assert res["answer"] == "Deep convolutional neural networks led to breakthroughs."
    assert "citations" in res
    assert len(res["citations"]) == 1
    citation = res["citations"][0]
    assert citation["id"] == chunk0.passage_id
    assert citation["page"] == chunk0.page_number
    assert citation["text"] == chunk0.text
    assert "retrieved" in res
    assert len(res["retrieved"]) > 0


def test_ask_not_found(sample_three_page_pdf: Path):
    """ask not_found: stub returns {"answerable": false, "answer": "", "citations": []}."""
    store = DocumentStore()
    record = ingest_pdf(sample_three_page_pdf, store)

    llm_payload = json.dumps({
        "answerable": False,
        "answer": "",
        "citations": [],
    })
    stub = StubLLMClient(llm_payload)

    res = ask(record, "breakthroughs in convolutional neural networks", stub, top_k=5)

    assert res["status"] == "not_found"
    assert "retrieved" in res
    assert "answer" not in res
    assert "citations" not in res
    assert len(res.keys()) == 2  # exactly status and retrieved


def test_ask_error_unparseable(sample_three_page_pdf: Path):
    """ask error: stub returns 'not json at all' -> status == 'error', never 'not_found'."""
    store = DocumentStore()
    record = ingest_pdf(sample_three_page_pdf, store)

    stub = StubLLMClient("not json at all")

    res = ask(record, "breakthroughs in convolutional neural networks", stub, top_k=5)

    assert res["status"] == "error"
    assert res["status"] != "not_found"
    assert res["reason"] == "unparseable_model_output"
    assert "retrieved" in res
    assert len(res.keys()) == 3  # exactly status, reason, retrieved


def test_ask_citation_does_not_exist(sample_three_page_pdf: Path):
    """ask with citation ID that does not exist -> status is not 'answered'."""
    store = DocumentStore()
    record = ingest_pdf(sample_three_page_pdf, store)

    llm_payload = json.dumps({
        "answerable": True,
        "answer": "Hallucinated statement",
        "citations": ["nonexistent_doc::p99::c99"],
    })
    stub = StubLLMClient(llm_payload)

    res = ask(record, "breakthroughs in convolutional neural networks", stub, top_k=5)

    assert res["status"] != "answered"
    assert res["status"] == "not_found"
    assert "retrieved" in res


def test_ask_propagates_llm_exception(sample_three_page_pdf: Path):
    """ask propagates an LLM exception (a small LLMClient subclass whose generate raises RuntimeError)."""
    store = DocumentStore()
    record = ingest_pdf(sample_three_page_pdf, store)

    class FailingLLMClient(LLMClient):
        def generate(self, system: str, user: str) -> str:
            raise RuntimeError("Underlying LLM API rate limit exceeded")

    with pytest.raises(RuntimeError, match="Underlying LLM API rate limit exceeded"):
        ask(record, "breakthroughs in convolutional neural networks", FailingLLMClient(), top_k=5)


def test_retrieved_present_in_all_three_statuses(sample_three_page_pdf: Path):
    """'retrieved' is present in all three statuses."""
    store = DocumentStore()
    record = ingest_pdf(sample_three_page_pdf, store)
    chunk0 = record.chunks[0]

    # Status: answered
    stub_answered = StubLLMClient(json.dumps({
        "answerable": True,
        "answer": "Test answer",
        "citations": [chunk0.passage_id],
    }))
    res_answered = ask(record, "breakthroughs in convolutional neural networks", stub_answered)
    assert res_answered["status"] == "answered"
    assert "retrieved" in res_answered
    assert len(res_answered["retrieved"]) > 0
    for r in res_answered["retrieved"]:
        assert set(r.keys()) == {"id", "page", "score", "text"}

    # Status: not_found
    stub_not_found = StubLLMClient(json.dumps({
        "answerable": False,
        "answer": "",
        "citations": [],
    }))
    res_not_found = ask(record, "breakthroughs in convolutional neural networks", stub_not_found)
    assert res_not_found["status"] == "not_found"
    assert "retrieved" in res_not_found
    assert len(res_not_found["retrieved"]) > 0
    for r in res_not_found["retrieved"]:
        assert set(r.keys()) == {"id", "page", "score", "text"}

    # Status: error
    stub_error = StubLLMClient("<<<bad json>>>")
    res_error = ask(record, "breakthroughs in convolutional neural networks", stub_error)
    assert res_error["status"] == "error"
    assert "retrieved" in res_error
    assert len(res_error["retrieved"]) > 0
    for r in res_error["retrieved"]:
        assert set(r.keys()) == {"id", "page", "score", "text"}

