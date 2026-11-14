"""
test_api_ask.py: Tests for POST /documents/{doc_id}/ask endpoint.

Verifies:
1. answered: HTTP 200, status 'answered', citation has id, page, text matching that chunk.
2. not_found: HTTP 200, status 'not_found'.
3. stub returns 'not json': HTTP 200, status 'error', reason 'unparseable_model_output'.
4. unknown doc_id: HTTP 404.
5. empty question, whitespace-only question, missing question field, 1001-char question: all HTTP 422.
6. llm raises RuntimeError: HTTP 503; raises EnvironmentError: HTTP 503; raises ValueError: HTTP 502.
7. llm_factory itself raises EnvironmentError: HTTP 503.
8. 503/502 response body does not contain the original exception message (e.g. 'SECRET-123').
9. Response of answered contains exactly the keys: status, answer, citations, retrieved.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.llm_client import LLMClient, StubLLMClient
from src.service import DocumentStore


def _upload_pdf(client: TestClient, pdf_path: Path) -> str:
    """Helper to upload a PDF and return its doc_id."""
    with open(pdf_path, "rb") as f:
        resp = client.post("/documents", files={"file": (pdf_path.name, f, "application/pdf")})
    assert resp.status_code == 201
    return resp.json()["doc_id"]


def test_ask_answered(tmp_path: Path, sample_three_page_pdf: Path):
    """Answered query returns 200, status 'answered', valid citation and exact keys."""
    store = DocumentStore()
    stub_holder = []
    app = create_app(store, tmp_path / "uploads", llm_factory=lambda: stub_holder[0])
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)
    target_chunk = store.get(doc_id).chunks[0]

    payload = json.dumps({
        "answerable": True,
        "answer": "Deeper neural networks are more difficult to train.",
        "citations": [target_chunk.passage_id],
    })
    stub_holder.append(StubLLMClient(payload))

    resp = client.post(
        f"/documents/{doc_id}/ask",
        json={"question": "difficult to train neural networks"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "answered"
    assert data["answer"] == "Deeper neural networks are more difficult to train."
    assert len(data["citations"]) == 1
    cit = data["citations"][0]
    assert cit["id"] == target_chunk.passage_id
    assert cit["page"] == target_chunk.page_number
    assert cit["text"] == target_chunk.text

    # Response of answered contains exactly the keys: status, answer, citations, retrieved
    assert set(data.keys()) == {"status", "answer", "citations", "retrieved"}


def test_ask_not_found(tmp_path: Path, sample_three_page_pdf: Path):
    """Unanswerable query returns 200 and status 'not_found'."""
    store = DocumentStore()
    stub = StubLLMClient(json.dumps({"answerable": False, "answer": "", "citations": []}))
    app = create_app(store, tmp_path / "uploads", llm_factory=lambda: stub)
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)

    resp = client.post(
        f"/documents/{doc_id}/ask",
        json={"question": "difficult to train neural networks"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_found"
    assert "retrieved" in data


def test_ask_unparseable_stub(tmp_path: Path, sample_three_page_pdf: Path):
    """Stub returning invalid JSON yields HTTP 200 with status 'error' and reason 'unparseable_model_output'."""
    store = DocumentStore()
    stub = StubLLMClient("not json")
    app = create_app(store, tmp_path / "uploads", llm_factory=lambda: stub)
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)

    resp = client.post(
        f"/documents/{doc_id}/ask",
        json={"question": "difficult to train neural networks"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["reason"] == "unparseable_model_output"
    assert data["status"] != "not_found"
    assert "retrieved" in data


def test_ask_unknown_doc_id(tmp_path: Path):
    """Querying an unknown doc_id returns 404."""
    store = DocumentStore()
    stub = StubLLMClient("unused")
    app = create_app(store, tmp_path / "uploads", llm_factory=lambda: stub)
    client = TestClient(app)

    resp = client.post(
        "/documents/nonexistent-doc-id-999/ask",
        json={"question": "What is deep learning?"},
    )
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_ask_question_validation(tmp_path: Path, sample_three_page_pdf: Path):
    """Empty, whitespace-only, missing, or >1000 char question fields all return HTTP 422."""
    store = DocumentStore()
    stub = StubLLMClient("unused")
    app = create_app(store, tmp_path / "uploads", llm_factory=lambda: stub)
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)

    # Empty question
    resp_empty = client.post(f"/documents/{doc_id}/ask", json={"question": ""})
    assert resp_empty.status_code == 422

    # Whitespace-only question
    resp_ws = client.post(f"/documents/{doc_id}/ask", json={"question": "   \n\t  "})
    assert resp_ws.status_code == 422

    # Missing question field
    resp_missing = client.post(f"/documents/{doc_id}/ask", json={})
    assert resp_missing.status_code == 422

    # 1001-character question
    resp_long = client.post(f"/documents/{doc_id}/ask", json={"question": "a" * 1001})
    assert resp_long.status_code == 422


def test_ask_llm_runtime_error(tmp_path: Path, sample_three_page_pdf: Path):
    """RuntimeError in LLM raises HTTP 503 and hides secret exception text."""
    store = DocumentStore()

    class FailingClient(LLMClient):
        def generate(self, system: str, user: str) -> str:
            raise RuntimeError("SECRET-123-runtime-error-occurred")

    app = create_app(store, tmp_path / "uploads", llm_factory=FailingClient)
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)

    resp = client.post(f"/documents/{doc_id}/ask", json={"question": "difficult to train"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "LLM unavailable or rate limited, try again later"
    assert "SECRET-123" not in resp.text


def test_ask_llm_environment_error(tmp_path: Path, sample_three_page_pdf: Path):
    """EnvironmentError in LLM raises HTTP 503 and hides secret exception text."""
    store = DocumentStore()

    class MissingEnvClient(LLMClient):
        def generate(self, system: str, user: str) -> str:
            raise EnvironmentError("SECRET-123-missing-gemini-key")

    app = create_app(store, tmp_path / "uploads", llm_factory=MissingEnvClient)
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)

    resp = client.post(f"/documents/{doc_id}/ask", json={"question": "difficult to train"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "LLM is not configured"
    assert "SECRET-123" not in resp.text


def test_ask_llm_value_error(tmp_path: Path, sample_three_page_pdf: Path):
    """Generic/ValueError in LLM raises HTTP 502 and hides secret exception text."""
    store = DocumentStore()

    class BadValueClient(LLMClient):
        def generate(self, system: str, user: str) -> str:
            raise ValueError("SECRET-123-invalid-internal-value")

    app = create_app(store, tmp_path / "uploads", llm_factory=BadValueClient)
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)

    resp = client.post(f"/documents/{doc_id}/ask", json={"question": "difficult to train"})
    assert resp.status_code == 502
    assert resp.json()["detail"] == "LLM request failed"
    assert "SECRET-123" not in resp.text


def test_ask_llm_factory_raises_environment_error(tmp_path: Path, sample_three_page_pdf: Path):
    """When llm_factory itself raises EnvironmentError, returns HTTP 503 without secret text."""
    store = DocumentStore()

    def bad_factory():
        raise EnvironmentError("SECRET-123-factory-env-error")

    app = create_app(store, tmp_path / "uploads", llm_factory=bad_factory)
    client = TestClient(app)

    doc_id = _upload_pdf(client, sample_three_page_pdf)

    resp = client.post(f"/documents/{doc_id}/ask", json={"question": "difficult to train"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "LLM is not configured"
    assert "SECRET-123" not in resp.text

