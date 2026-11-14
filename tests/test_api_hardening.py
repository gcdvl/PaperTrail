"""
test_api_hardening.py: Tests for API security and hardening features.

Verifies:
1. Non-PDF magic bytes return HTTP 400 and write nothing to disk.
2. Path traversal client filename (e.g. '../../evil.pdf') is sanitized and file on disk
   is named <32 hex>.pdf inside upload_dir with nothing written outside.
3. Upload exceeding max_upload_bytes returns HTTP 413 and upload_dir remains empty.
4. Upload of small valid PDF with a large limit returns HTTP 201.
5. LLM raising ConnectionError returns 503 with detail 'LLM unreachable, try again later'
   and hides secret text.
6. LLM raising TimeoutError returns 503 with detail 'LLM unreachable, try again later'
   and hides secret text.
7. LLM raising EnvironmentError returns 503 with detail 'LLM is not configured'
   and hides secret text.
"""

from pathlib import Path
from fastapi.testclient import TestClient

from src.api import create_app
from src.llm_client import LLMClient
from src.service import DocumentStore


def test_non_pdf_bytes_rejected_empty_dir(tmp_path: Path):
    """File with non-PDF bytes named .pdf returns 400, and upload dir is empty."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    app = create_app(store, upload_dir)
    client = TestClient(app)

    resp = client.post(
        "/documents",
        files={"file": ("non_pdf.pdf", b"NOT-A-PDF-CONTENT", "application/pdf")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "File is not a PDF"
    assert list(upload_dir.iterdir()) == []


def test_path_traversal_filename(tmp_path: Path, sample_three_page_pdf: Path):
    """Client filename with path traversal is not used on disk; saved as 32 hex inside upload_dir."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    app = create_app(store, upload_dir)
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        resp = client.post(
            "/documents",
            files={"file": ("../../evil.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 201

    files_in_upload_dir = list(upload_dir.iterdir())
    assert len(files_in_upload_dir) == 1
    disk_file = files_in_upload_dir[0]

    assert disk_file.name != "../../evil.pdf"
    assert disk_file.name != "evil.pdf"
    assert len(disk_file.stem) == 32
    int(disk_file.stem, 16)
    assert disk_file.suffix == ".pdf"

    # Verify no file written outside upload_dir
    assert not (tmp_path / "evil.pdf").exists()
    assert not (tmp_path.parent / "evil.pdf").exists()


def test_upload_exceeds_max_bytes(tmp_path: Path, sample_three_page_pdf: Path):
    """Upload larger than max_upload_bytes returns 413 and upload dir is empty."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    assert sample_three_page_pdf.stat().st_size > 1000

    app = create_app(store, upload_dir, max_upload_bytes=1000)
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        resp = client.post(
            "/documents",
            files={"file": ("paper.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "File too large"
    assert list(upload_dir.iterdir()) == []


def test_upload_small_valid_pdf_with_large_limit(tmp_path: Path, sample_three_page_pdf: Path):
    """Upload of small valid PDF with a large limit still returns 201."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    app = create_app(store, upload_dir, max_upload_bytes=25 * 1024 * 1024)
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        resp = client.post(
            "/documents",
            files={"file": ("paper.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 201
    assert len(list(upload_dir.iterdir())) == 1


def test_llm_connection_error(tmp_path: Path, sample_three_page_pdf: Path):
    """LLM raising ConnectionError returns 503 LLM unreachable and hides secret text."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"

    class ConnClient(LLMClient):
        def generate(self, system: str, user: str) -> str:
            raise ConnectionError("SECRET-9-connection-broken")

    app = create_app(store, upload_dir, llm_factory=ConnClient)
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        upload_resp = client.post("/documents", files={"file": ("paper.pdf", f, "application/pdf")})
    doc_id = upload_resp.json()["doc_id"]

    resp = client.post(f"/documents/{doc_id}/ask", json={"question": "difficult to train"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "LLM unreachable, try again later"
    assert "SECRET-9" not in resp.text


def test_llm_timeout_error(tmp_path: Path, sample_three_page_pdf: Path):
    """LLM raising TimeoutError returns 503 LLM unreachable and hides secret text."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"

    class TimeoutClient(LLMClient):
        def generate(self, system: str, user: str) -> str:
            raise TimeoutError("SECRET-9-timed-out")

    app = create_app(store, upload_dir, llm_factory=TimeoutClient)
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        upload_resp = client.post("/documents", files={"file": ("paper.pdf", f, "application/pdf")})
    doc_id = upload_resp.json()["doc_id"]

    resp = client.post(f"/documents/{doc_id}/ask", json={"question": "difficult to train"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "LLM unreachable, try again later"
    assert "SECRET-9" not in resp.text


def test_llm_environment_error(tmp_path: Path, sample_three_page_pdf: Path):
    """LLM raising EnvironmentError returns 503 LLM is not configured and hides secret text."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"

    class EnvClient(LLMClient):
        def generate(self, system: str, user: str) -> str:
            raise EnvironmentError("SECRET-9-missing-env")

    app = create_app(store, upload_dir, llm_factory=EnvClient)
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        upload_resp = client.post("/documents", files={"file": ("paper.pdf", f, "application/pdf")})
    doc_id = upload_resp.json()["doc_id"]

    resp = client.post(f"/documents/{doc_id}/ask", json={"question": "difficult to train"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "LLM is not configured"
    assert "SECRET-9" not in resp.text

