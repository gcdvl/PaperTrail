"""
test_api.py: Tests for PaperTrail FastAPI endpoints.

Verifies:
1. GET /health returns 200 and status ok.
2. POST /documents with valid PDF returns 201 and expected metadata fields.
3. GET /documents lists uploaded documents.
4. GET /documents/{doc_id} returns detail for known doc.
5. GET /documents/{doc_id} returns 404 for unknown id.
6. POST /documents with a .txt file renamed to .pdf returns 400.
7. POST /documents with a blank-page PDF returns 422.
8. Failed uploads leave no lingering files in upload_dir.
9. Saved file name on disk is a UUID hex and not the client's filename.
"""

from pathlib import Path
from fastapi.testclient import TestClient

from src.api import create_app
from src.service import DocumentStore


def test_health(tmp_path: Path):
    """GET /health returns 200 with status ok."""
    store = DocumentStore()
    app = create_app(store, tmp_path / "uploads")
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_valid_pdf(tmp_path: Path, sample_three_page_pdf: Path):
    """POST /documents with a valid PDF returns 201 and doc metadata."""
    store = DocumentStore()
    app = create_app(store, tmp_path / "uploads")
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        response = client.post(
            "/documents",
            files={"file": (sample_three_page_pdf.name, f, "application/pdf")},
        )

    assert response.status_code == 201
    data = response.json()
    assert "doc_id" in data
    assert data["filename"] == sample_three_page_pdf.name
    assert data["total_pages"] == 3
    assert data["chunk_count"] > 0


def test_list_documents(tmp_path: Path, sample_three_page_pdf: Path):
    """GET /documents returns all ingested documents."""
    store = DocumentStore()
    app = create_app(store, tmp_path / "uploads")
    client = TestClient(app)

    # Empty initially
    resp_empty = client.get("/documents")
    assert resp_empty.status_code == 200
    assert resp_empty.json() == []

    # Upload document
    with open(sample_three_page_pdf, "rb") as f:
        resp_upload = client.post(
            "/documents",
            files={"file": ("custom_paper.pdf", f, "application/pdf")},
        )
    assert resp_upload.status_code == 201
    doc_id = resp_upload.json()["doc_id"]

    # List shows uploaded document
    resp_list = client.get("/documents")
    assert resp_list.status_code == 200
    docs = resp_list.json()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == doc_id
    assert docs[0]["filename"] == "custom_paper.pdf"
    assert docs[0]["total_pages"] == 3
    assert docs[0]["chunk_count"] > 0


def test_get_document_detail(tmp_path: Path, sample_three_page_pdf: Path):
    """GET /documents/{doc_id} returns document detail for known id."""
    store = DocumentStore()
    app = create_app(store, tmp_path / "uploads")
    client = TestClient(app)

    with open(sample_three_page_pdf, "rb") as f:
        resp_upload = client.post(
            "/documents",
            files={"file": ("research.pdf", f, "application/pdf")},
        )
    doc_id = resp_upload.json()["doc_id"]

    resp_detail = client.get(f"/documents/{doc_id}")
    assert resp_detail.status_code == 200
    detail = resp_detail.json()
    assert detail["doc_id"] == doc_id
    assert detail["filename"] == "research.pdf"
    assert detail["total_pages"] == 3
    assert detail["chunk_count"] > 0


def test_get_document_unknown_id(tmp_path: Path):
    """GET /documents/{doc_id} returns 404 for unknown id."""
    store = DocumentStore()
    app = create_app(store, tmp_path / "uploads")
    client = TestClient(app)

    response = client.get("/documents/nonexistent_id_abc123")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_upload_txt_file_renamed_to_pdf(tmp_path: Path):
    """Upload of a .txt file renamed to .pdf returns 400."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    app = create_app(store, upload_dir)
    client = TestClient(app)

    fake_bytes = b"Just plain text disguised as a PDF file."
    response = client.post(
        "/documents",
        files={"file": ("fake_document.pdf", fake_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_upload_blank_page_pdf(tmp_path: Path, blank_pdf: Path):
    """Upload of a blank-page PDF returns 422."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    app = create_app(store, upload_dir)
    client = TestClient(app)

    with open(blank_pdf, "rb") as f:
        response = client.post(
            "/documents",
            files={"file": ("blank_document.pdf", f, "application/pdf")},
        )
    assert response.status_code == 422
    assert response.json()["detail"] == "PDF has no extractable text"


def test_failed_upload_cleans_up_file(tmp_path: Path, blank_pdf: Path):
    """After a failed upload the upload dir contains no leftover file."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    app = create_app(store, upload_dir)
    client = TestClient(app)

    # Failed upload: blank PDF (422)
    with open(blank_pdf, "rb") as f:
        resp_blank = client.post(
            "/documents",
            files={"file": ("blank.pdf", f, "application/pdf")},
        )
    assert resp_blank.status_code == 422
    assert list(upload_dir.glob("*.pdf")) == []

    # Failed upload: invalid PDF (400)
    resp_invalid = client.post(
        "/documents",
        files={"file": ("corrupt.pdf", b"garbage not a pdf", "application/pdf")},
    )
    assert resp_invalid.status_code == 400
    assert list(upload_dir.glob("*.pdf")) == []


def test_saved_filename_is_not_client_filename(tmp_path: Path, sample_three_page_pdf: Path):
    """Saved file name on disk is NOT the client's filename."""
    store = DocumentStore()
    upload_dir = tmp_path / "uploads"
    app = create_app(store, upload_dir)
    client = TestClient(app)

    client_filename = "original_user_paper_name.pdf"
    with open(sample_three_page_pdf, "rb") as f:
        response = client.post(
            "/documents",
            files={"file": (client_filename, f, "application/pdf")},
        )
    assert response.status_code == 201

    files_on_disk = list(upload_dir.glob("*.pdf"))
    assert len(files_on_disk) == 1
    disk_file = files_on_disk[0]

    assert disk_file.name != client_filename
    assert not (upload_dir / client_filename).exists()
    assert disk_file.suffix == ".pdf"
    # Stem is 32-character hex uuid4
    assert len(disk_file.stem) == 32
    int(disk_file.stem, 16)

