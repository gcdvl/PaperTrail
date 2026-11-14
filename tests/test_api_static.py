"""
test_api_static.py: Tests for serving static frontend files.

Verifies:
1. GET / returns 200 with content-type text/html and body matching index.html.
2. GET / returns 404 when index.html does not exist in static_dir.
3. GET /health still returns 200 with status 'ok'.
"""

from pathlib import Path
from fastapi.testclient import TestClient

from src.api import create_app
from src.service import DocumentStore


def test_index_serves_html(tmp_path: Path):
    """GET / serves index.html with 200 and text/html content-type."""
    store = DocumentStore()
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    index_file = static_dir / "index.html"
    sample_content = "<!DOCTYPE html><html><body><h1>PaperTrail</h1></body></html>"
    index_file.write_text(sample_content, encoding="utf-8")

    app = create_app(store, tmp_path / "uploads", static_dir=static_dir)
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert resp.text == sample_content


def test_index_missing_returns_404(tmp_path: Path):
    """GET / returns 404 with detail when index.html is missing."""
    store = DocumentStore()
    static_dir = tmp_path / "empty_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(store, tmp_path / "uploads", static_dir=static_dir)
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Frontend not found"}


def test_health_still_returns_200(tmp_path: Path):
    """GET /health continues to return 200 ok."""
    store = DocumentStore()
    app = create_app(store, tmp_path / "uploads")
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

