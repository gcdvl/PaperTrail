"""
api.py: FastAPI REST application for PaperTrail.

Provides HTTP endpoints for health checking, document upload, document inspection,
and question-answering grounded in paper passages.
"""

import gc
import logging
from pathlib import Path
import shutil
import threading
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from src.extractor import PDFExtractionError
from src.service import DocumentStore, NoTextError, ask, ingest_pdf

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AskRequest(BaseModel):
    """Request payload for querying an ingested document."""
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty or whitespace-only")
        if len(v) > 1000:
            raise ValueError("question cannot exceed 1000 characters")
        return v


def create_app(
    store: DocumentStore,
    upload_dir: Path,
    llm_factory: Optional[Any] = None,
    max_upload_bytes: int = 25 * 1024 * 1024,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    """
    Factory function to construct a configured FastAPI application.

    Args:
        store: DocumentStore instance for managing documents.
        upload_dir: Directory where uploaded PDFs are stored on disk.
        llm_factory: Optional callable/factory for instantiating LLM clients.
        max_upload_bytes: Maximum allowed size for uploaded files in bytes.
        static_dir: Directory containing static files including index.html.

    Returns:
        Configured FastAPI instance.
    """
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    static_path = Path(static_dir) if static_dir is not None else (PROJECT_ROOT / "static")

    app = FastAPI(title="PaperTrail API")
    app.state.store = store
    app.state.upload_dir = upload_path
    app.state.static_dir = static_path
    app.state.llm_factory = llm_factory
    app.state.max_upload_bytes = max_upload_bytes
    app.state.llm_instance = None
    app.state.llm_lock = threading.Lock()

    def get_llm():
        if app.state.llm_factory is not None:
            return app.state.llm_factory()
        if app.state.llm_instance is None:
            with app.state.llm_lock:
                if app.state.llm_instance is None:
                    from src.llm_client import GeminiLLMClient
                    app.state.llm_instance = GeminiLLMClient(delay_between_calls=0.0)
        return app.state.llm_instance

    app.state.get_llm = get_llm

    @app.get("/")
    def get_index():
        index_file = static_path / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return FileResponse(index_file, media_type="text/html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/documents", status_code=status.HTTP_201_CREATED)
    def upload_document(file: UploadFile = File(...)):
        # a. Read first 5 bytes. If not b"%PDF-", return 400 and write nothing to disk.
        header = file.file.read(5)
        if header != b"%PDF-":
            raise HTTPException(status_code=400, detail="File is not a PDF")

        saved_name = f"{uuid4().hex}.pdf"
        saved_path = upload_path / saved_name
        chunk_size = 1024 * 1024  # 1 MB
        total_bytes = len(header)

        if total_bytes > max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large")

        # b. Stream the upload to upload_dir/<uuid4 hex>.pdf in 1 MB pieces, counting bytes.
        too_large = False
        try:
            with open(saved_path, "wb") as buffer:
                buffer.write(header)
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_upload_bytes:
                        too_large = True
                        break
                    buffer.write(chunk)
        except Exception as exc:
            gc.collect()
            if saved_path.exists():
                saved_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

        if too_large:
            gc.collect()
            if saved_path.exists():
                saved_path.unlink()
            raise HTTPException(status_code=413, detail="File too large")

        # c. Ingestion and error handling
        exc_status = None
        exc_detail = None
        record = None

        try:
            record = ingest_pdf(saved_path, store, original_filename=file.filename)
        except PDFExtractionError as exc:
            exc_status = 400
            exc_detail = str(exc)
        except NoTextError:
            exc_status = 422
            exc_detail = "PDF has no extractable text"
        except Exception as exc:
            exc_status = 500
            exc_detail = str(exc)

        if exc_status is not None:
            gc.collect()
            if saved_path.exists():
                try:
                    saved_path.unlink()
                except Exception:
                    gc.collect()
                    saved_path.unlink(missing_ok=True)
            raise HTTPException(status_code=exc_status, detail=exc_detail)

        return {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "total_pages": record.total_pages,
            "chunk_count": len(record.chunks),
        }

    @app.get("/documents")
    def list_documents():
        records = store.list()
        return [
            {
                "doc_id": r.doc_id,
                "filename": r.filename,
                "total_pages": r.total_pages,
                "chunk_count": len(r.chunks),
            }
            for r in records
        ]

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str):
        record = store.get(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "total_pages": record.total_pages,
            "chunk_count": len(record.chunks),
        }

    @app.post("/documents/{doc_id}/ask")
    def ask_document(doc_id: str, request: AskRequest):
        record = store.get(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Document not found")

        try:
            llm = app.state.get_llm()
            return ask(record, request.question, llm, top_k=5)
        except (ConnectionError, TimeoutError):
            logger.exception("LLM unreachable")
            raise HTTPException(
                status_code=503,
                detail="LLM unreachable, try again later",
            )
        except EnvironmentError:
            logger.exception("LLM configuration error")
            raise HTTPException(status_code=503, detail="LLM is not configured")
        except RuntimeError:
            logger.exception("LLM unavailable or rate limited")
            raise HTTPException(
                status_code=503,
                detail="LLM unavailable or rate limited, try again later",
            )
        except Exception:
            logger.exception("LLM request failed")
            raise HTTPException(status_code=502, detail="LLM request failed")

    return app


app = create_app(DocumentStore(), PROJECT_ROOT / "uploads")
