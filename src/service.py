"""
service.py: Core application service layer for PaperTrail.

Provides document ingestion, thread-safe storage, and question-answering
orchestration without dependencies on web frameworks or Pydantic.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from src.answerer import answer_question
from src.chunker import Chunk, chunk_document
from src.extractor import extract_text_from_pdf
from src.retriever import retrieve


class NoTextError(Exception):
    """Raised when the PDF has no extractable text or chunking yields 0 chunks."""
    pass


@dataclass(frozen=True)
class DocumentRecord:
    """
    Persistent record representing an ingested PDF document and its chunks.

    Attributes:
        doc_id: Unique identifier for the document record.
        filename: User-facing filename of the document.
        total_pages: Total number of pages extracted.
        chunks: Tuple of all Chunk objects belonging to the document.
    """
    doc_id: str
    filename: str
    total_pages: int
    chunks: Tuple[Chunk, ...]


class DocumentStore:
    """In-memory document store guarded by threading.Lock."""

    def __init__(self) -> None:
        self._records: Dict[str, DocumentRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.doc_id] = record

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            return self._records.get(doc_id)

    def list(self) -> List[DocumentRecord]:
        with self._lock:
            return list(self._records.values())


def ingest_pdf(
    path: Union[str, Path],
    store: DocumentStore,
    original_filename: Optional[str] = None,
) -> DocumentRecord:
    """
    Extracts, chunks, stores, and returns a DocumentRecord for a PDF.

    Args:
        path: Path to the target PDF file.
        store: DocumentStore instance to store the record in.
        original_filename: Optional user-facing filename override.

    Returns:
        DocumentRecord with generated doc_id and chunk tuple.

    Raises:
        PDFExtractionError subclasses: Propagated directly from extractor.
        NoTextError: If the document has no extractable text or yields 0 chunks.
    """
    doc_result = extract_text_from_pdf(path)

    if not doc_result.has_text:
        raise NoTextError(f"PDF contains no extractable text: {path}")

    chunks = tuple(chunk_document(doc_result))
    if not chunks:
        raise NoTextError(f"Chunking yielded 0 chunks for PDF: {path}")

    filename = original_filename or Path(path).name
    doc_id = uuid4().hex

    record = DocumentRecord(
        doc_id=doc_id,
        filename=filename,
        total_pages=doc_result.total_pages,
        chunks=chunks,
    )

    store.add(record)
    return record


def ask(
    record: DocumentRecord,
    question: str,
    llm: Any,
    top_k: int = 5,
) -> dict:
    """
    Answers a question grounded on a document's chunks using retrieval and LLM.

    Args:
        record: Target DocumentRecord containing chunks.
        question: Query string.
        llm: LLM client supporting generate(system, user).
        top_k: Number of candidate chunks to retrieve.

    Returns:
        A dict matching exactly one of three shapes:
        - error: {"status": "error", "reason": "unparseable_model_output", "retrieved": [...]}
        - not_found: {"status": "not_found", "retrieved": [...]}
        - answered: {"status": "answered", "answer": str, "citations": [...], "retrieved": [...]}
    """
    retrieval_results = retrieve(record.chunks, question, top_k=top_k)
    passage_dicts = [{"id": r.passage_id, "text": r.passage} for r in retrieval_results]
    retrieved = [
        {
            "id": r.passage_id,
            "page": r.page_number,
            "score": r.score,
            "text": r.passage,
        }
        for r in retrieval_results
    ]

    answer_result = answer_question(question, passage_dicts, llm)

    if not answer_result.parse_ok:
        return {
            "status": "error",
            "reason": "unparseable_model_output",
            "retrieved": retrieved,
        }

    if not answer_result.answerable:
        return {
            "status": "not_found",
            "retrieved": retrieved,
        }

    chunk_map = {c.passage_id: c for c in record.chunks}
    citations = []
    for cid in answer_result.citations:
        chunk = chunk_map.get(cid)
        if chunk is None:
            return {
                "status": "not_found",
                "retrieved": retrieved,
            }
        citations.append({
            "id": chunk.passage_id,
            "page": chunk.page_number,
            "text": chunk.text,
        })

    if not citations:
        return {
            "status": "not_found",
            "retrieved": retrieved,
        }

    return {
        "status": "answered",
        "answer": answer_result.answer,
        "citations": citations,
        "retrieved": retrieved,
    }

