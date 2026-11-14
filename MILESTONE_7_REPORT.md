# PaperTrail -- Milestone 7 Report: Web API & Interactive UI

**Date:** 2026-11-08  
**System:** PaperTrail (Evidence-First Research Paper Question-Answering System)  
**Report File:** `MILESTONE_7_REPORT.md`  
**Environment:** Windows 11, Python 3.14.5, `GEMINI_MODEL=gemini-3.5-flash-lite`

---

## 1. Summary

Milestone 7 transitions PaperTrail from a set of command-line tools and evaluation harnesses into an interactive web application backed by a hardened REST API.

The architecture strictly decouples application business logic from web transport:
1. **Core Service Layer (`src/service.py`):** Pure Python module encapsulating document ingestion, thread-safe in-memory storage, retrieval orchestration, and grounded answer synthesis without importing web frameworks or schema libraries.
2. **REST API (`src/api.py`):** FastAPI application providing endpoints for health checking, document upload, metadata inspection, question answering, and static frontend delivery. Uploads are strictly validated, streamed, and bounded to 25 MB. Errors and LLM exceptions are mapped to clean HTTP status codes without leaking internal details.
3. **Frontend (`static/index.html`):** A zero-dependency, single-file HTML/CSS/JavaScript client. It renders three mutually exclusive states (`answered`, `not_found`, `error`) and constructs all DOM nodes using `createElement` and `textContent` to prevent script injection.

All pre-existing components (`src/extractor.py`, `src/chunker.py`, `src/retriever.py`, `src/search.py`, `src/answerer.py`, `src/dense_retriever.py`, `src/hybrid_retriever.py`, `src/verifier.py`) were preserved without modification. The test suite expanded from 159 tests to 196 tests (+37 new tests).

---

## 2. What Was Built

### Core Service (`src/service.py`)
- **`NoTextError`**: Raised when a PDF contains no extractable text or chunking yields zero passages.
- **`DocumentRecord`**: Immutable frozen dataclass storing `doc_id` (UUID hex), `filename` (user-facing document name), `total_pages`, and `chunks` (tuple of `Chunk` objects).
- **`DocumentStore`**: Thread-safe in-memory store guarded by a `threading.Lock`. Supports `add(record)`, `get(doc_id)`, and `list()`.
- **`ingest_pdf(path, store, original_filename=None)`**: Extracts text via PyMuPDF, chunks the document, generates a 32-character hex UUID, registers the record in the store, and propagates `PDFExtractionError` subclasses unchanged.
- **`ask(record, question, llm, top_k=5)`**: Retrieves top-5 chunks via BM25, formats passage context, calls `answer_question`, and returns one of exactly three dict shapes:
  - `{"status": "answered", "answer": str, "citations": [{"id", "page", "text"}], "retrieved": [...]}`
  - `{"status": "not_found", "retrieved": [...]}`
  - `{"status": "error", "reason": "unparseable_model_output", "retrieved": [...]}`
  Exceptions from the LLM client propagate directly to the caller.

### REST API (`src/api.py`)
- **`create_app(store, upload_dir, llm_factory=None, max_upload_bytes=25*1024*1024, static_dir=None)`**: Application factory returning a configured FastAPI instance with module-level `app = create_app(DocumentStore(), PROJECT_ROOT / "uploads")`.
- **Endpoints (implemented as plain `def` handlers)**:
  - `GET /health` -> `{"status": "ok"}` (HTTP 200).
  - `POST /documents` -> Multipart file upload (`"file"`). Validates `%PDF-` magic bytes, streams up to 25 MB in 1 MB chunks to `upload_dir/<uuid4 hex>.pdf`, calls `ingest_pdf`, returns HTTP 201 with metadata `{doc_id, filename, total_pages, chunk_count}`. Deletes uploaded files if extraction fails.
  - `GET /documents` -> Lists all documents in store with `{doc_id, filename, total_pages, chunk_count}` (HTTP 200).
  - `GET /documents/{doc_id}` -> Returns metadata for a specific document (HTTP 200), or HTTP 404 if unknown.
  - `POST /documents/{doc_id}/ask` -> Validates `AskRequest(question: str)` (stripped, 1 to 1000 characters, returning HTTP 422 if empty or too long). Resolves document from store (HTTP 404 if missing), invokes `ask()`, and returns result with HTTP 200. Mapped exceptions return sanitized HTTP 503 or 502.
  - `GET /` -> Serves `static/index.html` via `FileResponse(media_type="text/html")` (HTTP 200), or HTTP 404 if the file is missing.

### Web Client (`static/index.html`)
- Single standalone file containing responsive CSS and vanilla JavaScript.
- Uses system font stack, light-mode palette, and maximum width of 800px.
- Zero external dependencies: no CDN links, no external fonts or scripts, no inline SVGs, and no `http://` or `https://` string literals.
- DOM constructed strictly using `document.createElement` and `element.textContent` (no `innerHTML` or `document.write`).

---

## 3. Design Decisions

1. **Strict Dependency Isolation in Service Layer**:
   `src/service.py` contains no imports of `fastapi`, `pydantic`, or `starlette`, so it can be tested and reused without the web layer.
2. **Three Discrete Response States**:
   `ask()` yields exactly three return shapes: `"answered"`, `"not_found"`, and `"error"`. Unparseable model outputs are classified as `"error"` with `reason: "unparseable_model_output"`, never misreported as `"not_found"`.
3. **Server-Side Citation Resolution**:
   Citation page numbers and text snippets are resolved on the server by looking up citation IDs in the indexed `record.chunks`. The LLM's raw text output is never used for citation content. Cited IDs that are not among the passages shown to the model are dropped by `answer_question`; if no valid citation remains, the result is `"not_found"`.
4. **Synchronous Endpoint Execution (`def` vs `async def`)**:
   Endpoints are defined as standard `def` functions. Because PyMuPDF extraction, BM25 indexing, and LLM network calls with exponential backoff are blocking operations, FastAPI automatically offloads them to a threadpool worker, avoiding event-loop starvation.
5. **Lazy LLM Singleton**:
   A helper on `app.state` lazily instantiates a single `GeminiLLMClient(delay_between_calls=0.0)` inside a `threading.Lock()` on the first request. The class is imported inside the helper function so that application startup and test suites run without requiring API keys. When an `llm_factory` is injected, the factory is invoked instead.
6. **Upload Hardening & Stream Bounding**:
   - The first 5 bytes of every upload are inspected before any disk write. If they do not match `b"%PDF-"`, the endpoint immediately aborts with HTTP 400 (`"File is not a PDF"`).
   - Files are streamed to `<uuid4 hex>.pdf` in 1 MB chunks while accumulating bytes. If the total exceeds 25 MB, writing halts, the partial file is removed, and HTTP 413 (`"File too large"`) is returned.
   - Client-provided filenames (such as path traversal attempts like `../../evil.pdf`) are stored purely as metadata on `DocumentRecord.filename`; the physical file on disk is always a random 32-character hexadecimal UUID.
7. **Sanitized Error Mapping**:
   In `POST /documents/{doc_id}/ask`:
   - `(ConnectionError, TimeoutError)` -> HTTP 503 (`"LLM unreachable, try again later"`)
   - `EnvironmentError` -> HTTP 503 (`"LLM is not configured"`)
   - `RuntimeError` -> HTTP 503 (`"LLM unavailable or rate limited, try again later"`)
   - Any other `Exception` -> HTTP 502 (`"LLM request failed"`)
   Real exceptions are recorded via `logger.exception()`. Internal error strings (tested with `"SECRET-123"` markers) are stripped from responses.
8. **Frontend Passage ID Abstraction**:
   While internal passage IDs reflect the disk UUID (e.g. `<uuid>::p8::c3`), the frontend evidence card maps citations to the user's selected filename (`<filename> - page <page>`).

---

## 4. Automated Verification

The test suite grew by 37 new unit and integration tests across five dedicated modules:

| Test File | Number of Tests (`def test_`) | Scope & Coverage |
|---|:---:|---|
| `tests/test_service.py` | 9 | `ingest_pdf` metadata, `NoTextError`, `PDFExtractionError` propagation, `ask()` answered/not-found/error states, invalid citation IDs, LLM exception bubbling, and `retrieved` presence. |
| `tests/test_api.py` | 9 | Health check, PDF upload, document listing, detail retrieval, unknown ID 404, non-PDF 400, blank PDF 422, cleanup on failure, and UUID disk isolation. |
| `tests/test_api_ask.py` | 9 | Ask answered, not found, unparseable stub error, unknown doc 404, question length/whitespace validation (422), LLM exception mapping (502/503), factory errors, and secret masking. |
| `tests/test_api_hardening.py` | 7 | Non-PDF magic bytes rejection (400) with empty directory, path traversal sanitization (`../../evil.pdf`), upload size limit enforcement (413, tested with a small limit), large-limit success (201), `ConnectionError` (503), `TimeoutError` (503), and `EnvironmentError` (503). |
| `tests/test_api_static.py` | 3 | Static `index.html` delivery with `text/html` (200), missing file 404, and health check retention (200). |
| **Total New M7 Tests** | **37** | |

### Full Test Suite Run:
```powershell
python -m pytest -q
```
**Result:** `196 passed, 1 warning` (159 pre-existing + 37 new).  
The suite also passed (196 passed) with Wi-Fi turned off.

---

## 5. Live Verification

Manual verification was conducted on Windows 11 with Python 3.14.5 and `GEMINI_MODEL=gemini-3.5-flash-lite`:

1. **Document Upload**:
   - Uploaded `attention.pdf` (15 pages): returned HTTP 201 with `total_pages: 15` and `chunk_count: 66`.
2. **Answered Query**:
   - Question: *"What BLEU score did the big Transformer model achieve on the WMT 2014 English-to-German task?"*
   - Outcome: Status `answered`. Answer returned `28.4`. Cited page 8 (`attention.pdf::p8::c3` in the saved copy). The cited passage text contains `28.4`.
3. **Response Caching**:
   - Repeating the attention-heads question (an identical request) took **0.067 s** (cache hit from `./.cache/llm/`).
   - The first, uncached call for that question took about 5 s (5.04 s measured).
4. **Explicit Refusal (Unanswerable Query)**:
   - Question: *"What accuracy does the model reach on ImageNet?"*
   - Outcome: Status `not_found`. The interface presented an amber refusal banner with the top 5 searched candidate passages (pages 2, 8, 8, 8, 8).
5. **Validation & Error Handling**:
   - Empty question, whitespace-only, missing question field, and 1001-character string all returned HTTP 422.
   - Requesting an unknown document ID returned HTTP 404.
6. **Upload Hardening & Cleanup**:
   - Non-PDF bytes returned HTTP 400 (`"File is not a PDF"`).
   - A 30 MB file returned HTTP 413 (`"File too large"`).
   - Valid PDF upload returned HTTP 201.
   - After executing all three requests in sequence, the `uploads/` directory had gained exactly one new file (the valid PDF); the rejected uploads left nothing behind.
7. **Browser UI Checks**:
   - Both `answered` (green badge, evidence card) and `not_found` (amber badge, searched passages dropdown) rendered correctly.
   - Uploading a text file produced an inline error message under the upload card.
   - A file named `<b>bold</b>.pdf` displayed as literal text with no HTML markup execution.
   - Stopping the backend server produced the expected `"Cannot reach the server."` banner in the interface.

---

## 6. Honest Findings

1. **Retrieval Miss on Base Attention Heads**:
   - Question: *"How many attention heads does the base Transformer model use?"*
   - Outcome: Returned `not_found`.
   - Investigation: The sentence stating the number of heads (h = 8, in the "parallel attention layers" passage) is located on physical page 5 in chunk `attention.pdf::p5::c1`. BM25 ranks 1 through 6 contained no page-5 passages (retrieved passages came from pages 9, 15, 13, 7, 6, and 8); the first page-5 chunk was ranked at position 7.
   - Analysis: Because the retrieval pipeline supplies only the top-5 candidates to the generator, the model was never provided the evidence passage. The refusal was correct given the context provided. A second-stage reranker over the BM25 top-20 candidates would at least have this chunk in its candidate pool (rank 7); whether it would move it into the top 5 has not been tested.
2. **Automated Phrase-Match vs Supporting Context**:
   - In `MILESTONE_5_REPORT.md`, the 73.3% passage Recall@5 metric reflects exact automated matches against a single ground-truth phrase. A manual audit of raw chunk text in Milestone 5 demonstrated that supporting context was present in the top-5 pool for at least 38 of 45 answerable questions.

---

## 7. Known Limitations

- **Volatile Document Store**: Document metadata and passage chunks reside exclusively in memory. All registered documents are lost when the server process terminates. If run under multiple worker processes, each worker maintains its own isolated in-memory store.
- **Upload Retention**: Files saved to `uploads/` remain on disk indefinitely until manually removed.
- **Local Single-Process Demonstration**: The API contains no authentication, user session isolation, or per-client rate limiting. It is designed solely for local execution and should not be exposed to untrusted networks.
- **Narrow UI Evaluation Scope**: Live interactive browser verification was conducted exclusively with `attention.pdf`. `bert.pdf` and `drs.pdf` were used only in the offline evaluation scripts, not through the web interface.
- **Unexercised Browser Error Banner**: The API returns the `error` status for unparseable model output (tested with `StubLLMClient("not json")`), but the page's display of that state (`"Could not read the model's answer"`) was not seen in the browser.
- **Concurrency & Load**: No concurrent load testing or multi-user throughput benchmarks were conducted.
- **Free-Tier Quota & Retry Latency**: As implemented in `src/llm_client.py`, a request is attempted up to 6 times in total. On HTTP 429 each wait is at least 15 s (capped at 60 s), so under sustained rate limiting one ask can take at least about 75 s before the API returns 503, longer than the UI's "up to a minute" message.
- **Offline Judge**: The Milestone 6 answer verification system is an offline evaluation script and is not integrated into the live API serving path.

---

## 8. Candidate Next Steps (Not Started)

- **Cross-Encoder Reranker**: Implement a second-stage cross-encoder over the top-20 BM25 passages to boost Recall@5 and address retrieval misses like the one above.
- **Tokenizer Improvements**: Improve tokenization handling for hyphenated compounds and numeric decimals across all retrieval engines.
- **Coordinate-Based Table Extraction**: Reconstruct table rows from PyMuPDF word bounding boxes (`page.get_text("words")`) rather than PyMuPDF's `find_tables()`, which found no tables on 6 of the 15 table pages in M6.5.
- **Live Verification Integration**: Incorporate the Milestone 6 verification engine into the API response payload to show a verdict per citation alongside each answer.
- **Persistent Storage**: Back `DocumentStore` with SQLite to retain ingested documents across restarts.
- **Expanded Evaluation**: Benchmark live UI behavior across a wider variety of PDF structures and document layouts.

---

## 9. How to Reproduce

### 1. Run Automated Test Suite
```powershell
python -m pytest -q
```
Expected output: `196 passed, 1 warning`.

### 2. Launch Local Server
```powershell
$env:GEMINI_API_KEY = "<your key>"
$env:GEMINI_MODEL = "gemini-3.5-flash-lite"
python -m uvicorn src.api:app --port 8000
```
Must be executed from the project root directory.

### 3. Access Interactive UI
Open `http://127.0.0.1:8000/` in a web browser.

