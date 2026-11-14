# PaperTrail — Milestone 2 Change Report: Keyword Search

**Date:** 2026-10-16  
**System:** PaperTrail (Evidence-First Research Paper Question-Answering System)  
**Report File:** `MILESTONE_2_REPORT.md`  

---

## 1. Milestone Summary

### What functionality was added:
- **Core Keyword Search Engine (`src/search.py`):**
  - Case-insensitive keyword and phrase matching across text extracted from PDF documents.
  - Paragraph-aware passage segmentation (`split_into_passages`) to ensure multi-line sentences and complete paragraphs remain intact.
  - Multi-word matching that handles arbitrary whitespace and newline wraps (`\s+`) via compiled regex with `re.IGNORECASE` and `re.escape`.
  - An immutable data model (`SearchResult`) that couples each matching passage with its source document name and physical 1-based page number.
  - Layered search functions (`search_pages`, `search_document`, `search_pdf`) that operate directly on existing extraction data models.
- **CLI Search Capability (`src/cli.py`):**
  - Added an optional `--search` flag (`python -m src.cli <file.pdf> --search "<query>"`).
  - Clean terminal presentation showing document name, 1-based page number, and matching passage text for every hit.
  - Graceful messaging for empty queries and queries yielding zero matches.
- **Dedicated Keyword Search Test Suite (`tests/test_search.py`):**
  - 16 new automated unit and integration tests covering passage splitting, single/multi-word queries, case-insensitivity, multiple matching pages, blank pages, non-existent terms, empty queries, and CLI behavior.

### What existing functionality was modified:
- **`src/cli.py`:** Extended argument parsing with `--search` (using `nargs="?"` and `const=""` for clean PowerShell/bash compatibility). Added execution branching: if `--search` is supplied, it executes search mode; otherwise, it executes the existing extraction inspection flow.
- **`src/__init__.py`:** Updated package exports to expose `SearchResult`, `search_document`, `search_pages`, `search_pdf`, and `split_into_passages`.
- **`README.md`:** Updated project overview, directory tree, CLI search commands, and architecture design rationale.
- **`PROJECT_EXPLANATION.txt`:** Added Section 4 (Keyword Search Architecture deep dive), updated Section 5 (CLI branching), updated Section 6 (30 automated tests explained), updated Section 7 (execution walkthroughs), and updated Section 8 (interview questions & answers for Milestone 2).

### Whether any bugs were fixed:
- In Milestone 1, there were zero pre-existing bugs (all 14 original tests passed).
- During Milestone 2 development, a potential line-wrap issue inherent to PDF extraction was proactively identified and resolved: PDF text streams insert single newlines (`\n`) at line boundaries. A naive substring match (`query in passage`) would fail if a search phrase crossed a line break (e.g., `"supervised\nlearning"`). By tokenizing query terms and joining them with `\s+` via regex, multi-word matching across line wraps was made completely reliable.

---

## 2. Files Changed

| File Path (Relative) | Status | Specific Changes and Rationale |
|---|---|---|
| `src/search.py` | **Created** | Created core search engine: defined `SearchResult` dataclass, `split_into_passages()`, `search_pages()`, `search_document()`, and `search_pdf()`. Keeps search logic fully decoupled from the CLI. |
| `tests/test_search.py` | **Created** | Created 16 unit and integration tests covering passage chunking, case-insensitive matching, multi-page hits, 1-based page preservation, blank PDFs, empty queries, and CLI search output. |
| `src/cli.py` | **Modified** | Added `--search` argument to `build_parser()`. Implemented search output formatting and branching in `main()` while keeping the default extraction flow and `--full-text` flag intact. |
| `src/__init__.py` | **Modified** | Exported new search functions and data classes in `__all__` to provide a unified public package API. |
| `README.md` | **Modified** | Added Milestone 2 overview, updated folder structure map, added CLI search examples, and documented search design decisions. |
| `PROJECT_EXPLANATION.txt` | **Modified** | Comprehensive documentation update: added Section 4 (search architecture deep dive), updated Section 5 (CLI), Section 6 (testing breakdown for all 30 tests), Section 7 (step-by-step walkthroughs), and Section 8 (placement interview Q&A). |
| `requirements.txt` | **Unchanged** | Verified that no third-party packages beyond `pymupdf` and `pytest` are needed. Standard library modules (`re`, `dataclasses`, `pathlib`, `typing`, `argparse`) were used exclusively. |
| `sample_paper.pdf` | **Unchanged** | Sample mock document retained in original working state. |
| `scripts/create_sample_paper.py` | **Unchanged** | Retained utility script for generating mock research papers. |
| `tests/conftest.py` | **Unchanged** | Retained dynamic PDF fixtures without modification. |
| `tests/test_extractor.py` | **Unchanged** | All 14 original Milestone 1 tests retained without alteration. |

---

## 3. Implementation Details

### How Keyword Search Works (End-to-End Workflow)
1. **Query Ingestion:**
   The user supplies a query string via Python API (`search_document`, `search_pages`, `search_pdf`) or terminal flag (`--search "<query>"`).
2. **Defensive Normalization:**
   The query is stripped of leading and trailing whitespace. If the stripped query is empty, the search returns an empty list `[]` immediately without executing regex or looping over pages.
3. **Query Compilation:**
   The query is split into individual whitespace-delimited words. Each word is escaped with `re.escape()` to prevent math symbols or punctuation (e.g., `H(x)`, `[1]`, `F(x) := H(x) - x`) from acting as regex operators. The words are joined by `\s+` and compiled with `re.IGNORECASE`.
4. **Document Traversal:**
   The engine iterates over the list of `PageContent` objects:
   - Pages with `has_text = False` or empty text are skipped.
   - For valid pages, `split_into_passages(page.text)` segments the page into candidate passages by splitting on double newlines (`\n\s*\n+`).
5. **Passage Matching & Provenance Preservation:**
   Each candidate passage is evaluated against the compiled pattern using `pattern.search(passage)`. When a match occurs, a `SearchResult` is instantiated:
   - `document_name`: Inherited from `page.document_name`.
   - `page_number`: Inherited from `page.page_number` (strictly 1-based).
   - `passage`: The matching text block.
   - `query`: The normalized query string.
6. **Result Delivery:**
   A list of `SearchResult` instances is returned to the caller or printed by the CLI.

### Functions and Classes Involved
- `SearchResult` (`src/search.py`): Immutable dataclass (`frozen=True`) storing `document_name: str`, `page_number: int`, `passage: str`, `query: str`.
- `split_into_passages(text: str) -> List[str]` (`src/search.py`): Normalizes carriage returns and splits text into cohesive paragraphs.
- `search_pages(pages: List[PageContent], query: str) -> List[SearchResult]` (`src/search.py`): Core search algorithm matching across a collection of pages.
- `search_document(document: DocumentExtractionResult, query: str) -> List[SearchResult]` (`src/search.py`): Convenience bridge accepting `DocumentExtractionResult`.
- `search_pdf(file_path: Union[str, Path], query: str) -> List[SearchResult]` (`src/search.py`): End-to-end extraction and search helper.
- `build_parser() -> argparse.ArgumentParser` (`src/cli.py`): Configures command-line options including `--search`.
- `main(args=None) -> int` (`src/cli.py`): Entry point directing execution to search or extraction display.

### Key Design Decisions
- **Reuse of Extraction Data Models:** `src/search.py` directly ingests `PageContent` and `DocumentExtractionResult` from `src/extractor.py`. No redundant intermediary data structures were introduced.
- **Decoupled Architecture:** `src/search.py` contains zero CLI imports, zero `print` calls, and zero `sys.exit` calls. It is 100% pure Python and independently testable.
- **Deterministic Keyword Baseline:** Consciously avoided embeddings, vector databases, or LLM integrations in Milestone 2 to establish an immutable, verifiable baseline for citation provenance.

---

## 4. Test Results

### Exact Test Command Executed:
```powershell
pytest -v
```

### Actual Test Results:
- **Total Tests Collected:** 30
- **Passed:** 30
- **Failed:** 0
- **Skipped:** 0
- **Duration:** 0.45 seconds
- **Environment:** Windows 11, Python 3.14.5, pytest-9.1.1, pluggy-1.6.0

### Breakdown by Test Module:
1. **`tests/test_extractor.py` (Milestone 1 — 14 tests):**
   - `test_normal_extraction_basic_metadata`: **PASSED**
   - `test_one_based_page_numbering_preserved`: **PASSED**
   - `test_page_content_accuracy`: **PASSED**
   - `test_mixed_blank_and_text_pages_preserves_numbering`: **PASSED**
   - `test_blank_pdf_handled_gracefully`: **PASSED**
   - `test_file_not_found`: **PASSED**
   - `test_directory_path_raises_error`: **PASSED**
   - `test_non_pdf_file_raises_invalid_pdf_error`: **PASSED**
   - `test_corrupted_pdf_raises_invalid_pdf_error`: **PASSED**
   - `test_cli_success_output`: **PASSED**
   - `test_cli_full_text_flag`: **PASSED**
   - `test_cli_blank_pdf_warning`: **PASSED**
   - `test_cli_missing_file_returns_error`: **PASSED**
   - `test_cli_non_pdf_returns_error`: **PASSED**

2. **`tests/test_search.py` (Milestone 2 — 16 tests):**
   - `test_split_double_newline`: **PASSED**
   - `test_split_single_paragraph_with_linebreaks`: **PASSED**
   - `test_split_empty_and_whitespace`: **PASSED**
   - `test_matching_single_term`: **PASSED**
   - `test_case_insensitive_matching`: **PASSED**
   - `test_multiple_matching_pages`: **PASSED**
   - `test_no_matches_returns_empty_list`: **PASSED**
   - `test_empty_and_whitespace_queries_return_empty_list`: **PASSED**
   - `test_page_number_preservation_with_intermediate_blank_page`: **PASSED**
   - `test_blank_pdf_returns_no_matches`: **PASSED**
   - `test_multi_word_phrase_matching`: **PASSED**
   - `test_search_pdf_convenience_function`: **PASSED**
   - `test_cli_search_matching_results`: **PASSED**
   - `test_cli_search_no_matches`: **PASSED**
   - `test_cli_search_empty_query`: **PASSED**
   - `test_cli_search_on_missing_file_reports_error`: **PASSED**

---

## 5. Requirements Checklist

| Requirement | Status | Evidence / Verification |
|---|---|---|
| **1. Allow user to search extracted PDF text using a query** | **PASS** | Verified via `search_document()`, `search_pages()`, and `search_pdf()` in `src/search.py`. |
| **2. Return matching passages with document name and 1-based page number** | **PASS** | `SearchResult` pairs passage text with `document_name` and `page_number`. Verified by `test_matching_single_term`. |
| **3. Simple case-insensitive keyword matching (no embeddings/LLMs/vector DBs)** | **PASS** | Implemented purely with Python standard library `re` (`re.IGNORECASE`). Verified by `test_case_insensitive_matching`. Zero vector DB/LLM dependencies. |
| **4. Handle empty queries and queries with no matches clearly** | **PASS** | Returns `[]` in library code; CLI prints clear notices. Verified by `test_no_matches_returns_empty_list`, `test_empty_and_whitespace_queries_return_empty_list`, and `test_cli_search_empty_query`. |
| **5. Preserve association between every result and original page** | **PASS** | `SearchResult.page_number` directly inherits `PageContent.page_number`. Verified by `test_page_number_preservation_with_intermediate_blank_page` (Page 3 match remains Page 3 when Page 2 is blank). |
| **6. Keep search logic separate from CLI** | **PASS** | Pure functions in `src/search.py` have zero terminal I/O dependencies. CLI in `src/cli.py` handles input parsing and terminal formatting. |
| **7. Unit tests for matching terms, case-insensitivity, multi-page matches, no matches, empty queries, page preservation** | **PASS** | All required scenarios are covered by dedicated tests in `tests/test_search.py`. |
| **8. Reuse existing extraction data models where appropriate** | **PASS** | Ingests `PageContent` and `DocumentExtractionResult` from `src/extractor.py`. |
| **9. No unnecessary dependencies or unrelated changes** | **PASS** | `requirements.txt` unchanged (`pymupdf`, `pytest`). Extraction logic untouched. |
| **10. Full test suite actually executed and results reported** | **PASS** | Executed `pytest -v`. All 30 tests passed in 0.45s. |
| **CLI 1: Support `python -m src.cli <file.pdf> --search "<query>"`** | **PASS** | Implemented in `src/cli.py` and verified by `test_cli_search_matching_results`. |
| **CLI 2: Display document name, page number, and matching passage** | **PASS** | Formatted output verified in terminal and by capsys in `test_cli_search_matching_results`. |
| **CLI 3: Display clear message if no results are found** | **PASS** | Outputs `No matching passages found for query: '<query>'`. Verified by `test_cli_search_no_matches`. |
| **CLI 4: Keep extraction behavior and `--full-text` working** | **PASS** | Verified by `test_cli_success_output` and `test_cli_full_text_flag`. |

---

## 6. Important Code Changes

### Excerpt 1: `SearchResult` Data Model
**File:** [`src/search.py`](file:///d:/Projects/PaperTrail/src/search.py#L21-L36)  
```python
@dataclass(frozen=True)
class SearchResult:
    """
    Represents a matching passage found within a document page.

    Attributes:
        document_name: The filename of the source document (e.g., 'sample_paper.pdf').
        page_number: The 1-based physical page number where the match was found.
        passage: The text of the matching passage or paragraph.
        query: The search term that produced this match.
    """
    document_name: str
    page_number: int
    passage: str
    query: str = ""
```

### Excerpt 2: Passage Splitting
**File:** [`src/search.py`](file:///d:/Projects/PaperTrail/src/search.py#L38-L64)  
```python
def split_into_passages(text: str) -> List[str]:
    """Splits page text into candidate passages (paragraphs) separated by double newlines."""
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    raw_passages = re.split(r"\n\s*\n+", normalized)
    passages = [p.strip() for p in raw_passages if p.strip()]
    return passages if passages else [normalized]
```

### Excerpt 3: Multi-Word Case-Insensitive Matching
**File:** [`src/search.py`](file:///d:/Projects/PaperTrail/src/search.py#L90-L119)  
```python
    # Build regex pattern matching words separated by any whitespace (including newlines)
    pattern_str = r"\s+".join(re.escape(w) for w in words)
    pattern = re.compile(pattern_str, re.IGNORECASE)

    results: List[SearchResult] = []
    for page in pages:
        if not page.has_text or not page.text.strip():
            continue

        passages = split_into_passages(page.text)
        for passage in passages:
            if pattern.search(passage):
                results.append(
                    SearchResult(
                        document_name=page.document_name,
                        page_number=page.page_number,
                        passage=passage,
                        query=query_clean,
                    )
                )
```

### Excerpt 4: CLI `--search` Integration
**File:** [`src/cli.py`](file:///d:/Projects/PaperTrail/src/cli.py#L29-L38)  
```python
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        nargs="?",
        const="",
        metavar="QUERY",
        help="Search extracted PDF text for passages matching the given keyword or phrase.",
    )
```

---

## 7. Known Issues and Limitations

1. **Exact Substring vs. Semantic Equivalence:**
   Keyword search matches literal words (e.g., `"neural networks"` matches `"neural networks"` or `"neural\nnetworks"`). It cannot identify synonyms or semantic equivalents (e.g., searching `"deep learning"` will not retrieve passages mentioning only `"convolutional models"`). Semantic search and embedding-based retrieval are slated for upcoming milestones.
2. **Sequential Occurrence vs. Relevance Scoring:**
   Matches are currently returned in sequential document order (Page 1 matches, then Page 2 matches). No frequency-based or BM25/TF-IDF statistical ranking is applied yet.
3. **Scanned Documents Without OCR:**
   PyMuPDF extracts digital text layers. If a PDF consists of image-only scanned pages, extraction yields `has_text = False` and search returns `[]`. OCR pipeline integration would be needed to extract text from scanned documents.
4. **Complex Two-Column Reading Flow:**
   PyMuPDF extracts text streams in the order recorded in the PDF file. In certain complex two-column layouts, line wraps can occasionally interleave across columns. Advanced layout analysis will be addressed in passage chunking improvements.

---

## 8. Beginner Learning Guide

### Key Python Concepts Introduced in Milestone 2
1. **Regular Expressions (`re` Module):**
   - `re.compile(pattern, re.IGNORECASE)`: Pre-compiles regular expressions for efficient repetitive matching without case sensitivity.
   - `re.escape(string)`: Escapes special regex characters (such as `()`, `[]`, `+`, `.`, `*`), ensuring math formulas and scientific citations in papers are searched literally rather than evaluated as regex operators.
   - `r"\s+".join(...)`: Matches one or more whitespace characters, enabling multi-word searches to match even when terms are separated by line breaks (`\n`) in wrapped PDF text.
2. **Immutable Dataclasses (`@dataclass(frozen=True)`):**
   - Prevents accidental modification of objects after creation. If downstream code attempts to change `result.page_number = 5`, Python raises a `FrozenInstanceError`.
3. **CLI Argument Optionals with Defaults (`nargs='?'` and `const=''`):**
   - In `argparse`, setting `nargs="?"` and `const=""` allows the flag to be provided without an argument (e.g. `--search ""`), which prevents shell argument stripping errors in PowerShell.
4. **Software Decoupling (Separation of Concerns):**
   - By keeping `src/search.py` strictly decoupled from terminal I/O (`print`, `sys.argv`), the search engine can be tested directly in pytest, run from the CLI, or imported into future REST API endpoints without changing any search code.

### Recommended Functions to Inspect
1. [`src/search.py: search_pages()`](file:///d:/Projects/PaperTrail/src/search.py#L67-L121) — The core search loop connecting query compilation to extracted pages.
2. [`src/search.py: split_into_passages()`](file:///d:/Projects/PaperTrail/src/search.py#L38-L64) — How raw extracted page text is turned into cohesive paragraphs.
3. [`src/cli.py: main()`](file:///d:/Projects/PaperTrail/src/cli.py#L46-L125) — How execution branches cleanly between search and extraction inspection.

---

## 9. How to Reproduce

Run these commands from the project root (`d:\Projects\PaperTrail`) in Windows PowerShell:

### 1. Run the Full Automated Test Suite (30 Tests)
```powershell
pytest -v
```

### 2. Search for an Existing Term
```powershell
python -m src.cli sample_paper.pdf --search "attention"
```

### 3. Search for a Multi-Word Phrase
```powershell
python -m src.cli sample_paper.pdf --search "encoder-decoder"
```

### 4. Search for a Non-Existent Term (Demonstrates "No Matches" Handling)
```powershell
python -m src.cli sample_paper.pdf --search "supervised learning"
```

### 5. Search with an Empty Query (Demonstrates Edge-Case Handling)
```powershell
python -m src.cli sample_paper.pdf --search "   "
```

### 6. Verify Original Milestone 1 Extraction Flow Still Works
```powershell
# Default preview snippets
python -m src.cli sample_paper.pdf

# Full extracted page text
python -m src.cli sample_paper.pdf --full-text
```

---

## 10. Git Diff Summary

**Git Availability:** `git` is not installed or not available on `PATH` in this environment (verified via `where.exe git`, which returned exit code 1). Therefore, native `git diff` output cannot be generated directly.

### File-Level Change Summary Relative to Milestone 1:
- **`src/search.py` (New file, 151 lines):** Implements `SearchResult`, `split_into_passages()`, `search_pages()`, `search_document()`, and `search_pdf()`.
- **`tests/test_search.py` (New file, 172 lines):** Implements 16 unit and integration tests across 3 test classes.
- **`src/cli.py` (Modified):** Added `--search` argument to argument parser, added search execution branch, added search formatting output. Original extraction branch preserved intact.
- **`src/__init__.py` (Modified):** Added exports for search functions and `SearchResult`.
- **`README.md` (Modified):** Added Milestone 2 sections, CLI commands, and search architecture documentation.
- **`PROJECT_EXPLANATION.txt` (Modified):** Added Section 4, updated Sections 5, 6, 7, and 8 to detail keyword search, test breakdown, and interview Q&As.
- **All other files:** Intact and identical to Milestone 1.
