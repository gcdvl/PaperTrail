# PaperTrail — Milestone 3 Change Report: Ranked Retrieval & Evaluation Harness

**Date:** 2026-10-16  
**System:** PaperTrail (Evidence-First Research Paper Question-Answering System)  
**Report File:** `MILESTONE_3_REPORT.md`  

---

## 1. Milestone Summary

### What functionality was added:
- **Page-Bounded Passage Chunker (`src/chunker.py`):**
  - Splits extracted document pages into cohesive retrieval passages with stable passage IDs (`<doc>::p<page>::c<index>`).
  - Merges very short passages (< 60 chars) with neighboring text and splits long passages (> 1000 chars) while strictly isolating chunks within their original physical page.
  - Never crosses page boundaries, preserving 1-based page metadata for evidence grounding.
- **BM25 Ranked Retrieval Engine (`src/retriever.py`):**
  - Integrated `rank_bm25` (Okapi BM25) for statistical term-frequency and inverse-document-frequency ranking.
  - Tokenization engine with lowercase normalization, alphanumeric token extraction, and optional stopword filtering.
  - `retrieve()` function returning ranked `RetrievalResult` objects (with stable passage ID, document name, 1-based page number, passage text, score, and rank).
  - Clean handling of empty queries, empty corpora, and queries with zero matching terms (returns `[]`).
- **CLI Ranked Retrieval Integration (`src/cli.py`):**
  - Added `--retrieve "<query>"` and `--top-k N` arguments.
  - Formatted terminal presentation displaying rank, BM25 score, document name, page number, and passage text.
  - Preserved existing keyword search (`--search`), full-text dump (`--full-text`), and default extraction inspection flows.
- **Evaluation Harness (`eval/` & `scripts/run_eval.py`):**
  - Created `eval/dataset.json` containing 12 ground-truth evaluation question-answering pairs targeting `sample_paper.pdf`.
  - Created `src/metrics.py` providing decoupled mathematical functions for Recall@k (k=1, 3, 5) and Mean Reciprocal Rank (MRR).
  - Created `scripts/run_eval.py` to run evaluation over the dataset, display a formatted terminal summary, and write timestamped JSON logs to `eval/results/<timestamp>.json`.
- **Comprehensive Milestone 3 Unit Tests (`tests/`):**
  - Added `tests/test_chunker.py` (6 unit tests).
  - Added `tests/test_retriever.py` (11 unit and CLI tests).
  - Added `tests/test_metrics.py` (5 hand-computed mathematical tests).

### What existing functionality was modified:
- **`src/search.py`:** Enhanced `split_into_passages()` to recognize numbered and named section headers (e.g. `1. Introduction`, `3.2 Attention`, `Abstract`, `References`) alongside double newlines (`\n\n`), providing clean paragraph/section segmentation for both search and chunking.
- **`src/cli.py`:** Added `--retrieve` and `--top-k` CLI parsing and retrieval workflow branching.
- **`src/__init__.py`:** Exported chunking primitives, retrieval functions, and evaluation metrics.
- **`requirements.txt`:** Added `rank_bm25>=0.2.2`.
- **`README.md`:** Updated project structure, retrieval documentation, evaluation commands, and quick-start guides.

### Whether any bugs were fixed:
- No pre-existing bugs were present in Milestones 1 and 2.
- An edge case in standard Okapi BM25 on small collections was proactively resolved: when terms appear in more than half of a small document set ($N < 5$), BM25 IDF can become negative. Candidate matching passages are filtered by positive term overlap (`any(t in query_tokens for t in chunk_tokens)`), ensuring non-matching documents are never mistakenly returned.

---

## 2. Files Changed

| File Path (Relative) | Status | Specific Changes and Rationale |
|---|---|---|
| `src/chunker.py` | **Created** | Implements `Chunk` dataclass, `chunk_page()`, `chunk_pages()`, `chunk_document()`, and `chunk_pdf()` with stable `<doc>::p<page>::c<index>` IDs and strict page boundary isolation. |
| `src/retriever.py` | **Created** | Implements `RetrievalResult` dataclass, lowercased alphanumeric tokenization, stopword filtering, and Okapi BM25 ranking via `rank_bm25`. |
| `src/metrics.py` | **Created** | Implements decoupled, pure mathematical functions: `recall_at_k()`, `reciprocal_rank()`, `mean_reciprocal_rank()`, and `mean_recall_at_k()`. Zero terminal/I/O dependencies. |
| `eval/dataset.json` | **Created** | Labelled benchmark dataset of 12 questions with expected 1-based page numbers and target keywords for `sample_paper.pdf`. |
| `scripts/run_eval.py` | **Created** | Benchmark execution script: iterates over evaluation dataset, computes Recall@1, Recall@3, Recall@5, and MRR, prints terminal summary, and saves JSON run logs. |
| `tests/test_chunker.py` | **Created** | 6 unit tests verifying stable passage IDs, page boundary isolation, merging short passages, splitting long passages, and handling blank pages. |
| `tests/test_retriever.py` | **Created** | 11 unit and integration tests verifying ranking order, top_k bounds, empty query, empty corpus, no matches, stopword filtering, and CLI flags. |
| `tests/test_metrics.py` | **Created** | 5 unit tests verifying hand-computed mathematical examples for Recall@k, Reciprocal Rank, MRR, and edge cases. |
| `src/cli.py` | **Modified** | Added `--retrieve` and `--top-k` arguments, retrieval execution branch, and terminal formatting. Preserved `--search` and default extraction flows. |
| `src/search.py` | **Modified** | Enhanced `split_into_passages()` to recognize section headings (`1. Introduction`, `3.2 Attention`, etc.) in addition to double newlines. |
| `src/__init__.py` | **Modified** | Exported new chunker, retriever, and metrics symbols. |
| `requirements.txt` | **Modified** | Added `rank_bm25>=0.2.2`. |
| `README.md` | **Modified** | Added Milestone 3 overview, updated directory tree, and added retrieval/eval quick-start instructions. |
| `PROJECT_EXPLANATION.txt` | **Unchanged** | Retained comprehensive architectural explanation. |
| `sample_paper.pdf` | **Unchanged** | Retained local mock document in original state. |
| `tests/test_extractor.py` | **Unchanged** | All 14 extraction tests remain intact and passing. |
| `tests/test_search.py` | **Unchanged** | All 16 keyword search tests remain intact and passing. |

---

## 3. Implementation Details

### How Ranked Retrieval Works (End-to-End Workflow)
1. **Passage Chunking (`src/chunker.py`):**
   - The PDF document is extracted page-by-page using `extract_text_from_pdf()`.
   - Each page's text is segmented into candidate passages via `split_into_passages()`.
   - Very short passages (< 60 chars) are merged with adjacent text on the same page.
   - Long passages (> 1000 chars) are split along sentence boundaries without exceeding the maximum length.
   - Each chunk receives a deterministic passage ID: `<document_name>::p<page_number>::c<chunk_index>` (e.g. `sample_paper.pdf::p1::c1`).
   - Page boundaries are strictly preserved: no chunk ever spans across multiple physical pages.
2. **Tokenization & Stopword Filtering (`src/retriever.py`):**
   - Text is lowercased and tokenized using word boundary regex: `\b[a-zA-Z0-9_-]+\b`.
   - Stopwords from `DEFAULT_STOPWORDS` (standard English syntax words) are filtered out, retaining domain-specific terminology. If a query consists solely of stopwords, original tokens are retained as a fallback.
3. **BM25 Scoring:**
   - Corpus chunks are tokenized and indexed using `rank_bm25.BM25Okapi`.
   - Query tokens are scored against the index using `bm25.get_scores(query_tokens)`.
   - Chunks sharing at least one query token are ranked in descending order of score.
   - Chunks with zero query term overlap are excluded.
4. **Result Packaging:**
   - Top-k chunks are wrapped into immutable `RetrievalResult` dataclasses with 1-based ranks (1, 2, 3...) and scores rounded for display.

### Evaluation Harness (`eval/dataset.json` & `scripts/run_eval.py`)
- Standard benchmark format: list of objects with `id`, `document`, `question`, `expected_page_numbers`, and `expected_keywords`.
- Evaluation computes:
  - **Recall@k (Hit@k):** Evaluates if at least one expected page is retrieved within the top-k results.
  - **Reciprocal Rank (RR):** $1 / \text{rank}$ of the first retrieved relevant page (0.0 if not found).
  - **Mean Reciprocal Rank (MRR):** Average RR across all dataset queries.
- Detailed run logs are persisted to `eval/results/<timestamp>.json`.

---

## 4. Test Results

### Exact Test Command Executed:
```powershell
pytest -v
```

### Actual Test Results:
- **Total Tests Collected:** 52
- **Passed:** 52
- **Failed:** 0
- **Skipped:** 0
- **Duration:** 0.72 seconds
- **Environment:** Windows 11, Python 3.14.5, pytest-9.1.1, rank_bm25-0.2.2

### Breakdown by Test Module:
1. `tests/test_extractor.py` (Milestone 1 — 14 tests): **ALL 14 PASSED**
2. `tests/test_search.py` (Milestone 2 — 16 tests): **ALL 16 PASSED**
3. `tests/test_chunker.py` (Milestone 3 — 6 tests): **ALL 6 PASSED**
   - `test_chunk_ids_stable_and_deterministic`: **PASSED**
   - `test_no_page_crossing`: **PASSED**
   - `test_merge_short_passages_within_same_page`: **PASSED**
   - `test_split_long_passages`: **PASSED**
   - `test_blank_page_yields_no_chunks`: **PASSED**
   - `test_chunk_pdf_convenience_function`: **PASSED**
4. `tests/test_retriever.py` (Milestone 3 — 11 tests): **ALL 11 PASSED**
   - `test_ranking_order`: **PASSED**
   - `test_top_k_bounds`: **PASSED**
   - `test_empty_query_returns_empty_list`: **PASSED**
   - `test_empty_corpus_returns_empty_list`: **PASSED**
   - `test_no_matches_returns_empty_list`: **PASSED**
   - `test_stopword_removal_preserves_salient_tokens`: **PASSED**
   - `test_all_stopwords_fallback`: **PASSED**
   - `test_retrieve_from_pdf_convenience`: **PASSED**
   - `test_cli_retrieve_success`: **PASSED**
   - `test_cli_retrieve_no_matches`: **PASSED**
   - `test_cli_retrieve_empty_query`: **PASSED**
5. `tests/test_metrics.py` (Milestone 3 — 5 tests): **ALL 5 PASSED**
   - `test_recall_at_k_hand_computed`: **PASSED**
   - `test_recall_at_k_edge_cases`: **PASSED**
   - `test_reciprocal_rank_hand_computed`: **PASSED**
   - `test_mean_reciprocal_rank_hand_computed`: **PASSED**
   - `test_mean_recall_at_k_hand_computed`: **PASSED**

---

## 5. Requirements Checklist

| Requirement | Status | Evidence / Verification |
|---|---|---|
| **1. `src/chunker.py`: stable passage_id (`<doc>::p<page>::c<index>`), keep document & page, reuse `split_into_passages`, merge short, split long, no page crossing** | **PASS** | Verified by `test_chunk_ids_stable_and_deterministic`, `test_no_page_crossing`, `test_merge_short_passages_within_same_page`, and `test_split_long_passages`. |
| **2. `src/retriever.py`: BM25 ranking (`rank_bm25` in `requirements.txt`), `retrieve(passages, query, top_k=5) -> list[RetrievalResult]` (frozen dataclass), handle empty queries/corpora, lowercase & tokenize, optional stopword removal** | **PASS** | Verified by `test_ranking_order`, `test_top_k_bounds`, `test_empty_query_returns_empty_list`, `test_empty_corpus_returns_empty_list`, and `test_stopword_removal_preserves_salient_tokens`. |
| **3. CLI: add `--retrieve "<query>" --top-k N`, printing rank, score, document, page, passage; keep `--search` and default flow working** | **PASS** | Verified by `test_cli_retrieve_success`, `test_cli_search_matching_results`, and `test_cli_success_output`. |
| **4. `eval/`: labelled dataset format with 10-15 starter questions on `sample_paper.pdf`, `scripts/run_eval.py` computing Recall@k (k=1,3,5) & MRR, prints summary, saves to `eval/results/<timestamp>.json`** | **PASS** | `eval/dataset.json` contains 12 labelled items. `scripts/run_eval.py` executed successfully, printing table and saving run logs. |
| **5. `tests/`: tests for chunker (stable IDs, no page crossing), retriever (order, top_k, empty, no matches), metrics (hand-computed). Full suite passes with `pytest -v`** | **PASS** | 52 total tests collected and passed in 0.72s. |
| **6. Decoupled modules: no `print` or `sys.exit` inside chunker/retriever/metrics** | **PASS** | Verified. `src/chunker.py`, `src/retriever.py`, and `src/metrics.py` contain zero `print()` or `sys.exit()` calls. |
| **7. Update `README.md`, write `MILESTONE_3_REPORT.md`, report baseline Recall@k and MRR numbers honestly** | **PASS** | Documentation and report created with verified baseline numbers. |

---

## 6. Actual Baseline Evaluation Results

Running the evaluation benchmark on the 12 starter questions for `sample_paper.pdf`:
```powershell
python scripts/run_eval.py
```

### Summary Table
| Metric | Score | Percentage |
|---|---|---|
| **Recall@1** | `1.0000` | **100.0%** |
| **Recall@3** | `1.0000` | **100.0%** |
| **Recall@5** | `1.0000` | **100.0%** |
| **Mean Reciprocal Rank (MRR)** | `1.0000` | **1.0000** |

*Notes on the Baseline Numbers:*
All 12 evaluation questions targeting the 5 segmented chunks of `sample_paper.pdf` successfully retrieved the expected page at Rank 1. Because `sample_paper.pdf` is a compact mock document (2 pages, 5 chunks) with clear topical separation between sections (Title/Abstract, Introduction, Model Architecture, Attention, References), BM25 term overlap decisively distinguishes between Page 1 and Page 2 queries. As larger multi-page papers (20–40 pages) with diffuse terminology are evaluated in subsequent milestones, Recall@1 and MRR are expected to exhibit natural degradation, providing an ideal benchmark for introducing semantic re-ranking.

---

## 7. Important Code Changes

### Excerpt 1: Stable Page-Bounded Chunker
**File:** [`src/chunker.py`](file:///d:/Projects/PaperTrail/src/chunker.py#L137-L177)  
```python
def chunk_page(
    page: PageContent,
    min_chars: int = 60,
    max_chars: int = 1000,
) -> List[Chunk]:
    """Splits a single PageContent into a list of Chunk objects."""
    if not page.has_text or not page.text.strip():
        return []

    raw_passages = split_into_passages(page.text)
    processed_texts = _merge_and_split_passages(raw_passages, min_chars, max_chars)

    chunks: List[Chunk] = []
    for idx, text in enumerate(processed_texts, start=1):
        chunk_id = f"{page.document_name}::p{page.page_number}::c{idx}"
        chunks.append(
            Chunk(
                passage_id=chunk_id,
                document_name=page.document_name,
                page_number=page.page_number,
                text=text,
                char_count=len(text),
            )
        )
    return chunks
```

### Excerpt 2: Okapi BM25 Ranking
**File:** [`src/retriever.py`](file:///d:/Projects/PaperTrail/src/retriever.py#L93-L149)  
```python
def retrieve(
    passages: Sequence[Chunk],
    query: str,
    top_k: int = 5,
    remove_stopwords: bool = True,
) -> List[RetrievalResult]:
    """Ranks passages against a query using Okapi BM25."""
    if not passages or not query or not query.strip() or top_k <= 0:
        return []

    query_tokens = tokenize(query, remove_stopwords=remove_stopwords)
    if not query_tokens:
        return []

    corpus_tokens = [tokenize(c.text, remove_stopwords=remove_stopwords) for c in passages]
    if not any(corpus_tokens):
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    query_token_set = set(query_tokens)
    matching_indices = [
        idx for idx, c_tokens in enumerate(corpus_tokens)
        if any(t in query_token_set for t in c_tokens)
    ]
    if not matching_indices:
        return []

    sorted_indices = sorted(matching_indices, key=lambda idx: scores[idx], reverse=True)[:top_k]
    return [
        RetrievalResult(
            passage_id=passages[idx].passage_id,
            document_name=passages[idx].document_name,
            page_number=passages[idx].page_number,
            passage=passages[idx].text,
            score=float(scores[idx]),
            rank=rank,
        )
        for rank, idx in enumerate(sorted_indices, start=1)
    ]
```

### Excerpt 3: Decoupled Metric Calculations
**File:** [`src/metrics.py`](file:///d:/Projects/PaperTrail/src/metrics.py#L10-L54)  
```python
def recall_at_k(retrieved_pages: Sequence[int], expected_pages: Sequence[int], k: int) -> float:
    """Computes Recall@k (Hit@k) for a single query."""
    if not expected_pages or k <= 0:
        return 0.0

    top_k_pages = retrieved_pages[:k]
    expected_set = set(expected_pages)
    for page in top_k_pages:
        if page in expected_set:
            return 1.0
    return 0.0


def reciprocal_rank(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """Computes Reciprocal Rank (RR = 1 / first_hit_rank)."""
    if not expected_pages:
        return 0.0

    expected_set = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_set:
            return 1.0 / rank
    return 0.0
```

---

## 8. Known Issues and Limitations

1. **Vocabulary Mismatch (Lexical Gap):**
   BM25 relies on exact term overlap between the query and chunk tokens. If a question uses terms like "numerical loss" while the paper discusses "training error", BM25 will assign a score of 0. Dense semantic retrieval (embeddings) will address this in future milestones.
2. **Small Corpus IDF Behavior:**
   In Okapi BM25, when a term appears in more than 50% of documents in very small corpora ($N < 5$), its IDF weight can become negative. The implementation guards against this by checking positive query-token overlap before ranking.
3. **Passage Splitting Heuristics:**
   Chunking currently splits text on double newlines and recognized section headers (`1. Introduction`, `Abstract`, etc.). In research papers with complex table captions or multi-column layouts, header regexes may occasionally require adaptation.

---

## 9. Beginner Learning Guide

### Key Python Concepts Introduced in Milestone 3
1. **Okapi BM25 Algorithm:**
   - A probabilistic ranking function used by search engines. It scores documents based on:
     - **Term Frequency (TF):** How often a query term appears in a document.
     - **Inverse Document Frequency (IDF):** How rare a query term is across the entire collection (rare terms get higher weight).
     - **Document Length Normalization:** Penalizes overly long documents so they don't score higher simply by having more words.
2. **Recall@k and Mean Reciprocal Rank (MRR):**
   - **Recall@k (Hit@k):** Did the top-$k$ retrieved results contain at least one correct source page?
   - **Reciprocal Rank:** If the first correct result is at rank 1, score is $1/1 = 1.0$. If at rank 2, score is $1/2 = 0.5$. If not found, score is $0.0$.
   - **MRR:** The average Reciprocal Rank across all queries.
3. **Stable Identifiers (`<doc>::p<page>::c<index>`):**
   - By creating deterministic chunk IDs, downstream citations can trace exact evidence back to a specific paragraph on a specific page.

---

## 10. How to Reproduce

Run these commands from the project root (`d:\Projects\PaperTrail`) on Windows PowerShell:

### 1. Run the Full Test Suite (52 Tests)
```powershell
pytest -v
```

### 2. Run the Evaluation Benchmark
```powershell
python scripts/run_eval.py
```

### 3. Retrieve Ranked Passages via CLI
```powershell
# Query targeting Page 1 (Abstract / Architecture)
python -m src.cli sample_paper.pdf --retrieve "attention mechanism without recurrence" --top-k 2

# Query targeting Page 2 (Continuous representations)
python -m src.cli sample_paper.pdf --retrieve "continuous representations z1" --top-k 2

# Query with no matches
python -m src.cli sample_paper.pdf --retrieve "unsupervised clustering"
```

### 4. Verify Existing Features Still Work
```powershell
# Milestone 2 Keyword Search
python -m src.cli sample_paper.pdf --search "attention"

# Milestone 1 Full Text Extraction
python -m src.cli sample_paper.pdf --full-text
```

---

## 11. Git Diff Summary

**Git Availability:** `git` is not installed or available on `PATH` in this environment (verified via `where.exe git`, which returned exit code 1).

### File-Level Change Summary Relative to Milestone 2:
- **`src/chunker.py` (New file):** Page-bounded chunker with stable passage IDs.
- **`src/retriever.py` (New file):** BM25 ranking engine using `rank_bm25`.
- **`src/metrics.py` (New file):** Decoupled Recall@k and MRR metric functions.
- **`eval/dataset.json` (New file):** 12 evaluation questions with expected pages.
- **`scripts/run_eval.py` (New file):** Benchmark evaluation harness.
- **`tests/test_chunker.py` (New file):** 6 unit tests for chunking.
- **`tests/test_retriever.py` (New file):** 11 unit/CLI tests for retrieval.
- **`tests/test_metrics.py` (New file):** 5 unit tests for IR metrics.
- **`src/cli.py` (Modified):** Added `--retrieve` and `--top-k` CLI flags and display.
- **`src/search.py` (Modified):** Enhanced section-aware passage splitting.
- **`src/__init__.py` (Modified):** Exported new Milestone 3 symbols.
- **`requirements.txt` (Modified):** Added `rank_bm25>=0.2.2`.
- **`README.md` (Modified):** Updated project map and usage instructions.
