# PaperTrail -- Milestone 3.5 Change Report: Trustworthy Evaluation Harness

**Date:** 2026-10-17  
**System:** PaperTrail (Evidence-First Research Paper Question-Answering System)  
**Report File:** `MILESTONE_3_5_REPORT.md`  

---

## 1. Milestone Summary

### Goal
Make the evaluation harness trustworthy. No new retrieval methods, LLMs, or
embeddings were added. All changes are to the measurement and validation
infrastructure.

### What functionality was added:
- **Dataset Format v2 (`eval/dataset_v2.json`):**
  - New schema fields: `id`, `document`, `question`, `answerable` (bool), `expected_page_numbers`, and `evidence_phrase` (string or null).
  - Answerable items require a non-null `evidence_phrase` and non-empty `expected_page_numbers`.
  - Unanswerable items require `evidence_phrase: null` and empty `expected_page_numbers`.
  - JSON Schema provided at `eval/dataset_v2_schema.json` for external validation.
  - 2 placeholder example items (one answerable, one unanswerable) for schema reference. Real questions to be written by the user.

- **Dataset Validator (`scripts/validate_dataset.py`):**
  - Schema validation: checks required fields, types, and consistency between `answerable` flag and `evidence_phrase`/`expected_page_numbers`.
  - Evidence grounding: for every answerable item, verifies the `evidence_phrase` actually appears (whitespace-normalized, case-insensitive) on at least one expected page of the referenced PDF.
  - Duplicate ID detection.
  - PDF resolution: checks `papers/` directory first, then project root.

- **Passage-Level Metrics (`src/metrics.py`):**
  - `normalize_whitespace()`: collapses all whitespace to single spaces.
  - `passage_contains_evidence()`: whitespace-normalized, case-insensitive substring match.
  - `passage_recall_at_k()`: a retrieved chunk counts as a hit only if it contains the `evidence_phrase`.
  - `passage_reciprocal_rank()`: RR based on rank of first chunk containing the evidence phrase.
  - `score_distribution_stats()`: computes count, min, max, mean, median for score sequences.
  - All existing page-level metric functions (`recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `mean_recall_at_k`) preserved with unchanged signatures.

- **Unanswerable Item Handling:**
  - Unanswerable items are excluded from page-level and passage-level metrics.
  - Top-1 BM25 score distributions for answerable vs. unanswerable items reported side-by-side for future refusal threshold analysis.
  - No threshold is implemented yet (as specified).

- **Enhanced Evaluation Harness (`scripts/run_eval.py`):**
  - Supports both v1 (`dataset.json`) and v2 (`dataset_v2.json`) formats.
  - Multiple documents: PDFs resolved from `papers/` folder then project root.
  - Page-level and passage-level metrics reported separately.
  - Per-document metric breakdown.
  - Failure listing: for every miss (page-level or passage-level at k=5), prints the question, expected evidence phrase, and top-3 retrieved passage IDs.
  - Score distribution table for answerable vs. unanswerable top-1 BM25 scores.

- **Comprehensive Tests (`tests/test_metrics_v2.py`):**
  - 48 new unit tests covering all new functionality.

### What existing functionality was modified:
- **`src/metrics.py`:** Extended with new functions. All 4 existing function signatures preserved.
- **`src/__init__.py`:** Exported new metrics symbols.
- **`scripts/run_eval.py`:** Rewritten to support v2 features while remaining backward-compatible with v1.

### Whether any bugs were fixed:
- No pre-existing bugs were found. A Unicode encoding issue on Windows (cp1252 console) in the validator output was caught and fixed during development.

---

## 2. Files Changed

| File Path (Relative) | Status | Specific Changes and Rationale |
|---|---|---|
| `eval/dataset_v2.json` | **Created** | V2 dataset with 2 placeholder items demonstrating the schema (1 answerable, 1 unanswerable). |
| `eval/dataset_v2_schema.json` | **Created** | JSON Schema for dataset v2 with conditional validation rules. |
| `scripts/validate_dataset.py` | **Created** | Dataset validator: schema checks, evidence grounding against PDFs, duplicate ID detection. |
| `tests/test_metrics_v2.py` | **Created** | 48 unit tests for passage-level metrics, evidence matching, score distributions, unanswerable handling, and validator logic. |
| `src/metrics.py` | **Modified** | Added `normalize_whitespace()`, `passage_contains_evidence()`, `passage_recall_at_k()`, `passage_reciprocal_rank()`, `score_distribution_stats()`. All existing functions preserved. |
| `src/__init__.py` | **Modified** | Exported 5 new symbols from metrics module. |
| `scripts/run_eval.py` | **Modified** | Rewritten for v2 support: multi-document, page+passage metrics, per-doc breakdown, failure listing, score distributions. Backward-compatible with v1. |
| `eval/dataset.json` | **Unchanged** | V1 dataset retained for backward compatibility. |
| `src/chunker.py` | **Unchanged** | No changes. |
| `src/retriever.py` | **Unchanged** | No changes. |
| `src/extractor.py` | **Unchanged** | No changes. |
| `src/search.py` | **Unchanged** | No changes. |
| `src/cli.py` | **Unchanged** | No changes. |
| `tests/test_extractor.py` | **Unchanged** | All 14 tests passing. |
| `tests/test_search.py` | **Unchanged** | All 16 tests passing. |
| `tests/test_chunker.py` | **Unchanged** | All 6 tests passing. |
| `tests/test_retriever.py` | **Unchanged** | All 11 tests passing. |
| `tests/test_metrics.py` | **Unchanged** | All 5 tests passing. |

---

## 3. Implementation Details

### Passage-Level Metrics
The key insight: page-level Recall@k only checks whether the correct *page* was retrieved, not whether the correct *passage* on that page was retrieved. With passage-level metrics, a chunk only counts as a hit if it contains the `evidence_phrase` (whitespace-normalized, case-insensitive substring match).

**Normalization:**
1. All whitespace sequences (spaces, tabs, newlines) collapsed to single space.
2. Leading/trailing whitespace stripped.
3. Case-insensitive comparison (both lowercased).

This handles PDF extraction artifacts where line breaks or extra spaces appear within phrases.

### Unanswerable Score Distribution
For each item, the top-1 BM25 score is recorded. Answerable and unanswerable items' scores are collected separately, and `score_distribution_stats()` reports count, min, max, mean, and median for each group. This enables future threshold selection (e.g., "if top-1 score < X, refuse to answer") without implementing the threshold in this milestone.

### Multi-Document Support
`run_eval.py` resolves each item's `document` field by checking `papers/` first, then the project root. Documents are cached (extracted and chunked once), so multiple questions targeting the same PDF do not re-extract.

### Failure Listing
For every answerable item where page-level or passage-level Recall@5 < 1.0, the report prints:
- The question text
- The expected evidence phrase
- The top-3 retrieved passage IDs

This enables rapid debugging of retrieval failures without inspecting JSON logs.

---

## 4. Test Results

### Exact Test Command Executed:
```powershell
pytest -v
```

### Actual Test Results:
- **Total Tests Collected:** 100
- **Passed:** 100
- **Failed:** 0
- **Skipped:** 0
- **Duration:** 0.90 seconds
- **Environment:** Windows 11, Python 3.14.5, pytest-9.1.1, rank_bm25-0.2.2

### Breakdown by Test Module:
1. `tests/test_extractor.py` (Milestone 1 -- 14 tests): **ALL 14 PASSED**
2. `tests/test_search.py` (Milestone 2 -- 16 tests): **ALL 16 PASSED**
3. `tests/test_chunker.py` (Milestone 3 -- 6 tests): **ALL 6 PASSED**
4. `tests/test_retriever.py` (Milestone 3 -- 11 tests): **ALL 11 PASSED**
5. `tests/test_metrics.py` (Milestone 3 -- 5 tests): **ALL 5 PASSED**
6. `tests/test_metrics_v2.py` (Milestone 3.5 -- 48 tests): **ALL 48 PASSED**
   - `TestNormalizeWhitespace` (5 tests): **ALL PASSED**
   - `TestPassageContainsEvidence` (8 tests): **ALL PASSED**
   - `TestPassageRecallAtK` (7 tests): **ALL PASSED**
   - `TestPassageReciprocalRank` (6 tests): **ALL PASSED**
   - `TestPassageMRR` (1 test): **PASSED**
   - `TestScoreDistributionStats` (5 tests): **ALL PASSED**
   - `TestUnanswerableHandling` (4 tests): **ALL PASSED**
   - `TestDatasetValidator` (12 tests): **ALL PASSED**

---

## 5. Requirements Checklist

| Requirement | Status | Evidence / Verification |
|---|---|---|
| **1. Dataset format v2 with schema, validator, 2 placeholders** | **PASS** | `eval/dataset_v2.json` (2 items), `eval/dataset_v2_schema.json`, `scripts/validate_dataset.py` validates schema + evidence grounding. Verified by 12 validator tests. |
| **2. Passage-level Recall@k and MRR using evidence_phrase** | **PASS** | `passage_recall_at_k()` and `passage_reciprocal_rank()` in `src/metrics.py`. Whitespace-normalized, case-insensitive matching. Verified by 14 hand-computed tests. |
| **3. Unanswerable top-1 BM25 score distribution** | **PASS** | `score_distribution_stats()` reports count/min/max/mean/median. Eval harness separates answerable vs. unanswerable scores. No threshold implemented. Verified by 4 tests. |
| **4. Multi-document eval, per-document metrics, failure listing** | **PASS** | `run_eval.py` resolves PDFs from `papers/` dir, caches per-doc chunks, reports page + passage metrics per document, prints failure listing with top-3 IDs. Verified by running against both v1 and v2 datasets. |
| **5. Tests for new metrics, validator, unanswerable handling** | **PASS** | 48 new tests in `test_metrics_v2.py`, all passing. All 52 existing tests unbroken. Total: 100 passing. |
| **6. MILESTONE_3_5_REPORT.md with honest numbers** | **PASS** | This report. |

---

## 6. Actual Evaluation Results

### V1 Dataset (dataset.json, 12 queries, backward compatibility)

| Metric | Score |
|---|---|
| **Page Recall@1** | 100.0% (1.0000) |
| **Page Recall@3** | 100.0% (1.0000) |
| **Page Recall@5** | 100.0% (1.0000) |
| **Page MRR** | 1.0000 |

*Notes:* Identical to Milestone 3 baseline. The v1 dataset has no `evidence_phrase` fields, so passage-level metrics are not computed. The `sample_paper.pdf` is a small 2-page mock document where BM25 trivially retrieves the correct page.

### V1 Score Distribution

| Group | Count | Min | Max | Mean | Median |
|---|---|---|---|---|---|
| Answerable | 12 | 0.7574 | 5.0813 | 3.2490 | 3.4993 |
| Unanswerable | 0 | -- | -- | -- | -- |

### V2 Dataset (dataset_v2.json, 2 placeholder items)

| Metric | Score |
|---|---|
| **Page Recall@1** | 0.0% (0.0000) |
| **Passage Recall@1** | 0.0% (0.0000) |
| **Page MRR** | 0.0000 |
| **Passage MRR** | 0.0000 |

*Notes:* The v2 dataset currently contains only placeholder items. The answerable placeholder question ("PLACEHOLDER -- replace with your real question") does not match any content, yielding 0% across all metrics. This is expected and honest. Once real questions are written, these numbers will reflect actual retrieval quality.

### V2 Score Distribution

| Group | Count | Min | Max | Mean | Median |
|---|---|---|---|---|---|
| Answerable | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Unanswerable | 1 | 0.8546 | 0.8546 | 0.8546 | 0.8546 |

*Observation:* The unanswerable placeholder scored 0.8546 on BM25 (some incidental term overlap), while the answerable placeholder scored 0.0 (no term overlap with "PLACEHOLDER"). This is an artifact of placeholder data and not meaningful for threshold analysis.

---

## 7. Important Code Changes

### Excerpt 1: Evidence Phrase Matching
**File:** [`src/metrics.py`](file:///d:/Projects/PaperTrail/src/metrics.py#L22-L44)
```python
def passage_contains_evidence(passage: str, evidence_phrase: str) -> bool:
    """
    Checks if a passage contains the evidence phrase.
    Matching is whitespace-normalized and case-insensitive.
    """
    if not passage or not evidence_phrase:
        return False
    norm_passage = normalize_whitespace(passage).lower()
    norm_evidence = normalize_whitespace(evidence_phrase).lower()
    return norm_evidence in norm_passage
```

### Excerpt 2: Passage-Level Recall@k
**File:** [`src/metrics.py`](file:///d:/Projects/PaperTrail/src/metrics.py#L116-L141)
```python
def passage_recall_at_k(
    retrieved_passages: Sequence[str],
    evidence_phrase: str,
    k: int,
) -> float:
    """A retrieved chunk counts as a hit only if it contains the evidence_phrase."""
    if not evidence_phrase or k <= 0:
        return 0.0
    for passage in retrieved_passages[:k]:
        if passage_contains_evidence(passage, evidence_phrase):
            return 1.0
    return 0.0
```

### Excerpt 3: Score Distribution Statistics
**File:** [`src/metrics.py`](file:///d:/Projects/PaperTrail/src/metrics.py#L170-L196)
```python
def score_distribution_stats(scores: Sequence[float]) -> dict:
    """Computes count, min, max, mean, median for threshold analysis."""
    if not scores:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    mean = sum(sorted_scores) / n
    if n % 2 == 1:
        median = sorted_scores[n // 2]
    else:
        median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0
    return {"count": n, "min": sorted_scores[0], "max": sorted_scores[-1],
            "mean": mean, "median": median}
```

### Excerpt 4: Validator Evidence Grounding Check
**File:** [`scripts/validate_dataset.py`](file:///d:/Projects/PaperTrail/scripts/validate_dataset.py#L89-L150)
```python
def validate_evidence_on_page(item, project_root, papers_dir, page_text_cache):
    """Verifies evidence_phrase appears on expected pages of the PDF."""
    norm_evidence = _normalize_whitespace(evidence).lower()
    for page_num in expected_pages:
        page_text = page_text_cache.get((str(pdf_path), page_num), "")
        norm_page = _normalize_whitespace(page_text).lower()
        if norm_evidence in norm_page:
            found_on_page = True
            break
    if not found_on_page:
        errors.append(f"[{item_id}] Evidence phrase not found on expected pages")
```

---

## 8. Known Issues and Limitations

1. **Placeholder Dataset:** The v2 dataset contains only 2 placeholder items. Real evaluation questions with proper `evidence_phrase` values need to be authored by the user before meaningful passage-level metrics can be computed.

2. **Whitespace Normalization Scope:** The current normalization collapses all whitespace to single spaces. This handles most PDF extraction artifacts but may miss edge cases involving Unicode non-breaking spaces or zero-width characters in specific PDF renderers.

3. **No Refusal Threshold:** Score distributions are reported for analysis but no threshold is implemented. A future milestone should pick a threshold based on real data and implement refusal logic.

4. **Small Corpus Caveat:** The `sample_paper.pdf` is a 2-page mock document. Page-level metrics showing 100% are an artifact of the trivially small corpus. Passage-level metrics on real multi-page papers will provide more meaningful evaluation.

---

## 9. How to Reproduce

Run these commands from the project root (`d:\Projects\PaperTrail`) on Windows PowerShell:

### 1. Run the Full Test Suite (100 Tests)
```powershell
pytest -v
```

### 2. Validate the V2 Dataset
```powershell
python scripts/validate_dataset.py --dataset eval/dataset_v2.json
```

### 3. Run Evaluation on V1 Dataset (backward compatible)
```powershell
python scripts/run_eval.py --dataset eval/dataset.json
```

### 4. Run Evaluation on V2 Dataset
```powershell
python scripts/run_eval.py --dataset eval/dataset_v2.json
```

### 5. Verify Existing Features Still Work
```powershell
# Milestone 3 Ranked Retrieval
python -m src.cli sample_paper.pdf --retrieve "attention mechanism" --top-k 3

# Milestone 2 Keyword Search
python -m src.cli sample_paper.pdf --search "attention"

# Milestone 1 Full Text Extraction
python -m src.cli sample_paper.pdf --full-text
```

---

## 10. File-Level Change Summary

- **`eval/dataset_v2.json` (New):** V2 dataset with 2 placeholder items.
- **`eval/dataset_v2_schema.json` (New):** JSON Schema for v2 format.
- **`scripts/validate_dataset.py` (New):** Dataset validator with evidence grounding.
- **`tests/test_metrics_v2.py` (New):** 48 unit tests for all M3.5 functionality.
- **`src/metrics.py` (Modified):** +5 new functions, all existing preserved.
- **`src/__init__.py` (Modified):** +5 new exports.
- **`scripts/run_eval.py` (Modified):** Rewritten for v2 with backward compatibility.
