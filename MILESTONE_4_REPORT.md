# PaperTrail -- Milestone 4 Report: Dense Vector & Hybrid Retrieval with Honest Benchmarking

**Date:** 2026-10-18  
**System:** PaperTrail (Evidence-First Research Paper Question-Answering System)  
**Report File:** `MILESTONE_4_REPORT.md`  

---

## 1. Milestone Summary

### Goal
Implement dense embedding retrieval (`BAAI/bge-small-en-v1.5` via `sentence-transformers`) and hybrid retrieval using Reciprocal Rank Fusion ($k=60$), then compare them side-by-side against BM25 on the unified multi-document evaluation dataset (`eval/dataset_all.json`). All metrics and win/loss tradeoffs are reported with strict scientific honesty without inflating scores.

### What Functionality Was Added
- **Dense Vector Retrieval Engine (`src/dense_retriever.py`):**
  - Uses `sentence-transformers` models (default: `BAAI/bge-small-en-v1.5`, fully configurable).
  - Encodes passages into normalized vector representations and ranks candidates via cosine similarity.
  - Implements on-disk caching of passage embeddings in `.cache/embeddings/` keyed by a SHA-256 hash of the passage texts, passage IDs, and model name. Subsequent evaluation runs bypass inference entirely.
  - Returns standard, immutable `RetrievalResult` dataclasses identical to the BM25 retriever.
  - Automatically prepends the recommended BGE query prefix (`"Represent this sentence for searching relevant passages: "`) to query strings when evaluating BGE model variants.
- **Reciprocal Rank Fusion Hybrid Retriever (`src/hybrid_retriever.py`):**
  - Combines lexical rankings (BM25) and semantic vector rankings (Dense) using standard Reciprocal Rank Fusion (RRF):
    $$RRF\_score(d) = \sum_{m \in \{\text{bm25}, \text{dense}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
    with default smoothing constant $k = 60$ and configurable weights ($w_{\text{bm25}}=1.0, w_{\text{dense}}=1.0$).
  - Pulls a candidate pool from both systems and performs deterministic tie-breaking.
- **Evaluation Harness Extensions (`scripts/run_eval.py`):**
  - Added CLI flag `--retriever {bm25,dense,hybrid}` (default: `bm25`).
  - Added configuration flags `--model-name`, `--cache-dir`, `--rrf-k`, `--weight-bm25`, `--weight-dense`.
  - Persists retriever type, model name, and fusion parameters in the evaluation summary JSON.
- **Comparative Analysis Tool (`scripts/compare_runs.py`):**
  - Accepts two or more evaluation results JSON files and formats a side-by-side terminal comparison table across Page-Level and Passage-Level Recall@1, 3, 5 and MRR.
  - Identifies and lists specific queries that one retriever gets right while the other gets wrong.
- **Comprehensive Unit Tests (`tests/test_dense_hybrid.py`):**
  - 18 fast, offline unit tests using a lightweight `StubEmbedder` covering ranking order, `top_k` bounds, empty query handling, disk cache hits, BGE query prefixing, RRF mathematics, and run comparison.
- **Pinned Dependency Manifest (`requirements.txt`):**
  - Pinned exact versions for all project dependencies.

### What Existing Functionality Was Modified
- **`src/__init__.py`:** Exported new dense and hybrid classes and convenience functions.
- **`scripts/run_eval.py`:** Added multi-retriever dispatching and metadata tracking while maintaining full backward compatibility.
- **`requirements.txt`:** Pinned exact dependency versions.

### Preserved Invariants
- `eval/dataset_all.json` was **not modified**.
- Pre-existing BM25 retrieval logic, page-bounded chunking, and scoring were **not modified**.
- All 112 pre-existing unit and integration tests remain 100% passing.

---

## 2. Files Changed

| File Path (Relative) | Status | Specific Changes and Rationale |
|---|---|---|
| `src/dense_retriever.py` | **Created** | Dense vector retrieval engine with cosine similarity, BGE query prefixing, and SHA-256 disk caching. |
| `src/hybrid_retriever.py` | **Created** | Hybrid retrieval engine with Reciprocal Rank Fusion ($k=60$) and per-retriever weighting. |
| `scripts/compare_runs.py` | **Created** | CLI utility for side-by-side tabular comparison of run JSON files and pairwise win/loss item auditing. |
| `tests/test_dense_hybrid.py` | **Created** | 18 fast, offline unit tests utilizing `StubEmbedder` for deterministic verification. |
| `src/__init__.py` | **Modified** | Exported `DenseRetriever`, `retrieve_dense`, `HybridRetriever`, `reciprocal_rank_fusion`, and `retrieve_hybrid`. |
| `scripts/run_eval.py` | **Modified** | Added `--retriever`, model tracking in summary, and hybrid parameter routing. |
| `requirements.txt` | **Modified** | Pinned dependencies: `pymupdf==1.28.2`, `pytest==9.1.1`, `rank_bm25==0.2.2`, `sentence-transformers==6.1.0`. |
| `eval/dataset_all.json` | **Unchanged** | Benchmark ground truth preserved without modifications. |

---

## 3. Implementation Details

### Dense Vector Retrieval & Caching Architecture
Dense retrieval maps passage chunks and questions into a shared 384-dimensional continuous semantic vector space using `BAAI/bge-small-en-v1.5`.

```
Query -> [Prefix Prepend if BGE] -> [Dense Embedder] ---------\
                                                                -> [Cosine Similarity] -> [Sort & Top-K] -> RetrievalResult
Passages -> [Check Hash Key in .cache/] -> [Load or Embed] ---/
```

1. **Query Formatting:** For BGE models, questions receive the recommended prefix: `"Represent this sentence for searching relevant passages: "`. Passages are encoded without prefix.
2. **Deterministic Caching:** Passages are hashed via SHA-256 over model identifier + passage IDs + text content. Cached vectors are serialized via `numpy.save()` into `.cache/embeddings/<hash>.npy`. On subsequent runs, embedding inference for the corpus is 0 ms.
3. **Cosine Similarity:** Computed via unit-normalized dot products:
   $$\text{sim}(\mathbf{p}, \mathbf{q}) = \frac{\mathbf{p} \cdot \mathbf{q}}{\|\mathbf{p}\|_2 \|\mathbf{q}\|_2}$$

### Reciprocal Rank Fusion (RRF) Architecture
BM25 and Dense produce scores on completely different scales (BM25: uncalibrated logits $[0, \infty)$; Dense: cosine similarities $[-1, 1]$). Standard score normalization (min-max or z-score) is sensitive to outliers and distribution shifts. RRF bypasses score calibration by fusing rank positions:

$$RRF\_score(d) = \frac{w_{\text{bm25}}}{60 + \text{rank}_{\text{bm25}}(d)} + \frac{w_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(d)}$$

- Candidate passages retrieved by either system enter the candidate pool.
- Items appearing in both rankings receive accumulated reciprocal rank boosts.
- Ties are broken deterministically by best individual rank, followed by lexical `passage_id`.

---

## 4. Test Results

### Test Command Executed:
```powershell
pytest -v
```

### Summary:
- **Total Tests Collected:** 130
- **Passed:** 130
- **Failed:** 0
- **Duration:** 1.15 seconds
- **Offline Guarantee:** All dense and hybrid unit tests run against `StubEmbedder` without network calls or GPU dependencies.

### Module Breakdown:
1. `tests/test_extractor.py`: 15 passed (including ligature normalization)
2. `tests/test_search.py`: 16 passed
3. `tests/test_chunker.py`: 6 passed
4. `tests/test_retriever.py`: 12 passed
5. `tests/test_metrics.py`: 5 passed
6. `tests/test_metrics_v2.py`: 58 passed
7. `tests/test_dense_hybrid.py`: 18 passed

---

## 5. Requirements Checklist

| Requirement | Status | Evidence / Verification |
|---|---|---|
| **1. `src/dense_retriever.py` with `BAAI/bge-small-en-v1.5`, cosine similarity, disk cache, BGE prefix, `RetrievalResult`** | **PASS** | Implemented in `src/dense_retriever.py`. Verified by 10 unit tests and end-to-end evaluation. |
| **2. `src/hybrid_retriever.py` with RRF ($k=60$) and optional weights** | **PASS** | Implemented in `src/hybrid_retriever.py`. Verified by hand-computed mathematical unit tests. |
| **3. `scripts/run_eval.py` `--retriever {bm25,dense,hybrid}` and `scripts/compare_runs.py` side-by-side tool** | **PASS** | Evaluated on `eval/dataset_all.json`; `compare_runs.py` outputs side-by-side metric tables and item-by-item win/loss analysis. |
| **4. Offline unit tests with stub embedder** | **PASS** | 18 tests in `tests/test_dense_hybrid.py` run in < 0.15s without downloading models. |
| **5. `MILESTONE_4_REPORT.md` with honest numbers on `eval/dataset_all.json`, pinned `requirements.txt`** | **PASS** | Complete honest comparative analysis below; dependencies pinned in `requirements.txt`. |

---

## 6. Benchmark Evaluation Results (`eval/dataset_all.json`)

The evaluation was executed across all 66 items (45 answerable, 21 unanswerable) across three major papers: `attention.pdf`, `bert.pdf`, and `drs.pdf`.

### Overall Comparative Metrics

| Metric | BM25 Baseline | Dense (`bge-small`) | Hybrid (RRF $k=60$) | Delta (Hybrid vs BM25) |
|---|:---:|:---:|:---:|:---:|
| **Page Recall@1** | **71.1%** (0.7111) | 57.8% (0.5778) | 68.9% (0.6889) | -2.2% |
| **Page Recall@3** | **93.3%** (0.9333) | 75.6% (0.7556) | 86.7% (0.8667) | -6.6% |
| **Page Recall@5** | **95.6%** (0.9556) | 88.9% (0.8889) | 91.1% (0.9111) | -4.5% |
| **Page MRR** | **0.8224** | 0.6770 | 0.7788 | -0.0436 |
| **Passage Recall@1** | **44.4%** (0.4444) | 24.4% (0.2444) | 42.2% (0.4222) | -2.2% |
| **Passage Recall@3** | **71.1%** (0.7111) | 44.4% (0.4444) | 60.0% (0.6000) | -11.1% |
| **Passage Recall@5** | **73.3%** (0.7333) | 62.2% (0.6222) | 68.9% (0.6889) | -4.4% |
| **Passage MRR** | **0.5808** | 0.3806 | 0.5356 | -0.0452 |

### Per-Document Breakdown

#### 1. `attention.pdf` (n=15 answerable)
- **BM25:** Page R@1: 73.3%, Page MRR: 0.8667 | Pass R@1: 60.0%, Pass MRR: 0.7306
- **Dense:** Page R@1: 73.3%, Page MRR: 0.7911 | Pass R@1: 40.0%, Pass MRR: 0.5595
- **Hybrid:** **Page R@1: 86.7%**, **Page MRR: 0.9000** | **Pass R@1: 66.7%**, **Pass MRR: 0.7417**
*(On `attention.pdf`, Hybrid outperformed both BM25 and Dense across all metrics!)*

#### 2. `bert.pdf` (n=15 answerable)
- **BM25:** **Page R@1: 80.0%**, **Page MRR: 0.8762** | **Pass R@1: 40.0%**, **Pass MRR: 0.5540**
- **Dense:** Page R@1: 46.7%, Page MRR: 0.5911 | Pass R@1: 13.3%, Pass MRR: 0.2411
- **Hybrid:** Page R@1: 66.7%, Page MRR: 0.7456 | Pass R@1: 26.7%, Pass MRR: 0.3867

#### 3. `drs.pdf` (n=15 answerable)
- **BM25:** **Page R@1: 60.0%**, **Page MRR: 0.7244** | **Pass R@1: 33.3%**, Pass MRR: 0.4578
- **Dense:** Page R@1: 53.3%, Page MRR: 0.6489 | Pass R@1: 20.0%, Pass MRR: 0.3411
- **Hybrid:** Page R@1: 53.3%, Page MRR: 0.6907 | **Pass R@1: 33.3%**, **Pass MRR: 0.4784**

---

## 7. Honest Analysis: Where Dense & Hybrid Win vs. Where They Lose

### Why Dense Retrieval Underperformed BM25 on This Benchmark
Dense retrieval achieved **24.4% Passage Recall@1** vs. **44.4% for BM25**. This is an expected and authentic result on academic research benchmarks for several distinct reasons:

1. **Exact Hyperparameters, Metrics, and Numbers:**
   - Academic benchmark questions frequently inquire about exact figures: `"36M sentences"`, `"100,000 steps or 12 hours"`, `"30,000 token vocabulary"`, `"3.6 billion FLOPs"`, `"1.7M parameters"`, `"3.57% error"`.
   - Small dense embedding models (`bge-small-en-v1.5`, 384 dimensions) map all number-related and training-related sentences to nearby regions in latent space. They lack token-level discrimination to differentiate between `"12 hours"` vs `"24 hours"`, or `"100,000 steps"` vs `"1,000,000 steps"`.
   - BM25 indexes exact character n-grams and tokens, matching exact numerals with high IDF weights.
2. **Dense Semantic Drift:**
   - On `bert.pdf`, Dense suffered significantly (Passage R@1 of 13.3% vs 40.0% for BM25). For questions asking about specific ablations (e.g. `[bert-11]` SST-2 accuracy without NSP), the dense embedder repeatedly selected general introductory passages defining pre-training rather than the specific tabular results passage.

### Where Dense Retrieval Won Over BM25
Dense retrieval succeeded on questions requiring conceptual or paraphrased matching where users did not use the exact phrasing from the paper:

- **`[drs-02]` Training Degradation:**
  - *Question:* "What is the primary underlying cause identified when increased depth leads to higher training error in plain deep networks?"
  - *Evidence:* "degradation problem has been exposed"
  - *Outcome:* **Dense hit at Rank 1**; **BM25 failed** because the question used descriptive phrasing rather than the keyword `"degradation"`.
- **`[attn-10]` Parsing Evaluation Score:**
  - *Question:* "Which evaluation metric score was recorded on Section 23 of the WSJ Penn Treebank dataset when using the four-layer Transformer in a semi-supervised setup?"
  - *Evidence:* "Transformer (4 layers) semi-supervised 92.7"
  - *Outcome:* **Dense hit at Rank 1**; **BM25 missed** because query terms were dispersed across the text.
- **`[drs-12]` Region of Interest Pooling Layer:**
  - *Question:* "Which specific convolutional feature layer in ResNet-101 is selected to perform Region of Interest pooling during object detection processing?"
  - *Evidence:* "is performed before conv5 1"
  - *Outcome:* **Dense hit at Rank 1**; **BM25 missed** due to terminology variations between "Region of Interest pooling" and "RoI".

### Why Equal-Weight Hybrid Diluted BM25 Top-1 Precision
In Hybrid retrieval ($w_{\text{bm25}}=1.0, w_{\text{dense}}=1.0$):
- Hybrid achieved a notable win on `attention.pdf`: **Page Recall@1 rose to 86.7%** (vs 73.3% for BM25) and **Passage Recall@1 rose to 66.7%** (vs 60.0% for BM25).
- However, on `bert.pdf` and `drs.pdf`, because dense retrieval assigned high similarity scores to irrelevant background paragraphs for numerical questions, equal-weight RRF occasionally pushed the correct BM25 top-1 chunk down to rank 2 or 3 (e.g. `[bert-04]`, `[bert-05]`, `[drs-08]`).
- As instructed, no hyperparameter tuning was performed to artificially inflate the hybrid numbers. In production environments, applying non-equal retriever weighting (e.g. $w_{\text{bm25}}=1.5, w_{\text{dense}}=0.8$) or query-intent routing (lexical vs. semantic) is recommended.

---

## 8. Summary of Results Artifacts

All benchmark runs and comparison files are persisted and reproducible:
- **BM25 Run:** `eval/results/20261018_080436.json`
- **Dense Run:** `eval/results/20261018_080520.json`
- **Hybrid Run:** `eval/results/20261018_080554.json`
- **Comparison Command:**
  ```powershell
  python scripts/compare_runs.py eval/results/20261018_080436.json eval/results/20261018_080520.json eval/results/20261018_080554.json
  ```
