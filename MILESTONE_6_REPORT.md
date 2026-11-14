# PaperTrail -- Milestone 6 Report: LLM-as-a-Judge Answer Verification

**Date:** 2026-10-29  
**System:** PaperTrail (Evidence-First Research Paper Question-Answering System)  
**Report File:** `MILESTONE_6_REPORT.md`  
**Evaluation Source Artifact:** `eval/results/answers_20261024_140625.json`  
**Verification Result Artifact:** `eval/results/verify_20261029_082052.json`  
**Human-Readable Verification Audit:** `eval/verify_items.txt`  
**Evaluator Model:** `gemini-3.5-flash-lite` (via official `google-genai` SDK, temperature = 0)

---

## 1. Regressions & Surprises (Listed First)

1. **False Positive on `attn-10` (Judge Misread Flattened Table):**
   - **Question:** *"Which evaluation metric score was recorded on Section 23 of the WSJ Penn Treebank dataset when using the four-layer Transformer in a semi-supervised setup?"*
   - **Model Answer:** *"An F1 score of 92.7 was recorded on Section 23 of the WSJ Penn Treebank dataset when using the four-layer Transformer in a semi-supervised setup."*
   - **Citations Cited:** `['attention.pdf::p10::c1', 'attention.pdf::p10::c6']`
   - **Automated Metric in M5:** Grounding was marked `True` in Milestone 5 because the dataset's ground truth `evidence_phrase` (`"Transformer (4 layers) semi-supervised 92.7"`) appeared verbatim in `attention.pdf::p10::c6`.
   - **Judge Verdict:** Both cited chunks were evaluated as **`not_supported`**.
     - For `p10::c1`: Table 4 header only; no F1 score or model details present. This verdict is correct (the header chunk lacks scores).
     - For `p10::c6`: The judge noted: *"The cited passage associates the 92.7 score with Luong et al. (2015) using a multi-task setup, while the 4-layer Transformer in a semi-supervised setup is associated with a score of 92.1."*
   - **Correction:** The `not_supported` verdict on `p10::c6` is a **false positive caused by table flattening**. The columns of Table 4 are (Parser, Training, WSJ 23 F1). In the flattened chunk text (`92.1 / Transformer (4 layers) / semi-supervised / 92.7 / Luong et al. (2015) [23] / multi-task`), the row labels for the 92.1 entry fall outside this chunk. Checked against Table 4 on page 10 of attention.pdf: the Transformer (4 layers) semi-supervised row has F1 92.7; 92.1 belongs to the McClosky et al. and Vinyals & Kaiser semi-supervised rows; Luong et al. (2015) multi-task is 93.0, which falls outside the chunk. The model's answer is correct.

2. **Addresses-question check did not detect the known soft failures:**
   - All 40 answered items received `answers_question` (40/40, 100.0%), including the two unanswerable items (`attn-u07` and `bert-u03`).
   - The check flagged 0 of 2 known soft failures (attn-u07, bert-u03) because it sees only the question and answer, not the paper, so it cannot tell that the paper does not state the figure. Detecting this needs a different design that gives the judge the retrieved passages; out of scope for M6.
   - For `bert-u03`, the judge's own reason notes the answer describes the parameters "functionally rather than giving a numeric count", yet it still returned `answers_question`. The judge noticed the gap but did not act on it.

3. **Multi-Citation Partial Support on `drs-03`:**
   - `drs-03` cited 3 chunks (`drs.pdf::p3::c5`, `p3::c2`, `p3::c6`) to answer how residual mapping operations are mathematically defined.
   - The verifier judged all 3 chunks as **`partially_supported`**. Each cited chunk supported a subset of the composite answer (one defined $\mathcal{F}(x) := \mathcal{H}(x) - x$, another defined $y = \mathcal{F}(x, \{W_i\}) + x$, and the third defined $y = \mathcal{F}(x, \{W_i\}) + W_s x$), but none individually supported the entire multi-part definition. Together, however, the three citations fully support the response.

---

## 2. Evaluation Setup & Methodology

- **Input Scope:** Frozen eval results from Milestone 5 (`eval/results/answers_20261024_140625.json`). The answering pipeline was not re-run.
- **Evaluated Set:** Exactly the 40 answered items from the benchmark (26 refusals were skipped).
- **Two Decoupled Verification Checks:**
  1. `verify_citation(question, answer, cited_chunk_text)`: Evaluates whether a cited passage chunk substantiates the answer claim. Verdicts $\in$ `{supported, partially_supported, not_supported}` + 1-sentence reason.
  2. `verify_addresses_question(question, answer)`: Evaluates whether the answer directly addresses the question asked. Verdicts $\in$ `{answers_question, states_not_in_paper, off_topic_or_wrong_figure}` + 1-sentence reason.
- **Judge Prompts:** Frozen prior to running evaluation, strict JSON output required. Parse failures are tracked explicitly and never silently defaulted.
- **Execution & Rate Limiting:** `gemini-3.5-flash-lite`, delay between calls = 5.0s, caching enabled via `.cache/llm/`.

---

## 3. Verification Metrics & Summary

### API Execution Statistics
- **Total Answered Items in Source:** 40
- **Total Answered Items Evaluated:** 40
- **Total Citations Evaluated:** 50
- **Live API Calls Made:** 84
- **Cache Hits:** 6
- **Quota Hit:** False (0 quota errors encountered)
- **JSON Parse Failures:** **0** across all 90 evaluations (40 addresses checks + 50 citation checks)

### Citation Support Verdicts (50 citations across 40 items)

| Verdict | Count | Percentage | Description |
|---|:---:|:---:|---|
| **`supported`** | **43** | **86.0%** | The cited passage explicitly and fully substantiates the claim in the answer. |
| **`partially_supported`** | **4** | **8.0%** | The cited passage supports part of the claim or context, but leaves aspects unverified. |
| **`not_supported`** | **3** | **6.0%** | The cited passage does not contain information supporting the claim or is incorrect. |
| **Parse Failures** | **0** | **0.0%** | Strict JSON decoding failures. |

> [!NOTE]
> At least 1 of the 3 `not_supported` verdicts is a confirmed false positive (`attn-10` `p10::c6`, caused by table flattening; see Section 1, item 1). Raw judge counts are preserved as-is.

#### Complete Breakdown of Non-Supported Citations (7 citations across 4 items):
1. **`attn-10` (`attention.pdf::p10::c1` - `not_supported`):**
   *Reason:* The cited passage only shows the header of Table 4 and lacks the specific F1 score and model details mentioned in the claim.
2. **`attn-10` (`attention.pdf::p10::c6` - `not_supported`, confirmed false positive):**
   *Reason:* The cited passage associates the 92.7 score with Luong et al. (2015) using a multi-task setup, while the 4-layer Transformer in a semi-supervised setup is associated with a score of 92.1.
3. **`attn-u07` (`attention.pdf::p3::c2` - `not_supported`):**
   *Reason:* The cited passage discusses the general model architecture with N = 6 layers and does not contain any information about the English constituency parsing task or a 4-layer transformer.
4. **`drs-02` (`drs.pdf::p1::c6` - `partially_supported`):**
   *Reason:* The cited passage supports the claims about degradation not being caused by overfitting and higher training error, but it does not mention Batch Normalization, vanishing gradients, or exponentially low convergence rates.
5. **`drs-03` (`drs.pdf::p3::c5` - `partially_supported`):**
   *Reason:* The cited passage supports the definition of y = F(x, {Wi}) + x and the representation of F(x, {Wi}) as the residual mapping, but it does not mention the alternative definitions involving Wsx or F(x) := H(x) - x.
6. **`drs-03` (`drs.pdf::p3::c2` - `partially_supported`):**
   *Reason:* The cited passage supports the definition of F(x) := H(x) - x and the original function as F(x) + x, but does not mention the specific notation {Wi} or the dimensional matching formula involving Wsx.
7. **`drs-03` (`drs.pdf::p3::c6` - `partially_supported`):**
   *Reason:* The cited passage supports the definition of $y = F(x, \{Wi\}) + Wsx$ and mentions $x$ and $F$, but it does not explicitly define the general building block formula $y = F(x, \{Wi\}) + x$ or the alternative $F(x) := H(x) - x$ as stated in the answer.

---

### Addresses-Question Verdicts (40 items)

| Verdict | Count | Percentage |
|---|:---:|:---:|
| **`answers_question`** | **40** | **100.0%** |
| **`states_not_in_paper`** | **0** | **0.0%** |
| **`off_topic_or_wrong_figure`** | **0** | **0.0%** |
| **Parse Failures** | **0** | **0.0%** |

#### Items Not Answering Question Among 45 Answerables:
- **None (0 items).** Every answerable item that generated an answer directly attempted to answer the prompt.

#### Unanswerable Items Answered by Model (2 items):
1. **`attn-u07`:**
   - *Question:* "How many total layers were used in the decoder stack for the constituency parsing task?"
   - *Addresses Verdict:* `answers_question`
   - *Reason:* *"The answer directly provides the number of layers used in the decoder stack for the constituency parsing task based on the text."*
   - *Citation Verdicts:* `p9::c3` (`supported` for 4-layer transformer), `p3::c2` (`not_supported` for parsing decoder).
2. **`bert-u03`:**
   - *Question:* "How many total parameters were added to the architecture for the multi-choice classification head on SWAG?"
   - *Addresses Verdict:* `answers_question`
   - *Reason:* *"The answer explains the parameters added for the multi-choice classification head on SWAG, even though it describes them functionally rather than giving a numeric count."*
   - *Citation Verdicts:* `p7::c6` (`supported`).

---

## 4. Sanity Check on Milestone 5 Manual Audit

The verifier matched the M5 manual audit on 6 of 9 items, partly disagreed on 1 (drs-03), and missed 2 (attn-u07, bert-u03). Two of the 6 agreements (bert-08, drs-13) are table items, so see the table-reading limitation in Section 5.

| Item ID | Target Issue | Human Verdict in M5 | Verifier Verdict & Notes | Agreement? |
|---|---|---|---|:---:|
| `attn-07` | Evidence phrase boundary split | Supported (36M sentences in `p7::c5`) | `supported` (*"The cited passage explicitly states that the English-French dataset consisted of 36M sentences."*) | **Agreed** |
| `bert-03` | Evidence phrase location mismatch | Supported (1M steps in `p14::c2`) | `supported` (*"The cited passage explicitly states that 'BERT was trained for 1M steps'."*) | **Agreed** |
| `bert-08` | Table text variation | Supported (Concat Last Four 96.1 in `p9::c7`) | `supported` (*"The cited passage shows in Table 7 that 'Concat Last Four Hidden' achieved the highest score of 96.1 among the feature-based BERTBASE approaches for CoNLL-2003 Named Entity Recognition."*) | **Agreed** |
| `bert-14` | Evidence phrase location mismatch | Supported (Concat last 4 layers in `p16::c5`) | `supported` (*"The cited passage explicitly states that for the feature-based approach, they concatenate the last 4 layers of BERT."*) | **Agreed** |
| `drs-01` | Evidence phrase location mismatch | Supported (3.57% top-5 error in `p2::c4`) | `supported` (*"The cited passage explicitly states that their ensemble achieved a 3.57% top-5 error on the ImageNet test set."*) | **Agreed** |
| `drs-03` | Multiple building block equations | Supported across 3 cited chunks | `partially_supported` on each of `p3::c5`, `p3::c2`, `p3::c6` (individually each chunk supports part of the composite answer). | **Partial disagreement (per-chunk judging understates composite support)** |
| `drs-13` | Table phrasing variation | Supported (11.7% in `p12::c8`) | `supported` (*"The cited passage explicitly states that with dense and multi-scale testing, the ResNet-101 has an error of 11.7% using ground truth classes."*) | **Agreed** |
| `attn-u07` | Unanswerable answered | Soft failure (supported claims, but paper doesn't give parsing decoder layers) | Addresses: `answers_question`; Citations: `supported` on `p9::c3` (4-layer parsing model), `not_supported` on `p3::c2` (base model decoder not relevant to parsing task). | **Not flagged (soft failure missed by the addresses check)** |
| `bert-u03` | Unanswerable answered | Soft failure (vector parameter description supported, but no parameter count given) | Addresses: `answers_question`; Citations: `supported` on `p7::c6` (describes choice-scoring vector, but does not provide numerical count). | **Not flagged (soft failure missed by the addresses check)** |

---

## 4b. Hand-Labelled Check of Judge Verdicts

The project owner labelled 14 citation checks (9 chosen as hard cases, 5 chosen at random) as supported / partial / not, without seeing the judge's verdicts. Table rows were checked against the PDF. Raw sheet: `eval/judge_labels_blind.csv`.

- Judge and labeller agreed on 11 of 14 rows (7 of 9 targeted rows, 4 of 5 random rows).
- Disagreements (3):
  - `attn-10` `p10::c6`: judge `not_supported`, labeller `supported`. The judge misread the flattened table (see Section 1, item 1).
  - `attn-u07` `p3::c2`: judge `not_supported`, labeller `partial`. The chunk supports the decoder N = 6 claim but not the 4-layer parsing claim; the judge penalised the chunk for not covering the whole answer.
  - `attn-04` `p6::c1`: judge `supported`, labeller `not`. The chunk gives O(1) maximum path length, but not the claim that this makes long-range dependencies easier to learn (borderline row).
- All 3 disagreements are borderline or table rows. No disagreement was on a plain, clear-cut row.
- This is one labeller and 14 rows, deliberately weighted toward hard cases, so the 11 of 14 figure is not an estimate of the judge's overall accuracy.

---

## 5. Limitations of Verification System

1. **Same-Model Family Bias:**
   - Both the generator and the judge use `gemini-3.5-flash-lite`. Models from the same family share pre-training biases, tokenization artifacts, and stylistic preferences. A cross-family judge would provide greater independence.
2. **Single-Run Stochasticity:**
   - Evaluated on a single run with temperature = 0. While zero-temperature generation is largely deterministic, minor API variations or edge cases were not multi-sampled.
3. **Per-Citation vs. Composite Support Evaluation:**
   - The citation verifier assesses each cited chunk independently against the full answer (`drs-03`). When an answer synthesizes information across multiple passages, each individual chunk receives `partially_supported`, underestimating aggregate document-level support.
4. **Lack of Human Calibration on Edge Cases:**
   - The judge has been validated only by the small hand-labelled check in Section 4b (14 rows, one labeller), not by a multi-annotator agreement study (e.g. Cohen's $\kappa$).
5. **Table Flattening Corrupts Numeric Verdicts:**
   - The judge misreads numeric tables when the PDF extractor flattens them, because row/column boundaries are lost. Table-based verdicts (attn-10, bert-08, drs-13) are unreliable in either direction — a `supported` verdict on a table chunk may be a lucky column-order match, and a `not_supported` may be a false positive (as confirmed for attn-10 `p10::c6`). This is the same table-flattening issue identified in Milestone 5's retrieval diagnostics (drs-11, bert-11). No fix is applied in M6.
6. **Addresses-Question Check Lacks Document Context:**
   - The addresses-question check has no recall on the known soft failures (0 of 2), because it never sees the paper text.
7. **Small, single-labeller calibration.**
   - The hand-labelled check covers 14 of 50 citation checks, by one labeller, and was weighted toward hard cases. It is not a formal agreement study.

---

## 6. Reproducibility

- **Answers Run Verified:** `eval/results/answers_20261024_140625.json`
- **Verification Script:** `scripts/run_verify_eval.py`
- **Raw Result JSON:** `eval/results/verify_20261029_082052.json`
- **Human-Readable Audit:** `eval/verify_items.txt`
- **Hand-Labelled Sheet:** `eval/judge_labels_blind.csv`
- **Execution Command:**
  ```powershell
  python scripts/run_verify_eval.py --call-delay 5
  ```
- **Test Suite Verification:**
  ```powershell
  pytest -v
  ```
  *(159 passed, 1 warning in 7.02s)*

