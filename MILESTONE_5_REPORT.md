# PaperTrail -- Milestone 5 Report: RAG Generation & Evidence-Grounded Answering

**Date:** 2026-10-24  
**System:** PaperTrail (Evidence-First Research Paper Question-Answering System)  
**Report File:** `MILESTONE_5_REPORT.md`  
**Evaluation Source Artifact:** `eval/results/answers_20261024_140625.json`

---

## 1. Setup & Methodology

- **Model:** `gemini-3.5-flash-lite` (via official `google-genai` SDK).
- **Retrieval Engine:** Lexical BM25 (`rank_bm25.BM25Okapi`), returning top-5 ranked passage chunks per document.
- **Prompt Strategy:** The prompt was written once and not tuned or iteratively optimized. It enforces a strict instruction: if the retrieved passages do not contain sufficient evidence to answer the question, the model must refuse and return `"answerable": false` with an empty answer string and empty citations list.
- **Evaluation Benchmark:** Single execution run on all 66 items in `eval/dataset_all.json` across three foundational papers (`attention.pdf`, `bert.pdf`, `drs.pdf`):
  - 45 answerable items
  - 21 unanswerable items
- **Note on Initial Model Pilot:** `gemini-3.6-flash` was originally piloted on a 19-item subset. However, Google AI Studio's free tier imposes a cap of 5 requests per minute and 20 requests per day for `gemini-3.6-flash` (shown on the Google AI Studio rate-limit page). The pilot was promptly abandoned to avoid rate-limit exhaustion, and all benchmark results in this report represent the complete 66-item run executed with `gemini-3.5-flash-lite` (free tier: 15 requests per minute, 500 requests per day, handled via client-side delay and exponential backoff). The 19-item pilot run was discarded and is not included in the results.

---

## 2. Benchmark Metrics

The table below presents the evaluation results from the run `answers_20261024_140625.json`.

> **Important Clarification on Citation Grounding:**  
> The **Citation Grounding** metric checks whether the NFKC-normalized `evidence_phrase` from the dataset appears verbatim as a substring inside any of the passages cited by the model. It measures **retrieval/citation evidence alignment**, not semantic answer correctness.

| Metric | Overall (All 66 Items) | Retrieval Hit (Top-5 Hit, n=33) | Retrieval Miss (Top-5 Miss, n=33) |
|---|:---:|:---:|:---:|
| **Total Items** | 66 | 33 | 33 |
| **Unanswerable Items** | 21 | 0 | 21 |
| **Refusals on Unanswerables (True Negatives)** | **90.5%** (19 / 21) | N/A (0 unanswerables in Hit) | **90.5%** (19 / 21) |
| **Answerable Items** | 45 | 33 | 12 |
| **False Refusal Rate (Answerable Refused)** | **15.6%** (7 / 45) | **0.0%** (0 / 33) | **58.3%** (7 / 12) |
| **Citation Validity Rate (Cited IDs in Top-5 Pool)** | **100.0%** (40 / 40) | **100.0%** (33 / 33) | **100.0%** (7 / 7) |
| **Citation Grounding Rate (Evidence Phrase in Cited Chunk)** | **81.6%** (31 / 38) | **93.9%** (31 / 33) | **0.0%** (0 / 5) |
| **Parse Failure Rate (JSON Decoding Failures)** | **0.0%** (0 / 66) | **0.0%** (0 / 33) | **0.0%** (0 / 33) |

### Key Takeaways from Metrics:
1. **Zero Hallucinated Passage IDs:** In all 40 generated answers with citations, 100% of the cited passage identifiers were members of the candidate pool passed in the context.
2. **Hit vs. Miss Divide:** When BM25 retrieval successfully places the evidence passage into the top-5 candidates (**Retrieval Hit**, n=33):
   - False refusal rate drops to **0.0%** (0/33).
   - Citation grounding reaches **93.9%** (31/33).
3. **Retrieval Miss Behavior:** When the evidence passage is absent from the top 5 candidates (**Retrieval Miss**, n=33):
   - For unanswerable items (n=21), the model correctly refuses **90.5%** of the time (19/21).
   - For answerable items where retrieval failed (n=12), the model refuses **58.3%** (7/12). All 7 ungrounded-but-answered items were found to be supported by their cited chunks, so the automated grounding rate (81.6%) is a lower bound caused by the single-phrase evidence labels (different wording, other chunk, chunk boundary); the remaining failures are the two unanswerable items answered (attn-u07, bert-u03).

**Manual Audit (not an automated metric):**  
After reading raw chunk text in `eval/inspect_items.txt`, all 7 ungrounded-but-answered items were supported by their cited chunks, and every cited chunk was inside the top-5 pool, so at least 38 of 45 answerable items (33 phrase-matched hits + 5 supported on phrase-miss) had supporting text in the top 5. The 73.3% / 33-of-45 ceiling is evidence-phrase based and understates this. The 38/45 figure is a manual audit finding, not an automated metric (headline automated metrics stay exactly as measured: 19/21 refusals, 7/45 false refusals, 31/38 grounding).

---

## 3. Manual Review of 9 Flagged Items

Nine specific items from the run were selected for deep manual inspection: items flagged with ungrounded citations (`attn-07`, `bert-03`, `bert-08`, `drs-01`, `drs-03`, `bert-14`, `drs-13`) and unanswerable items answered by the model (`attn-u07`, `bert-u03`). 

The analysis below evaluates the question, the model answer, excerpts of the cited passages, the target evidence phrase, and an honest verdict based **strictly on the passage text**, without guessing.

| Item ID | Question | Model Answer | Cited Passage Excerpt | Target Evidence Phrase | Verdict & Detailed Analysis |
|---|---|---|---|---|---|
| `attn-07` | How many sentences were contained in the English-French training corpus? | The English-French dataset consisted of 36M sentences. | `[attention.pdf::p7::c5]`: *"2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together..."* | `WMT 2014 English-French dataset consisting of 36M sentences` | Supported. The cited chunk contains "2014 English-French dataset consisting of 36M sentences"; the evidence phrase spans the p7::c4/c5 boundary, so no single chunk can match it. |
| `bert-03` | How many total pre-training updates were executed during the initial training phase? | BERT was trained for 1M steps during pre-training. | `[bert.pdf::p14::c2]`: *"... • GPT was trained for 1M steps with a batch size of 32,000 words; BERT was trained for 1M steps with a batch size of 128,000 words."* | `for 1,000,000 steps` | Supported. p14::c2 states "BERT was trained for 1M steps"; the evidence phrase is in p13::c3. |
| `bert-08` | Which combination of hidden layer outputs yielded the strongest F1 score in the feature extraction experiment for entity recognition? | The combination of hidden layer outputs that yielded the strongest F1 score in the feature extraction experiment for entity recognition was the concatenation of the last four hidden layers, which achieved a score of 96.1. | `[bert.pdf::p9::c7]`: *"Feature-based approach (BERTBASE) ... Concat Last Four Hidden 96.1 ... Table 7: CoNLL-2003 Named Entity Recognition results."* | `token representations from the top four hidden` | Supported. p9::c7 contains Table 7 "Concat Last Four Hidden 96.1"; the phrase is in p9::c8. |
| `drs-01` | What standard evaluation error rate was achieved on ImageNet by an ensemble of these deep architectures? | An ensemble of these deep architectures achieved a 3.57% top-5 error on the ImageNet test set. | `[drs.pdf::p2::c4]`: *"Our ensemble has 3.57% top-5 error on the ImageNet test set, and won the 1st place in the ILSVRC 2015 classification competition."* | `ensemble of these residual nets achieves 3.57% error` | Supported. p2::c4 states "Our ensemble has 3.57% top-5 error on the ImageNet test set"; the phrase is in p1::c2. |
| `drs-03` | How are residual mapping operations mathematically defined relative to the input and output functions? | The residual mapping to be learned is represented by the function F(x, {Wi}), where x and y are the input and output vectors, and it is defined within a building block as y = F(x, {Wi}) + x ... Alternatively, the residual function is defined as F(x) := H(x) - x... | `[drs.pdf::p3::c5]`: *"3.2 Identity Mapping by Shortcuts ... y = F(x, {Wi}) + x. (1)"*<br>`[drs.pdf::p3::c2]`: *"3.1 Residual Learning ... explicitly let these layers approximate a residual function F(x) := H(x) −x."*<br>`[drs.pdf::p3::c6]`: *"...y = F(x, {Wi}) + Wsx. (2)"* | `underlying mapping as H(x), we let the stacked nonlinear` | Supported. The cited chunks contain Eq. 1, Eq. 2 and F(x) := H(x) - x; the phrase is in p2::c1. |
| `bert-14` | Which layer activations are extracted and concatenated to feed into the BiLSTM classifier for the feature-based CoNLL-2003 NER approach? | For the feature-based approach, the last 4 layers of BERT are extracted and concatenated. | `[bert.pdf::p16::c5]`: *"Table 8: Ablation over different masking strategies... For the feature-based approach, we concatenate the last 4 layers of BERT as the features, which was shown to be the best approach in Section 5.3."* | `Concat Last Four Hidden` | Supported. p16::c5 states "For the feature-based approach, we concatenate the last 4 layers of BERT as the features"; the phrase is in p9::c7. |
| `drs-13` | What localization error percentage was reached on ground truth classes when using dense and multi-scale testing on ResNet-101? | When using dense and multi-scale testing with ResNet-101, the localization error on ground truth classes is 11.7%. | `[drs.pdf::p12::c8]`: *"With dense (fully convolutional) and multi-scale testing, our ResNet-101 has an error of 11.7% using ground truth classes."* | `RPN ResNet-101 dense 11.7` | Supported. p12::c8 states 11.7% using ground truth classes; the phrase is in p12::c2. |
| `attn-u07` | How many total layers were used in the decoder stack for the constituency parsing task? | For the English constituency parsing task, a 4-layer transformer was trained [attention.pdf::p9::c3]. The decoder stack in the general model architecture is composed of a stack of N = 6 identical layers [attention.pdf::p3::c2], but the specific task description states they trained a 4-layer transformer [attention.pdf::p9::c3]. | `[attention.pdf::p3::c2]`: *"The decoder is also composed of a stack of N = 6 identical layers"* | `None` (Unanswerable) | Soft failure: every statement is supported by its cited chunks (4-layer transformer for parsing; N=6 decoder in the base model), and no unsupported number is claimed, but the model did not state that the paper does not specify the decoder layer count for parsing; it should have refused or said so. |
| `bert-u03` | How many total parameters were added to the architecture for the multi-choice classification head on SWAG? | The only task-specific parameters introduced for the SWAG dataset are a vector whose dot product with the [CLS] token representation denotes a score for each choice. | `[bert.pdf::p7::c6]`: *"...The only task-specific parameters introduced is a vec-tor whose dot product with the [CLS] token rep-resentation C denotes a score for each choice..."* | `None` (Unanswerable) | Soft failure: the answer is accurate and supported (the only added parameters are a vector scoring each choice) but gives no count and does not state that the paper gives none; it should have refused or said so. |

---

## 4. Exemplary Grounded Responses (5 Samples)

Below are five representative answers where BM25 achieved a retrieval hit and the model produced accurate, fully grounded answers with verified citations:

1. **`attn-01` (Attention Heads Count):**
   - *Question:* "What is the total number of attention heads utilized in the standard Transformer baseline?"
   - *Answer:* "In the work, they employ h = 8 parallel attention layers, or heads."
   - *Citation:* `['attention.pdf::p5::c1']`
   - *Passage Support:* Explicitly states: *"we employ $h = 8$ parallel attention layers, or heads."*

2. **`attn-02` (Training Duration):**
   - *Question:* "How long was the standard translation architecture trained on the hardware setup before completion?"
   - *Answer:* "The base models (the standard translation architecture) were trained for a total of 100,000 steps or 12 hours."
   - *Citation:* `['attention.pdf::p7::c6']`
   - *Passage Support:* States: *"The base models were trained for a total of 100,000 steps or 12 hours."*

3. **`attn-03` (Attention Scaling Factor):**
   - *Question:* "Which mathematical scaling term is applied to prevent small gradients in dot-product attention calculations?"
   - *Answer:* "To prevent small gradients, the dot products are scaled by 1 over the square root of dk (or divided by the square root of dk)."
   - *Citation:* `['attention.pdf::p4::c3', 'attention.pdf::p4::c2']`
   - *Passage Support:* States: *"we scale the dot products by $\frac{1}{\sqrt{d_k}}$."*

4. **`attn-04` (Path Length Advantage of Self-Attention):**
   - *Question:* "What primary advantage does self-attention offer regarding the maximum computational distance between token representations?"
   - *Answer:* "Self-attention offers the advantage of a maximum path length (or maximum computational distance) of O(1), connecting all positions with a constant number of operations, which makes it easier to learn long-range dependencies compared to recurrent or convolutional layers."
   - *Citation:* `['attention.pdf::p6::c1', 'attention.pdf::p6::c5']`
   - *Passage Support:* Table 1 and Section 4 text clearly demonstrate $O(1)$ maximum path length.

5. **`attn-05` (English-to-German Translation Benchmark):**
   - *Question:* "What BLEU score did the large configuration achieve on the English-to-German evaluation dataset?"
   - *Answer:* "The large configuration (big transformer model) achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task."
   - *Citation:* `['attention.pdf::p8::c3']`
   - *Passage Support:* Table 2 reports Big Transformer BLEU score of 28.4 on EN-DE.

---

## 5. False Refusals & Root Cause Analysis

Across the benchmark, exactly 7 answerable questions were refused by the model (`answerable: false`). 

Notably, **reading the retrieved text, 6 of 7 had no answer-bearing text in the top 5 (refusal appropriate); drs-11 had the value in a flattened table chunk (refusal defensible). No clear model-side false refusal was found.**

| Item ID | Question | Retrieval Hit? | Reason for Refusal |
|---|---|:---:|---|
| `attn-14` | In the model variation experiments, what development BLEU score was observed when substituting sinusoidal encodings with learned positional representations? | **False** (Miss) | p9::c1, rank 8. Exact BLEU (Table 3 row E) not in top 5; ranks 2 and 3 only say learned embeddings give "nearly identical results". Refusing the number was correct. |
| `bert-02` | Which specific vocabulary algorithm and size were selected for processing text units? | **False** (Miss) | p4::c1, rank 7. No top-5 chunk mentions the 30,000-token WordPiece vocabulary. |
| `bert-11` | In the pre-training task ablation study, what exact accuracy was recorded on the SST-2 dataset when removing the next sentence prediction objective? | **False** (Miss) | p8::c2, no query-token overlap. The "No NSP ... 92.6" table row is a numbers-and-acronyms body with no word shared with the question. |
| `drs-05` | What floating-point operation count is needed for the 34-layer plain architecture? | **False** (Miss) | p4::c3 (Fig. 3 caption "3.6 billion FLOPs"), rank 15; p3::c9 has no query-token overlap. Not Table 1. |
| `drs-09` | What top-1 accuracy error rate was registered on the ImageNet validation set by the 34-layer residual model using option B shortcut projections? | **False** (Miss) | p6::c3, rank 40. The ResNet-34 option B row is not in the top 5. |
| `drs-10` | What total parameter count is required by the 110-layer deep architecture designed for the CIFAR-10 experiment? | **False** (Miss) | p7::c9, no query-token overlap. The "ResNet 110 1.7M" row is a table row, not Section 4.2 text. |
| `drs-11` | What mean average precision score was achieved on the PASCAL VOC 2012 evaluation set when utilizing the baseline architecture paired with Faster R-CNN? | **False** (Miss) | p11::c3, rank 6. Borderline: the top-5 chunk p8::c8 holds Table 7's numbers but the columns are flattened, so VOC 2012 vs 2007 cannot be told apart. |

### Diagnostic Insight
None of the 9 manually audited items contained an unsupported claim. The other 31 grounded items have the evidence phrase in a cited chunk, but their full answer text was not audited (no LLM judge yet).

---

## 6. Honest System Limitations

1. **Unanswerable Sample Size ($n=21$):**
   - The unanswerable test suite contains 21 items. The observed true refusal rate is $90.5\%$ (19/21).
   - With $n=21$, the 95% Wilson score confidence interval spans approximately **[71.1%, 97.4%]**. A larger suite of unanswerable questions ($n \ge 100$) is necessary to establish precise statistical bounds on unanswerable refusal reliability.
2. **Single-Run & Single-Model Scope:**
   - Benchmarking was conducted with one run of `gemini-3.5-flash-lite`.
3. **Absence of an LLM-as-Judge for Semantic Correctness:**
   - Automated grounding relies on substring matching against an annotator-selected `evidence_phrase`. As demonstrated in Section 3 (`attn-07`, `bert-08`, `drs-01`, `drs-13`), an answer can be completely factually correct and cite the correct paragraph, yet fail the automated grounding check due to slight phrasing or table-formatting variations.
   - Evidence-phrase labels point to one location, so valid support elsewhere is counted as a miss; attn-07's phrase cannot match any single chunk.
   - None of the 9 manually audited items contained an unsupported claim. The other 31 grounded items have the evidence phrase in a cited chunk, but their full answer text was not audited (no LLM judge yet).
4. **Retrieval Upper Bound (Ceiling of 33/45):**
   - By the automated phrase-match metric, BM25 retrieves the evidence into the top 5 for 33/45 answerable items (73.3%). A manual audit found supporting text in the top 5 for at least 38/45 (84.4%), so the true ceiling is higher but not automatically measured.
5. **Known Retrieval Failure Modes:**
   - **Vocabulary Mismatch:** Paraphrased questions (e.g., asking about "floating-point operation count" instead of "FLOPs") fail under lexical BM25.
   - **Tabular Data Splitting:** PyMuPDF extraction often separates table headers from data rows across chunks, making numerical lookup fragile for standard chunkers.
   - **False-Refusal Retrieval Diagnostics:** Two failure modes: (a) ranked below the top 5 (attn-14 rank 8, bert-02 rank 7, drs-11 rank 6, drs-05 rank 15, drs-09 rank 40), (b) no query-token overlap, so the chunk is excluded by the retriever at any top_k (bert-11, drs-05 p3::c9, drs-10). Token diagnostics (observation only; the tokenizer was not changed or tested): drs-05 p3::c9 and drs-10: the retriever's tokenizer keeps hyphenated compounds whole in the question ('34-layer', '110-layer') while the chunks contain the parts separately ('34','layer'; '110'), so the token sets do not intersect; decimals are also split ('3.6' to '3','6'). bert-11: p8::c2 is a numbers-and-acronyms table body whose header words are in other chunks, and the question spells out 'next sentence prediction'.
   - **Table Flattening & Structural Limitations:** Table flattening by the extractor (drs-11), table-body chunks separated from their headers, and hyphen handling in the tokenizer; any tokenizer or chunker change must be a separate experiment re-run on all retrievers together.
6. **API Quota Limitations (`gemini-3.6-flash`):**
   - The Google AI Studio free tier caps `gemini-3.6-flash` at 5 requests per minute and 20 requests per day (shown on the Google AI Studio rate-limit page), making it impractical for automated testing loops or multi-item benchmarks without paid enterprise tier commitments. `gemini-3.5-flash-lite` (free tier: 15 requests per minute, 500 requests per day) serves as the viable alternative for developer-tier continuous evaluation.

---

## 7. Artifact Reproducibility

- **Benchmark Run:** `eval/results/answers_20261024_140625.json`
- **Inspection Script:** `scripts/inspect_items.py`
- **Inspection Artifact:** `eval/inspect_items.txt`
- **False-Refusal Inspection Artifact:** `eval/inspect_refusals.txt` (BM25 retrieval diagnostics for the 7 false-refusal items: top-5 retrieved chunks, evidence-phrase chunk ranks)
- **Evaluation Command:**
  ```powershell
  python scripts/run_answer_eval.py --call-delay 5
  ```
- **Inspection Command:**
  ```powershell
  python scripts/inspect_items.py
  ```
  Produces both `eval/inspect_items.txt` (9 flagged items, cited-ID based) and `eval/inspect_refusals.txt` (7 false-refusal items, BM25-based retrieval inspection with full chunk ranking).
- **Test Suite Verification:**
  ```powershell
  pytest -v
  ```
  *(146 tests passed in ~1.2s)*
