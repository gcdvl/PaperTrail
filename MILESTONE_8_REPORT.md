# PaperTrail -- Milestone 8 Report: Reranker Experiment (Not Adopted)

**Date:** 2026-11-08 to 2026-11-09
**Status:** Experiment completed. The reranker did not meet the pre-registered adoption rule and is **not wired into the API**.

---

## 1. Summary

M8 tested whether a cross-encoder reranker over the BM25 top-20 improves passage recall on the frozen evaluation set. The adoption rule was written in `MILESTONE_8_PLAN.md` before any result existed.

- BM25 has the evidence somewhere in its top 20 for **40 of 45** answerable items, against 33 of 45 in the top 5, so a perfect reranker could gain at most 7 items.
- The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescued 4 of those 7 items but pushed 4 items that BM25 already had in its top 5 out of it. Net passage R@5 change: **+0** (33/45 before and after).
- On the held-out table set, R@5 **fell from 8/15 to 6/15**.
- Three of the four adoption conditions failed, so the result is **ADOPT: NO**. No file in `src/api.py` or `src/service.py` was changed.

---

## 2. Question and pre-registered rule

Question: does reranking the BM25 top-20 with a cross-encoder improve passage recall over plain BM25 top-5?

The plan (`MILESTONE_8_PLAN.md`, written 2026-11-08 17:41, before the candidate-recall script and before any reranker run) fixed:

- Data: `eval/dataset_all.json`, the 45 answerable items; held-out: `eval/heldout_tables.json` (15 items, BM25 baseline 8/15 at R@5).
- One model only: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Pool 20, final top-5. No second model, no pool or model changes after seeing results.
- Stop rule for Step 1: stop if fewer than 3 answerable items have their evidence at BM25 rank 6 to 20.
- Adoption rule (all four must hold):
  (a) net R@5 gain of at least 3 items on the 45;
  (b) R@1 does not drop;
  (c) no paper has a net R@5 change below -1;
  (d) held-out R@5 stays at or above 8 of 15.

---

## 3. Step 1: how much can a reranker gain? (BM25 candidate recall)

`scripts/candidate_recall.py` (BM25 only, no model, no network). Self-check against the published M4 numbers passed (R@1/R@3/R@5 = 20/32/33 of 45). Output saved in `eval/results/candidate_recall_20261108_175355.txt`.

| k | 1 | 3 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|
| Items with evidence in top k (of 45) | 20 | 32 | 33 | 37 | 40 | 40 | 41 |
| Percent | 44.4 | 71.1 | 73.3 | 82.2 | 88.9 | 88.9 | 91.1 |

Per paper (of 15 each): R@5 / R@20 = attention 13 / 14, bert 11 / 13, drs 9 / 13.

Items whose best rank is outside the top 5:

| Item | Best BM25 rank | Item | Best BM25 rank |
|---|---|---|---|
| attn-07 | not retrieved | bert-11 | not retrieved |
| attn-14 | 8 | drs-01 | 18 |
| bert-02 | 7 | drs-03 | 6 |
| bert-03 | not retrieved | drs-05 | 15 |
| bert-08 | 17 | drs-09 | 40 |
| drs-10 | not retrieved | drs-11 | 6 |

Seven items sit at ranks 6 to 20 ("rescuable"), so the stop rule was not triggered. Five items (attn-07, bert-03, bert-11, drs-10 at "not retrieved" and drs-09 at rank 40) are outside the top 20 and cannot be reached by any reranker over a 20-item pool. The ranks for attn-14, bert-02, drs-11, drs-05 and drs-09 match those reported in `MILESTONE_5_REPORT.md`.

---

## 3.1 What was built

| File | Purpose |
|---|---|
| `src/reranker.py` | `CrossEncoderReranker`, `retrieve_with_rerank`; `sentence_transformers` is imported lazily |
| `tests/test_reranker.py` | 16 offline tests using a stub scorer (no model download) |
| `scripts/candidate_recall.py` | BM25 candidate recall (Step 1) |
| `scripts/run_rerank_eval.py` | Baseline vs reranked comparison and adoption rule (Step 3) |
| `MILESTONE_8_PLAN.md` | Pre-registered plan |

Test suite: 196 before M8, **212 passing** after (+16). The reranker module remains in `src/` but nothing in the API uses it.

---

## 4. Results (`eval/results/rerank_eval_20261109_111616.txt` and `.json`)

All three self-checks passed: baseline R@1/R@3/R@5 = 20/32/33; net gained minus lost equals the R@5 difference; held-out baseline R@5 = 8.

### 4.1 Frozen set (45 answerable items), passage recall

| | Baseline (BM25 top-5) | Reranked (BM25 top-20 -> cross-encoder -> top-5) |
|---|---|---|
| R@1 | 20/45 (44.4%) | 21/45 (46.7%) |
| R@3 | 32/45 (71.1%) | 29/45 (64.4%) |
| R@5 | 33/45 (73.3%) | 33/45 (73.3%) |

Per paper (baseline -> reranked, of 15):

| Paper | R@1 | R@3 | R@5 |
|---|---|---|---|
| attention | 9 -> 9 | 13 -> 12 | 13 -> 14 |
| bert | 6 -> 7 | 11 -> 8 | 11 -> 9 |
| drs | 5 -> 5 | 8 -> 9 | 9 -> 10 |

### 4.2 Items that changed at R@5

Gained (baseline miss, reranked hit): 

| Item | BM25 rank | Reranked rank |
|---|---|---|
| attn-14 | 8 | 2 |
| bert-02 | 7 | 5 |
| drs-01 | 18 | 2 |
| drs-05 | 15 | 5 |

Lost (baseline hit, reranked miss):

| Item | BM25 rank | Reranked rank |
|---|---|---|
| bert-06 | 2 | 18 |
| bert-12 | 3 | 12 |
| bert-13 | 3 | 9 |
| drs-13 | 5 | 12 |

Gained 4, lost 4, **net +0**. Three of the seven rescuable items (bert-08, drs-03, drs-11) were not rescued.

### 4.3 Held-out table set (15 items)

| | Baseline | Reranked |
|---|---|---|
| R@1 | 3/15 | 3/15 |
| R@3 | 5/15 | 5/15 |
| R@5 | 8/15 | **6/15** |

Two held-out hits were lost (attn-table-04 and bert-table-02, both BM25 rank 4 -> miss). No held-out miss became a hit. Two items changed rank within the top 5 (drs-table-01: 3 -> 5; drs-table-03: 4 -> 2).

### 4.4 Timing (this machine)

Model load plus one warm-up call: 19.88 s. Rerank call per query (scoring 20 passages and sorting, not including BM25): mean 1058.3 ms, max 1611.7 ms over 60 queries.

---

## 5. Adoption rule outcome

| Condition | Result | |
|---|---|---|
| (a) net R@5 gain >= 3 | +0 | FAIL |
| (b) reranked R@1 >= baseline R@1 | 21 >= 20 | PASS |
| (c) no paper net R@5 change below -1 | attention +1, bert -2, drs +1 | FAIL |
| (d) held-out reranked R@5 >= 8 of 15 | 6/15 | FAIL |

**ADOPT: NO.** The rule was applied as written.

---

## 6. What this does and does not show

- On this evaluation set, with this model and a 20-item pool, reranking did not improve top-5 passage recall overall. It changed *which* items are hit: it moved four deep BM25 hits up and four shallow ones down.
- R@3 got worse (32 -> 29) and held-out R@5 got worse (8 -> 6). R@1 improved by one item. The R@3 figure was not part of the adoption rule, but it is reported because it is a regression.
- The set is small: one item is 2.2 points on the 45. A 4-up, 4-down result is a "no detectable benefit" result and is not evidence about the model in general.
- Possible explanations were **not tested**. The model was trained on web-search passages, and the two held-out hits it lost are from the table questions, whose chunks are flattened numbers. The passages of the lost items were not inspected.
- The metric is evidence-phrase matching, a proxy. `MILESTONE_5_REPORT.md` found by manual audit that at least 38 of 45 answerable items already had supporting text in the BM25 top 5, so the true headroom is smaller than 33 vs 40. The effect on final answers (answerer or refusals) was **not measured**; this experiment used retrieval metrics only and made no Gemini calls.

---

## 7. Known limitations of this experiment

- One reranker model, one pool size, one run; no second model was tried because the plan fixed that in advance.
- Three papers; the frozen set and held-out set are small.
- Timing is from one machine and one run.
- 5 of the 45 items are unreachable by any reranker over BM25 top-20.

---

## 8. Candidate next steps (not started)

Each would be a new experiment needing its own written plan before running:

- Combine BM25 and reranker scores instead of replacing the BM25 order.
- A different or larger reranker, declared in advance.
- Tokenizer fix for hyphenated compounds and decimals (see `MILESTONE_5_REPORT.md`), re-run on all retrievers together.
- Rebuilding table rows from `page.get_text("words")` coordinates (see `MILESTONE_6_5_REPORT.md`).

---

## 9. How to reproduce

```powershell
python scripts\candidate_recall.py
python scripts\run_rerank_eval.py
python -m pytest -q
```

The first reranker run downloads the model (about 86 MB) from the Hugging Face Hub and needs internet; later runs load it from the local cache. The scripts use BM25 and the cross-encoder only, with no Gemini calls.
