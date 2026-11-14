# PaperTrail -- Milestone 6.5 Report: Table-Aware Chunking Experiment (Not Implemented)

**Date:** 2026-11-07  
**Outcome:** Experiment stopped before implementation. The chunker is unchanged. No new-chunker results exist.  
**Reason:** PyMuPDF `find_tables()` does not detect or reconstruct the tables in the three evaluation papers (Section 4).

---

## 1. Question

Frozen-set failures and judge errors traced back to flattened tables (`drs-11`, `bert-11`, `attn-10`, and the M6 judge's misreading of Table 4). Question: does a table-aware extractor/chunker, which keeps a table's caption, header and rows together, improve retrieval of table cells?

## 2. Experimental Design (written before any implementation)

- **Held-out set:** `eval/heldout_tables.json`, 15 questions (5 per paper: attention.pdf, bert.pdf, drs.pdf), each 3 single-cell lookups + 2 two-row comparisons per paper, spread over 11 tables. No question was taken from `eval/dataset_all.json`.
- **How it was built:** drafted by Gemini, then every question and answer verified by the project owner against the PDFs; `must_contain` strings (1 to 4 per question) checked with `scripts/check_heldout.py` against the extracted text of the stated page and against other pages.
- **Hit rule:** a retrieved chunk from the question's document is a hit if it contains every `must_contain` string (case-insensitive substring), on any page. `drs-table-03` has the same fact on pages 8 and 11; either counts.
- **Adoption rule (saved before the baseline run):** adopt only if old-path results reproduce the M4 frozen numbers, BM25 held-out R@5 improves by at least 3 questions (8 to 11 of 15), and frozen-set non-table items drop by no more than 1. Rule file: `eval/EXPERIMENT_TABLES_RULE.txt` (exact name: see project folder).
- **Frozen-set groups for the rule:** 11 table-evidence items (`attn-10, attn-11, attn-14, bert-10, bert-11, bert-12, bert-14, drs-09, drs-10, drs-11, drs-13`) and the other 34 answerable items.

### Held-out set corrections made during verification
- `attn-table-02`: Gemini's answer (1.0 x 10^20) was wrong; the ByteNet row has a BLEU score only. Replaced with a BLEU question (23.75).
- `drs-table-04`: Gemini cited "Option B" and 25.02; Table 2 shows plain-34 28.54 and ResNet-34 25.03 (difference 3.51). Corrected.
- `drs-table-03` duplicated frozen `drs-11` (same cell). Replaced with a Table 10 question (76.4).
- `bert-table-02` duplicated frozen `bert-05` (SQuAD 2.0 test F1 83.1). Replaced with a Table 4 SWAG question (ESIM+ELMo Dev 59.1).
- These two replacements happened after the first baseline run and before any new-chunker result.

## 3. Baseline (current chunker, old path)

Final baseline (`eval/results/heldout_baseline_20261106_201915.txt`), counts out of 15:

| Retriever | R@1 | R@3 | R@5 |
|---|:---:|:---:|:---:|
| BM25 | 3 | 5 | 8 |
| Hybrid (RRF k=60) | 2 | 5 | 5 |
| Dense (bge-small) | 1 | 2 | 3 |

First baseline, before the two replacements (`eval/results/heldout_baseline_20261106_160450.txt`): BM25 4/6/8, hybrid 3/5/6, dense 2/4/4.

- BM25 hits 6 of 9 lookups and 2 of 6 comparison questions in the first baseline's per-question ranks; comparison questions need both values and labels in one chunk.
- Current BM25 misses (7): `attn-table-02`, `attn-table-05`, `bert-table-01`, `bert-table-04`, `drs-table-02`, `drs-table-04`, `drs-table-05`.
- Agent-reported chunk dump (consistent with the `p10::c1` and `p10::c6` texts in `eval/verify_items.txt`): Table 4 (attention p10) is split across `p10::c1` to `p10::c7`; a row's label and its value often fall in different chunks, so for `attn-table-05` and `drs-table-04` no single chunk contains all required strings under the old chunker.

## 4. Why the Experiment Stopped

A table-aware chunker needs table rows as input. A read-only coverage run (`scripts/table_coverage.py`, PyMuPDF `find_tables()`, 15 table pages) showed:

- **Default strategy finds 0 tables on 6 of 15 pages:** attention p6 and p8; bert p6, p7, p8, p9. These pages hold 8 of the 15 held-out questions (attention 3, bert 5).
- **Where tables are detected, rows are not recovered.** Example: Table 4 (attention p10) returns the parser names in one cell and the F1 values in another; drs p6 and p7 show merged columns (e.g. `28.07 9.33 - 9.15 24.27 7.38`). A row label still is not tied to its value.
- **`strategy="text"` returns one large pseudo-table per page** with words split mid-word (e.g. `Maxi | mum`).
- **Consequence for the rule:** of the 7 current BM25 misses, 3 are on undetected pages (`attn-table-02`, `bert-table-01`, `bert-table-04`). The other 4 would all have to flip to reach the +3 bar, and the coverage run did not show that rows could be rebuilt for them.

Decision: do not implement a `find_tables()`-based chunker. The chunker, extractor and retrievers are unchanged.

## 5. Process Note

An earlier agent session reported a completed Stage 1 implementation (flag, `TableBlock`, dump script, new tests). Verification showed it was not real: `pytest` stayed at 159 passed with the flag on and off, `scripts/dump_table_chunks.py` did not exist, and a search of `src/*.py` found no `PAPERTRAIL_TABLE_AWARE`, `find_tables` or `TableBlock`. A file listing showed no modified files in `src/` or `tests/`. Nothing from that report is used here.

## 6. Not Pursued

- **Coordinate-based row reconstruction** (rebuild rows from `page.get_text("words")`, anchored on "Table N" captions). Generic and independent of ruling lines, but a real build; not attempted.
- **`pymupdf_layout`** (suggested by a PyMuPDF warning). Untested; compatibility with Python 3.14 unknown.
- **Raising `min_chars` to merge short row chunks.** Rejected: it could raise the `must_contain` score without aligning rows to labels, so a higher score would not mean better retrieval.

## 7. Limitations

- 15 held-out questions, one labeller, drafted by Gemini (the model family used for answering in M5 and judging in M6) and verified by the project owner.
- No new-chunker numbers exist, so no claim is made about whether table-aware chunking would help.

## 8. Reproducibility

- Held-out questions: `eval/heldout_tables.json`
- Checker: `scripts/check_heldout.py` (`--show ID` prints text windows)
- Baseline runner: `scripts/run_heldout_eval.py`; results in `eval/results/heldout_baseline_*.txt`
- Coverage: `scripts/table_coverage.py`
- Rule file: `eval/EXPERIMENT_TABLES-*.txt`
- Tests: `pytest -q` gives `159 passed, 1 warning` with and without `PAPERTRAIL_TABLE_AWARE=1` (the flag does nothing in the current code)
