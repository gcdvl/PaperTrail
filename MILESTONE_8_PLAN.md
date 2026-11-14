# M8 plan: reranker experiment (written 2026-11-08, before any run)

Question: does a cross-encoder reranker over the BM25 top-20 improve passage recall
on the frozen set?

Data: eval/dataset_all.json, the 45 answerable items (same metric as M4).
Held-out: eval/heldout_tables.json (15 items), BM25 baseline 8 of 15 at R@5.

Step 1 (no model): BM25 candidate recall at k = 5, 10, 20, 30, 50.
Stop rule: if fewer than 3 answerable items have their evidence at BM25 rank 6 to 20,
the adoption rule cannot be met. Stop and write a short report.

Step 2: one model only, cross-encoder/ms-marco-MiniLM-L-6-v2. BM25 top-20, rerank,
take top-5. Pool size and model are not changed after seeing results. No second model.

Adoption rule (all must hold): net passage R@5 gain of at least 3 items (6.7 points)
on the 45; R@1 does not drop; no paper loses more than 1 item net; held-out R@5 stays
at or above 8 of 15. Nothing is wired into src/api.py unless all hold.

Report regardless: per-item wins and losses, regressions, time per query, model name.

## Step 1 result (run 17:53, eval/results/candidate_recall_20261108_175355.txt)
BM25 R@5 33/45, R@10 37/45, R@20 40/45, R@50 41/45. Rescuable items (rank 6 to 20): 7.
Stop rule not triggered (7 >= 3). Upper bound for a top-20 reranker: 40/45 (+7 items).
Adoption needs net +3 on R@5.

## Step 2 log (2026-11-09)
src/reranker.py and tests/test_reranker.py built, 212 tests passing. Smoke test on one
question that is not in the datasets (heads question) only confirmed that the model
loads and returns 5 sorted results. It is not evidence. The rule above is unchanged.

## Step 3 result (run 2026-11-09 11:16, eval/results/rerank_eval_20261109_111616.txt)
   Net R@5 +0 (4 gained, 4 lost), R@1 20 -> 21, held-out R@5 8 -> 6, bert net -2.
   Rule failed on (a), (c), (d). ADOPT: NO. Nothing wired into src/api.py.