# Labelling rules for the new evaluation set (written BEFORE labelling)

Scope: new papers only. The old 66-item set (dataset_all.json) stays frozen and
had no written rules, so old and new numbers are not strictly comparable.

## Files
- PDFs: eval/new/papers_new/ (not committed). Items: eval/new/dataset_new.json (draft) ->
  eval/new/dataset_new_frozen.json (after freeze). Never touch dataset_all.json.

## Item fields
Required (same as the old set): id, document, question, answerable,
expected_page_numbers (1-based), evidence_phrase, notes.
Optional: Optional: qtype (number|definition|method|comparison|table|other),
subtype (unanswerable only: a|b|c), table (true|false), search_terms (unanswerable only),
drafted_by (author|gpt-edited|claude).
IDs: <paper>-01.. for answerable, <paper>-u01.. for unanswerable.

## Per paper: 15 answerable + 7 unanswerable.
## Size (fixed before labelling, not to be changed after seeing any results)
5 new papers x (15 answerable + 7 unanswerable) = 75 answerable + 35 unanswerable = 110 items.
Total set after adding: 8 papers (3 old + 5 new), 66 old + 110 new = 176 items.

## Answerable
1. 1. Questions may be drafted with an AI (dlrm: Claude; GPT earlier drafts were replaced). Every
   question is read against the paper and recorded in "drafted_by". The evidence phrase and the
   answerable/unanswerable label are always checked with the dump and the checker; an AI's claim
   that "the paper doesn't say this" is never trusted. Never run PaperTrail on a new paper
   before it is frozen. Amended 2026-11-24, before any labelling was finished or any system run.
2. Paraphrase. Do not copy the evidence sentence's wording.
3. The question must make sense alone (no "in Table 2 of this paper").
4. evidence_phrase: copy-paste from the extractor text (dump_pages output),
   not from a PDF viewer. 4 to 25 words, inside ONE chunk, no table-cell soup.
5. expected_page_numbers must list EVERY page where the phrase appears.
   If it is repeated, choose a more specific phrase.
6. Mix qtypes. At most 20% of a paper's answerable items may be table items
   (table: true).
7. The phrase must contain the answer, not just the topic.
8. Prefer evidence phrases that don’t contain a line-end hyphenated word. If unavoidable, copy it exactly as the dump shows it.


## Unanswerable
Subtypes: (a) on-topic but the paper does not state it, (b) false premise
(wrong entity or number), (c) off-topic. At least 4 of 7 per paper must be (a).
Before labelling, search the extracted text for 3+ key terms and record them
in search_terms. If the paper states it anywhere, it is answerable.

## Label consistency check (replaces the second labeller; fixed 2026-11-24)
No second labeller is available. Instead: at least 7 days after finishing, re-label a
random 20% of items blind (question + PDF only, without looking at the old labels).
Report agreement on answerable/unanswerable and on page. This measures consistency of one
person over time, not agreement between people, and the README says so. Items that
disagree are removed before freezing. If a classmate or friend later labels a subset,
report that too, and keep it separate.

## Freeze
validator passes -> check_new_set passes -> record SHA-256 -> zip/tag.
After freezing: no edits, no removals, no "fixing" items because of results.
Any later change = a new version file with a written reason.



## Allowed evidence pages (fixed 2026-11-14, before labelling)
creatine 1-12, ddpm 1-8, dlrm 1-8, lora 1-12, rag 1-9 (pages before the reference list).
No appendix evidence. Unanswerable items are checked against the WHOLE paper.
Enforced by eval/new/allowed_pages.json via check_new_set.py --allowed-pages.
Before labelling a paper, read one body page of its dump in VS Code; if it reads wrongly
(columns mixed, garbled), stop and tell Claude.
