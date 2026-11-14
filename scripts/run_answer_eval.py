import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_document
from src.extractor import extract_text_from_pdf
from src.retriever import retrieve
from src.answerer import answer_question, AnswerResult, _strip_code_fences
from src.llm_client import GeminiLLMClient
from src.metrics import passage_contains_evidence, compute_answer_eval_metrics

def _resolve_pdf(doc_name: str, papers_dir: Path, project_root: Path) -> Path:
    """Resolves a document name to a PDF path, checking papers_dir first, then project root."""
    pdf_path = papers_dir / doc_name
    if pdf_path.exists():
        return pdf_path
    pdf_path = project_root / doc_name
    if pdf_path.exists():
        return pdf_path
    raise FileNotFoundError(
        f"PDF document '{doc_name}' not found in '{papers_dir}' or '{project_root}'"
    )

def _format_rate(num: int, den: int) -> str:
    """Formats count and percentage cleanly."""
    if den == 0:
        return "0/0 (N/A)"
    pct = (num / den) * 100.0
    return f"{num}/{den} ({pct:.1f}%)"

def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick answerer evaluation (limit N items).")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "eval" / "dataset_all.json"),
        help="Path to the JSON evaluation dataset (default: eval/dataset_all.json).",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=str(PROJECT_ROOT / "papers"),
        help="Directory containing PDF documents referenced in the dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to process (e.g., 10).",
    )
    parser.add_argument(
        "--call-delay",
        type=float,
        default=13.0,
        help="Delay in seconds between API calls (default: 13.0).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    papers_dir = Path(args.papers_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Dataset: {dataset_path}  ({len(data)} items loaded)")

    if not os.getenv("GEMINI_API_KEY"):
        os.environ.setdefault("GEMINI_API_KEY", "dummy")
    if not os.getenv("GEMINI_MODEL"):
        os.environ.setdefault("GEMINI_MODEL", "gemini-3.6-flash")

    llm = GeminiLLMClient(delay_between_calls=args.call_delay)
    processed = 0
    results = []
    failures = []

    for item in data:
        if args.limit is not None and processed >= args.limit:
            break
        item_id = item.get("id", "N/A")
        doc_name = item.get("document")
        question = item.get("question")
        expected_answerable = item.get("answerable")
        evidence_phrase = item.get("evidence_phrase")

        try:
            pdf_path = _resolve_pdf(doc_name, papers_dir, PROJECT_ROOT)
            doc_result = extract_text_from_pdf(pdf_path)
            chunks = chunk_document(doc_result)
            passages = retrieve(chunks, question, top_k=5)
            passage_dicts = [{"id": p.passage_id, "text": p.passage} for p in passages]
            answer_res: AnswerResult = answer_question(question, passage_dicts, llm)

            passage_map = {p.passage_id: p.passage for p in passages}
            retrieved_ids = set(passage_map.keys())

            # Check retrieval hit: Does any top-5 retrieved passage contain evidence_phrase?
            retrieval_hit = (
                any(passage_contains_evidence(p.passage, evidence_phrase) for p in passages)
                if (evidence_phrase and expected_answerable is not False)
                else False
            )

            # Check raw citations validity from raw LLM response
            raw_citations = []
            raw_answerable = answer_res.answerable
            if answer_res.parse_ok and answer_res.raw_response:
                try:
                    cleaned_raw = _strip_code_fences(answer_res.raw_response)
                    raw_data = json.loads(cleaned_raw)
                    raw_citations = list(raw_data.get("citations") or [])
                    raw_answerable = bool(raw_data.get("answerable", False))
                except Exception:
                    raw_citations = list(answer_res.citations)
            else:
                raw_citations = list(answer_res.citations)

            # Citations validity:
            # - If citations are provided, all must be in retrieved set.
            # - If model claimed answerable, it must provide citations.
            if raw_citations:
                citations_valid = all(cid in retrieved_ids for cid in raw_citations)
            elif raw_answerable:
                citations_valid = False
            else:
                citations_valid = True

            # Check citation grounding: Does at least one cited passage contain evidence_phrase?
            if expected_answerable is True and answer_res.answerable and answer_res.citations:
                grounded = (
                    any(passage_contains_evidence(passage_map[cid], evidence_phrase)
                        for cid in answer_res.citations if cid in passage_map)
                    if evidence_phrase else False
                )
            else:
                grounded = False

            # Check failure conditions
            item_issues = []
            if not answer_res.parse_ok:
                item_issues.append("Parse failure: Response could not be parsed as JSON")
            if expected_answerable is True and not answer_res.answerable:
                item_issues.append("False refusal: Answerable item was refused by model")
            if expected_answerable is False and answer_res.answerable:
                item_issues.append(f"Wrongly answered unanswerable: Model answered with citations {answer_res.citations}")
            if answer_res.answerable and not citations_valid:
                item_issues.append(f"Invalid citations: Cited IDs {raw_citations} not in retrieved pool {list(retrieved_ids)}")
            if expected_answerable is True and answer_res.answerable and not grounded:
                item_issues.append(f"Ungrounded answer: Cited passage(s) {answer_res.citations} do not contain evidence_phrase {evidence_phrase!r}")

            ans_preview = answer_res.answer.replace("\n", " ").strip()
            if len(ans_preview) > 60:
                ans_preview = ans_preview[:57] + "..."
            flag_str = f"ans={answer_res.answerable} hit={retrieval_hit} ground={grounded}"
            print(f"[{item_id}] {flag_str} | citations={answer_res.citations} | {ans_preview}")

            rec = {
                "id": item_id,
                "document": doc_name,
                "question": question,
                "expected_answerable": expected_answerable,
                "model_answerable": answer_res.answerable,
                "evidence_phrase": evidence_phrase,
                "answer_res": answer_res,
                "citations": answer_res.citations,
                "raw_citations": raw_citations,
                "retrieval_hit": retrieval_hit,
                "grounded": grounded,
                "citations_valid": citations_valid,
                "parse_ok": answer_res.parse_ok,
            }
            results.append(rec)
            for issue in item_issues:
                failures.append({"id": item_id, "issue": issue})

        except Exception as e:
            print(f"[{item_id}] ERROR: {e}")
            failures.append({"id": item_id, "issue": f"Execution error: {e}"})
        finally:
            processed += 1

    print(f"\nProcessed {processed} of {len(data)} items")
    summary_data = compute_answer_eval_metrics(results)

    col_w = 26
    print("\n" + "=" * 90)
    print("ANSWER EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Metric':<36} | {'Overall':<{col_w}} | {'Retrieval Hit (in top-5)':<{col_w}} | {'Retrieval Miss':<{col_w}}")
    print("-" * 90)

    for metric_name, key_cnt, key_tot in [
        ("Total items", "total", None),
        ("Refusal rate on unanswerable", "refusal_count_unanswerable", "unanswerable_count"),
        ("False-refusal rate on answerable", "false_refusal_count", "answerable_count"),
        ("Citation validity (IDs in pool)", "citation_validity_count", "citation_validity_total"),
        ("Citation grounding (has evidence)", "citation_grounding_count", "citation_grounding_total"),
        ("Parse failures", "parse_failure_count", "total"),
    ]:
        if key_tot is None:
            ov = str(summary_data["overall"][key_cnt])
            hit = str(summary_data["retrieval_hit"][key_cnt])
            miss = str(summary_data["retrieval_miss"][key_cnt])
        else:
            ov = _format_rate(summary_data["overall"][key_cnt], summary_data["overall"][key_tot])
            hit = _format_rate(summary_data["retrieval_hit"][key_cnt], summary_data["retrieval_hit"][key_tot])
            miss = _format_rate(summary_data["retrieval_miss"][key_cnt], summary_data["retrieval_miss"][key_tot])
        print(f"{metric_name:<36} | {ov:<{col_w}} | {hit:<{col_w}} | {miss:<{col_w}}")
    print("=" * 90)

    print(f"\nFAILURE LIST ({len(failures)} issue{'s' if len(failures) != 1 else ''} found)")
    print("=" * 90)
    if failures:
        for f in failures:
            print(f"  - [{f['id']}] {f['issue']}")
    else:
        print("  None")
    print("=" * 90)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"answers_{timestamp_str}.json"

    output_obj = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary_data,
        "results": [
            {
                "id": r["id"],
                "document": r["document"],
                "question": r["question"],
                "expected_answerable": r["expected_answerable"],
                "answerable": r["model_answerable"],
                "answer": r["answer_res"].answer,
                "citations": r["citations"],
                "retrieval_hit": r["retrieval_hit"],
                "grounding": r["grounded"],
                "citations_valid": r["citations_valid"],
                "parse_ok": r["parse_ok"],
            }
            for r in results
        ],
        "failures": failures,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation results to: {out_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

