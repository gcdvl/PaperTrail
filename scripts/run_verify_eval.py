"""
run_verify_eval.py: Verification runner for PaperTrail Milestone 6.

Evaluates answered items from an existing answer run (default: eval/results/answers_20261024_140625.json)
using src.verifier:
1. verify_citation: Evaluates whether each cited passage chunk supports the answer's claim.
2. verify_addresses_question: Evaluates whether the answer addresses the question asked.

Outputs:
- eval/results/verify_<timestamp>.json: Full structured verification results and statistics.
- eval/verify_items.txt: Human-readable audit report with question, answer, cited chunk text,
  and both verdicts for every evaluated item.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import Chunk, chunk_document
from src.extractor import extract_text_from_pdf
from src.llm_client import GeminiLLMClient
from src.verifier import verify_citation, verify_addresses_question


def resolve_pdf(doc_name: str, papers_dir: Path, project_root: Path) -> Path:
    """Resolves document name to a PDF path."""
    pdf_path = papers_dir / doc_name
    if pdf_path.exists():
        return pdf_path
    pdf_path = project_root / doc_name
    if pdf_path.exists():
        return pdf_path
    raise FileNotFoundError(f"PDF document '{doc_name}' not found.")


class TrackingGeminiLLMClient(GeminiLLMClient):
    """Subclass of GeminiLLMClient to track API calls made vs cache hits and quota status."""

    def __init__(self, delay_between_calls: float = 5.0):
        super().__init__(delay_between_calls=delay_between_calls)
        self.api_calls_made = 0
        self.cache_hits = 0

    def generate(self, system: str, user: str) -> str:
        cache_key = self._cache_key(system, user)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Cache miss: will make live API call
        self.api_calls_made += 1
        return super().generate(system, user)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 6 Answer Verification Runner")
    parser.add_argument(
        "--answers",
        type=str,
        default=str(PROJECT_ROOT / "eval" / "results" / "answers_20261024_140625.json"),
        help="Path to answers JSON file to verify.",
    )
    parser.add_argument(
        "--call-delay",
        type=float,
        default=5.0,
        help="Delay in seconds between Gemini API calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of answered items to process (for debugging).",
    )
    args = parser.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")

    papers_dir = PROJECT_ROOT / "papers"
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(answers_path, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    all_results = answers_data.get("results", [])
    # Filter to answered items only (not refusals)
    answered_items = [r for r in all_results if r.get("answerable") is True]

    if args.limit is not None:
        answered_items = answered_items[:args.limit]

    print(f"Loaded {len(all_results)} total items from {answers_path.name}")
    print(f"Processing {len(answered_items)} answered items (refusals skipped)...")

    # Environment variable check (do NOT print keys)
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("Missing GEMINI_API_KEY environment variable")
    if not os.getenv("GEMINI_MODEL"):
        os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash-lite")

    llm = TrackingGeminiLLMClient(delay_between_calls=args.call_delay)

    # Document chunk text cache
    doc_chunk_map_cache: Dict[str, Dict[str, str]] = {}

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_output_path = results_dir / f"verify_{timestamp_str}.json"
    txt_output_path = PROJECT_ROOT / "eval" / "verify_items.txt"

    verification_records = []
    quota_hit = False
    quota_error_msg = ""

    # Statistics counters
    citation_verdict_counts = {
        "supported": 0,
        "partially_supported": 0,
        "not_supported": 0,
    }
    citation_parse_failures = 0

    addresses_verdict_counts = {
        "answers_question": 0,
        "states_not_in_paper": 0,
        "off_topic_or_wrong_figure": 0,
    }
    addresses_parse_failures = 0

    for idx, item in enumerate(answered_items, start=1):
        item_id = item.get("id")
        doc_name = item.get("document", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        citations = item.get("citations", [])
        expected_answerable = item.get("expected_answerable")

        # Cache document chunk texts
        if doc_name not in doc_chunk_map_cache:
            pdf_path = resolve_pdf(doc_name, papers_dir, PROJECT_ROOT)
            doc_result = extract_text_from_pdf(pdf_path)
            chunks = chunk_document(doc_result)
            doc_chunk_map_cache[doc_name] = {c.passage_id: c.text for c in chunks}

        chunk_map = doc_chunk_map_cache[doc_name]

        print(f"[{idx}/{len(answered_items)}] Verifying item {item_id} ({len(citations)} citations)...")

        # 1. verify_addresses_question
        addresses_res = None
        try:
            addresses_res = verify_addresses_question(question, answer, llm)
            if addresses_res.parse_ok and addresses_res.verdict:
                addresses_verdict_counts[addresses_res.verdict] += 1
            else:
                addresses_parse_failures += 1
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                quota_hit = True
                quota_error_msg = err_str
                print(f"QUOTA / RATE LIMIT HIT during addresses check on {item_id}: {err_str}")
                break
            raise

        # 2. verify_citation for each cited chunk
        citation_results = []
        for cid in citations:
            chunk_text = chunk_map.get(cid, "[Chunk not found in document]")
            try:
                c_res = verify_citation(question, answer, chunk_text, llm)
                if c_res.parse_ok and c_res.verdict:
                    citation_verdict_counts[c_res.verdict] += 1
                else:
                    citation_parse_failures += 1
                citation_results.append({
                    "citation_id": cid,
                    "chunk_text": chunk_text,
                    "verdict": c_res.verdict,
                    "reason": c_res.reason,
                    "parse_ok": c_res.parse_ok,
                    "raw_response": c_res.raw_response,
                })
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    quota_hit = True
                    quota_error_msg = err_str
                    print(f"QUOTA / RATE LIMIT HIT during citation check on {item_id} ({cid}): {err_str}")
                    break
                raise

        record = {
            "id": item_id,
            "document": doc_name,
            "question": question,
            "answer": answer,
            "expected_answerable": expected_answerable,
            "addresses_question": {
                "verdict": addresses_res.verdict if addresses_res else None,
                "reason": addresses_res.reason if addresses_res else None,
                "parse_ok": addresses_res.parse_ok if addresses_res else False,
                "raw_response": addresses_res.raw_response if addresses_res else None,
            },
            "citations_verification": citation_results,
        }
        verification_records.append(record)

        if quota_hit:
            break

    # Build human-readable text file
    txt_lines: List[str] = []
    txt_lines.append("=" * 80)
    txt_lines.append(f"PAPERTRAIL MILESTONE 6: ANSWER VERIFICATION REPORT")
    txt_lines.append(f"Timestamp: {timestamp_str}")
    txt_lines.append(f"Source Answer Run: {answers_path.name}")
    txt_lines.append(f"Model: {llm.model_name}")
    txt_lines.append(f"Evaluated Items: {len(verification_records)} of {len(answered_items)} answered items")
    if quota_hit:
        txt_lines.append(f"STATUS: HALTED EARLY DUE TO QUOTA / RATE LIMIT: {quota_error_msg}")
    else:
        txt_lines.append("STATUS: COMPLETE")
    txt_lines.append("=" * 80)
    txt_lines.append("")

    for r in verification_records:
        txt_lines.append("=" * 80)
        txt_lines.append(f"ITEM ID: {r['id']} (Document: {r['document']})")
        txt_lines.append(f"Expected Answerable: {r['expected_answerable']}")
        txt_lines.append(f"QUESTION:\n{r['question']}")
        txt_lines.append(f"ANSWER:\n{r['answer']}")
        txt_lines.append("")

        addr = r["addresses_question"]
        txt_lines.append(f"ADDRESSES QUESTION CHECK:")
        txt_lines.append(f"  Verdict: {addr['verdict']}")
        txt_lines.append(f"  Reason: {addr['reason']}")
        txt_lines.append(f"  Parse OK: {addr['parse_ok']}")
        txt_lines.append("")

        txt_lines.append(f"CITATION CHECKS ({len(r['citations_verification'])} citations):")
        if not r["citations_verification"]:
            txt_lines.append("  None")
        for cv in r["citations_verification"]:
            txt_lines.append(f"  - Citation ID: {cv['citation_id']}")
            txt_lines.append(f"    Verdict: {cv['verdict']}")
            txt_lines.append(f"    Reason: {cv['reason']}")
            txt_lines.append(f"    Parse OK: {cv['parse_ok']}")
            txt_lines.append(f"    Cited Chunk Text:")
            for line in cv["chunk_text"].split("\n"):
                txt_lines.append(f"      {line}")
            txt_lines.append("")
        txt_lines.append("")

    txt_content = "\n".join(txt_lines)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    total_citations_evaluated = sum(len(r["citations_verification"]) for r in verification_records)
    summary_data = {
        "timestamp": timestamp_str,
        "source_answers": str(answers_path),
        "model": llm.model_name,
        "total_answered_in_source": len(answered_items),
        "items_evaluated": len(verification_records),
        "quota_hit": quota_hit,
        "quota_error_msg": quota_error_msg if quota_hit else None,
        "api_calls_made": llm.api_calls_made,
        "cache_hits": llm.cache_hits,
        "citation_verification": {
            "total_evaluated": total_citations_evaluated,
            "verdicts": citation_verdict_counts,
            "parse_failures": citation_parse_failures,
        },
        "addresses_question_verification": {
            "total_evaluated": len(verification_records),
            "verdicts": addresses_verdict_counts,
            "parse_failures": addresses_parse_failures,
        },
        "records": verification_records,
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print()
    print("=" * 60)
    print("VERIFICATION RUN SUMMARY")
    print("=" * 60)
    print(f"Items evaluated: {len(verification_records)} / {len(answered_items)}")
    print(f"API calls made: {llm.api_calls_made} | Cache hits: {llm.cache_hits}")
    print(f"Quota hit: {quota_hit}")
    print()
    print(f"Citation Support ({total_citations_evaluated} citations across {len(verification_records)} items):")
    for v, c in citation_verdict_counts.items():
        pct = (c / total_citations_evaluated * 100) if total_citations_evaluated else 0.0
        print(f"  {v}: {c} ({pct:.1f}%)")
    print(f"  Parse failures: {citation_parse_failures}")
    print()
    print(f"Addresses Question ({len(verification_records)} items):")
    for v, c in addresses_verdict_counts.items():
        pct = (c / len(verification_records * 100)) if verification_records else 0.0
        print(f"  {v}: {c} ({pct:.1f}%)")
    print(f"  Parse failures: {addresses_parse_failures}")
    print()
    print(f"JSON Output: {json_output_path}")
    print(f"Text Output: {txt_output_path}")


if __name__ == "__main__":
    main()

