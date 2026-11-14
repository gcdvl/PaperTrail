"""
Generates an annotation CSV sheet for human review of citation verifications.
Reads from eval/results/verify_20261029_082052.json and outputs eval/judge_labels.csv
with UTF-8 with BOM encoding for Excel compatibility.
"""

import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    input_file = root / "eval" / "results" / "verify_20261029_082052.json"
    output_file = root / "eval" / "judge_labels.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    fieldnames = [
        "item_id",
        "citation_id",
        "question",
        "answer",
        "cited_chunk_text",
        "judge_verdict",
        "judge_reason",
        "my_label",
        "my_notes",
    ]

    rows = []
    for record in data.get("records", []):
        item_id = record.get("id", "")
        question = record.get("question", "")
        answer = record.get("answer", "")

        for cit in record.get("citations_verification", []):
            citation_id = cit.get("citation_id", "")
            cited_chunk_text = cit.get("chunk_text", "")
            judge_verdict = cit.get("verdict", "")
            judge_reason = cit.get("reason", "")

            rows.append({
                "item_id": item_id,
                "citation_id": citation_id,
                "question": question,
                "answer": answer,
                "cited_chunk_text": cited_chunk_text,
                "judge_verdict": judge_verdict,
                "judge_reason": judge_reason,
                "my_label": "",
                "my_notes": "",
            })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} citation rows to {output_file}")


if __name__ == "__main__":
    main()

