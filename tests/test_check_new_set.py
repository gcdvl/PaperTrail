"""
test_check_new_set.py: Unit tests for scripts/check_new_set.py.

Uses synthetic in-memory PDFs created with pymupdf in pytest tmp_path.
No files from papers/ are read or required.
"""

import json
from pathlib import Path
import pymupdf
import pytest

from scripts.check_new_set import check_dataset, main


def _make_pdf(path: Path, pages: list[str]) -> None:
    """Creates a tiny PDF with the given text on each page (1-based)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text)
    doc.save(str(path))
    doc.close()


def test_good_item(tmp_path: Path):
    """A valid answerable item with 4-25 words in one chunk on listed page passes."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "test.pdf"
    _make_pdf(pdf_path, [
        "This is a comprehensive study of deep neural network architectures and their performance."
    ])

    dataset = [
        {
            "id": "good-01",
            "document": "test.pdf",
            "question": "What is the subject of this study?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "study of deep neural",
            "notes": "clean test item",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, warnings = check_dataset(ds_path, papers_dir)
    assert errors == []


def test_phrase_straddling_two_chunks(tmp_path: Path):
    """An evidence phrase split across two separate chunks fails."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "straddle.pdf"

    # Creating text that splits into two chunks (> 1000 chars total)
    s1 = ("Alpha " * 150) + "boundaryone. "
    s2 = "boundarytwo " + ("Beta " * 50)
    _make_pdf(pdf_path, [f"{s1}\n\n{s2}"])

    # Phrase crosses from chunk 1 to chunk 2
    phrase = "Alpha boundaryone. boundarytwo Beta"

    dataset = [
        {
            "id": "straddle-01",
            "document": "straddle.pdf",
            "question": "What are the boundary values?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": phrase,
            "notes": "cross-chunk phrase",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir)
    assert any("No single chunk contains evidence_phrase" in e for e in errors)


def test_wrong_page(tmp_path: Path):
    """Evidence phrase on page 1 listed as page 2 fails."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "multipage.pdf"

    _make_pdf(pdf_path, [
        "This is page one text discussing the original transformer architecture.",
        "This is page two text discussing residual neural network representations.",
    ])

    dataset = [
        {
            "id": "wrong-page-01",
            "document": "multipage.pdf",
            "question": "Which architecture is discussed on page one?",
            "answerable": True,
            "expected_page_numbers": [2],  # Wrong! Phrase is on page 1
            "evidence_phrase": "original transformer architecture",
            "notes": "wrong page test",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir)
    assert any("missing from listed page 2" in e for e in errors)
    assert any("unlisted page 1" in e for e in errors)


def test_phrase_over_25_words(tmp_path: Path):
    """An evidence phrase with more than 25 words fails."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "long.pdf"

    long_sentence = (
        "This sentence has more than twenty five words in it because we are intentionally "
        "testing the validation logic that prevents overly long evidence phrases in our "
        "evaluation dataset items."
    )
    _make_pdf(pdf_path, [long_sentence])

    dataset = [
        {
            "id": "long-01",
            "document": "long.pdf",
            "question": "What does the sentence test?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": long_sentence,  # 27 words
            "notes": "too long",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir)
    assert any("must be 4 to 25 words" in e for e in errors)


def test_unanswerable_with_too_few_search_terms(tmp_path: Path):
    """An unanswerable item with fewer than 3 search_terms fails."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "unans.pdf"
    _make_pdf(pdf_path, ["General research text about machine learning models."])

    dataset = [
        {
            "id": "unans-01",
            "document": "unans.pdf",
            "question": "What is the quantum computing algorithm used?",
            "answerable": False,
            "expected_page_numbers": [],
            "evidence_phrase": None,
            "search_terms": ["quantum", "algorithm"],  # Only 2 terms!
            "notes": "too few search terms",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir)
    assert any("fewer than 3 entries" in e for e in errors)


def test_duplicate_id(tmp_path: Path):
    """Duplicate item IDs are flagged as errors."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "dup.pdf"
    _make_pdf(pdf_path, [
        "This is an introductory section on deep learning models and architectures."
    ])

    dataset = [
        {
            "id": "dup-01",
            "document": "dup.pdf",
            "question": "What is the first question?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "section on deep learning",
            "notes": "first item",
        },
        {
            "id": "dup-01",  # Duplicate!
            "document": "dup.pdf",
            "question": "What is the second question?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "section on deep learning",
            "notes": "duplicate item",
        },
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir)
    assert any("Duplicate id" in e for e in errors)


def test_unanswerable_all_terms_in_one_chunk(tmp_path: Path):
    """Unanswerable item where all search_terms appear in one chunk fails with 'review by hand'."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "hand.pdf"
    _make_pdf(pdf_path, [
        "The quick brown fox jumps over the lazy dog in the sunny park with trees."
    ])

    dataset = [
        {
            "id": "hand-01",
            "document": "hand.pdf",
            "question": "What does the quick fox do with the dog?",
            "answerable": False,
            "expected_page_numbers": [],
            "evidence_phrase": None,
            "search_terms": ["quick brown", "lazy dog", "sunny park"],
            "notes": "all in one chunk",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir)
    assert any("review by hand" in e for e in errors)


def test_table_items_exceed_twenty_percent(tmp_path: Path):
    """When table items exceed 20% of answerable items for a paper, an error is raised."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "table.pdf"
    _make_pdf(pdf_path, [
        "Page text with multiple pieces of information about neural network training."
    ])

    # 4 answerable items, 2 of which are table items (2/4 = 50% > 20%)
    dataset = [
        {
            "id": f"tab-{i}",
            "document": "table.pdf",
            "question": f"Question number {i}?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "information about neural network",
            "table": (i <= 2),  # items 1 and 2 are table items
            "notes": "table test",
        }
        for i in range(1, 5)
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir)
    assert any("Table items exceed 20%" in e for e in errors)


def test_main_cli_success_and_failure(tmp_path: Path, monkeypatch):
    """Test main() CLI wrapper exit codes."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "cli.pdf"
    _make_pdf(pdf_path, [
        "Valid text snippet for the CLI integration test case."
    ])

    good_ds = tmp_path / "good.json"
    good_ds.write_text(json.dumps([
        {
            "id": "cli-01",
            "document": "cli.pdf",
            "question": "What is this text snippet?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "Valid text snippet for",
            "notes": "cli test",
        }
    ]), encoding="utf-8")

    # Success case: returns 0
    monkeypatch.setattr("sys.argv", ["check_new_set.py", "--dataset", str(good_ds), "--papers-dir", str(papers_dir)])
    assert main() == 0

    # Failure case: missing field -> returns 1
    bad_ds = tmp_path / "bad.json"
    bad_ds.write_text(json.dumps([{"id": "bad-01"}]), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_new_set.py", "--dataset", str(bad_ds), "--papers-dir", str(papers_dir)])
    assert main() == 1


def test_allowed_pages_inside_range(tmp_path: Path):
    """When --allowed-pages is given, answerable item with expected_page_numbers inside range passes."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "range_test.pdf"
    _make_pdf(pdf_path, [
        "Page one content that is not the target.",
        "Page two content containing the specific evidence phrase for testing.",
        "Page three content following the evidence.",
    ])

    dataset = [
        {
            "id": "range-01",
            "document": "range_test.pdf",
            "question": "What is the specific evidence phrase?",
            "answerable": True,
            "expected_page_numbers": [2],
            "evidence_phrase": "specific evidence phrase for testing",
            "notes": "inside range",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    allowed_pages = {"range_test.pdf": [1, 3]}
    ap_path = tmp_path / "allowed_pages.json"
    ap_path.write_text(json.dumps(allowed_pages), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir, allowed_pages_path=ap_path)
    assert errors == []


def test_allowed_pages_outside_range(tmp_path: Path):
    """When --allowed-pages is given, expected_page_numbers outside range or missing doc fails."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_path = papers_dir / "range_test.pdf"
    _make_pdf(pdf_path, [
        "Page one content containing the specific evidence phrase for testing.",
        "Page two content.",
        "Page three content.",
    ])

    dataset = [
        {
            "id": "range-02",
            "document": "range_test.pdf",
            "question": "What is the specific evidence phrase?",
            "answerable": True,
            "expected_page_numbers": [1],
            "evidence_phrase": "specific evidence phrase for testing",
            "notes": "outside range",
        }
    ]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    # Allowed range is pages 2 to 3, but item is on page 1
    allowed_pages = {"range_test.pdf": [2, 3]}
    ap_path = tmp_path / "allowed_pages.json"
    ap_path.write_text(json.dumps(allowed_pages), encoding="utf-8")

    errors, _ = check_dataset(ds_path, papers_dir, allowed_pages_path=ap_path)
    assert any("outside allowed range" in e for e in errors)

    # Document missing from allowed_pages JSON is also an ERROR
    ap_missing_path = tmp_path / "allowed_pages_missing.json"
    ap_missing_path.write_text(json.dumps({"other.pdf": [1, 3]}), encoding="utf-8")

    errors_missing, _ = check_dataset(ds_path, papers_dir, allowed_pages_path=ap_missing_path)
    assert any("missing from allowed-pages" in e for e in errors_missing)
