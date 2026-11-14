"""
conftest.py: Pytest fixtures for creating temporary, self-contained test PDFs.

All test PDFs are generated dynamically in temporary directories using PyMuPDF,
ensuring tests do not require external network access or pre-existing assets.
"""

from pathlib import Path
import pytest
import pymupdf


@pytest.fixture
def sample_three_page_pdf(tmp_path: Path) -> Path:
    """
    Creates a 3-page mock research paper PDF with distinct text on each page.
    """
    pdf_path = tmp_path / "deep_residual_learning.pdf"
    doc = pymupdf.open()

    # Page 1: Abstract & Intro
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 72),
        "Abstract: Deeper neural networks are more difficult to train.\n"
        "We present a residual learning framework to ease the training.\n"
        "1. Introduction\n"
        "Deep convolutional neural networks have led to a series of breakthroughs.",
        fontsize=11,
    )

    # Page 2: Methodology
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 72),
        "2. Deep Residual Learning\n"
        "Formally, denoting the desired underlying mapping as H(x),\n"
        "we let the stacked nonlinear layers fit another mapping of F(x) := H(x) - x.",
        fontsize=11,
    )

    # Page 3: Experiments & Results
    page3 = doc.new_page(width=595, height=842)
    page3.insert_text(
        (50, 72),
        "3. Experiments on ImageNet\n"
        "Our baseline 34-layer plain net has higher training error than 18-layer plain net.\n"
        "In contrast, 34-layer ResNet achieves significantly lower training error.",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    """Creates a 1-page PDF document with no text (blank page)."""
    pdf_path = tmp_path / "blank_paper.pdf"
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)  # blank page without text
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def mixed_blank_and_text_pdf(tmp_path: Path) -> Path:
    """
    Creates a 3-page PDF where Page 1 has text, Page 2 is blank, and Page 3 has text.
    Used to verify that page numbers remain accurate (1, 2, 3) even if intermediate
    pages contain no extractable text.
    """
    pdf_path = tmp_path / "mixed_document.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 72), "Content on page one.", fontsize=11)

    # Page 2 is deliberately left empty
    doc.new_page(width=595, height=842)

    p3 = doc.new_page(width=595, height=842)
    p3.insert_text((50, 72), "Content on page three.", fontsize=11)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def non_pdf_file(tmp_path: Path) -> Path:
    """Creates a plain text file with a .txt extension."""
    text_path = tmp_path / "notes.txt"
    text_path.write_text("This is not a PDF file.", encoding="utf-8")
    return text_path


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    """Creates a file with a .pdf extension containing invalid binary data."""
    bad_pdf_path = tmp_path / "corrupted.pdf"
    bad_pdf_path.write_bytes(b"%PDF-1.4\nSome random garbage that is not valid PDF syntax")
    return bad_pdf_path


@pytest.fixture
def unicode_characters_pdf(tmp_path: Path) -> Path:
    """Creates a 1-page PDF containing mathematical and Greek unicode symbols (\\u2217 and \\u03b1)."""
    pdf_path = tmp_path / "unicode_paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(
        pymupdf.Rect(50, 50, 500, 500),
        "<p>Attention weights \u2217 with learning rate \u03b1 parameter.</p>",
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

