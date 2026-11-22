import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.extractor import extract_text_from_pdf

for pdf in sorted((ROOT / "eval" / "new" / "papers_new").glob("*.pdf")):
    pages = extract_text_from_pdf(pdf).pages
    sizes = [len(p.text) for p in pages]
    text = "\n".join(p.text for p in pages)
    print(f"{pdf.name}: pages={len(pages)} chars={sum(sizes)} "
          f"min_page={min(sizes)} short_pages(<800)={[p.page_number for p in pages if len(p.text) < 800]}")
    print(f"   mojibake={text.count('â€') + text.count('Ã')} "
          f"replacement_char={text.count(chr(0xfffd))} "
          f"line_end_hyphens={len(re.findall(r'[a-z]-\n[a-z]', text))}")