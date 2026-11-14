"""
create_sample_paper.py: Utility to generate a sample 2-page PDF paper for quick local testing.
"""

from pathlib import Path
import pymupdf


def generate_sample_pdf(output_path: str = "sample_paper.pdf") -> Path:
    target = Path(output_path)
    doc = pymupdf.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 72),
        "Attention Is All You Need (Sample Mock Document)\n\n"
        "Abstract\n"
        "The dominant sequence transduction models are based on complex recurrent or\n"
        "convolutional neural networks that include an encoder and a decoder. We propose\n"
        "the Transformer, a model architecture eschewing recurrence and relying entirely\n"
        "on an attention mechanism to draw global dependencies between input and output.\n\n"
        "1. Introduction\n"
        "Recurrent neural networks, long short-term memory and gated recurrent neural\n"
        "networks have been firmly established as state of the art approaches in sequence\n"
        "modeling and transduction problems.",
        fontsize=11,
    )

    # Page 2
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 72),
        "3. Model Architecture\n"
        "Most competitive neural sequence transduction models have an encoder-decoder structure.\n"
        "Here, the encoder maps an input sequence of symbol representations to a sequence of\n"
        "continuous representations z = (z1, ..., zn).\n\n"
        "3.2 Attention\n"
        "An attention function can be described as mapping a query and a set of key-value\n"
        "pairs to an output, where the query, keys, values, and output are all vectors.\n\n"
        "References\n"
        "[1] Vaswani et al., 2017. Attention is all you need. In NeurIPS.",
        fontsize=11,
    )

    doc.save(str(target))
    doc.close()
    print(f"Sample PDF successfully created at: {target.resolve()}")
    return target


if __name__ == "__main__":
    generate_sample_pdf()
