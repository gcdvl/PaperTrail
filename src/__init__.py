"""PaperTrail: Verifiable PDF research paper citation and question-answering system."""

from src.extractor import (
    DocumentExtractionResult,
    InvalidPDFError,
    PageContent,
    PDFExtractionError,
    PDFFileNotFoundError,
    extract_text_from_pdf,
)
from src.search import (
    SearchResult,
    search_document,
    search_pages,
    search_pdf,
    split_into_passages,
)
from src.chunker import (
    Chunk,
    chunk_document,
    chunk_page,
    chunk_pages,
    chunk_pdf,
)
from src.retriever import (
    RetrievalResult,
    retrieve,
    retrieve_from_document,
    retrieve_from_pdf,
    tokenize,
)
from src.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    normalize_whitespace,
    passage_contains_evidence,
    passage_recall_at_k,
    passage_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    score_distribution_stats,
    compute_answer_eval_metrics,
    summarize_answer_metrics,
)

from src.dense_retriever import (
    DenseRetriever,
    retrieve_dense,
)
from src.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    retrieve_hybrid,
)

__all__ = [
    "Chunk",
    "compute_answer_eval_metrics",
    "DenseRetriever",
    "DocumentExtractionResult",
    "HybridRetriever",
    "InvalidPDFError",
    "PageContent",
    "PDFExtractionError",
    "PDFFileNotFoundError",
    "RetrievalResult",
    "SearchResult",
    "chunk_document",
    "chunk_page",
    "chunk_pages",
    "chunk_pdf",
    "extract_text_from_pdf",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "normalize_whitespace",
    "passage_contains_evidence",
    "passage_recall_at_k",
    "passage_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
    "reciprocal_rank_fusion",
    "retrieve",
    "retrieve_dense",
    "retrieve_from_document",
    "retrieve_from_pdf",
    "retrieve_hybrid",
    "score_distribution_stats",
    "search_document",
    "search_pages",
    "search_pdf",
    "split_into_passages",
    "summarize_answer_metrics",
    "tokenize",
]

