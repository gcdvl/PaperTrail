"""
retriever.py: BM25 ranked retrieval module for PaperTrail.

Provides Okapi BM25 scoring across passage chunks using the rank_bm25 package,
sensible lowercased tokenization, optional stopword filtering, and returns
ranked RetrievalResult objects with stable passage IDs and 1-based page numbers.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, List, Optional, Sequence, Set, Union

from rank_bm25 import BM25Okapi

from src.chunker import Chunk, chunk_document, chunk_pdf
from src.extractor import DocumentExtractionResult, extract_text_from_pdf


@dataclass(frozen=True)
class RetrievalResult:
    """
    Represents a ranked passage retrieved for a given query.

    Attributes:
        passage_id: Unique identifier of the underlying chunk ('<doc>::p<page>::c<index>').
        document_name: Filename of the source document.
        page_number: 1-based physical page number.
        passage: Text content of the retrieved passage.
        score: BM25 relevance score.
        rank: 1-based position in ranked order (1 = most relevant).
    """
    passage_id: str
    document_name: str
    page_number: int
    passage: str
    score: float
    rank: int


DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Lowercases and tokenizes text into word tokens using alphanumeric boundaries.

    Args:
        text: Input string to tokenize.
        remove_stopwords: If True, filters out standard English stopwords.
                          If all tokens are stopwords, preserves original tokens
                          so queries like 'what is it' are not emptied.

    Returns:
        List of cleaned token strings.
    """
    if not text:
        return []

    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    if not tokens:
        return []

    if remove_stopwords:
        filtered = [t for t in tokens if t not in DEFAULT_STOPWORDS]
        return filtered if filtered else tokens

    return tokens


def retrieve(
    passages: Sequence[Chunk],
    query: str,
    top_k: int = 5,
    remove_stopwords: bool = True,
) -> List[RetrievalResult]:
    """
    Ranks passages against a query using Okapi BM25.

    Args:
        passages: Sequence of Chunk objects to search.
        query: User question or search phrase.
        top_k: Maximum number of ranked results to return.
        remove_stopwords: Whether to remove stopwords during tokenization.

    Returns:
        List of RetrievalResult objects ranked descending by BM25 score.
        Returns [] if query is empty, corpus is empty, or no passages match.
    """
    if not passages or not query or not query.strip() or top_k <= 0:
        return []

    query_tokens = tokenize(query, remove_stopwords=remove_stopwords)
    if not query_tokens:
        return []

    # Tokenize corpus chunks
    corpus_tokens = [tokenize(c.text, remove_stopwords=remove_stopwords) for c in passages]

    # Guard against completely empty tokenized corpus
    if not any(corpus_tokens):
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    # A passage is a candidate match if it shares at least one query token
    query_token_set = set(query_tokens)
    matching_indices = [
        idx for idx, c_tokens in enumerate(corpus_tokens)
        if any(t in query_token_set for t in c_tokens)
    ]
    if not matching_indices:
        return []

    # Rank matching indices descending by score
    sorted_indices = sorted(matching_indices, key=lambda idx: scores[idx], reverse=True)[:top_k]

    results: List[RetrievalResult] = []
    for rank, idx in enumerate(sorted_indices, start=1):
        chunk = passages[idx]
        results.append(
            RetrievalResult(
                passage_id=chunk.passage_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                passage=chunk.text,
                score=float(scores[idx]),
                rank=rank,
            )
        )

    return results


def retrieve_from_document(
    document: DocumentExtractionResult,
    query: str,
    top_k: int = 5,
    min_chars: int = 100,
    max_chars: int = 1200,
) -> List[RetrievalResult]:
    """
    Chunks an extracted document and retrieves ranked passages.

    Args:
        document: DocumentExtractionResult from extractor.py.
        query: User question or search phrase.
        top_k: Maximum number of ranked results.
        min_chars: Chunker minimum character threshold for merging.
        max_chars: Chunker maximum character threshold for splitting.

    Returns:
        List of RetrievalResult objects.
    """
    chunks = chunk_document(document, min_chars=min_chars, max_chars=max_chars)
    return retrieve(chunks, query, top_k=top_k)


def retrieve_from_pdf(
    file_path: Union[str, Path],
    query: str,
    top_k: int = 5,
    min_chars: int = 100,
    max_chars: int = 1200,
) -> List[RetrievalResult]:
    """
    Convenience function that extracts text, chunks, and retrieves ranked passages.

    Args:
        file_path: Path to the local PDF file.
        query: User question or search phrase.
        top_k: Maximum number of ranked results.
        min_chars: Chunker minimum character threshold for merging.
        max_chars: Chunker maximum character threshold for splitting.

    Returns:
        List of RetrievalResult objects.
    """
    document = extract_text_from_pdf(file_path)
    return retrieve_from_document(document, query, top_k=top_k, min_chars=min_chars, max_chars=max_chars)
