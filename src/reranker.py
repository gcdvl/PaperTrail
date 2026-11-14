"""
reranker.py: Cross-encoder reranking for PaperTrail retrieval results.

Provides a CrossEncoderReranker that rescores BM25 candidates using a
cross-encoder model, and a convenience function retrieve_with_rerank
that chains BM25 retrieval with reranking.

sentence_transformers is NOT imported at module level so that tests and
lightweight usage never trigger a model download.
"""

from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple

from src.retriever import RetrievalResult, retrieve
from src.chunker import Chunk

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_POOL = 20


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder scoring function.

    Args:
        model_name: Identifier for the cross-encoder model. Used only when
            scorer is None and the model is loaded lazily.
        scorer: Optional callable that takes a list of (query, passage) pairs
            and returns a list of float scores. When provided, no model is
            loaded; this is the primary mechanism for offline testing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        scorer: Optional[Callable[[List[Tuple[str, str]]], List[float]]] = None,
    ) -> None:
        self._model_name = model_name
        self._scorer = scorer
        self._model = None  # lazy-loaded CrossEncoder instance

    def _get_scorer(self) -> Callable[[List[Tuple[str, str]]], List[float]]:
        """Returns the scoring callable, loading the model lazily if needed."""
        if self._scorer is not None:
            return self._scorer

        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

        model = self._model

        def _model_scorer(pairs: List[Tuple[str, str]]) -> List[float]:
            scores = model.predict(pairs)
            return [float(s) for s in scores]

        return _model_scorer

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Reranks retrieval results by cross-encoder score.

        Args:
            query: The user query string.
            results: Sequence of RetrievalResult objects from BM25 retrieval.
            top_k: Maximum number of results to return.

        Returns:
            New list of RetrievalResult objects sorted by rerank score
            descending, with ties broken by original input order. Each
            result has score set to the rerank score and rank set to its
            1-based position. Input objects are never mutated.
        """
        if not results or top_k <= 0:
            return []

        scorer = self._get_scorer()

        pairs: List[Tuple[str, str]] = [(query, r.passage) for r in results]
        scores = scorer(pairs)

        # Build (rerank_score, original_index, result) tuples.
        # Sorting by (-score, original_index) ensures descending score with
        # ties broken by the original BM25 order (position in input list).
        indexed = [
            (-scores[i], i, results[i])
            for i in range(len(results))
        ]
        indexed.sort(key=lambda t: (t[0], t[1]))

        reranked: List[RetrievalResult] = []
        for rank, (neg_score, _orig_idx, orig) in enumerate(indexed[:top_k], start=1):
            reranked.append(
                replace(orig, score=-neg_score, rank=rank)
            )

        return reranked


def retrieve_with_rerank(
    chunks: Sequence[Chunk],
    query: str,
    reranker: CrossEncoderReranker,
    top_k: int = 5,
    pool: int = DEFAULT_POOL,
) -> List[RetrievalResult]:
    """
    BM25 retrieval followed by cross-encoder reranking.

    Args:
        chunks: Document chunks to search.
        query: User query string.
        reranker: CrossEncoderReranker instance.
        top_k: Number of final results after reranking.
        pool: Number of BM25 candidates to feed into the reranker.

    Returns:
        List of at most top_k RetrievalResult objects, reranked.
    """
    candidates = retrieve(chunks, query, top_k=pool)
    return reranker.rerank(query, candidates, top_k=top_k)

