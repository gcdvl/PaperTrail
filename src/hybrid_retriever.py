"""
hybrid_retriever.py: Hybrid retrieval engine for PaperTrail combining BM25 and dense retrieval.

Implements Reciprocal Rank Fusion (RRF) with default k=60 to fuse BM25 keyword rankings
and dense vector semantic rankings, supporting optional per-retriever weighting.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.chunker import Chunk
from src.dense_retriever import DenseRetriever
from src.retriever import RetrievalResult, retrieve as bm25_retrieve

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]],
    weights: Optional[Sequence[float]] = None,
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """
    Fuses multiple ranked lists of RetrievalResult using Reciprocal Rank Fusion (RRF).

    Formula for document d:
        RRF_score(d) = sum_{m} (w_m / (k + rank_m(d)))

    Args:
        rankings: Sequence of ranked result lists (each from a distinct retrieval system).
        weights: Optional sequence of weights w_m for each ranking. Defaults to 1.0 each.
        k: Smoothing constant (default 60).
        top_k: Optional maximum number of fused results to return.

    Returns:
        Fused list of RetrievalResult objects ranked descending by RRF score.
    """
    if not rankings:
        return []

    if weights is None:
        eff_weights = [1.0] * len(rankings)
    else:
        if len(weights) != len(rankings):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of rankings ({len(rankings)})"
            )
        eff_weights = list(weights)

    # Accumulate RRF score and retain metadata per passage_id
    rrf_scores: Dict[str, float] = {}
    passage_metadata: Dict[str, Tuple[str, int, str]] = {}
    best_ranks: Dict[str, int] = {}

    for ranking, w in zip(rankings, eff_weights):
        for result in ranking:
            pid = result.passage_id
            rank = result.rank
            contribution = float(w) / (k + rank)

            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + contribution

            if pid not in passage_metadata:
                passage_metadata[pid] = (
                    result.document_name,
                    result.page_number,
                    result.passage,
                )

            if pid not in best_ranks or rank < best_ranks[pid]:
                best_ranks[pid] = rank

    if not rrf_scores:
        return []

    # Sort descending by RRF score; break ties by best individual rank, then passage_id
    sorted_pids = sorted(
        rrf_scores.keys(),
        key=lambda pid: (-rrf_scores[pid], best_ranks[pid], pid),
    )

    if top_k is not None and top_k > 0:
        sorted_pids = sorted_pids[:top_k]

    fused_results: List[RetrievalResult] = []
    for rank, pid in enumerate(sorted_pids, start=1):
        doc_name, page_num, text = passage_metadata[pid]
        fused_results.append(
            RetrievalResult(
                passage_id=pid,
                document_name=doc_name,
                page_number=page_num,
                passage=text,
                score=float(rrf_scores[pid]),
                rank=rank,
            )
        )

    return fused_results


class HybridRetriever:
    """
    Combines BM25 and dense retrieval using Reciprocal Rank Fusion.

    Attributes:
        bm25_retriever: Callable or function executing BM25 retrieval.
        dense_retriever: DenseRetriever instance executing vector retrieval.
        k: RRF smoothing constant (default: 60).
        weight_bm25: Weight assigned to BM25 ranking (default: 1.0).
        weight_dense: Weight assigned to dense ranking (default: 1.0).
    """

    def __init__(
        self,
        bm25_retriever: Optional[Callable[..., List[RetrievalResult]]] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        k: int = DEFAULT_RRF_K,
        weight_bm25: float = 1.0,
        weight_dense: float = 1.0,
    ):
        self.bm25_retriever = bm25_retriever if bm25_retriever is not None else bm25_retrieve
        self.dense_retriever = dense_retriever if dense_retriever is not None else DenseRetriever()
        self.k = k
        self.weight_bm25 = weight_bm25
        self.weight_dense = weight_dense

    def retrieve(
        self,
        passages: Sequence[Chunk],
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieves top_k passages using fused BM25 and dense retrieval.

        Args:
            passages: Sequence of Chunk objects.
            query: Query string.
            top_k: Maximum number of fused results to return.

        Returns:
            List of RetrievalResult objects.
        """
        if not passages or not query or not query.strip() or top_k <= 0:
            return []

        # Pull a wider candidate pool from both retrievers to ensure robust fusion
        candidate_k = max(top_k * 3, 30)

        bm25_results = self.bm25_retriever(passages, query, top_k=candidate_k)
        dense_results = self.dense_retriever.retrieve(passages, query, top_k=candidate_k)

        return reciprocal_rank_fusion(
            rankings=[bm25_results, dense_results],
            weights=[self.weight_bm25, self.weight_dense],
            k=self.k,
            top_k=top_k,
        )


def retrieve_hybrid(
    passages: Sequence[Chunk],
    query: str,
    top_k: int = 5,
    k: int = DEFAULT_RRF_K,
    weight_bm25: float = 1.0,
    weight_dense: float = 1.0,
    dense_retriever: Optional[DenseRetriever] = None,
) -> List[RetrievalResult]:
    """
    Convenience function executing hybrid BM25 + dense retrieval.

    Args:
        passages: Sequence of Chunk objects.
        query: Query string.
        top_k: Max ranked results to return.
        k: RRF smoothing constant (default: 60).
        weight_bm25: BM25 ranking weight.
        weight_dense: Dense ranking weight.
        dense_retriever: Optional configured DenseRetriever.

    Returns:
        Fused list of RetrievalResult objects.
    """
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        k=k,
        weight_bm25=weight_bm25,
        weight_dense=weight_dense,
    )
    return retriever.retrieve(passages, query, top_k=top_k)
