"""
test_dense_hybrid.py: Unit tests for dense and hybrid retrieval engines.

Verifies:
1. Dense ranking order based on cosine similarity.
2. top_k restrictions and 1-based ranks.
3. Clean handling of empty queries and empty corpora.
4. On-disk and in-memory passage embedding caching (cache hits).
5. BGE query prefix application and avoidance of double-prefixing.
6. Reciprocal Rank Fusion (RRF k=60) mathematical behavior, ranking, and weighting.
7. HybridRetriever end-to-end integration with stub embedders.

All tests use a lightweight StubEmbedder to execute offline in milliseconds without model downloads.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pytest

from src.chunker import Chunk
from src.dense_retriever import (
    BGE_QUERY_PREFIX,
    DenseRetriever,
    compute_corpus_cache_key,
    retrieve_dense,
)
from src.hybrid_retriever import (
    DEFAULT_RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion,
    retrieve_hybrid,
)
from src.retriever import RetrievalResult


class StubEmbedder:
    """Deterministic, lightweight stub embedder mapping tokens to unit vectors."""

    def __init__(self, vocab_dims: Optional[Dict[str, int]] = None, dim: int = 8):
        self.dim = dim
        self.call_count = 0
        self.encoded_queries: List[str] = []
        self.vocab_dims = vocab_dims or {
            "attention": 0,
            "transformer": 1,
            "residual": 2,
            "resnet": 3,
            "convolution": 4,
            "bert": 5,
        }

    def encode(self, texts: Union[str, Sequence[str]], **kwargs) -> np.ndarray:
        self.call_count += 1
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        vectors = []
        for text in text_list:
            if is_single:
                self.encoded_queries.append(text)
            v = np.zeros(self.dim, dtype=np.float32)
            lower = text.lower()
            for word, idx in self.vocab_dims.items():
                if word in lower:
                    v[idx] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)

        return vectors[0] if is_single else np.array(vectors, dtype=np.float32)


@pytest.fixture
def sample_dense_corpus() -> List[Chunk]:
    """Provides a synthetic 4-chunk corpus for deterministic dense retrieval testing."""
    return [
        Chunk(
            passage_id="doc.pdf::p1::c1",
            document_name="doc.pdf",
            page_number=1,
            text="The Transformer model relies on attention mechanisms for sequential tasks.",
            char_count=75,
        ),
        Chunk(
            passage_id="doc.pdf::p1::c2",
            document_name="doc.pdf",
            page_number=1,
            text="Convolution layers apply spatial filters across 2D image matrices.",
            char_count=67,
        ),
        Chunk(
            passage_id="doc.pdf::p2::c1",
            document_name="doc.pdf",
            page_number=2,
            text="Residual learning frameworks train deeper neural networks easily.",
            char_count=66,
        ),
        Chunk(
            passage_id="doc.pdf::p2::c2",
            document_name="doc.pdf",
            page_number=2,
            text="BERT uses masked language modeling and bidirectional representation.",
            char_count=69,
        ),
    ]


# ---------------------------------------------------------------------------
# Dense Retriever Tests
# ---------------------------------------------------------------------------

class TestDenseRetriever:
    """Unit tests for dense vector retrieval and cosine similarity ranking."""

    def test_ranking_order(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """Passage with highest vector cosine similarity is ranked #1."""
        stub = StubEmbedder()
        retriever = DenseRetriever(embedder=stub, cache_dir=tmp_path)

        results = retriever.retrieve(sample_dense_corpus, query="transformer attention", top_k=4)

        assert len(results) > 0
        assert results[0].passage_id == "doc.pdf::p1::c1"
        assert results[0].rank == 1
        assert "Transformer" in results[0].passage
        # Scores are in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_bounds(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """top_k restricts the number of returned results and ranks are sequential 1-based."""
        stub = StubEmbedder()
        retriever = DenseRetriever(embedder=stub, cache_dir=tmp_path)

        res_1 = retriever.retrieve(sample_dense_corpus, query="neural networks", top_k=1)
        res_2 = retriever.retrieve(sample_dense_corpus, query="neural networks", top_k=2)

        assert len(res_1) == 1
        assert res_1[0].rank == 1
        assert len(res_2) == 2
        assert res_2[0].rank == 1
        assert res_2[1].rank == 2

    def test_empty_query_returns_empty_list(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """Empty or whitespace queries return [] without raising or invoking embedder."""
        stub = StubEmbedder()
        retriever = DenseRetriever(embedder=stub, cache_dir=tmp_path)

        assert retriever.retrieve(sample_dense_corpus, "") == []
        assert retriever.retrieve(sample_dense_corpus, "   ") == []
        assert retriever.retrieve(sample_dense_corpus, "\n\t") == []

    def test_empty_corpus_returns_empty_list(self, tmp_path: Path):
        """Retrieval against an empty corpus returns []."""
        stub = StubEmbedder()
        retriever = DenseRetriever(embedder=stub, cache_dir=tmp_path)
        assert retriever.retrieve([], "transformer") == []

    def test_top_k_zero_or_negative_returns_empty(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """Requesting top_k <= 0 returns []."""
        stub = StubEmbedder()
        retriever = DenseRetriever(embedder=stub, cache_dir=tmp_path)
        assert retriever.retrieve(sample_dense_corpus, "transformer", top_k=0) == []
        assert retriever.retrieve(sample_dense_corpus, "transformer", top_k=-1) == []

    def test_disk_and_memory_cache_hit(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """Embeddings are cached on disk and reused across retriever instances without re-encoding."""
        cache_dir = tmp_path / "embeddings_cache"
        stub1 = StubEmbedder()
        retriever1 = DenseRetriever(embedder=stub1, cache_dir=cache_dir)

        # 1. First retrieval computes embeddings and creates disk cache
        embs1 = retriever1.get_passage_embeddings(sample_dense_corpus)
        assert stub1.call_count == 1
        cache_key = compute_corpus_cache_key(sample_dense_corpus, retriever1.model_name)
        cache_file = cache_dir / f"{cache_key}.npy"
        assert cache_file.exists(), "Cache file must be created on disk"

        # 2. In-memory cache hit on the same instance
        embs_mem = retriever1.get_passage_embeddings(sample_dense_corpus)
        assert stub1.call_count == 1
        np.testing.assert_array_equal(embs1, embs_mem)

        # 3. New retriever instance with fresh memory cache loads from disk
        stub2 = StubEmbedder()
        retriever2 = DenseRetriever(embedder=stub2, cache_dir=cache_dir)
        embs2 = retriever2.get_passage_embeddings(sample_dense_corpus)

        assert stub2.call_count == 0, "Disk cache hit should not call embedder"
        np.testing.assert_array_equal(embs1, embs2)

    def test_cache_key_changes_when_corpus_changes(self, sample_dense_corpus: List[Chunk]):
        """Modifying a chunk text changes the cache key."""
        key1 = compute_corpus_cache_key(sample_dense_corpus, "model_a")
        modified_corpus = list(sample_dense_corpus)
        modified_corpus[0] = Chunk(
            passage_id=sample_dense_corpus[0].passage_id,
            document_name=sample_dense_corpus[0].document_name,
            page_number=sample_dense_corpus[0].page_number,
            text="Completely different text content.",
            char_count=32,
        )
        key2 = compute_corpus_cache_key(modified_corpus, "model_a")
        assert key1 != key2

    def test_bge_query_prefix_applied(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """BGE models automatically prepend recommended query prefix."""
        stub = StubEmbedder()
        retriever = DenseRetriever(
            model_name="BAAI/bge-small-en-v1.5",
            embedder=stub,
            cache_dir=tmp_path,
        )

        retriever.retrieve(sample_dense_corpus, query="what is attention?", top_k=2)

        assert len(stub.encoded_queries) > 0
        expected = f"{BGE_QUERY_PREFIX}what is attention?"
        assert stub.encoded_queries[-1] == expected

    def test_bge_query_prefix_no_double_prefix(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """If query already contains prefix, it is not prepended twice."""
        stub = StubEmbedder()
        retriever = DenseRetriever(
            model_name="BAAI/bge-small-en-v1.5",
            embedder=stub,
            cache_dir=tmp_path,
        )
        query = f"{BGE_QUERY_PREFIX}what is attention?"
        retriever.retrieve(sample_dense_corpus, query=query, top_k=2)

        assert stub.encoded_queries[-1] == query

    def test_convenience_retrieve_dense(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """retrieve_dense convenience function executes cleanly."""
        stub = StubEmbedder()
        results = retrieve_dense(
            sample_dense_corpus,
            query="residual",
            top_k=1,
            embedder=stub,
            cache_dir=tmp_path,
        )
        assert len(results) == 1
        assert "Residual" in results[0].passage


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion & Hybrid Retriever Tests
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    """Tests for mathematical correctness of Reciprocal Rank Fusion."""

    def test_rrf_hand_computed_scores(self):
        """Verifies exact RRF scores and ranking with k=60."""
        # Ranking 1 (BM25): doc1 @ rank 1, doc2 @ rank 2
        r1 = [
            RetrievalResult("doc1", "paper.pdf", 1, "Text 1", score=5.0, rank=1),
            RetrievalResult("doc2", "paper.pdf", 2, "Text 2", score=3.0, rank=2),
        ]
        # Ranking 2 (Dense): doc2 @ rank 1, doc3 @ rank 2
        r2 = [
            RetrievalResult("doc2", "paper.pdf", 2, "Text 2", score=0.9, rank=1),
            RetrievalResult("doc3", "paper.pdf", 3, "Text 3", score=0.8, rank=2),
        ]

        fused = reciprocal_rank_fusion([r1, r2], k=60)

        # Hand calculations:
        # doc1: 1/(60+1) = 1/61 ≈ 0.0163934
        # doc2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.0161290 + 0.0163934 = 0.0325224
        # doc3: 1/(60+2) = 1/62 ≈ 0.0161290
        # Expected rank order: doc2 (rank 1), doc1 (rank 2), doc3 (rank 3)
        assert len(fused) == 3
        assert fused[0].passage_id == "doc2"
        assert fused[0].rank == 1
        assert pytest.approx(fused[0].score, rel=1e-4) == 0.0325224

        assert fused[1].passage_id == "doc1"
        assert fused[1].rank == 2
        assert pytest.approx(fused[1].score, rel=1e-4) == 1.0 / 61.0

        assert fused[2].passage_id == "doc3"
        assert fused[2].rank == 3
        assert pytest.approx(fused[2].score, rel=1e-4) == 1.0 / 62.0

    def test_rrf_with_weights(self):
        """Verifies that per-retriever weights properly scale reciprocal contributions."""
        # BM25 weight 2.0, Dense weight 1.0
        r_bm25 = [
            RetrievalResult("doc_bm25", "paper.pdf", 1, "Text BM25", score=5.0, rank=1),
        ]
        r_dense = [
            RetrievalResult("doc_dense", "paper.pdf", 2, "Text Dense", score=0.9, rank=1),
        ]

        fused = reciprocal_rank_fusion([r_bm25, r_dense], weights=[2.0, 1.0], k=60)

        # doc_bm25 score: 2.0 / 61 ≈ 0.0327869
        # doc_dense score: 1.0 / 61 ≈ 0.0163934
        assert fused[0].passage_id == "doc_bm25"
        assert fused[1].passage_id == "doc_dense"
        assert pytest.approx(fused[0].score, rel=1e-4) == 2.0 / 61.0
        assert pytest.approx(fused[1].score, rel=1e-4) == 1.0 / 61.0

    def test_rrf_empty_rankings(self):
        """Fusing empty rankings safely returns []."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_rrf_top_k_cutoff(self):
        """top_k restricts maximum number of fused results."""
        r1 = [
            RetrievalResult(f"d{i}", "p.pdf", 1, f"T{i}", 1.0, i)
            for i in range(1, 10)
        ]
        fused = reciprocal_rank_fusion([r1], top_k=3)
        assert len(fused) == 3
        assert [r.rank for r in fused] == [1, 2, 3]


class TestHybridRetriever:
    """Integration tests for HybridRetriever with BM25 and stub dense retriever."""

    def test_hybrid_retrieve_end_to_end(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """HybridRetriever combines BM25 and dense candidate rankings."""
        stub = StubEmbedder()
        dense_ret = DenseRetriever(embedder=stub, cache_dir=tmp_path)
        hybrid = HybridRetriever(dense_retriever=dense_ret, k=60)

        results = hybrid.retrieve(sample_dense_corpus, query="Transformer attention", top_k=3)

        assert len(results) > 0
        assert results[0].rank == 1
        assert results[0].passage_id == "doc.pdf::p1::c1"
        assert results[0].score > 0

    def test_hybrid_empty_query(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """Empty query returns [] cleanly in hybrid retrieval."""
        stub = StubEmbedder()
        dense_ret = DenseRetriever(embedder=stub, cache_dir=tmp_path)
        hybrid = HybridRetriever(dense_retriever=dense_ret)

        assert hybrid.retrieve(sample_dense_corpus, "") == []
        assert hybrid.retrieve(sample_dense_corpus, "   ") == []

    def test_convenience_retrieve_hybrid(self, sample_dense_corpus: List[Chunk], tmp_path: Path):
        """retrieve_hybrid convenience function works as expected."""
        stub = StubEmbedder()
        dense_ret = DenseRetriever(embedder=stub, cache_dir=tmp_path)

        results = retrieve_hybrid(
            sample_dense_corpus,
            query="BERT representation",
            top_k=2,
            dense_retriever=dense_ret,
        )

        assert len(results) == 2
        assert results[0].rank == 1
        assert "BERT" in results[0].passage


# ---------------------------------------------------------------------------
# Compare Runs Script Tests
# ---------------------------------------------------------------------------

class TestCompareRuns:
    """Tests for scripts/compare_runs.py formatting and win/loss item analysis."""

    def test_compare_runs_formatting_and_item_outcomes(self):
        from scripts.compare_runs import compare_item_outcomes, format_table

        run_a = (
            "BM25",
            {
                "summary": {
                    "retriever": "bm25",
                    "page_level": {"mean_recall@1": 0.8, "mean_recall@3": 0.9, "mean_recall@5": 1.0, "mean_reciprocal_rank": 0.85},
                    "passage_level": {"mean_recall@1": 0.6, "mean_recall@3": 0.7, "mean_recall@5": 0.8, "mean_reciprocal_rank": 0.65},
                },
                "details": [
                    {"id": "q1", "question": "Question 1", "answerable": True, "passage_recall@1": 1.0, "evidence_phrase": "ev 1"},
                    {"id": "q2", "question": "Question 2", "answerable": True, "passage_recall@1": 0.0, "evidence_phrase": "ev 2"},
                ],
            },
        )
        run_b = (
            "DENSE",
            {
                "summary": {
                    "retriever": "dense",
                    "model_name": "BAAI/bge-small-en-v1.5",
                    "page_level": {"mean_recall@1": 0.75, "mean_recall@3": 0.85, "mean_recall@5": 0.95, "mean_reciprocal_rank": 0.80},
                    "passage_level": {"mean_recall@1": 0.65, "mean_recall@3": 0.75, "mean_recall@5": 0.85, "mean_reciprocal_rank": 0.70},
                },
                "details": [
                    {"id": "q1", "question": "Question 1", "answerable": True, "passage_recall@1": 0.0, "evidence_phrase": "ev 1"},
                    {"id": "q2", "question": "Question 2", "answerable": True, "passage_recall@1": 1.0, "evidence_phrase": "ev 2"},
                ],
            },
        )

        table_str = format_table([run_a, run_b])
        assert "BM25" in table_str
        assert "DENSE" in table_str
        assert "Passage Recall@1" in table_str

        outcomes_str = compare_item_outcomes(run_a, run_b, k=1)
        assert "[q1] Question 1" in outcomes_str
        assert "[q2] Question 2" in outcomes_str
        assert "Items RIGHT in [BM25] but WRONG in [DENSE] (1 items):" in outcomes_str
        assert "Items RIGHT in [DENSE] but WRONG in [BM25] (1 items):" in outcomes_str
