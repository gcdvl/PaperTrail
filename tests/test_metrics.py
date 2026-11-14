"""
test_metrics.py: Unit tests for information retrieval evaluation metrics.

Tests hand-computed mathematical examples for:
1. recall_at_k (hits at k=1, k=2, k=3, misses, empty inputs).
2. reciprocal_rank (rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.333, miss -> 0.0).
3. mean_reciprocal_rank (averaging across query runs).
4. mean_recall_at_k (averaging recall values across query runs).
"""

import pytest

from src.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


class TestMetrics:
    """Hand-computed mathematical verification of IR metrics."""

    def test_recall_at_k_hand_computed(self):
        """Verifies Recall@k on known sequences."""
        retrieved = [1, 2, 3, 4, 5]

        # Case 1: Target is at index 0 (page 1)
        assert recall_at_k(retrieved, [1], k=1) == 1.0
        assert recall_at_k(retrieved, [1], k=3) == 1.0

        # Case 2: Target is at index 1 (page 2)
        assert recall_at_k(retrieved, [2], k=1) == 0.0
        assert recall_at_k(retrieved, [2], k=2) == 1.0
        assert recall_at_k(retrieved, [2], k=5) == 1.0

        # Case 3: Target is at index 3 (page 4)
        assert recall_at_k(retrieved, [4], k=1) == 0.0
        assert recall_at_k(retrieved, [4], k=3) == 0.0
        assert recall_at_k(retrieved, [4], k=4) == 1.0

        # Case 4: Target not in retrieved list
        assert recall_at_k(retrieved, [99], k=5) == 0.0

        # Case 5: Multiple expected pages (hit if any is retrieved in top-k)
        assert recall_at_k(retrieved, [10, 2], k=1) == 0.0
        assert recall_at_k(retrieved, [10, 2], k=2) == 1.0

    def test_recall_at_k_edge_cases(self):
        """Verifies boundary conditions return 0.0 without errors."""
        assert recall_at_k([], [1], k=5) == 0.0
        assert recall_at_k([1, 2], [], k=5) == 0.0
        assert recall_at_k([1, 2], [1], k=0) == 0.0
        assert recall_at_k([1, 2], [1], k=-1) == 0.0

    def test_reciprocal_rank_hand_computed(self):
        """Verifies 1/rank calculations on known positions."""
        retrieved = [10, 20, 30, 40]

        # Rank 1: 1 / 1 = 1.0
        assert reciprocal_rank(retrieved, [10]) == 1.0

        # Rank 2: 1 / 2 = 0.5
        assert reciprocal_rank(retrieved, [20]) == 0.5

        # Rank 3: 1 / 3 ≈ 0.3333333333333333
        assert pytest.approx(reciprocal_rank(retrieved, [30]), 0.0001) == 1.0 / 3.0

        # Rank 4: 1 / 4 = 0.25
        assert reciprocal_rank(retrieved, [40]) == 0.25

        # Miss: 0.0
        assert reciprocal_rank(retrieved, [99]) == 0.0

        # Empty inputs
        assert reciprocal_rank([], [10]) == 0.0
        assert reciprocal_rank([10], []) == 0.0

    def test_mean_reciprocal_rank_hand_computed(self):
        """Verifies MRR averaging."""
        # Query 1: RR=1.0, Query 2: RR=0.5, Query 3: RR=0.0 -> Mean = 1.5 / 3 = 0.5
        rr_scores = [1.0, 0.5, 0.0]
        assert mean_reciprocal_rank(rr_scores) == 0.5

        # Empty list returns 0.0
        assert mean_reciprocal_rank([]) == 0.0

    def test_mean_recall_at_k_hand_computed(self):
        """Verifies average Recall@k computation."""
        # 3 queries: 1.0, 1.0, 0.0 -> Mean = 2.0 / 3 ≈ 0.6667
        recalls = [1.0, 1.0, 0.0]
        assert pytest.approx(mean_recall_at_k(recalls), 0.0001) == 2.0 / 3.0

        # Empty list returns 0.0
        assert mean_recall_at_k([]) == 0.0
