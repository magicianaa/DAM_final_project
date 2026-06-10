from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    average_precision_at_k,
    coverage,
    diversity,
    hit_rate_at_k,
    mean_average_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_by_user,
)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

    def test_partial(self):
        assert precision_at_k([1, 2, 3], {1, 3, 5}, 3) == 2 / 3

    def test_none(self):
        assert precision_at_k([1, 2, 3], {4, 5}, 3) == 0.0

    def test_k_zero(self):
        assert precision_at_k([1, 2, 3], {1}, 0) == 0.0

    def test_k_larger_than_list(self):
        assert precision_at_k([1, 2], {1, 3}, 5) == 1 / 5


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

    def test_partial(self):
        assert recall_at_k([1, 2, 3], {1, 3, 5}, 3) == 2 / 3

    def test_empty_relevant(self):
        assert recall_at_k([1, 2, 3], set(), 3) == 0.0


class TestNDCGAtK:
    def test_perfect(self):
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

    def test_partial_relevant(self):
        val = ndcg_at_k([4, 2, 1], {1, 2}, 3)
        assert 0.0 < val < 1.0

    def test_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k([1, 2, 3], {3}, 3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k([1, 2, 3], {4}, 3) == 0.0


class TestAveragePrecisionAtK:
    def test_perfect(self):
        # hits at positions 1,2,3 -> (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert average_precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

    def test_partial(self):
        # hits at positions 1,3 -> (1/1 + 2/3) / 2 = 0.8333...
        result = average_precision_at_k([1, 2, 3], {1, 3}, 3)
        assert abs(result - 0.833333) < 1e-5

    def test_no_relevant(self):
        assert average_precision_at_k([1, 2, 3], {4, 5}, 3) == 0.0

    def test_k_zero(self):
        assert average_precision_at_k([1, 2, 3], {1}, 0) == 0.0


class TestMeanAveragePrecision:
    def test_basic(self):
        recs = [[1, 2, 3], [4, 5, 6]]
        rel = [{1, 3}, {4, 6}]
        result = mean_average_precision_at_k(recs, rel, 3)
        assert 0.0 < result <= 1.0

    def test_empty(self):
        assert mean_average_precision_at_k([], [], 3) == 0.0


class TestMeanReciprocalRank:
    def test_first_position(self):
        result = mean_reciprocal_rank([[1, 2, 3]], [{1}], 3)
        assert result == 1.0

    def test_third_position(self):
        result = mean_reciprocal_rank([[1, 2, 3]], [{3}], 3)
        assert result == 1.0 / 3

    def test_no_hit(self):
        result = mean_reciprocal_rank([[1, 2, 3]], [{4}], 3)
        assert result == 0.0


class TestCoverage:
    def test_full(self):
        assert coverage([[1, 2], [2, 3]], 3) == 1.0

    def test_partial(self):
        assert coverage([[1, 2], [1, 2]], 4) == 0.5

    def test_empty_catalog(self):
        assert coverage([[1, 2]], 0) == 0.0

    def test_empty_recs(self):
        assert coverage([], 10) == 0.0
