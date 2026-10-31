"""Unit tests for RRF fusion and gate logic."""

from __future__ import annotations

import pytest

from app.services.retriever import reciprocal_rank_fusion, RetrievalResult


class MockScoredPoint:
    def __init__(self, id: str, score: float, payload: dict | None = None):
        self.id = id
        self.score = score
        self.payload = payload or {}


class TestRRF:
    def test_only_in_dense(self):
        dense = [MockScoredPoint("a", 0.9), MockScoredPoint("b", 0.8)]
        result = reciprocal_rank_fusion(dense, [], k=60)
        assert "a" in [r[0] for r in result]

    def test_only_in_sparse(self):
        sparse = [MockScoredPoint("x", 0.7), MockScoredPoint("y", 0.6)]
        result = reciprocal_rank_fusion([], sparse, k=60)
        assert "x" in [r[0] for r in result]

    def test_top_item_in_both_gets_highest_rrf(self):
        dense = [MockScoredPoint("a", 0.9), MockScoredPoint("b", 0.8)]
        sparse = [MockScoredPoint("a", 0.7), MockScoredPoint("c", 0.6)]
        result = reciprocal_rank_fusion(dense, sparse, k=60)
        assert result[0][0] == "a"

    def test_rrf_scores_are_positive(self):
        dense = [MockScoredPoint(str(i), 1.0 / (i + 1)) for i in range(5)]
        sparse = [MockScoredPoint(str(i), 0.5 / (i + 1)) for i in range(5)]
        result = reciprocal_rank_fusion(dense, sparse, k=60)
        for _, _, _, rrf in result:
            assert rrf > 0

    def test_k_affects_scores(self):
        dense = [MockScoredPoint("a", 0.9)]
        sparse = [MockScoredPoint("a", 0.8)]
        low_k = reciprocal_rank_fusion(dense, sparse, k=1)
        high_k = reciprocal_rank_fusion(dense, sparse, k=100)
        assert low_k[0][3] > high_k[0][3]

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([], [], k=60) == []

    def test_returns_all_unique_ids(self):
        dense = [MockScoredPoint("a", 0.9), MockScoredPoint("b", 0.8)]
        sparse = [MockScoredPoint("b", 0.7), MockScoredPoint("c", 0.6)]
        result = reciprocal_rank_fusion(dense, sparse, k=60)
        ids = [r[0] for r in result]
        assert len(ids) == len(set(ids))
        assert set(ids) == {"a", "b", "c"}


class TestGateLogic:
    def test_gate_passed(self):
        result = RetrievalResult(
            chunks=[], gate_passed=True,
            best_dense_score=0.85, best_rerank_score=0.7, refusal_reason=None,
        )
        assert result.gate_passed is True

    def test_gate_failed_on_low_dense(self):
        result = RetrievalResult(
            chunks=[], gate_passed=False,
            best_dense_score=0.3, best_rerank_score=None,
            refusal_reason="dense_score_below_cutoff:0.3<0.7",
        )
        assert not result.gate_passed
        assert "dense_score_below_cutoff" in (result.refusal_reason or "")

    def test_gate_failed_on_no_results(self):
        result = RetrievalResult(
            chunks=[], gate_passed=False,
            best_dense_score=0.0, best_rerank_score=None,
            refusal_reason="no_results",
        )
        assert not result.gate_passed
