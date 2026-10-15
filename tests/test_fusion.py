"""Unit tests for Reciprocal Rank Fusion (hand-computed expectations)."""

import pytest

from retrieval_lab.fusion import reciprocal_rank_fusion


def test_rrf_hand_computed_scores_and_order() -> None:
    rankings = [
        [("d1", 9.0), ("d2", 8.0), ("d3", 7.0)],
        [("d3", 5.0), ("d1", 4.0)],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60)
    # d1: 1/61 + 1/62 = 123/3782
    # d3: 1/63 + 1/61 = 124/3843
    # d2: 1/62
    assert [doc_id for doc_id, _ in fused] == ["d1", "d3", "d2"]
    assert fused[0][1] == pytest.approx(123 / 3782)
    assert fused[1][1] == pytest.approx(124 / 3843)
    assert fused[2][1] == pytest.approx(1 / 62)


def test_rrf_ignores_input_scores() -> None:
    rankings = [[("a", 0.1), ("b", 100.0)], [("b", 0.2), ("a", 50.0)]]
    fused = reciprocal_rank_fusion(rankings, k=60)
    # Both appear at ranks 1 and 2 -> identical fused scores, tie broken by id.
    assert [doc_id for doc_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_rrf_top_k_truncates() -> None:
    rankings = [[("d1", 1.0), ("d2", 1.0), ("d3", 1.0)]]
    fused = reciprocal_rank_fusion(rankings, k=60, top_k=2)
    assert [doc_id for doc_id, _ in fused] == ["d1", "d2"]


def test_rrf_empty_rankings() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
