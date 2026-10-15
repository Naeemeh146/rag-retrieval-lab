"""Unit tests for ranking metrics (hand-computed expectations)."""

import math

import pytest

from retrieval_lab.metrics import compute_metrics, mrr_at_k, ndcg_at_k, recall_at_k

RANKED = ["d1", "d2", "d3"]
RELEVANT = {"d2", "d4"}


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.0)
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.5)
    assert recall_at_k(RANKED, set(), 3) == pytest.approx(0.0)
    assert recall_at_k(RANKED, RELEVANT, 0) == pytest.approx(0.0)


def test_mrr_at_k() -> None:
    assert mrr_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.0)
    assert mrr_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)
    assert mrr_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.5)
    assert mrr_at_k(["d4", "d1"], RELEVANT, 2) == pytest.approx(1.0)


def test_ndcg_at_k() -> None:
    # gains=[0, 1]; dcg = 1/log2(3); idcg = 1 + 1/log2(3)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx(expected)
    # Perfect ranking -> 1.0
    assert ndcg_at_k(["d2", "d4", "d1"], RELEVANT, 3) == pytest.approx(1.0)
    # No relevant retrieved -> 0.0
    assert ndcg_at_k(["d1", "d3"], RELEVANT, 2) == pytest.approx(0.0)
    assert ndcg_at_k(RANKED, set(), 2) == pytest.approx(0.0)


def test_compute_metrics_averages_over_queries() -> None:
    results = {"q1": ["d2", "d1"], "q2": ["d9"]}
    qrels = {"q1": {"d2"}, "q2": {"d8"}}
    metrics = compute_metrics(results, qrels, ks=[2])
    # q1: recall=1.0, mrr=1.0, ndcg=1.0 ; q2: all 0.0
    assert metrics["recall@2"] == pytest.approx(0.5)
    assert metrics["mrr@2"] == pytest.approx(0.5)
    assert metrics["ndcg@2"] == pytest.approx(0.5)


def test_compute_metrics_rejects_empty_ks() -> None:
    with pytest.raises(ValueError, match="ks must not be empty"):
        compute_metrics({"q1": ["d1"]}, {"q1": {"d1"}}, ks=[])
