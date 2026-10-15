"""Ranking metrics implemented in numpy (no pytrec_eval C dependency)."""

from __future__ import annotations

import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents retrieved within the top ``k``.

    Returns 0.0 when ``relevant`` is empty or ``k`` is not positive.
    """
    if not relevant or k <= 0:
        return 0.0
    retrieved = set(ranked[:k])
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Mean reciprocal rank: 1 / rank of the first relevant document.

    Returns 0.0 when no relevant document appears within the top ``k``.
    """
    if not relevant or k <= 0:
        return 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance.

    ``DCG@k = sum(rel_i / log2(i + 2))`` over the top ``k`` ranks;
    ``nDCG@k = DCG@k / IDCG@k`` where the ideal ranking places all relevant
    documents first. Returns 0.0 when ``relevant`` is empty or ``k`` is not
    positive.
    """
    if not relevant or k <= 0:
        return 0.0
    gains = np.array(
        [1.0 if doc_id in relevant else 0.0 for doc_id in ranked[:k]],
        dtype=np.float64,
    )
    discounts = np.log2(np.arange(2, 2 + len(gains), dtype=np.float64))
    dcg = float(np.sum(gains / discounts))
    ideal_hits = min(len(relevant), k)
    ideal_discounts = np.log2(np.arange(2, 2 + ideal_hits, dtype=np.float64))
    idcg = float(np.sum(1.0 / ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    results: dict[str, list[str]],
    qrels: dict[str, set[str]],
    ks: list[int],
) -> dict[str, float]:
    """Average ranking metrics over queries.

    Args:
        results: Mapping of query id to ranked document ids.
        qrels: Mapping of query id to relevant document ids.
        ks: Cutoff values, e.g. ``[5, 10]``.

    Returns:
        Mapping like ``{"recall@5": 0.83, "mrr@5": 0.71, "ndcg@5": 0.79, ...}``
        with means over the queries present in ``results``.
    """
    if not ks:
        raise ValueError("ks must not be empty")
    totals: dict[str, float] = {}
    n_queries = 0
    for query_id, ranked in results.items():
        relevant = qrels.get(query_id, set())
        n_queries += 1
        for k in ks:
            totals[f"recall@{k}"] = totals.get(f"recall@{k}", 0.0) + recall_at_k(
                ranked, relevant, k
            )
            totals[f"mrr@{k}"] = totals.get(f"mrr@{k}", 0.0) + mrr_at_k(ranked, relevant, k)
            totals[f"ndcg@{k}"] = totals.get(f"ndcg@{k}", 0.0) + ndcg_at_k(ranked, relevant, k)
    if n_queries == 0:
        return dict.fromkeys(totals, 0.0)
    return {name: total / n_queries for name, total in totals.items()}
