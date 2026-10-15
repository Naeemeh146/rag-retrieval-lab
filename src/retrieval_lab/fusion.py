"""Rank fusion: Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Each document's fused score is ``sum(1 / (k + rank))`` over the rankings
    in which it appears (ranks are 1-based). Input scores are ignored — only
    the order within each ranking matters, which is what makes RRF robust
    when combining lexical and dense scores on incomparable scales.

    Args:
        rankings: Ranked ``(doc_id, score)`` lists, best first.
        k: RRF smoothing constant damping the influence of lower ranks.
        top_k: Maximum number of fused results; ``None`` returns all.

    Returns:
        ``(doc_id, fused_score)`` pairs sorted by descending fused score.
        Ties are broken by ``doc_id`` for determinism.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        return ordered[:top_k]
    return ordered
