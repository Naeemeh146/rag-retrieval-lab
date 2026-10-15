"""Unit tests for providers and builders (external services mocked)."""

import numpy as np
import pytest

from retrieval_lab.chunking import build_chunker
from retrieval_lab.embeddings import (
    DeterministicEmbeddingProvider,
    build_embedding_provider,
)
from retrieval_lab.rerank import NoOpReranker, build_reranker


def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimension=64)
    first = provider.embed(["hello world", "hello world", "different text"])
    second = provider.embed(["hello world", "hello world", "different text"])
    assert provider.dimension == 64
    assert first.shape == (3, 64)
    np.testing.assert_allclose(first, second)
    # Identical texts embed identically; rows are L2-normalized.
    np.testing.assert_allclose(first[0], first[1])
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(3))
    # Distinct text usually yields a distinct vector.
    assert not np.allclose(first[0], first[2])


def test_deterministic_embeddings_reject_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="positive"):
        DeterministicEmbeddingProvider(dimension=0)


def test_cohere_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        build_embedding_provider("cohere")
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        build_reranker("cohere")


def test_builders_reject_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        build_embedding_provider("openai")
    with pytest.raises(ValueError, match="Unknown reranker"):
        build_reranker("bm25")
    with pytest.raises(ValueError, match="Unknown chunker"):
        build_chunker("sentence")


def test_noop_reranker_preserves_order() -> None:
    reranker = NoOpReranker()
    ranked = reranker.rerank("q", ["d1", "d2", "d3"], ["t1", "t2", "t3"], top_n=2)
    assert [doc_id for doc_id, _ in ranked] == ["d1", "d2"]
    assert ranked[0][1] > ranked[1][1]
