"""Tests requiring model downloads or API keys (marked slow, skipped in CI)."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_sentence_transformer_provider() -> None:
    st = pytest.importorskip("sentence_transformers")
    _ = st
    from retrieval_lab.embeddings import SentenceTransformerProvider

    provider = SentenceTransformerProvider()
    vectors = provider.embed(["hello world", "information retrieval"])
    assert vectors.shape == (2, provider.dimension)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-5)


def test_cross_encoder_reranker_orders_by_relevance() -> None:
    pytest.importorskip("sentence_transformers")
    from retrieval_lab.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    ranked = reranker.rerank(
        "What is the capital of France?",
        ["d1", "d2"],
        ["Paris is the capital of France.", "Dogs are friendly animals."],
        top_n=2,
    )
    assert [doc_id for doc_id, _ in ranked] == ["d1", "d2"]
