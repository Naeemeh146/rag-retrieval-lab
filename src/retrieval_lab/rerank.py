"""Reranker interface and implementations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class Reranker(ABC):
    """Interface for second-stage rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, doc_ids: list[str], texts: list[str], top_n: int
    ) -> list[tuple[str, float]]:
        """Rerank candidate documents for ``query``.

        Args:
            query: Query text.
            doc_ids: Candidate document identifiers, in first-stage order.
            texts: Candidate document texts, aligned with ``doc_ids``.
            top_n: Maximum number of results to return.

        Returns:
            ``(doc_id, score)`` pairs sorted by descending relevance score.
        """
        raise NotImplementedError


class NoOpReranker(Reranker):
    """Reranker that preserves the first-stage order.

    Useful for ablations (measuring the marginal value of a real reranker)
    and for fully offline runs.
    """

    def rerank(
        self, query: str, doc_ids: list[str], texts: list[str], top_n: int
    ) -> list[tuple[str, float]]:
        """Return candidates in their original order with decaying scores."""
        _ = (query, texts)
        return [(doc_id, float(len(doc_ids) - i)) for i, doc_id in enumerate(doc_ids)][:top_n]


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker (default: ms-marco-MiniLM-L-6-v2).

    Scores each ``(query, document)`` pair jointly. Requires the
    ``sentence-transformers`` package and downloads the model weights on
    first use.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Args:
        model_name: Hugging Face cross-encoder identifier.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install 'retrieval-lab[models]'"
            ) from exc
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, doc_ids: list[str], texts: list[str], top_n: int
    ) -> list[tuple[str, float]]:
        """Score ``(query, doc)`` pairs and return the top ``top_n``."""
        if not doc_ids:
            return []
        scores = self._model.predict([(query, text) for text in texts])
        ranked = sorted(
            zip(doc_ids, (float(s) for s in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_n]


class CohereReranker(Reranker):
    """Cohere Rerank API reranker.

    Requires the ``COHERE_API_KEY`` environment variable and the ``cohere``
    package.
    """

    def __init__(self, model: str = "rerank-english-v3.0", api_key: str | None = None) -> None:
        """Args:
        model: Cohere rerank model identifier.
        api_key: API key; defaults to the ``COHERE_API_KEY`` environment
            variable.

        Raises:
            RuntimeError: If no API key is available.
            ImportError: If the ``cohere`` package is not installed.
        """
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "CohereReranker requires an API key. Set the COHERE_API_KEY "
                "environment variable (see .env.example) or pass api_key explicitly."
            )
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "The 'cohere' package is required for CohereReranker. "
                "Install it with: pip install 'retrieval-lab[models]'"
            ) from exc
        self.model = model
        self._client = cohere.ClientV2(api_key=key)

    def rerank(
        self, query: str, doc_ids: list[str], texts: list[str], top_n: int
    ) -> list[tuple[str, float]]:
        """Rerank candidates via the Cohere Rerank API."""
        if not doc_ids:
            return []
        response = self._client.rerank(query=query, documents=texts, model=self.model, top_n=top_n)
        return [
            (doc_ids[result.index], float(result.relevance_score)) for result in response.results
        ]


def build_reranker(name: str, model: str = "") -> Reranker:
    """Instantiate a reranker by name.

    Args:
        name: One of ``"none"``, ``"cross-encoder"``, ``"cohere"``.
        model: Model identifier for learned rerankers (ignored for ``"none"``).

    Raises:
        ValueError: If ``name`` is unknown.
    """
    normalized = name.lower()
    if normalized == "none":
        return NoOpReranker()
    if normalized == "cross-encoder":
        return CrossEncoderReranker(model_name=model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if normalized == "cohere":
        return CohereReranker(model=model or "rerank-english-v3.0")
    raise ValueError(f"Unknown reranker {name!r}; expected 'none', 'cross-encoder' or 'cohere'")
