"""Embedding provider interface and implementations."""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(ABC):
    """Interface for text embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: Input texts.

        Returns:
            Array of shape ``(len(texts), dimension)`` with L2-normalized rows.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension produced by this provider."""
        raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by offline components."""
    return _TOKEN_RE.findall(text.lower())


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Hash-based bag-of-words embeddings.

    Each token is hashed into a fixed-size vector and term counts are
    accumulated, then L2-normalized. Fully offline and deterministic, which
    makes it ideal for unit tests, demos, and reproducible ablations. It is
    *not* a semantic embedding model — use it to validate plumbing, not to
    measure real dense-retrieval quality.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Args:
        dimension: Output vector size. Must be positive.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts with hashed term counts (L2-normalized)."""
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self._dimension
                vectors[row, index] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings (default: all-MiniLM-L6-v2).

    Requires the ``sentence-transformers`` package and downloads the model
    weights on first use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Args:
        model_name: Hugging Face model identifier.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install it with: pip install 'retrieval-lab[models]'"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # `get_embedding_dimension` is the current name; fall back to the
        # legacy `get_sentence_embedding_dimension` on older releases.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        dimension = (
            get_dim() if callable(get_dim) else self._model.get_sentence_embedding_dimension()
        )
        if dimension is None:
            raise RuntimeError(f"Could not determine embedding dimension for {model_name}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Embedding dimension reported by the model."""
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts (L2-normalized)."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float64)


class CohereProvider(EmbeddingProvider):
    """Cohere Embed API provider.

    Requires the ``COHERE_API_KEY`` environment variable and the ``cohere``
    package.
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
    ) -> None:
        """Args:
        model: Cohere embedding model identifier.
        api_key: API key; defaults to the ``COHERE_API_KEY`` environment
            variable.

        Raises:
            RuntimeError: If no API key is available.
            ImportError: If the ``cohere`` package is not installed.
        """
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "CohereProvider requires an API key. Set the COHERE_API_KEY "
                "environment variable (see .env.example) or pass api_key explicitly."
            )
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "The 'cohere' package is required for CohereProvider. "
                "Install it with: pip install 'retrieval-lab[models]'"
            ) from exc
        self.model = model
        self._client = cohere.ClientV2(api_key=key)

    @property
    def dimension(self) -> int:
        """Embedding dimension (1024 for the default v3 English model)."""
        return 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts via the Cohere API (L2-normalized)."""
        response = self._client.embed(texts=texts, model=self.model, input_type="search_document")
        raw: object = response.embeddings
        vectors = np.asarray(raw, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


def build_embedding_provider(name: str, model: str = "", dimension: int = 384) -> EmbeddingProvider:
    """Instantiate an embedding provider by name.

    Args:
        name: One of ``"deterministic"``, ``"sentence-transformer"``,
            ``"cohere"``.
        model: Model identifier for learned providers (ignored otherwise).
        dimension: Vector size for the deterministic provider.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    normalized = name.lower()
    if normalized == "deterministic":
        return DeterministicEmbeddingProvider(dimension=dimension)
    if normalized == "sentence-transformer":
        return SentenceTransformerProvider(model_name=model or "all-MiniLM-L6-v2")
    if normalized == "cohere":
        return CohereProvider(model=model or "embed-english-v3.0")
    raise ValueError(
        f"Unknown embedding provider {name!r}; expected 'deterministic', "
        "'sentence-transformer' or 'cohere'"
    )
