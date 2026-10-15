"""Two-stage retrieval pipeline with pydantic-v2 YAML configs."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

from retrieval_lab.chunking import BaseChunker, Chunk, build_chunker
from retrieval_lab.datasets import RetrievalDataset
from retrieval_lab.embeddings import EmbeddingProvider, build_embedding_provider
from retrieval_lab.fusion import reciprocal_rank_fusion
from retrieval_lab.lexical import BM25Retriever
from retrieval_lab.rerank import Reranker, build_reranker


class ChunkerConfig(BaseModel):
    """Chunking stage configuration."""

    name: str = Field(default="recursive", description="fixed | overlapping | recursive")
    chunk_size: int = Field(default=200, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)


class EmbeddingConfig(BaseModel):
    """Dense embedding configuration."""

    provider: str = Field(
        default="deterministic",
        description="deterministic | sentence-transformer | cohere",
    )
    model: str = Field(default="all-MiniLM-L6-v2")
    dimension: int = Field(default=384, gt=0)


class BM25Config(BaseModel):
    """Lexical retrieval configuration."""

    enabled: bool = True
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    top_k: int = Field(default=50, gt=0)


class DenseConfig(BaseModel):
    """Dense retrieval configuration."""

    enabled: bool = True
    top_k: int = Field(default=50, gt=0)


class FusionConfig(BaseModel):
    """Rank fusion configuration."""

    method: str = Field(default="rrf", description="only 'rrf' is supported")
    k: int = Field(default=60, gt=0)
    top_k: int = Field(default=50, gt=0)


class RerankConfig(BaseModel):
    """Second-stage reranking configuration."""

    enabled: bool = False
    provider: str = Field(default="none", description="none | cross-encoder | cohere")
    model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_n: int = Field(default=10, gt=0)


class RetrieverConfig(BaseModel):
    """Full two-stage retriever configuration (YAML-serializable)."""

    name: str = "experiment"
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RetrieverConfig:
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as handle:
            payload: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls.model_validate(payload)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(), handle, sort_keys=False)


@dataclass(frozen=True)
class RetrievalResult:
    """A single ranked retrieval hit."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int


class TwoStageRetriever:
    """Two-stage retriever: hybrid first-stage retrieval + reranking.

    Stage 1 retrieves candidates with BM25 and/or dense embeddings and fuses
    the rankings with Reciprocal Rank Fusion. Stage 2 (optional) reranks the
    fused candidates with a cross-encoder or API reranker.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        """Args:
        config: Pipeline configuration.
        """
        self.config = config
        self._chunker: BaseChunker = build_chunker(
            config.chunker.name,
            chunk_size=config.chunker.chunk_size,
            chunk_overlap=config.chunker.chunk_overlap,
        )
        self._embedding: EmbeddingProvider = build_embedding_provider(
            config.embedding.provider,
            model=config.embedding.model,
            dimension=config.embedding.dimension,
        )
        self._bm25 = BM25Retriever(k1=config.bm25.k1, b=config.bm25.b)
        self._reranker: Reranker | None = (
            build_reranker(config.rerank.provider, model=config.rerank.model)
            if config.rerank.enabled
            else None
        )
        self._chunks: list[Chunk] = []
        self._chunk_embeddings: np.ndarray | None = None
        self._built = False

    @property
    def is_built(self) -> bool:
        """Whether :meth:`build` has been called."""
        return self._built

    @property
    def num_chunks(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def build(self, dataset: RetrievalDataset) -> None:
        """Chunk the corpus and build lexical + dense indexes.

        Args:
            dataset: Corpus to index (queries/qrels are not needed here).
        """
        chunks: list[Chunk] = []
        for document in dataset.documents:
            chunks.extend(self._chunker.chunk(document.text, document.doc_id))
        if not chunks:
            raise ValueError("no chunks produced; check the chunker configuration")
        self._chunks = chunks
        self._bm25.index(
            [chunk.chunk_id for chunk in chunks],
            [chunk.text for chunk in chunks],
        )
        self._chunk_embeddings = self._embedding.embed([c.text for c in chunks])
        self._built = True

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Run two-stage retrieval for ``query``.

        Args:
            query: Query text.
            top_k: Maximum results; defaults to the fusion ``top_k``.

        Returns:
            Ranked retrieval results (chunk-level).
        """
        self._require_built()
        assert self._chunk_embeddings is not None
        limit = top_k or self.config.fusion.top_k

        rankings: list[list[tuple[str, float]]] = []
        if self.config.bm25.enabled:
            rankings.append(self._bm25.search(query, top_k=self.config.bm25.top_k))
        if self.config.dense.enabled:
            rankings.append(self._dense_search(query, self.config.dense.top_k))
        if not rankings:
            raise ValueError("at least one of BM25 or dense retrieval must be enabled")

        fused = reciprocal_rank_fusion(
            rankings, k=self.config.fusion.k, top_k=self.config.fusion.top_k
        )

        if self._reranker is not None:
            by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
            reranked = self._reranker.rerank(
                query,
                [chunk_id for chunk_id, _ in fused],
                [by_id[chunk_id].text for chunk_id, _ in fused],
                top_n=min(self.config.rerank.top_n, limit),
            )
            fused = reranked[:limit]
        else:
            fused = fused[:limit]

        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                doc_id=by_id[chunk_id].doc_id,
                text=by_id[chunk_id].text,
                score=score,
                rank=rank,
            )
            for rank, (chunk_id, score) in enumerate(fused, start=1)
        ]

    def search_docs(self, query: str, top_k: int | None = None) -> list[str]:
        """Return ranked document ids (chunk hits deduplicated, best rank wins)."""
        seen: set[str] = set()
        doc_ids: list[str] = []
        for hit in self.search(query, top_k=top_k):
            if hit.doc_id not in seen:
                seen.add(hit.doc_id)
                doc_ids.append(hit.doc_id)
        return doc_ids

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Cosine-similarity search over chunk embeddings (chunk ids)."""
        self._require_built()
        assert self._chunk_embeddings is not None
        query_vec = self._embedding.embed([query])[0]
        scores = self._chunk_embeddings @ query_vec
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self._chunks[int(i)].chunk_id, float(scores[int(i)])) for i in order]

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("TwoStageRetriever.search called before build()")

    def save(self, directory: str | Path) -> None:
        """Persist the built index (chunks, embeddings, BM25) to ``directory``.

        Args:
            directory: Target directory (created if missing).
        """
        self._require_built()
        assert self._chunk_embeddings is not None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(directory / "config.yaml")
        with open(directory / "chunks.json", "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "start": chunk.start,
                        "end": chunk.end,
                    }
                    for chunk in self._chunks
                ],
                handle,
            )
        np.save(directory / "embeddings.npy", self._chunk_embeddings)
        with open(directory / "bm25.pkl", "wb") as handle:
            pickle.dump(self._bm25, handle)

    @classmethod
    def load(cls, directory: str | Path) -> TwoStageRetriever:
        """Load a previously saved index from ``directory``."""
        directory = Path(directory)
        config = RetrieverConfig.from_yaml(directory / "config.yaml")
        retriever = cls(config)
        with open(directory / "chunks.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        retriever._chunks = [
            Chunk(
                text=item["text"],
                doc_id=item["doc_id"],
                chunk_id=item["chunk_id"],
                start=item["start"],
                end=item["end"],
            )
            for item in payload
        ]
        retriever._chunk_embeddings = np.load(directory / "embeddings.npy")
        with open(directory / "bm25.pkl", "rb") as handle:
            retriever._bm25 = pickle.load(handle)
        retriever._built = True
        return retriever
