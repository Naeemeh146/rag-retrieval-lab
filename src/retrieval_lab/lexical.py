"""BM25 lexical retrieval built on rank-bm25."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from retrieval_lab.embeddings import tokenize


class BM25Retriever:
    """Lexical retriever using the Okapi BM25 ranking function.

    Documents are indexed once; queries are tokenized with the same
    lowercase alphanumeric tokenizer used across the lab.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Args:
        k1: Term-frequency saturation parameter.
        b: Document-length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def index(self, doc_ids: list[str], texts: list[str]) -> None:
        """Build the BM25 index over ``texts``.

        Args:
            doc_ids: Document identifiers, one per text.
            texts: Corpus texts.

        Raises:
            ValueError: If ``doc_ids`` and ``texts`` have different lengths
                or the corpus is empty.
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if not texts:
            raise ValueError("cannot index an empty corpus")
        self._doc_ids = list(doc_ids)
        tokenized = [tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Rank documents for ``query``.

        Args:
            query: Query text.
            top_k: Maximum number of results.

        Returns:
            ``(doc_id, score)`` pairs sorted by descending score.

        Raises:
            RuntimeError: If called before :meth:`index`.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.search called before index()")
        if top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._doc_ids, (float(s) for s in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
