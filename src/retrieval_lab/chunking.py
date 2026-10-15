"""Chunking strategies for splitting documents into retrievable passages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single retrievable passage cut from a source document.

    Attributes:
        text: The chunk text.
        doc_id: Identifier of the source document.
        chunk_id: Unique identifier, ``f"{doc_id}#c{index}"``.
        start: Character offset where the chunk starts in the source document.
        end: Character offset where the chunk ends in the source document.
    """

    text: str
    doc_id: str
    chunk_id: str
    start: int
    end: int


class BaseChunker(ABC):
    """Interface for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split ``text`` into chunks belonging to ``doc_id``.

        Args:
            text: Full document text.
            doc_id: Identifier of the source document.

        Returns:
            Ordered list of chunks covering the document.
        """
        raise NotImplementedError

    def _make_chunk(self, text: str, doc_id: str, index: int, start: int, end: int) -> Chunk:
        """Build a Chunk with a conventional id."""
        return Chunk(text=text, doc_id=doc_id, chunk_id=f"{doc_id}#c{index}", start=start, end=end)


class FixedSizeChunker(BaseChunker):
    """Split text into consecutive fixed-size character windows (no overlap).

    The final chunk may be shorter than ``chunk_size``. Empty input yields
    no chunks.
    """

    def __init__(self, chunk_size: int = 500) -> None:
        """Args:
        chunk_size: Maximum number of characters per chunk. Must be positive.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split ``text`` into fixed-size windows."""
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(text), self.chunk_size)):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end]
            if piece:
                chunks.append(self._make_chunk(piece, doc_id, i, start, end))
        return chunks


class OverlappingChunker(BaseChunker):
    """Sliding-window chunking with a fixed character overlap between windows.

    Windows advance by ``chunk_size - overlap`` characters. The final window
    may be shorter than ``chunk_size``.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        """Args:
        chunk_size: Number of characters per chunk. Must be positive.
        overlap: Characters shared with the previous chunk. Must satisfy
            ``0 <= overlap < chunk_size``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= overlap < chunk_size:
            raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split ``text`` into overlapping windows."""
        if not text:
            return []
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        index = 0
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(self._make_chunk(text[start:end], doc_id, index, start, end))
            if end == len(text):
                break
            start += step
            index += 1
        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """Recursively split on separators, then pack pieces into sized chunks.

    The text is split on the first separator that produces pieces (tried in
    order: paragraph, newline, sentence, word, character). Pieces are then
    greedily packed into chunks whose body is at most ``chunk_size``
    characters, joined with a single space. Consecutive chunks share a
    ``chunk_overlap`` character tail of the previous chunk so boundary
    context is preserved (chunks may therefore exceed ``chunk_size`` by up
    to ``chunk_overlap`` characters).

    Chunk ``start``/``end`` offsets are best-effort: they point at the first
    and last contributing split in the source document.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
    ) -> None:
        """Args:
        chunk_size: Maximum characters per chunk. Must be positive.
        chunk_overlap: Characters of trailing context carried into the next
            chunk. Must satisfy ``0 <= chunk_overlap < chunk_size``.
        separators: Separators tried in order, finest last.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                f"chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size, got {chunk_overlap}"
            )
        if not separators:
            raise ValueError("separators must not be empty")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split ``text`` recursively and pack pieces into chunks."""
        if not text:
            return []
        splits = self._split_text(text, self.separators)
        # Track source offsets for each split.
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for split in splits:
            idx = text.find(split, cursor)
            if idx == -1:  # Fallback: should not happen for clean splits.
                idx = cursor
            offsets.append((idx, idx + len(split)))
            cursor = idx + len(split)

        chunks: list[Chunk] = []
        current: list[str] = []
        cur_len = 0
        first_split_idx = 0

        def flush(end_split_idx: int) -> None:
            body = " ".join(current)
            prefix = ""
            if chunks and self.chunk_overlap > 0:
                prefix = chunks[-1].text[-self.chunk_overlap :].rstrip()
            chunk_text = f"{prefix} {body}" if prefix else body
            start = offsets[first_split_idx][0]
            end = offsets[end_split_idx][1]
            chunks.append(self._make_chunk(chunk_text, doc_id, len(chunks), start, end))

        for i, split in enumerate(splits):
            add_len = len(split) + (1 if current else 0)
            if current and cur_len + add_len > self.chunk_size:
                flush(i - 1)
                current = []
                cur_len = 0
                first_split_idx = i
                add_len = len(split)
            current.append(split)
            cur_len += add_len
        if current:
            flush(len(splits) - 1)
        return chunks

    def _split_text(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Recursively split ``text`` with the first usable separator."""
        if not separators:
            return [text]
        separator = separators[0]
        remaining = separators[1:]
        if separator == "":
            return list(text)
        if separator not in text:
            return self._split_text(text, remaining)
        parts = [part for part in text.split(separator) if part]
        splits: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                splits.append(part)
            elif remaining:
                splits.extend(self._split_text(part, remaining))
            else:
                # No finer separator left: hard-split overlong part.
                splits.extend(
                    part[i : i + self.chunk_size] for i in range(0, len(part), self.chunk_size)
                )
        return splits


def build_chunker(name: str, chunk_size: int = 500, chunk_overlap: int = 100) -> BaseChunker:
    """Instantiate a chunker by name.

    Args:
        name: One of ``"fixed"``, ``"overlapping"``, ``"recursive"``.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap characters (ignored by ``"fixed"``).

    Raises:
        ValueError: If ``name`` is unknown.
    """
    normalized = name.lower()
    if normalized == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size)
    if normalized == "overlapping":
        return OverlappingChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    if normalized == "recursive":
        return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(f"Unknown chunker {name!r}; expected 'fixed', 'overlapping' or 'recursive'")
