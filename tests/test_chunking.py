"""Unit tests for chunking strategies (exact boundary assertions)."""

import pytest

from retrieval_lab.chunking import (
    FixedSizeChunker,
    OverlappingChunker,
    RecursiveCharacterChunker,
    build_chunker,
)


def test_fixed_size_exact_boundaries() -> None:
    chunker = FixedSizeChunker(chunk_size=4)
    chunks = chunker.chunk("abcdefghij", "d")
    assert [(c.text, c.start, c.end, c.chunk_id) for c in chunks] == [
        ("abcd", 0, 4, "d#c0"),
        ("efgh", 4, 8, "d#c1"),
        ("ij", 8, 10, "d#c2"),
    ]


def test_fixed_size_empty_text() -> None:
    assert FixedSizeChunker(chunk_size=4).chunk("", "d") == []


def test_fixed_size_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        FixedSizeChunker(chunk_size=0)


def test_overlapping_exact_boundaries() -> None:
    chunker = OverlappingChunker(chunk_size=4, overlap=2)
    chunks = chunker.chunk("abcdefghij", "d")
    assert [(c.text, c.start, c.end) for c in chunks] == [
        ("abcd", 0, 4),
        ("cdef", 2, 6),
        ("efgh", 4, 8),
        ("ghij", 6, 10),
    ]


def test_overlapping_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        OverlappingChunker(chunk_size=4, overlap=4)
    with pytest.raises(ValueError, match="overlap"):
        OverlappingChunker(chunk_size=4, overlap=-1)


def test_recursive_exact_packing_and_overlap() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk("aaa bbb ccc ddd eee fff", "doc")
    assert [c.text for c in chunks] == ["aaa bbb ccc ddd eee", "d eee fff"]
    assert chunks[0].chunk_id == "doc#c0"
    assert chunks[1].chunk_id == "doc#c1"
    # Offsets point at the contributing splits in the source text.
    assert (chunks[0].start, chunks[0].end) == (0, 19)
    assert (chunks[1].start, chunks[1].end) == (20, 23)


def test_recursive_splits_long_word_by_characters() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=4, chunk_overlap=0)
    chunks = chunker.chunk("abcdefgh", "d")
    # Character-level fallback splits are space-joined when packed.
    assert [c.text for c in chunks] == ["a b", "c d", "e f", "g h"]


def test_recursive_empty_text() -> None:
    assert RecursiveCharacterChunker().chunk("", "d") == []


def test_build_chunker_names() -> None:
    assert isinstance(build_chunker("fixed"), FixedSizeChunker)
    assert isinstance(build_chunker("overlapping"), OverlappingChunker)
    assert isinstance(build_chunker("recursive"), RecursiveCharacterChunker)
    with pytest.raises(ValueError, match="Unknown chunker"):
        build_chunker("sliding")
