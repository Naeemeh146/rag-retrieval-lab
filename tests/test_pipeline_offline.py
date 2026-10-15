"""Offline end-to-end tests: build, search, save/load, and full eval harness."""

from pathlib import Path

import pytest

from retrieval_lab.datasets import load_synthetic_corpus
from retrieval_lab.evaluate import run_experiment, run_experiments
from retrieval_lab.pipeline import RetrieverConfig, TwoStageRetriever


def _hybrid_config() -> RetrieverConfig:
    return RetrieverConfig(
        name="test-hybrid",
        embedding={"provider": "deterministic", "dimension": 64},  # type: ignore[dict-item]
    )


def test_end_to_end_offline_eval_metrics_are_sane() -> None:
    dataset = load_synthetic_corpus()
    metrics = run_experiment(_hybrid_config(), dataset, ks=[5, 10])
    assert metrics["name"] == "test-hybrid"
    for key, value in metrics.items():
        if key == "name":
            continue
        assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
    # The synthetic corpus is designed so hybrid retrieval finds signal.
    assert metrics["recall@5"] > 0.0
    assert metrics["mrr@5"] > 0.0


def test_bm25_only_config_runs_offline() -> None:
    dataset = load_synthetic_corpus()
    config = RetrieverConfig.from_yaml(Path("configs/bm25-only.yaml"))
    retriever = TwoStageRetriever(config)
    retriever.build(dataset)
    hits = retriever.search("What is nDCG in information retrieval evaluation?", top_k=5)
    assert hits, "expected at least one hit"
    # The nDCG document must surface in the top results.
    assert "d10" in {hit.doc_id for hit in hits}


def test_search_before_build_raises() -> None:
    retriever = TwoStageRetriever(_hybrid_config())
    with pytest.raises(RuntimeError, match="before build"):
        retriever.search("query")


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    dataset = load_synthetic_corpus()
    retriever = TwoStageRetriever(_hybrid_config())
    retriever.build(dataset)
    index_dir = tmp_path / "index"
    retriever.save(index_dir)
    assert (index_dir / "config.yaml").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "bm25.pkl").exists()

    loaded = TwoStageRetriever.load(index_dir)
    query = "Why use a vector database with HNSW for similarity search?"
    assert [h.chunk_id for h in loaded.search(query, top_k=5)] == [
        h.chunk_id for h in retriever.search(query, top_k=5)
    ]


def test_run_experiments_writes_json_and_csv(tmp_path: Path) -> None:
    dataset = load_synthetic_corpus()
    out_dir = tmp_path / "results"
    rows = run_experiments(
        [Path("configs/bm25-only.yaml"), Path("configs/dense-only.yaml")],
        dataset,
        out_dir,
        ks=[5],
    )
    assert [row["name"] for row in rows] == ["bm25-only", "dense-only"]
    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.csv").exists()
