"""Evaluation harness: run experiment configs and record metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from retrieval_lab.datasets import RetrievalDataset, load_dataset
from retrieval_lab.metrics import compute_metrics
from retrieval_lab.pipeline import RetrieverConfig, TwoStageRetriever

DEFAULT_KS = [5, 10]


def run_experiment(
    config: RetrieverConfig,
    dataset: RetrievalDataset,
    ks: list[int] | None = None,
) -> dict[str, float | str]:
    """Run one experiment config end to end and return averaged metrics.

    Args:
        config: Retriever configuration under test.
        dataset: Evaluation dataset with queries and qrels.
        ks: Metric cutoffs.

    Returns:
        Row mapping with the experiment ``name`` plus averaged metric values.
    """
    cutoffs = ks or DEFAULT_KS
    retriever = TwoStageRetriever(config)
    retriever.build(dataset)
    results = {
        query.query_id: retriever.search_docs(query.text, top_k=max(cutoffs))
        for query in dataset.queries
    }
    metrics = compute_metrics(results, dataset.qrels, cutoffs)
    row: dict[str, float | str] = {"name": config.name}
    row.update(metrics)
    return row


def run_experiments(
    config_paths: list[Path],
    dataset: RetrievalDataset,
    out_dir: Path,
    ks: list[int] | None = None,
) -> list[dict[str, float | str]]:
    """Run several configs and write ``results.json`` + ``results.csv``.

    Args:
        config_paths: YAML experiment configs to evaluate.
        dataset: Evaluation dataset.
        out_dir: Directory receiving ``results.json`` and ``results.csv``.
        ks: Metric cutoffs.

    Returns:
        Per-experiment metric rows in input order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for config_path in config_paths:
        config = RetrieverConfig.from_yaml(config_path)
        rows.append(run_experiment(config, dataset, ks=ks))
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_dir / "results.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def evaluate_dataset(
    config_paths: list[Path],
    dataset_name: str,
    out_dir: Path,
    data_dir: Path,
    ks: list[int] | None = None,
) -> list[dict[str, float | str]]:
    """Load a dataset by name and run the experiment suite on it."""
    dataset = load_dataset(dataset_name, data_dir=data_dir)
    return run_experiments(config_paths, dataset, out_dir, ks=ks)
