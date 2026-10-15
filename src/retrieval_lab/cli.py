"""Command-line interface for the retrieval lab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from retrieval_lab.datasets import download_scifact, load_dataset
from retrieval_lab.evaluate import evaluate_dataset
from retrieval_lab.pipeline import RetrieverConfig, TwoStageRetriever

app = typer.Typer(
    help="Two-stage RAG retrieval lab: ingest, query, and benchmark retrieval pipelines."
)


@app.command("download-data")
def download_data(
    out: Annotated[Path, typer.Option(help="Cache directory for datasets.")] = Path("data"),
) -> None:
    """Download the BEIR SciFact dataset (network required)."""
    dataset = download_scifact(out)
    typer.echo(
        f"Downloaded {dataset.name}: {len(dataset.documents)} docs, "
        f"{len(dataset.queries)} queries, {len(dataset.qrels)} qrels "
        f"-> {out.resolve()}"
    )


@app.command()
def ingest(
    config: Annotated[Path, typer.Option(help="Retriever YAML config.")] = Path(
        "configs/hybrid.yaml"
    ),
    dataset: Annotated[str, typer.Option(help="'synthetic' or 'scifact'.")] = "synthetic",
    data_dir: Annotated[Path, typer.Option(help="Dataset cache directory.")] = Path("data"),
    out: Annotated[Path, typer.Option(help="Index output directory.")] = Path("data/index"),
) -> None:
    """Chunk a dataset and build the lexical + dense index on disk."""
    corpus = load_dataset(dataset, data_dir=data_dir)
    retriever = TwoStageRetriever(RetrieverConfig.from_yaml(config))
    retriever.build(corpus)
    retriever.save(out)
    typer.echo(
        f"Indexed {len(corpus.documents)} documents "
        f"({retriever.num_chunks} chunks) -> {out.resolve()}"
    )


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Query text.")],
    index: Annotated[Path, typer.Option(help="Index directory from ingest.")] = Path("data/index"),
    top_n: Annotated[int, typer.Option(help="Results to show.")] = 5,
) -> None:
    """Run two-stage retrieval against a built index."""
    retriever = TwoStageRetriever.load(index)
    for hit in retriever.search(question, top_k=top_n):
        typer.echo(f"[{hit.rank}] {hit.doc_id} ({hit.chunk_id}) score={hit.score:.4f}")
        typer.echo(f"    {hit.text[:200]}")


@app.command()
def evaluate(
    configs: Annotated[
        list[Path] | None, typer.Option(help="Experiment YAML configs (repeatable).")
    ] = None,
    dataset: Annotated[str, typer.Option(help="'synthetic' or 'scifact'.")] = "synthetic",
    data_dir: Annotated[Path, typer.Option(help="Dataset cache directory.")] = Path("data"),
    out: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
) -> None:
    """Run experiment configs and write results.json + results.csv."""
    selected = configs or [
        Path("configs/bm25-only.yaml"),
        Path("configs/dense-only.yaml"),
        Path("configs/hybrid.yaml"),
    ]
    rows = evaluate_dataset(selected, dataset, out, data_dir)
    for row in rows:
        metrics = "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k != "name" and isinstance(v, float)
        )
        typer.echo(f"{row['name']}: {metrics}")


if __name__ == "__main__":
    app()
