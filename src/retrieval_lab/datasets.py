"""Dataset loading: built-in synthetic corpus and BEIR SciFact downloader."""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


@dataclass(frozen=True)
class Document:
    """A corpus document."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Query:
    """An evaluation query."""

    query_id: str
    text: str


@dataclass
class RetrievalDataset:
    """Documents, queries, and relevance judgments for retrieval evaluation."""

    name: str
    documents: list[Document] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    qrels: dict[str, set[str]] = field(default_factory=dict)


def load_synthetic_corpus() -> RetrievalDataset:
    """Load the tiny built-in corpus used for offline tests and demos.

    Sixteen short documents across six topics (hybrid search, reranking,
    chunking, dense embeddings, evaluation metrics, vector databases) with
    eight queries and known relevant documents. Runs fully offline.
    """
    docs = [
        (
            "d01",
            "Hybrid search combines BM25 lexical matching with dense vector "
            "retrieval. The two ranked lists are merged with reciprocal rank "
            "fusion, which sums inverse ranks so that documents ranked highly "
            "by either retriever rise to the top of the fused list.",
        ),
        (
            "d02",
            "Reciprocal rank fusion needs no score normalization because it "
            "only uses ranks. A common smoothing constant k of 60 dampens the "
            "influence of low-ranked documents when fusing BM25 and dense "
            "embedding results.",
        ),
        (
            "d03",
            "A cross-encoder reranker scores query and document pairs jointly "
            "with a transformer, producing more accurate relevance scores than "
            "bi-encoder retrieval. It is slower, so it reranks only the top "
            "candidates from first-stage hybrid retrieval.",
        ),
        (
            "d04",
            "Two-stage retrieval pipelines first recall candidates with fast "
            "retrieval and then rerank them. The reranker promotes the most "
            "relevant passages, which improves answer quality for RAG systems "
            "while keeping latency acceptable.",
        ),
        (
            "d05",
            "Chunking strategy controls how documents are split into passages. "
            "Fixed-size chunking cuts text every N characters, while recursive "
            "character chunking splits on paragraph and sentence boundaries "
            "first, preserving semantic coherence of each chunk.",
        ),
        (
            "d06",
            "Overlapping sliding-window chunking shares context between "
            "adjacent chunks so that information near a boundary is not lost. "
            "A typical setup uses 500-character chunks with 100 characters "
            "of overlap for long documents.",
        ),
        (
            "d07",
            "Dense embeddings map text into a vector space where similar "
            "meanings are close together. Bi-encoder models embed queries and "
            "documents independently, enabling fast cosine-similarity search "
            "over millions of passages.",
        ),
        (
            "d08",
            "Sentence transformer models such as all-MiniLM-L6-v2 produce "
            "compact 384-dimensional embeddings that work well for semantic "
            "search. They run locally without an API key and index quickly.",
        ),
        (
            "d09",
            "Recall at k measures the fraction of relevant documents retrieved "
            "in the top k results. It answers whether the retriever found "
            "everything it should have, ignoring ranking order within the "
            "top k.",
        ),
        (
            "d10",
            "Normalized discounted cumulative gain, nDCG, rewards placing "
            "relevant documents at high ranks. It discounts gains logarithmically "
            "by rank and normalizes by the ideal ranking, making it the "
            "standard metric for evaluating ranked retrieval.",
        ),
        (
            "d11",
            "Mean reciprocal rank, MRR, is the inverse rank of the first "
            "relevant document. It is widely used for question answering "
            "evaluation where a single correct passage matters most.",
        ),
        (
            "d12",
            "Vector databases store embeddings with approximate nearest "
            "neighbor indexes such as HNSW. They return similar vectors in "
            "milliseconds, which makes large-scale dense retrieval practical "
            "for production RAG pipelines.",
        ),
        (
            "d13",
            "Metadata filtering in a vector database restricts similarity "
            "search to documents matching attributes like date or source. "
            "Combining filters with vector search improves precision for "
            "domain-specific retrieval tasks.",
        ),
        (
            "d14",
            "Query expansion adds synonyms or generated terms to the original "
            "query before retrieval. It helps BM25 match documents that use "
            "different vocabulary than the query, raising recall.",
        ),
        (
            "d15",
            "Sparse embeddings keep most dimensions at zero and only activate "
            "terms present in the text. They offer an efficient middle ground "
            "between BM25 term matching and dense semantic search.",
        ),
        (
            "d16",
            "Evaluation harnesses run fixed experiment configurations over a "
            "dataset and record metrics to JSON and CSV. Comparing ablations "
            "side by side reveals which retrieval component drives gains.",
        ),
    ]
    queries = [
        ("q1", "How does reciprocal rank fusion combine BM25 and dense results?"),
        ("q2", "What is a cross-encoder reranker and why is it slower?"),
        ("q3", "Which chunking strategy preserves semantic coherence?"),
        ("q4", "How do dense embeddings enable fast similarity search?"),
        ("q5", "What is nDCG in information retrieval evaluation?"),
        ("q6", "Why use a vector database with HNSW for similarity search?"),
        ("q7", "Describe a two-stage retrieval pipeline with reranking"),
        ("q8", "Which metrics evaluate ranked search engine results?"),
    ]
    qrels = {
        "q1": {"d01", "d02"},
        "q2": {"d03"},
        "q3": {"d05", "d06"},
        "q4": {"d07", "d08"},
        "q5": {"d10"},
        "q6": {"d12"},
        "q7": {"d03", "d04"},
        "q8": {"d09", "d10", "d11"},
    }
    return RetrievalDataset(
        name="synthetic",
        documents=[Document(doc_id=doc_id, text=text) for doc_id, text in docs],
        queries=[Query(query_id=query_id, text=text) for query_id, text in queries],
        qrels=qrels,
    )


def download_scifact(data_dir: str | Path) -> RetrievalDataset:
    """Download the BEIR SciFact dataset and parse it without the `beir` package.

    Fetches the public ``scifact.zip`` archive, extracts ``corpus.jsonl``,
    ``queries.jsonl`` and ``qrels/test.tsv`` from it, and caches the parsed
    dataset as JSON under ``data_dir`` for subsequent offline loads.

    Args:
        data_dir: Directory used for download cache and parsed output.

    Returns:
        The SciFact retrieval dataset (test split qrels).

    Raises:
        URLError: If the download fails (network required).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "scifact.json"

    if cache_path.exists():
        return _dataset_from_json(json.loads(cache_path.read_text()))

    archive_path = data_dir / "scifact.zip"
    if not archive_path.exists():
        request = urllib.request.Request(
            SCIFACT_URL, headers={"User-Agent": "rag-retrieval-lab/0.1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            open(archive_path, "wb") as handle,
        ):
            handle.write(response.read())

    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, set[str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("scifact/corpus.jsonl") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                record = json.loads(line)
                title = record.get("title", "")
                body = record.get("text", "")
                corpus[record["_id"]] = f"{title} {body}".strip()
        with archive.open("scifact/queries.jsonl") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                record = json.loads(line)
                queries[record["_id"]] = record["text"]
        with archive.open("scifact/qrels/test.tsv") as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"), delimiter="\t")
            for row in reader:
                if int(row["score"]) > 0:
                    qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])

    dataset = RetrievalDataset(
        name="scifact",
        documents=[Document(doc_id=doc_id, text=text) for doc_id, text in corpus.items()],
        queries=[Query(query_id=query_id, text=text) for query_id, text in queries.items()],
        qrels=qrels,
    )
    cache_path.write_text(
        json.dumps(
            {
                "name": dataset.name,
                "documents": [
                    {"doc_id": doc.doc_id, "text": doc.text} for doc in dataset.documents
                ],
                "queries": [{"query_id": q.query_id, "text": q.text} for q in dataset.queries],
                "qrels": {qid: sorted(ids) for qid, ids in dataset.qrels.items()},
            }
        )
    )
    return dataset


def _dataset_from_json(payload: dict[str, object]) -> RetrievalDataset:
    """Rebuild a RetrievalDataset from its cached JSON form."""
    documents = payload["documents"]
    queries = payload["queries"]
    qrels = payload["qrels"]
    assert isinstance(documents, list)
    assert isinstance(queries, list)
    assert isinstance(qrels, dict)
    return RetrievalDataset(
        name=str(payload["name"]),
        documents=[Document(doc_id=str(d["doc_id"]), text=str(d["text"])) for d in documents],
        queries=[Query(query_id=str(q["query_id"]), text=str(q["text"])) for q in queries],
        qrels={str(qid): set(ids) for qid, ids in qrels.items()},
    )


def load_dataset(name: str, data_dir: str | Path = "data") -> RetrievalDataset:
    """Load a dataset by name.

    Args:
        name: ``"synthetic"`` for the built-in offline corpus or
            ``"scifact"`` for the cached BEIR download.
        data_dir: Cache directory for downloaded datasets.

    Raises:
        ValueError: If ``name`` is unknown.
        FileNotFoundError: If ``"scifact"`` was never downloaded.
    """
    normalized = name.lower()
    if normalized == "synthetic":
        return load_synthetic_corpus()
    if normalized == "scifact":
        cache_path = Path(data_dir) / "scifact.json"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"SciFact cache not found at {cache_path}. Run `retrieval-lab download-data` first."
            )
        return _dataset_from_json(json.loads(cache_path.read_text()))
    raise ValueError(f"Unknown dataset {name!r}; expected 'synthetic' or 'scifact'")
