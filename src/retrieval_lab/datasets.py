"""Dataset loading: built-in synthetic corpus, BEIR SciFact, and NVD CVE downloads."""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
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
        name: ``"synthetic"`` for the built-in offline corpus,
            ``"scifact"`` for the cached BEIR download, or ``"cve"``
            for the cached NVD CVE corpus.
        data_dir: Cache directory for downloaded datasets.

    Raises:
        ValueError: If ``name`` is unknown.
        FileNotFoundError: If ``"scifact"`` or ``"cve"`` was never downloaded.
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
    if normalized == "cve":
        return load_cve_corpus(data_dir)
    raise ValueError(f"Unknown dataset {name!r}; expected 'synthetic', 'scifact', or 'cve'")


# ---------------------------------------------------------------------------
# NVD CVE corpus (public data only — https://nvd.nist.gov)
# ---------------------------------------------------------------------------

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_RESULTS_PER_PAGE = 2000
# Unauthenticated NVD API 2.0 limit: 5 requests per rolling 30 seconds.
NVD_REQUEST_DELAY_SECONDS = 7.0

# Natural-language probe queries. Each pairs a query with the exact substring
# rule used to build its qrels, so relevance is deterministic and defensible.
CVE_NL_QUERIES: list[tuple[str, str, str]] = [
    ("nl-buffer-overflow", "buffer overflow vulnerabilities", "buffer overflow"),
    ("nl-sql-injection", "SQL injection vulnerabilities", "sql injection"),
    (
        "nl-xss",
        "cross-site scripting (XSS) vulnerabilities",
        "cross-site scripting",
    ),
]


def _nvd_get(params: dict[str, str], delay: float = NVD_REQUEST_DELAY_SECONDS) -> dict:
    """GET one NVD API 2.0 page, raising on HTTP errors.

    Args:
        params: Query parameters for the request.
        delay: Seconds to wait before the request (rate-limit courtesy).
    """
    import time

    time.sleep(delay)
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{NVD_API_URL}?{query}",
        headers={"User-Agent": "rag-retrieval-lab/0.1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload: dict = json.loads(response.read().decode("utf-8"))
        return payload


def download_nvd_pages(
    data_dir: str | Path,
    max_cves: int = 3000,
    pub_start: str = "2026-07-01T00:00:00.000",
    pub_end: str = "2026-10-15T00:00:00.000",
) -> list[dict]:
    """Download raw NVD CVE records with resumable per-page caching.

    Pages are cached under ``<data_dir>/nvd-pages/page-<startIndex>.json`` so
    an interrupted download resumes without re-requesting finished pages.

    Args:
        data_dir: Cache directory.
        max_cves: Upper bound on CVE records to fetch.
        pub_start: NVD ``pubStartDate`` (ISO-8601). Must be within 120 days
            of ``pub_end`` (NVD API limit).
        pub_end: NVD ``pubEndDate`` (ISO-8601).

    Returns:
        Raw ``vulnerabilities`` entries from the NVD API.

    Raises:
        ValueError: If the date range exceeds NVD's 120-day limit.
    """
    from datetime import datetime

    start = datetime.fromisoformat(pub_start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(pub_end.replace("Z", "+00:00"))
    if (end - start).days > 120:
        raise ValueError("NVD API 2.0 limits pubStartDate/pubEndDate to a 120-day range")
    data_dir = Path(data_dir)
    pages_dir = data_dir / "nvd-pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    start_index = 0
    total_results: int | None = None
    first_request = True
    while len(records) < max_cves:
        page_path = pages_dir / f"page-{start_index}.json"
        if page_path.exists():
            page = json.loads(page_path.read_text())
        else:
            page = _nvd_get(
                {
                    "pubStartDate": pub_start,
                    "pubEndDate": pub_end,
                    "resultsPerPage": str(NVD_RESULTS_PER_PAGE),
                    "startIndex": str(start_index),
                },
                delay=0.0 if first_request else NVD_REQUEST_DELAY_SECONDS,
            )
            page_path.write_text(json.dumps(page))
        first_request = False
        if total_results is None:
            total_results = int(page.get("totalResults", 0))
        batch = page.get("vulnerabilities", [])
        if not batch:
            break
        records.extend(batch)
        start_index += len(batch)
        if total_results is not None and start_index >= total_results:
            break
    return records[:max_cves]


def parse_cpe(criteria: str) -> tuple[str, str] | None:
    """Extract ``(vendor, product)`` from a CPE 2.3 string.

    Args:
        criteria: e.g. ``cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*``.

    Returns:
        Lower-cased ``(vendor, product)`` or ``None`` if unparsable.
    """
    parts = criteria.split(":")
    if len(parts) < 5 or parts[0] != "cpe" or parts[1] != "2.3":
        return None
    vendor, product = parts[3].strip().lower(), parts[4].strip().lower()
    if not vendor or not product or vendor == "*" or product == "*":
        return None
    return (vendor, product)


def _cve_products(cve: dict) -> list[tuple[str, str]]:
    """Collect normalized ``(vendor, product)`` pairs from a CVE's CPE matches."""
    products: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in cve.get("configurations", []) or []:
        for match in node.get("nodes", []) or []:
            for cpe in match.get("cpeMatch", []) or []:
                parsed = parse_cpe(cpe.get("criteria", ""))
                if parsed and parsed not in seen:
                    seen.add(parsed)
                    products.append(parsed)
    return products


def _cve_english_description(cve: dict) -> str | None:
    """Return the English description, or ``None`` for rejected/undescribed CVEs."""
    for desc in cve.get("descriptions", []) or []:
        if desc.get("lang") == "en":
            raw = desc.get("value", "")
            value = raw.strip() if isinstance(raw, str) else ""
            if value and "** REJECT **" not in value:
                return value
    return None


def _cve_severity(cve: dict) -> str:
    """Best-effort CVSS base severity label, else ``""``."""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        for metric in (cve.get("metrics", {}) or {}).get(key, []) or []:
            data = metric.get("cvssData", {})
            if data.get("baseSeverity"):
                return f"{data['baseSeverity']} ({data.get('baseScore', '?')})"
    return ""


def cve_to_document(cve: dict) -> tuple[Document, list[tuple[str, str]]] | None:
    """Convert one NVD ``cve`` object to a corpus document.

    Returns:
        ``(Document, products)`` or ``None`` when the CVE has no usable
        English description.
    """
    cve_id = cve.get("id", "")
    description = _cve_english_description(cve)
    if not cve_id or not description:
        return None
    products = _cve_products(cve)
    parts = [f"{cve_id}.", description]
    if products:
        affected = ", ".join(f"{vendor} {product}" for vendor, product in products[:8])
        parts.append(f"Affected products: {affected}.")
    severity = _cve_severity(cve)
    if severity:
        parts.append(f"Severity: {severity}.")
    return (Document(doc_id=cve_id, text=" ".join(parts)), products)


def build_cve_dataset(
    records: list[dict],
    top_products: int = 10,
    max_queries: int = 20,
) -> RetrievalDataset:
    """Build an evaluation dataset from raw NVD ``vulnerabilities`` records.

    Product queries take the form ``"vulnerabilities affecting <vendor>
    <product>"`` for the most-affected products; their qrels are exact by
    construction (derived from the same CPE data as the documents).
    Additional natural-language queries use explicit substring rules
    (see :data:`CVE_NL_QUERIES`) so their qrels are deterministic.

    Args:
        records: Raw ``vulnerabilities`` entries from the NVD API.
        top_products: How many top products become queries.
        max_queries: Cap on total queries (product + NL).

    Returns:
        A :class:`RetrievalDataset` named ``"cve"``.
    """
    documents: list[Document] = []
    doc_products: dict[str, list[tuple[str, str]]] = {}
    for record in records:
        parsed = cve_to_document(record.get("cve", {}))
        if parsed is None:
            continue
        document, products = parsed
        documents.append(document)
        doc_products[document.doc_id] = products

    queries: list[Query] = []
    qrels: dict[str, set[str]] = {}

    # Product queries: exact qrels from CPE data.
    product_counts: dict[tuple[str, str], int] = {}
    product_docs: dict[tuple[str, str], set[str]] = {}
    for doc_id, products in doc_products.items():
        for product in products:
            product_counts[product] = product_counts.get(product, 0) + 1
            product_docs.setdefault(product, set()).add(doc_id)
    ranked = sorted(product_counts.items(), key=lambda item: (-item[1], item[0]))
    for (vendor, product_name), _count in ranked[:top_products]:
        query_id = f"product-{vendor}-{product_name}".replace(" ", "-")
        queries.append(
            Query(
                query_id=query_id,
                text=f"vulnerabilities affecting {vendor} {product_name}",
            )
        )
        qrels[query_id] = set(product_docs[(vendor, product_name)])
        if len(queries) >= max_queries:
            break

    # Natural-language queries with explicit substring rules.
    descriptions = {document.doc_id: document.text.lower() for document in documents}
    for query_id, text, phrase in CVE_NL_QUERIES:
        if len(queries) >= max_queries:
            break
        matched = {doc_id for doc_id, lowered in descriptions.items() if phrase in lowered}
        if len(matched) >= 3:
            queries.append(Query(query_id=query_id, text=text))
            qrels[query_id] = matched

    return RetrievalDataset(name="cve", documents=documents, queries=queries, qrels=qrels)


def download_cve_dataset(
    data_dir: str | Path = "data",
    max_cves: int = 3000,
) -> RetrievalDataset:
    """Download recent NVD CVEs and cache the parsed evaluation dataset.

    Writes the parsed dataset to ``<data_dir>/nvd.json`` (gitignored);
    raw pages are cached under ``<data_dir>/nvd-pages/`` for resumability.

    Args:
        data_dir: Cache directory.
        max_cves: Upper bound on CVE records to fetch.

    Returns:
        The parsed CVE :class:`RetrievalDataset`.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    records = download_nvd_pages(data_dir, max_cves=max_cves)
    dataset = build_cve_dataset(records)
    cache_path = data_dir / "nvd.json"
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


def load_cve_corpus(data_dir: str | Path = "data") -> RetrievalDataset:
    """Load the cached NVD CVE corpus.

    Raises:
        FileNotFoundError: If ``nvd.json`` was never downloaded.
    """
    cache_path = Path(data_dir) / "nvd.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"CVE cache not found at {cache_path}. Run `retrieval-lab download-cve` first."
        )
    return _dataset_from_json(json.loads(cache_path.read_text()))
