"""Tests for the NVD CVE dataset builder (fully offline via fixture)."""

import json
from pathlib import Path

import pytest

from retrieval_lab.datasets import (
    build_cve_dataset,
    cve_to_document,
    load_cve_corpus,
    parse_cpe,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cve-mini.json"


def _records() -> list[dict]:
    return json.loads(FIXTURE.read_text())["records"]


def test_parse_cpe_extracts_vendor_product() -> None:
    assert parse_cpe("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*") == ("apache", "log4j")
    assert parse_cpe("cpe:2.3:o:microsoft:windows_server:2019:*:*:*:*:*:*:*") == (
        "microsoft",
        "windows_server",
    )
    assert parse_cpe("not-a-cpe") is None
    assert parse_cpe("cpe:2.3:a:*:log4j:*:*:*:*:*:*:*:*") is None
    assert parse_cpe("cpe:2.3:a:apache:*:*:*:*:*:*:*:*:*") is None


def test_cve_to_document_rejects_unusable_records() -> None:
    records = _records()
    rejected = next(r for r in records if r["cve"]["id"] == "CVE-2024-99999")
    assert cve_to_document(rejected["cve"]) is None
    french_only = next(r for r in records if r["cve"]["id"] == "CVE-2024-88888")
    assert cve_to_document(french_only["cve"]) is None


def test_cve_to_document_includes_id_products_and_severity() -> None:
    records = _records()
    log4shell = next(r for r in records if r["cve"]["id"] == "CVE-2021-44228")
    parsed = cve_to_document(log4shell["cve"])
    assert parsed is not None
    document, products = parsed
    assert document.doc_id == "CVE-2021-44228"
    assert "CVE-2021-44228" in document.text
    assert "apache log4j" in document.text
    assert ("apache", "log4j") in products
    assert "CRITICAL" in document.text


def test_build_cve_dataset_filters_and_builds_exact_qrels() -> None:
    dataset = build_cve_dataset(_records())
    assert dataset.name == "cve"
    # 12 records in fixture; rejected + French-only are dropped.
    assert len(dataset.documents) == 10

    qrels = dataset.qrels
    assert qrels["product-apache-log4j"] == {"CVE-2021-44228", "CVE-2021-45046"}
    assert qrels["product-microsoft-windows_server"] == {"CVE-2019-0708", "CVE-2020-1472"}
    # Every qrel must reference a real document.
    doc_ids = {doc.doc_id for doc in dataset.documents}
    for query_id, relevant in qrels.items():
        assert relevant, f"empty qrels for {query_id}"
        assert relevant <= doc_ids, f"dangling qrels for {query_id}"

    # Natural-language query from the substring rule.
    assert qrels["nl-buffer-overflow"] == {
        "CVE-2014-0160",
        "CVE-2023-4966",
        "CVE-2019-0708",
    }


def test_build_cve_dataset_queries_are_unique_and_capped() -> None:
    dataset = build_cve_dataset(_records(), top_products=3, max_queries=4)
    assert len(dataset.queries) <= 4
    query_ids = [q.query_id for q in dataset.queries]
    assert len(set(query_ids)) == len(query_ids)


def test_load_cve_corpus_missing_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download-cve"):
        load_cve_corpus(tmp_path)


def test_end_to_end_bm25_on_cve_fixture() -> None:
    from retrieval_lab.evaluate import run_experiment
    from retrieval_lab.pipeline import RetrieverConfig

    dataset = build_cve_dataset(_records())
    config = RetrieverConfig.from_yaml(Path("configs/bm25-only.yaml"))
    metrics = run_experiment(config, dataset, ks=[5])
    assert metrics["recall@5"] > 0.0
    assert metrics["mrr@5"] > 0.0
