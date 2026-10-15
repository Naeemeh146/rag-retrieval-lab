# Case study: CVE retrieval over public NVD data

## Problem statement

Security analysts triage thousands of new CVEs every month. When a new
vulnerability drops — or when an asset inventory needs screening — the first
question is always retrieval: *which CVEs affect the products I care about?*
Analysts typically grep NVD or rely on keyword search, which misses
paraphrased descriptions ("remote code execution" vs "arbitrary code
execution") and drowns in false positives on popular terms.

This case study asks whether a two-stage retrieval pipeline (hybrid
BM25 + dense search with Reciprocal Rank Fusion, plus cross-encoder reranking)
materially beats plain keyword search on real vulnerability data — and *where*
each method wins. All data is public NVD (no proprietary sources).

## Dataset construction

- **Source:** NVD API 2.0 (`services.nvd.nist.gov`), unauthenticated,
  respecting the 5-requests-per-30-seconds limit. CVEs published
  2026-07-01 → 2026-10-15 (NVD's 120-day window).
- **Corpus:** 2,995 CVE documents. Each document is the CVE ID, the English
  description, affected `vendor product` pairs parsed from CPE 2.3 strings,
  and the CVSS base severity where available. Rejected and non-English
  entries are dropped.
- **Queries (13) with exact ground truth:**
  - *Product queries* (10): `"vulnerabilities affecting <vendor> <product>"`
    for the ten most-affected products (google chrome: 79 CVEs,
    linux linux_kernel: 63, microsoft edge_chromium: 47, microsoft windows:
    43, apache camel: 34, foxit pdf_editor/pdf_reader: 28 each, dell
    data_domain_operating_system: 25, mediawiki mediawiki: 24, coder coder:
    20). Qrels are exact by construction — derived from the same CPE data
    as the documents.
  - *Natural-language queries* (3): "buffer overflow vulnerabilities" (76),
    "SQL injection vulnerabilities" (176), "cross-site scripting (XSS)
    vulnerabilities" (235), with qrels from explicit substring rules on the
    descriptions (deterministic and auditable).
- Reproduce: `retrieval-lab download-cve --out data` then
  `retrieval-lab evaluate --dataset cve --configs configs/cve-*.yaml`.

## Configurations tested

| config | stage 1 | stage 2 | notes |
|---|---|---|---|
| `cve-bm25` | BM25 | — | lexical baseline, fully offline |
| `cve-dense-st` | sentence-transformer (`all-MiniLM-L6-v2`) | — | learned dense baseline |
| `cve-hybrid-st` | BM25 + dense, RRF | — | |
| `cve-hybrid-rerank` | BM25 + dense, RRF | cross-encoder (`ms-marco-MiniLM-L-6-v2`) | |
| `cve-chunking-fixed` / `overlapping` / `recursive` | BM25, chunking ablations | — | fully offline |

## Results

Real numbers from `retrieval-lab evaluate --dataset cve` (2,995 docs,
13 queries; recall/MRR/nDCG @ 5, 10):

| config | recall@5 | mrr@5 | ndcg@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| cve-bm25 | 0.1264 | 1.0000 | 1.0000 | 0.2412 | 1.0000 | 0.9641 |
| cve-dense-st | 0.0998 | 0.9231 | 0.8603 | 0.1928 | 0.9359 | 0.8157 |
| cve-hybrid-st | 0.1228 | 1.0000 | 0.9734 | 0.2212 | 1.0000 | 0.9147 |
| cve-hybrid-rerank | 0.1134 | 1.0000 | 0.9413 | 0.2334 | 1.0000 | 0.9408 |
| cve-chunking-fixed | 0.1246 | 1.0000 | 0.9888 | 0.2353 | 1.0000 | 0.9292 |
| cve-chunking-overlapping | 0.1264 | 1.0000 | 1.0000 | 0.2415 | 1.0000 | 0.9649 |
| cve-chunking-recursive | 0.1264 | 1.0000 | 1.0000 | 0.2323 | 1.0000 | 0.9386 |

Per-query recall@10 (BM25 vs dense) shows where the gap lives:

| query | relevant | BM25 hit@10 | dense hit@10 |
|---|---|---|---|
| product-google-chrome | 79 | 10 | 9 |
| product-linux-linux_kernel | 63 | 10 | 9 |
| product-microsoft-edge_chromium | 47 | 10 | 9 |
| product-microsoft-windows | 43 | 10 | **3** |
| product-apache-camel | 34 | 7 | 8 |
| product-foxit-pdf_editor | 28 | 10 | 10 |
| product-foxit-pdf_reader | 28 | 10 | 10 |
| product-dell-data_domain_operating_system | 25 | 10 | 10 |
| product-mediawiki-mediawiki | 24 | 9 | 9 |
| product-coder-coder | 20 | 10 | **2** |
| nl-buffer-overflow | 76 | 9 | 6 |
| nl-sql-injection | 176 | 10 | 9 |
| nl-xss | 235 | 8 | 8 |

## Takeaways

1. **BM25 dominates this corpus — and MRR@5 = 1.0 says why.** Every
   query's key terms appear verbatim in relevant documents (product names in
   the "Affected products" line, phrases like "buffer overflow"). BM25 puts a
   relevant CVE at rank 1 for all 13 queries. When the vocabulary is
   controlled, lexical search is nearly optimal.
2. **Dense retrieval underperforms — and fails instructively.** The
   sentence-transformer trails BM25 on recall@10 (0.193 vs 0.241) and nDCG@10
   (0.816 vs 0.964). Its worst misses are on *generic* product terms:
   `microsoft windows` (3/10 vs 10/10) and `coder coder` (2/10 vs 10/10),
   where semantic similarity can't disambiguate what exact string matching
   nails. A general-purpose embedding model adds noise where precision on
   named entities is what matters.
3. **Hybrid lands between its parents.** RRF fusion (recall@10 0.221) can't
   rescue dense's misses on this corpus, but doesn't drag BM25 down much
   either — the expected behavior when one signal is strictly stronger.
4. **Reranking trades recall@5 for nDCG@10.** The cross-encoder slightly
   lowers recall@5 (0.113 vs 0.123) but lifts nDCG@10 (0.941 vs 0.915):
   better ordering of what was found, at the cost of ~650 cross-encoder
   inferences for 13 queries — the classic stage-2 latency/quality tradeoff,
   now measured.
5. **Chunking barely matters here.** CVE documents are short (median 454
   chars); all three strategies land within noise of each other. Chunking
   strategy matters for long documents, not for CVE-sized passages.
6. **Read the absolute recall numbers with the qrels in mind.** Qrels run
   20–235 relevant per query, so recall@10 is capped by construction (e.g.
   10/79 = 0.127 max for the chrome query). MRR/nDCG — ranking quality —
   are the discriminating metrics here, and that's where BM25 is near-perfect.

## Limitations & next steps

- Single NVD snapshot (Jul–Oct 2026); CPE coverage varies by CVE age.
- Dense numbers use a general-purpose embedding model, not a
  security-tuned one — domain fine-tuning (or a CVE-specific
  bi-encoder) is the obvious next experiment, and the harness is ready for it.
- Natural-language qrels come from substring rules, so a description saying
  "not vulnerable to XSS" would count as relevant — a known, documented
  heuristic. Product qrels are exact.
- No temporal split: a production system should evaluate on *future* CVEs
  relative to its index build date.
