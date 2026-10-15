# rag-retrieval-lab

**rag-retrieval-lab is a hands-on laboratory for two-stage retrieval in RAG systems: stage 1 runs hybrid BM25+dense search fused with Reciprocal Rank Fusion, stage 2 applies cross-encoder reranking, and every experiment is benchmarked with recall/MRR/nDCG so you can compare chunking strategies, fusion methods, and retriever configurations on real numbers.**

[![CI](https://github.com/Naeemeh146/rag-retrieval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Naeemeh146/rag-retrieval-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

## Problem statement

Retrieval quality decides RAG answer quality, yet most tutorials stop at "embed and
cosine-search". Real pipelines need answers to harder questions: does hybrid
lexical+dense search beat either alone? How much does a reranker add? Which
chunking strategy preserves the passages your queries actually need? This lab
gives you a clean, fully offline codebase to run those ablations yourself —
with BEIR SciFact one command away when you want a real benchmark.

## Architecture

```mermaid
flowchart LR
    Docs[Documents] --> Chunk[Chunking\nfixed / overlapping / recursive]
    Chunk --> BM25[BM25 index]
    Chunk --> Dense[Dense embeddings]
    Q[Query] --> BM25
    Q --> Dense
    BM25 --> RRF[Reciprocal Rank Fusion]
    Dense --> RRF
    RRF --> Rerank{Cross-encoder\nrerank?}
    Rerank -->|yes| Out[Ranked passages]
    Rerank -->|no| Out
    Out --> Eval[recall@k / MRR@k / nDCG@k]
```

**Stage 1 — hybrid retrieval.** BM25 (`rank-bm25`) and dense cosine search each
return top-k chunk rankings; Reciprocal Rank Fusion merges them by rank alone,
so incomparable lexical/dense score scales never need normalization.

**Stage 2 — reranking.** An optional cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores query–passage pairs jointly and
keeps the top-n. A Cohere Rerank provider is available behind `COHERE_API_KEY`.

**Evaluation.** `recall@k`, `MRR@k`, and `nDCG@k` are implemented in numpy (no
`pytrec_eval` C dependency) and averaged over queries.

## Quickstart

```bash
pip install -e .

# Run the offline benchmark suite on the built-in synthetic corpus
retrieval-lab evaluate \
  --configs configs/bm25-only.yaml \
  --configs configs/dense-only.yaml \
  --configs configs/hybrid.yaml \
  --dataset synthetic \
  --out results

# Ask a question against a built index
retrieval-lab ingest --config configs/hybrid.yaml --dataset synthetic --out data/index
retrieval-lab query "What is nDCG in information retrieval evaluation?" --index data/index

# Optional: download BEIR SciFact for a real benchmark (needs network)
retrieval-lab download-data --out data
```

The default path needs **no API keys and no model downloads**: deterministic
hash-based embeddings and BM25 run fully offline. Learned providers
(`sentence-transformer`, `cohere`, `cross-encoder`) are opt-in.

## Benchmark results

Real numbers from `make eval` on the built-in synthetic corpus (16 documents,
8 queries, deterministic offline embeddings, `recall/MRR/nDCG @ 5, 10`):

| config                | recall@5 | mrr@5 | ndcg@5 | recall@10 | mrr@10 | ndcg@10 |
|-----------------------|----------|-------|--------|-----------|--------|---------|
| bm25-only             | 0.896    | 0.760 | 0.756  | 0.958     | 0.760  | 0.783   |
| dense-only            | 0.625    | 0.588 | 0.505  | 0.875     | 0.601  | 0.597   |
| hybrid                | 0.688    | 0.667 | 0.608  | 0.958     | 0.708  | 0.719   |
| chunking-fixed        | 0.854    | 0.681 | 0.672  | 0.958     | 0.681  | 0.718   |
| chunking-overlapping  | 0.688    | 0.604 | 0.577  | 0.917     | 0.643  | 0.667   |
| chunking-recursive    | 0.688    | 0.667 | 0.608  | 0.958     | 0.708  | 0.719   |

Takeaways (corpus-specific, not universal truths):

- **BM25 dominates this corpus** — queries were written with distinctive
  keywords, which is exactly the regime where lexical matching shines.
- **Hash-based "dense" is a weak dense model** — it captures term overlap, not
  semantics, so treat `dense-only` as a plumbing check rather than a real
  dense-retrieval result. Swap in `sentence-transformer` for meaningful numbers.
- **RRF fusion lands between its parents** here; on corpora where lexical and
  dense fail on *different* queries, hybrid typically beats both.
- `chunking-recursive` matches `hybrid` exactly — same config, confirming the
  harness is deterministic.

Reproduce with `make eval`; results land in `results/results.{json,csv}`.

## Configuration guide

Experiments are YAML files validated by pydantic v2 (`RetrieverConfig`):

```yaml
name: hybrid
chunker:   {name: recursive, chunk_size: 200, chunk_overlap: 50}
embedding: {provider: deterministic, dimension: 384}   # or sentence-transformer / cohere
bm25:      {enabled: true, k1: 1.5, b: 0.75, top_k: 50}
dense:     {enabled: true, top_k: 50}
fusion:    {method: rrf, k: 60, top_k: 50}
rerank:    {enabled: false, provider: none, top_n: 10}  # or cross-encoder / cohere
```

| Config                  | What it ablates                          | Offline? |
|-------------------------|------------------------------------------|----------|
| `bm25-only.yaml`        | Lexical baseline                         | ✅        |
| `dense-only.yaml`       | Dense baseline                           | ✅        |
| `hybrid.yaml`           | BM25 + dense with RRF                    | ✅        |
| `hybrid-rerank.yaml`    | + cross-encoder rerank (stage 2)         | ❌ needs model download |
| `chunking-fixed.yaml`   | Fixed-size chunking                      | ✅        |
| `chunking-overlapping.yaml` | Sliding-window chunking              | ✅        |
| `chunking-recursive.yaml`   | Recursive character chunking         | ✅        |

Provider selection: `deterministic` (offline default) · `sentence-transformer`
(local, `pip install "retrieval-lab[models]"`) · `cohere` (needs
`COHERE_API_KEY`, see `.env.example`).

## Project layout

```
src/retrieval_lab/
  chunking.py    # fixed / overlapping / recursive chunkers
  embeddings.py  # provider interface: deterministic, sentence-transformers, Cohere
  lexical.py     # BM25 via rank-bm25
  fusion.py      # Reciprocal Rank Fusion
  rerank.py      # reranker interface: cross-encoder, Cohere, no-op
  pipeline.py    # TwoStageRetriever + pydantic YAML configs
  metrics.py     # recall@k, MRR@k, nDCG@k in numpy
  datasets.py    # synthetic corpus + BEIR SciFact downloader
  evaluate.py    # experiment harness -> results/{json,csv}
  cli.py         # typer CLI: download-data, ingest, query, evaluate
configs/         # ablation YAML configs
tests/           # pytest suite (model tests marked `slow`)
```

## Development

```bash
make install-dev   # editable install with dev extras
make lint          # ruff check + format check
make typecheck     # mypy src/
make test          # pytest -m "not slow" (offline)
make test-slow     # includes model-download tests
pre-commit install # ruff + mypy hooks
```

## Roadmap

- [ ] Vector-database backend (Qdrant/Chroma) for the dense index
- [ ] Serving API (FastAPI) exposing `/search` over a built index
- [ ] BEIR benchmark table (SciFact, NFCorpus) with learned embeddings
- [ ] Hybrid score normalization ablations (vs. RRF)
- [ ] Query rewriting / HyDE as a pre-retrieval stage

## Citation

If you use this lab in your work, please cite it:

```bibtex
@software{yadollahpour2026ragretrievallab,
  author  = {Yadollahpour, Naeemeh},
  title   = {rag-retrieval-lab: Two-Stage RAG Retrieval Lab},
  version = {0.1.0},
  date    = {2026-10-15},
  url     = {https://github.com/Naeemeh146/rag-retrieval-lab},
  license = {MIT}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable record.

## License

MIT — see [LICENSE](LICENSE).
