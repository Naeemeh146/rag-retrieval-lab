# Contributing to rag-retrieval-lab

Thanks for your interest! This is a portfolio lab, but issues and pull requests
are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
make install-dev
pre-commit install
```

## Workflow

1. Create a branch from `main`: `git checkout -b feat/my-change`.
2. Add or update tests for any behavior change (`pytest -m "not slow"` must pass).
3. Run the gates before pushing:
   ```bash
   make lint && make typecheck && make test
   ```
4. Format with `ruff format`; keep line length at 100.
5. Open a PR with a clear description and the experiment results if you touch
   retrieval behavior.

## Conventions

- Type hints on everything; Google-style docstrings on public functions.
- New providers/rerankers go behind the existing `build_*` factories.
- Tests needing model downloads or API keys get `@pytest.mark.slow`.
- No secrets in code or configs — use `.env` (see `.env.example`).
