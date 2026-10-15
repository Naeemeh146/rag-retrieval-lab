.PHONY: install install-dev lint format typecheck test test-slow eval download-data clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src/

test:
	pytest -m "not slow" -q

test-slow:
	pytest -q

eval:
	retrieval-lab evaluate \
		--configs configs/bm25-only.yaml \
		--configs configs/dense-only.yaml \
		--configs configs/hybrid.yaml \
		--configs configs/chunking-fixed.yaml \
		--configs configs/chunking-overlapping.yaml \
		--configs configs/chunking-recursive.yaml \
		--dataset synthetic \
		--out results

download-data:
	retrieval-lab download-data --out data

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
