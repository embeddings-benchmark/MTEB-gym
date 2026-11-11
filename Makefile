install:
	uv sync --group dev --frozen

lint:
	uv run --no-sync ruff format .
	uv run --no-sync ruff check . --fix --exit-non-zero-on-fix

lint-check:
	uv run --no-sync ruff format . --check
	uv run --no-sync ruff check .

test:
	uv run --no-sync --group test pytest -q

typecheck:
	uv run --locked --group typing mypy

pr: lint typecheck test
