# The verification gate. CONTRIBUTING.md, RELEASING.md and CI all run exactly this.
.PHONY: gate lint typecheck test
gate: lint typecheck test

lint:
	uv run ruff check src tests benchmarks

typecheck:
	uv run mypy src/chad

test:
	uv run pytest -q
