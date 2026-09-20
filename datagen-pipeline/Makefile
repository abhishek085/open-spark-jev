PY ?= ../.venv/bin/python
export PYTHONPATH := src

.PHONY: test lint dry-run clean
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests examples
dry-run:
	$(PY) -m os_datagen.cli generate --task-pack foundation_document_type_v1 --count 100 --seed 42 --dry-run --out artifacts/dry_run
clean:
	rm -rf artifacts/dry_run .pytest_cache
