PYTHON  = uv run python
SRC     = src
ARGS    ?=

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m $(SRC) $(ARGS)

debug:
	$(PYTHON) -m pdb -m $(SRC) $(ARGS)

clean:
	find . -path ./.venv -prune -o -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) $(MYPY_FLAGS)

lint-strict:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --strict
