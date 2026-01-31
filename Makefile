.PHONY: install install-dev test lint examples examples-all docs

install:
	python -m pip install -e . --no-build-isolation

install-dev:
	python -m pip install -e ".[dev]" --no-build-isolation

test:
	python -m pytest -q

lint:
	ruff check src tests examples
	ruff format src tests examples
	black src tests examples

examples:
	python examples/run_slab_scan.py

examples-all:
	python examples/run_slab_scan.py
	python examples/run_tabulated_geom.py
	python examples/run_circular_tokamak.py
	python examples/run_cyclone_salpha.py

docs:
	mkdocs build --strict

