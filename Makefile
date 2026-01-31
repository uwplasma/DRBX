.PHONY: install install-dev test lint examples examples-all examples-literature examples-stellarator docs

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

examples-literature:
	python examples/literature/mosetto2012_driftwaves.py
	python examples/literature/mosetto2012_ballooning.py
	python examples/literature/halpern2013_gradient_removal.py
	python examples/literature/cyclone_kxky_scan.py

examples-stellarator:
	PYTHONPATH=../pyQSC-main python examples/run_pyqsc_stellarator.py

docs:
	mkdocs build --strict
