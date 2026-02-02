.PHONY: install install-dev test lint docs examples examples-simple examples-intermediate examples-advanced examples-all examples-stellarator

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
	$(MAKE) examples-simple

examples-simple:
	python examples/1_simple/01_slab_ky_scan.py
	python examples/1_simple/02_circular_tokamak_ky_scan.py
	python examples/1_simple/03_salpha_cyclone_ky_scan.py

examples-intermediate:
	python examples/2_intermediate/01_tabulated_geometry_roundtrip.py
	python examples/2_intermediate/02_cyclone_kxky_scan.py
	python examples/2_intermediate/03_jax_autodiff_optimize_ky_star.py

examples-advanced:
	python examples/3_advanced/01_mosetto2012_driftwave_branches.py
	python examples/3_advanced/02_mosetto2012_ballooning_branches.py
	python examples/3_advanced/03_halpern2013_gradient_removal_lp.py
	python examples/3_advanced/05_jorge2016_isttok_linear_workflow.py

examples-all: examples-simple examples-intermediate examples-advanced

examples-stellarator:
	PYTHONPATH=../pyQSC-main python examples/3_advanced/04_stellarator_nearaxis_pyqsc.py

docs:
	mkdocs build --strict
