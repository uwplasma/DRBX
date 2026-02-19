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
	python examples/01_linear_basics/slab_ky_scan.py
	python examples/01_linear_basics/circular_tokamak_ky_scan.py
	python examples/01_linear_basics/salpha_cyclone_ky_scan.py
	python examples/03_sheath_mpse/open_slab_sheath_ky_scan.py

examples-intermediate:
	python examples/02_geometry/tabulated_geometry_roundtrip.py
	python examples/01_linear_basics/cyclone_kxky_scan.py
	python examples/05_jax_autodiff/autodiff_optimize_ky_star.py
	python examples/01_linear_basics/em_beta_scan.py
	python examples/01_linear_basics/hot_ions_tau_scan.py
	python examples/04_closures_transport/parallel_closures_effects.py
	python examples/03_sheath_mpse/sheath_heat_see_effects.py --out out_sheath_heat
	python examples/04_closures_transport/braginskii_closures_effects.py --out out_braginskii
	python examples/08_nonlinear_hw2d/mms_hw2d_convergence.py --out out_mms_hw2d
	python examples/10_verification/poisson_cg_verification.py --out out_poisson_cg_verify

examples-advanced:
	python examples/06_literature_tokamak_sol/mosetto2012_driftwave_branches.py
	python examples/06_literature_tokamak_sol/mosetto2012_ballooning_branches.py
	python examples/06_literature_tokamak_sol/halpern2013_gradient_removal_lp.py
	python examples/06_literature_tokamak_sol/jorge2016_isttok_linear_workflow.py
	python examples/06_literature_tokamak_sol/mosetto2012_regime_map.py
	python examples/06_literature_tokamak_sol/halpern2013_salpha_ideal_ballooning_map.py
	python examples/03_sheath_mpse/loizu2012_full_mpse_bc.py
	python examples/08_nonlinear_hw2d/hw2d_driftwave_turbulence.py
	python examples/08_nonlinear_hw2d/hw2d_neutrals_effect.py
	python examples/08_nonlinear_hw2d/hw2d_movie.py
	python examples/09_fci/fci_slab_parallel_derivative_mms.py --out out_fci_mms

examples-all: examples-simple examples-intermediate examples-advanced

examples-stellarator:
	python examples/07_essos_geometries/stellarator_nearaxis_essos.py

docs:
	mkdocs build --strict
