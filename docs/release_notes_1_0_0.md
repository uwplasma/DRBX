# Release Notes: 1.0.0

`jax_drb 1.0.0` is the first package-oriented release of the project.

## Highlights

- packaged Python distribution through `pyproject.toml`
- PyPI Trusted Publishing workflow in [`publish-pypi.yml`](../.github/workflows/publish-pypi.yml)
- Python 3.10, 3.11, and 3.12 test workflow in [`test.yml`](../.github/workflows/test.yml)
- `95%` bounded closeout coverage gate through [`scripts/run_closeout_coverage.py`](../scripts/run_closeout_coverage.py)
- promoted native validation lanes in 1D, 2D, and reduced 3D workflows
- structured runtime, comparison, convergence, and profiling artifact bundles
- bounded controller, recycling, neutral, impurity, and geometry-adapter validation surfaces

## Installation

From PyPI:

```bash
pip install jax-drb
```

From a checkout:

```bash
git clone https://github.com/uwplasma/jax_drb
cd jax_drb
pip install -e .
```

## Included Capabilities

Core user-facing capabilities:

- TOML-driven native runs
- Python API entry points for curated and deck-driven cases
- verbose run logs and restart bundles
- portable JSON and NPZ artifact outputs
- 2D and 3D movies, plots, and benchmark summaries

Validation and engineering surfaces:

- parity ladders and geometry adapters
- Hermes-backed reactions/collisions and impurity/radiation campaigns
- controller-feedback, temperature-feedback, and detachment-controller bounded gates
- native 3D runtime, convergence, and profiling bundles

## Current Scope

This release supports:

- a strong standalone public research-code release,
- native and benchmark-backed workflows on the promoted validation matrix.

Broader production workflows remain active engineering work, especially:

- larger temperature and detachment control workflows,
- longer-window direct tokamak recycling,
- broader production 3D campaigns beyond the reduced native matrix.

Detailed status is tracked in:

- [implementation_inventory.md](implementation_inventory.md)
- [parity_harness.md](parity_harness.md)
- [parity_matrix.md](parity_matrix.md)

## Recommended First Commands

Smoke-test the packaged runtime:

```bash
jax_drb examples/inputs/restartable_diffusion.toml --verbose
```

Run the bounded closeout coverage gate:

```bash
python scripts/run_closeout_coverage.py
```

Build the package locally:

```bash
python -m build
```
