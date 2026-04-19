# Release Notes: 1.0.0

`jax_drb 1.0.0` is the first package-oriented release of the project.

This release is built around a selected-lane research claim:

- native CLI and Python-driver workflows;
- restartable runs with structured artifact output;
- promoted native validation lanes in 1D, 2D, and reduced 3D workflows;
- committed comparison, profiling, convergence, and runtime bundles;
- bounded controller, recycling, neutral, impurity, and geometry-adapter validation surfaces.

## Highlights

- packaged Python distribution through `pyproject.toml`
- PyPI Trusted Publishing workflow in
  [`publish-pypi.yml`](../.github/workflows/publish-pypi.yml)
- explicit release/packaging guide in [release_packaging.md](release_packaging.md)
- `95%` bounded release-closeout coverage gate through
  [`scripts/run_closeout_coverage.py`](../scripts/run_closeout_coverage.py)
- reduced but real 3D native tokamak, traced-field-line, and stellarator artifact bundles
- richer direct tokamak recycling windows, including neon-enabled bounded
  `nout=3` and `nout=5` short-window gates
- bounded reduced `temperature_feedback` and `detachment_controller` lanes

## Installation

From PyPI after publish:

```bash
pip install jax-drb
```

From a checkout:

```bash
pip install -e .[dev,integrators,models,validation]
```

## What Is Included

Core user-facing capabilities:

- TOML-driven native runs
- Python API entry points for curated and deck-driven cases
- verbose run logs and restart bundles
- portable JSON/NPZ artifact outputs
- documentation galleries with publication-ready figures and GIFs

Validation and engineering surfaces:

- parity ladders and benchmark adapters
- Hermes-backed reactions/collisions and impurity/radiation campaigns
- controller-feedback, temperature-feedback, and detachment-controller reduced gates
- native 3D runtime, convergence, and profiling bundles

## Claim Boundary

This release supports:

- a strong public research-code release;
- a selected-lane JCP manuscript draft boundary.

It does **not** claim:

- a broad parity-complete standalone replacement across the full intended
  DRB reference-workflow matrix;
- a full-production temperature/detachment workflow;
- a broad end-to-end production 3D workflow.

Those remaining items are tracked in:

- [jcp_readiness_audit.md](jcp_readiness_audit.md)
- [hermes_capability_audit.md](hermes_capability_audit.md)
- [PLAN.md](../PLAN.md)

## Recommended First Commands

Smoke-test the packaged runtime:

```bash
jax_drb examples/inputs/restartable_diffusion.toml --verbose
```

Run the release-closeout coverage gate:

```bash
python scripts/run_closeout_coverage.py
```

Build the package locally:

```bash
python -m build
```

## Next Research Steps

- broaden production temperature/detachment workflows beyond reduced controller gates
- widen direct tokamak recycling beyond the current bounded windows
- extend reduced/native-selected 3D workflows into broader production 3D campaigns
- draft the selected-lane JCP paper using the committed artifact bundles
