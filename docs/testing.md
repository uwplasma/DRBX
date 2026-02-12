# Testing

This page describes what is tested in `jaxdrb`, how to run the test suite locally, and what
guarantees (and limitations) the tests provide.

## Quick start

From the repository root:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

To run the docs build in “strict” mode (fails on broken links / missing pages):

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
```

## What is covered

The tests aim to provide:

- **Regression protection** for the linear operators and model RHS implementations.
- **Basic physics sanity checks** in known “no-drive” limits.
- **Hard conservative gates** for selected nonlinear and field-line DRB subsets.
- **API stability** for key user-facing functions and dataclasses.

Examples of covered checks:

- Neutral stability in the *no-drive* limit ($\omega_n=\omega_{T_e}=0$) for the periodic case.
- Consistency of scan outputs (`gamma_eigs`, `omega_eigs`, eigenvalues) and file writing behavior.
- Open-field-line sheath closures: volumetric end-loss proxy and (simplified) MPSE boundary
  enforcement in small problems.
- Quantitative sheath gates for EM current closure and heat/SEE closure terms.
- Conservative invariant gates:
  - HW2D ideal subset (`tests/test_hw2d_conservative_gate.py`),
  - cold-ion DRB periodic subset (`tests/test_drb_nonlinear_conservative_gate.py`),
  - cold-ion DRB operator split and residual gates (`tests/test_drb_operator_rates.py`, `tests/test_drb_operator_split.py`).

## Optional ESSOS tests

Some geometry pipelines require ESSOS (VMEC / near-axis / Biot–Savart). These tests are **optional**:

- If ESSOS is **not installed**, the tests are skipped.
- If ESSOS **is installed**, the tests perform smoke checks that the conversion routines produce a
  valid `TabulatedGeometry` file.

This behavior is implemented using `pytest.importorskip("essos")` in:

- `tests/test_essos_geometry_optional.py`

## CI

GitHub Actions runs:

- linting (`ruff`, `black`),
- unit tests (`pytest`),
- docs build (`mkdocs build --strict`),
- packaging build (sdist/wheel),
- performance gate (`benchmarks/check_core_kernels.py`),
- physics-conservation benchmark gate (`benchmarks/check_drb_conservative_gate.py`).

The CI definition lives in:

- `.github/workflows/ci.yml`

## Philosophy and limitations

`jaxdrb` is a reduced-model solver suite with **both linear and nonlinear** workflows. The tests focus on:

- catching broken numerics and regressions,
- enforcing “known-limit” behaviors where appropriate,
- validating geometry pipelines and file I/O,
- enforcing **conservation / budget closure** identities for conservative operator subsets,
- preventing nonlinear-regime regressions via broad-band statistics gates.

The repository’s benchmark gates are designed to be reviewer-auditable. They still do **not** claim
that any single closure set matches all SOL regimes across all devices; instead, the code keeps
closures modular (toggleable), documents assumptions explicitly, and provides reproducible gates that
quantify when identities/limits should hold.
