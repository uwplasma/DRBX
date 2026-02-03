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
- **API stability** for key user-facing functions and dataclasses.

Examples of covered checks:

- Neutral stability in the *no-drive* limit ($\omega_n=\omega_{T_e}=0$) for the periodic case.
- Consistency of scan outputs (`gamma_eigs`, `omega_eigs`, eigenvalues) and file writing behavior.
- Open-field-line sheath closures: volumetric end-loss proxy and (simplified) MPSE boundary
  enforcement in small problems.

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
- packaging build (sdist/wheel).

The CI definition lives in:

- `.github/workflows/ci.yml`

## Philosophy and limitations

`jaxdrb` is a reduced model intended for fast, exploratory linear studies. The tests focus on:

- catching broken numerics and regressions,
- enforcing “known-limit” behaviors where appropriate,
- validating geometry pipelines and file I/O.

They do **not** claim quantitative agreement with any one published SOL closure set in all regimes.
Where the implementation uses simplified closures (e.g. linearized MPSE BC enforcement), the docs
and tests aim to make that explicit.

