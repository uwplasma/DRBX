# Release And Packaging

`jax_drb` is packaged as a standard Python project and published through GitHub Actions using PyPI Trusted Publishing.

## Install Paths

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

The default package install already includes the runtime, solver, plotting, and geometry dependencies used by the main CLI and analysis workflows.

## Private Release Artifacts

The repository keeps large generated `.npz`, `.png`, and `.gif` files out of
git. They are stored in the private release
`validation-artifacts-2026-04-28` as two bundles:

- `jax_drb_docs_media.zip` restores README/docs figures, movie GIFs, and
  example arrays under `docs/data/`;
- `jax_drb_reference_baselines.zip` restores heavy validation baselines under
  `references/baselines/`.

Users with repository access can restore both bundles from a fresh clone with:

```bash
gh auth login --hostname github.com
python scripts/fetch_example_artifacts.py
```

For non-CLI automation, set `GH_TOKEN` or `GITHUB_TOKEN` to a token with access
to `uwplasma/jax_drb`. The downloader uses the GitHub CLI first because private
release assets need authentication, then falls back to token-authenticated HTTPS.
Set `JAX_DRB_ARTIFACT_CACHE=/path/to/cache` to reuse downloaded archives across
checkouts, or `JAX_DRB_OFFLINE_ARTIFACTS=1` to require that artifacts already
exist locally.

This artifact path is the supported self-contained user workflow. Users do not
need to download any external plasma code to run the examples, view or
regenerate the README/docs movies, or execute the cached validation checks.
Fresh live-reference reruns are developer-maintenance tasks for refreshing the
release bundles.

## Build The Package

Build the source distribution and wheel locally:

```bash
python -m pip install build
python -m build
```

Expected outputs:

- `dist/jax_drb-<version>.tar.gz`
- `dist/jax_drb-<version>-py3-none-any.whl`

Validate the built metadata:

```bash
python -m pip install twine
python -m twine check dist/*
```

## GitHub Workflows

The repository includes:

- [`publish-pypi.yml`](../.github/workflows/publish-pypi.yml) for package publishing
- [`test.yml`](../.github/workflows/test.yml) for the Python 3.10, 3.11, and 3.12 test matrix

The PyPI publish workflow:

1. builds the wheel and sdist on GitHub Actions,
2. stores them as workflow artifacts,
3. publishes them to PyPI through OIDC with `id-token: write`,
4. uses the `pypi` GitHub environment for the publish job.

Publishing is triggered by a published GitHub release or by manual
`workflow_dispatch`. It is intentionally not triggered directly by tag pushes,
so creating a version tag and then publishing its GitHub release cannot submit
the same distribution to PyPI twice.

## Release Checklist

Before publishing a version:

1. run the bounded closeout and promoted solver coverage gates:

```bash
python scripts/run_closeout_coverage.py
python scripts/run_promoted_solver_coverage.py
```

2. run the fast bounded validation slice:

```bash
python scripts/run_fast_research_checks.py
```

3. build the distributions locally:

```bash
python -m build
```

4. verify the public docs and artifact surface:

```bash
mkdocs build --strict
pytest -q tests/test_release_surface.py
```

5. verify the release artifact bundle and docs-media restore path when release
   assets have changed:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines
pytest -q tests/test_runtime_artifacts.py
```

6. dispatch the bounded research campaign workflows that are expected for the
   release candidate, then wait for GitHub `test`, `docs`, and `coverage` to
   complete successfully on the target commit.

7. optionally run the Python version matrix locally or through CI.

## Current Release Boundary

The current package release is intended to support:

- standalone CLI and Python-driver workflows,
- promoted native-exact and native-operational validation lanes,
- reduced but real 3D tokamak, traced-field-line, and stellarator workflows,
- artifact-driven parity, runtime, convergence, and profiling reports.

It is not the full closure of every research workflow in the broader validation matrix. The detailed status remains in:

- [hermes_capability_audit.md](hermes_capability_audit.md)
- [implementation_inventory.md](implementation_inventory.md)
- [parity_harness.md](parity_harness.md)
- [parity_matrix.md](parity_matrix.md)

## After The First Package Release

The main post-release technical targets are:

- broader production temperature and detachment workflows,
- longer-window direct tokamak recycling closure,
- broader production 3D workflows beyond the reduced native matrix.
