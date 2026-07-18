# Release And Packaging

`drbx` is packaged as a standard Python project and published through GitHub Actions using PyPI Trusted Publishing.

## Install Paths

From PyPI:

```bash
pip install drbx
```

From a checkout:

```bash
git clone https://github.com/uwplasma/drbx
cd drbx
pip install -e .
```

The default package install already includes the runtime, solver, plotting, and geometry dependencies used by the main CLI and analysis workflows.

## Private Release Artifacts

The repository keeps large generated `.npz`, `.png`, and `.gif` files out of
git. They are stored in the private release
`validation-artifacts-2026-04-28` as one bundle:

- `drbx_docs_media.zip` restores README/docs figures, movie GIFs, and
  example arrays under `docs/data/`.

The release-hosted media map is defined by the artifact-restore helper
`scripts/fetch_example_artifacts.py`, backed by
`src/drbx/runtime/artifacts.py`, so release reviewers can verify which PNG,
GIF, MP4, and NPZ URLs are expected for a given artifact tag.

The current docs-media bundle
contains `174` media files, including the diverted-tokamak movie arrays, the
3D tokamak GIF, the compact stellarator FCI showcase, and the imported-field
QA-hybrid stationarity/Jacobi movie used in the README.

Users with repository access can restore the docs-media bundle from a fresh clone with:

```bash
gh auth login --hostname github.com
python scripts/fetch_example_artifacts.py
```

For non-CLI automation, set `GH_TOKEN` or `GITHUB_TOKEN` to a token with access
to `uwplasma/drbx`. The downloader uses the GitHub CLI first because private
release assets need authentication, then falls back to token-authenticated HTTPS.
Set `DRBX_ARTIFACT_CACHE_DIR=/path/to/cache` to reuse downloaded archives
across checkouts; the older `DRBX_ARTIFACT_CACHE` name is also accepted.
Set `DRBX_ARTIFACT_DOWNLOAD_TIMEOUT` and
`DRBX_ARTIFACT_DOWNLOAD_ATTEMPTS` to tune the HTTPS fallback used when the
GitHub CLI is unavailable. Set `DRBX_OFFLINE_ARTIFACTS=1` to require that
artifacts already exist locally.

This artifact path is the supported self-contained user workflow. Users do not
need to download any external plasma code to run the examples, view or
regenerate the README/docs movies, or execute the cached validation checks.
Fresh local reruns are developer-maintenance tasks for refreshing the
release bundles.

## Repository Footprint Audit

Before release closeout, run the read-only footprint audit:

```bash
python scripts/audit_repository_footprint.py --top 20 --min-size-mib 1
```

The audit reports tracked large files from the current working tree, current
`HEAD` blob sizes, top reachable-history blobs across all refs, untracked files
that are not excluded by gitignore, and `.git/objects/pack` size. It only runs
read-only `git` queries and filesystem stats; it does not run garbage
collection, `git filter-repo`, or any other history-rewriting command.

For automation or an external release record, emit JSON and redirect it outside
the checkout:

```bash
python scripts/audit_repository_footprint.py --format json --top 20 \
  --min-size-mib 1 > /tmp/drbx_repository_footprint.json
```

## Build The Package

Build the source distribution and wheel locally:

```bash
python -m pip install build
python -m build
```

Expected outputs:

- `dist/drbx-<version>.tar.gz`
- `dist/drbx-<version>-py3-none-any.whl`

Validate the built metadata:

```bash
python -m pip install twine
python -m twine check dist/*
```

## GitHub Workflows

The repository includes:

- [`publish-pypi.yml`](../.github/workflows/publish-pypi.yml) for package publishing
- [`test.yml`](../.github/workflows/test.yml) for the Python 3.10, 3.11, and 3.12 test matrix
- [`docs.yml`](../.github/workflows/docs.yml) for `tests/test_release_surface.py`
  and `mkdocs build --strict --clean`
- [`coverage.yml`](../.github/workflows/coverage.yml) for public-surface
  coverage

The PyPI publish workflow:

1. builds the wheel and sdist on GitHub Actions,
2. stores them as workflow artifacts,
3. publishes them to PyPI through OIDC with `id-token: write`,
4. uses the `pypi` GitHub environment for the publish job.

Publishing is triggered by manual `workflow_dispatch` or by publishing a
GitHub release whose tag starts with `v`. It is intentionally not triggered
directly by tag pushes, so creating a version tag and then publishing its
GitHub release cannot submit the same distribution to PyPI twice. Artifact-only
releases, such as validation media refreshes, should use non-version tags and
are ignored by the PyPI jobs.

## Coverage And Validation Lanes

The release-readiness lanes are intentionally split:

- `test.yml` runs the targeted shipping regression slice on Python 3.10, 3.11,
  and 3.12.
- `docs.yml` checks the public release surface and builds the docs strictly.
- `coverage.yml` enforces the public-surface coverage gate.

## Release Checklist

Before publishing a version:

1. run the whole-package coverage gate (the same job enforced by
   `coverage.yml`):

```bash
pytest -q -m "not slow" --cov=drbx --cov-branch
coverage report
```

2. run the fast bounded validation slice:

```bash
python scripts/run_fast_research_checks.py
```

3. check the repository footprint before creating release artifacts:

```bash
python scripts/audit_repository_footprint.py --top 20 --min-size-mib 1
```

4. build the distributions locally:

```bash
python -m build
```

6. verify the public docs and artifact surface:

```bash
mkdocs build --strict
pytest -q tests/test_release_surface.py
```

7. verify the release artifact bundle and docs-media restore path when release
   assets have changed:

```bash
python scripts/fetch_example_artifacts.py
pytest -q tests/test_runtime_artifacts.py
```

Use a version tag such as `v1.0.3` only for package releases. Use an artifact
tag such as `validation-artifacts-YYYY-MM-DD` for docs-media or baseline
refreshes so the publish workflow remains skipped.

8. dispatch the bounded research campaign workflows that are expected for the
   release candidate, then wait for GitHub `test`, `docs`, and `coverage` to
   complete successfully on the target commit.

9. optionally run the Python version matrix locally or through CI.

## Current Release Boundary

The current package release is `2.0.0`. The earlier `v1.0.2`
tag must not be moved; publish the next package release as `v1.0.3` when the
remaining local and hosted gates are accepted.

The current package release is intended to support:

- standalone CLI and Python-driver workflows,
- promoted native-exact and native-operational validation lanes,
- reduced but real 3D tokamak, traced-field-line, and stellarator workflows,
- artifact-driven runtime, convergence, and profiling reports,
- software citation by citing the repository directly (version metadata
  lives in `pyproject.toml`).

Latest local closeout evidence:

- whole-package coverage: `95.16%`, `804` passed, `14`
  skipped, `10` deselected, and `1` expected xfail;
- fast bounded research checks: all default slices passed locally;
- docs build: `mkdocs build --strict --clean` passed locally;
- docs-media artifact restore: `174/174` media files restored from
  the private release bundle using `scripts/fetch_example_artifacts.py
  --force` with an isolated root and cache;
- self-contained example slice: `11` docs/example subprocess tests passed;
- representative user examples: diverted tokamak movie/profile, model
  selection guide, stellarator geometry, VMEC-extender import, and compact
  nonlinear stellarator movie commands passed locally;
- footprint/package audit: `.git` about `27M`, reachable pack about
  `6.43 MiB`, largest tracked file below `328 KiB`, wheel about `709 KiB`,
  and sdist about `614 KiB`.

It is not the full closure of every research workflow in the broader validation matrix. The detailed status is tracked in the project planning notes.

## After The First Package Release

The main post-release technical targets are:

- broader production temperature workflows,
- broader production 3D workflows beyond the reduced native matrix.
