# Release And Packaging

`jax_drb` is now set up to ship as a Python package and to publish through
GitHub Actions using PyPI Trusted Publishing.

This page is the practical release surface:

- how to install the package locally and from PyPI;
- how to build and verify the wheel/sdist;
- how the PyPI publishing workflow is wired;
- what still belongs to future research rather than the first public package.

## Install Paths

Editable development install:

```bash
pip install -e .[dev,integrators,models,validation]
```

Minimal runtime install from a checkout:

```bash
pip install -e .
```

Expected PyPI install after release:

```bash
pip install jax-drb
```

Validation and plotting extras from a checkout:

```bash
pip install -e .[validation]
```

## Build The Package

Build the source distribution and wheel locally:

```bash
python -m pip install build
python -m build
```

The expected outputs are:

- `dist/jax_drb-<version>.tar.gz`
- `dist/jax_drb-<version>-py3-none-any.whl`

Validate the built metadata:

```bash
python -m pip install twine
python -m twine check dist/*
```

## PyPI Publishing Workflow

The repository now includes
[`publish-pypi.yml`](../.github/workflows/publish-pypi.yml).

It follows the current PyPI Trusted Publishing model:

1. build the wheel and sdist on GitHub Actions;
2. store them as workflow artifacts;
3. publish them to PyPI through OIDC with `id-token: write`;
4. require the `pypi` GitHub environment on the publish job.

The workflow triggers on:

- GitHub release publication;
- manual `workflow_dispatch`.

To activate this on PyPI, configure the project’s Trusted Publisher to trust:

- repository: `uwplasma/jax_drb`
- workflow: `publish-pypi.yml`
- environment: `pypi`

## Release Checklist

Before publishing a version:

1. run the bounded release-closeout coverage gate:

```bash
python scripts/run_closeout_coverage.py
```

2. run the fast research-grade gate:

```bash
python scripts/run_fast_research_checks.py
```

3. build the distributions locally:

```bash
python -m build
```

4. verify that the public docs/artifact surface is still clean:

```bash
pytest -q tests/test_release_surface.py
```

5. confirm the selected-lane claim boundary in
[`jcp_readiness_audit.md`](jcp_readiness_audit.md).

## Claim Boundary For The First Public Package

The first package release is meant to support:

- standalone native runs through the CLI and Python API;
- promoted native-exact and native-operational validation lanes;
- reduced but real 3D tokamak, traced-field-line, and stellarator workflows;
- artifact-driven validation, parity, runtime, and profiling reports.

It is **not** meant to claim that every research workflow is already parity
complete across the full intended DRB reference-workflow matrix. That broader claim remains
tracked explicitly in the readiness audit and the plan.

## Why This Release Surface Is Reasonable

The release shape is aligned with the current reproducibility and validation bar
in the surrounding literature:

- the Hermes-3 multi-component plasma paper emphasizes flexible multi-species
  edge/SOL workflows and benchmark-driven validation rather than one monolithic
  “everything is done” claim;
- the TCV-X21 validation papers emphasize explicit benchmark contracts and
  observable families;
- the 2023 reproducibility statement shared by JCP/JSC/SISC asks for public
  code and data that reproduce the results of the paper, which maps well onto
  the committed artifact bundles and validation packages already in-tree.

Useful external references:

- [Hermes-3 multi-component plasma paper](https://arxiv.org/abs/2303.12131)
- [Validation of Hermes-3 turbulence simulations against the TCV-X21 diverted L-mode reference case](https://arxiv.org/abs/2506.12180)
- [Enhancing reproducibility of research papers in SISC, JSC and JCP](https://www.sciencedirect.com/journal/journal-of-computational-physics/about/announcements/enhancing-reproducibility-of-research-papers-in-sisc-jsc-and-jcp)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)

## Next Steps After The First Package Release

The main post-release technical targets are:

- broader production temperature/detachment workflows beyond the reduced
  controller gates;
- longer-window direct tokamak recycling closure beyond the currently bounded
  windows;
- broader production 3D workflows beyond the current reduced/native-selected
  matrix;
- manuscript drafting on the selected-lane JCP claim boundary.
