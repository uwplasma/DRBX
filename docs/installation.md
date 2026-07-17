# Installation

`drbx` is packaged as `drbx` on PyPI and can also be installed from a
local checkout.

## PyPI

```bash
pip install drbx
```

This installs the runtime dependencies used by the public solver, geometry,
and plotting paths: `jax`, `diffrax`, `scipy`, `equinox`, `matplotlib`,
`netCDF4`, `rich`, `pillow`, and `tomli` on Python versions that do not provide
`tomllib`.

## Editable Checkout

```bash
git clone https://github.com/uwplasma/drbx
cd drbx
pip install -e .
```

For local documentation builds:

```bash
pip install -e .[docs]
python -m mkdocs build --clean
```

Read the Docs builds the same `mkdocs.yml` configuration through the root
`.readthedocs.yaml` file. The public site is
[drbx.readthedocs.io](https://drbx.readthedocs.io/).

## Verify The Install

Run a small deck:

```bash
drbx examples/inputs/restartable_diffusion.toml
```

Inspect the same deck without advancing the simulation:

```bash
drbx inspect examples/inputs/restartable_diffusion.toml
```

Build the lightweight docs locally:

```bash
python -m mkdocs build --clean --site-dir /tmp/drbx_docs
```

## Optional External Geometry Campaigns

Some validation campaigns import externally traced field-line data. Those
workflows use environment variables such as `DRBX_ESSOS_ROOT` and are
documented on their campaign pages. They are not required for installation or
for the basic native examples.
