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

## Optional Coil Fields

Coil fields and field-line tracing use the installed ESSOS package:
`pip install 'drbx[essos]'`. The Landreman--Paul QA coil set is bundled.

The 3D renders in the field-line, HSX and FCI-plane examples use PyVista (offscreen) through
`drbx.viz`; install it with `pip install 'drbx[viz]'`. It is optional and not needed by the solvers.
Examples that need the matching VMEC `wout` read it from an ESSOS checkout
named by `DRBX_ESSOS_ROOT`; they are not required for the basic native
examples.
