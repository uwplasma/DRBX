# jaxdrb

`jaxdrb` is a small, CPU-friendly JAX package for linear (flux-tube / field-line) stability analysis of
cold-ion, drift-reduced Braginskii-like edge/SOL models.

## Install (editable)

From this folder:

```bash
python -m pip install -e .
```

If you are offline (or have restricted network access), add:

```bash
python -m pip install -e . --no-build-isolation
```

For dev tools (pytest/ruff/black):

```bash
python -m pip install -e ".[dev]"
```

For docs tooling:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

## CLI

Example ky scan in a shear-slab geometry:

```bash
jaxdrb-scan --geom slab --ky-min 0.05 --ky-max 1.0 --nky 32 --out out_slab
```

Circular tokamak:

```bash
jaxdrb-scan --geom tokamak --ky-min 0.05 --ky-max 1.0 --nky 32 --out out_tok
```

Cyclone-like s-alpha:

```bash
jaxdrb-scan --geom salpha --q 1.4 --shat 0.796 --epsilon 0.18 --alpha 0.0 --ky-min 0.05 --ky-max 1.0 --nky 32 --out out_cyclone
```

Tabulated geometry:

```bash
jaxdrb-scan --geom tabulated --geom-file mygeom.npz --ky-min 0.05 --ky-max 1.0 --nky 32 --out out_tab
```

## Status

This is a v1 implementation aimed at fast iteration:
- Fourier representation in perpendicular directions: `exp(i kx psi + i ky alpha)`
- 1D grid along the field line (`l`)
- Boussinesq polarization closure: `Omega = -k_perp^2(l) * phi`
