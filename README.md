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

2D eigenvalue scan over `(kx, ky)`:

```bash
jaxdrb-scan2d --geom salpha --kx-min -1.0 --kx-max 1.0 --nkx 33 --ky-min 0.1 --ky-max 1.0 --nky 24 --out out_kxky
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

## Examples (including literature workflows)

The `examples/` tree is organized by complexity:

- `examples/1_simple/`: quick “hello world” ky scans + diagnostics
- `examples/2_intermediate/`: tabulated-geometry round-trip, kx–ky scans, JAX autodiff workflow
- `examples/3_advanced/`: literature-inspired workflows + stellarator (pyQSC) geometry

Quick smoke test (runs the simple examples):

```bash
make examples
```

Run everything except the pyQSC stellarator case:

```bash
make examples-all
```

Near-axis stellarator example (requires a pyQSC checkout next to this repo):

```bash
make examples-stellarator
```

## Docs

Build locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```
