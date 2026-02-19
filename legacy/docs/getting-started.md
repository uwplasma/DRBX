# Getting started

## Install

From the `jaxdrb/` repository root:

```bash
python -m pip install -e .
```

If you are offline (or have restricted network access), add:

```bash
python -m pip install -e . --no-build-isolation
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

For documentation tooling:

```bash
python -m pip install -e ".[docs]"
```

## First run (CLI)

Run a simple scan in the analytic slab geometry:

```bash
jaxdrb-scan --geom slab --ky-min 0.05 --ky-max 1.0 --nky 32 --out out_slab
```

You should get:

- `out_slab/params.json` with the run configuration
- `out_slab/results.npz` containing `ky`, growth rates and eigenvalues
- `out_slab/gamma_ky.png` plotting growth rate vs `ky`

## Key concepts

### Field-line model (flux tube)

`jaxdrb` models perturbations along a single field line coordinate `l` (often a poloidal angle
`theta`), with Fourier structure in the perpendicular directions:

$$
\tilde{f}(\psi,\alpha,l,t) = \hat{f}(l,t)\,\exp\{ i k_x \psi + i k_y \alpha \}.
$$

This is the primary simplification enabling:

- matrix-free eigenvalue calculations (no global Poisson solves),
- fast parameter scans over `(kx, ky)` and geometry parameters.

### Geometry independence

The model core never asks for “VMEC”, “EFIT”, etc. It only needs three operations, each acting
on 1D arrays over `l`:

- $k_\perp^2(l)$,
- $\nabla_\parallel$,
- curvature drive $C(\cdot)$.

Everything else (how these are obtained) lives in the geometry provider.

## Next steps

- See `examples.md` for more complete scans (circular tokamak and s-alpha/Cyclone geometry).
- See `model/equations.md` for the implemented PDE system and closures.
- See `geometry/index.md` for supported geometries and the `.npz` tabulated format.

