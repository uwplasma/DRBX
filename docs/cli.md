# CLI reference

The package installs a console script:

```bash
jaxdrb-scan ...
```

and a 2D scan helper:

```bash
jaxdrb-scan2d ...
```

The CLI performs a local scan over `ky` (and a fixed `kx`) for a selected geometry, and computes:

- a leading eigenvalue estimate (matrix-free Arnoldi),
- a time-domain growth rate estimate (Diffrax initial-value).

It writes a results folder with machine-readable output and a plot.

## Common options

- `--model {cold-ion-es, hot-ion-es, em}` selects the physics model variant
- `--geom {slab, tokamak, salpha, tabulated}`
- `--kx FLOAT` (default 0.0)
- `--ky-min FLOAT`, `--ky-max FLOAT`, `--nky INT`
- `--nl INT` number of grid points along the field line
- `--out PATH` output directory

## Geometry options

These apply to analytic geometries (`slab`, `tokamak`, `salpha`):

- `--length FLOAT` domain length in `l` (default `2π`)
- `--shat FLOAT` magnetic shear parameter
- `--curvature0 FLOAT` curvature magnitude (for tokamak/s-alpha, if left at 0, defaults to `epsilon`)

Tokamak/s-alpha specific:

- `--q FLOAT`
- `--R0 FLOAT`
- `--epsilon FLOAT`
- `--alpha FLOAT` (s-alpha only)

Tabulated:

- `--geom-file PATH` path to the `.npz` file

## Physics options

- `--omega-n FLOAT`, `--omega-Te FLOAT` background-gradient drives
- `--omega-Ti FLOAT` ion temperature-gradient drive (hot-ion model)
- `--eta FLOAT` resistive coupling
- `--me-hat FLOAT` electron inertia knob
- `--beta FLOAT` inductive coupling strength (electromagnetic model)
- `--tau-i FLOAT` ion-to-electron temperature ratio (hot-ion model)
- `--no-curvature` disables curvature terms
- `--no-boussinesq` uses a linearized non-Boussinesq polarization closure (about equilibrium `n0(l)`)
- `--Dn`, `--DOmega`, `--DTe` perpendicular diffusion coefficients
- `--DTi` ion temperature diffusion (hot-ion model)
- `--Dpsi` psi diffusion (electromagnetic model)

## Solver options

Arnoldi:

- `--arnoldi-m INT` initial Krylov dimension
- `--arnoldi-max-m INT` cap (default: full dimension `5*nl`)
- `--arnoldi-tol FLOAT` target relative residual
- `--nev INT` number of Ritz values saved per `ky`

Initial-value:

- `--tmax FLOAT`, `--dt0 FLOAT`, `--nsave INT`
- `--no-initial-value` disables the initial-value solver (eigenvalues only)

## Outputs

`params.json` records the run configuration.

`results.npz` contains at least:

- `ky`: ky grid (float, shape `(nky,)`)
- `gamma_eigs`: leading eigenvalue real part (shape `(nky,)`)
- `omega_eigs`: leading eigenvalue imag part (shape `(nky,)`)
- `gamma_iv`: initial-value growth estimate (shape `(nky,)`)
- `eigs`: complex eigenvalues returned by Arnoldi (shape `(nky, nev)`)

The CLI also writes a set of diagnostic plots:

- `scan_panel.png`: $\gamma(k_y)$, $\omega(k_y)$, and $\max(\gamma,0)/k_y$
- `geometry_overview.png`: basic geometry coefficients along `l`
- `spectrum.png`: Ritz spectrum at the maximizing $k_{y,*}$
- `eigenfunctions.png`: eigenfunction amplitudes/real parts along `l` at $k_{y,*}$

## 2D scan (kx, ky)

`jaxdrb-scan2d` performs a 2D eigenvalue scan over a `kx×ky` grid and writes:

- `results_2d.npz`: `kx`, `ky`, `gamma_eigs(kx,ky)`, `omega_eigs(kx,ky)`
- `gamma_kxky.png`: heatmap of $\gamma(k_x,k_y)$
- `gamma_ky_max_over_kx.png`: $\max_{k_x}\gamma$ vs $k_y$
- `kx_star_vs_ky.png`: $k_x^*(k_y)$ that maximizes $\gamma$
