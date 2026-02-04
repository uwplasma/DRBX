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
- `--geom {slab, slab-open, tokamak, tokamak-open, salpha, salpha-open, tabulated}`
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
- `--chi-par-Te FLOAT` parallel electron heat conduction coefficient
- `--nu-par-e FLOAT`, `--nu-par-i FLOAT` parallel flow diffusion/viscosity coefficients
- `--nu-sink-n FLOAT`, `--nu-sink-Te FLOAT`, `--nu-sink-vpar FLOAT` simple volumetric sinks
- `--line-bc {none,dirichlet,neumann}` applies a user-defined BC along `l` to all fields (benchmarking/nonlinear-prep)
- `--line-bc-value FLOAT`, `--line-bc-grad FLOAT` set Dirichlet value or Neumann gradient for `--line-bc`
- `--line-bc-nu FLOAT` sets the RHS relaxation rate (0 disables BC enforcement)
- `--sheath` enables Loizu-style MPSE Bohm sheath BCs (alias for `--sheath-bc`)
- `--sheath-bc` enables Loizu-style magnetic-pre-sheath entrance BCs (only active for `*-open` geometries)
- `--no-sheath-bc` disables MPSE Bohm sheath BCs for `*-open` geometries (Bohm is the default for open-field-line runs)
- `--sheath-bc-model {simple,loizu2012}` selects the MPSE enforcement model
- `--sheath-bc-nu-factor FLOAT` multiplies the BC enforcement rate (~`2/L_parallel`)
- `--sheath-cos2 FLOAT` sets a proxy for $\cos^2(a)$ used in the Loizu 2012 vorticity relation
- `--sheath-lambda FLOAT` sets $\Lambda = 0.5\ln(m_i/(2\pi m_e))$ (default ~3.28)
- `--sheath-delta FLOAT` ion transmission correction (cold ions → 0)
- `--sheath-loss` enables a volumetric end-loss proxy (not a substitute for MPSE BCs)
- `--sheath-loss-nu-factor FLOAT` multiplies the loss rate `nu_sh ~ 2/L_parallel`
- `--sheath-end-damp`, `--no-sheath-end-damp` toggles an additional boundary-localized damping term at the sheath nodes (robust default)
- `--sheath-heat` enables a lightweight sheath heat transmission / energy-loss closure at the MPSE nodes
- `--sheath-gamma-auto`, `--no-sheath-gamma-auto` toggles $\gamma_e \approx 2 + \Lambda_{\mathrm{eff}}$ vs a manual $\gamma_e$
- `--sheath-gamma-e FLOAT` manual electron heat transmission factor (used with `--no-sheath-gamma-auto`)
- `--sheath-gamma-i FLOAT` ion heat transmission factor (hot-ion model; default 3.5)
- `--sheath-see` enables a simple secondary electron emission (SEE) correction
- `--sheath-see-yield FLOAT` sets a constant SEE yield $\delta$ (used to form $\Lambda_{\mathrm{eff}} = \Lambda + \ln(1-\delta)$)

## Solver options

Arnoldi:

- `--arnoldi-m INT` initial Krylov dimension
- `--arnoldi-max-m INT` cap (default: full dimension `N` of the state)
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

## Nonlinear HW2D

`jaxdrb-hw2d` runs a 2D periodic nonlinear HW-like drift-wave testbed (see `docs/nonlinear/`):

```bash
jaxdrb-hw2d --nx 96 --ny 96 --tmax 40 --dt 0.05 --out out_hw2d_cli
```

With neutrals enabled:

```bash
jaxdrb-hw2d --neutrals --nu-ion 0.2 --nu-rec 0.02 --out out_hw2d_neutrals_cli
```

Outputs include `params.json`, `timeseries.npz`, and snapshot plots (`n.png`, `phi.png`, `omega.png`, and `N.png` if enabled).

Boundary condition experiments (non-periodic, FD+CG path):

```bash
jaxdrb-hw2d --poisson cg_fd --bracket centered --bc-x dirichlet --bc-y dirichlet --bc-enforce-nu 10.0 --out out_hw2d_bc
```

HW2D-specific options include:

- `--bracket {spectral,arakawa,centered}`
- `--poisson {spectral,cg_fd}`
- `--bc-x`, `--bc-y` in `{periodic,dirichlet,neumann}`
- `--bc-enforce-nu FLOAT` (boundary relaxation rate; useful for non-periodic experiments)
