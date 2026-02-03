# Inputs & Outputs

This page summarizes what goes into `jaxdrb` runs (inputs) and what comes out (outputs), for both
the CLI and the Python API.

## Inputs

### Geometry inputs

`jaxdrb` operates on a **field line** parameterized by a 1D coordinate `l` (periodic by default).

You can provide geometry in two ways:

- **Analytic geometry providers** (Python objects), e.g.:
  - `jaxdrb.geometry.SlabGeometry`
  - `jaxdrb.geometry.CircularTokamakGeometry`
  - `jaxdrb.geometry.SAlphaGeometry`
- **Tabulated geometry** from an `.npz` file:
  - `jaxdrb.geometry.TabulatedGeometry.from_npz("geom.npz")`

Tabulated geometry files use coefficient arrays along the `l` grid:

- Required keys:
  - `l`: 1D array of shape `(nl,)` (uniform grid; periodic endpoint excluded)
  - `gxx`, `gxy`, `gyy`: arrays of shape `(nl,)` defining
    $$k_\perp^2(l) = k_x^2 g^{xx}(l) + 2 k_x k_y g^{xy}(l) + k_y^2 g^{yy}(l)$$
- Optional keys:
  - `curv_x`, `curv_y`: curvature coefficients used in the operator
    $$\mathcal{C}(f) = i\,(k_x\,\mathrm{curv}_x + k_y\,\mathrm{curv}_y)\,f$$
  - `dpar_factor`: multiplicative factor for the parallel derivative (default 1), so
    $$\nabla_\parallel f \approx \mathrm{dpar\_factor}(l)\,\partial_l f$$
  - `B`: magnetic field magnitude along the line (used for plotting and future extensions)

ESSOS integration helpers that produce `.npz` files live in [`src/jaxdrb/geometry/essos.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/geometry/essos.py).

### Physics / model parameters

Most runs are controlled by a `jaxdrb.models.params.DRBParams` object. Common parameters include:

- `omega_n`: density-gradient drive (a proxy for background gradients)
- `omega_Te`: electron-temperature-gradient drive
- `eta`: collisionality/resistive parameter (model-dependent interpretation; see model docs)
- `me_hat`: electron inertia toggle/strength for inertial branches
- `curvature_on`: enable/disable curvature drive
- `Dn`, `DOmega`, `DTe`, `Dpsi`, `DTi`: small diffusion terms for stabilization (model-dependent)
- `chi_par_Te`: optional parallel electron heat conduction coefficient (closure)
- `nu_par_e`, `nu_par_i`: optional parallel flow diffusion/viscosity coefficients (closures)
- `nu_sink_n`, `nu_sink_Te`, `nu_sink_vpar`: optional volumetric sinks (proxies for sources/sinks)
- `beta`: electromagnetic beta parameter (electromagnetic model)
- `tau_i`: ion-temperature ratio (hot-ion model)
- `boussinesq`: polarization closure toggle (Boussinesq vs linearized non-Boussinesq)

### Wavenumbers and grids

`jaxdrb` uses a ballooning/flux-tube representation:

$$\delta f \propto \exp(i k_x\,\psi + i k_y\,\alpha).$$

Inputs:

- `kx`: scalar (or a grid for 2D scans)
- `ky`: scalar or a 1D grid for scans
- `nl`: number of grid points along `l` (for analytic geometries) or inferred from a tabulated file

### Solver inputs

#### Matrix-free eigen solver (Arnoldi)

Key knobs (used by the CLI and by `jaxdrb.analysis.scan.scan_ky`):

- `arnoldi_m`: Arnoldi Krylov dimension per cycle
- `arnoldi_tol`: relative residual tolerance (used to adapt `m` upward if needed)
- `nev`: number of Ritz values retained for reporting/plots
- `seed`: RNG seed for the initial Krylov vector

#### Initial-value solver (Diffrax)

Key knobs (used by `jaxdrb.linear.growthrate.estimate_growth_rate`):

- `tmax`: final time
- `dt0`: initial step size (adaptive integrator)
- `nsave`: number of saved time points (affects diagnostics + fits)

## Outputs

### Scan results (Python)

`jaxdrb.analysis.scan.scan_ky(...)` returns a `Scan1DResult` with:

- `ky`: ky grid
- `gamma_eigs`, `omega_eigs`: leading eigenvalue real/imag parts
- `eigs`: small set of Ritz eigenvalues (complex)
- `arnoldi_m_used`, `arnoldi_rel_resid`: Arnoldi diagnostics
- optional initial-value estimates `gamma_iv`, `omega_iv`

### CLI output folder layout

The CLI writes an output folder (e.g. `--out out_slab/`) containing:

- `results.npz`: arrays for `ky`, `gamma`, `omega`, eigenvalues, and metadata
- `params.json`: a complete record of the inputs used for the run
- one or more `.png` figures (e.g. `gamma_vs_ky.png`)

Examples write a similar structure under `out/.../`, typically adding more diagnostic plots:

- growth-rate spectra (gamma vs ky),
- eigenvalue spectra in the complex plane,
- geometry overview plots (metric and curvature coefficients),
- eigenfunction panels (field structure along `l`).

### SOL-width workflow outputs

The SOL width proxy used in Halpern-style “gradient removal” workflows is:

$$\left(\frac{\gamma}{k_y}\right)_{\max} = \max_{k_y>0}\,\frac{\max(\gamma(k_y),0)}{k_y}.$$

The fixed-point estimate returns:

- `Lp`: converged width proxy
- `ky_star`: argmax location in ky
- `gamma_over_ky_star`: maximized ratio value
- a `history` array for plotting convergence

See [`src/jaxdrb/analysis/lp.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/analysis/lp.py).

## Environment variables

- `JAXDRB_FAST=1`: some advanced examples use reduced resolution for quick runs.
- `JAXDRB_FAST=0`: run those examples at higher resolution (slower).

## Reproducibility tips

- Prefer saving a `params.json` next to any `results.npz` file (many examples do this).
- If you are reporting the SOL-width proxy, use the normalized quantity
  $\max(\gamma,0)/(k_y\,c_s)$ or its dimensionless counterpart consistently; see
  `docs/literature/sol_width.md` for the workflow assumptions.
