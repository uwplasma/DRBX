# DRB2D model (2D drift-reduced Braginskii testbed)

This page documents the **DRB2D** nonlinear testbed used throughout `jaxdrb` to
validate conservative advection, polarization closures, and turbulence diagnostics
in a fast 2D setting. It is a **perpendicular (x–y)** reduction of the drift-reduced
Braginskii system with a simplified parallel closure and curvature drive.

The DRB2D testbed is not a full SOL/tokamak simulation; it is a **research-grade,
end‑to‑end differentiable** numerical kernel that reproduces key limits used in
SOL turbulence codes (GBS/Hermes-style slab proxies) and in drift‑wave testbeds.

References in the local literature cache:

- `conserving_drb.pdf` (energy-consistent DRB formulation and invariants)
- `camargo_biskamp_scott95.pdf` (2D drift‑wave / HW‑like turbulence)
- `Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf` and `EPFL_TH6197.pdf`
  (GBS SOL slab modeling)
- `hermes-2/hermes_paper.pdf` (Hermes edge/SOL model and blob2d example)

## Fields and normalization

The DRB2D state is

$$
Y = (n,\ \Omega,\ v_{\parallel e},\ v_{\parallel i},\ T_e),
$$

with electrostatic potential obtained from a polarization closure.
The model is expressed in normalized units consistent with DRB slab models
(see `conserving_drb.pdf` and `Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf`).

The 2D domain is a uniform grid in $(x,y)$:

- $x$: radial-like direction,
- $y$: poloidal/binormal-like direction,
- periodic or non-periodic boundary conditions (Dirichlet/Neumann) are supported.

## Core equations (Boussinesq form)

The Boussinesq DRB2D system implemented in `jaxdrb` uses a standard perpendicular
advection + curvature form with a simplified parallel coupling (modeled by a
constant $k_\parallel$). In compact form:

$$
\partial_t n + \{\phi, n\} = -\nabla_\parallel v_{\parallel e} + C(p) - C(\phi) + \cdots
$$

$$
\partial_t \Omega + \{\phi, \Omega\} = \nabla_\parallel j_\parallel + C(p) + \cdots
$$

$$
\partial_t v_{\parallel e} + \{\phi, v_{\parallel e}\}
 = \frac{1}{\hat m_e}\nabla_\parallel(\phi - n - \alpha_{T_e} T_e) + \cdots
$$

$$
\partial_t v_{\parallel i} + \{\phi, v_{\parallel i}\}
 = -\nabla_\parallel \phi + \cdots
$$

$$
\partial_t T_e + \{\phi, T_e\} = -\frac{2}{3}\nabla_\parallel v_{\parallel e}
 + \frac{2}{3} C\!\left(\frac{7}{2}T_e + n - \phi\right) + \cdots
$$

where:

- $\{\phi, f\} = \partial_x \phi\,\partial_y f - \partial_y \phi\,\partial_x f$
  is the 2D Poisson bracket (implemented with the conservative Arakawa stencil by default),
- $j_\parallel = v_{\parallel i} - v_{\parallel e}$,
- $p = n + T_e$ in normalized units,
- $C(\cdot)$ is the slab curvature operator.

The dots indicate optional drive/dissipation terms (diffusion, linear damping,
SOL losses, and sources). The energy-consistent forms and invariant diagnostics
are aligned with `conserving_drb.pdf`.

## Polarization closure

The electrostatic potential is obtained from:

### Boussinesq:

$$
\Omega = -\nabla_\perp^2 \phi,
$$

solved spectrally on periodic grids or via a matrix‑free CG solve on non‑periodic
grids (see `docs/nonlinear/algorithms.md`).

### Non‑Boussinesq:

$$
 -\nabla_\perp \cdot (n\,\nabla_\perp \phi) = \Omega,
$$

with $n$ floored for SPD stability and solved by CG with a spectral/Jacobi
preconditioner. This is documented in the non‑Boussinesq gate and is aligned with
the conservative DRB formulation (`conserving_drb.pdf`).

## Curvature and gradient drives

The slab curvature operator is

$$
C(f) = -\omega_c\,\partial_y f,
$$

consistent with standard slab/SOL models (GBS, Hermes; see `Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf`
and `hermes-2/hermes_paper.pdf`). The sign of $\omega_c$ controls whether
interchange transport drives outward (positive $x$) or inward motion.

Optional background-gradient drives use a Fourier‑$k_y$ proxy:

$$
\partial_t n \leftarrow -\omega_n\,\partial_y \phi,\quad
\partial_t T_e \leftarrow -\omega_{T_e}\,\partial_y \phi.
$$

## Dissipation, damping, and SOL closures

The DRB2D testbed includes:

- Laplacian diffusion ($D_n$, $D_\Omega$, $D_{T_e}$),
- Biharmonic diffusion ($D_{n4}$, $D_{\Omega4}$, $D_{T_e4}$),
- Linear damping $-\mu\,f$ (used as a simple proxy for parallel losses),
- Optional SOL closed→open masks with relaxation/sink terms (GBS‑style proxy).

These knobs are explicitly documented in the validation gates and are used to keep
short runs stable and reproducible while preserving conservative nonlinear dynamics.

## Boundary conditions

`jaxdrb` supports:

- **Periodic** BCs in $x,y$ for spectral inversions and conservative tests.
- **Neumann/Dirichlet** BCs in $x$ for open‑boundary SOL proxies.

For non‑periodic Poisson solves, a small gauge‑lifting term removes the nullspace;
this is controlled by `poisson_gauge_epsilon`. When the geometry is *Neumann in x*
and *periodic in y*, the solver can use a fast mixed FFT (DCT‑I in $x$ + FFT in $y$)
that matches the discrete FD Laplacian spectrum and is substantially faster than
iterative CG while remaining differentiable.

## Numerics

Key numerical choices:

- **Poisson bracket**: Arakawa (default), centered, or spectral.
- **Poisson/polarization** (periodic): spectral.
- **Poisson/polarization** (Neumann $x$ + periodic $y$): mixed FFT.
- **Poisson/polarization** (general non‑periodic): matrix‑free CG.
- **Time integration**: Diffrax fixed‑step for reproducibility and differentiability.

See `docs/nonlinear/algorithms.md` for implementation details.

## Benchmarks and examples

The DRB2D testbed is anchored to a set of reproducible benchmarks:

- **Kelvin–Helmholtz benchmark** (`drb2d_kelvin_helmholtz.py`):
  conservative advection + Poisson inversion validation (see KH references in `docs/references.md`).
- **Hermes‑2 blob2d proxy** (`drb2d_hermes2_blob2d.py`):
  Gaussian blob + curvature drive, tuned to Hermes‑2 `blob2d` parameters
  (`hermes-2/hermes_paper.pdf`). A weak stochastic vorticity forcing is available
  to keep short runs nonlinear and visually active on reduced grids. The README
  movie uses Neumann $x$ + periodic $y$ with the mixed‑FFT Poisson solver, and an
  optional initial $\phi$ dipole to strengthen radial propagation on short runs.
- **GBS SOL proxy** (`drb2d_sol_movie.py`):
  closed→open LCFS masks and Bohm‑sheath closures (`Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf`,
  `EPFL_TH6197.pdf`).

Each benchmark has a corresponding gate in `benchmarks/` and documentation in
`docs/validation.md`.

## References (local)

- `conserving_drb.pdf`
- `camargo_biskamp_scott95.pdf`
- `Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf`
- `EPFL_TH6197.pdf`
- `hermes-2/hermes_paper.pdf`
