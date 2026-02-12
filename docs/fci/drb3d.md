# DRB3D with FCI: equations, discretization, and parameters

This page documents the current **3D FCI (“flux-coordinate independent”) DRB3D milestone** in
`jaxdrb`:

- the *model state* and what each field means,
- the *FCI parallel operator* (including **target-aware** stencils),
- the *plate/sheath closure coupling* used by the current benchmark gates,
- the *perpendicular operators* used on each plane,
- a parameter reference (with links to the authoritative code definitions).

For map data structures and file formats, see:

- [`docs/fci/maps.md`](maps.md)

For requirements and benchmark gates targeting a fully nonlinear 3D edge/SOL solver, see:

- [`docs/fci/requirements.md`](requirements.md)
- [`docs/validation.md`](../validation.md)

---

## Coordinate picture (toroidal planes + field-line mapping)

The FCI discretization used here represents the 3D domain as a **stack of structured planes**
labelled by an angle-like coordinate (typically toroidal angle) $\phi_k$:

$$
\phi_k = \phi_0 + k\,\Delta\phi,\qquad k=0,\ldots,N_\phi-1.
$$

Each plane is a structured grid in **in-plane coordinates** (commonly cylindrical $(R,Z)$ for
ESSOS/VMEC use cases, or Cartesian $(x,y)$ for analytic/MMS cases).

The **parallel derivative** is not taken by differentiating within the plane. Instead, we follow
field lines to map values from plane $k\pm 1$ back to plane $k$ grid points, then form finite
differences using the mapped values and the field-line distance $\Delta l$ carried by the map.

This is the “flux-coordinate independent” idea: the parallel operator follows the magnetic field
without requiring a flux-aligned mesh.

---

## DRB3D milestone state (what is evolved)

The full drift-reduced Braginskii set is implemented in 1D field-line and 2D branches; for the FCI
3D milestone the code currently focuses on:

1) **conservative/budget identities** and
2) **target/sheath coupling** with target-aware parallel operators.

The canonical DRB state (field-line solver) is:

$$
y = (n,\Omega,v_{\parallel e},v_{\parallel i},T_e),
$$

with electrostatic potential $\phi$ recovered via a polarization closure (Boussinesq in the current
FCI milestones unless otherwise noted).

Authoritative state containers:

- Linear/field-line:  
  [`src/jaxdrb/models/cold_ion_drb.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/cold_ion_drb.py)
- Nonlinear 2D:  
  [`src/jaxdrb/nonlinear/drb2d.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/drb2d.py)
- FCI operators and milestones:  
  [`src/jaxdrb/fci/`](https://github.com/uwplasma/jax_drb/tree/main/src/jaxdrb/fci)

---

## Parallel operator: FCI target-aware $\nabla_\parallel$

### A) Mapped values by bilinear interpolation

Let $f_k$ be a scalar field on plane $k$ with values $f_k(i,j)$ on the structured grid. The forward
FCI map provides a **footpoint** $(x^+,y^+)$ on plane $k+1$ for each grid point $(x,y)$ on plane $k$,
plus a bilinear stencil:

$$
f_{k+1}(x^+,y^+) \;\approx\; \sum_{m=1}^4 w_m\, f_{k+1}(i_m, j_m).
$$

The same holds backward to plane $k-1$.

Implementation (runtime map + interpolation):

- [`src/jaxdrb/fci/map.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/map.py)
  (`FCIBilinearMap`, `apply_bilinear_map`)

### B) Central difference with nonuniform $\Delta l$

The map also carries a distance-to-plane along the field line for each grid point:

- forward: $\Delta l^{+}$ (plane $k \to k+1$)
- backward: $\Delta l^{-}$ (plane $k \to k-1$)

Using the mapped values $f_{k+1\to k}$ and $f_{k-1\to k}$, a second-order central approximation is:

$$
\nabla_\parallel f\big|_k \;\approx\;
\frac{f_{k+1\to k} - f_{k-1\to k}}{\Delta l^{+} + \Delta l^{-}}.
$$

### C) Target-aware treatment (open field lines)

For open field lines, a trajectory may hit a target *before* reaching the next/previous plane. The
map can encode:

- a boolean `hit` mask, and
- a distance-to-hit $\Delta l_{\rm hit}$ (plus intersection metadata).

Near hits, the operator switches from a symmetric stencil to a **one-sided, nonuniform** stencil
that is compatible with plate/sheath boundary conditions:

- if the **forward** segment hits a target, use a one-sided stencil anchored at the target point,
- similarly for a backward hit.

Implementation:

- [`src/jaxdrb/fci/parallel.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/parallel.py)
  (`parallel_derivative_target_aware_3d`)

The scheme is designed to be:

- JAX-friendly (pure array operations),
- differentiable end-to-end (maps and fields enter algebraically),
- regression-testable (MMS convergence gates).

---

## Plate/sheath closure coupling (budget-aware)

For edge/SOL physics, the parallel operator must be coupled to plate boundary conditions. The
repository implements and tests **Bohm/Loizu-like** entrance conditions in 1D field-line workflows,
and reuses the same building blocks in the FCI milestone gates.

Key components:

- Sheath/MPSE closure functions:  
  [`src/jaxdrb/models/sheath.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/models/sheath.py)
- Boundary-condition “relaxation” (Dirichlet/Neumann/periodic) helpers used across branches:  
  [`src/jaxdrb/nonlinear/bcs.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/bcs.py)

### Budget channels

The CI gate for the current “full multiphysics DRB3D milestone” checks:

- explicit particle-rate channels (parallel vs sheath/plate sinks),
- explicit energy-rate channels (sheath heat transmission / losses),
- split-operator reconstruction residuals (when operator splitting is enabled).

Gate implementation:

- [`benchmarks/check_fci_drb3d_full_multiphysics_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/benchmarks/check_fci_drb3d_full_multiphysics_gate.py)

---

## Perpendicular operators on each plane

On each plane, `jaxdrb` uses structured-grid finite-difference/finite-volume building blocks:

- Poisson bracket (Arakawa): energy/enstrophy-friendly advection in 2D branches.
- Laplacians / hyper-Laplacians for diffusion/hyperdiffusion.
- Polarization closure:
  - Boussinesq: $\Omega = \nabla_\perp^2 \phi$ (spectral solve available in periodic boxes),
  - non-Boussinesq milestone (2D): $-\nabla_\perp\cdot(n\nabla_\perp\phi)=\Omega$ (SPD solve).

Operator kernels live in:

- [`src/jaxdrb/nonlinear/`](https://github.com/uwplasma/jax_drb/tree/main/src/jaxdrb/nonlinear)

---

## Time integration (Diffrax)

All nonlinear branches use Diffrax for time integration, enabling:

- fixed-step runs (fast and reproducible for gates),
- adaptive runs (robustness for stiff or strongly driven regimes),
- solver comparisons as regression checks.

Entry point patterns:

- `DRB2DModel.diffeqsolve(...)` in  
  [`src/jaxdrb/nonlinear/drb2d.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/nonlinear/drb2d.py)

---

## Parameter reference (authoritative code links)

FCI/DRB3D involves **three parameter layers**:

1) **Map/build parameters** (how maps are constructed)  
   see `docs/fci/maps.md` and:
   - [`src/jaxdrb/fci/builder.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/builder.py)
2) **Operator parameters** (how operators use maps, BCs, and closures)
3) **Model parameters** (physical/normalized coefficients and toggles)

The most important operator-layer definitions to consult are:

- Map object + interpolation:  
  [`src/jaxdrb/fci/map.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/map.py)
- Parallel operators (target-aware stencils):  
  [`src/jaxdrb/fci/parallel.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/parallel.py)
- FCI IO (npz format):  
  [`src/jaxdrb/fci/io.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/fci/io.py)

For the multiphysics milestone gate thresholds and what is checked:

- [`benchmarks/check_fci_drb3d_full_multiphysics_gate.py`](https://github.com/uwplasma/jax_drb/blob/main/benchmarks/check_fci_drb3d_full_multiphysics_gate.py)

---

## What is “3D” right now, and what remains

The repository now contains **true 3D geometry/mapping infrastructure** (toroidal-plane FCI map
builders, target intersection metadata, target-aware parallel stencils) and **budget-aware benchmark
gates** for short multiphysics runs.

The main remaining steps to become a production 3D edge/SOL turbulence code are:

- longer-time 3D turbulence regression gates (statistics, spectra, profile diagnostics),
- robust non-Boussinesq polarization in 3D with strong preconditioning,
- device-grade geometry pipelines (diverted tokamak / island divertor / stellarators) with robust
  target detection and wall BCs.

These items are tracked in:

- [`docs/fci/requirements.md`](requirements.md)
- [`docs/roadmap.md`](../roadmap.md)

