# Conservative nonlinear DRB (preparation)

This page describes ongoing preparation steps to evolve `jaxdrb` toward **nonlinear**, **conservative**
drift-reduced Braginskii (DRB) simulations.

The key motivation is that standard drift-reduced formulations commonly used in SOL turbulence codes can
lose exact conservation properties at the order at which polarisation effects enter, unless the implicit
polarisation relation is handled carefully.

## Primary reference

- B. De Lucca et al., *Conservative formulation of the drift-reduced fluid plasma model* (2026),
  arXiv: [`2601.05704`](https://arxiv.org/abs/2601.05704).

The paper constructs a conservative formulation by **analytically inverting** the implicit relation
that defines the polarisation velocity in terms of the time derivative of the electric field, and shows
exact conservation laws (energy, mass, charge, momentum) in arbitrary magnetic geometry (including EM).

## What exists today in `jaxdrb`

`jaxdrb` currently includes:

- robust matrix-free linear solvers and geometry abstraction for linear stability analysis,
- nonlinear HW2D as a fast testbed for numerical kernels and end-to-end differentiability,
- MPSE/sheath boundary-condition infrastructure for open field lines in the **linear** model,
- optional neutral interactions (minimal milestone model).

Nonlinear conservative DRB is **not yet implemented**; the HW2D milestone and the conservative utilities
in `src/jaxdrb/nonlinear/conservative/` exist to make the transition more systematic.

## What “conservative nonlinear DRB” requires

### 1) A discrete energy functional and a budget diagnostic

For any candidate nonlinear DRB formulation, we need:

- a precise discrete energy functional $E(y)$ matching the continuous conservation statement,
- a budget tool to compute $\dot E$ from the discrete RHS term-by-term,
- tests that demonstrate conservation in the appropriate limits (periodic, no sources/sinks, etc.).

This is why HW2D includes an explicit budget diagnostic (`HW2DModel.energy_budget`) and why conservation
checks are centralized in `src/jaxdrb/nonlinear/conservative/checks.py`.

### 2) A conservation-respecting discretization (perpendicular + parallel)

Perpendicular (in $(x,y)$ / $(\psi,\\alpha)$):

- For Poisson brackets / $E\\times B$ advection, Arakawa-style conservative Jacobians are a strong default on
  structured grids, and are already used in HW2D.
- For higher-order methods, an energy/enstrophy-preserving discretization should be maintained (e.g. compatible
  finite-difference/SBP operators or a conservative DG formulation).

Parallel (along $l$):

- For open-field-line SOL physics, parallel boundary conditions and sheath closures make the system stiff.
- A stable, differentiable discretization benefits from SBP-like finite differences or DG with a stable numerical
  flux, together with an IMEX or semi-implicit time integrator.

### 3) Implicit / semi-implicit solves in JAX (stiffness)

Nonlinear SOL DRB typically has stiffness from:

- parallel conduction and resistivity,
- sheath losses and boundary closures,
- (eventual) electromagnetic coupling.

To remain fast and differentiable in JAX, we prefer:

- matrix-free Krylov solvers (`jax.scipy.sparse.linalg.cg` / custom GMRES) with JAX-friendly preconditioners,
- IMEX time integrators (explicit nonlinear advection + implicit stiff linear terms),
- careful control of allocations (scan-based stepping, static shapes, `vmap` for parameter scans).

## Next nonlinear milestone (proposed)

The lowest-risk path is:

1. **Nonlinear periodic slab DRB (no sheath)**: add nonlinear $E\\times B$ advection terms to the existing
   reduced equations and verify conservation and known turbulence diagnostics in periodic settings.
2. **Introduce open-field-line parallel direction** (3D flux-tube) with sheath BCs in a controlled way, keeping
   the perpendicular discretization conservative.
3. **Adopt the conservative polarisation formulation** (De Lucca et al. 2026) and verify energy conservation with
   dedicated tests.

These steps allow progressively stronger validation before adding the full complexity of SOL boundaries, sources,
and multi-physics closures.

