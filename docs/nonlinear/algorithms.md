# Algorithms for nonlinear stepping

This page summarizes the algorithms used in the nonlinear milestone and the rationale for selecting them, with an eye toward extending the approach to the full nonlinear drift-reduced Braginskii (DRB) system.

## Poisson bracket

Two bracket options are provided:

1. **Arakawa bracket (finite difference)**:
   - conservative and antisymmetric in a discrete sense,
   - useful when working on grids that are not naturally spectral.

2. **Pseudo-spectral bracket (FFT)**:
   - uses FFT-based derivatives and computes products in real space,
   - uses a **2/3-rule dealiasing** mask to reduce aliasing errors.

## Dealiasing

When products are formed in real space (e.g. in the pseudo-spectral bracket), high-wavenumber content can alias back into resolved modes. The implementation uses a 2/3 truncation mask in Fourier space to reduce this effect.

## Polarization solve

For periodic domains, polarization is solved spectrally by inverting $-k_\perp^2$ mode-by-mode, with the $k=0$ mode fixed to enforce a gauge ($\hat{\phi}(0)=0$).

For non-periodic boundary-condition experiments, the HW2D milestone also includes a finite-difference Laplacian with a **matrix-free CG solve**. This keeps the code:

- end-to-end differentiable (CG is a JAX primitive),
- modular (the Poisson solve is the only elliptic step in the electrostatic closure),
- close to what will be required for nonlinear DRB with nontrivial geometry and boundaries.

## Time integration

Two time-stepping paths are supported:

- **Fixed-step Diffrax integration**:
  - uses `diffrax.diffeqsolve` with a constant step size controller,
  - preserves the fixed-step semantics used in conservative gates and regression tests,
  - keeps the implementation end-to-end differentiable.

- **Diffrax adaptive integration**:
  - provides a convenient, robust reference integrator,
  - useful for verification and for problems where adaptive stepping is important.

## Differentiability

The nonlinear milestone is implemented to remain compatible with JAX transformations:

- RHS functions are JAX-pure (no side effects, no Python data-dependent control flow in jitted regions).
- Fixed-step stepping uses Diffrax with constant step sizes (still differentiable and XLA-friendly).
- The non-periodic Poisson solve uses JAX's matrix-free CG, which is differentiable through the solver iterations.

This enables end-to-end differentiation of scalar objectives that depend on simulation outputs, e.g.
optimizing parameters of the drive/damping terms or (in future) geometry parameters.

## Toward nonlinear DRB

The full nonlinear drift-reduced Braginskii system adds:

- 3D geometry (at least 2D perpendicular + 1D parallel),
- open-field-line boundary conditions (sheath / MPSE),
- stiff parallel operators and strong anisotropy,
- additional closures (viscosity, conduction), sources/sinks, and optional electromagnetic coupling.

The likely next algorithmic step for nonlinear DRB is an **IMEX** or **operator-split** approach:

- explicit: $E\times B$ advection and curvature terms,
- implicit or semi-implicit: stiff parallel diffusion/closure operators.

Diffrax provides implicit and IMEX-capable solvers; this is a natural fit for the nonlinear transition plan.
