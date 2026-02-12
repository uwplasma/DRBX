# Limitations and interpretation

`jaxdrb` is a research code, but it is **not** a “toy” or purely qualitative model: the repository
includes **quantitative verification and regression gates** (MMS, conservation/budget closure checks,
and literature-aligned proxy benchmarks) intended to make both the numerics and the physics
reviewer-auditable.

This page documents the main limitations that remain on the path to a **fully nonlinear, 3D,
energy-conserving, multiphysics edge/SOL DRB solver**.

## Physical scope

### What is implemented and benchmarked

- Linear field-line DRB workflows (cold-ion, hot-ion, EM) with matrix-free `J·v` solvers.
- Open-field-line MPSE/sheath-entrance closures (Bohm/Loizu-style) with quantitative consistency gates.
- Nonlinear 2D milestones (HW2D and DRB2D) with conservative gates and energy-budget closure checks.
- FCI/3D *preparation milestones* (maps + parallel operators + minimal 3D slab operators) with MMS and
  conservative/sheath budget gates.

For the authoritative list of checks and where they live (tests/examples), see `docs/validation.md`.

### What is still a roadmap item (for quantitative SOL turbulence)

The core missing pieces for full 3D SOL turbulence prediction are:

- A full 3D open-field-line model with **target plates** and a complete sheath closure set in 3D
  (not just 1D MPSE constraints or “sheath damping” milestones).
- A real **non-Boussinesq polarization** solve in real space:
  $$
  -\nabla_\perp\cdot(n\,\nabla_\perp\phi)=\Omega,
  $$
  including robust SPD preconditioning and energy-rate gates.
- More complete edge/SOL physics: realistic sources/sinks, improved neutral models, gyroviscosity and
  higher-fidelity Braginskii closures, and full EM coupling (including inductive dynamics) in 3D.
- A geometry pipeline that produces FCI maps for diverted tokamaks / islands / stellarators with robust
  boundary intersection detection.

### Interpretation of “drives” and experimental mapping

Several workflows use **controlled proxy parameters** (e.g. `omega_n`, `omega_Te`, or geometry knobs like
`curvature0`) to reproduce reduced-model literature scans and to separate instability branches.

This makes tests and benchmarks reproducible, but it is not the same as a full experimental mapping.
Mapping a simulation to a specific device requires choosing reference scales and translating gradients,
collisionality, and geometry consistently (see `docs/model/normalization.md`).

## Numerical scope

- The field-line coordinate `l` uses periodic finite differences by default.
- The eigenvalue solver is a basic, matrix-free Arnoldi implementation without implicit restarting.
- For cases that converge slowly (near-marginal growth, nearly-degenerate modes, or non-normal operators),
  the CLI increases the Krylov dimension up to the full state dimension
  `N = 5 * nl`.

## Coordinate and operator conventions

Geometry is abstracted, so the meaning of `curvature0`, `omega_d`, and metric coefficients is tied
to the chosen geometry model. This is deliberate: it keeps the solver core geometry-agnostic.

If you intend to compare to a specific reference (e.g., a published dispersion relation), ensure
that:

- your geometry model matches the reference’s operator definitions,
- the parameter normalization matches the reference,
- the equilibrium drives (e.g. `omega_n`) map to the reference gradients.
