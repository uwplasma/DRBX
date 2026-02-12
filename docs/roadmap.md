# Roadmap

This project focuses on correctness, extensibility, and reviewer‑proof verification over feature‑count.
The code now includes **both linear field‑line solvers and nonlinear 2D DRB testbeds**, plus FCI/3D
scaffolding. The roadmap below tracks the remaining steps toward a **fully nonlinear 3D, energy‑conserving,
multiphysics DRB solver** for edge/SOL turbulence.

## Current scope (implemented)

- Linear field‑line DRB solvers with matrix‑free Arnoldi and initial‑value growth estimation.
- Nonlinear HW2D and DRB2D models with conservative gates, energy budgets, and solver comparisons.
- Hot‑ion and EM branches in DRB2D with curvature‑drive benchmarks.
- Neutral coupling and MMS verification.
- FCI preparation: analytic slab maps, curved‑map regression, and minimal 3D slab operators with
  conservative + sheath budget gates.

## Near‑term (next milestones)

- Add shift‑invert support (GMRES solve of `(J - σ I)x = b`) to better target unstable eigenvalues.
- Extend non‑Boussinesq polarization beyond linearized forms (state‑dependent `n` in 2D/3D gates).
- Add multi‑mode perpendicular spectral support to enable nonlinear brackets in 3D.
- Expand FCI to include non‑uniform `B` maps, target plates, and one‑sided parallel stencils.

## Medium‑term (physics completeness)

- Full sheath boundary conditions (Bohm/Loizu) in 3D with energy‑consistent closures.
- Braginskii closures beyond current scalings (gyroviscosity and realistic sources/sinks).
- Full EM coupling (inductive fields) with robust current closure and energy budgets in 3D.
- Geometry pipelines: VMEC / stellarator / diverted tokamak field‑line tracing into FCI maps.

## Long‑term (production‑level 3D SOL turbulence)

- Fully nonlinear 3D DRB with energy‑conserving discretizations and open‑field‑line physics.
- IMEX / implicit solvers for stiff closures with matrix‑free preconditioning in JAX.
- Regression gates tied to published benchmarks and long‑time turbulence statistics.
