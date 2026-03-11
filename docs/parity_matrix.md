# Parity Matrix

This document tracks the parity buildout against the private reference implementation. `legacy/` is archival only and is not part of the active implementation plan.

## Stage 1: Configuration And Runtime Skeleton

Goal: reproduce reference input semantics, component scheduling semantics, and normalization bookkeeping before any PDE kernels are ported.

Deliverables:

- `BOUT.inp` parser with section/key order preservation.
- Scalar expression resolver for reference-style numeric expressions such as `AA = 1/1836`, `dy = Ly / ny`, `Bnorm = mesh:Bxy`, and `bxcvz = 1./Rxy^2`.
- Component expansion from `[model] components = ...` and per-species `type = ...`.
- Normalization model reproducing `Nnorm`, `Tnorm`, `Bnorm`, `Cs0`, `Omega_ci`, `rho_s0`, and output-unit metadata.
- Scheduler contract that executes all `transform()` hooks before any finalization hook, matching `ComponentScheduler::transform()` semantics in reference.
- Immutable state container for fields, diagnostics, and metadata.
- CLI skeleton for `inspect` and `run --dry-run`.

Tests:

- parser ordering and comment stripping;
- multiline component lists;
- numeric expression resolution;
- normalization formulas;
- scheduler ordering;
- live reference output summary extraction from `BOUT.dmp.0.nc`.

## Stage 2: Parity Harness

Goal: support `nout = 0` one-RHS checks, one fixed step, and short-time windows against selected reference inputs.

Deliverables:

- manifest of reference cases;
- live reference runner that stages isolated work directories and applies parity-mode overrides such as `nout=0` and `nout=1`;
- NetCDF summary extraction for selected compare variables and scalar metadata;
- reference/JAX comparison harness;
- reference dump metadata schema;
- first regression baselines;
- native JAX `one_rhs` execution for `evolve_density_rhs`, including structured-mesh coordinates, array-expression evaluation, boundary reconstruction, portable summary emission, and baseline regression tests;
- native JAX `one_step` execution for `diffusion_one_step`, including strict `H(...)` support, structured metric normalization, Neumann guard reconstruction, and an exact one-step radial transport advance.

## Stage 3+: Physics Buildout

The remaining stages stay as defined in [PLAN.md](/Users/rogerio/local/jax_drb/PLAN.md):

- mesh and metric parity;
- finite-volume operators and MMS parity;
- 1D open-field fluid core;
- sheath, recycling, and control terms;
- 2D electrostatic drifts and vorticity;
- 3D electromagnetic capabilities;
- neutrals, reactions, and impurities;
- performance, packaging, validation, and documentation.
