# Hermes-3 Parity Matrix

This repository treats Hermes-3 as the numerical and interface reference. `legacy/` is archival only and is not part of the implementation plan.

## Stage 1: Configuration And Runtime Skeleton

Goal: reproduce Hermes input semantics, component scheduling semantics, and normalization bookkeeping before any PDE kernels are ported.

Deliverables:

- `BOUT.inp` parser with section/key order preservation.
- Scalar expression resolver for Hermes-style numeric expressions such as `AA = 1/1836`, `dy = Ly / ny`, `Bnorm = mesh:Bxy`, and `bxcvz = 1./Rxy^2`.
- Component expansion from `[hermes] components = ...` and per-species `type = ...`.
- Normalization model reproducing `Nnorm`, `Tnorm`, `Bnorm`, `Cs0`, `Omega_ci`, `rho_s0`, and output-unit metadata.
- Scheduler contract that executes all `transform()` hooks before any finalization hook, matching `ComponentScheduler::transform()` semantics in Hermes.
- Immutable state container for fields, diagnostics, and metadata.
- CLI skeleton for `inspect` and `run --dry-run`.

Tests:

- parser ordering and comment stripping;
- multiline component lists;
- numeric expression resolution;
- normalization formulas;
- scheduler ordering;
- live Hermes output summary extraction from `BOUT.dmp.0.nc`.

## Stage 2: Parity Harness

Goal: support `nout = 0` one-RHS checks, one fixed step, and short-time windows against selected Hermes reference inputs.

Deliverables:

- manifest of reference Hermes cases;
- live Hermes runner that stages isolated work directories and applies parity-mode overrides such as `nout=0` and `nout=1`;
- NetCDF summary extraction for selected compare variables and scalar metadata;
- hermes/jax comparison harness;
- reference dump metadata schema;
- first regression baselines.

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
