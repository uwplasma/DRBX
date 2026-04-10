# Research-Grade Validation Matrix

This document is the public-facing contract for how `jax_drb` will justify a strong-subset publication claim.

## Capability Tiers

- `native_exact`
  - Fully native solve path.
  - Clean enough to anchor a parity claim in the paper.
  - Must satisfy the full promotion gate below.
- `native_operational`
  - Native path with bounded residuals.
  - Useful for research iteration and internal validation.
  - Not counted as headline evidence until promoted.
- `scaffolded_reference_backed`
  - Replay, dump, or cached-history assisted path.
  - Useful for diagnostics, operator localization, and bridge coverage.
  - Must not be presented as equivalent to native closure.

## Promotion Gate

Before any case family is promoted to `native_exact`, it must have:

1. one-RHS parity on the smallest exercising case
2. one-step parity on the same family
3. short-window parity for transient workflows
4. unit tests for every new operator or boundary branch used on that family
5. at least one physics-facing diagnostic test
6. restart/resume equivalence if the family is user-facing
7. output/log/provenance artifact coverage if the family is exposed in CLI/examples
8. explicit capability-tier labeling in the manifest, docs, and run log

## Validation Layers

### Unit / Operator

- sheath boundary formulas
- recycling source reconstruction
- ion viscosity scaling and geometry dependence
- reaction parser, rate interpolation, and source partition
- vorticity / `phi` / `Apar` operator identities

### Parity Regression

- exact summary and array baselines for `native_exact`
- bounded residual checks for `native_operational`
- committed cache-backed diagnostics for `scaffolded_reference_backed`

### Physics

- conservation / decay / symmetry / steady-state checks
- benchmark-specific diagnostics for drift-wave, blob, Alfvén, recycling, and tokamak lanes

### Convergence

- timestep refinement on promoted transient lanes
- spatial refinement where practical on promoted benchmark families

### Runtime

- restart equivalence
- output/log completeness
- precision-mode behavior
- verbose logging coverage

## Current Strategic Focus

The current critical path is not “add more staged cases.” It is:

1. finish one fully native open-field recycling transient backbone
2. promote that backbone through `one_rhs -> one_step -> short_window`
3. reuse it for integrated and direct-tokamak recycling/production lanes
4. widen the matrix only after that native closure is stable

## Required Campaigns

- operator-focused recycling / ion-viscosity campaign
- direct tokamak convergence campaign
- TORPEX seeded blob benchmark package
- TCV-X21 diverted L-mode benchmark package
- detachment-scaling package
- performance and memory benchmark package on promoted native paths

## Publication Scope

The first paper target is:

- a research-grade, restartable, well-documented JAX edge/SOL code
- a clearly stated supported matrix
- exact or tightly bounded parity on that matrix
- explicit labeling of anything still operational or scaffolded

Differentiability remains a design requirement, but it is staged after the first publication-grade parity release rather than being used to block the core evidence program.
