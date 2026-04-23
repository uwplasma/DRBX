# Research-Grade Validation Matrix

This document is the public-facing contract for how `jax_drb` justifies its promoted validation surfaces.

## Capability Tiers

- `native_exact`
  - Fully native solve path.
  - Clean enough to anchor the main parity claim.
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
- manufactured-solution order-of-accuracy reports on promoted operator lanes, starting with the public `fluid_1d_mms_convergence` campaign and its standalone `scripts/run_fluid_1d_mms_convergence.py` wrapper

### Runtime

- restart equivalence
- output/log completeness
- precision-mode behavior
- verbose logging coverage
- fast-gate execution with bounded wall time for curated research slices

## Current Figure Standard

The literature does not treat every dashboard as equal evidence. The current
`jax_drb` public artifact surface should be interpreted in two classes.

### Main scientific figures

These are the current figure families that are close to the literature pattern
used in verification, validation, and differentiable-science papers:

- [fluid_1d_mms_convergence](data/fluid_1d_mms_convergence_artifacts/images/fluid_1d_mms_convergence.png)
  for order-of-accuracy evidence
- [reactions_collisions_campaign](data/reactions_collisions_campaign_artifacts/images/reactions_collisions_campaign.png)
  for rate, source-partition, and closure checks
- [neutral_parallel_diffusion_campaign](data/neutral_parallel_diffusion_campaign_artifacts/images/neutral_parallel_diffusion_campaign.png)
  for AFN-versus-multispecies neutral closure comparison
- [collision_closure_campaign](data/collision_closure_campaign_artifacts/images/collision_closure_campaign.png)
  for friction, conduction, and viscosity closure activity
- [tokamak_anomalous_diffusion_campaign](data/tokamak_anomalous_diffusion_campaign_artifacts/images/tokamak_anomalous_diffusion_campaign.png)
  for geometry-sensitive transport effects on an evolved tokamak state
- [target_recycling_campaign](data/target_recycling_campaign_artifacts/images/target_recycling_campaign.png)
  for target-localized recycling and sheath-conditioned closure activity
- [autodiff_diffusion_uncertainty](data/autodiff_diffusion_uncertainty_artifacts/images/autodiff_diffusion_uncertainty.png)
  for uncertainty propagation on the differentiable lane
- [local_cpu_scaling_campaign](data/local_cpu_scaling_campaign_artifacts/images/local_cpu_scaling_campaign.png)
  for workstation throughput on repeated heavy production solves

### Supporting engineering figures

These remain useful and should stay in the docs, but they should not be the
main evidence panels in the future paper unless they are paired with a more
physics-facing interpretation:

- [hermes_comparison_summary](data/hermes_comparison_summary_artifacts/images/hermes_comparison_summary.png)
  is an index figure across heterogeneous lanes, not a direct literature-style
  benchmark figure
- [jax_native_profile_audit](data/jax_native_profile_audit_artifacts/images/jax_native_profile_audit.png)
  is an engineering/profile figure rather than a physics validation figure
- [native_3d_runtime_campaign](data/native_3d_runtime_campaign_artifacts/images/native_3d_runtime_campaign.png)
  is a runtime/supporting figure rather than a primary scientific result

This distinction matters because papers such as Roy 2005, the GBS
parallel-gradient CPC paper, the TCV-X21 benchmark paper, SOLPS-ITER against
TCV-X21, and Hermes-3 against TCV-X21 all emphasize convergence curves,
profile/target comparisons, source or diagnostic maps, and scan figures tied to
 a physical question. Summary dashboards are acceptable only as supporting
 context.

## Fast Validation Policy

The default developer/research gate is now:

- [scripts/run_fast_research_checks.py](../scripts/run_fast_research_checks.py)

It runs curated slices covering:

- runtime / CLI / restart surfaces
- portable parity payload helpers
- manufactured-solution convergence/history checks
- open-field and implicit-operator checks
- recycling operator and blocker diagnostics

Each slice has a hard 5-minute timeout by default. If a slice exceeds that limit, the gate fails immediately and the underlying pytest process is terminated. The point is to keep research iteration bounded and avoid stale long-running local checks from replacing focused evidence.

Coverage is opt-in on this gate. The default run is intentionally a fast no-coverage pass; use `--with-coverage` when you are explicitly measuring coverage rather than iterating on operator changes.

Longer transient-solver history tests should be marked `slow` and kept out of this default gate unless they are the specific subject of the current iteration.

Reviewer-facing convergence campaigns should live outside the default gate and be run explicitly, for example through the optional `convergence_campaign` slice or the public `fluid_1d_mms_convergence` artifact package and its standalone `scripts/run_fluid_1d_mms_convergence.py` wrapper.

## Current Strategic Focus

The current critical path is not “add more staged cases.” It is:

1. finish one fully native open-field recycling transient backbone
2. promote that backbone through `one_rhs -> one_step -> short_window`
3. reuse it for integrated and direct-tokamak recycling/production lanes
4. widen the matrix only after that native closure is stable

## Required Campaigns

- operator-focused recycling / ion-viscosity campaign
- neutral parallel-diffusion closure campaign on prepared multispecies recycling states
- collision/conduction closure campaign on prepared multispecies recycling states
- tokamak anomalous-diffusion campaign on evolved direct-recycling states
- target-recycling and sheath-response campaign on prepared multispecies recycling states
- direct tokamak convergence campaign
- TORPEX seeded blob benchmark package
- TCV-X21 diverted L-mode benchmark package
- detachment-scaling package
- performance and memory benchmark package on promoted native paths
- differentiable sensitivity / inverse-design / scaling artifact package on promoted native paths

## Publication Scope

The first target is:

- a research-grade, restartable, well-documented JAX edge/SOL code
- a clearly stated supported matrix
- exact or tightly bounded parity on that matrix
- explicit labeling of anything still operational or scaffolded

Differentiability remains a design requirement, but it is staged after the first high-confidence parity release rather than being used to block the core evidence program.
