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
- manufactured-solution order-of-accuracy reports on promoted operator lanes,
  starting with the public `fluid_1d_mms_convergence` campaign and the
  open-field operator campaign for parallel-gradient and force-balance
  refinement

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

- [fluid_1d_mms_convergence](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__fluid_1d_mms_convergence_artifacts__images__fluid_1d_mms_convergence.png)
  for order-of-accuracy evidence
- [open_field_operator_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__open_field_operator_campaign_artifacts__images__open_field_operator_campaign.png)
  for open-field parallel-gradient, force-balance, target-recycling, and
  autodiff evidence
- [reactions_collisions_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__reactions_collisions_campaign_artifacts__images__reactions_collisions_campaign.png)
  for rate, source-partition, and closure checks
- [neutral_parallel_diffusion_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_parallel_diffusion_campaign_artifacts__images__neutral_parallel_diffusion_campaign.png)
  for AFN-versus-multispecies neutral closure comparison
- [collision_closure_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__collision_closure_campaign_artifacts__images__collision_closure_campaign.png)
  for friction, conduction, and viscosity closure activity
- [tokamak_anomalous_diffusion_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_anomalous_diffusion_campaign_artifacts__images__tokamak_anomalous_diffusion_campaign.png)
  for geometry-sensitive transport effects on an evolved tokamak state
- [target_recycling_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__target_recycling_campaign_artifacts__images__target_recycling_campaign.png)
  for target-localized recycling and sheath-conditioned closure activity
- [hermes_live_rerun_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__hermes_live_rerun_campaign_artifacts__images__hermes_live_rerun_campaign.png)
  for same-machine native-versus-live-Hermès code-to-code comparison across the
  current representative 1D and 2D matrix
- [neutral_mixed_boundary_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_boundary_campaign_artifacts__images__neutral_mixed_boundary_campaign.png)
  for boundary-localized neutral-mixed mismatch analysis on the live rerun
  surface
- [neutral_mixed_term_balance_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__neutral_mixed_term_balance_campaign_artifacts__images__neutral_mixed_term_balance_campaign.png)
  for term-level `NVh` residual localization on the neutral-mixed one-step
  Hermès mismatch
- [autodiff_diffusion_uncertainty](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_uncertainty_artifacts__images__autodiff_diffusion_uncertainty.png)
  for uncertainty propagation on the differentiable lane
- [autodiff_diffusion_sensitivity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_sensitivity_artifacts__images__autodiff_diffusion_sensitivity.png)
  for gradient-versus-finite-difference sensitivity evidence
- [autodiff_diffusion_inverse_design](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_inverse_design_artifacts__images__autodiff_diffusion_inverse_design.png)
  for a closed differentiable optimization example on the same native lane
- [local_cpu_scaling_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__local_cpu_scaling_campaign_artifacts__images__local_cpu_scaling_campaign.png)
  for workstation throughput on repeated heavy production solves
- [stellarator_fci_validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__operators__images__stellarator_fci_operator_campaign.png)
  for the first native non-axisymmetric metric, field-line-map, operator, and
  reduced 3D SOL dynamics evidence bundle
- [stellarator_metric_mms_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__metric_mms__images__stellarator_metric_mms_campaign.png)
  for the full \(J^{-1}\partial_i(JK g^{ij}\partial_j f)\) scalar-operator
  manufactured-solution gate, including non-orthogonal metric cross-term
  activity on the synthetic stellarator geometry
- [stellarator_sheath_recycling_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__sheath_recycling__images__stellarator_sheath_recycling_campaign.png)
  for non-axisymmetric traced-endpoint sheath losses, zero-current particle
  reconstruction, and exact recycling source accounting
- [essos_imported_drb_movie_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_diagnostics.png)
  for the first imported Landreman-Paul QA coil movie-grade transient with
  fixed-layout DRB state, sheath/recycling/neutrals, tight potential residual,
  and explicit reduced-transient scope labeling
- [stellarator_neutral_physics_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__neutral_physics__images__stellarator_neutral_physics_campaign.png)
  for neutral diffusion plus ionisation, recombination, and charge-exchange
  conservation on the non-axisymmetric map
- [stellarator_vorticity_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__vorticity__images__stellarator_vorticity_campaign.png)
  for metric-weighted vorticity inversion and the first non-axisymmetric
  radial \(E\times B\) diagnostic seam
- [stellarator_drb_pytree_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__pytree_drb__images__stellarator_drb_pytree_campaign.png)
  for the current fixed-layout 3D PyTree RHS, JVP derivative check, batched
  objective equivalence, and local/multi-device profiling seam
- [essos_landreman_paul_qa_fieldline_import](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_fieldline_import_artifacts__images__essos_landreman_paul_qa_fieldline_import.png)
  for ESSOS-owned field evaluation, adaptive field-line tracing, Poincare
  extraction, and portable geometry arrays that `jax_drb` can consume without
  maintaining duplicate coil-field or field-line-tracing code
- [essos_imported_fci_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_artifacts__images__essos_imported_fci_campaign.png)
  for imported field-line maps feeding JAXDRB sheath/recycling and neutral
  closures with fixed-shape FCI endpoint masks
- [essos_imported_pytree_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_pytree_artifacts__images__essos_imported_pytree_campaign.png)
  for imported field-line maps feeding the fixed-layout PyTree RHS, `jax.jvp`,
  and `jax.vmap` gates

### Supporting engineering figures

These remain useful and should stay in the docs, but they should not be the
main evidence panels in the future paper unless they are paired with a more
physics-facing interpretation:

- [hermes_comparison_summary](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__hermes_comparison_summary_artifacts__images__hermes_comparison_summary.png)
  is an index figure across heterogeneous lanes, not a direct literature-style
  benchmark figure
- [jax_native_profile_audit](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__jax_native_profile_audit_artifacts__images__jax_native_profile_audit.png)
  is an engineering/profile figure rather than a physics validation figure
- [native_3d_runtime_campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__native_3d_runtime_campaign_artifacts__images__native_3d_runtime_campaign.png)
  is a runtime/supporting figure rather than a primary scientific result
- [strong_scaling_diffusion](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__strong_scaling_diffusion_artifacts__images__strong_scaling_diffusion.png)
  is useful as a differentiable-kernel scaling check, while the heavier local
  CPU scaling campaign remains the stronger workstation result

This distinction matters because verification and validation papers such as
Roy 2005 emphasize order studies, explicit error measures, and model-versus-data
separation; the GBS parallel-gradient and GBS code papers use analytic-wave,
operator, and nonlinear-code comparisons; the TCV-X21 and SOLPS-ITER TCV-X21
papers use profile, target, neutral, and source observables tied to a physical
question; and JAX performance guidance emphasizes profiling, persistent
compilation caching, and transformable kernels before broad accelerator claims.
Summary dashboards are acceptable only as supporting context.

## Literature Anchors For The Validation Ladder

The current validation plan is aligned with the following literature patterns:

- Verification and solution quality: manufactured-solution convergence,
  observed order, and explicit numerical-error reporting following the
  verification/validation standards summarized by Roy and related CFD V&V
  literature.
- SOL turbulence and open-field numerics: parallel-gradient accuracy, shear
  Alfvén/operator tests, and nonlinear SOL comparisons following the GBS
  parallel-gradient and GBS code papers.
- Diverted tokamak validation: profile and target observables, diagnostic
  families, neutral sensitivity, and source-distribution interpretation
  following the TCV-X21, SOLPS-ITER TCV-X21, and Hermes-3 validation style.
- Differentiable scientific computing: explicit gradient checks, UQ pushforward
  against Monte Carlo, inverse design, `jax.linearize`/`jvp` Jacobian actions,
  and measured compile/execute/profiling behavior following the JAX autodiff,
  profiling, `pmap`, and persistent-cache documentation.

Key public references for this ladder include Roy's NASA V&V overview, the
TCV-X21 validation case, SOLPS-ITER against TCV-X21, the Hermès-3 code paper,
the Hermès-3 documentation/source diagnostics, and the official JAX profiling,
`pmap`, autodiff, and persistent-cache documentation.

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

Reviewer-facing convergence campaigns should live outside the default gate and
be run explicitly, for example through the optional `convergence_campaign` slice
or the public `fluid_1d_mms_convergence` and `open_field_operator_campaign`
artifact packages.

## Current Strategic Focus

The current critical path is not “add more staged cases.” It is:

1. finish one fully native open-field recycling transient backbone
2. promote that backbone through `one_rhs -> one_step -> short_window`
3. reuse it for integrated and direct-tokamak recycling/production lanes
4. widen the matrix only after that native closure is stable

The current live rerun evidence sharpens that priority:

- compact tokamak transport/turbulence lanes are already exact and much faster
  on the guarded compare surface
- integrated and direct-tokamak recycling are now at or below wall-time parity
  on this machine, but their current one-step mismatch is dominated by near-zero
  `NVd` on the guarded compare surface rather than by large absolute profile
  error
- heavy 1D recycling and the neutral mixed lane remain the main fidelity and
  runtime gaps
- the neutral lane now also has a dedicated live rerun boundary-audit figure,
  plus a term-level `NVh` residual-balance figure, so the remaining mismatch is
  no longer represented only by matrix-level summary scalars
- full live 3D Hermès reruns are still missing, so 3D remains a distinct
  selected-field evidence track rather than part of the live rerun matrix

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
- repository-slimming package that removes legacy and paper-only material from
  the active release branch while preserving necessary benchmark artifacts

## Publication Scope

The first target is:

- a research-grade, restartable, well-documented JAX edge/SOL code
- a clearly stated supported matrix
- exact or tightly bounded parity on that matrix
- explicit labeling of anything still operational or scaffolded

Differentiability remains a design requirement, but it is staged after the first high-confidence parity release rather than being used to block the core evidence program.
