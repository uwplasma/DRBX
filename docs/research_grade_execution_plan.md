# Research-Grade Execution Plan

This is the single authoritative execution plan for making JAXDRB a
research-grade, lightweight, documented, validated, performant, differentiable
scrape-off-layer and edge-plasma code. It consolidates the older refactoring,
geometry, validation, runtime, parity, documentation, and release notes into one
ordered plan.

## Plan Authority

This file is the only active plan. All implementation work should update this
file first, then update a subordinate technical page only when the extra detail
belongs there.

Subordinate pages:

- [refactoring_plan.md](refactoring_plan.md): source and test architecture
  appendix.
- [testing_strategy.md](testing_strategy.md): test policy and evidence-layer
  appendix.
- [runtime_gap_remediation.md](runtime_gap_remediation.md): parity, runtime,
  and memory offender appendix.
- [geometry_roadmap.md](geometry_roadmap.md): reusable 3D geometry architecture
  appendix.
- [non_axisymmetric_stellarator_sol_plan.md](non_axisymmetric_stellarator_sol_plan.md):
  stellarator geometry, FCI, and validation appendix.
- [vmec_extender_edge_fields.md](vmec_extender_edge_fields.md): VMEC-extender
  import-contract appendix.
- [parity_matrix.md](parity_matrix.md): historical parity buildout appendix.
- [research_directions.md](research_directions.md): research-context appendix.
- [release_packaging.md](release_packaging.md): release, packaging, PyPI, and
  heavy-artifact appendix.
- `../plan_jax_drb.md`: compatibility redirect to this file.

Other Markdown files under `docs/` are status pages, validation reports,
example pages, literature notes, or rendered-artifact documentation. They may
contain historical "next step" or "remaining" language for a specific campaign,
but they are not execution plans. If any page conflicts with this file, follow
this file and update the page after the implementation decision is made.

Hosted GitHub Actions should not be polled while hosted runners are blocked by
account billing/spending-limit state. Billing is exhausted for the current
period, so development should use local release gates and only recheck hosted
CI after billing is restored or the user explicitly asks for a CI rerun.

Cross-document audit policy:

- Markdown pages under `docs/` may contain background, derivations, historical
  measurements, campaign reports, or detailed technical appendices.
- They must not introduce a different priority order, a different completion
  status, or a different definition of done from this file.
- Historical "remaining work" notes in appendices are acceptable only when they
  describe the evidence behind the current blocker recorded here.
- If an appendix still contains an obsolete next step, update this file first,
  then edit the appendix to say whether the item is complete, superseded, or
  still active.
- Campaign/status docs may keep local evidence narratives. Their priorities,
  completion percentages, and release blockers are authoritative only if they
  match the current completion snapshot and implementation backlog below.
- `plan_jax_drb.md` remains a compatibility redirect. Do not add execution
  steps there.

## How To Execute This Plan

This file should be used as the working checklist for every future pass.

Working rules:

1. Start every work pass by reading the current completion snapshot, the ordered
   execution plan, and the execution log.
2. Pick the highest-priority unblocked lane from the ordered plan, or a later
   independent lane only when the current lane is blocked by external data,
   reference runs, hardware, or reviewer decision.
3. Before broadening any README, docs, release, or paper claim, add or update
   the matching validation, performance, coverage, geometry, or parity evidence
   in this file.
4. Use subordinate pages only for technical detail that is too long for this
   master plan. Do not create new competing roadmap files.
5. Keep heavy generated artifacts out of git. If an example needs large
   fixtures, movies, traces, or dumps, use release assets plus a documented
   downloader/restoration command.
6. Keep ordinary user workflows self-contained. Reference-code runs are
   developer validation tools, not required dependencies for normal examples.
7. Log decisions and negative benchmarks. A failed optimization, failed
   preconditioner, or rejected geometry/movie is useful evidence and should be
   recorded before moving to the next option.
8. Commit and push coherent plan, docs, tests, or code batches frequently, but
   do not wait on hosted CI while the runner account is blocked.
9. When a user asks for a plan-only pass, stop after updating this file and
   reviewing the diff. Do not start solver changes, validation runs, or test
   campaigns until the plan is approved.

Definition of a completed lane:

- implementation exists behind a stable public API or explicitly opt-in
  experimental API;
- tests cover equations, numerics, differentiability or parity as appropriate;
- documentation explains equations, assumptions, inputs, outputs, examples,
  plots, and limitations;
- figures/movies are reproducible by documented scripts and have passed visual
  QA when promoted;
- runtime, memory, and scaling claims have same-fidelity evidence;
- repo-footprint and package audits do not show accidental heavy artifacts.

## Target Claim Boundary

The release target is not simply "examples run." The target is a codebase that
an external researcher can clone, install, run, audit, and cite.

JAXDRB should support the following promoted claim classes only when the listed
evidence exists:

- Drift-reduced Braginskii edge/SOL models with clearly documented equations,
  closures, normalizations, and implementation locations.
- Open-field and closed-field simulations in tokamak and stellarator geometry.
- Sheath, recycling, neutral, detachment, and target-response physics with
  equation-level and component-level tests.
- Diverted tokamak examples that are self-contained for users and can generate
  movies, target profiles, neutral/source plots, and turbulent diagnostics.
- 3D non-axisymmetric stellarator examples for VMEC, VMEC-extender,
  ESSOS-imported coil maps, hybrid VMEC/coil maps, HSX, NCSX,
  Landreman-Paul QA, and Dommaschk potentials.
- JAX-native residual and differentiability lanes where host-side SciPy or
  reference-code barriers are absent from the promoted objective.
- CPU and GPU performance claims tied to real heavy kernels, profiler evidence,
  memory evidence, and fidelity-preserving comparisons.
- Documentation and examples that teach users how to select models, geometries,
  boundary conditions, input files, outputs, plotting, validation, profiling,
  and differentiability workflows.

Unpromoted examples, reduced forcing terms, and visualization-only movies must
be explicitly labeled as reduced, pedagogical, exploratory, or scaffolded. A
movie is never a validation gate by itself.

## Current Completion Snapshot

Audit date: 2026-06-18. Percentages are approximate and evidence-based. A lane
moves only when implementation, validation, plots or diagnostics, documentation,
and tests all move together.

| Lane | Completion | Current blocker |
| --- | ---: | --- |
| Plan authority and release hygiene | 95% | Keep this file current and prevent new competing roadmap files. |
| Meaningful promoted coverage | 96% | Keep `scripts/run_promoted_solver_coverage.py` above `95%` after each solver and geometry promotion. |
| Reference-backed parity | 99.1% | Keep the closed neutral `NVh` source split locked while extending the same term-level parity discipline to recycling, sheath, target-source, and longer-window diverted-tokamak campaigns. |
| JAX-native recycling solver | 90% | Make the documented full-output JAX-transformable recycling path fast enough for broader opt-in promotion beyond bounded fixture gates. |
| Effective preconditioning | 42% | Move beyond opt-in local-block reuse speedup evidence to a transport-aware or Schur-style preconditioner that reduces Krylov budget. |
| Performance and scaling | 57% | Rerun heavy CPU/GPU profiles after solver changes and show real-kernel speedup, not only bounded fixture or compact-kernel throughput. |
| Drift-reduced Braginskii model surface | 65% | Finish equation-to-code maps, Boussinesq/non-Boussinesq comparisons, vorticity/potential gates, and EM selected-field promotion. |
| Neutral, recycling, sheath, detachment | 78% | Finish term-level neutral/recycling/sheath gates and detachment observables across promoted tokamak lanes. |
| Diverted tokamak self-contained tutorials | 70% | Ensure clean-clone users can fetch small/release-hosted fixtures, run simulations, create movies, and analyze turbulent profiles. |
| 3D stellarator imported-field/VMEC SOL | 67% | Promote live connection-length, endpoint, FCI, grid-refinement, and time-refinement gates before turbulence/movie claims. |
| Code architecture split | 60% | Split broad recycling, neutral, runner, CLI, and large test files into narrow directly tested modules. |
| Docs and examples | 86% | Make every advertised README figure/movie reproducible by a documented example and move extended validation detail into docs. |
| Repo footprint | 90% | Repeat `.git`, tracked-large-file, wheel/sdist, docs-media, and local-cache audits before every tag. |

## Milestone Map

The ordered plan below is detailed. This milestone map is the compact route to
finish the code without re-planning at every step.

| Milestone | Lanes | Exit criteria |
| --- | --- | --- |
| M0: planning and artifact control | plan authority, repo footprint, self-contained examples | This file is current, subordinate plans do not conflict, heavy assets are release-hosted or excluded, and users do not need private reference-code installs. |
| M1: reference-backed physics parity | neutral `NVh`, recycling, sheath, target sources, detachment observables | Remaining accepted-step sequencing offender is closed or bounded, parity reports rank remaining errors by term and field, and regression tests lock each closed offender. |
| M2: JAX-native recycling and preconditioning | fixed-layout residuals, JVP/Jacobian actions, physics/block preconditioners | Full-output recycling residual can run through the JAX-transformable seam, JVP/finite-difference gates pass, and at least one preconditioner gives same-case solver-health or runtime improvement before default promotion. |
| M3: complete DRB physics surface | Braginskii closures, vorticity/potential, Boussinesq/non-Boussinesq, electromagnetic selected fields, open/closed field lines | Each promoted term has equations, implementation links, limiting-case tests, and literature-anchored plots; reduced surrogate terms are either removed from promoted examples or labeled. |
| M4: self-contained diverted tokamak program | 1D/2D/3D diverted geometry, recycling, detachment, turbulence, movies, analysis scripts | Clean-clone users can run tokamak tutorials, generate movies/profiles, fetch only release-hosted fixtures when needed, and reproduce README visuals. |
| M5: 3D stellarator SOL program | VMEC, VMEC-extender, ESSOS coil maps, hybrid maps, HSX, NCSX, Landreman-Paul QA, Dommaschk | Boundary, Poincare, connection-length, endpoint, FCI, open/closed, refinement, sheath/recycling/neutral, and reduced/full transient gates pass before movies are promoted. |
| M6: performance and differentiability evidence | CPU, GPU, multi-device, `jit`, `vmap`, `jvp`, VJP/grad, UQ/inverse design | Real heavy kernels have cProfile/RSS/JAX-trace evidence, CPU/GPU scaling is same-fidelity, and differentiability examples compare against finite differences. |
| M7: release package | coverage, docs, README, examples, package, release notes | Promoted coverage is above `95%`, docs build locally, package is small, release notes state experimental boundaries, and tag/release/PyPI workflow are ready. |

## Target Capability Matrix

This matrix is the implementation contract for the next phase. A row can be
called promoted only after its equations, tests, artifacts, docs, examples, and
claim boundaries are complete.

| Capability | Required cases | Required evidence |
| --- | --- | --- |
| Effective preconditioning | hydrogen recycling, multispecies D/T/He recycling, diverted tokamak recycling, imported-field compact 3D RHS | same-case solver health, residual/JVP call counts, Krylov iterations, memory, runtime, and no parity degradation. |
| Diverted tokamak simulations | 1D recycling, 1D detachment, 2D diverted transport, 2D recycling, multispecies D/T/He, impurity/radiation, nonlinear turbulence window | target profiles, OMP profiles, source/radiation maps, heat/particle flux, neutral density, detachment metrics, movie, and clean-clone tutorial. |
| Drift-reduced Braginskii physics | diffusion-only, electrostatic DRB, vorticity/potential, Boussinesq, non-Boussinesq, selected electromagnetic/Alfven, open and closed field lines | equation-to-code map, limiting-case tests, MMS/operator convergence, vorticity inversion, bracket tests, Boussinesq comparison, and labeled reduced/full status. |
| JAX-native recycling | fixed-layout residual, full-output BDF residual, sparse-JVP/JVP action, VJP/grad objectives, matrix-free or sparse linearization, D/T/He adaptive BDF | correctness parity with compatibility solver, JVP versus finite-difference tests, solver-health report, cProfile/RSS/JAX trace, and opt-in/default decision. |
| Sheath/recycling/neutrals | no-flow, zero-current, sheath heat transmission, target recycling, neutral diffusion, ionization, recombination, charge exchange, radiation, detachment | source accounting, boundary reconstruction, target-response tests, term-level parity, detachment scan, neutral/source/target plots, and docs derivations. |
| Open and closed field-line simulations | tokamak closed surfaces, tokamak open SOL, stellarator closed VMEC maps, stellarator open endpoint maps, hybrid maps | separate closed/open validation reports, connection-length definitions, endpoint masks, profile diagnostics, and no cross-use of closed-map metrics as open-target claims. |
| 3D stellarator geometries | VMEC, VMEC-extender, ESSOS coil import, hybrid VMEC/coil maps, HSX QHS, NCSX, Landreman-Paul QA, Dommaschk potentials | boundary/surface plot, Poincare/field-line plot, connection-length map, endpoint mask, FCI convergence, grid/time refinement, physics transient, movie QA. |
| Reference parity | neutral mixed, recycling, direct tokamak, production/diverted tokamak, selected-field geometry, target/sheath/recycling sources | one-RHS, one-step, short-window, accepted-step traces, field/term/location offender register, and regression tests for every closed offender. |
| CPU/GPU performance | Mac CPU ensemble, real heavy recycling, imported-field compact RHS, single GPU, multi-GPU ensemble/sharding when available | same-fidelity scaling figure, compile/execute split, memory, persistent cache state, and profiler bundle linked from docs. |
| Documentation/examples | README, ReadTheDocs, model-selection guide, tokamak tutorial, stellarator tutorial, validation gallery, performance guide | every advertised figure/movie has a command, no private reference-code dependency for users, extended derivations in docs, and concise README. |
| Coverage/release | promoted solver surface, public CLI/API, examples, validation campaigns, package, footprint | `95%` promoted coverage, local release gates, package audit, no large blobs, release-hosted assets, release notes, tag, and PyPI workflow readiness. |

## Current Implementation Backlog

This backlog is the executable checklist for the next implementation phase.
Work should proceed in this order unless a task is blocked by missing external
reference data, unavailable GPU hardware, or reviewer/user decision. Later
independent lanes may run in parallel only when they do not broaden unsupported
claims.

| Priority | Track | Concrete next actions | Exit gate |
| --- | --- | --- | --- |
| P0 | Plan authority and repo hygiene | Keep this file as the only active plan; keep `plan_jax_drb.md` as a redirect; audit roadmap-like Markdown pages after each major decision; keep hosted CI out of the critical path while billing is exhausted; keep heavy traces, NetCDF dumps, profile bundles, and movies out of git. | Clean plan diff, no competing roadmap, clean `git status`, footprint audit before tag. |
| P1 | Reference-backed neutral parity | Keep the closed neutral `NVh` accepted-step source split under regression, rerun the reference monitor after future neutral/recycling changes, and apply the same exact-reference-state source-register pattern to remaining recycling, sheath, target-source, and longer-window parity campaigns. | Accepted-step trace matches the reference time grid, pressure-gradient/diffusion/viscosity remain roundoff-closed, `SNVh_parallel_inertia` stays roundoff-bounded, and regression tests lock the flux-mode decision. |
| P2 | JAX-native recycling residual | Promote the full-output recycling BDF residual through the fixed-layout PyTree/array seam; port sheath/no-flow, zero-current, target recycling, collisions, neutral diffusion, D/T/He reactions, scalar feedback, BE/BDF2/adaptive history, and artifact output without host-side residual loops. | Compatibility solver parity, JVP versus finite-difference tests, no hidden fallback, bounded solver-health report, and docs labeling default versus opt-in paths. |
| P3 | Effective preconditioning | Build physics/block preconditioners in increasing complexity: same-cell blocks, sparse-JVP materialized block controls, parallel-line transport, neutral/plasma Schur approximation, target/sheath local blocks, and 2D/3D FCI transport blocks. Use PETSc-style field-split and line-solve ideas as algorithms, not runtime dependencies. | Same-case speedup or reduced residual/JVP/Krylov count after build cost, no parity degradation, memory not worse than compatibility baseline, CPU/GPU trace evidence. |
| P4 | Drift-reduced Braginskii model surface | Finish equations, symbol definitions, normalizations, Boussinesq/non-Boussinesq polarization, vorticity/potential solve, ExB bracket, curvature/interchange, selected electromagnetic/Alfven lanes, open/closed-field semantics, and limiting-case/MMS tests. Remove or clearly label demo-only nonlinear forcing. | Equation-to-code map and tests for every promoted term, Boussinesq comparison plot, vorticity/bracket gates, EM selected-field report, no unsupported README claim. |
| P5 | Neutral/recycling/sheath/detachment physics | Complete source accounting for ionization, recombination, charge exchange, radiation, neutral diffusion, recycling, pumping, no-flow, zero-current, sheath heat transmission, target sources, and detachment controller metrics. | Term-level parity or analytic tests, target flux/source balance, detachment scan plots, docs derivations with implementation links. |
| P6 | Self-contained diverted tokamak program | Provide clean-clone tutorials that generate or fetch release-hosted fixtures; run 1D recycling, 1D detachment, 2D/3D diverted transport, multispecies D/T/He, impurity/radiation, and nonlinear turbulence windows; generate movies, OMP/target profiles, source maps, and neutral/radiation diagnostics. | Users can run examples and regenerate advertised README/docs media without installing reference codes; large fixtures remain release-hosted. |
| P7 | 3D stellarator geometry and SOL | Promote VMEC, VMEC-extender, ESSOS coil, hybrid VMEC/coil, HSX QHS, NCSX, Landreman-Paul QA, and Dommaschk lanes. For each device: source metadata, boundary/surface plot, Poincare/field-line plot, connection-length or closed-map parallel-step metric, endpoint masks, FCI MMS/convergence, linear dynamics, sheath/recycling/neutral gates, nonlinear transient, and movie QA. | Per-device validation bundle with grid/time refinement and frame-by-frame movie QA before any turbulence claim. |
| P8 | Performance, parallelization, and differentiability | Use `jit`, `vmap`, `jvp`, VJP/grad, persistent compilation cache, batched ensembles, CPU-device scaling, GPU kernels, and multi-GPU/sharded ensembles only on parity-proven JAX-native paths. Compare derivatives against finite differences and timings against same-fidelity baselines. | cProfile/RSS/JAX trace/XLA evidence, CPU/GPU scaling plots on real kernels, differentiability plots with finite-difference error curves. |
| P9 | Docs, examples, coverage, and release | Make README concise and docs comprehensive; every advertised figure/movie has a command; examples follow SIMSOPT-style top-level parameters plus imported API functions; maintain promoted coverage above `95%`; run footprint/package audits; prepare release notes/tag/PyPI workflow when hosted CI can run. | `mkdocs build --strict --clean`, closeout/promoted coverage above `95%`, clean-clone examples, package audit, release notes and experimental boundaries current. |

## Ordered Execution Plan

This order should be followed unless a lane is blocked and a later independent
lane can progress without broadening unsupported claims.

### 1. Freeze Planning, Claim Boundaries, And Release Hygiene

Objective:

Keep the project moving from one plan, one claim boundary, and one definition
of done.

Steps:

1. Keep this file as the active plan and keep `plan_jax_drb.md` as a redirect.
2. Add every implementation decision, validation gate, and negative benchmark
   result to the execution log before broadening a README/docs claim.
3. Keep hosted CI checks periodic while runner billing is blocked; do not spend
   time polling jobs that fail before steps start.
4. Before any tag, run local release gates:

```bash
PYTHONPATH=src python scripts/run_closeout_coverage.py
PYTHONPATH=src python scripts/run_promoted_solver_coverage.py
PYTHONPATH=src python scripts/run_fast_research_checks.py
PYTHONPATH=src mkdocs build --strict --clean
```

Promotion evidence:

- clean `git status` review;
- no staged unrelated source, generated dump, trace, or media artifact;
- plan log entry for the work being promoted;
- local release gates green or a documented, locally reproducible failure.

### 2. Close Reference-Backed Parity At Component Level

Objective:

Use reference comparisons as component diagnostics, not only final-state
aggregate plots.

Current state:

- Many RHS, one-step, short-window, and campaign gates exist.
- Direct neutral-mixed `SNVh_*` pressure-gradient and viscosity source formulas
  are closed against written reference diagnostics.
- The remaining neutral `NVh` offender is localized to the accepted-step
  parallel-inertia/Lax-flux source split plus state/history sequencing feeding
  neutral pressure/log-pressure preparation, with near-target
  `Grad(logPnlim)` as a secondary stencil check.

Steps:

1. Preserve clean reference rerun mode through `JAX_DRB_REFERENCE_ROOT`.
2. Keep users independent of reference-code installation by using committed
   lightweight fixtures and release-hosted artifacts for examples.
3. Add richer accepted-step reference monitor output with full active-field
   accepted states, scalar limiter inputs, guard-cell history, target-adjacent
   stencils, and component source terms.
4. Test native residual pieces directly on reference accepted states before
   changing formulas.
5. Patch state/history replay first. Only change the near-target
   `Grad(logPnlim)` stencil if the state/history patch fails to close the
   scalar flux-cap drift.
6. Maintain an offender register ranked by case, field, term, location,
   absolute error, relative/scaled error, runtime, memory proxy, and artifact.
7. Every closed offender gets a regression test and a docs note linking the
   equation, implementation function, reference field, and plot/report.

Promotion evidence:

- matched native/reference accepted-step traces on the same time grid;
- term-level source comparison for pressure-gradient, viscosity, diffusion,
  reaction, boundary, target, and sheath contributions;
- absolute and relative errors reported together, especially for near-zero
  fields;
- clean reference provenance, not a dirty exploratory checkout;
- offender register updated after the fix.

### 3. Finish The JAX-Native Recycling Solver Backbone

Objective:

Move the heavy recycling path from host-side residual loops and sparse
finite-difference Jacobian assembly toward a transformable fixed-layout JAX
residual with JVP/Jacobian-action and eventually matrix-free solves.

Target architecture:

- Stable compatibility tier: current NumPy/SciPy sparse finite-difference
  Newton/BDF path remains validated and available.
- Differentiable tier: fixed-layout PyTree/array residual, JAX-transformable
  kernels, JVP/Jacobian-action linearization, VJP/gradient objectives, and
  matrix-free or sparse-JVP solver options.

Steps:

1. Keep current negative solver/backend evidence opt-in: Lineax, BiCGSTAB,
   row/field scaling, linearized diagonal, local block-Jacobi, residual-JIT
   variants, and GMRES-control sweeps should not become defaults unless a
   same-case gate proves a runtime and solver-health win.
2. Complete fixed-layout residual assembly in this order:
   collisions, neutral diffusion, target recycling, sheath/no-flow
   reconstruction, zero-current ion-sum/potential reconstruction, controller
   feedback, multispecies D/T/He reaction accumulation, scalar feedback, BE
   residual, BDF2 residual, adaptive history/state preparation.
3. Promote the full-output BDF residual through the fixed-layout seam, not only
   compact tests.
4. Replace sparse finite-difference Jacobian assembly where feasible with
   sparse JVP materialization or matrix-free JVP actions.
5. Keep full-output BDF/JVP parity separate from fixed-BDF2 residual promotion
   until both pass correctness and runtime gates.
6. Add JVP versus centered finite-difference tests for every promoted residual
   component.
7. Add VJP/gradient tests for scalar objectives that will be used in UQ,
   inverse design, control, or optimization.
8. Rerun heavy recycling profiles after each solver switch.

Promotion evidence:

- hydrogen and D/T/He fixed-BDF2 gates pass with bounded residual, no failed
  linear solves, and no hidden fallback;
- adaptive-BDF hydrogen gate passes accepted-error, convergence, and
  solver-health checks;
- full-output BDF/JVP path matches compatibility solver on the same case;
- cProfile, RSS, JAX trace, residual-call count, Jacobian/JVP timing, and
  linear-solver health are archived;
- differentiability examples pass `jit`, `jvp`, `grad`, batched sensitivity,
  and finite-difference comparison.

### 4. Build Effective Physics Preconditioning

Objective:

Make the JAX-native solver practically useful by reducing Krylov cost and
accepted-trial cost with a physics/block preconditioner, not just backend
switching.

Reference engineering lessons:

- Edge-fluid solvers commonly separate the exact nonlinear residual from a
  cheaper preconditioner that approximates dominant local and parallel
  transport couplings.
- PETSc-style strategies are useful as design patterns: block Jacobi, line
  solves, field-split/Schur approximations, approximate diffusion/advection
  inverses, and solver-health reporting.
- Direct sparse solvers such as SuperLU-style and MUMPS-style factorizations
  are useful compatibility baselines and small-case diagnostics, but they are
  not the long-term accelerator path unless the assembled sparse operator is
  cheaper than the current finite-difference Jacobian.
- JAX promotion requires the preconditioner to preserve shape stability, avoid
  host loops in hot paths, and keep memory bounded.

Source-code audit targets before implementation:

1. Inspect the current reference implementation's implicit-solver,
   preconditioner, and field-split usage for the matching recycling and
   neutral cases.
2. Inspect PETSc field-split, block-Jacobi, additive-Schwarz, and Schur
   complement patterns as algorithms to reproduce with JAX-compatible arrays,
   not as a runtime dependency.
3. Inspect MUMPS, SuperLU_DIST, and local SciPy sparse-solve behavior only as
   baselines for robustness and diagnostics.
4. Inspect `sfincs_jax` and other JAX plasma codes for practical patterns:
   static layouts, batched linear solves, JVP actions, compile-cache use,
   and objective differentiation.

Candidate implementation order:

1. Existing baselines: no preconditioner, state scaling, field scaling,
   linearized diagonal, and same-cell local blocks. These are negative or
   neutral evidence so far and remain diagnostic controls.
2. Same-cell block-Jacobi with reuse across nonlinear iterations only if it
   reduces total Krylov cost after build cost.
3. Sparse-JVP materialization control: build only the linearized columns needed
   for the preconditioner, not the full finite-difference Jacobian. The first
   target is the active-field block structure of density, pressure, parallel
   momentum, neutral density/pressure/momentum, and controller scalars.
4. Parallel-line transport preconditioner: approximate each field with mass
   plus parallel advection/diffusion along the open-field direction, using
   batched tridiagonal or banded solves where possible. This is the first
   physically motivated candidate for open-field recycling and divertor cases.
5. Neutral/plasma Schur-style preconditioner: approximate stiff neutral
   diffusion/reaction coupling and plasma pressure/momentum coupling as
   separate blocks with cheap coupling corrections. This is the candidate most
   likely to help detached and high-recycling cases.
6. Sheath/target local block preconditioner: include Bohm/sheath heat-flux,
   zero-current reconstruction, recycling, and target-source coupling in
   target-adjacent local blocks, because the current parity offenders are
   often target-adjacent.
7. 2D/3D FCI transport preconditioner: approximate field-line diffusion and
   perpendicular diffusion with metric-weighted separable blocks for compact
   imported-field kernels.
8. Matrix-free Krylov promotion only after the residual and preconditioner are
   both parity-proven.

Promotion evidence:

- same-case hydrogen and D/T/He comparison against stable default;
- fewer Krylov iterations or fewer residual/JVP calls after accounting for
  preconditioner build time;
- no linear-solver failures or hidden fallbacks;
- no parity degradation;
- memory does not exceed the compatibility baseline;
- CPU/GPU timing and JAX trace show the win is in the promoted kernel, not in
  reduced output work.

### 5. Complete Drift-Reduced Braginskii Model Coverage

Objective:

Expose the physics model surface honestly, with selectable equations,
closures, geometries, and boundary conditions.

Model levels:

- diffusion-only scalar transport;
- 1D open-field plasma fluid;
- electrostatic drift-reduced Braginskii;
- vorticity/potential solve;
- Boussinesq polarization;
- non-Boussinesq polarization;
- selected electromagnetic/Alfven lanes;
- sheath, target, recycling, neutral, and detachment closures;
- open-field, closed-field, tokamak, and stellarator geometry variants.

User-facing selection ladder:

1. Start with diffusion-only scalar transport to teach meshes, inputs,
   outputs, restart, plotting, and convergence.
2. Move to 1D open-field plasma fluids to teach parallel losses, sheath
   boundary conditions, recycling, neutral coupling, and detachment metrics.
3. Move to 2D electrostatic drift-reduced Braginskii to teach curvature drive,
   ExB bracket, vorticity/potential solve, Boussinesq versus non-Boussinesq
   polarization, and turbulent transport diagnostics.
4. Move to selected electromagnetic lanes only through validated Alfven or
   selected-field benchmarks.
5. Move to 3D tokamak or stellarator geometry only after the geometry,
   connection-length, endpoint, and FCI/operator gates pass.
6. For every step, examples should show the smallest useful deck, the relevant
   equations, expected outputs, plotting commands, and capability tier.

Steps:

1. Finish the equation-to-code map for density, pressure/energy, parallel
   momentum, vorticity, potential, ExB bracket, curvature/interchange,
   diamagnetic terms, Braginskii closures, electromagnetic selected-field
   terms, neutral terms, recycling terms, and source terms.
2. For every promoted term, define all symbols, units, normalizations, floors,
   limiters, boundary semantics, and implementation functions in docs.
3. Add limiting-case tests where the term vanishes or reduces analytically.
4. Add MMS or operator-convergence tests for promoted spatial operators.
5. Add paired Boussinesq/non-Boussinesq comparisons: potential/vorticity
   response, particle/heat flux, symmetry or up-down/asymmetry metrics, and
   runtime cost.
6. Promote electromagnetic selected-field examples through Alfven/selected
   field verification before any electromagnetic turbulence claim.
7. Replace demo-only nonlinear forcing with the real bracket/vorticity path, or
   label it explicitly as a pedagogical reduced closure.

Promotion evidence:

- equation, implementation link, and test for each promoted term;
- Boussinesq/non-Boussinesq plot generated from a documented script;
- vorticity/potential and bracket gates with physical diagnostics;
- selected electromagnetic figure and validation report;
- no README/docs claim that a reduced surrogate is full DRB physics.

### 6. Finish Neutral, Recycling, Sheath, And Detachment Physics

Objective:

Make neutral and target physics detailed enough for diverted tokamak and
stellarator open-field studies, including detached regimes where enabled.

Physics scope:

- neutral density, pressure, and momentum where enabled;
- neutral parallel and perpendicular diffusion;
- ionization, recombination, charge exchange, and radiation;
- AMJUEL/OpenADAS or documented rate models;
- neutral floors, pressure floors, flux limits, and limiter preparation;
- target recycling, reflection, pumping, fast and thermal recycling;
- no-flow, zero-current, sheath heat-transmission, and target boundary
  reconstruction;
- source, radiation, and target-temperature detachment metrics.

Required model documentation:

- Density equations must define advective, diffusive, ionization,
  recombination, recycling, pumping, and imposed-source terms separately.
- Pressure or energy equations must define conductive, collisional,
  atomic/radiation, sheath, recycling, and control/source terms separately.
- Momentum equations must define pressure-gradient, inertia, viscosity,
  neutral drag/charge-exchange, boundary, and target-source terms separately.
- Every floor, limiter, guard-cell reconstruction, flux cap, target condition,
  rate coefficient, and normalization must be documented with code links.
- Reduced or approximate neutral models must state what Hermès-style/BOUT-style
  term is omitted or simplified and why the resulting case is still useful.

Steps:

1. Close neutral `NVh` parity using accepted-step state/history sequencing.
2. Add direct tests for neutral diffusion, pressure-gradient, viscosity,
   reaction, boundary, and target-source terms.
3. Add target recycling accounting tests for particle and energy balance.
4. Add sheath heat-transmission tests, no-flow guard tests, and zero-current
   ion-sum reconstruction tests.
5. Add detachment scan campaigns with target temperature, ionization-front
   position, pressure loss, radiation/source balance, neutral buildup, and
   recycling coefficient sensitivity.
6. Generate publication-ready neutral/source/target plots from the same scripts
   used by the validation campaigns.
7. Add clean-clone neutral/recycling tutorial outputs: source maps, target
   fluxes, neutral-density lineouts, radiation/source partitions, and a short
   explanation of detached versus attached behavior.

Promotion evidence:

- component-level parity for implemented neutral/recycling/sheath terms;
- detachment plots with physically interpretable monotonicity or bounded
  response;
- docs with full equations, derivations, normalizations, and code links;
- tests are physics/numerics tests, not placeholder smoke tests.

### 7. Make Diverted Tokamak Simulations Self-Contained

Objective:

Users should be able to clone JAXDRB, install it, run the diverted tokamak
examples, create movies, and analyze turbulent profiles without installing
external reference codes.

Steps:

1. Provide clean-clone tutorials that either generate small native fixtures or
   fetch release-hosted fixtures.
2. Keep large `tokamak.nc`, `BOUT.dmp*.nc`, movies, and profile bundles out of
   git history. Store them as release assets or external benchmark artifacts
   with stable downloader scripts.
3. For the current private repository, the fetch path must support authenticated
   GitHub release downloads through `gh` or `GITHUB_TOKEN`. For any future
   public release, the same commands should work without private reference-code
   access.
4. Promote a tokamak example ladder:
   1D recycling, 1D detachment scan, 2D direct tokamak diffusion/transport,
   2D recycling, multispecies D/T/He, impurity/neon radiation, detached target
   scan, then longer nonlinear turbulence windows.
5. Generate movies and profile plots from the same documented scripts:
   OMP profiles, target profiles, heat flux, particle flux, source maps,
   neutral maps, target temperature, detachment indicators, and turbulence
   fluctuation metrics.
6. Add runtime progress, ETA, restart, output, and provenance examples for the
   tokamak tutorials.
7. Keep reference-code comparison workflows available to developers, but never
   make them required for ordinary user examples.
8. Add a "how to regenerate the README tokamak movie" recipe that starts from
   a clean clone, fetches or creates required fixtures, runs the simulation or
   documented reduced campaign, and writes the same GIF/PNG profile outputs.

Promotion evidence:

- clean-clone tokamak tutorial passes without private paths;
- advertised README movie and plots are reproducible from example commands;
- large fixtures are absent from git and package distributions;
- profile and target plots are validated against reference-backed or
  literature-anchored diagnostics.

### 8. Promote 3D Stellarator Geometry And Open/Closed Field-Line SOL

Objective:

Build the first reviewer-proof 3D stellarator SOL examples, with open and
closed field-line regions, sheath/recycling/neutrals where physically
applicable, and polished non-axisymmetric figures/movies only after geometry
and operator gates pass.

Geometry sources:

- VMEC-coordinate field-line maps.
- VMEC-extender exterior grids from the current importer contract and upstream
  export path.
- ESSOS-imported coil maps, including `ESSOS_biot_savart_LandremanPaulQA.json`.
- Hybrid VMEC/coil maps, where VMEC gives smooth coordinate surfaces and coil
  maps supply open-field endpoint behavior.
- HSX QHS vacuum equilibrium from
  [landreman/vmec_equilibria HSX/QHS_vac](https://github.com/landreman/vmec_equilibria/tree/master/HSX/QHS_vac).
- NCSX VMEC examples with explicit source metadata.
- Landreman-Paul QA with ESSOS coils and VMEC references.
- Dommaschk potentials using the BSTING/Zoidberg workflow as an implementation
  and visualization reference.

Source anchors:

- [BSTING workflow files](https://github.com/rogeriojorge/bsting_files).
- Optional local BSTING checkout: set `BSTING_FILES_ROOT` when present.
- [Hermes-3 reference commit](https://github.com/boutproject/hermes-3/tree/eebf98fd18198101bebe7cdb5c85f25dc1ff3474).
- [Zoidberg reference commit](https://github.com/rogeriojorge/zoidberg/tree/a7ed260123508c35939002d96412c0dd84491fe4).
- [ESSOS](https://github.com/uwplasma/ESSOS).

Promotion ladder:

1. Geometry import contract: metadata, coordinate conventions, field period,
   boundary source, units, and provenance.
2. Boundary/surface plots: VMEC surface, coil-field trace surface, open-field
   shell, and rendered device boundary.
3. Poincare and field-line validation against the imported source.
4. Connection-length validation with clear definitions:
   one-sided connection length, target-to-target connection length, and
   effective parallel step length for closed maps.
5. Endpoint-mask and sheath-target map convergence.
6. FCI map interpolation and metric-weighted conservative operator MMS.
7. Open/closed-field linear tests.
8. Sheath/recycling/neutral gates on imported endpoints.
9. Nonlinear turbulent transient with grid and timestep refinement.
10. Polished movie and profile analysis.

Required devices/examples:

- Landreman-Paul QA, using ESSOS coil, VMEC, and hybrid maps.
- HSX QHS vacuum, using VMEC and Boozer/source metadata where available.
- NCSX, using VMEC-coordinate and optional exterior-field lanes.
- Dommaschk potentials, following the BSTING/Zoidberg pattern for FCI maps and
  stellarator SOL visualization.

Per-device execution sequence:

1. Record the source artifact, field periods, coordinate conventions, units,
   boundary/wall source, and whether the map is closed, open, or hybrid.
2. Render the imported boundary and field-line context before any plasma
   transient.
3. Run field-line/Poincare and connection-length diagnostics at multiple
   resolutions. For closed VMEC maps, report periodic parallel-step metrics
   instead of open target-to-target connection length.
4. Build endpoint masks and wall/target maps for open or hybrid lanes.
5. Run FCI interpolation, parallel-gradient, conservative diffusion, and metric
   MMS gates.
6. Run reduced linear dynamics on the validated map and compare mode
   propagation, damping, and conservation to expectations.
7. Add sheath/recycling/neutral gates only after endpoint masks are validated.
8. Run nonlinear transients with grid and timestep refinement. Movies are
   promoted only after frame-by-frame QA shows non-axisymmetry, smooth
   dynamics, correct boundary shape, visible open/closed structure, colorbar,
   time annotation, and no jitter.
9. Move polished media to release assets and keep only small source artifacts,
   thumbnails, or scripts in git.

BSTING and Zoidberg files are implementation and visualization references.
They define the expected FCI workflow, figure types, and movie quality bar, but
JAXDRB must own its imported arrays, operator validation, physics residuals,
diagnostics, examples, and claim boundaries.

Promotion evidence:

- boundary/surface plot;
- Poincare or field-line plot;
- connection-length or parallel-step map;
- endpoint/sheath mask map;
- FCI operator convergence plot;
- one linear or reduced nonlinear physics output figure;
- grid/time refinement report before any turbulence movie is promoted;
- frame-by-frame movie QA showing smooth dynamics, visible non-axisymmetry,
  correct boundary shape, informative colorbar, time annotation, and no jitter.

### 9. Validate Open And Closed Field-Line Physics

Objective:

Make open/closed-field-line claims precise, especially for stellarators where
closed VMEC surfaces and open SOL/target maps have different mathematical
semantics.

Steps:

1. Closed-field tests:
   parallel-gradient periodicity, effective parallel-step convergence,
   closed-surface drift-wave or reduced linear mode, and conserved/controlled
   quantities.
2. Open-field tests:
   endpoint masks, target-to-target connection length, one-sided connection
   length, sheath response, recycling response, neutral source localization,
   and target heat/particle flux.
3. Hybrid tests:
   compare VMEC-coordinate closed-map locations with coil-derived open-field
   endpoint behavior and document where the hybrid approximation is valid.
4. Profile diagnostics:
   flux-surface or surface-shell averages, radial profiles, fluctuation levels,
   skewness, spectra, radial particle/heat flux, neutral/source maps, and
   target localization.

Promotion evidence:

- separate closed-field and open-field validation reports;
- no closed-field connection-length quantity used as an open-field target claim;
- stellarator profiles and movies linked to field-line/endpoint evidence;
- literature-style plots, not only rendered surfaces.

### 10. Performance, Parallelization, GPU, And Differentiability

Objective:

Show real speedups and differentiability on promoted kernels without hiding
behind toy examples.

CPU plan:

- Use Mac CPU cores for batched/ensemble fixed-work solves, batched JVPs,
  parameter scans, UQ, finite-difference comparisons, and optimization
  batches.
- Keep single SciPy-BDF solve threading as compatibility evidence only; it is
  not the main strong-scaling story.
- Use `vmap` for single-device batch throughput and explicit host-device count
  experiments for local CPU-device scaling.

GPU plan:

- Start with compact JAX-native residuals where compilation and device
  execution dominate host orchestration.
- Move to full-output recycling only after the residual is transformable and
  host/SciPy barriers are reduced.
- Capture persistent compilation cache status, compile time, execution time,
  memory, and CPU/GPU comparison.
- Multi-GPU should focus on naturally sharded ensembles, parameter scans, or
  domain slabs after single-GPU evidence is credible.

Differentiability plan:

- Promote only lanes with pure-JAX objectives.
- Add `grad`, `jvp`, VJP/adjoint or implicit-function tests where valid.
- Compare derivatives with finite differences on well-conditioned objectives.
- Provide UQ, inverse design/control, sensitivity maps, and optimization
  examples.
- State clearly when a path is not end-to-end differentiable because it uses
  host-side SciPy, file I/O, reference replay, or non-transformable callbacks.

Promotion evidence:

- same-machine timing before/after each solver change;
- cProfile/RSS/JAX trace/XLA or kernel timing for heavy paths;
- CPU strong-scaling figure on real kernels or real ensembles;
- single-GPU and multi-GPU bundle only after the JAX-native kernel is stable;
- derivative plots with finite-difference checks and error curves;
- docs explain compile time, execution time, memory, and limitations.

### 11. Refactor Code Architecture Without Changing Physics

Objective:

Split large files into maintainable modules while preserving parity and public
API behavior.

Target source layout:

```text
src/jax_drb/native/
  recycling/
    state.py
    layout.py
    operators.py
    closures.py
    reactions.py
    collisions.py
    neutral_diffusion.py
    target_sources.py
    boundaries.py
    residual.py
    stepping.py
    diagnostics.py
  neutral/
    state.py
    operators.py
    closures.py
    boundaries.py
    residual.py
    diagnostics.py
  tokamak/
    metrics.py
    mapping.py
    transport.py
    lineouts.py
  stellarator/
    vmec.py
    vmec_extender.py
    essos_import.py
    dommaschk.py
    fci.py
    connection_length.py
    diagnostics.py
  runner/
    registry.py
    references.py
    cache.py
    execution.py
    comparison.py
    artifacts.py
```

Steps:

1. Extract behavior-preserving helpers from `recycling_1d.py`,
   `neutral_mixed.py`, `runner.py`, and `cli.py`.
2. Preserve compatibility shims while moving call sites.
3. Split large tests into operator, parity, campaign, CLI, artifact, and slow
   transient files.
4. Add direct tests for each extracted module before relying on broad
   integration tests.
5. Add docstrings for public APIs and comments only where the algorithm is not
   self-explanatory.
6. Keep physics behavior unchanged unless a parity-backed bug fix is explicitly
   documented.

Promotion evidence:

- no broad active source module remains above about 1000 lines without a
  documented reason;
- extracted modules have direct tests and docs links;
- promoted parity/coverage gates remain green after each extraction;
- no user-facing import breaks without a compatibility alias and release note.

### 12. Build Documentation And Examples As The User Interface

Objective:

Make docs and examples complete enough that users do not need the paper or
private context to run and understand the code.

Documentation requirements:

- installation and quick start;
- ReadTheDocs configuration and docs build instructions;
- model selection guide;
- equations, derivations, normalizations, and symbols;
- equation-to-code map with source links;
- input deck schema and examples;
- output fields, units, diagnostics, provenance, restart, and ETA;
- validation strategy and capability tiers;
- validation gallery and artifact links;
- performance, profiling, CPU/GPU, and differentiability guide;
- geometry guide for tokamak, VMEC, VMEC-extender, ESSOS, hybrid, HSX, NCSX,
  Landreman-Paul QA, and Dommaschk lanes;
- neutral/recycling/sheath/detachment guide;
- testing and contribution guide;
- release and PyPI publishing guide.

Example requirements:

- User-facing examples follow SIMSOPT-style scripts:
  parameters at the top, imported JAXDRB API functions, small auxiliary
  functions only, explicit output paths, and no hidden monolithic driver.
- Every README figure/movie has a corresponding documented example command.
- Extended validation plots live in docs galleries instead of overflowing the
  README.
- Examples are grouped as tutorials, benchmarks, validation campaigns,
  performance, differentiability, and geometry.
- Examples should not require users to install reference codes. Developer
  reference-comparison scripts may remain available separately.

Promotion evidence:

- `PYTHONPATH=src mkdocs build --strict --clean`;
- clean-clone tokamak and stellarator tutorials regenerate advertised plots and
  movies from native outputs or release-hosted fixtures;
- README contains a small polished set of visuals, not the full validation
  archive;
- docs state capability tier, limitations, and reproducibility commands for
  each promoted figure.

### 13. Keep Coverage Above 95 Percent With Meaningful Tests

Objective:

Maintain high coverage by testing physics, numerics, autodiff, parity, and
failure behavior, not by adding placeholder smoke tests.

Required test classes:

- unit tests for equations, closures, limiters, floors, and boundary formulas;
- operator tests for gradients, divergence, diffusion, FCI maps, interpolation,
  vorticity/potential, sheath, recycling, and neutral terms;
- MMS convergence tests for promoted operators;
- Jacobian/JVP/VJP/gradient tests against finite differences;
- pack/unpack, layout, restart, artifact, and provenance tests;
- reference-backed parity tests for promoted component surfaces;
- CLI dispatch, failure reporting, and clean-clone example tests;
- coverage audits for promoted solver surfaces.

Promotion evidence:

- `scripts/run_promoted_solver_coverage.py` reports at least `95%`;
- `scripts/run_closeout_coverage.py` remains green;
- new coverage corresponds to real physics/numerical surfaces;
- any skipped slow/live tests have a documented manual command and artifact.

### 14. Keep The Repository Lightweight

Objective:

Make `git clone` and `pip install` fast while preserving reproducibility.

Rules:

- Keep source, docs, tests, small fixtures, and small rendered figures in git
  only when they are necessary and small.
- Keep large movies, large NetCDF dumps, heavy reference baselines, JAX traces,
  profile bundles, and manuscript-only artifacts out of git history.
- Use GitHub releases or external artifact storage for heavy assets.
- Provide downloader/restoration scripts for examples and docs assets.
- Keep paper-only material in the paper repository, not the code repository.

Pre-tag footprint audit:

```bash
du -sh .git
git count-objects -vH
git ls-files -z | xargs -0 du -h | sort -hr | head -40
python -m build
tar -tf dist/*.tar.gz | sort | sed -n '1,120p'
python -m zipfile -l dist/*.whl | sed -n '1,120p'
```

Promotion evidence:

- no tracked large simulation dump or trace;
- no accidental media/profile/cache artifact in wheel or sdist;
- release-hosted media links work;
- no history rewrite needed unless a new large blob enters git history.

### 15. Final Release, Tag, And Publication Package

Objective:

Ship a version only after the code, docs, tests, examples, performance evidence,
and validation claims line up.

Steps:

1. Run local release gates and docs build.
2. Run representative clean-clone tutorials for tokamak and stellarator.
3. Run the latest promoted solver coverage and closeout coverage.
4. Run the latest offender register, parity reports, and selected validation
   campaigns.
5. Run heavy CPU/GPU performance bundles when the promoted solver kernel
   changed.
6. Run footprint audit and package audit.
7. Refresh README, docs, release notes, changelog, and examples matrix.
8. Decide which experimental lanes remain opt-in and label them.
9. Bump version.
10. Tag release.
11. Create GitHub release with notes and release-hosted validation assets.
12. Let PyPI workflow publish once hosted CI can run again.

Release cannot claim:

- end-to-end differentiability for paths still using host-side SciPy solves;
- broad 3D stellarator turbulence for movie-only reduced transients;
- GPU or multi-GPU speedups without real-kernel evidence;
- reference parity for components without term-level diagnostics or bounded
  error reports.

## Literature And Code Baseline

The following references define the validation, architecture, and writing
standard for the project. The plan uses them as anchors for tests, plots, and
documentation style. JAXDRB should not copy any code blindly; it should reuse
validated ideas through its own APIs, tests, artifacts, and documentation.

### Edge/SOL Fluid And Braginskii Codes

- BOUT++: Dudson et al. 2009,
  [BOUT++: a framework for parallel plasma fluid simulations](https://arxiv.org/abs/0810.5757),
  [source](https://github.com/boutproject/BOUT-dev),
  [physics models](https://bout-dev.readthedocs.io/en/latest/user_docs/physics_models.html).
- Hermes-3: Dudson et al. 2024,
  [CPC paper](https://www.sciencedirect.com/science/article/pii/S0010465523003363),
  [preprint](https://arxiv.org/abs/2303.12131),
  [source](https://github.com/boutproject/hermes-3),
  [equations](https://hermes3.readthedocs.io/en/latest/equations.html),
  [closures](https://hermes3.readthedocs.io/en/latest/closure.html),
  [reactions](https://hermes3.readthedocs.io/en/latest/reactions.html),
  [boundary conditions](https://hermes3.readthedocs.io/en/latest/boundary_conditions.html).
- GBS:
  [code paper](https://www.sciencedirect.com/science/article/pii/S0021999116001923),
  [neutral extension](https://arxiv.org/abs/2112.03573).
- TOKAM3X:
  [diverted edge/SOL turbulence paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999116301838).
- GRILLIX:
  [CPC paper](https://www.sciencedirect.com/science/article/pii/S0010465525003765).
- GDB:
  [CPC paper](https://www.sciencedirect.com/science/article/abs/pii/S001046551830208X).

Lessons for JAXDRB:

- Separate physics components, geometry, solver, diagnostics, and
  postprocessing.
- Validate equation terms, not only final fields.
- Include convergence, target profiles, source maps, runtime, and scaling
  evidence.
- Keep complicated boundary and neutral models documented at equation level.

### Diverted Tokamak And Detachment Benchmarks

- TCV-X21 benchmark:
  [preprint](https://arxiv.org/abs/2109.01618),
  [FAIR dataset](https://zenodo.org/records/5776286).
- SOLPS-ITER TCV-X21 validation:
  [preprint](https://arxiv.org/abs/2310.17390).
- Hermes-3 TCV-X21 validation:
  [preprint](https://arxiv.org/abs/2506.12180).

Lessons for JAXDRB:

- Target-region observables are central: target temperature, target density,
  heat flux, particle flux, pressure loss, neutral density, source location,
  and radiation/source partition.
- Agreement should be reported as profiles and physical metrics, not only
  scalar norms.
- Neutral and sheath assumptions must be explicit because they strongly affect
  target agreement.

### Stellarator SOL, FCI, VMEC, Coils, And Dommaschk Geometry

- BSTING stellarator-filament paper:
  [preprint](https://arxiv.org/abs/1808.08899).
- BSTING files:
  [rogeriojorge/bsting_files](https://github.com/rogeriojorge/bsting_files).
- Zoidberg reference workflow:
  [specific branch/commit](https://github.com/rogeriojorge/zoidberg/tree/a7ed260123508c35939002d96412c0dd84491fe4).
- HSX QHS vacuum VMEC equilibrium:
  [Landreman equilibrium repository](https://github.com/landreman/vmec_equilibria/tree/master/HSX/QHS_vac).
- ESSOS:
  [uwplasma/ESSOS](https://github.com/uwplasma/ESSOS).
- Recent stellarator SOL fluid literature:
  [Coelho et al. 2022](https://arxiv.org/abs/2201.10871),
  [Shanahan et al. 2024](https://arxiv.org/abs/2403.18220),
  [JPP stellarator island-divertor turbulence article](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/global-fluid-turbulence-simulations-in-the-scrapeoff-layer-of-a-stellarator-island-divertor/BA86AE2B67AE1F224800F2A0BB7193C1).

Lessons for JAXDRB:

- A non-axisymmetric movie is credible only after geometry and operator
  validation.
- Report boundary surfaces, Poincare plots, connection length, endpoint masks,
  target localization, profile statistics, fluctuation statistics, and
  grid/time refinement.
- Keep field-line tracing and coil-field evaluation owned by imported tools
  where appropriate, while JAXDRB owns imported arrays, FCI maps, equations,
  solver execution, diagnostics, and documentation.

### Differentiable Scientific Computing

- JAX JVP/VJP docs:
  [jacobian-vector-products](https://docs.jax.dev/en/latest/jacobian-vector-products.html).
- JAX persistent compilation cache:
  [docs](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).
- JAX sharded computation:
  [docs](https://docs.jax.dev/en/latest/sharded-computation.html).
- JAX `shard_map`:
  [notebook](https://docs.jax.dev/en/latest/notebooks/shard_map.html).
- Lineax matrix-free solves:
  [no materialisation example](https://docs.kidger.site/lineax/examples/no_materialisation/).
- Lineax operators:
  [API](https://docs.kidger.site/lineax/api/operators/).
- Diffrax adjoints:
  [API](https://docs.kidger.site/diffrax/api/adjoints/).
- Equinox:
  [paper](https://arxiv.org/abs/2111.00254).
- Lineax:
  [paper](https://arxiv.org/abs/2311.17283).
- JAX-Fluids:
  [CPC paper](https://www.sciencedirect.com/science/article/pii/S0010465522002466).
- Algorithmic differentiation for plasma edge codes:
  [JCP paper](https://www.sciencedirect.com/science/article/pii/S0021999123004989).

Lessons for JAXDRB:

- The differentiability target is a pure-JAX residual and objective path.
- `jax.jvp`, `jax.vjp`, `vmap`, `jit`, `shard_map`, and matrix-free linear
  operators should be used where they reduce host barriers and preserve
  validation.
- Persistent compilation cache and explicit compile/execution timing should be
  documented in performance examples.
- Forward-mode, reverse-mode, and implicit-function sensitivities should be
  tied to concrete scientific objectives.

## Evidence Matrix

Each promoted feature should carry the following evidence:

| Feature class | Required evidence |
| --- | --- |
| Equation term | Equation, symbol definitions, implementation link, unit/operator test, limiting-case test. |
| Spatial operator | MMS or analytic convergence, metric/coordinate tests, boundary tests. |
| Time integration | Timestep refinement, restart equivalence, history/state sequencing tests. |
| Reference parity | One-RHS, one-step, short-window or accepted-step trace, absolute/relative errors, component owner. |
| Neutral/recycling/sheath | Source accounting, boundary reconstruction, target flux, heat transmission, detachment metrics. |
| Differentiability | `jit`, `jvp`, `grad`, VJP/implicit sensitivity where valid, finite-difference comparison. |
| Performance | cProfile/RSS/JAX trace, compile/execute split, memory, same-machine baseline, CPU/GPU evidence. |
| Geometry | Boundary plot, Poincare/field-line plot, connection length, endpoint mask, FCI convergence. |
| User example | Clean-clone command, input/output docs, generated plot/movie, capability tier, no private dependency. |
| Release | Coverage, docs build, footprint audit, wheel/sdist audit, release notes, tagged assets. |

## Execution Log

Use this log for concise decision records. Do not paste terminal output here.

- 2026-06-18: Closed the neutral-mixed accepted-step `NVh` local source
  offender by making the native `SNVh_parallel_inertia` term use the
  reference-matching `Div_par_fvv(..., fix_flux=False)` Lax boundary mode. A
  reference-state RHS rerun on the existing `309` accepted-step trace drops
  `SNVh_parallel_inertia` from `1.42e-4` to `1.11e-12` target-adjacent
  pointwise error, while pressure-gradient, perpendicular diffusion, viscosity,
  diffusion-limiter, velocity, and `Grad(logPnlimh)_*` remain at roundoff. The
  parity report now includes a `parallel_inertia_flux_variant_register` that
  ranks `Div_par_fvv_fix_flux_false` best and `Div_par_fvv_fix_flux_true`
  worst on exact reference states. Focused native/operator/validation tests
  pass (`74 passed`) and `compileall` is clean. The next active lanes are P2
  JAX-native recycling residual promotion and P3 preconditioning/performance,
  with P1 retained as a regression discipline for future reference runs.
- 2026-06-18: Revisited the roadmap/status Markdown pages after billing was
  exhausted and CI polling was explicitly deprioritized. The plan now has a
  single `Current Implementation Backlog` with priorities `P0` through `P9`,
  covering plan authority, neutral parity, JAX-native recycling,
  preconditioning, drift-reduced Braginskii coverage, neutral/recycling/sheath
  physics, self-contained diverted tokamak examples, 3D stellarator SOL
  geometry, performance/differentiability, docs/examples/coverage/release, and
  repo-footprint controls. Roadmap-like docs were audited and status/example
  pages now carry explicit authority notes pointing back to this file, so old
  campaign-local "next step" text cannot override the master plan.
- 2026-06-18: Rechecked CI after commit `9a068ca`. Hosted `test`,
  `coverage`, and `docs` failed before runner assignment: each job completed in
  a few seconds with an empty step list, empty runner name, no downloadable log,
  and no GitHub job failure message. That is account/runner startup evidence,
  not a repository-code failure. Local workflow equivalents are green on the
  current worktree: Python `3.10`, `3.11`, and `3.12` each pass the 43-test
  hosted test slice in clean throwaway venvs; `mkdocs build --strict --clean`
  passes; closeout coverage is `97.0%`; and promoted solver coverage is
  `95.12%` (`565 passed`, `14 skipped`, `1 xfailed`). The repo is ready for the
  next hosted CI rerun when Actions runners can start.
- 2026-06-18: Extended the neutral-mixed accepted-step reference monitor with
  direct `SNVh_parallel_inertia` and `SNVh_perpendicular_diffusion` components.
  A patched clean reference build and max-order-2 accepted-step rerun produced
  `309` matched accepted records. Evaluating the native RHS on exact reference
  accepted states closes pressure-gradient, perpendicular diffusion,
  viscosity, diffusion-limiter, and `Grad(logPnlimh)_*` terms to roundoff, but
  ranks `SNVh_parallel_inertia` first at about `1.42e-4` target-pointwise. This
  historical blocker was later closed by the explicit `fix_flux=False`
  `Div_par_fvv` momentum-flux mode recorded above; accepted-state/history
  sequencing remains a broader transient-monitoring concern, not the active
  local-source explanation for this trace.
- 2026-06-18: Finished the repo-side CI triage after the hosted Actions runs
  for `test`, `docs`, and `coverage` failed before executing any steps and
  exposed no job logs. The local CI-equivalent gates are green on this
  worktree: docs build, release-surface tests, workflow test slice, fast
  research checks, closeout coverage at `97.0%`, and promoted solver coverage
  at `95.12%`. The only actionable local failure was a stale
  neutral-mixed test expectation after adding `git apply --recount`; the test
  now verifies the recount-aware forward and reverse patch checks. Hosted
  Actions should be rechecked periodically, but the current zero-step failures
  remain runner/account-startup evidence rather than repo-code evidence.
- 2026-06-18: Rechecked the latest hosted `test`, `coverage`, and `docs`
  Actions runs for commit `3edf668`. Every failed job still has an empty step
  list and the same GitHub annotation: the job was not started because recent
  account payments failed or the spending limit must be increased. This closes
  the actionable CI lane for now; no repo-side CI failure is available to fix
  until GitHub runners can start again.
- 2026-06-18: Tightened the neutral-mixed accepted-step reference monitor patch
  by setting 17-digit precision inside the nested JSON helper streams. The
  high-precision rerun keeps `309/309` matched accepted steps and zero
  solver-order mismatches, and it moves the reference-state residual diagnostic
  away from density-roundoff noise to a small target-adjacent `NVh` residual
  (`2.76e-6`). The dominant parity lane remains accepted-state/history
  preparation feeding `Grad(logPnlimh)` and the `Dnnh_flux_max` limiter ladder.
- 2026-06-18: Added a `reference_active_state_rhs_register` to the accepted-step
  parity report. It reconstructs each reference accepted state and evaluates the
  native neutral RHS/source/preboundary payloads on that state, separating local
  operator algebra from time-discretization history. On the high-precision
  max-order-2 trace the source/preboundary target-adjacent mismatch closes to
  roundoff (`1.42e-14`), including `grad_logPnlimh_y`, `Dnnh_flux_max`, `Dnnh`,
  `eta_h`, and the `SNVh_*` terms. The all-field RHS-on-reference ranking is
  dominated by `ddt(NVh)` at about `1.42e-4`; the later component-split and
  flux-variant register localized and closed this as the neutral
  `SNVh_parallel_inertia` boundary-flux-mode issue, rather than a
  pressure-gradient, viscosity, or diffusion-limit formula issue.
- 2026-06-18: Refined the master plan into an execution checklist with
  explicit working rules, lane completion criteria, milestone map, and immediate
  next work package. Confirmed that `plan_jax_drb.md` is only a redirect and
  that the older refactoring, geometry, non-axisymmetric, parity, runtime, and
  research-direction pages are subordinate appendices rather than competing
  plans. This was a plan-only pass and preserved pre-existing uncommitted
  source/test diagnostics for the neutral-mixed lane.
- 2026-06-18: Refactored this master plan around the current finish-line goals
  before further implementation: effective preconditioning, self-contained
  diverted tokamak simulations, complete drift-reduced Braginskii model
  coverage, JAX-native recycling, sheath/recycling/neutral/detachment physics,
  Boussinesq and non-Boussinesq comparisons, electromagnetic selected-field
  lanes, open and closed field-line simulations, VMEC/VMEC-extender/coil/hybrid
  stellarator geometry, HSX/NCSX/Landreman-Paul QA/Dommaschk examples,
  reference-backed parity, release-hosted heavy assets, full documentation,
  `95%` meaningful coverage, CPU/GPU performance, and lightweight packaging.
  Added a cross-document conflict policy, a target capability matrix,
  device-by-device stellarator validation sequence, stronger neutral-model
  documentation requirements, and a more concrete preconditioning audit and
  implementation ladder. No tests, solver changes, or validation campaigns
  were started during this plan-only pass.
- 2026-06-18: Consolidated the plan into this single authoritative file and
  reorganized it around ordered workstreams: plan authority, reference-backed
  parity, JAX-native recycling, effective preconditioning, drift-reduced
  Braginskii models, neutral/recycling/sheath/detachment physics, diverted
  tokamak tutorials, 3D stellarator geometry, open/closed field-line physics,
  performance/differentiability, code architecture, docs/examples, meaningful
  coverage, repository footprint, and release. The plan explicitly includes
  VMEC, VMEC-extender, ESSOS coil maps, hybrid maps, HSX, NCSX,
  Landreman-Paul QA, Dommaschk potentials, BSTING/Zoidberg workflow anchors,
  self-contained examples, release-hosted heavy assets, and no hosted-CI
  polling while runner billing is blocked.
- 2026-06-18: Finished the sparse Newton linear-solver health reporting patch
  used by adaptive-BDF sparse-JVP recycling gates. Direct sparse solves now
  report `scipy_spsolve`, GMRES reports `scipy_gmres`, failed GMRES with direct
  fallback reports `scipy_gmres_spsolve_fallback`, and immediate convergence
  with zero linear iterations is no longer counted as unknown linear-solver
  status. The bounded `recycling_1d_one_step` sparse-JVP adaptive-BDF gate
  (`timestep=0.25`, `steps=1`, `max_nonlinear_iterations=3`) now reports `24`
  sparse-JVP solver steps, `0` failed linear solves, and `0` unknown
  linear-solver steps. The gate still takes about `16.45 s` and remains
  dominated by JVP Jacobian assembly, so this is solver-health evidence rather
  than a speedup claim.
- 2026-06-18: Batched the exact JVP-derived `parallel_line` preconditioner
  build across multiple field-line blocks with a bounded
  `max_batch_unknowns` control. This improves the infrastructure for future
  2D/3D transport preconditioners, but the current 1D hydrogen gate remains
  negative default-promotion evidence: with residual JIT, skipped initial
  residual check, and batched JAX GMRES, the same-worktree medians were
  `3.07 s` unpreconditioned, `3.79 s` with `parallel_line`, and `3.76 s` with
  `parallel_line` plus preconditioner reuse, all with the same residual band
  and full `800` GMRES update budget. The next performance step remains
  reducing residual/JVP cost or developing a Schur/transport preconditioner
  that measurably reduces iteration count.
- 2026-06-18: Added explicit preconditioner-evidence gates to the recycling
  promotion scripts. `scripts/run_recycling_jvp_promotion_gate.py` can now pass
  `--fixed-bdf2-linear-preconditioner=<name>`, which forwards the matching
  runtime override and requires `fixed_bdf2_linear_preconditioner=<name>` plus
  a positive build count in `scripts/compare_recycling_transient_modes.py`.
  The lower-level compare script also supports adaptive-BDF required
  preconditioner diagnostics. Focused verification:
  `PYTHONPATH=src pytest -q tests/test_compare_recycling_transient_modes.py
  tests/test_recycling_jvp_promotion_gate.py` passed with `46` tests. This does
  not claim a speedup; it closes a promotion-gate gap by ensuring future
  preconditioner campaigns cannot silently fall back to an unpreconditioned
  path.
- 2026-06-18: Ran the new required-preconditioner gate on the lightweight
  `recycling_1d_one_step` fixed-BDF2 fixture with `local_block_diag`,
  `timestep=10`, and `steps=2`. The fixed-full-field route passed with
  residual `1.90e-6`, `9` preconditioner builds, `0` failed linear solves, and
  `22.7 s`; the active-array route passed with the same residual/build count,
  `0` failed linear solves, and `30.3 s`. Both routes still consumed the full
  `3600` JAX-GMRES update budget, so this is promotion-gate correctness
  evidence rather than performance-promotion evidence.
- 2026-06-18: Added
  `--fixed-bdf2-linear-preconditioner-refresh=<n>` to
  `scripts/run_recycling_jvp_promotion_gate.py` so the bounded fixed-BDF2 gate
  can reproduce preconditioner reuse without manual overrides. On the same
  `recycling_1d_one_step` gate, `local_block_diag` with refresh `100` reduced
  preconditioner builds from `9` to `2`, improved fixed-full-field time from
  `22.7 s` to `18.9 s`, improved active-array time from `30.3 s` to `23.4 s`,
  preserved zero failed linear solves, and kept the residual below `1e-5`.
  Because the JAX-GMRES update budget remains saturated, this stays opt-in and
  the next real performance target is reducing Krylov iterations through a
  stronger Schur/transport preconditioner or cheaper residual/JVP kernels.
- 2026-06-18: Added optional fixed-BDF2 performance-promotion gates for
  total JAX-GMRES budget and dynamic-preconditioner build count. The lower-level
  compare script now accepts
  `--require-fixed-bdf2-max-linear-iterations=<n>` and
  `--require-fixed-bdf2-max-preconditioner-builds=<n>`; the promotion wrapper
  forwards them as `--fixed-bdf2-max-linear-iterations=<n>` and
  `--fixed-bdf2-max-preconditioner-builds=<n>`. These gates intentionally
  separate solver-correctness evidence from performance-promotion evidence:
  the current refresh-100 local-block gate is correct and faster than rebuilding
  blocks, but still fails the stricter "reduced Krylov budget" standard until a
  stronger Schur/transport preconditioner or cheaper residual/JVP path lands.
- 2026-06-18: Ran the stricter bounded performance gate on
  `recycling_1d_one_step` with `local_block_diag`, refresh `100`,
  `--require-fixed-bdf2-max-linear-iterations=3200`, and
  `--require-fixed-bdf2-max-preconditioner-builds=2`. Both fixed-full-field and
  active-array fixed-BDF2 routes passed with residual `3.76e-6`, zero failed
  linear solves, exactly `3200` linear iterations, and exactly `2`
  preconditioner builds. Timings were `18.9 s` and `22.5 s`. This promotes
  preconditioner reuse as a bounded opt-in fixture gate, not as heavy-kernel or
  default-solver evidence.
- 2026-06-18: Reran the local promoted-solver coverage slice after the
  recycling budget-gate changes:
  `PYTHONPATH=src python scripts/run_promoted_solver_coverage.py` passed with
  `565` tests, `14` skips, `7` deselected tests, `1` xfail, and total coverage
  `95.12%`, above the required `95.00%`. The largest remaining promoted-slice
  coverage gaps are still `src/jax_drb/native/recycling_1d.py`,
  `src/jax_drb/native/runner.py`, `src/jax_drb/parity/reference.py`, and
  `src/jax_drb/cli.py`; future tests should target real solver, reference,
  runner, and CLI behavior rather than smoke-only coverage.
- 2026-06-18: Promoted the same preconditioner and solver-health budget
  discipline into the real-kernel JAX-linearized recycling profiler.
  `scripts/profile_recycling_jax_linearized_gate.py` now forwards
  `--linear-preconditioner=<name>` and
  `--linear-preconditioner-refresh=<n>` into the profiled solve, writes gate
  requirements and failures into `profile_summary.json`, and returns nonzero
  when `--require-linear-preconditioner=<name>`,
  `--require-max-linear-iterations=<n>`, or
  `--require-max-preconditioner-builds=<n>` are not satisfied. Dynamic
  JVP-derived preconditioners must report finite build diagnostics when
  required. Focused verification:
  `PYTHONPATH=src pytest -q tests/test_profile_recycling_jax_linearized_gate.py
  tests/test_recycling_jvp_promotion_gate.py
  tests/test_compare_recycling_transient_modes.py
  tests/test_research_campaign_bundle.py` passed with `71` tests. This closes
  another heavy-profile gate gap, but it does not by itself prove a D/T/He
  speedup; the next required evidence is a reference-backed CPU/GPU run using
  these gates.
- 2026-06-18: Ran a reference-backed local D/T/He JAX-linearized profile with
  the new preconditioner gates. The default tiny-timestep deck exited before a
  Newton update (`nonlinear_iterations=0`, residual `2.41e-11`), which
  correctly failed the required dynamic-preconditioner-build gate. Rerunning at
  `timestep=1.0` exercised the heavy path. The unpreconditioned control passed
  in `7.59 s` with residual `7.315`, clean JAX-GMRES status, and the full
  `400` update budget. `local_block_diag` with refresh `100` also passed
  correctness gates with the same residual and one preconditioner build, but it
  took `29.84 s`, spent `1.65 s` building the preconditioner, and still used
  the full `400` update budget. This is negative default-promotion evidence for
  local dense blocks on D/T/He and strengthens the next-step requirement:
  develop a Schur/transport preconditioner or cheaper residual/JVP kernel that
  actually reduces Krylov work.
- 2026-06-18: Added no-op rejection gates to the real-kernel JAX-linearized
  recycling profiler. `scripts/profile_recycling_jax_linearized_gate.py` now
  supports `--require-min-nonlinear-iterations=<n>` and
  `--require-min-linear-iterations=<n>`, records those floors in
  `profile_summary.json`, and returns nonzero if a supposedly heavy profile
  exits before Newton/JAX-GMRES work. The local `dthe-jax-linearized-gate` in
  `scripts/run_research_campaign_bundle.py` now uses `timestep=1.0`, skips the
  redundant initial residual check, requires at least one nonlinear and one
  linear iteration, and caps the current one-step linear budget at `400`. This
  turns the D/T/He research-campaign command into a real heavy-kernel profiling
  gate instead of a tiny-timestep residual-check artifact. A local wrapper run
  against `/Users/rogerio/local/hermes-3` passed in `20.8 s`; the profiled solve
  took `7.86 s`, the RSS sample run took `11.76 s`, the peak RSS delta was
  `683 MiB`, and the solver reported one nonlinear iteration, `400` linear
  iterations, residual `7.315`, and clean JAX-GMRES status.
- 2026-06-18: Ran the remaining cheap D/T/He JAX-linearized preconditioner and
  Krylov-control probes on the same `timestep=1.0` gate. `field_scale`
  completed in `8.01 s`, `linearized_diag` in `8.31 s`, `state_scale` in
  `27.42 s`, and all retained the full `400` update budget. Reducing the
  unpreconditioned control to `10 x 10` updates gave residual `7.316` but did
  not speed up the local CPU run (`8.49 s`), while `5 x 10` updates slowed to
  `27.84 s` and worsened the residual to `8.09`; incremental GMRES at
  `10 x 10` was also slow (`26.46 s`). The profiler now exposes
  `--linear-restart`, `--linear-maxiter`, and
  `--linear-tolerance-factor` as first-class sweep controls, plus
  `--require-max-residual-inf-norm` as a quality gate. The local
  `dthe-jax-linearized-gate` keeps the measured-fast `20 x 20` batched GMRES
  path and now requires residual below `7.4`. A wrapper verification passed in
  `18.9 s`; the profiled solve took `7.46 s`, the RSS sample run took
  `10.29 s`, the peak RSS delta was `830 MiB`, and the summary reported
  residual `7.315 < 7.4`, one nonlinear iteration, `400` linear iterations, and
  clean JAX-GMRES status. This closes simple scalar, diagonal, local-block, and
  reduced-budget probes as default speedup lanes; the next implementation
  target remains residual/JVP kernel cost or a real transport/Schur
  preconditioner.
- 2026-06-18: Tested residual JIT on the same D/T/He quality-gated profile. A
  single non-warmed `--jit-residual` run passed the residual and solver-health
  gates but took `30.90 s`, with `7.37 s` in residual evaluations and
  `23.37 s` in linear solves. A warmed run with one warmup and two timed solves
  still missed the non-JIT baseline: warmup `20.08 s`, timed runs `11.11 s` and
  `9.33 s`, median `10.22 s`. This keeps residual JIT as an opt-in diagnostic
  and possible GPU probe, not the local CPU default.

## Definition Of Done

JAXDRB is ready for a versioned research release when all of the following are
true:

- The repository is lightweight, fast to clone, and free of large tracked dumps,
  traces, and paper-only artifacts.
- Installation is simple and unpinned for runtime dependencies.
- Public APIs, CLI commands, examples, docs, and README claims are consistent.
- Promoted solver coverage remains above `95%` with meaningful tests.
- Every promoted equation term has documentation, source mapping, and tests.
- Every promoted physics lane has verification, regression, and validation
  evidence appropriate to its claim.
- Reference-backed mismatches are either closed or component-local with
  bounded documented errors.
- JAX-native differentiability claims are limited to pure-JAX paths with
  derivative tests and finite-difference comparisons.
- CPU/GPU performance claims are backed by profiler, memory, scaling, and
  same-fidelity comparison artifacts.
- Tokamak and stellarator tutorials run from a clean clone using native outputs
  or release-hosted fixtures.
- README figures/movies are polished, reproducible, and backed by validation
  gates.
- Unsupported or exploratory surfaces are clearly labeled as such.
- Release notes, footprint audit, docs build, coverage gates, package audit,
  tag, GitHub release assets, and PyPI workflow are ready.
