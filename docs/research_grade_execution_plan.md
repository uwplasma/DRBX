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
- `../plan_jax_drb.md`: compatibility redirect to this file.

If any subordinate page conflicts with this file, follow this file and update
the subordinate page after the implementation decision is made.

This planning pass is intentionally not a code/test implementation pass. A
pre-existing uncommitted source edit is present in
`src/jax_drb/solver/implicit.py` from the previous sparse solver-health lane and
is outside the scope of this plan-only pass.

Hosted GitHub Actions should not be polled repeatedly while hosted runners are
blocked by account billing/spending-limit state. Use local release gates during
development and check hosted CI periodically after billing is restored.

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
| Plan authority and release hygiene | 92% | Keep this file current and prevent new competing roadmap files. |
| Meaningful promoted coverage | 96% | Keep `scripts/run_promoted_solver_coverage.py` above `95%` after each solver and geometry promotion. |
| Reference-backed parity | 98.5% | Close remaining neutral `NVh` accepted-step state/history sequencing feeding `Pnlim`, `logPnlim`, and `Grad(logPnlim)`. |
| JAX-native recycling solver | 89% | Make a JAX-transformable full-output recycling path faster and stable enough for default or documented opt-in promotion. |
| Effective preconditioning | 35% | Move beyond negative row/field/local-block evidence to a transport-aware or Schur-style preconditioner with same-case speedup. |
| Performance and scaling | 52% | Rerun heavy CPU/GPU profiles after solver changes and show real-kernel speedup, not only compact-kernel throughput. |
| Drift-reduced Braginskii model surface | 65% | Finish equation-to-code maps, Boussinesq/non-Boussinesq comparisons, vorticity/potential gates, and EM selected-field promotion. |
| Neutral, recycling, sheath, detachment | 78% | Finish term-level neutral/recycling/sheath gates and detachment observables across promoted tokamak lanes. |
| Diverted tokamak self-contained tutorials | 70% | Ensure clean-clone users can fetch small/release-hosted fixtures, run simulations, create movies, and analyze turbulent profiles. |
| 3D stellarator imported-field/VMEC SOL | 67% | Promote live connection-length, endpoint, FCI, grid-refinement, and time-refinement gates before turbulence/movie claims. |
| Code architecture split | 60% | Split broad recycling, neutral, runner, CLI, and large test files into narrow directly tested modules. |
| Docs and examples | 86% | Make every advertised README figure/movie reproducible by a documented example and move extended validation detail into docs. |
| Repo footprint | 90% | Repeat `.git`, tracked-large-file, wheel/sdist, docs-media, and local-cache audits before every tag. |

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
- The remaining neutral `NVh` offender is localized to accepted-step
  state/history sequencing feeding neutral pressure/log-pressure preparation,
  with near-target `Grad(logPnlim)` as a secondary stencil check.

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
- JAX promotion requires the preconditioner to preserve shape stability, avoid
  host loops in hot paths, and keep memory bounded.

Candidate order:

1. Existing baselines: no preconditioner, state scaling, field scaling,
   linearized diagonal, and same-cell local blocks. These are negative or
   neutral evidence so far and remain diagnostic controls.
2. Same-cell block-Jacobi with reuse across nonlinear iterations only if it
   reduces total Krylov cost after build cost.
3. Parallel-line transport preconditioner: approximate each field with
   mass plus parallel advection/diffusion along the open-field direction,
   using batched tridiagonal or banded solves where possible.
4. Neutral/plasma Schur-style preconditioner: approximate stiff neutral
   diffusion/reaction coupling and plasma pressure/momentum coupling as
   separate blocks with cheap coupling corrections.
5. Reduced sparse-JVP preconditioner: assemble a smaller or lower-order sparse
   operator only if memory and build time stay below the compatibility
   finite-difference Jacobian.
6. Matrix-free Krylov promotion only after the residual and preconditioner are
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
3. Promote a tokamak example ladder:
   1D recycling, 1D detachment scan, 2D direct tokamak diffusion/transport,
   2D recycling, multispecies D/T/He, impurity/neon radiation, detached target
   scan, then longer nonlinear turbulence windows.
4. Generate movies and profile plots from the same documented scripts:
   OMP profiles, target profiles, heat flux, particle flux, source maps,
   neutral maps, target temperature, detachment indicators, and turbulence
   fluctuation metrics.
5. Add runtime progress, ETA, restart, output, and provenance examples for the
   tokamak tutorials.
6. Keep reference-code comparison workflows available to developers, but never
   make them required for ordinary user examples.

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
- Local BSTING files: `/Users/rogeriojorge/local/bsting_files` when present.
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
