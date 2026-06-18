# JAXDRB Refactoring Plan

!!! note "Plan authority"
    This page is a subordinate technical appendix for source/test structure.
    The active execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    roadmap conflicts with that plan, follow the execution plan and update this
    appendix afterward.

This document is the clean engineering roadmap for bringing `jax_drb` to a
research-grade, maintainable, and fully auditable state without changing the
scientific claim boundary or breaking the existing promoted parity surface.

The current consolidated execution plan lives in
[research_grade_execution_plan.md](research_grade_execution_plan.md). That page
is the active cross-cutting plan for literature anchors, comparison-code
lessons, git-history hygiene, performance, validation, documentation, examples,
CI/CD, and release readiness. This page remains the detailed refactoring
roadmap for source/test structure.

It is intentionally separate from historical work logs. The goal here is not to
log everything that happened; it is to state what the codebase should look like
when the refactor is complete, what evidence must exist for that state, and in
what order the work should proceed.

## Scope

The refactor target is:

- preserve current promoted functionality and parity behavior against Hermes-3
- split monolithic source and test files into smaller units with clear
  responsibilities
- raise test coverage to a meaningful `95%` on the promoted solver surface, not
  by padding with smoke tests alone
- widen the evidence program so that physics, numerics, performance, and
  autodiff capabilities are all benchmarked and reviewable
- make important solver paths explainable through comments, docstrings, and
  equation-to-code traceability
- generate publication-ready plots directly from validated tests and campaigns
- apply a common publication-plotting standard in the validation layer so new
  figures are high-resolution, information-dense, and tied to the same checked
  artifacts as the tests

The refactor is not an excuse to silently change closures, broaden the claim
boundary, or weaken the current Hermes-backed evidence tiers. If a refactor
changes behavior, the change must be explicit and justified by new evidence.

The current runtime campaign also adds one more hard rule:

- optimization work must be reported against isolated profiler runs and, when
  the lane is Hermès-backed, against the same-machine live comparison surface;
  timing numbers collected under mixed local load should not be promoted to the
  docs or the paper.

## External Comparison Class

The code should be engineered and validated at the standard set by the main
reduced-fluid and edge/SOL comparison class, with kinetic codes used to define
the outer scientific boundary rather than as direct like-for-like competitors.

### Reduced-fluid edge and SOL codes

- BOUT++ framework:
  [Dudson et al. 2009](https://arxiv.org/abs/0810.5757)
- Hermes-3 multi-component edge/SOL model:
  [Dudson et al. 2024](https://www.sciencedirect.com/science/article/pii/S0010465523003363)
- GBS drift-reduced Braginskii code:
  [Halpern et al. 2016](https://www.sciencedirect.com/science/article/pii/S0021999116001923)
- GDB global edge drift-ballooning code:
  [Shi et al. 2018](https://www.sciencedirect.com/science/article/abs/pii/S001046551830208X)
- TOKAM3X full-torus edge/SOL fluid code:
  [Bufferand et al. 2016](https://www.sciencedirect.com/science/article/pii/S0021999116301838)
- SOLEDGE3X detached-regime turbulence:
  [Mosetto et al. 2024](https://www.sciencedirect.com/science/article/pii/S2352179124001790)
- GRILLIX FCI edge turbulence family:
  [Kube et al. 2025](https://www.sciencedirect.com/science/article/pii/S0010465525003765)

### Validation culture and benchmark references

- code and solution verification review:
  [Roy 2005](https://www.sciencedirect.com/science/article/pii/S0021999104004747)
- TCV-X21 benchmark:
  [Bufetov et al. 2022](https://arxiv.org/abs/2109.01618)
- SOLPS-ITER against TCV-X21:
  [Wang et al. 2023](https://arxiv.org/abs/2310.17390)
- Hermes-3 against TCV-X21:
  [Dudson et al. 2025 preprint](https://arxiv.org/abs/2506.12180)
- algorithmic differentiation for plasma edge codes:
  [Carli et al. 2023](https://www.sciencedirect.com/science/article/pii/S0021999123004989)

### Kinetic boundary and differentiable scientific computing context

- Gkeyll SOL turbulence:
  [Shi et al. 2016](https://arxiv.org/abs/1610.09056)
- GENE-X edge and SOL gyrokinetics:
  [Frei et al. 2025](https://www.sciencedirect.com/science/article/pii/S0010465525003194)
- JAX-based differentiable CFD example:
  [JAX-Fluids](https://www.sciencedirect.com/science/article/pii/S0010465522002466)
- official JAX transformation and profiling docs:
  [automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html),
  [profiling](https://docs.jax.dev/en/latest/profiling.html),
  [`pmap` / `shard_map` guidance](https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html)

These references imply the standard we should meet:

- clear equation provenance
- explicit verification and validation separation
- benchmark and experiment-facing diagnostics rather than only operator tests
- maintainable architecture with reusable geometry, solver, and diagnostics
  layers
- honest capability boundaries for differentiability, scaling, and geometry

## Current Codebase Assessment

The main software debt is concentrated in a small number of oversized source
and test files.

### Source hotspots

The primary split candidates are:

- `src/jax_drb/native/recycling_1d.py`
- `src/jax_drb/native/runner.py`
- `src/jax_drb/native/neutral_mixed.py`
- `src/jax_drb/cli.py`
- `src/jax_drb/validation/tokamak_tcv_x21_scaffold.py`
- `src/jax_drb/validation/temperature_feedback_campaign.py`
- `src/jax_drb/validation/detachment_controller_campaign.py`
- `src/jax_drb/validation/tokamak_tcv_x21_toroidal_movie.py`
- `src/jax_drb/validation/__init__.py`

These files currently mix several responsibilities:

- equation assembly
- source and closure evaluation
- pack/unpack and state-layout logic
- time stepping
- diagnostics
- plotting
- artifact writing
- validation policy and registry export

### Test hotspots

The same pattern appears in the tests, especially in:

- `tests/test_native_integrated_2d_recycling.py`
- `tests/test_native_recycling_1d.py`
- `tests/test_native_tokamak_cases.py`
- `tests/test_native_runner.py`
- `tests/test_parity_recycling.py`
- `tests/test_native_neutral_mixed.py`

These tests already contain valuable evidence, but too much of it is trapped in
large multipurpose files. That makes coverage harder to interpret and makes it
too easy to leave a branch untested because the only exercising test is a large
integration case.

### Architectural gaps

The current main gaps are:

- solver logic is still concentrated in large physics-family files rather than
  in small, typed, testable operator modules
- the validation layer mixes campaign orchestration, plotting, and registry
  export
- differentiable examples are real, but they are still a separate island rather
  than a staged pathway into promoted physics lanes
- the current `95%` closeout slice is useful, but it is not yet the same thing
  as `95%` meaningful coverage of the promoted solver surface
- `scripts/run_promoted_solver_coverage.py --audit` now exposes that gap
  directly for the native mesh/metric/open-field/recycling/runner/parity/CLI
  surface; the next step is raising that audit to the default `95%` gate through
  extracted operator tests rather than through more broad smoke coverage
- many examples are useful scientifically, but not all of them have an explicit
  status as tutorial, benchmark, campaign, or publication artifact generator
- the runtime/performance story still mixes two different classes of workloads:
  compact native JAX kernels, which are now GPU-auditable, and heavy promoted
  recycling transients, which remain primarily CPU/host-side optimization
  targets

## Refactor Principles

The refactor should follow a few hard rules.

1. No scientific drift during structural work.
   The first phase should move code without changing equations or parity bands.

2. Split by responsibility, not by file length alone.
   A smaller file that still mixes residual assembly, plotting, and artifact
   writing is not a successful refactor.

3. Every extracted operator gets its own tests.
   Refactoring is only complete when the extracted code has direct unit
   coverage, not just inherited integration coverage.

4. Every promoted validation lane must stay evidence-backed.
   No refactor should weaken the committed benchmark artifacts or capability
   labeling.

5. Keep the JAX boundary explicit.
   Pure-JAX transformable paths and host-backed implicit paths should be
   separate in the architecture and in the docs.

6. Separate CPU and GPU evidence by workload class.
   GPU claims should come from compact native JAX lanes unless and until the
   heavier transient backbone is sufficiently de-hosted to make accelerator
   measurements scientifically meaningful.

## Target Package Architecture

The current top-level package structure is already reasonable. The main issue is
the lack of internal subdivision inside the physics and validation packages.

### Native solver layout

The target `native` layout should move toward subpackages such as:

```text
src/jax_drb/native/
  recycling/
    __init__.py
    state.py
    layout.py
    operators.py
    closures.py
    boundaries.py
    sources.py
    residual.py
    stepping.py
    diagnostics.py
  neutral/
    __init__.py
    state.py
    operators.py
    closures.py
    boundaries.py
    residual.py
    diagnostics.py
  tokamak/
    __init__.py
    metrics.py
    mapping.py
    lineouts.py
    selected_field.py
```

Specific file splits:

- `recycling_1d.py`
  - extract state/result and small boundary-setting layouts: started in
    `src/jax_drb/native/recycling_1d_state.py`, with public and private
    compatibility re-exports kept from `recycling_1d.py`
  - extract progress/ETA bookkeeping: started in
    `src/jax_drb/native/recycling_progress.py`, with the existing private
    `_build_recycling_progress_details` import path preserved from
    `recycling_1d.py`
  - extract packed-state layout and metadata
  - extract transport operators
  - extract atomic/recycling/controller source closures
  - extract sheath and wall boundary handling
  - extract residual and Newton-facing assembly
  - extract diagnostics and summaries
- `neutral_mixed.py`
  - extract state/result layouts: started in
    `src/jax_drb/native/neutral_mixed_state.py`, with re-exports kept from
    `neutral_mixed.py` for compatibility
  - extract parallel transport operators: started in
    `src/jax_drb/native/neutral_mixed_operators.py`, including the open-field
    parallel advection/diffusion operators, perpendicular diffusion operator,
    parallel gradient, and gradient-magnitude helper used by the neutral
    diffusion closure
  - extract wall/guard reconstruction: started in
    `src/jax_drb/native/neutral_mixed_boundaries.py`, with the original
    `neutral_mixed.py` private aliases preserved for compatibility while tests
    and callers are migrated
  - extract source and exchange closures
  - extract residual history/step bookkeeping
- `runner.py`
  - split deck parsing and curated case resolution
  - split restart and trim-window resolution: started in
    `src/jax_drb/native/runner_state.py`, which now owns native run/restart
    result state and restart-bundle construction
  - split case dispatch registry
  - split result/artifact writing
  - split parity-mode and compare-surface handling

### Solver layout

`src/jax_drb/solver/implicit.py` should be prepared for a clearer boundary
between:

- finite-difference Jacobian assembly
- colored sparse structure handling
- linear solve backends
- JAX-linearized Newton path
- diagnostics and timing

Longer-term, this makes it possible to stage `lineax`-style or fully JAX-native
linearization work without entangling it with the host-backed sparse path.
That should remain a second-phase improvement, not a precondition for the
initial split.

### Validation layout

The validation package should move toward three layers:

```text
src/jax_drb/validation/
  campaigns/
  reports/
  plots/
  geometry/
  publication/
```

Specific split targets:

- geometry scaffolds and selected-field helpers should share a `geometry/`
  namespace rather than living as independent monoliths
- campaign modules should assemble data and metrics but delegate plotting and
  report serialization
- `validation/__init__.py` should become a small registry surface rather than a
  large import/export wall

### CLI layout

`src/jax_drb/cli.py` should be decomposed into:

- command parsing
- command handlers
- render/progress helpers
- shared artifact/log formatting

The goal is a CLI that is easy to test without triggering full runs.

## Testing And Coverage Strategy

The `95%` target should be redefined as meaningful coverage of the promoted
scientific and software surface. It should not be reached by counting only
smoke tests, nor by excluding the hard code.

### Target test taxonomy

The test suite should evolve into explicit layers:

- `tests/unit/`
  - pure operator and parser tests
  - geometry helpers
  - state-layout tests
  - serialization and report helpers
- `tests/regression/`
  - fixed numerical baselines without external references
  - restart equivalence
  - pack/unpack invariants
- `tests/parity/`
  - Hermes-backed or reference-backed comparisons
  - bounded residual surfaces for operational lanes
- `tests/autodiff/`
  - `grad` vs finite-difference
  - `jvp` / `vjp` consistency
  - `jit` invariance
  - `vmap` / batching invariance
- `tests/publication/`
  - campaign JSON/NPZ/plot generation and schema checks
  - artifact completeness and public-surface sanitization
- `tests/smoke/`
  - bounded fast CLI/tutorial checks only

The exact on-disk migration can be phased, but the logical distinction should
exist immediately in naming, markers, and CI slices.

### Coverage definition

The coverage target should have three explicit metrics.

1. Unit and operator coverage
   - every extracted operator and closure branch directly exercised
2. Promoted-lane coverage
   - every `native_exact` and `native_operational` path exercised by parity or
     physics regression tests
3. Public-surface coverage
   - CLI, examples, reports, plots, manifests, and artifact schemas exercised

The release target is:

- at least `95%` on the promoted solver and public-surface slice
- no critical operator family below `90%`
- no monolithic module left “green” only because an integration test passes

### Physics and literature-anchored tests

The test program should include:

- manufactured-solution convergence for promoted operator families
- conservation and decay tests on compact lanes
- sheath and wall boundary condition tests against the reduced-fluid literature
- collisional exchange and thermal-force tests against documented formulas
- recycling source partition and controller reconstruction tests
- neutral/plasma exchange and trim-window compare-surface tests
- TCV-X21, TCV-X21-derived, and direct-tokamak bounded windows
- TORPEX-style blob dynamics and drift-wave benchmark diagnostics
- 3D selected-field, runtime, and convergence reports

The benchmark hierarchy should mirror the literature:

- verification:
  MMS, order tests, restart equivalence, operator identities
- code-to-code validation:
  Hermes-3 and curated reference payloads
- benchmark and experiment-facing validation:
  TCV-X21, TORPEX-style blob/filament, detachment and divertor scaling, 3D
  geometry portability

## Literature-Anchored Baseline Audit

Before any baseline is treated as publication-grade, it should be checked
against the figure and diagnostic patterns used in the comparison literature.
The main references inspected for this plan were:

- Hermes-3:
  [Dudson et al. 2024](https://www.sciencedirect.com/science/article/pii/S0010465523003363)
- GBS:
  [Giacomin et al. 2022](https://www.sciencedirect.com/science/article/pii/S0021999122003280)
- SOLPS-ITER against TCV-X21:
  [Wang et al. 2024](https://doi.org/10.1088/1741-4326/ad3562)
- TCV-X21 benchmark:
  [Sales de Oliveira et al. 2022](https://doi.org/10.1088/1741-4326/ac74b4)
- GBS parallel-gradient discretization study:
  [Mosetto et al. 2015](https://www.sciencedirect.com/science/article/pii/S001046551400366X)
- TORPEX X-point validation:
  [Galassi et al. 2022](https://orbit.dtu.dk/en/publications/validation-of-edge-turbulence-codes-in-a-magnetic-x-point-scenari/)
- detachment scaling with Hermes-1D:
  [Body et al. 2024](https://www.sciencedirect.com/science/article/pii/S2352179124002424)
- SPLEND1D detachment model:
  [Delaporte-Mathurin et al. 2024](https://arxiv.org/abs/2402.04656)

These papers repeatedly use the same figure classes:

- convergence curves and MMS order plots
- workflow or domain-geometry schematics
- 2D and 3D snapshots with magnetic topology or separatrix overlays
- profile overlays against experiment or a trusted reference code
- scan or optimization figures showing improved agreement across observables
- runtime or algorithmic figures only when they are tied to a scientific claim

This implies a rule for `jax_drb` baselines:

- a baseline is not anchored in the literature if it exists only as arrays or a
  JSON summary
- every promoted benchmark family should have at least one figure that matches a
  recognizable literature pattern and at least one machine-readable artifact
- where the literature compares profiles, targets, or diagnostic maps, our
  baseline should expose those same observables rather than only reduced norms

The current codebase review adds one more rule:

- cross-case summary dashboards, profiler summaries, and runtime rollups are
  useful supporting artifacts, but they are not primary publication figures
  unless they are paired with a literature-style physical or numerical question
  on the same surface

The current live native-versus-live-Hermes rerun matrix adds a concrete update
to that audit:

- exact compact tokamak transport/turbulence lanes are now demonstrated on the
  same machine, not only through committed reference arrays
- heavy 1D recycling and the neutral mixed lane remain the main remaining live
  runtime and fidelity gaps
- the neutral mixed gap now also has a dedicated live rerun boundary-audit
  figure, which should be treated as the first manuscript-facing neutral
  mismatch surface rather than relying only on matrix-level summary bars
- integrated and direct-tokamak recycling are already competitive in wall time
  on this machine, but their current one-step mismatch is dominated by near-zero
  `NVd` on the guarded compare surface rather than by large absolute profile
  error
- full live 3D Hermès reruns are still absent, so 3D should remain a distinct
  selected-field evidence track until that gap closes

## Manuscript Figure Plan From Tests And Campaigns

The future paper figures should come out of validated tests and campaigns,
rather than from paper-only scripts that reimplement logic.

### Verification figures

- MMS convergence figure for promoted operator families
  - current seed: `fluid_1d` MMS and future MMS extensions
  - literature anchor: Hermes-3 figure style for 1D convergence
- operator verification figure for parallel-gradient and elliptic closures
  - target: compare exact/analytic or manufactured expectations, numerical
    order, and bounded operator residuals
  - literature anchor: GBS parallel-gradient operator paper

### Geometry and benchmark figures

- domain and diagnostic map figure for tokamak validation
  - target: magnetic geometry plus diagnostic or compare window overlay
  - literature anchor: SOLPS-ITER / TCV-X21 diagnostic map figure
- 3D snapshot figure with topology overlay
  - target: tokamak/traced-field-line/stellarator selected-field snapshot with
    separatrix or surface annotation
  - literature anchor: GBS whole-volume snapshot figure

### Physics validation figures

- direct tokamak recycling ladder
  - one-step, `nout=3`, and `nout=5` bounded overlays
  - target quantities: density, pressure, momentum, recycling-related source
    terms, and summary residuals
- neutral short-window validation
  - full-array and centerline comparisons, not only scalar residual summaries
- detachment and controller figures
  - target: controller history, temperature or recycling target tracking,
    response transients, and agreement to reference histories
- impurity/radiation and reactions figures
  - target: rate closure agreement, radiation loss trends, and source partition
  - include profile-level ionisation, isotope-resolved charge exchange, and
    collisionality-closure figures from the reactions/collisions campaign so
    the paper does not rely only on scalar pass/fail metrics

### Differentiability figures

- gradient-vs-finite-difference comparison
- covariance pushforward vs Monte Carlo uncertainty comparison
- inverse-design convergence and recovered profile/design overlay
- workstation throughput scaling for repeated heavy solves

These should remain anchored to currently promoted differentiable lanes until a
stronger open-field or recycling differentiable lane exists.

### Figure-generation policy

Every figure destined for the manuscript should be produced from:

- a validated test or campaign script in the code repo
- a machine-readable analysis artifact
- a plotting function that can be regression-checked for completeness

The paper repo may compose panels, but it should not be the first place where
the data product is generated.

In practical terms, this means that any new test strong enough to justify a
scientific claim should be accompanied by one of:

- a validation campaign in `src/jax_drb/validation`
- a benchmark/example entry point that writes stable artifacts
- a direct artifact-producing script under `examples/engineering`

The test alone is not enough if the result is expected to appear in the paper.

## Autodiff, JAX, And Optimization Roadmap

The differentiable lane should become a first-class part of the architecture,
but the plan must stay honest about where full autodiff exists today and where
host-backed boundaries still dominate.

Performance, differentiability, and accuracy are joint constraints during the
refactor. A structural change is incomplete if it makes the code cleaner while
quietly degrading runtime, transformability, or numerical fidelity.

### Immediate differentiable targets

Keep and harden the current compact native lanes for:

- sensitivity analysis
- uncertainty propagation
- inverse design
- throughput and scaling experiments

Every differentiable example should have:

- `grad` or `value_and_grad` agreement with finite differences
- `jit`-compiled and eager invariance
- `vmap` batching invariance where relevant
- published analysis JSON and figure outputs

### Next promoted differentiable physics lanes

Promote differentiable tasks in this order:

1. compact diffusion lane
2. drift-wave or vorticity lane with scalar QoIs
3. reduced selected-field 3D operators
4. one open-field or recycling transient with a fully documented differentiation
   boundary

Only after those are stable should the plan expand to:

- mirror optimization
- tokamak geometry/control optimization
- stellarator or traced-field-line optimization

Those workflows will require cleaner parameterized geometry and solver APIs than
the current campaign scripts expose. They are realistic goals, but they depend
on the refactor of geometry adapters and on explicit pure-JAX parameter
boundaries.

### JAX ecosystem usage policy

Use JAX ecosystem tools where they provide a clear engineering win:

- `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jvp`, `jax.vjp`, `jax.linearize`
  where the solver path is already transformable
- `jax.checkpoint` or rematerialization only when memory-pressure measurements
  justify it
- `diffrax` only where its solver abstractions improve clarity or capability on
  genuinely JAX-native paths
- `lineax` only when the linear algebra boundary is ready for it
- `equinox` only where module structure materially clarifies stateful
  differentiable components

Do not add these libraries to hot paths just to say they are used. The standard
is stronger code and stronger evidence, not ecosystem completeness for its own
sake.

## Additional Validation And Benchmark Lanes Worth Adding

The current plan already covers the main promoted lanes, but the literature
suggests a few additional benchmark families that would materially strengthen a
future JCP paper.

### Parallel-gradient and operator verification

The GBS parallel-gradient paper shows that operator papers can make a strong
scientific point if they combine:

- analytical or reduced-model expectations
- convergence and dispersion-style verification
- nonlinear benchmark comparison

`jax_drb` should add a dedicated operator campaign for:

- parallel-gradient discretization
- sheath boundary sensitivity
- geometry-metric consistency in selected-field reductions

### TORPEX and X-point blob validation

The TORPEX X-point validation literature provides a natural bridge between
compact blob lanes and divertor/X-point geometry claims. This should become an
explicit planned benchmark package, not just a “nice to have.”

Target deliverables:

- seeded blob trajectory and morphology figures
- center-of-mass and amplitude diagnostics
- X-point topology or null-region geometry figure
- comparison against published blob propagation trends

### Detachment scaling and 1D reduced-model comparison

The Hermes-1D detachment-scaling and SPLEND1D papers imply that a strong
reduced-model paper should include:

- rollover and detachment-front trends
- scaling against standard detachment models such as Lengyel–Goedheer
- explicit scan artifacts rather than isolated one-off transients

`jax_drb` should therefore plan:

- a 1D detachment-scaling campaign
- comparison against reduced theoretical scaling where appropriate
- controller and radiation scans tied to reusable analysis JSON and plots

### TCV-X21 neutrals and diagnostics extension

The SOLPS-ITER TCV-X21 paper strengthens the benchmark by adding neutral
pressure and Balmer-line observables, not just density and temperature. This is
worth adding to the plan explicitly:

- neutral pressure comparisons
- Balmer-line or proxy synthetic-diagnostic comparisons where feasible
- ionization source distribution summaries

### Mirror and stellarator optimization lane

For the differentiable roadmap beyond compact diffusion and reduced 3D selected
field lanes, the highest-value physics extensions are:

- mirror geometry parameter sensitivity and optimization
- tokamak source/control or metric parameter optimization
- stellarator or traced-field-line geometry optimization on reduced observables

These are longer-term tasks, but they should be planned early because they
shape how geometry parameters, diagnostics, and JAX-transformable APIs are
designed during the refactor.

## Example And Campaign Triage

Every example should have an explicit status.

### Benchmark-grade examples

These should be kept, validated, and tied to public artifacts:

- `examples/blob2d_meeting_demo.py`
- `examples/diverted_tokamak_movie_demo.py`
- `examples/engineering/*campaign*_demo.py`
- `examples/autodiff_diffusion_*`
- `examples/strong_scaling_diffusion_demo.py`

### Tutorial-grade examples

These should stay simple and user-facing, with lighter evidence requirements:

- `examples/restartable_diffusion_tutorial.py`
- `examples/inputs/restartable_diffusion.toml`

### Promotion policy

For each example, decide whether it is:

- a tutorial
- a benchmark generator
- a validation campaign
- a publication figure generator

If an example does not fit one of these roles, it should be merged, simplified,
or removed.

## Documentation Expansion Plan

The public docs are already useful, but the refactor should add much more
developer-facing documentation on testing and code structure.

### Maintained documentation surfaces

- `docs/code_structure.md`
  - package map
  - module responsibilities
  - import and registry boundaries
  - JAX-native vs host-backed solver surfaces
- `docs/testing_strategy.md`
  - test taxonomy
  - markers and expected runtime classes
  - what counts as verification, regression, parity, and benchmark validation
  - how coverage is measured and interpreted
- `docs/equation_to_code_map.md`
  - equation terms
  - closure names
  - implementation modules
  - direct tests and campaigns that exercise each term
- `docs/example_status_matrix.md`
  - tutorial vs benchmark vs campaign vs publication generator role for each
    example
- `docs/profiling_runtime.md`
  - supported runtime-profiling workflow
  - case-selection policy for worst offenders
  - Perfetto / TensorBoard / memory-profile guidance
  - CPU versus GPU execution notes
- `docs/runtime_gap_remediation.md`
  - current live Hermès runtime/fidelity offenders
  - measured before/after fixes
  - next remediation order
  - office GPU execution plan

These pages now exist in the MkDocs navigation and are included in the
release-surface checks. The remaining work is maintenance: update this set
whenever a module is split, a physics gate is promoted, or a performance claim
changes.

### Documentation requirements during refactor

Every refactor milestone should update:

- code docstrings
- public docs for the affected package
- the equation-to-code map
- the testing strategy page when a new validation layer or marker is introduced

## Runtime And Fidelity Remediation Plan

The current public live matrix is now strong enough to drive the next runtime
and mismatch work directly instead of relying on generic intuition.

### Current priority order

1. `neutral_mixed_one_step`
   - real fidelity gap on `NVh`
   - runtime now improved substantially, so further work should be driven by
     closure/operator comparison rather than by bulk vectorization alone
2. `recycling_dthe_one_step`
   - current worst live runtime ratio
   - already tight enough in fidelity that runtime work can proceed with strong
     regression protection
3. `recycling_1d_one_step`
   - same class as the multispecies case, but slightly less urgent after the
     latest speedup
4. integrated/direct tokamak recycling observables
   - runtime already competitive
   - next work should move toward target/source/ionization observables closer to
     the TCV-X21, SOLPS-ITER, and Hermès comparison style

### Required profiler workflow

Every one of those runtime tasks should use the same public workflow:

- [profiling_runtime.md](profiling_runtime.md)
- [scripts/profile_curated_case.py](../scripts/profile_curated_case.py)

That workflow should be used for:

- local CPU cProfile diagnosis
- JAX trace collection
- device-memory snapshots on GPU
- compilation-cache and XLA-dump comparisons when compile or kernel behavior is
  the suspected bottleneck

### GPU execution plan

The reachable `office` machine is now the planned GPU profiling host. It
already exposes two CUDA-visible JAX devices, so the next blocker is only
environment setup rather than hardware availability.

The expected next GPU sequence is:

1. create a clean repo-local environment on `office`
2. install matching JAX/JAXLIB plus `jax_drb`
3. rerun the current worst-offender cases with trace and memory profiling
4. only then decide whether the next acceleration target is:
   - single-case runtime
   - batched independent solves
   - multi-device sharding

This is required because the code is intended for research and paper
production, not only for internal development.

## Comments, Docstrings, And Equation Traceability

The refactor should add documentation in the code, not just in Markdown.

### Docstring standard

For every public function, class, and campaign entry point:

- one-sentence summary
- parameter descriptions
- return structure
- invariants or important assumptions
- parity or benchmark role if relevant

### Inline comment standard

Comments should be added only where they materially help:

- non-obvious operator forms
- geometry indexing and trim-window rules
- pack/unpack layout assumptions
- JAX transformation boundaries and host barriers
- boundary-condition branches that correspond to literature formulas

### Equation-to-code traceability

Every promoted closure term should be traceable from:

- `docs/physics_models.md`
- its operator or closure module docstring
- its unit tests
- at least one validation campaign or regression test

The end state is that a reviewer or new developer can answer:

- where is `Q_cond,s` implemented?
- where is the thermal-force term tested?
- where is the wall reconstruction or sheath closure validated?

without reading a monolithic 5000-line file.

## Publication-Ready Plots From Tests

The strongest publication figures should come from validated tests and campaign
artifacts, not ad hoc notebooks.

### Priority figure families

- MMS convergence and order plots
- direct tokamak recycling transient ladders
- live native-vs-live-Hermes rerun matrix
- neutral short-window full-array diagnostics
- blob or drift-wave benchmark diagnostics
- TCV-X21 scaffold and selected-field comparison summaries
- 3D runtime and convergence campaigns
- JAX profile and local CPU throughput campaigns
- sensitivity, UQ, and inverse-design figures on promoted differentiable lanes
- workstation throughput figures on repeated heavy solves, not only tiny-kernel
  timing plots, for local research workflows

### Current research-grade figure subset

After the latest validation pass, the strongest existing figure families are:

- fluid 1D MMS convergence
- reactions and collisions
- neutral parallel diffusion
- collision closure
- tokamak anomalous diffusion
- target recycling
- live Hermès rerun matrix across representative 1D and 2D lanes
- autodiff uncertainty propagation
- local CPU scaling on repeated heavy tokamak-recycling solves

The current summary/runtime/profile figures remain useful, but they should stay
secondary in the future paper unless they are extended:

- Hermes comparison summary
- native 3D runtime campaign
- JAX native profile audit

The next paper-strengthening benchmark additions should therefore focus on the
literature-recognizable missing classes rather than on more dashboards:

- TCV-X21 target and divertor observable maps
- source and ionisation-distribution figures on benchmark tokamak lanes
- TORPEX/X-point blob trajectories and morphology figures
- detachment-scaling curves and front-location trends

### Plot-generation policy

Each figure-producing campaign should:

- write machine-readable analysis JSON
- write arrays NPZ when useful
- generate a publication-grade image with stable labels and units
- have a regression test for artifact completeness and basic schema
- be callable from a small script or example entry point

This keeps the paper and the code aligned: important figures should be
reproducible by the same scripts that prove the underlying result.

The current template for this policy is the public
[fluid_1d_mms_convergence.md](fluid_1d_mms_convergence.md) bundle: one shared
verification implementation now feeds the standalone script wrapper, direct
tests, docs artifacts, and future manuscript-ready convergence figures.

The next closure-facing template is the public
[neutral_parallel_diffusion_campaign.md](neutral_parallel_diffusion_campaign.md)
bundle: one shared closure implementation now feeds direct operator tests, docs
artifacts, and a manuscript-usable AFN-versus-multispecies profile figure on a
prepared multispecies recycling state.

The next collisional template is the public
[collision_closure_campaign.md](collision_closure_campaign.md) bundle: one
shared Braginskii-style collision module now feeds direct operator tests, docs
artifacts, and a manuscript-usable closure-activity figure covering friction,
ion viscosity, and conduction on the prepared multispecies recycling state.

The next tokamak-geometry template is the public
[tokamak_anomalous_diffusion_campaign.md](tokamak_anomalous_diffusion_campaign.md)
bundle: one shared anomalous-diffusion operator now feeds direct non-orthogonal
tokamak tests, docs artifacts, and a manuscript-usable geometry-contrast figure
on the evolved direct-recycling state.

The next boundary-and-recycling template is the public
[target_recycling_campaign.md](target_recycling_campaign.md) bundle: one shared
target-recycling support layer now feeds direct boundary tests, docs artifacts,
and a manuscript-usable prepared-state figure for recycling sources and the
electron boundary response.

The next code-to-code comparison template is the public
[hermes_live_rerun_campaign.md](hermes_live_rerun_campaign.md) bundle: one
shared live rerun implementation now feeds direct campaign tests, docs
artifacts, and a manuscript-usable same-machine native-versus-Hermes matrix
that exposes both fidelity and runtime competitiveness across representative 1D
and 2D lanes.

## Sequencing

The safest order is:

1. freeze scientific behavior and record current parity baselines
2. split the largest source modules without changing numerics
3. split the giant test files into explicit layers
4. add direct unit/operator coverage on the extracted modules
5. upgrade the coverage gate from “closeout slice” to “promoted solver surface”
6. widen literature-anchored benchmark campaigns
7. promote the next differentiable physics lanes
8. regenerate publication-quality figures from the validated campaigns

## Definition Of Done

The refactor is complete when all of the following are true:

- promoted parity behavior against Hermes-3 is unchanged or explicitly improved
- the main monolithic files are split into coherent modules
- the promoted solver surface reaches a meaningful `95%` coverage target
- every promoted closure family has direct operator tests and at least one
  physics-facing validation surface
- differentiable lanes include sensitivity, UQ, inverse design, and at least one
  promoted physics family beyond compact diffusion
- examples have explicit roles and evidence tiers
- public docs, code docstrings, and tests all point to the same equations and
  operator names
- the paper figures can be regenerated from the validated campaign scripts

That is the standard required to move `jax_drb` from “promising and already
useful” to “well-structured, research-grade, and straightforward to extend
without breaking scientific trust.”
