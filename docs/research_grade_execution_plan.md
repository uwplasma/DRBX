# Research-Grade Execution Plan

This is the active engineering and validation plan for turning `jax_drb` into a
research-grade, low-overhead, end-to-end differentiable edge/SOL plasma code
that can be shipped to external researchers without relying on private context.

This plan consolidates the refactoring roadmap, validation matrix, profiling
notes, documentation status, comparison-code audit, and recent git history. It
is intentionally not a work log. It defines the target state, the evidence
needed for that target state, and the next execution sequence.

## Current Repository State

Audit date: 2026-04-23.

Active repository:

- working tree: `/Users/rogerio/local/jax_drb`
- remote: `https://github.com/uwplasma/jax_drb.git`
- branch: `main`
- audit base commit: `631c2f74 Cache target geometry and add neutral mismatch audit`
- unrelated local change intentionally left untouched:
  `docs/data/detachment_controller_campaign_artifacts/data/detachment_controller_campaign.json`

Recent history is concentrated in four work streams:

- solver/runtime performance: sparse finite-difference Jacobian threading,
  reaction/source allocation cleanup, cached target-boundary geometry, runtime
  ETA reporting, profiling scripts, persistent compilation-cache support
- validation evidence: live reference rerun matrix, neutral-mixed boundary
  mismatch audit, MMS convergence package, reaction/collision/neutral/target
  campaigns, local CPU scaling, differentiability/UQ figures
- architecture extraction: recycling layout, setup, state preparation, atomic
  rates, reactions, collisions, neutral diffusion, target recycling,
  anomalous diffusion, runner helpers, cache/reference/compare helpers
- release hardening: Python 3.10-3.12 workflow, PyPI publish workflow, MIT
  license, release-surface path sanitation, runtime install dependency cleanup

The current codebase is not empty scaffolding. It already has a meaningful
native and reference-backed validation surface. The remaining problem is that
the strongest evidence and the most maintainable architecture do not yet line
up perfectly: the highest-fidelity recycling lanes still carry host-side
implicit-solver bottlenecks, the largest files remain too broad, and several
strong validations live next to legacy or manuscript-only material that should
not be part of a clean research-code release.

## External Literature And Code Baseline

The validation and engineering standard should be set by the major edge/SOL
fluid, gyrofluid, gyrokinetic, and differentiable scientific-computing codes.
The useful lesson is not that `jax_drb` should copy any one of them. The useful
lesson is that a credible plasma code separates equations, closures, numerics,
inputs, diagnostics, tests, and benchmark artifacts in a way that external
users can audit.

### BOUT++

Primary references:

- Dudson et al. 2009, "BOUT++: a framework for parallel plasma fluid
  simulations": https://arxiv.org/abs/0810.5757
- BOUT++ source: https://github.com/boutproject/BOUT-dev
- BOUT++ physics-model documentation:
  https://bout-dev.readthedocs.io/en/latest/user_docs/physics_models.html
- BOUT++ code-layout documentation:
  https://bout-dev.readthedocs.io/en/latest/developer_docs/code_layout.html

Relevant findings:

- BOUT++ is a general 3D plasma-fluid framework in curvilinear coordinates.
- The code/paper emphasizes implicit time evolution, high-order advection
  options, benchmark problems, and scaling with processor count.
- The source tree is organized around `manual`, `examples`, `tests`, `include`,
  `src`, `solver`, `mesh`, `physics`, and external solver interfaces.
- The model pattern is explicit: initialize variables, declare evolved state,
  communicate guard cells, assemble the RHS, and delegate time integration to a
  solver backend.
- The current source exposes MPI, OpenMP-aware regions, PETSc/SUNDIALS
  interfaces, preconditioner hooks, and stencil abstractions.

Implication for `jax_drb`:

- Keep physics operators small and testable.
- Keep solver backends separate from model assembly.
- Document every evolved equation through an equation-to-code bridge.
- Treat performance/scaling as a first-class validation artifact, not a README
  claim.

### Hermes-3

Primary references:

- Dudson et al. 2024, "Hermes-3: Multi-component plasma simulations with
  BOUT++": https://www.sciencedirect.com/science/article/pii/S0010465523003363
- Hermes-3 preprint: https://arxiv.org/abs/2303.12131
- Hermes-3 source: https://github.com/boutproject/hermes-3
- Hermes-3 docs: https://hermes3.readthedocs.io/en/latest/
- Hermes-3 equations:
  https://hermes3.readthedocs.io/en/latest/equations.html
- Hermes-3 closures:
  https://hermes3.readthedocs.io/en/latest/closure.html
- Hermes-3 reactions:
  https://hermes3.readthedocs.io/en/latest/reactions.html
- Hermes-3 boundary conditions:
  https://hermes3.readthedocs.io/en/latest/boundary_conditions.html
- Hermes-3 tests:
  https://hermes3.readthedocs.io/en/turbulence-docs/tests.html

Relevant source-code findings from a shallow source audit:

- `src/` and `include/` contain component-level implementations for density,
  pressure, energy, momentum, neutral models, Braginskii closures, reactions,
  sheath closure, recycling, controllers, anomalous/classical diffusion,
  electromagnetic terms, vorticity, and source/diagnostic components.
- The component architecture exposes whole-equation components and term
  components. This matches the way the public documentation explains the model.
- The test/doc strategy includes unit, integrated, and convergence tests.
- The neutral model family is explicitly split between 1D
  `neutral_parallel_diffusion` and 2D/3D `neutral_mixed`.
- The boundary-condition documentation identifies exact target/sheath,
  recycling, no-flow, and neutral reflection semantics that must be matched or
  deliberately bounded in `jax_drb`.

Implication for `jax_drb`:

- The promoted parity surface must be component-aware, not just field-aware.
- Every implemented component should have a local unit/operator test, a
  reference-backed gate when appropriate, and a doc page that states exactly
  what is implemented.
- Remaining mismatches should be localized to components or boundary terms,
  not described only through aggregate summary errors.

### GBS, GDB, TOKAM3X, SOLEDGE3X, GRILLIX

Primary references:

- GBS code paper: https://www.sciencedirect.com/science/article/pii/S0021999116001923
- GBS PDF: https://infoscience.epfl.ch/bitstreams/70937adb-6f71-4d41-98bc-852b4b5b50e9/download
- GBS kinetic-neutral extension: https://arxiv.org/abs/2112.03573
- GDB paper: https://www.sciencedirect.com/science/article/abs/pii/S001046551830208X
- TOKAM3X paper: https://www.sciencedirect.com/science/article/abs/pii/S0021999116301838
- SOLEDGE3X detached-regime paper:
  https://www.sciencedirect.com/science/article/pii/S2352179124001790
- GRILLIX CPC paper:
  https://www.sciencedirect.com/science/article/pii/S0010465525003765

Relevant findings:

- GBS is a 3D global, flux-driven, two-fluid SOL turbulence code that evolved
  toward finite ion temperature, neutral physics, closed-field regions,
  non-Boussinesq polarization, and scalable parallel multigrid.
- TOKAM3X emphasizes full-torus edge/SOL turbulence in limited and diverted
  geometry with verification and validation.
- GRILLIX and related flux-coordinate-independent codes provide the benchmark
  standard for geometry portability and edge turbulence in complex geometry.
- These codes use convergence studies, profile comparisons, fluctuation
  diagnostics, target observables, and scaling studies as primary figures.

Implication for `jax_drb`:

- Publication-grade validation should not be only scalar error dashboards.
  It must include profile/target plots, convergence curves, closure lineouts,
  transport/source maps, and scaling plots tied to a physical question.
- The 3D story needs geometry portability tests, not only a single diverted
  tokamak scaffold.

### TCV-X21 And Diverted-Tokamak Benchmark Culture

Primary references:

- TCV-X21 benchmark paper: https://arxiv.org/abs/2109.01618
- SOLPS-ITER validation against TCV-X21:
  https://arxiv.org/abs/2310.17390
- Hermes-3 validation against TCV-X21:
  https://arxiv.org/abs/2506.12180
- TCV-X21 FAIR dataset: https://zenodo.org/records/5776286

Relevant findings:

- TCV-X21 was designed as a diverted L-mode validation reference case for edge
  turbulence codes.
- The benchmark uses many 1D and 2D observables across multiple diagnostics and
  both field directions.
- The original turbulence-code validation found good upstream agreement and
  weaker target agreement; sensitivity was tied to resistivity, heat
  conductivities, source rates, and sheath boundary conditions.
- SOLPS-ITER validation added neutral observables and concluded that neutral
  dynamics and ionization source distribution are central, even in a case
  designed to reduce neutral impact.
- Hermes-3 TCV-X21 simulations reproduce important profile and target-shift
  features while still identifying neutrals as a likely missing factor for
  remaining target-region differences.

Implication for `jax_drb`:

- TCV-X21 is the right external diverted benchmark, but it should come after the
  native recycling/tokamak transient backbone is stable.
- The benchmark package must include target profiles, OMP profiles, neutral
  observables when available, field-direction-sensitive diagnostics, and a
  transparent quantitative agreement metric.

### Gkeyll And Kinetic Boundary Codes

Primary references:

- Gkeyll docs: https://gkeyll.readthedocs.io/en/latest/
- Gkeyll source: https://github.com/ammarhakim/gkeyll
- Gkeyll SOL turbulence paper: https://arxiv.org/abs/1610.09056
- GENE-X CPC paper: https://www.sciencedirect.com/science/article/pii/S0010465521000989
- GENE-X spectrally accelerated edge/SOL work:
  https://arxiv.org/abs/2411.09232

Relevant findings:

- Gkeyll is a compiled kinetic/multifluid code with MPI, CUDA, NCCL, OpenBLAS,
  SuperLU, generated DG kernels, and a separate postprocessing ecosystem.
- The source tree contains machine-specific CPU/GPU configuration scripts and
  separates `core`, `gyrokinetic`, `moments`, `pkpm`, and `vlasov` solvers.
- GENE-X and Gkeyll define a kinetic outer boundary for what reduced-fluid
  results should and should not claim.

Implication for `jax_drb`:

- `jax_drb` should not claim kinetic fidelity. It should instead be explicit
  about reduced-fluid closure assumptions and use kinetic codes as context for
  future-model boundaries.
- The practical engineering lesson is to separate solver kernels, machine
  configuration, tests, and postprocessing enough that accelerator claims can
  be reproduced.

### Differentiable Scientific Computing Baseline

Primary references:

- JAX JVP/VJP docs:
  https://docs.jax.dev/en/latest/jacobian-vector-products.html
- JAX persistent compilation cache:
  https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- JAX sharded computation:
  https://docs.jax.dev/en/latest/sharded-computation.html
- JAX `shard_map`:
  https://docs.jax.dev/en/latest/notebooks/shard_map.html
- Lineax matrix-free solves:
  https://docs.kidger.site/lineax/examples/no_materialisation/
- Diffrax adjoints: https://docs.kidger.site/diffrax/api/adjoints/
- Equinox paper: https://arxiv.org/abs/2111.00254
- Lineax paper: https://arxiv.org/abs/2311.17283
- JAX-Fluids paper:
  https://www.sciencedirect.com/science/article/pii/S0010465522002466
- Algorithmic differentiation for plasma edge codes:
  https://www.sciencedirect.com/science/article/pii/S0021999123004989

Relevant findings:

- JAX can compute JVPs and VJPs directly and can batch basis pushes/pulls using
  `vmap`.
- The persistent compilation cache is a real performance feature and should be
  configured before the first compilation.
- CPU multi-device testing can use explicit host-device count configuration.
- `shard_map` is the forward-looking SPMD API for mapping work across device
  shards.
- Lineax documents exactly the Newton-matrix use case that matters here:
  matrix-vector products can be built from `jax.jvp` without materializing the
  full Jacobian.
- Diffrax adjoints distinguish direct solver differentiation, forward-mode
  sensitivity, reverse/checkpointed sensitivity, and implicit-function
  differentiation for steady-state solves.

Implication for `jax_drb`:

- The main differentiability target is a pure-JAX residual path with
  matrix-free linearization. Finite-difference sparse Jacobian assembly is a
  useful bridge, not the final differentiable solver architecture.
- The scaling target should separate single-solve latency from ensemble and
  parameter-scan throughput. JAX is strongest when a family of solves can be
  batched or sharded rather than driven through host-side loops.

## Current Strengths

The codebase already has several strong surfaces:

- install metadata is clean: no pinned runtime dependency versions, Python
  `>=3.10`, `tomli` only for Python versions without `tomllib`
- public CLI supports TOML decks, structured run logs, restart bundles,
  detailed runtime progress, ETA on long native transient lanes, precision
  control, and persistent compilation cache setup
- PyPI publishing workflow exists and uses OIDC-based publishing on tags,
  releases, or manual dispatch
- CI covers Python 3.10, 3.11, and 3.12, although only a narrow shipping slice
  is active while CI billing is constrained
- operator-level validation campaigns exist for MMS, reactions, collisions,
  neutral diffusion, collision closure, target recycling, anomalous diffusion,
  live reference reruns, neutral-mixed boundary mismatch, local CPU scaling,
  differentiability, and 3D selected-field reductions
- recent performance work reduced the heavy open-field D/T recycling one-step
  local mean from about `75.3 s` to about `52.76 s` through source allocation
  cleanup and cached target-boundary geometry
- recent neutral-mixed work reduced the one-step local mean from about
  `1.15 s` to about `0.63 s` through vectorized gradient magnitude evaluation
- the validation layer has a common publication-quality figure helper and
  stores machine-readable JSON/NPZ/PNG artifacts for promoted campaigns

## Current Gaps And Risks

### Repository Hygiene

The current repository is too heavy for a clean research release:

- pre-slimming `.git` was about `428M`, with about `411M` in pack files
- pre-slimming working tree was about `1.7G`
- pre-slimming `docs/data` was about `22M`
- `references` is about `26M`
- the tracked `legacy/` tree was about `26M` and has been moved out of the
  active release branch
- the largest historical blobs are old GIFs and legacy/readme assets, including
  one historical `examples/assets/readme/drb2d_kh.gif` object of about
  `63.45 MiB`
- current large tracked baselines include several multi-MB NPZ/GIF artifacts
  that are scientifically useful but should be explicitly classified as
  release artifacts, optional artifacts, or externally hosted artifacts

This is a blocker for a lightweight public research code. The next cleanup must
separate three classes of data:

- minimal baseline artifacts required by fast CI
- documentation/demo artifacts useful for users
- heavy research artifacts that should live in releases, external storage, or
  the paper/benchmark artifact repository

### Paper And Legacy Material In The Code Repo

The active code repo previously contained historical legacy code, manuscript/JCP
docs, manuscript figure artifacts, and publication-only example wrappers. Those
surfaces have been removed from the active release branch after confirming that
the paper repository already carries the relevant archive. Some validation campaigns
should absolutely remain in the code repo because they are tested, documented,
and useful. But paper-only prose plans and manuscript-only panel generators
belong outside the code release branch.

The target rule is:

- keep reusable validation code and tested scientific figure generators
- move manuscript drafting, paper planning, paper-only panel composition, and
  historical logs out of the code release branch
- keep `legacy/` only if it is moved to a separate archival branch or external
  archive; it should not be imported or shipped as part of the active package

### Architecture

The largest active source/test files remain too broad:

- `src/jax_drb/native/recycling_1d.py`: about `3235` lines
- `src/jax_drb/native/runner.py`: about `2445` lines
- `src/jax_drb/native/neutral_mixed.py`: about `1403` lines
- `src/jax_drb/cli.py`: about `1133` lines
- `tests/test_native_integrated_2d_recycling.py`: about `2942` lines
- `tests/test_native_tokamak_cases.py`: about `2713` lines
- `tests/test_native_recycling_1d.py`: about `2005` lines
- `tests/test_native_runner.py`: about `1093` lines

The recent extraction work is correct but incomplete. The target structure is
still a set of subpackages with narrow responsibilities:

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
  runner/
    registry.py
    references.py
    cache.py
    execution.py
    comparison.py
    artifacts.py
```

The split must be behavior-preserving. Each extracted module needs direct unit
coverage before the broad integration tests are allowed to be the only safety
net.

### Solver And Differentiability

The biggest technical gap is the heavy implicit path:

- finite-difference sparse Jacobian assembly still dominates some heavy runs
- residual evaluation still crosses host/device boundaries on promoted
  recycling paths
- pack/unpack and prepared-state assembly still allocate and copy too much
- SciPy BDF/Newton paths are useful but block full end-to-end differentiability
- thread-parallel finite-difference coloring can help on a laptop, but it is
  not the final scaling model

The target solver stack should have two explicit tiers:

- compatibility tier: current NumPy/SciPy sparse finite-difference Newton/BDF
  path, optimized and validated, used until each lane is promoted
- differentiable tier: pure-JAX residual, JVP/VJP linearization, matrix-free
  Lineax/iterative solve or implicit-function sensitivity path, no unnecessary
  host barriers

The first differentiable-tier primitive is now in-tree and tested:
`build_sparse_jvp_jacobian` reuses the finite-difference coloring contract but
fills the sparse Jacobian from `jax.linearize` plus batched `jax.vmap` tangent
pushes. That gives a bridge for pure-JAX residuals that still need a
materialized sparse matrix. The stronger long-term path is the existing
matrix-free JAX Newton/GMRES solver, which applies the linearized Jacobian
action directly and avoids materializing a sparse matrix altogether. The
promotion rule is therefore explicit: do not claim a JAX-native implicit
backend for a lane until its residual can be differentiated without host-side
NumPy/SciPy barriers.

### Validation

The validation structure is strong but needs tighter promotion gates:

- one fully native open-field recycling transient backbone still needs to move
  from operational to exact/tightly bounded across RHS, one-step, and
  short-window surfaces
- the neutral-mixed boundary mismatch has a dedicated figure now, but the
  mismatch still needs a component-level fix
- full live 3D reference reruns are still missing; current 3D evidence is
  selected-field/reduced/scaffolded rather than broad production 3D
- TCV-X21 should remain a benchmark target, but it should not be promoted until
  the recycling/tokamak transient backbone is stable enough to produce profile
  and target comparisons honestly
- every paper-strength figure should already exist as a docs artifact from a
  tested validation campaign

### Hermes Parity, Runtime, And Memory Offender Register

The next validation push should explicitly rank the main offenders between
JAXDRB and Hermes-3 instead of treating each mismatch or slow run as an
isolated bug. The offender register should be regenerated whenever a promoted
Hermes-backed lane changes and should contain, for each case, the compare
surface, dominant field, dominant component when known, absolute error,
relative/scaled error, JAXDRB wall time, Hermes wall time, peak resident memory
or best available memory proxy, and the artifact path used to reproduce the
diagnosis.

The committed implementation of this triage layer is
`docs/data/hermes_offender_register_artifacts/data/hermes_offender_register.json`,
with a publication-ready summary figure at
`docs/data/hermes_offender_register_artifacts/images/hermes_offender_register.png`.
On the current promoted live matrix, the top parity target is
`neutral_mixed_one_step` on `NVh`, the top runtime target is
`recycling_dthe_one_step`, and the top measured peak-RSS ratio is also
`recycling_dthe_one_step` at about `0.95`. The current peak-RSS result is
useful because it rules out a broad native memory regression relative to
Hermès on this matrix; the remaining memory task is phase-resolved profiling of
Jacobian assembly, residual evaluation, packing, and artifact extraction.

The current priority parity offenders are:

- heavy 1D open-field recycling transients, especially neutral and recycling
  closure drift over longer windows
- neutral mixed transport, especially the boundary-local `NVh` mismatch that
  already has a dedicated boundary-audit figure
- integrated 2D production/recycling target-band residuals, historically led
  by `Pe`, `Pd+`, `NVd+`, and neutral-side transient terms depending on the
  rung
- direct tokamak multispecies recycling windows where the largest scaled
  mismatch is often a near-zero `NVd`/`NVt` field and therefore must be reported
  with absolute-error context
- OpenADAS/neon-enabled recycling paths where table lookup, radiation/source
  partitioning, and species-state preparation must stay component-local

The current priority runtime offenders are:

- sparse finite-difference Jacobian construction and repeated residual calls in
  the host-backed implicit recycling path
- active-state pack/unpack and prepared-state reconstruction inside implicit
  iterations
- target boundary geometry, target recycling, neutral diffusion, and collision
  closure assembly on heavy multispecies cases
- long-history artifact writing and compare-surface extraction for transient
  validation
- any live Hermes rerun that requires a high-rank launch or writes large dumps
  before the guarded compare surface is extracted

The first concrete runtime-instrumentation step is now in the shared implicit
solver: sparse Newton steps expose residual, Jacobian, linear-solve,
line-search, and fallback diagnostics, and the colored finite-difference
Jacobian builder reuses a precomputed extraction plan across refreshes. The
next `recycling_dthe_one_step` run should therefore report both external
cProfile/JAX trace evidence and solver-phase timings from the same solve.

The current priority memory offenders are:

- materialized sparse Jacobians and temporary colored finite-difference states
  in heavy implicit solves
- repeated full-field copies in state preparation, boundary reconstruction, and
  active-domain packing
- stored long histories and optional reference snapshots that are useful for
  validation but should not be required for ordinary package use
- multi-MB NPZ/GIF artifacts that need explicit classification as CI baselines,
  documentation assets, release assets, or external research artifacts

The register should drive the next fixes in this order:

1. localize the largest parity error to an equation term, closure, boundary
   rule, or compare-window convention
2. add a direct unit/operator test and, where useful, a small artifact-producing
   diagnostic plot for that term
3. profile the same lane before and after the fix, including memory when
   available
4. update the validation report with both absolute and relative errors so
   near-zero-field mismatches are not overstated
5. only then widen the case matrix or promote a new manuscript/docs figure

### CI And Automation

The current workflow is intentionally narrow because of billing constraints.
That is acceptable short-term but not the final research-code standard.

The final automated gate should have tiers:

- `fast`: pure unit/operator/release-surface tests on every push/PR
- `research-fast`: `scripts/run_fast_research_checks.py` default slices on PRs
  when billing is available
- `coverage`: promoted solver/public slice with `95%` target
- `artifact`: regenerate selected lightweight validation artifacts and compare
  schema/metrics
- `nightly`: slower live/reference-backed cases, convergence campaigns, memory
  and performance reports
- `release`: build distributions, import check, metadata check, tag/release
  publishing to PyPI

## Physics Gates

The promoted model surface should be guarded by physical, numerical, and
software gates. A case is not research-grade until it passes all gates relevant
to its claim.

### Equation And Closure Gates

Every implemented term must have:

- equation definition in docs
- implementation file/function listed in docs
- unit tests on representative scalar/vector fields
- limiting-case test where the term analytically vanishes or reduces to a
  simple expression
- reference-backed test when it corresponds to an external component
- artifact-producing campaign when it is scientifically important enough to be
  shown in the paper or docs

Required closure families:

- density, pressure, energy, and parallel-momentum evolution
- ExB, diamagnetic, anomalous, and classical transport where implemented
- Braginskii collisions, friction, heat exchange, conduction, viscosity, and
  thermal force where implemented
- reaction parsing, AMJUEL/ADAS rates, ionization, recombination, charge
  exchange, radiation, and source partitioning
- sheath boundary, no-flow boundary, neutral boundary, recycling, pumping, and
  target-localized fast/thermal recycling
- neutral parallel diffusion and neutral mixed transport
- electrostatic vorticity/potential and electromagnetic selected-field lanes

### Verification Gates

Minimum verification suite:

- MMS spatial convergence for promoted finite-difference operators
- time-step refinement for promoted transient lanes
- conservation or controlled-loss tests for density, energy, and momentum
  where the modeled physics allows it
- symmetry and zero-flux tests for boundary operators
- restart/resume equivalence
- pack/unpack/state-layout round trips
- finite-difference Jacobian versus matrix-free JVP consistency
- `grad`, `jvp`, and `vjp` versus finite-difference checks on promoted
  differentiable lanes

### Code-To-Code Gates

Minimum code-to-code promotion ladder:

- one-RHS parity
- one-step parity
- short-window parity
- longer-window diagnostic comparison when transient drift matters
- absolute and relative errors, with near-zero reference fields explicitly
  identified
- same-machine runtime comparison for any performance claim
- capability tier stored in manifest, run log, docs, and validation report

### Benchmark And Experiment-Facing Gates

Priority benchmark packages:

- TORPEX/blob-style seeded filament dynamics for compact electrostatic lanes
- TCV-X21 diverted L-mode profiles and target diagnostics
- SOLPS-ITER/Hermes-3-style neutral observables where available
- 1D detachment and recycling scans with target temperature, ionization source,
  pressure loss, radiation/source balance, and upstream/target profiles
- geometry-portability tests across diverted tokamak, traced-field-line, and
  VMEC-style stellarator data

## Performance And Memory Plan

The performance target is not "make one plot look good." The target is low
wall time, low memory use, reproducible scaling evidence, and no hidden loss of
fidelity.

### Measurement Rules

- Run isolated cProfile measurements for heavy host-side cases.
- Run JAX traces for compiled native kernels.
- Capture device-memory profiles for GPU-visible lanes.
- Use XLA dumps only for targeted fusion/layout investigations.
- Always pair speed improvements with the same fidelity surface.
- Report CPU and GPU separately.
- Separate single-solve latency from ensemble/scan throughput.

### Hotspots To Attack

Current heavy-path hotspots:

- sparse finite-difference Jacobian assembly
- repeated residual evaluation under SciPy implicit stepping
- neutral parallel diffusion and collision closure
- target recycling and target boundary-source assembly
- prepared-state setup and boundary reconstruction
- active-field pack/unpack and scalar feedback packing
- host-device synchronization in profiling and artifact paths

### JAX-Native Solver Direction

The next solver architecture should use:

- pure-JAX residual functions for promoted lanes
- static PyTree state layouts instead of repeated dict/array reconstruction
- `jax.jit` around residual kernels with shape-stable arguments
- `jax.jvp` for Jacobian-vector products
- `jax.vjp` for adjoint and objective gradients
- `vmap` for batched sensitivities, finite-difference validation, and ensemble
  scans
- Lineax matrix-free operators for Newton/Krylov experiments
- Diffrax forward/direct/implicit adjoint ideas where the problem is naturally
  an ODE or steady-state solve

The first target should be a compact open-field recycling residual on a fixed
layout. It should be promoted only after it matches the current NumPy/SciPy
path on the existing one-RHS and one-step gates.

The implementation sequence for that first target is:

1. define a small fixed-layout recycling residual dataclass/PyTree with arrays
   only and no mutable dictionaries in the hot path
2. port the measured hot kernels one at a time: reaction/source rates,
   collision closure, neutral diffusion, target recycling, and BDF residual
   assembly
3. add unit parity tests for each port against the current NumPy helper
4. add derivative gates for each promoted kernel: JVP versus centered finite
   difference, VJP/gradient versus finite difference for scalar quantities of
   interest, and batched `vmap` agreement with serial evaluation
5. benchmark the pure-JAX residual with three solver backends: current sparse
   finite-difference compatibility, batched sparse-JVP materialization, and
   matrix-free JVP/GMRES
6. only after parity and timing are stable, wire the JAX residual into a
   promoted one-step or short-window recycling lane

The first helper-level port under item 2 is complete: the atomic-rate helpers
for AMJUEL, OpenADAS, and hydrogen charge exchange now preserve JAX arrays and
have `jit`/`grad` tests. The existing single-isotope reaction-source formulas
also preserve JAX arrays through the source terms themselves. The next
implementation step is not another rate formula. It is the fixed-layout
reaction-source accumulator that can call those helpers without Python
dictionaries or host conversions inside the residual.

The first artifact-level gate for this port is the atomic-rate
differentiability campaign. It produces a publication-ready figure comparing
JAX autodiff slopes against centered finite differences on the AMJUEL,
OpenADAS, and hydrogen charge-exchange rate surfaces.

The first fixed-layout reaction-source kernel is also in-tree for the
hydrogenic same-isotope reaction block. It returns array-only source terms,
matches the existing dictionary implementation, and supports `jit`/`grad`.
The next residual refactor should extend this pattern to the D/T/He
multispecies reaction matrix and then replace the dictionary accumulation in
the packed recycling residual.
That D/T/He extension now exists at helper level:
`fixed_layout_dthe_reaction_sources` stacks D, T, and He neutral/ion source
arrays, includes the D/T same-isotope and cross-isotope charge-exchange matrix,
matches the dictionary source path on the Hermès `1D-recycling-dthe` deck, and
is differentiable under JAX. The next refactor is therefore a wiring task:
replace the mutable reaction-source accumulation inside the packed residual
with this fixed-layout PyTree and measure the BDF residual/Jacobian call count
again.

The direct-tokamak recycling validation surface now also has a target/neutral
observable campaign. It promotes the `tokamak_recycling_dthe_one_step` lane
from scalar parity evidence to target-index charged-density profiles,
momentum-flux proxies, neutral parallel-density buildup, and target
electron-temperature proxy errors. This is the profile-observable bridge needed
before the JAXDRB paper compares tokamak recycling behavior with the TCV-X21,
SOLPS-ITER, and Hermes-3 validation literature.

### CPU Parallelization

Laptop CPU speedups are possible, but the right target is not one host-side
BDF solve. The right CPU-parallel targets are:

- batched parameter scans
- batched RHS/JVP evaluations
- color-group Jacobian evaluations where the residual is still host-side
- ensemble steady-state fixed-work solves
- uncertainty propagation and inverse-design batches

Implementation choices:

- keep `JAX_DRB_HOST_DEVICE_COUNT` for explicit CPU device experiments
- use `vmap` for vectorized single-device batch throughput
- use `shard_map` or `pmap` only for workloads that are naturally independent
  across CPU devices
- keep SciPy thread parallelism as a compatibility optimization, not a primary
  scaling claim

### Memory Plan

- Replace repeated full-field copies with view/slice-aware active-domain
  layouts where possible.
- Keep long histories optional and compressed.
- Store validation arrays only when needed for regression or public figures.
- Move heavyweight research artifacts out of the package distribution and, if
  necessary, out of the main git branch.
- Add memory regression checks for promoted large examples.

## Documentation Plan

The documentation should be useful even if nobody reads the paper.

Required public documentation:

- installation and quick start
- CLI and Python API
- input deck schema and examples
- runtime progress, ETA, restart, outputs, and provenance
- physics models with equations and implementation mapping
- code structure and contribution guide
- validation strategy and capability tiers
- validation gallery with reproducible artifacts
- profiling and performance workflow
- differentiability guide with sensitivity/UQ/inverse-design examples
- testing guide for users and contributors
- release/PyPI packaging guide

Immediate documentation gaps:

- one active roadmap should point to this plan instead of scattering decisions
  across historical files
- docs should include a testing/how-to-validate page that explains which tests
  are unit, regression, parity, benchmark, artifact, and slow
- docs should include an examples taxonomy so users can distinguish tutorials,
  benchmarks, engineering campaigns, and paper/research artifact generators
- code-facing docs should avoid paper-only language unless the file is clearly
  in the paper repo or a manuscript-specific artifact area

## Examples Plan

Examples should be easy to run and informative. The target taxonomy:

- `examples/tutorials/`: first-run examples, no private references, fast
- `examples/benchmarks/`: literature-anchored comparisons
- `examples/validation/`: artifact-producing campaign entry points
- `examples/performance/`: profiling, scaling, memory
- `examples/differentiability/`: sensitivity, UQ, inverse design,
  optimization
- `examples/geometry/`: tokamak, traced-field-line, stellarator geometry

Current `examples/engineering/*` can remain during transition, but the final
layout should be user-facing rather than internally named.

## CI/CD And PyPI Plan

Current state:

- `.github/workflows/test.yml` runs a narrow targeted shipping slice on Python
  3.10, 3.11, and 3.12
- `.github/workflows/publish-pypi.yml` builds distributions and publishes to
  PyPI on tags/releases/manual dispatch

Target state:

- PR gate:
  - packaging metadata
  - release surface
  - fast unit/operator tests
  - runtime precision/import tests on Python 3.10-3.12
- Research gate:
  - `scripts/run_fast_research_checks.py`
  - promoted solver slice coverage
  - artifact schema tests
- Nightly/manual heavy gate:
  - live reference reruns
  - convergence campaigns
  - selected memory/performance campaigns
  - optional remote GPU smoke/audit if credentials and billing allow
- Release gate:
  - build wheel/sdist
  - import check from wheel
  - artifact-free install check
  - PyPI publish on semver tag

## Active Execution Sequence

### Phase 0: Planning And Release Hygiene

Deliverables:

- add this consolidated plan and link it from the refactoring/testing docs
- fix stale repository URLs in user-facing docs
- add the new neutral-mixed boundary campaign to the release-surface audit
- decide which current docs/artifacts are release-critical versus paper-only
- create an artifact-pruning and history-rewrite checklist

Exit criteria:

- public docs point to the active plan
- release-surface tests pass
- no accidental edits to unrelated local artifacts

### Phase 1: Repository Slimming

Deliverables:

- remove `legacy/` from the active release branch after confirming it is
  archived in the paper repository
- remove manuscript-only docs/examples/artifacts from the active release branch
- remove old historical large GIFs and unused blobs with `git filter-repo`
- keep only small, necessary baselines in the main branch
- document where heavyweight benchmark artifacts are stored

Exit criteria:

- `.git` pack size reduced by at least half
- active checkout contains no paper-only planning files
- package distribution excludes research artifacts that are not needed at
  runtime

### Phase 2: Architecture Split

Deliverables:

- split `recycling_1d.py` into `native/recycling/*`
- split `neutral_mixed.py` into `native/neutral/*`
- split `runner.py` into `native/runner/*`
- split large tests into operator, parity, campaign, and slow-transient files
- preserve public imports with compatibility shims during transition

Exit criteria:

- no active source module over about 1000 lines unless explicitly justified
- extracted modules have direct unit tests
- existing promoted parity/campaign tests remain green

### Phase 3: Solver And Differentiability Backbone

Deliverables:

- implement a pure-JAX residual for the smallest open-field recycling lane
- add JVP-based Jacobian-action checks against finite-difference sparse
  Jacobian columns
- prototype Lineax matrix-free Newton/Krylov on the compact lane
- add implicit-function sensitivity for a steady-state fixed-work solve
- remove unnecessary host-device barriers in recycling residual setup

Exit criteria:

- compact lane matches current solver on one-RHS and one-step gates
- `grad`, `jvp`, and finite-difference sensitivities agree on a promoted
  differentiable objective
- runtime and memory are no worse than the current compatibility path on the
  same case, or the tradeoff is explicitly documented

### Phase 4: Fidelity Closure

Deliverables:

- maintain a Hermes parity/runtime/memory offender register that ranks cases by
  dominant field, component, absolute error, scaled error, wall time, and memory
  footprint
- fix neutral-mixed `NVh` boundary-local mismatch or document the exact
  bounded cause with an equation-level explanation
- promote the open-field recycling transient ladder through one-RHS,
  one-step, short-window, and a selected longer diagnostic window
- extend target/source/closure campaigns to exact term-by-term equations in
  docs
- add direct tests for every closure branch used in the promoted lanes

Exit criteria:

- each promoted lane has absolute and relative error metrics
- mismatch causes are component-local and documented
- no paper/docs figure relies on an untested path

### Phase 5: Benchmark Expansion

Deliverables:

- TORPEX/blob benchmark package
- TCV-X21 profile/target/neutral observable package
- detachment/recycling scan package
- geometry-portability package across tokamak, traced-field-line, and
  stellarator-style data
- performance and memory campaign on promoted native paths

Exit criteria:

- benchmark figures look like literature figures: profiles, targets,
  convergence, scans, source maps, and physically interpretable diagnostics
- all benchmark packages have runnable scripts, JSON/NPZ artifacts, and tests
- claim boundary is explicit in docs and run logs

### Phase 6: Full Release Automation

Deliverables:

- turn the fast research gate into a CI job when billing is available
- use `scripts/run_promoted_solver_coverage.py --audit` to track the promoted
  solver/public slice until the refactor raises it from the April 23, 2026
  local baseline of `73%` to the required `95%`
- add coverage gate for promoted solver/public slice once that audit clears the
  threshold without `--audit`
- add nightly/manual heavy reference reruns
- add release smoke install from wheel
- validate PyPI workflow on a test release before versioned public release

Exit criteria:

- a clean clone can install quickly with `pip install jax-drb`
- source install works with `pip install -e .`
- CI gives fast feedback without running unbounded heavy jobs on every PR
- release builds and publishing are reproducible

## Definition Of Done

`jax_drb` is research-grade when:

- the repo is slim enough to clone quickly
- install is simple and unpinned
- the public API, CLI, examples, and docs are coherent
- the promoted solver surface has meaningful `95%` coverage
- every promoted equation term has unit/operator tests and docs
- every promoted physics lane has verification, regression, and validation
  evidence appropriate to its claim
- performance claims are tied to profiler artifacts and same-machine
  comparisons
- differentiability claims are tied to `grad`/`jvp`/`vjp` tests and
  finite-difference agreement
- CI/CD can test and publish without manual steps
- all unsupported surfaces are clearly labeled rather than implied
