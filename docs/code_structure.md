# Code Structure

!!! note "Plan authority"
    This page is a developer map and refactoring context appendix. The active
    execution plan is
    [Research-Grade Execution Plan](research_grade_execution_plan.md). If this
    page conflicts with that plan, follow the execution plan and update this
    page afterward.

This page is the developer-facing map of the `jax_drb` source tree. The goal is
to make the package understandable before reading the monolithic solver files or
the validation campaigns in detail.

The comparison standard is the architecture and validation culture seen in
codes such as [BOUT++](https://arxiv.org/abs/0810.5757),
[Hermes-3](https://www.sciencedirect.com/science/article/pii/S0010465523003363),
[GBS](https://www.sciencedirect.com/science/article/pii/S0021999122003280), and
[TOKAM3X](https://www.sciencedirect.com/science/article/pii/S0021999116301838):
separate the governing operators from the orchestration layer, separate
verification from benchmark validation, and keep geometry, numerics, and
plotting reusable.

The validation package now also carries a shared publication-plot helper in
[src/jax_drb/validation/publication_plotting.py](../src/jax_drb/validation/publication_plotting.py).
That helper is part of the research-grade validation surface: the figure
standard should live next to the tested campaigns, not only in downstream paper
scripts.

## Package Map

The current top-level layout is:

- `src/jax_drb/native`
  native solvers and problem-family implementations
- `src/jax_drb/solver`
  reusable linear, elliptic, and implicit-solver helpers
- `src/jax_drb/validation`
  benchmark campaigns, geometry diagnostics, plots, and publication-oriented
  artifacts
- `src/jax_drb/parity`
  portable summary/array payload helpers and comparison tooling
- `src/jax_drb/config`
  BOUT/HERMES-style input parsing and numeric option resolution
- `src/jax_drb/runtime`
  runtime configuration, precision, profiling, and execution helpers
- `src/jax_drb/reference`
  curated reference-case metadata

## Current Responsibilities

The current native solver families are:

- `fluid_1d.py`
  compact manufactured-solution and differentiable verification lane
- `vorticity.py`, `blob2d.py`, `drift_wave.py`, `electromagnetic.py`
  reduced benchmark and turbulence families
- `recycling_1d.py`
  open-field and direct-tokamak recycling, reactions, sheath, controller, and
  implicit transient backbone
- `neutral_mixed.py`
  mixed neutral transport and exchange closures
- `runner.py`
  case resolution, deck execution, restart orchestration, and artifact writing

The validation layer contains four kinds of modules, although they are not yet
split cleanly on disk:

- campaign builders
- geometry adapters and diagnostics
- plotting/report helpers
- publication-facing summary packages

## Refactor Direction

The refactor plan in [refactoring_plan.md](refactoring_plan.md) moves the code
toward smaller internal namespaces:

- `native/recycling/`
- `native/neutral/`
- `native/tokamak/`
- `validation/campaigns/`
- `validation/geometry/`
- `validation/plots/`
- `validation/reports/`

The first structural extraction in that direction is the packed-state layout
layer used by the implicit recycling solver:

- [src/jax_drb/native/recycling_layout.py](../src/jax_drb/native/recycling_layout.py)

That module exists so the active-domain packing and unpacking rules can be unit
tested directly, instead of being implied only through large transient-solver
tests.

The next solver-facing extraction is the fixed-layout recycling residual lane:

- [src/jax_drb/native/recycling_fixed_residual.py](../src/jax_drb/native/recycling_fixed_residual.py)

That module defines `RecyclingFixedState`, a JAX PyTree containing active
field blocks and controller scalars, plus transformable backward-Euler and BDF2
residual builders. It is the migration target for the heavy recycling residual:
the existing dictionary/full-field path remains the Hermès-compatible
production path, while the fixed-layout state gives a small, directly tested
surface for JVPs, sparse-JVP Jacobian assembly, and eventual matrix-free
linearized solves. The same module owns both migration adapters. The
host-oracle bridge reconstructs full guard-cell fields and
controller-integral dictionaries, calls the current packed RHS, and returns a
fixed-state RHS for parity against the D/T/He Hermès recycling deck. The
active-array transient backend is the production-facing lane for new ports: it
uses the same fixed-state interface as the full-field bridge, returns
fixed-layout RHS arrays directly, and is JVP-tested without repacking through
the older field/feedback split callback. Coupled kernels that compute field
and controller-feedback derivatives from the same source evaluation can use
`build_fixed_array_state_rhs` to return a complete `RecyclingFixedState` in one
pass. A second adapter,
`build_fixed_full_field_array_rhs`, stages guard-cell kernels and closure terms
such as collision friction/heat exchange, neutral parallel diffusion, and
target recycling through the same fixed-state interface while each term is
still being migrated to active-array form. New source, collision, diffusion,
target, and sheath terms should enter through one of those adapters before
being promoted into the full transient solve. The current sheath extraction
follows that rule: no-flow electron preparation,
zero-current ion-sum reconstruction, simple ion, full ion, and full electron
sheath response formulas now live in
[src/jax_drb/native/open_field.py](../src/jax_drb/native/open_field.py) as
backend-preserving helpers, while the remaining full-field sheath orchestration
stays in [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)
until its Hermès parity gates are ready.
The production backward-Euler/BDF2 recycling steppers now construct their
nonlinear residual through this fixed-state bridge, and expose both
`sparse_jvp` and `jax_linearized` solver modes for transformable residual
surfaces. The legacy SciPy BDF history callback remains separate until the
entire heavy residual no longer needs host-backed dictionary assembly.

The next low-risk extraction is the recycling field metadata layer:

- [src/jax_drb/native/recycling_fields.py](../src/jax_drb/native/recycling_fields.py)

That module owns:

- evolving variable-name ordering
- field template construction
- runtime field-override application

These rules are small, but they are part of the implicit-state contract and are
therefore worth testing directly rather than only through end-to-end recycling
cases.

The current boundary-helper extraction is:

- [src/jax_drb/native/recycling_boundaries.py](../src/jax_drb/native/recycling_boundaries.py)

That module owns the small but scientifically relevant guard-cell rules used by
the recycling backbone:

- neutral target density extrapolation
- open-field scalar Neumann guards
- open-field scalar Dirichlet guards

These rules influence parity and compare-window surfaces, so they need direct
tests and should later feed artifact-producing benchmark campaigns when they are
used in literature-facing operator studies.

The current atomic-data and rate-layer extraction is:

- [src/jax_drb/native/recycling_atomic.py](../src/jax_drb/native/recycling_atomic.py)

That module isolates:

- packaged AMJUEL and OpenADAS table loading
- AMJUEL polynomial evaluation
- OpenADAS bilinear rate evaluation
- charge-exchange fit evaluation
- normalized reaction-rate and energy-loss helpers

This is an important split because it separates atomic-data handling from the
larger recycling residual assembly and makes the accuracy/performance boundary
of the reaction closures easier to test directly.

The current reaction/source assembly extraction is:

- [src/jax_drb/native/recycling_reactions.py](../src/jax_drb/native/recycling_reactions.py)

That module owns:

- reaction parsing for ionisation, recombination, and charge exchange
- grouped source, momentum, and energy assembly
- reaction diagnostics used by the recycling and reactions/collisions validation
  surfaces
- effective neutral ionisation and charge-exchange collision-rate helpers

This is the first recycling submodule whose outputs already map directly onto a
publication-facing validation package:

- [src/jax_drb/validation/reactions_collisions_campaign.py](../src/jax_drb/validation/reactions_collisions_campaign.py)

The current collision-frequency and viscosity-input extraction is:

- [src/jax_drb/native/recycling_collisions.py](../src/jax_drb/native/recycling_collisions.py)

That module isolates:

- charge-weighted electron-density assembly for multispecies states
- Braginskii-style collision-frequency assembly across electron, ion, and
  neutral pairs
- ion-parallel-viscosity collisionality, collision time, and viscosity-coefficient
  inputs

This is a scientifically meaningful split because it separates the collisional
closure backbone from the larger recycling residual assembly. It also maps
directly to the profile-level collisionality and charge-exchange figures now
produced by:

- [src/jax_drb/validation/reactions_collisions_campaign.py](../src/jax_drb/validation/reactions_collisions_campaign.py)

The current feedback-controller state extraction is:

- [src/jax_drb/native/recycling_feedback.py](../src/jax_drb/native/recycling_feedback.py)

That module isolates:

- upstream density-error evaluation on active recycling states
- trapezoidal controller-integral updates
- predictor-stage controller-integral updates
- integral sanitization and compact vector packing helpers

This matters because the controller-oriented validation packages should not have
to depend on the full recycling residual file just to exercise controller-state
logic. It is also the path toward more direct tests of the temperature and
detachment controller lanes.

The current field-sanitization extraction is:

- [src/jax_drb/native/recycling_sanitize.py](../src/jax_drb/native/recycling_sanitize.py)

That module isolates:

- density-floor enforcement
- temperature and pressure floor enforcement
- charge-weighted electron-pressure floor reconstruction

This is a useful split because these rules are small, branchy, and easy to test
directly, yet they affect solver robustness and controller behavior on the
recycling lanes.

The current recycling setup and runtime-model extraction is:

- [src/jax_drb/native/recycling_setup.py](../src/jax_drb/native/recycling_setup.py)

That module now owns:

- open-field species template construction from BOUT-style decks
- literal-reference and field-expression evaluation for setup-time options
- explicit pressure-source normalization
- density-feedback controller loading and source-shape normalization
- runtime-model assembly for the implicit recycling backbone

This is a high-value seam because it separates deck interpretation and runtime
model construction from the residual assembly itself. It also makes the
equation-to-implementation bridge clearer in the docs and future paper:
scientifically meaningful setup contracts such as source normalization,
controller loading, and evolving-field ordering no longer live only inside the
large recycling solver file.

The current prepared-state and field-conditioning extraction is:

- [src/jax_drb/native/recycling_state.py](../src/jax_drb/native/recycling_state.py)

That module now owns:

- soft floor and safe-temperature reconstruction helpers
- species velocity reconstruction from density and momentum
- axisymmetric profile reduction used by anomalous-diffusion closures
- target-guard merge helpers
- prepared-species-state construction before sheath and collisional closures

This split matters because it isolates the branchy preconditioning rules that
sit between raw species fields and the physical closures. Those rules affect
accuracy, solver robustness, and the meaning of compare-window states, so they
should be tested directly instead of being exercised only through the larger
recycling and sheath integration paths.

The current neutral parallel-diffusion closure extraction is:

- [src/jax_drb/native/recycling_neutral_diffusion.py](../src/jax_drb/native/recycling_neutral_diffusion.py)

That module now owns:

- component gating for the neutral parallel-diffusion family
- AFN versus multispecies collision-mode selection
- density, energy, and momentum parallel-diffusion assembly
- diffusion and closure diagnostics used to interpret the neutral closure

This split matters because neutral parallel diffusion is a distinct physical
closure family rather than just bookkeeping inside the recycling residual. It
is a good candidate for future literature-facing operator and closure figures,
so it should be directly testable and separable from the rest of the implicit
recycling backbone.

The current anomalous-diffusion extraction is:

- [src/jax_drb/native/recycling_anomalous_diffusion.py](../src/jax_drb/native/recycling_anomalous_diffusion.py)

That module now owns:

- anomalous `D`, `chi`, and `nu` coefficient resolution on recycling species
- orthogonal and non-orthogonal tokamak anomalous-transport assembly
- the `nz = 1` non-orthogonal `g23 / g_23` upwind operator used on the direct
  tokamak lane
- anomalous-transport diagnostics used by direct tests and public validation
  artifacts

This split matters because anomalous perpendicular transport is both a real
physics closure and a real geometry boundary. It should be explainable,
testable, and plottable without hiding the implementation inside the full
recycling residual file. It also now maps directly onto the public tokamak
operator package:

- [src/jax_drb/validation/tokamak_anomalous_diffusion_campaign.py](../src/jax_drb/validation/tokamak_anomalous_diffusion_campaign.py)

The current target-recycling support extraction is:

- [src/jax_drb/native/recycling_targets.py](../src/jax_drb/native/recycling_targets.py)

That module now owns:

- target recycling source assembly on prepared ion states
- current-free electron-velocity reconstruction from prepared ion densities
- the open-field centered gradient used by the electron force-balance path

This split matters because target recycling and boundary-conditioned electron
response are physically meaningful closure families, not just bookkeeping. They
now map directly onto the public prepared-state validation package:

- [src/jax_drb/validation/target_recycling_campaign.py](../src/jax_drb/validation/target_recycling_campaign.py)

The current collision/conduction closure extraction is:

- [src/jax_drb/native/recycling_collision_closure.py](../src/jax_drb/native/recycling_collision_closure.py)

That module now owns:

- Braginskii friction coefficients and thermal-force gating
- ion-ion thermal-force pair construction
- parallel ion viscous stress and divergence helpers
- species conductivity coefficients and collision-time selection
- the assembled collision/conduction closure application used by the recycling RHS

This split matters because the collisional closure is one of the main places
where reduced-fluid physics assumptions, transport coefficients, and numerical
stiffness meet. Keeping it isolated makes it much easier to test directly,
profile separately, and tie manuscript figures back to the actual implemented
closure formulas.

The first runner-side compare-window extraction is:

- [src/jax_drb/native/runner_compare.py](../src/jax_drb/native/runner_compare.py)

That module owns:

- guard-cell trimming for compare surfaces
- compare-variable selection for payload emission

These are not physics operators, but they are part of the public benchmark and
artifact contract. Extracting them makes the native execution path easier to
test directly without routing every check through the full runner dispatch.

The current runner execution-option extraction is:

- [src/jax_drb/native/runner_execution.py](../src/jax_drb/native/runner_execution.py)

That module owns:

- parity-mode to output-step mapping
- default-plus-case override merging
- restart-variable selection for the promoted placeholder families

These helpers are also part of the public execution contract. Pulling them out
keeps the runner file focused on dispatch and case execution rather than on
small policy functions.

The current runner cache-policy extraction is:

- [src/jax_drb/native/runner_cache.py](../src/jax_drb/native/runner_cache.py)

That module owns:

- capability-tier defaulting for native-only runs
- cache-path construction for integrated-2D, open-field, and tokamak reference
  bundles
- policy checks for which curated cases should read snapshot or history caches

This is a useful split because cache use is part of the promoted benchmark
contract rather than a private implementation detail. It affects runtime,
provenance, and reproducibility, so it should be directly tested instead of
being inferred only through the larger runner and tokamak integration suites.

The current runner reference-resolution extraction is:

- [src/jax_drb/native/runner_reference.py](../src/jax_drb/native/runner_reference.py)

That module owns:

- reference-root recovery from curated input paths
- application of case-specific override templates
- reference-case lookup by curated case name

These helpers sit on the reproducibility path for curated benchmark runs. They
determine which deck is loaded and how reference-root-dependent overrides are
resolved, so they are worth isolating and testing directly rather than leaving
them buried in the dispatch file.

The current recycling-specific runner-helper extraction is:

- [src/jax_drb/native/runner_recycling.py](../src/jax_drb/native/runner_recycling.py)

That module owns:

- direct recycling field-name and optional-diagnostic metadata
- source and velocity override extraction from cached optional fields
- guard-only restriction of field-template overrides
- recycling transient initial-case name mapping
- species-velocity reconstruction used by integrated 2D replay paths

These helpers influence replay fidelity and compare-surface construction for the
integrated and open-field recycling families. They are stable enough to unit
test directly and scientifically meaningful enough that they should not remain
hidden inside the main runner dispatch file.

The current transient solver-mode policy extraction is:

- [src/jax_drb/native/runner_solver_mode.py](../src/jax_drb/native/runner_solver_mode.py)

That module owns:

- configured recycling transient solver-mode parsing
- default solver-mode selection from parity mode and ion-species count
- the explicit BDF preference on promoted integrated and direct-tokamak one-step
  lanes

This is worth isolating because solver-mode choice affects both runtime and
scientific reproducibility. It is a policy layer rather than a transport
operator, and it should be directly tested rather than inferred only from
longer transient cases.

The current RHS-term assembly extraction is:

- [src/jax_drb/native/recycling_rhs_terms.py](../src/jax_drb/native/recycling_rhs_terms.py)

That module owns:

- electron pressure RHS decomposition into explicit, parallel-divergence,
  parallel-advection, and energy-source pieces
- ion density, pressure, and momentum RHS decomposition on prepared open-field
  states

This split matters because these assembled terms are the first place where the
closure stack becomes directly interpretable as a numerical balance. They are
already scientifically meaningful and directly tested, so they belong in a
small module rather than buried inside the larger recycling residual file.

## JAX Boundary

The architecture should keep the JAX boundary explicit:

- compact verification and reduced-operator lanes are already JAX-native and
  are appropriate for `jit`, `vmap`, `grad`, `jvp`, and `vjp`
- the heavy recycling backbone still includes host-backed sparse Newton and
  finite-difference Jacobian logic, so it should be documented and tested as a
  mixed JAX/NumPy/SciPy path rather than marketed as end-to-end differentiable

This distinction matches the current literature boundary between purely
differentiable JAX-native workflows and larger multiphysics edge codes whose
implicit backbones remain host-oriented, even when they expose differentiated
reduced operators or optimization workflows.
