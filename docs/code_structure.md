# Code Structure

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
