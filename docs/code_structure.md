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
