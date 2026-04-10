# Physics Models

This page is the technical map from the governing equations to the source tree. It is meant to help both new users and developers find where a model lives before they change a case, add a term, or debug a result.

## Model Families

`jax_drb` currently organizes its native physics into a few main families:

- open-field recycling and multispecies edge/SOL transport
- electrostatic drift-wave and vorticity systems
- blob and interchange-style turbulence benchmarks
- Alfven-wave and annulus electromagnetic benchmarks
- direct tokamak geometry transport, recycling, and turbulence ladders

## Drift-Reduced Braginskii Core

The promoted electrostatic and open-field lanes are built around drift-reduced Braginskii-style density, momentum, pressure, and potential evolution.

Representative equations on supported lanes include:

- continuity:
  - `∂t n + ∇·Γ = S_n`
- parallel momentum:
  - `∂t (n V_∥) + ∇·(Γ V_∥) = -∇_∥ p + F_coll + F_E + ∇_∥·Π_∥ + S_m`
- pressure / energy:
  - `∂t p + ∇·(p u) + γ p ∇·u = Q_coll + Q_src + Q_cond`
- vorticity / potential closure:
  - Boussinesq and related elliptic closures depending on the promoted lane

Primary source files:

- open-field and recycling closure:
  - [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)
  - [src/jax_drb/native/open_field.py](../src/jax_drb/native/open_field.py)
- mesh and metric handling:
  - [src/jax_drb/native/mesh.py](../src/jax_drb/native/mesh.py)
  - [src/jax_drb/native/metrics.py](../src/jax_drb/native/metrics.py)
- transport helpers:
  - [src/jax_drb/native/transport.py](../src/jax_drb/native/transport.py)
- runner/orchestration:
  - [src/jax_drb/native/runner.py](../src/jax_drb/native/runner.py)

## Sheath And Recycling Closures

The open-field and tokamak recycling lanes use explicit target/sheath boundary conditioning, recycling source assembly, and neutral/ion feedback terms.

Key source locations:

- sheath boundary conditioning:
  - [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)
- recycling source diagnostics and transient stepping:
  - [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)
- restart/state packing for the recycling transient:
  - [src/jax_drb/runtime/output.py](../src/jax_drb/runtime/output.py)
  - [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)

Important operator terms currently under active review include:

- parallel ion viscosity `DivPiPar`
- target-corner guard-cell semantics
- reaction/source partitioning
- non-orthogonal transport terms in production-style geometries

## Electrostatic Drift-Wave And Blob Lanes

The benchmark electrostatic lanes cover:

- coupled density / electron-momentum / vorticity evolution
- potential inversion
- ExB transport
- blob curvature/interchange dynamics

Primary source files:

- drift-wave:
  - [src/jax_drb/native/drift_wave.py](../src/jax_drb/native/drift_wave.py)
- blob:
  - [src/jax_drb/native/blob2d.py](../src/jax_drb/native/blob2d.py)
- vorticity and elliptic operators:
  - [src/jax_drb/native/vorticity.py](../src/jax_drb/native/vorticity.py)
  - [src/jax_drb/solver/elliptic.py](../src/jax_drb/solver/elliptic.py)

## Electromagnetic Lanes

The current electromagnetic ladder is benchmark-first. It includes Alfven-wave and annulus-style validation problems with compact promoted surfaces.

Primary source files:

- electromagnetic operators:
  - [src/jax_drb/native/electromagnetic.py](../src/jax_drb/native/electromagnetic.py)
- Alfven-wave benchmark utilities:
  - [src/jax_drb/validation/alfven_wave.py](../src/jax_drb/validation/alfven_wave.py)
  - [src/jax_drb/validation/alfven_wave_meeting.py](../src/jax_drb/validation/alfven_wave_meeting.py)

## Neutral And Atomic Physics

Neutral and recycling-capable lanes depend on packaged rate data and source builders.

Primary source files:

- neutral benchmark analysis:
  - [src/jax_drb/validation/neutral_mixed.py](../src/jax_drb/validation/neutral_mixed.py)
- atomic/radiation data packaging:
  - [src/jax_drb/data/atomic_rates](../src/jax_drb/data/atomic_rates)
- source assembly and reaction evaluation:
  - [src/jax_drb/native/recycling_1d.py](../src/jax_drb/native/recycling_1d.py)

## Numerics And Solvers

The numerics are intentionally split between:

- native explicit/structured update kernels
- elliptic solvers for potential closures
- implicit or stiff transient stepping on selected promoted lanes

Primary source files:

- implicit solvers:
  - [src/jax_drb/solver/implicit.py](../src/jax_drb/solver/implicit.py)
- elliptic solvers:
  - [src/jax_drb/solver/elliptic.py](../src/jax_drb/solver/elliptic.py)
- runtime precision and performance settings:
  - [src/jax_drb/runtime/__init__.py](../src/jax_drb/runtime/__init__.py)
  - [src/jax_drb/runtime/performance.py](../src/jax_drb/runtime/performance.py)

## Output, Restart, And Provenance

Promoted user-facing runs produce:

- summary JSON
- arrays NPZ
- restart NPZ
- verbose event log JSON

Primary source files:

- CLI and argument model:
  - [src/jax_drb/cli.py](../src/jax_drb/cli.py)
- portable payload and restart writing:
  - [src/jax_drb/runtime/output.py](../src/jax_drb/runtime/output.py)
- parity/benchmark payload helpers:
  - [src/jax_drb/parity/portable.py](../src/jax_drb/parity/portable.py)
  - [src/jax_drb/parity/arrays.py](../src/jax_drb/parity/arrays.py)
  - [src/jax_drb/parity/compare.py](../src/jax_drb/parity/compare.py)

## Validation And Promotion Rules

Before a capability is promoted to `native_exact`, the working rule is:

- one-RHS parity on the smallest exercising case
- one-step parity on the same case
- short-window parity when transient behavior matters
- operator or boundary unit tests for every new branch
- at least one physics-facing diagnostic
- restart equivalence when the workflow is user-facing
- artifact and provenance checks for CLI/example surfaces

The reviewer-facing version of that contract is in:

- [research_grade_validation_matrix.md](research_grade_validation_matrix.md)
