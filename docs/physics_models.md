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

The promoted electrostatic, open-field, and direct-tokamak lanes are built
around drift-reduced Braginskii-style density, momentum, pressure, and
potential evolution.

At the level exposed in the current native ladders, the code is solving
discrete forms of the following model families.

### Continuity

For an evolved species density `n`:

```text
∂t n + ∇·Γ = S_n
```

where `Γ` is the resolved advective/diffusive flux and `S_n` collects
ionisation, recombination, recycling, pumping, and any case-specific source
terms.

### Parallel Momentum

For the evolved parallel momentum density `n V_∥`:

```text
∂t (n V_∥) + ∇·(Γ V_∥)
  = -∇_∥ p + F_coll + F_thermal + F_sheath + ∇_∥·Π_∥ + S_m
```

The exact active terms depend on the promoted lane:

- open-field recycling adds sheath, recycling, Braginskii friction, heat
  exchange, thermal force, and ion-viscosity closures;
- drift-wave/blob ladders carry the benchmark-consistent reduced momentum
  structure;
- direct tokamak ladders reuse the same promoted closures on the staged
  tokamak metric payload.

### Pressure / Energy

For the evolved scalar pressure `p`:

```text
∂t p + ∇·(p u) + γ p ∇·u = Q_cond + Q_coll + Q_src
```

with the right-hand side carrying the promoted conduction, collisional exchange,
radiation/source, and controller/recycling terms relevant to the active lane.

### Potential / Vorticity Closure

The electrostatic ladders solve benchmark-specific elliptic closures between
`phi`, `Vort`, and the underlying density/current state. On the promoted
benchmark surfaces this includes:

- Boussinesq closures on the vorticity ladder;
- drift-wave/quasineutral electron closures on the drift-wave ladder;
- benchmark-faithful `phi` reconstruction on the blob/interchange lanes.

### Electromagnetic Reduced Surfaces

The promoted electromagnetic benchmark lanes use compact selected-field
surfaces around:

```text
Ajpar = Σ_s Z_s n_s V_{∥,s}
```

plus the staged `Apar`/`NVe`/`Vort` benchmark closures documented in the
electromagnetic source and validation utilities.

## Numerical Algorithms

The code paths above are not solved with one monolithic algorithm. The current
native runtime uses a few distinct numerical patterns.

### Structured Finite-Volume / Flux-Form Updates

Most promoted 1D/2D native lanes use explicit flux-form field updates on the
structured mesh and metric payload. In practice this means:

- face reconstruction and metric-aware transport operators;
- explicit source assembly from the promoted physics components;
- trimming to the active domain when the curated parity surface excludes guard
  cells.

### Elliptic Solves

Potential and related closures are handled through the elliptic solver layer in
[solver/elliptic.py](../src/jax_drb/solver/elliptic.py), with lane-specific
setup coming from the surrounding physics module.

### Implicit / Stiff Transient Stepping

The heaviest recycling and neutral lanes use bounded implicit stepping rather
than pure explicit updates. The active release surface currently includes:

- sparse backward-Euler / BDF-style recycling transient ladders;
- matrix-free implicit neutral stepping on the promoted `neutral_mixed`
  windows;
- compact reduced controller lanes on staged CVODE-backed reference examples.

### Controller Reconstruction / Audit Algorithms

The controller campaign packages reconstruct proportional-integral source terms
from saved histories using the same signal conventions and trapezoid-style
integral bookkeeping expected by the promoted reference examples. These are
review/audit algorithms rather than hot-kernel solvers, but they are part of
the claimed validation surface.

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

The user-visible control-oriented closures currently exposed in the validation
surface are:

- upstream density feedback
- reduced temperature feedback
- reduced detachment controller

The bounded controller packages validate the saved control trajectories and
source identities, but the broader production temperature/detachment workflow is
still explicitly documented as beyond the current strong-subset claim.

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

## Differentiability Boundary

`jax_drb` intentionally separates:

- the fully user-facing CLI/runtime surface, which may use NumPy/SciPy
  boundary code where appropriate;
- the end-to-end differentiable research lane, which is expected to run through
  Python drivers on the strongest native JAX kernels.

Today the best differentiable lanes are still the compact native-exact kernels
such as diffusion, vorticity, drift-wave-style reduced paths, and the reduced
3D selected-field kernels used in the profiling/runtime campaigns. The heavier
recycling backbone remains the main differentiability and accelerator blocker.

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
