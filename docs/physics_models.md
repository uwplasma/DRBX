# Physics Models

This page is the technical map from the governing equations to the source tree.
It is meant to help both new users and developers find where a model lives
before they change a case, add a term, or debug a result. It also states the
main first-principles model forms and the numerical patterns actually used in
the current code, so the docs remain useful even without the manuscript.

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

In the promoted reduced-fluid lanes, the resolved particle flux is represented
schematically as

```text
Γ_s = n_s u_E + b n_s V_{∥,s} - D_{⊥,s} ∇_⊥ n_s + Γ_s^model
```

where `u_E = b × ∇⊥ φ / B` and `Γ_s^model` collects benchmark-specific
transport closures such as curvature-driven or reduced-annulus terms.

### Continuity

For an evolved species density `n_s`:

```text
∂t n_s + ∇·Γ_s = S_{n,s}
```

where `Γ_s` is the resolved advective/diffusive flux and `S_{n,s}` collects
ionisation, recombination, recycling, pumping, controller action, and any
case-specific source
terms.

### Parallel Momentum

For the evolved parallel momentum density `n_s V_{∥,s}`:

```text
∂t (n_s V_{∥,s}) + ∇·(Γ_s V_{∥,s})
  = -∇_∥ p_s + F_{coll,s} + F_{thermal,s} + F_{sheath,s}
    + ∇_∥·Π_{∥,s} + S_{m,s}
```

The exact active terms depend on the promoted lane:

- open-field recycling adds sheath, recycling, Braginskii friction, heat
  exchange, thermal force, and ion-viscosity closures;
- drift-wave/blob ladders carry the benchmark-consistent reduced momentum
  structure;
- direct tokamak ladders reuse the same promoted closures on the staged
  tokamak metric payload.

### Pressure / Energy

For the evolved scalar pressure `p_s`:

```text
∂t p_s + ∇·(p_s u_s) + γ p_s ∇·u_s
  = Q_{cond,s} + Q_{coll,s} + Q_{src,s}
```

with the right-hand side carrying the promoted conduction, collisional exchange,
radiation/source, and controller/recycling terms relevant to the active lane.

In the open-field and tokamak recycling lanes this includes explicit parallel
heat conduction, sheath energy losses, thermal-force coupling, reaction energy
exchange, and neutral/plasma exchange terms.

The dominant parallel conductive closure is the standard reduced form

```text
q_{∥,s} ≈ -κ_{∥,s} ∇_{∥} T_s
```

### Potential / Vorticity Closure

The electrostatic ladders solve benchmark-specific elliptic closures between
`phi`, `Vort`, and the underlying density/current state. On the promoted
benchmark surfaces this includes:

- Boussinesq closures on the vorticity ladder;
- drift-wave/quasineutral electron closures on the drift-wave ladder;
- benchmark-faithful `phi` reconstruction on the blob/interchange lanes.

At the operator level this is the familiar reduced electrostatic structure:

```text
ω = ∇⊥·(C ∇⊥ φ)
```

with lane-dependent coefficients `C`, metric terms, and source closures.

On the promoted electrostatic benchmark lanes, the corresponding transport
equation is represented schematically as

```text
∂t ω + ∇·(ω u_E) = ∇∥ J∥ + S_ω
```

with `S_ω` collecting curvature, sheath, and benchmark-specific source terms.

### Electromagnetic Reduced Surfaces

The promoted electromagnetic benchmark lanes use compact selected-field
surfaces around:

```text
Ajpar = Σ_s Z_s n_s V_{∥,s}
```

plus the staged `Apar`/`NVe`/`Vort` benchmark closures documented in the
electromagnetic source and validation utilities.

Where electron-parallel dynamics is retained explicitly, the reduced
parallel-force balance is represented in compact form as

```text
0 = -e n_e E_∥ - ∇∥ p_e - η_∥ J_∥ + S_{∥,e}
```

## Reduced-Fluid Operator Structure

Across the drift-reduced lanes, the discrete operators are built from the same
small set of physical ingredients:

- parallel derivatives `Grad_par(f)` and flux divergences `Div_par(F)`;
- perpendicular transport/divergence operators on the staged metric payload;
- electrostatic `E×B` transport, typically represented in reduced form through
  an advection bracket or equivalent face-flux reconstruction;
- sheath target closures and recycling source terms at open-field boundaries;
- collisional, viscous, thermal-force, and atomic-rate source operators.

The exact promoted equation set differs by benchmark, but the implementation
reuses these operator families rather than encoding each case as an unrelated
solver.

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

In the implementation, this is where the bulk of the transport kernels live:
- `native/fluid_1d.py`
- `native/drift_wave.py`
- `native/blob2d.py`
- `native/recycling_1d.py`
- `native/neutral_mixed.py`

Recent performance work removed several per-cell Python loops from this layer
and replaced them with array kernels, especially on the heavy neutral/recycling
operators.

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

The strongest production path today is the sparse Newton backbone in
`solver/implicit.py` plus `native/recycling_1d.py`:

- nonlinear residuals are assembled from the staged multispecies open-field or
  direct-tokamak state;
- sparse finite-difference quotient Jacobians are built on the packed active
  state;
- GMRES is used first, with direct sparse fallback where needed;
- backward-Euler and BDF2-style history stepping are used on the promoted
  recycling windows.

That path is still the main host/SciPy-heavy backbone and the main remaining
performance bottleneck. Recent optimization passes made it materially cheaper by
reusing packed-state metadata, vectorizing hot residual operators, and reducing
allocation overhead in sparse Jacobian assembly.

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

At the leading-order reduced level, the target closures are expressed through

```text
V_{∥,i}|target ~ c_s
q_{∥,e}|target ~ γ_e n_e T_e c_s
```

with `c_s` the local sound speed and `γ_e` the electron sheath heat
transmission factor.

## Implicit Transient Form

The strongest production-path recycling and direct-tokamak ladders use a
backward-Euler/BDF-style implicit residual of the form

```text
F(U^{n+1}) = U^{n+1} - Σ_k α_k U^{n-k} - Δt β R(U^{n+1}) = 0
```

with Newton updates

```text
J(U^{n+1,ℓ}) δU^ℓ = -F(U^{n+1,ℓ})
U^{n+1,ℓ+1} = U^{n+1,ℓ} + δU^ℓ
```

The current implementation builds sparse finite-difference quotient Jacobians
on the packed active state, solves the linearized system with GMRES first, and
falls back to direct sparse solves where required.

## Differentiable Analysis Surface

The compact differentiable lanes use the standard JAX gradient map

```text
g(θ) = ∇_θ J(θ)
```

and local Gaussian uncertainty propagation through the linearized pushforward

```text
Σ_Q ≈ G Σ_θ G^T ,  G = ∂Q/∂θ
```

These are the surfaces used by the published sensitivity, uncertainty, and
inverse-design examples. The heaviest implicit recycling backbone is still the
main boundary between the clean JAX-native lane and the host/SciPy-heavy lane.

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

## JAX Implementation Boundary

`jax_drb` is not one monolithic “all-JAX” runtime. The code deliberately
separates:

- fully JAX-native compact kernels used for differentiable reduced lanes,
  profiling, and selected-field 3D reductions;
- mixed host/JAX/SciPy production paths used where the strongest current parity
  surface still depends on sparse implicit workflows.

In practice, the current promoted JAX-native building blocks are:

- `jax.numpy` array kernels
- `@jax.jit`
- `jax.vmap`
- `jax.grad` / `jax.value_and_grad`
- `jax.lax.linalg.tridiagonal_solve`

The codebase does not currently rely on `diffrax`, `equinox`, or `lineax` to
power the promoted release results. Those libraries are useful ecosystem
context and future options, but the release-critical kernels are driven by the
core JAX primitives above.

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

## Background Context

The public code docs intentionally stay implementation-first. The model family
context is:

- reduced-fluid edge/SOL transport with explicit parallel losses, sheath
  closure, recycling, and neutral/atomic source terms;
- compact electrostatic and reduced-electromagnetic benchmark surfaces on the
  same operator stack;
- an end-to-end differentiable subset built from the strongest JAX-native
  kernels.

The code docs stop there deliberately. Broader literature comparisons and
paper-style code-family positioning belong in the separate manuscript repo, not
in the shipping package documentation.

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

The summary version of that contract is in:

- [research_grade_validation_matrix.md](research_grade_validation_matrix.md)
