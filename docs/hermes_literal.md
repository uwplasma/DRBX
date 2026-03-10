# Hermes Literal Path

## Purpose

`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal` is the active strict
parity implementation path.

Unlike the frozen hybrid mirror in
`/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_hermes`, this package is being
rebuilt from the Hermes/BOUT component execution model outward:

1. guard-aware field storage
2. boundary operators
3. shifted-metric communication
4. shared component state like `fastest_wave`
5. component `transform_impl()`
6. component `finally()`
7. stage-1 engine ordering

The goal is not to adapt the unified `jax_drb` term registry. The goal is to
translate the Hermes source graph directly into pure JAX so the same inputs
produce the same outputs, while remaining differentiable and JIT-friendly.

## Frozen Path

The previous hybrid parity work now lives in:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/legacy_hermes`

That path is kept only for traceability and regression comparison. It is not
the active Milestone A implementation target.

## Initial Modules

The first fresh modules are:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/field.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/boundary_standard.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/sound_speed.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/evolve_density.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/evolve_pressure.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/shifted_metric.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/fv.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/div_ops.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/exb.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/delp2.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/vorticity.py`

Implemented source translations in this first slice:

- `SoundSpeed::transform_impl`
  - source: `/Users/rogerio/local/hermes-3/src/sound_speed.cxx`
- `EvolveDensity::transform_impl`
  - source: `/Users/rogerio/local/hermes-3/src/evolve_density.cxx`
- `EvolvePressure::transform_impl`
  - source: `/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx`
- `BoundaryNeumann`-style guard filling and `Field3D::setBoundaryTo`
  midpoint preservation
  - sources:
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/boundary_standard.cxx`
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/field/field3d.cxx`
- `ShiftedMetricInterp` / `ShiftedMetric`
  - sources:
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/parallel/shiftedmetricinterp.cxx`
- `FV::Div_par_mod`
  - source:
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/include/bout/fv_ops.hxx`
- `Div_par(jpar)`
  - source:
    - `/Users/rogerio/local/hermes-3/src/div_ops.cxx`
- `Div_n_bxGrad_f_B_XPPM`
  - source:
    - `/Users/rogerio/local/hermes-3/src/div_ops.cxx`
- `Coordinates::Delp2(Field3D)`
  - source:
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`
- `Vorticity::finally` ExB branch
  - source:
    - `/Users/rogerio/local/hermes-3/src/vorticity.cxx`

## Differentiability and Performance Rules

- Runtime operators stay as pure array transforms.
- No Python mutation is allowed in JIT-executed numerical paths.
- Guard-aware helpers operate on explicit arrays and layouts, not opaque state.
- Reference and fused implementations should be separated when operator
  complexity grows.

## Current Status

The config/driver layer now recognizes `engine = "hermes_literal"` and routes
that choice to
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/engine.py`.

That engine is now executable for the strict Stage 1 baseline. It owns the
runtime scheduler interface used by the normal driver path:

- `rhs_split`
- `rhs`
- `rhs_explicit`
- `rhs_stiff`
- `rhs_with_phi`
- `rhs_with_phi_iters`
- `rhs_explicit_with_phi`
- `rhs_explicit_with_phi_iters`

The strict early parity config at
`/Users/rogerio/local/jax_drb/examples/open_field_line/input_tokamak_bxcv_alignment_strict_early.toml`
now sets `engine = "hermes_literal"`, and the smallest strict audit runs
through that engine directly:

- `/Users/rogerio/local/jax_drb/runs/audit_literal_engine_smoke`

Current 1-step array-ranked leaders on the literal engine are:

- `Te parallel`: `0.1474904091090806`
- `n parallel`: `0.13383127252151306`
- `omega parallel`: `0.11697795624618619`
- `Pe parallel`: `0.1133024567583403`
- `n advection`: `0.09623829491706752`
- `Pe advection`: `0.0676385260919583`

So the remaining Milestone A work is no longer “make the literal engine
exist.” It is “close the remaining operator/state contract gaps while the
strict audit is already running on that engine.”

The next runtime rehome is also landed now:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/parallel.py`

The reduced density/pressure cache in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/rhs.py` now uses that
module directly for:

- sheath-state reconstruction
- shifted boundary-plane handling
- `Div_par(jpar)` / `FV::Div_par_mod` selection
- fastest-wave and pressure transport coefficients

That removes another major hybrid dependency from the literal engine without
changing the live 1-step parity ordering. The rehome audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_parallel_runtime_rehome`
stays numerically aligned with the earlier literal-engine baseline while moving
`Te parallel`, `n parallel`, and `Pe parallel` slightly in the right direction.

The next runtime rehome is also landed now:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/advection.py`

The reduced density/pressure cache in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/rhs.py` now gets its
live ExB advection group from the literal module rather than importing
`core/terms/advection.py`. The literal module preserves the existing runtime
switches for:

- `exb_advection_simplified`
- `exb_advect_conservative`

and is covered by:

- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_advection_runtime.py`
- `/Users/rogerio/local/jax_drb/tests/test_vorticity_alignment_switches.py`

The 1-step strict audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_advection_parallel_rehome_1step_after_fix`
is numerically identical to the prior literal-engine baseline for the leading
rows:

- `n advection`: `0.09623829491706752`
- `Pe advection`: `0.0676385260919583`
- `n parallel`: `0.1338298917677307`
- `Te parallel`: `0.14748382093236653`

This matters because the leading strict channels now come from literal runtime
modules for both advection and parallel transport. The remaining gap is no
longer explained by those term-group imports falling back into the unified
core.

The literal engine now also owns its context builder in:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/context.py`

and `engine.py` now calls that builder directly instead of importing
`core.terms.build_context`. The new regression
`/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_context.py`
checks that the literal context reproduces the previous strict runtime
contract for:

- physical density and temperature fields
- prepared density and pressure fields
- solved `phi`
- the hot/EM/neutrals feature flags

This does not change the strict audit ranking, but it removes another shared
runtime dependency from the literal engine itself.

The literal engine now also imports its schedule and dispatch table through:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/registry.py`

instead of importing `core.terms.registry` directly from `engine.py`. The new
regression `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_registry.py`
locks the current Stage 1 contract by checking that the literal registry still
matches the active schedule and term names used by the strict engine.

The latest literal-runtime transport change is the staged local-subdomain ExB
path in:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/exb.py`

with the strict config now setting:

- `hermes_mirror_parallel_subdomain_size = 8`

This makes the runtime ExB wrapper evaluate the literal local operator over
consecutive Hermes-sized parallel chunks (`MYSUB`-style) instead of applying a
single global transform across the whole strict field. The new stitched-global
regression in
`/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_exb_runtime.py`
shows the expected improvement against the Hermes dump-backed global fixture.

The strict 1-step audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_subdomain_parallel_1step`
improves the live ExB rows to:

- `n advection`: `0.09623829491706752 -> 0.06021497597645309`
- `Te advection`: `0.03993444992422992 -> 0.03175328243530484`
- `Pe advection`: `0.0676385260919583 -> 0.0417892594173691`

while the leading parallel rows remain unchanged:

- `n parallel`: `0.1338298917677307`
- `omega parallel`: `0.11697795624618619`
- `Te parallel`: `0.14748382093236653`
- `Pe parallel`: `0.11330241103262646`

This is a real literal-runtime gain, but it is still not the `1e-2` strict
parity target. The remaining blocker is no longer “global vs local ExB” in
general; it is the residual processor-local communication/transform contract in
the ExB and parallel transport paths.

The next structural slice is now also landed in:

- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/communicate.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/bcs.py`
- `/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/ops.py`

`communicate.py` makes the local parallel-slab assembly explicit instead of
embedding that logic directly in the ExB runtime wrapper. The promoted runtime
currently uses the same internal edge-copy seam contract as the previously
validated subdomain path, while also exposing a neighbor-plane extraction mode
for the later, stricter processor-communication rewrite. The new unit coverage
is:

- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_communicate.py`

and the existing stitched-global regression in
`/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_exb_runtime.py`
still reports the same improvement:

- `Ne` raw relative error: `0.3242119312307518 -> 0.06090172693816785`
- `Pe` raw relative error: `0.3455716236178938 -> 0.06601079736963186`

The strict 1-step audit at
`/Users/rogerio/local/jax_drb/runs/audit_literal_comm_layer_1step` is
numerically identical to
`/Users/rogerio/local/jax_drb/runs/audit_literal_subdomain_parallel_1step` on
the leading rows. That is intentional for this slice: it removes more shared
runtime ownership from `core.terms` without regressing the promoted literal
baseline.

The shifted-metric layer is now landed and validated in
`/Users/rogerio/local/jax_drb/tests/hermes_literal/test_shifted_metric.py`
against both synthetic geometry-adapter checks and the dump-backed fixture
`/Users/rogerio/local/jax_drb/tests/fixtures/hermes_mirror_shiftedmetric_global_t1.npz`.

The literal parallel FV/divergence layer is now landed in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/fv.py` and
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/div_ops.py`, with
synthetic checks in
`/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_parallel.py` and
dump-backed term regressions in
`/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_parallel_dump.py`.

The literal ExB, `Delp2`, and vorticity layers are now also landed in
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/exb.py`,
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/delp2.py`, and
`/Users/rogerio/local/jax_drb/src/jaxdrb/hermes_literal/vorticity.py`, with
targeted regressions in:

- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_exb.py`
- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_exb_runtime.py`
- `/Users/rogerio/local/jax_drb/tests/hermes_literal/test_literal_vorticity.py`

The active strict runtime is now also consuming the fresh literal modules for:

- reduced species preparation
- shifted transforms
- ExB transport
- `Delp2` / vorticity ExB
- parallel FV transport
- mirror RHS cache assembly

The latest strict 1-step audit from
`/Users/rogerio/local/jax_drb/runs/audit_literal_runtime_promotion_1step`
preserves the previous fail-fast ordering rather than regressing it:

- `n parallel/par`: `weighted_array_rel = 0.13383127252151306`
- `omega parallel/jpar`: `weighted_array_rel = 0.11697795624618619`
- `Pe parallel/par_total`: `weighted_array_rel = 0.1133024567583403`
