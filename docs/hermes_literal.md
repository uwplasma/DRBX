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
