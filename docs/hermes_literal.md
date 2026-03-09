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

That engine entrypoint is still intentionally incomplete. The strict open-field
parity run should not switch to it until the communication, shifted-metric,
ExB, parallel FV, and vorticity component stack is landed.

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
