# Differentiable FCI status

The former standalone differentiable FCI example and its validation gate have
been retired during the sharded-only refactor. Differentiable numerical work
now composes local JAX operators inside a `jax.shard_map` step. The retained
`FciGeometry3D` object stages host geometry and metric data; it is lowered to
local shard payloads before evolution.

For supported examples, see the sharded two-field test and the shifted-torus
full-EB MMS test listed in [Examples](examples.md). For solver behavior and
whole-step compilation, see [Solvers and Design](solvers_and_design.md).
