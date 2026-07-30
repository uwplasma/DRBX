# Stellarator FCI validation

The former campaign page described deleted global FCI operators and campaign
drivers. The current validation surface is intentionally smaller:

- `tests/test_fci_sharded_2field.py` checks a local two-field step and its
  `jax.shard_map` execution;
- `tests/test_mms_shifted_torus_2_field.py` provides the shifted-torus
  two-field manufactured-solution driver;
- `tests/test_mms_shifted_torus_EB_sharded.py` checks the shard-local full-EB
  path;
- `examples/benchmarks/fci_sharded_strong_scaling.py` measures device scaling.

`FciGeometry3D` is retained for host-side field-line, metric, and boundary
staging. Numerical updates consume `LocalFciGeometry3D` payloads and local
operators inside `jax.shard_map`.
