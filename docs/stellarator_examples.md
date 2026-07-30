# Stellarator examples

The supported 3-D FCI examples use the local operator API and lower their
work to `jax.shard_map` for multi-device execution. `FciGeometry3D` remains
the host-side geometry and metric staging object.

Current entry points include:

- `tests/test_fci_sharded_2field.py`, a two-field local/sharded step;
- `tests/test_mms_shifted_torus_2_field.py`, a shifted-torus two-field MMS;
- `tests/test_mms_shifted_torus_EB_sharded.py`, the full-EB shifted-torus MMS;
- `examples/benchmarks/fci_sharded_strong_scaling.py`, a sharded scaling
  benchmark.

The former standalone stellarator FCI examples and global RHS APIs are no
longer supported reproduction paths. Historical figures in `docs/media/` are
retained where useful, but their deleted scripts should not be invoked.
