# Stellarator FCI validation status

The earlier tutorial described the global rotating-ellipse operator example
and global four-field turbulence tests. Those paths are retired. The supported
FCI validation workflow is now split between the ported sharded two-field
model and the sharded full-EB model.

## Ported sharded two-field path

The two-field manufactured-solution gate covers the shifted-torus geometry,
and `tests/test_fci_sharded_2field.py` checks that the `shard_map` step is
bit-exact with the single-device step. The reproducible performance path is:

```bash
pytest -q tests/test_mms_shifted_torus_2_field.py tests/test_fci_sharded_2field.py
PYTHONPATH=src python examples/benchmarks/fci_sharded_strong_scaling.py
```

## Sharded full-EB path

The seven-field electrostatic/electromagnetic drift-reduced Braginskii path
uses host-side `FciGeometry3D` only to stage data before lowering to
`LocalFciGeometry3D`. Its supported spatial and time-integration gate is:

```bash
pytest -q tests/test_mms_shifted_torus_EB_sharded.py
```

Analysis scripts that reconstruct shifted-torus geometry should import
`tests/shifted_torus_4field_mms_helpers.py` directly. The retired global
four-field analysis scripts and the global rotating-ellipse operator example
are not supported entry points.
