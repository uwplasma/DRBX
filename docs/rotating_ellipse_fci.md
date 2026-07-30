# Rotating-ellipse FCI status

The former global rotating-ellipse parallel-operator example and gate have
been retired with the global RHS path. This page remains as a pointer for
maintainers so old links do not imply that the removed global lane is a
supported validation route.

## Supported validation paths

Use the ported sharded two-field gate for reduced FCI validation:

```bash
pytest -q tests/test_fci_sharded_2field.py
```

Use the sharded full-EB manufactured-solution gate for the seven-field
drift-reduced Braginskii path:

```bash
pytest -q tests/test_mms_shifted_torus_EB_sharded.py
```

The supported multi-device performance path is the sharded two-field
strong-scaling benchmark:

```bash
PYTHONPATH=src python examples/benchmarks/fci_sharded_strong_scaling.py
```

The shared shifted-torus MMS geometry used by analysis code lives in
`tests/shifted_torus_4field_mms_helpers.py`; post-processing should import that
helper directly rather than importing a test driver or a retired global RHS.
