# DRBX Developer Documentation

This directory documents the architecture that is selectable in the current
codebase. Source and tests remain authoritative; these notes explain how the
pieces fit together and which combinations are supported.

## Current documents

| Document | Scope |
|---|---|
| [Geometry and metric pipeline](geometry_metric_evaluator_call_chain.md) | Magnetic inputs, wall geometry, square and toroidal metric construction, sampling, and caches |
| [Axis-regular angular RLP](axis_regular_angular_rlp.md) | Production toroidal owner topology, prolongation/restriction, fine-grid operator contract, and phi solve |
| [Embedded control volumes](embedded_control_volume_architecture.md) | Generic cut-wall agglomeration, reconstruction, face ownership, and sharding contracts |
| [FCI simulation architecture](fci_simulation_architecture.md) | HSX driver, operators, boundary closures, phi inversion, time integration, and supported configurations |

## Topology boundaries

The code contains three related but distinct geometry uses:

| Use | Logical topology | State representation | Runtime status |
|---|---|---|---|
| Square HSX comparison | `[0,1]^2 x S1` | One unknown per structured cell | Supported |
| Toroidal HSX simulation | Polar `D2 x S1` | Radius-dependent angular RLP owners | Production toroidal path |
| Embedded/cut wall | Structured chart cut by an implicit wall | Agglomerated control-volume owners | Reusable library and validation path; not selected by `simulate_hsx_blob.py` |

Angular RLP and embedded cut-wall agglomeration share owner-map and volume
containers, but they do not share a face algorithm. RLP applies an ordinary
fine polar operator between prolongation and restriction. Embedded walls use
explicit irregular-face geometry and reconstruction.

## Documentation policy

- Keep current contracts, invariants, source entry points, and validation
  boundaries here.
- Do not maintain chronological experiment logs, failed-run diaries, or
  prototype command transcripts in `dev_docs`.
- Record durable numerical evidence in tests or purpose-named analysis
  artifacts. Use version control for superseded designs.
- When a selectable path changes, update the relevant architecture document
  in the same change.
- A document must clearly distinguish production driver behavior from a
  reusable library or test-only scaffold.

## Focused verification

From the `DRBX` directory:

```bash
PYTHONPATH=src:.. XDG_CACHE_HOME=/tmp/drbx-cache python -m pytest -q \
  tests/test_MetricEvaluator_toroidal.py \
  tests/test_polar_angular_agglomeration_geometry.py \
  tests/test_fci_projected_fine_grid_control_volume.py \
  tests/test_fci_gmres_control_volume_owner_space.py \
  tests/test_simulate_hsx_blob_toroidal_geometry.py
```

Generic embedded-wall behavior has separate coverage in
`test_fci_cutwall_slab_operators.py`,
`test_fci_cutwall_shifted_torus_4field.py`, and
`test_fci_control_volume_field_closure.py`.
