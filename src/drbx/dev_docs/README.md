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
| [FCI geometry artifacts](fci_geometry_artifacts.md) | Independent HSX producer, directory-bundle contract, qualification, and trusted consumer boundary |

## Research roadmaps

These describe approved future work and progress gates, not currently verified
solver capabilities. Detailed run evidence belongs in linked research artifacts.

| Roadmap | Scope |
|---|---|
| [Second-order perpendicular RLP operators](perpendicular_second_order_roadmap.md) | P00–P11 global operator and elliptic/MMS solution certification, with regional diagnostics; authoritative task progress ledger |
| [Second-order parallel RLP operators](parallel_second_order_roadmap.md) | Q00–Q09 HSX diffusion certification, shared interface structure, and coupled parallel operator/solution MMS; authoritative task progress ledger |
| [Parallel FCI/direct architecture comparison](parallel_fci_direct_design_comparison.md) | Reopened Q design choice; traced FCI strongly preferred, direct reconstruction retained as an alternative and benchmark |
| [FCI return basis and support candidate](q_fci_geometry_aware_return_design.md) | Research design contract for geometry-aware cubic-potential face returns, adequate traced-leg support, and the next bounded verification; not production-certified |
| [Q05a direct-cubic extraction assignment](q05a_direct_cubic_extraction_assignment.md) | Authorized bounded package extraction and local-action replay contract; not full Q04/Q05 promotion |
| [Q06 gradient/divergence assignment](q06_gradient_divergence_assignment.md) | Authorized bounded real-HSX `G`, conservative `D`, product-identity, and composition certification contract |
| [Current traced Q06 contract](q06_traced_gradient_divergence_contract.md) and [direct-gradient assignment](q06_direct_gradient_assignment.md) | Frozen traced midpoint/cap `G` semantics and initial bounded implementation scope; older integrated-face Q06 plans are historical for this path |
| [Q08 shared reconstruction extraction plan](q08_shared_extraction_plan.md) | Implemented paired preparation, compact bank/runtime and shared owner-plane utilities; bounded replay/resource evidence, with full-grid/GPU and remaining Q08 gates open |

Both roadmaps require real HSX geometry from the first numerical audit.
Idealized fixtures remain algebra/bookkeeping controls, not evidence of HSX
convergence. They share the global operator L2 order gate and inexpensive
midpoint MMS convention; parallel structure is promoted only after diffusion
certification.

The [clean remote global qualification commands](../../../scripts/hsx_remote_qualification/REMOTE_COMMANDS.md)
and [implementation validation](../../../scripts/hsx_remote_qualification/VALIDATION.md)
support the new versioned perpendicular campaign. The roadmaps retain the
scientific acceptance gates and separate P/Q ownership.

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
