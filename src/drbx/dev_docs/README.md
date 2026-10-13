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
| [FCI simulation architecture](fci_simulation_architecture.md) | HSX driver, operators, boundary closures, phi inversion, time integration, supported configurations, and the nodal SBP perpendicular stack (module map, defaults, verified scope, cost) |
| [FCI geometry artifacts](fci_geometry_artifacts.md) | Independent HSX producer, directory-bundle contract, qualification, and trusted consumer boundary |

## Research roadmaps

These describe approved future work and progress gates, not currently verified
solver capabilities. Detailed run evidence belongs in linked research artifacts.

| Roadmap | Scope |
|---|---|
| [Second-order perpendicular RLP operators](perpendicular_second_order_roadmap.md) | P00–P11 global operator and elliptic/MMS solution certification, with regional diagnostics; authoritative task progress ledger |
| [Nodal FCI parallel operators](parallel_fci_roadmap.md) | QF00–QF07: plane-to-plane FCI on P's nodal layout, from maps and the static operators to the evolved six-field MMS and P coupling; authoritative Q progress ledger since 12 October 2026 |
| [Nodal FCI design contract](q_fci_nodal_design_contract.md) | The numerical method: maps, endpoint interpolation, collocated `G`, trace-hit LVF boundaries, support-operator divergence/diffusion in P's norm, references and open decisions |

### Archived Q records (superseded 12 October 2026)

These describe the owner-based, short-span traced Q operators and are kept for provenance only:
[Q00–Q09 roadmap](parallel_second_order_roadmap.md),
[FCI/direct comparison](parallel_fci_direct_design_comparison.md),
[FCI return basis](q_fci_geometry_aware_return_design.md),
[Q05a extraction](q05a_direct_cubic_extraction_assignment.md),
[Q06 assignment](q06_gradient_divergence_assignment.md),
[Q06 traced contract](q06_traced_gradient_divergence_contract.md),
[Q06 direct gradient](q06_direct_gradient_assignment.md),
[Q08 extraction plan](q08_shared_extraction_plan.md),
[Q09 harness](q09_evolved_mms_contract.md), and the remaining `q0*_` / `q_fci_*` / `q_traced_*` records.

Both roadmaps require real HSX geometry from the first numerical audit.
Idealized fixtures remain algebra/bookkeeping controls, not evidence of HSX
convergence. They share the global operator L2 order gate, P's nodal layout and
norm, and pointwise nodal MMS references.

The clean remote global qualification commands and implementation validation
(`scripts/hsx_remote_qualification/`, removed; see commit `c0269a95`)
supported the versioned perpendicular campaign. The roadmaps retain the
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
