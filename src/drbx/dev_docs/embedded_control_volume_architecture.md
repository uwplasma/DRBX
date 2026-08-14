# Embedded Control-Volume Architecture

The embedded control-volume stack represents a structured logical mesh cut by
an actual physical wall. It provides global agglomeration, local owner
geometry, moment reconstruction, canonical irregular faces, conservative
scatter, and cross-shard metadata.

This is distinct from toroidal angular RLP. The two paths share generic owner
and volume containers, but RLP has no irregular faces or face reconstruction.

## Integration status

The embedded-wall implementation is a reusable library with slab and shifted-
torus validation. It is not selected by `simulate_hsx_blob.py`. The seven-field
HSX RHS accepts a control-volume geometry only when it carries the production
angular-agglomeration profile, and then applies `R A_f P` rather than compact
face operators. Its parallel characteristic subsystem still uses five
material variables; that subsystem count is not the full state count.

## Host topology

[`geometry/fci_control_volumes.py`](../geometry/fci_control_volumes.py)
constructs decomposition-invariant topology before JAX tracing.

`GlobalControlVolumeTopology3D` records:

- fine-cell owner indices and aggregate IDs;
- active owners and merged sources;
- aggregate membership;
- regular-face masks;
- canonical irregular-face ownership;
- remote aggregate and residual-routing metadata.

`build_global_control_volume_topology` derives agglomeration from cut-cell
geometry. `build_global_control_volume_topology_from_owner_map` accepts an
already-defined owner map. `compile_local_control_volume_geometry` lowers the
global record for one shard without changing owner identity.

## Cell cases

Every local storage cell has orthogonal roles:

| Role | Meaning | Runtime value |
|---|---|---|
| Active owner | Stores one independent aggregate unknown | Evolved/solved |
| Merged source | Fine cell represented by another owner | Alias slot is inactive |
| Aggregate target | Owner receiving one or more sources | Uses aggregate volume and moments |
| Inactive/zero-fluid cell | Outside the represented fluid region | Masked |

A retained cut cell may be an active owner or may be merged. A geometrically
ordinary cell can be an aggregate target. Operator code must use masks and
owner maps rather than infer roles from volume fraction alone.

`LocalControlVolumeCellGeometry3D` carries raw and aggregate volumes,
centroids, second and third moments, membership counts, and owner routing.

## Faces

Dense regular faces remain in `LocalRegularFaceGeometry3D`. Faces affected by
cutting or agglomeration are represented by canonical rows in
`LocalControlVolumeFaceRows3D`.

An irregular row identifies:

- minus and optional plus aggregate owners;
- physical-boundary, interior, or partial-face kind;
- quadrature points and area covector weights;
- metric and magnetic coefficients at quadrature points;
- one global face ID and one evaluator;
- remote field and residual destinations when a face crosses a shard.

The same physical face must never be counted by both dense and irregular
paths. Interior flux is added once to one owner and subtracted once from the
other. Physical cut-wall flux is added only to the fluid-side owner.

## Reconstruction and field closure

`LocalMomentReconstruction3D` stores precomputed observation indices and
weights for recovering values and gradients from aggregate cell averages.
The basis uses aggregate moments, so its unknowns retain a cell-average
meaning rather than being treated as point samples.

`LocalMomentFittedFaceRows3D` stores direct integrated face functionals for
generic embedded faces. `build_local_control_volume_field_closure` evaluates
the selected face values, gradients, projected perpendicular flux, parallel
flux, and parallel-gradient flux for one field and boundary condition.

Boundary data are supplied separately through
`LocalControlVolumeBoundaryBC3D`. A scalar Dirichlet value does not by itself
determine a normal derivative; the reconstruction and operator-specific
closure determine the derivative or flux actually required by the operator.

## Conservative operator flow

```text
owner field
  -> gather complete local/halo observations
  -> evaluate reconstruction or direct face functional
  -> compute one integrated flux per canonical face
  -> scatter +F/-F to owner residuals
  -> divide by aggregate physical volume
```

Products such as pressure or particle flux require a closure for the derived
quantity. They must not generally be approximated as products of independently
averaged face values.

The relevant runtime code is in:

- [`native/fci_boundaries.py`](../native/fci_boundaries.py), which defines the
  packed geometry, reconstruction, and BC records;
- [`native/fci_operators.py`](../native/fci_operators.py), which builds field
  closures and conservative operators;
- [`native/fci_control_volume_operators.py`](../native/fci_control_volume_operators.py),
  which contains reusable control-volume operator helpers.

## Sharding contract

Global preprocessing owns topology identity. Local shards may evaluate a
canonical face only when designated by the global record. A remote interior
face carries both the remote owner sample location and the reverse residual
destination. Halo exchange supplies observations; reverse scatter preserves
the equal-and-opposite flux update.

RLP currently restricts toroidal production to one device, but that is not a
limitation of the generic embedded-wall topology records.

## Required invariants

- Every active fine cell maps to exactly one active owner.
- Owner mapping is idempotent.
- Aggregate volume equals the sum of member raw volumes.
- Active aggregate volumes are positive and finite.
- Every canonical irregular face has exactly one evaluator.
- Interior face updates are equal and opposite before volume division.
- Dense and irregular masks form an exclusive face partition.
- Active reconstruction rows reference valid observations and reproduce their
  declared polynomial basis to tolerance.
- Remote owner and remote residual metadata identify the same physical
  aggregate coupling.

## Validation

- [`tests/test_fci_control_volumes.py`](../../../tests/test_fci_control_volumes.py)
  validates global and local topology.
- [`tests/test_fci_control_volume_reconstruction_library.py`](../../../tests/test_fci_control_volume_reconstruction_library.py)
  validates moment bases and functionals.
- [`tests/test_fci_control_volume_field_closure.py`](../../../tests/test_fci_control_volume_field_closure.py)
  validates values, gradients, products, and boundary closure.
- [`tests/test_fci_cutwall_slab_operators.py`](../../../tests/test_fci_cutwall_slab_operators.py)
  validates isolated embedded-wall operators.
- [`tests/test_fci_cutwall_shifted_torus_4field.py`](../../../tests/test_fci_cutwall_shifted_torus_4field.py)
  validates the shifted-torus four-field scaffold and sharded face ownership.
