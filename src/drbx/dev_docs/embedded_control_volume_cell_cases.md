# Embedded Control-Volume Cell Cases

This guide explains the cases a storage cell can occupy in the embedded
control-volume discretization, which numerical machinery applies to each
case, and why that machinery is needed.

The detailed execution order and data structures are documented in
[cutwall_agglomeration_ls_call_chain.md](cutwall_agglomeration_ls_call_chain.md).
This document is organized around the geometry cases instead.

## Two Independent Classifications

A cell classification answers:

```text
Who owns this finite-volume unknown?
Where is its value and reconstruction anchored?
```

A face classification answers:

```text
Which geometric interface carries the flux?
Does the dense stencil or a compact quadrature row evaluate it?
```

These questions must remain separate. A regular active cell may use a compact
face on its cut-wall side and dense faces on its other sides. Promoting the
entire cell to one face path would recursively promote neighboring regular
cells and destroy the structured fast path.

The authoritative cell data is `LocalControlVolumeCellGeometry3D`. The
authoritative combined object is `LocalEmbeddedControlVolumeGeometry3D`.

## Case Summary

| Cell case | Owns a value | Reconstruction | Face handling |
| --- | --- | --- | --- |
| Ordinary interior owner | Yes | Structured by default | Dense regular faces |
| Coordinate-boundary owner | Yes | Structured by default | Dense boundary closure |
| Retained cut-cell owner | Yes | Moment-aware cubic | Compact irregular faces where needed |
| Aggregate target | Yes | Moment-aware cubic about aggregate centroid | Compact aggregate interfaces plus eligible dense faces |
| Merged source | No | None; output is zero | Its exterior faces route to the target owner |
| Zero-fluid or inactive storage | No | None; output is zero | No fluid flux |
| Dense-to-compact transition owner | Yes | Polynomial available for affected faces | Chosen independently per face |
| Shard-interface owner | Yes | Local polynomial plus exchanged coefficients | Dense halo path or mirrored compact row |

The cases can overlap. For example, an aggregate target can also touch a cut
wall and a shard interface.

## 1. Ordinary Interior Owner

An ordinary interior owner is a full-fluid cell that:

- maps to itself;
- is an active owner;
- receives no merged sources;
- does not touch a compact irregular interface.

Its stored value is the physical finite-volume average over the ordinary
cell. The dense structured operators already provide the desired accuracy and
are substantially cheaper than compact reconstruction.

### Machinery Used

- ordinary cell center, volume, and metric data;
- prepared field halos;
- centered dense face stencils;
- dense conservative divergence.

### Why Nothing New Is Needed

The regular grid supplies symmetric neighbors and complete faces. There is no
missing solid-side sample, displaced control-volume anchor, or partial face
area to represent.

## 2. Coordinate-Boundary Owner

This is an otherwise regular active cell adjacent to a full
coordinate-aligned physical boundary. It remains on the dense face path.

### Dirichlet Face

The field value is known on the boundary. In the finite-volume path, the BC
value is interpreted as the `J`-weighted average over that face patch, not
merely the value at its center.

`LocalRegularBoundaryMomentClosure3D` stores geometry-specific face and
first-owner-centroid normal-derivative functionals using:

```text
boundary face average + three inward cell averages
```

The weights reproduce the derivative of the cubic moment basis at both
locations. They are precomputed outside JIT and applied as four-value dot
products.

This machinery was added because the old one-sided formula combined a
pointwise wall value with finite-volume cell averages. That mismatch creates a
boundary-local truncation error even when the interior stencil is correct.

### Neumann Or Prescribed Flux Face

The supplied normal derivative or flux is applied directly. It does not pass
through the Dirichlet derivative functional.

### Halo Machinery

Physical face ghosts are followed by topology closure and then physical
corner closure. This ordering is needed because tangential derivatives on a
physical face can sample periodic or sharded tangential halo indices.

### What Is Not Used

- no compact cut-wall row for a complete coordinate face;
- no aggregate reconstruction merely because the cell is at the boundary;
- no embedded-wall BC object for the regular face.

## 3. Retained Cut-Cell Owner

A cut cell has positive fluid volume but is intersected by the embedded solid.
If no acceptable local merge target exists, or the merge policy elects to
keep it, it remains an independent active owner.

Its value is an average over the actual fluid fragment, so its anchor is the
fluid centroid rather than the storage-cell center. Its second moment also
differs from that of a full rectangular cell.

### Machinery Used

- raw fluid volume, centroid, and second moment;
- `LocalMomentReconstruction3D`;
- field-specific `LocalControlVolumeBoundaryBC3D`;
- compact partial, interior, and cut-wall face rows;
- face quadrature for conservative fluxes.

### Why It Is Needed

A coordinate stencil can cross into solid storage or assume a full face that
does not exist. Ghost-cell construction becomes ambiguous near oblique walls,
edges, and corners. Moment-aware reconstruction instead uses neighboring
control-volume averages and the actual Dirichlet wall equations.

One-wall and multi-wall cells use the same reconstruction system. There is no
separate hard projection of the gradient onto wall equations.

## 4. Merged Source Cell

A small or awkward cut cell may be merged into a neighboring active owner.
The source storage cell then ceases to own an independent unknown.

`owner_i/j/k` maps it directly to its target. It must not participate as an
independent sample, receive a gradient, or receive final operator output.

### Machinery Used

- direct, idempotent source-to-target mapping;
- source raw moments accumulated into the target;
- owner-value expansion when storage-shaped arrays are needed;
- conservative routing of exterior source faces to the target.

### Why It Is Needed

Dividing flux by a very small cut-cell volume produces severe explicit
timestep restrictions and amplifies geometric errors. Merging removes that
small independent control volume while preserving its fluid volume and
moments.

### Required Invariants

- a source is never an active owner;
- mappings do not form chains;
- the target is local and active;
- source storage values cannot affect reconstructed flux;
- source gradients and operator outputs are zero.

## 5. Aggregate Target

An aggregate target is an active owner with one or more merged sources:

```text
is_aggregate_target == (received_source_count > 0)
```

Its stored value is the finite-volume average over the union of all members.
Its reconstruction anchor is the aggregate `J`-weighted centroid, and its
second moment is the moment of that same union.

### Machinery Used

- aggregate volume, centroid, second moment, and third moment;
- moment-aware cubic reconstruction;
- unique owner samples rather than storage-cell samples;
- aggregate-volume division after integrated flux accumulation.

### Why It Is Needed

Treating the aggregate value as if it lived at the target storage-cell center
makes wall equations and neighboring reconstruction equations geometrically
inconsistent. A gradient can then satisfy a wall equation exactly while being
less accurate physically. The aggregate anchor keeps the tuple
`(value, position, moments)` consistent.

## 6. Zero-Fluid Or Inactive Storage Cell

A cell with no represented fluid volume has no degree of freedom.

### Machinery Used

- inactive masks;
- zero gradients, Hessians, and operator outputs;
- exclusion from reconstruction samples and face rows.

### Why It Is Needed

Solid storage is an implementation detail, not physical data. Reading a zero
or stale value from it as a neighboring sample introduces an artificial jump
at the wall.

Boundary information comes from BC objects at the actual wall geometry, never
from solid-cell storage.

## 7. Dense-To-Compact Transition Cell

A regular owner can neighbor a retained cut cell, aggregate member, partial
face, or cut-wall region. This does not force all six of its faces onto the
compact path.

### Face-Level Policy

- a full ordinary face with no compact row uses the dense path;
- an aggregate-internal face is omitted;
- a partial or embedded boundary face uses one compact row;
- a full interface represented by a compact row is excluded from dense
  divergence;
- a full face whose structured support touches nonregular storage is owned by
  its compact row, and both local face owners receive cubic reconstruction
  support;
- every other full face remains entirely on the structured dense path.

### Why Exclusive Face Ownership Is Needed

Earlier wiring allowed a dense face to be closed while its replacement compact
flux reached only one owner, or allowed both paths to count the interface.
Either case breaks conservation. Every physical interface must contribute
exactly once, with equal and opposite signs for an interior face.

## 8. Embedded Cut-Wall Face Owner

An active owner may touch one or several embedded wall patches. This is a face
condition layered on top of the retained-cut or aggregate-target cell case.

### Machinery Used

- compact `CV_FACE_CUT_WALL` rows;
- wall centroid and Gauss-point geometry;
- field-specific BC kind and values;
- wall equations in the cubic reconstruction;
- quadrature of boundary flux.

### Why BC Objects Are Required

Dirichlet means the field value is known on the physical wall. The value must
be collocated with the wall centroid or quadrature point where it is used.
Neumann and normal-flux conditions instead prescribe derivative or flux data.

A single scalar wall value cannot safely be substituted into an inactive
neighbor slot because the wall is generally not located at that cell center
and may not be normal to a coordinate direction.

## 9. Partial Interior Face

Two active control volumes can share only part of a storage-grid face because
the solid blocks the remainder.

### Machinery Used

- nonoverlapping rectangular decomposition of the open region;
- `2x2` Gauss points per rectangle;
- one compact `CV_FACE_PARTIAL` row;
- one numerical flux evaluated at shared physical points;
- equal-and-opposite owner scatter.

### Why It Is Needed

Multiplying a full-face stencil by only an area fraction does not capture
off-center geometry or metric variation and can sample the wrong control
volume. Explicit patch quadrature represents both the open measure and its
location.

## 10. Shard-Interface Owner

The cell remains locally owned, but one face or reconstruction sample can lie
on another shard.

### Dense Interface

Prepared halos supply the neighboring values through topology exchange.

### Compact Interface

Each shard stores a mirrored, locally outward row. Reconstructed owner value,
gradient, Hessian, and validity are exchanged. Both sides evaluate the same
physical interface flux and apply opposite local divergence signs.

### Why It Is Needed

`jnp.roll` wraps within a local shard and is not a global periodic neighbor
operation. Raw owned arrays therefore cannot provide cross-shard
reconstruction samples or face values.

Aggregate ownership itself remains shard local. Cross-shard aggregates would
require a separate reduction and ownership protocol and are not currently
supported.

## Operator Decision Sequence

For each field, the runtime logic can be read as:

1. Expand owner values into storage shape where dense infrastructure needs it.
2. Complete physical, topology, and corner halos.
3. Build cubic polynomials for active reconstruction owners.
4. Exchange polynomial coefficients needed by remote compact interfaces.
5. Evaluate untouched ordinary faces with dense structured stencils.
6. Apply the moment-derived closure on full regular Dirichlet boundaries.
7. Evaluate compact transition, partial, and cut-wall rows by quadrature.
8. Verify that no compact row has an open dense-face counterpart.
9. Scatter integrated fluxes to active owners exactly once.
10. Divide by the matching aggregate physical volume.
11. Set merged-source and inactive-storage outputs to zero.

## Diagnostic Questions

When a near-wall error appears, inspect these questions in order:

1. Is the reported index an active owner, aggregate target, merged source, or
   inactive cell?
2. Is the stored value interpreted at the matching centroid and moments?
3. Does each attached face use exactly one dense or compact representation?
4. Does an interior compact flux reach both owners with opposite signs?
5. Does any reconstruction or flux read merged-source or solid storage?
6. Are Dirichlet values collocated with the face centroid or quadrature point?
7. For a coordinate boundary, is the BC a face average rather than a point
   value?
8. For a shard interface, did the value or polynomial arrive through exchange
   rather than local wrapping?
9. Is the integrated sum divided by the same volume used by MMS projection?

This ordering separates geometry ownership errors from reconstruction errors,
face-routing errors, and boundary-data errors.
