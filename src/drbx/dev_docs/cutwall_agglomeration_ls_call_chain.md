# Embedded Control-Volume Call Chain

This document describes the unified aggregate-cell and cut-wall path used by
the native FCI operators and the shifted-torus four-field MMS harness.

For a case-by-case explanation of ordinary, boundary, cut, merged, aggregate,
transition, and shard-interface cells, see
[embedded_control_volume_cell_cases.md](embedded_control_volume_cell_cases.md).

## Core Contract

The numerical unknown on an active embedded control volume is a physical
finite-volume average:

```text
U_i = (1 / V_i) integral_CVi J u dxi
V_i = integral_CVi J dxi
```

Every reconstruction, boundary equation, face flux, norm, and MMS projection
must use the same control volume, volume, centroid, and moments. A merged
source cell is storage only. It contributes moments and volume to its owner but
does not own a degree of freedom or receive operator output.

## Complete Halo Closure

Operators consume halos prepared by `LocalHaloClosure3D`. The shared contract
has three ordered stages:

1. **Face closure** applies `LocalBoundaryFaceBC3D` through
   `PhysicalGhostCellFiller3D` over owned tangential face slabs.
2. **Topology closure** performs shard exchange followed by local or mapped
   topology rules. Because it runs after face closure, newly materialized
   physical ghosts propagate into physical-periodic and physical-shard edges.
3. **Corner closure** fills remaining physical-physical edges in codimension
   two, then three-physical-side corners in codimension three.

`PhysicalGhostCornerFiller3D` uses polynomial extrapolation from already
completed lower-codimension strips. Several directional candidates are
averaged. It never overwrites owned cells, ordinary face ghosts, or
physical-topology corners.

Boundary builders that depend on neighboring state first perform a pre-BC
topology exchange. After the dynamic BC payload is finalized, the complete
three-stage closure runs. Paths with an already known BC skip that prepass and
perform one closure pass.

This ordering is required by face-gradient operators. For example, a
tangential derivative on a physical radial face samples radial ghosts at
periodic or sharded tangential halo indices. Running physical filling after
the final topology exchange leaves those samples stale.

The closure is sharding compatible: collectives retain their established
mesh-axis ordering and physical-physical corner completion is local. It does
not create cross-shard aggregate ownership.

The authoritative local object is
`LocalEmbeddedControlVolumeGeometry3D`:

```text
LocalEmbeddedControlVolumeGeometry3D
  cells              LocalControlVolumeCellGeometry3D
  regular_faces      LocalRegularFaceGeometry3D
  irregular_faces    LocalControlVolumeFaceRows3D
  regular_transition_faces
                     LocalRegularTransitionFaceRows3D
  reconstruction     LocalMomentReconstruction3D (19-coefficient cubic on irregular owners)
  regular_boundary_closure
                     LocalRegularBoundaryMomentClosure3D
  centroid_*         operator coefficients at aggregate centroids
```

The former agglomeration map, aggregate geometry, gradient LS rows, and sparse
two-sided regular-face rows are not part of this call chain. Embedded operators
receive the unified geometry directly.

## Cell Geometry

`LocalControlVolumeCellGeometry3D` owns all cell-level embedded geometry.

- `owner_i/j/k` maps every storage cell directly to itself or to one local
  active owner.
- `is_merged_source` marks storage cells whose fluid region belongs to another
  owner.
- `is_active_owner` marks cells with an independent finite-volume unknown.
- `is_aggregate_target` is exactly `received_source_count > 0`.
- `raw_volume`, `raw_centroid`, `raw_second_moment`, and `raw_third_moment` describe the fluid
  portion of one storage cell.
- `aggregate_volume`, `centroid`, `second_moment`, and `third_moment` describe the union of all
  members mapped to an active owner.

Second moments are central logical-coordinate moments:

```text
M2_i = (1 / V_i) integral_CVi J
       (xi - centroid_i) (xi - centroid_i)^T dxi
```

`build_local_control_volume_cell_geometry` combines moments with
origin-moment accumulation. It rejects merge chains and requires all targets
to be local owned cells.

### Shifted-Torus Producer

The shifted-torus closed-box fixture computes full-cell and solid-overlap
moments with three-point Gauss integration in every coordinate. Fluid moments
are full moments minus solid moments.

A positive-volume storage cell is considered for merging when either:

- its center lies in the solid, or
- its fluid fraction is below `0.5`.

Candidate targets must be owned, positive-volume, face adjacent, and not merge
sources. Selection uses:

1. largest shared open-face physical measure;
2. shortest fluid-centroid distance;
3. fixed axis and sign order.

If no local target exists, the cut cell remains an active owner. Aggregates
never cross shards.

The geometry builder checks:

- local and idempotent owner mapping;
- no merge chains;
- positive active-owner volume;
- finite active-owner moments;
- exact `is_aggregate_target` semantics;
- no first-order reconstruction fallback in the convergence fixture.

## Face Geometry

The fast path retains dense structured faces when a face is a full-fluid
interface and does not have a compact row. Face ownership is exclusive:
every geometric face is represented by either the dense path or one compact
row, never both.

`LocalControlVolumeFaceRows3D` stores every other unique interface:

- `CV_FACE_INTERIOR` for an aggregate-touching full interface;
- `CV_FACE_PARTIAL` for a partially open interface;
- `CV_FACE_CUT_WALL` for an embedded physical wall;
- `CV_FACE_PHYSICAL_BOUNDARY` only when an embedded boundary leaves a partial
  domain-boundary face that the regular closure cannot represent.

Full coordinate-aligned physical boundaries stay on the dense structured
path. They use `LocalBoundaryFaceBC3D` and the physical face/topology/corner
halo closure. Merely being adjacent to a physical boundary does not activate
control-volume polynomial replacement on tangential dense faces.

An interior row has one minus owner and one local or remote plus owner. A
boundary row has only a minus owner. Each nonempty rectangular patch has four
two-dimensional Gauss points, oriented logical area covector weights, and
metric, magnetic-field, and projector data evaluated at those points.

Nonrectangular open regions are decomposed into nonoverlapping rectangles.
Faces internal to one aggregate are omitted. A local interior row is evaluated
once and scattered with equal and opposite signs.

A regular owner next to the compact region can still have ordinary dense faces
on its other sides. A full face whose structured support touches cut,
aggregate, or merged-source storage is a regular transition face: its dense
mask is closed and one compact `CV_FACE_INTERIOR` row owns the interface.
`LocalRegularTransitionFaceRows3D` indexes those rows and records validity,
remote ownership, and sample count without duplicating their metric or
quadrature payload.

Dense faces between untouched regular owners retain the original structured
stencil. There is no full-array polynomial replacement pass. This is
intentionally a face-level contract: requiring whole-cell path exclusivity
would recursively promote every cell in a connected fluid domain.

Partial physical-boundary rows use the same moment-aware cubic polynomial
as embedded cut-wall rows. Full coordinate-aligned physical faces do not
create reconstruction rows; their normal closure comes from the regular face
BC and their tangential faces retain centered structured stencils.

At a shard boundary, each shard stores a mirrored row oriented outward from its
local owner. Both rows use the same quadrature geometry and exchanged remote
polynomial, so they compute the same physical interface flux with opposite
local divergence signs.

## Field Boundary Data

`LocalControlVolumeBoundaryBC3D` is field specific and row aligned with
`irregular_faces`.

It carries:

- BC kind;
- value at the boundary-face centroid;
- value at every face quadrature point;
- active-row mask.

For Dirichlet data, "Dirichlet" means the field value is known on the physical
wall. The centroid value enters reconstruction. Quadrature values are
available for boundary face-value fluxes. `BC_NORMALFLUX` and `BC_NOFLUX`
apply directly to the integrated normal flux.

The native four-field EB path groups these objects in
`LocalFciDrbEBControlVolumeBCBundle`. Derived fields such as pressure, current,
and density times parallel velocity derive collocated BC values from the same
field bundles.

## Moment Reconstruction

`precompute_local_moment_reconstruction` builds compact cubic reconstruction
metadata outside JIT. Runtime code does not factor a least-squares matrix.

For an active owner `i`, reconstruction is centered at its aggregate centroid:

```text
u_i(x) = U_i
       + g_i . d
       + 1/2 H_i : (d d^T - M2_i)
       + 1/6 T_i : (d d d - M3_i)
d = x - centroid_i
```

The subtraction of `M2_i` makes the polynomial average over control volume
`i` equal to stored finite-volume value `U_i`.

For a neighboring owner `j`, the cell-average equation is:

```text
U_j - U_i =
  g_i . d_ij
  + 1/2 H_i : (M2_j + d_ij d_ij^T - M2_i)
  + 1/6 T_i : (M3_j + sym(d_ij, M2_j) + d_ij d_ij d_ij - M3_i)
```

For a Dirichlet boundary centroid `w`, the wall equation is:

```text
u_w - U_i =
  g_i . d_iw
  + 1/2 H_i : (d_iw d_iw^T - M2_i)
  + 1/6 T_i : (d_iw d_iw d_iw - M3_i)
```

There are three gradient, six symmetric Hessian, and ten symmetric
third-derivative coefficients. Coordinates
are normalized by local spacing and equations receive inverse-square distance
weights. The host precompute uses rank-revealing SVD:

- radius-one unique active owners are considered first;
- radius two is added when rank or conditioning is inadequate;
- at most 48 nearest unique samples are retained;
- transforms and condition diagnostics are stored in
  `LocalMomentReconstruction3D`.

The shifted-torus convergence fixture requires rank 19 and cubic order on
every active reconstruction row. Lower-order fallbacks remain available
for non-convergence callers.

`build_local_control_volume_polynomial_from_field` performs the runtime work:

1. read local and remote sample averages;
2. read field-specific Dirichlet centroid values;
3. assemble only the right-hand-side vector;
4. apply the precomputed transform by matrix-vector multiplication;
5. gather each active owner's authoritative row through
   `target_row_for_cell`;
6. leave merged-source gradient and Hessian zero;
7. exchange owner value, gradient, Hessian, and validity for remote faces.

The dense row gather is intentional. Padded compact rows have placeholder
target indices and must never be allowed to overwrite a real owner through a
scatter update.

For irregular and guard owners it solves for three gradient, six symmetric
Hessian, and ten symmetric third-derivative coefficients.  Dense bulk owners
retain the structured path.  The resulting `LocalControlVolumePolynomial3D` evaluates both point values and
point gradients. `as_cell_gradient()` preserves `LocalCellGradient3D` for
gradient-consuming operator APIs.

### Physical Boundary Gradient

Full coordinate-aligned physical faces remain on the dense regular path.
For a Dirichlet face, `LocalRegularBoundaryMomentClosure3D` stores two
precomputed finite-volume derivative functionals:

```text
du/dx|face = w_wall U_face
           + w_0 U_0 + w_1 U_1 + w_2 U_2

du/dx|centroid_0 = v_wall U_face
                 + v_0 U_0 + v_1 U_1 + v_2 U_2
```

`U_face` is the `J`-weighted average over the physical face patch and the
three inward `U_i` values are `J`-weighted cell averages. The weights reproduce
the normal derivative of cubic polynomials under those same moment
functionals. They are geometry metadata computed outside JIT; applying either
one at runtime is one four-value dot product. The face derivative feeds
conservative projected fluxes. The first-centroid derivative patches only the
normal component consumed by Poisson bracket, curvature, and parallel-gradient
operators. Tangential components retain the completed structured stencil.
The upper-face orientation is included in the stored weights, so both results
use the positive coordinate derivative convention.

This closure applies only to regular Dirichlet faces. Neumann and prescribed
normal-flux conditions enter the dense flux directly and do not use the
Dirichlet derivative functional. Embedded cut walls and partial physical
faces continue to evaluate the moment-aware control-volume polynomial at
their compact face quadrature points.

The shifted-torus finite-volume harness projects exact radial Dirichlet data
to the same `J`-weighted face average before constructing
`LocalBoundaryFaceBC3D`. Passing a point value at the face center into this
finite-volume functional is inconsistent and reintroduces a boundary-local
truncation error.

## Halo And Shard Flow

For a field entering an operator:

1. expand every positive-volume storage cell from its mapped owner;
2. inject owned storage into the halo array;
3. exchange topology and shard halos;
4. fill physical ghosts for dense ordinary faces;
5. build the polynomial from owner averages and compact BC data;
6. exchange reconstructed polynomial coefficients for remote compact faces.

Raw `jnp.roll` on owned arrays is never a valid cross-shard sample. Periodic
remote coordinates are unwrapped before reconstruction equations are formed.

The legacy runtime payload still requires aggregate ownership to be shard
local.  The canonical topology migration replaces that limitation with a
single global owner map and explicit remote-owner metadata; its runtime
aggregate exchange is the remaining migration item.  Interface fluxes remain
sharding compatible through coefficient exchange and mirrored compact rows.

## Conservative Flux And Divergence

Dense ordinary faces use the existing structured face kernels. Compact faces
evaluate the same field polynomial at face quadrature points.

Parallel scalar flux uses:

```text
F_q = J_q (area_covector_q . b_q) u_face_q
```

Interior `u_face_q` is the average of the two owner reconstructions at the same
quadrature point. A Dirichlet boundary uses its boundary face value.

Projected perpendicular or parallel diffusion uses:

```text
F_q = J_q area_covector_q . P_q . grad(u)_face_q
```

Interior face gradients are averages of the two owner reconstructions.
Embedded cut-wall and partial physical-boundary gradients come from the
polynomial whose equations include the relevant Dirichlet wall data.
Coordinate-aligned full physical boundaries use the regular face BC closure.
There is no field-specific gradient replacement after polynomial evaluation.

`_local_control_volume_integrated_divergence`:

1. converts open dense-face flux densities to integrated face fluxes;
2. scatters storage-cell dense sums to mapped owners;
3. adds each compact row to the minus owner;
4. subtracts the same row flux from a local plus owner;
5. divides by `cells.aggregate_volume`;
6. masks merged sources and inactive storage to zero.

Cut-wall boundary fluxes and active/aggregate interior fluxes therefore share
one face representation. No inactive source value participates in a compact
flux.

## Native Four-Field RHS

`LocalFciDrbEBRhs.evaluate_stage` follows this sequence:

1. build regular-face and control-volume BC bundles;
2. expand and prepare state halos;
3. reconstruct phi with `LocalPerpLaplacianInverseSolver`;
4. prepare phi storage and halo data;
5. build one polynomial per primitive field;
6. derive BCs and polynomials for pressure, current, and scalar flux products;
7. obtain gradients from those polynomials;
8. evaluate parallel gradients, Poisson brackets, and curvature using
   aggregate-centroid metric and magnetic coefficients;
9. evaluate conservative parallel and projected flux divergences through the
   unified face path;
10. add sources and mask nonowners.

The phi inverse solver uses the same `control_volume_geometry`, compact phi BC,
polynomial reconstruction, face quadrature, and aggregate volume as the RHS.
GMRES active masks and volume-weighted convergence diagnostics are aligned
with active owners. The shifted-torus MMS configuration requests an algebraic
target tolerance of `1e-11` and accepts no finite relative residual above
`5e-5` in the operator-convergence harness.
The solver convergence and failure flags are still reported, but exhausting the
iteration limit does not fail this diagnostic when the achieved residual meets
that acceptance threshold.

Products of primitive finite-volume fields are not formed as only
`U_i V_i`. For smooth fields, the leading moment covariance is retained:

```text
<u v>_i = U_i V_i + grad(u)_i^T M2_i grad(v)_i + O(h^3)
```

The derived owner average is then halo-prepared and reconstructed like any
other field.

Legacy operator arguments remain temporarily available for slab and
non-embedded callers. Supplying `control_volume_geometry` selects the unified
path and requires the aligned field BC and polynomial.

## Shifted-Torus MMS Flow

The convergence fixture projects all analytic quantities with the same
three-point `J dxi` integration used for geometry:

- initial and final state;
- phi;
- source;
- exact time derivative.

Raw storage-cell averages are scattered into aggregate owner averages using raw
physical volumes. Error norms use `aggregate_volume` and active owners:

```text
L2 = sqrt(sum_i V_i error_i^2 / sum_i V_i)
Linf = max over active owners
```

Combined norms are secondary. Per-field volume-L2 and active-owner Linf are
the acceptance quantities.

RK stage time is carried explicitly in stage data. It must never be inferred
from a projected field such as phi: finite-volume projection and aggregation
change pointwise correlations enough to create a false nonzero stage time,
which in turn corrupts source and wall data only near reconstruction rows.

`--operator-convergence-only` compiles separate kernels for:

- exact-phi parallel gradient;
- parallel density-flux divergence;
- a nondegenerate Poisson bracket;
- a nondegenerate curvature operator;
- perpendicular phi Laplacian;
- each full t=0 RHS field;
- optional phi algebraic solve.

It reports bulk, one-wall, multi-wall, aggregate-target, retained-cut-cell,
reconstruction-row, and remote-interface categories without constructing the
old high-volume LS diagnostic payload. It also partitions otherwise dense
cells into `dense_compact_d1`, `dense_compact_d2`, and `dense_far`. The first
two groups are one and two coordinate-neighbor layers from an irregular-face,
reconstruction-row, or aggregate-target owner. These bands distinguish an
error generated by compact-face machinery from an error propagated into the
dense operator.

For the largest parallel-divergence and perpendicular-Laplacian errors, the
diagnostic reports all six dense-face contributions. Each entry includes the
integrated numerical flux, the matching exact face quadrature, whether the
dense mask is closed in favor of a compact transition row, and whether the
neighboring owner has a valid polynomial reconstruction. This is
diagnostic-only work and does not alter face selection or operator output.

With `--skip-operator-phi-solve`, the full-RHS diagnostic consumes the
projected exact phi owner field directly and omits the separate algebraic solve
check. This mode isolates spatial RHS discretization from both GMRES error and
the discrete phi inversion. Without the flag, the full RHS reconstructs phi
through `LocalPerpLaplacianInverseSolver` before evaluating the equations.

The full time sweep scales RK steps linearly with resolution. The CLI defaults
to a minimum observed order of `1.8` for every field's volume-L2 and
active-owner Linf norm.

## Canonical Migration Status

The cleanup is intentionally staged.  The canonical host-side implementation
now lives in `drbx.geometry.fci_control_volumes` and
`drbx.native.fci_control_volume_operators`:

```text
raw moments -> GlobalControlVolumeTopology3D
            -> LocalControlVolumeGeometry3D
            -> LocalMomentReconstruction3D / direct face-functional weights
```

It has characterization coverage for direct owner maps, central-moment
translation, unique physical faces, periodic seams, cross-shard owner
references, and cubic finite-volume basis reproduction.  In particular,
`LocalControlVolumeGeometry3D.remote_aggregate_id` distinguishes an ID merely
referenced by a local source from an aggregate physically owned on the shard.

The shifted-torus fixture now enters cubic reconstruction through
`precompute_local_moment_reconstruction`.  Its JAX operator kernels still
consume the historical row payload while the direct face-functional gather and
mirrored cross-shard flux evaluator are migrated.  Therefore the following
legacy statements describe the current runtime compatibility, not the target
architecture.  Do not delete those row types until compact parallel and
projected fluxes both execute through direct functionals and their one- and
four-shard tests pass.

## Sharding Compatibility Matrix

| Subsystem | Compatible | Requirement |
| --- | --- | --- |
| Cell ownership and moments | Yes | Every source maps directly to a local owned target |
| Host geometry construction | Yes | Build per shard, then stack matching padded pytrees |
| Cubic moment precompute | Yes | Host-side local rows plus explicit remote sample metadata |
| Runtime polynomial build | Yes | Prepared halos for remote cell equations |
| Ordinary dense faces | Yes | Existing halo/topology preparation |
| Local compact interior face | Yes | One unique row, two local owners |
| Shard-boundary compact face | Yes | Mirrored rows plus polynomial coefficient exchange |
| Cut-wall boundary face | Yes | Field-specific compact BC on each local row |
| Regular transition face | Yes | Unique compact row; mirrored row and coefficient exchange when remote |
| Physical Dirichlet closure | Yes | Three inward owners must be ordinary and local; runtime applies face and first-centroid four-value functionals |
| Phi GMRES | Yes | Active-owner mask and collective global reductions |
| Cross-shard aggregate ownership | Metadata only | Canonical global topology and local remote IDs support it; legacy JAX operator rows still require direct functional gather/exchange migration |
| Global debug assembly | Debug only | Host gather, not a production SPMD kernel |

## Required Invariants

The following are correctness conditions, not optional diagnostics:

- owner mapping is local, direct, and idempotent;
- every active owner has positive finite aggregate volume and moments;
- ordinary cells are not labeled aggregate targets;
- aggregate volume is conserved;
- merged sources have zero gradient, Hessian, and operator output;
- dense faces exclude every compact or aggregate-internal interface;
- every nonregular full transition has one valid compact row and a closed
  dense mask;
- untouched dense faces retain the structured regular-face stencil;
- full coordinate-aligned physical faces remain dense and do not activate
  tangential polynomial replacement;
- local compact interior fluxes cancel to roundoff before volume division;
- mirrored shard rows compute equal physical flux;
- no convergence-fixture row uses linear fallback;
- MMS projection and operator divergence use the same physical volume.

## Accuracy Boundary

Cubic reconstruction exactly reproduces cubic finite-volume data on the
irregular-owner mask and supplies the moment data used by the direct compact
face-functional migration. Full coordinate boundaries use the regular
moment-derived boundary functional described above until physical-boundary
faces migrate to the same direct-functional representation. The two paths are
deliberately separate during this transition: the coordinate boundary has an
ordered inward stencil, whereas an embedded wall requires multidimensional
moment data.

The operator-only Linf gate remains authoritative. If a cut-wall category
converges below order `1.8`, the next change must add information to the
boundary reconstruction (for example, a higher-order wall-normal equation or
higher moments). Weight tuning or reintroducing one-wall/multi-wall hard
projection is not a valid substitute.
