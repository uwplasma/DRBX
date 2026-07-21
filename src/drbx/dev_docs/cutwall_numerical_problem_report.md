# Cut-Wall Numerical Problem and Repair Status

## Purpose

This report describes the fundamental numerical problem exposed by adding
embedded cut walls and cell agglomeration to the native FCI operators. It is
intended to support both implementation work and a literature review spanning
scrape-off-layer (SOL) fluid models, embedded-boundary finite-volume methods,
cut-cell stabilization, and high-order boundary closures.

This document is organized around the mathematical problem and the evidence
from the shifted-torus manufactured-solution (MMS) tests. For the detailed
runtime call chain, see
[cutwall_agglomeration_ls_call_chain.md](cutwall_agglomeration_ls_call_chain.md).
For the cases a logical cell can occupy, see
[embedded_control_volume_cell_cases.md](embedded_control_volume_cell_cases.md).

## Executive Summary

The cut-wall work exposed a representation inconsistency that was mostly
hidden in the original regular-grid solver.

The solver stores finite-volume cell averages, but many original operators
behave as if those values were point samples at geometric cell centers. On a
uniform, uncut mesh, symmetry makes that approximation second-order accurate
and the distinction is easy to overlook. Once cells are cut, merged, shifted
to aggregate centroids, or connected through irregular faces, those
cancellations disappear. The same formulas can then become inaccurate,
nonconservative, or unstable.

The central problem is therefore not one indexing bug. The following pieces
must all describe the same discrete control volume:

1. The location and finite-volume meaning of each stored unknown.
2. The volume, centroid, and moments of each active control volume.
3. The reconstruction used to recover point values and derivatives.
4. The ownership, geometry, and quadrature of every face.
5. The interpretation and enforcement of wall boundary conditions.
6. The conservative scatter of one face flux to adjacent owners.
7. Halo and shard communication for reconstruction support.

Several concrete wiring bugs have been found and repaired. The remaining
challenge is approximation quality near embedded boundaries, especially the
wall-normal derivative used by projected diffusive fluxes. A Dirichlet wall
value strongly constrains the polynomial trace, but does not by itself
determine a reliable one-sided normal derivative.

## 1. Discrete Meaning of a Stored Field

For an active control volume `CV_i`, the stored field is intended to be the
physical finite-volume average

```text
U_i = (1 / V_i) integral_CV_i J(xi) u(xi) dxi
V_i = integral_CV_i J(xi) dxi.
```

On a regular symmetric cell,

```text
U_i = u(x_i) + O(h^2),
```

so treating `U_i` as a point value at the cell center often still produces a
second-order approximation. Central differences also benefit from symmetry
and cancellation between opposite faces.

For a cut or agglomerated cell, however:

- the fluid centroid is not the logical cell center;
- the fluid volume is not the full logical-cell volume;
- the second and third moments are not symmetric;
- an aggregate value represents several logical cells;
- the aggregate centroid may be substantially displaced;
- the two sides of a derivative stencil are no longer geometrically paired.

Using an aggregate average as a point value at an original logical center is
therefore inconsistent. Differentiation can amplify a small value mismatch:

```text
value error O(h^2) / distance O(h) -> derivative error O(h).
```

If the cut distance is much smaller than `h`, the amplification can be larger.
This is why the failure appeared after cut walls and agglomeration were added.
The regular solver relied on geometric symmetry that is absent near an
embedded boundary.

## 2. The Small-Cut-Cell Problem

A conservative finite-volume update has the form

```text
dU_i/dt = -(1 / V_i) sum_f F_f + S_i.
```

When `V_i` is very small, both explicit stability restrictions and ordinary
face-flux errors are amplified by `1 / V_i`. This is the classical small-cell
problem in Cartesian cut-cell methods.

The current implementation uses local agglomeration:

- selected small or center-in-solid cut cells become merged sources;
- each source maps to a nearby active owner;
- source volume and geometric moments are accumulated into that owner;
- source fields are volume-averaged into the aggregate value;
- merged sources receive zero independent operator output;
- faces internal to one aggregate are omitted.

The authoritative location of the aggregate unknown is its fluid-volume
centroid, not either member's original logical center.

Current aggregate geometry includes:

- raw and aggregate physical volume;
- identity-or-target owner mapping;
- merged-source and active-owner masks;
- aggregate-target and member counts;
- aggregate centroid;
- symmetric second central moment;
- symmetric third central moment.

Agglomeration remains local to a shard. Cross-shard aggregates are forbidden
because they would require distributed ownership, volume reduction, and
synchronized state updates. Values, moments, and reconstruction coefficients
may cross shards; ownership does not.

## 3. Why the Original Directional Stencils Became Inadequate

The original local gradient builder used coordinate-direction stencils. On a
regular grid these consume samples at known, symmetric positions. Near an
embedded wall, a nominal stencil entry may instead refer to:

- an inactive solid storage cell;
- a merged source whose value is not an independent degree of freedom;
- an aggregate value located at a displaced centroid;
- a wall value at an intersection point;
- a remote owner requiring halo metadata;
- a compact transition reconstructed from another control volume.

Simply replacing the inactive sample with a wall value does not make the
three-dimensional gradient consistent with the finite-volume averages. It
also does not account for aggregate moments.

The earlier least-squares repair improved locality but retained several
problems:

- it initially replaced all three gradient components even when only one
  coordinate stencil was unsafe;
- it initially used raw local rolling at shard boundaries;
- wall values entered as weak samples rather than authoritative boundary data;
- aggregate values were anchored at logical centers instead of aggregate
  centroids;
- one-wall and multi-wall post-projections could satisfy the wall equation
  while degrading the physical gradient;
- a linear or quadratic polynomial did not provide sufficient order for some
  face derivatives.

Those observations motivated the unified embedded-control-volume and
moment-aware reconstruction path.

## 4. Moment-Aware Cubic Reconstruction

Irregular owners now use a cubic polynomial about the aggregate centroid. It
has 19 nonconstant coefficients:

- 3 gradient coefficients;
- 6 symmetric Hessian coefficients;
- 10 symmetric third-derivative coefficients.

For a neighboring control-volume average, the reconstruction equation is

```text
U_j - U_i =
    g . d
  + 1/2 H : (M2_j + d d^T - M2_i)
  + 1/6 T : (M3_j + sym(d, M2_j) + d d d - M3_i).
```

Here `d` is the displacement from owner `i`'s aggregate centroid to owner
`j`'s centroid. The translated third-moment term is

```text
sym(d, M2)_abc = d_a M2_bc + d_b M2_ac + d_c M2_ab.
```

For a Dirichlet wall point,

```text
u_w - U_i =
    g . d_w
  + 1/2 H : (d_w d_w^T - M2_i)
  + 1/6 T : (d_w d_w d_w - M3_i).
```

The system uses unique active owners and active Dirichlet quadrature points.
Coordinates are scaled by local grid spacing, and geometry-aware distance
weights are applied. A rank-revealing host-side factorization precomputes the
transform. Runtime reconstruction is a matrix-vector product, not an
iterative solve.

The reconstruction is restricted to the irregular region and a guard layer.
The dense regular bulk retains its structured kernels.

At `N=40`, current diagnostics report:

```text
cubic reconstruction rows       6816
rank                             19 on every selected row
quadratic fallbacks              0
linear fallbacks                 0
invalid rows                     0
maximum reported condition       about 28
```

The cubic reconstruction substantially improved near-wall first derivatives.
Compared with the previous quadratic path, the reconstruction-row errors for
parallel gradients dropped by nearly an order of magnitude in representative
fields. This confirms that the cubic moment transform is active, numerically
well-conditioned, and useful.

## 5. Dirichlet Values Do Not Determine Wall-Normal Derivatives

A Dirichlet boundary condition specifies

```text
u(x_w) = u_w.
```

It does not directly prescribe

```text
du/dn at x_w.
```

A multidimensional least-squares polynomial can match all wall values closely
while having a poor wall-normal derivative. Reasons include:

- reconstruction support is one-sided;
- wall equations constrain values rather than derivatives;
- cell-average and wall equations compete in an overdetermined fit;
- normal and tangential polynomial coefficients are correlated;
- differentiation amplifies small coefficient errors;
- the sample cloud may have weak leverage in the normal direction;
- aggregate centroids can be displaced relative to the wall patch.

The cubic diagnostic isolated precisely this behavior. At the worst cut-wall
face:

- polynomial trace residuals were only approximately `1e-5` to `1e-4`;
- tangential derivatives were reasonably accurate;
- the coordinate-normal derivative had the wrong magnitude and sometimes the
  wrong component sign;
- replacing only the normal derivative by the exact one recovered the exact
  projected face flux.

This shows that, for that failure:

- face geometry was correct;
- metric and projector evaluation were correct;
- quadrature and face scatter were correct;
- the remaining error was the normal derivative from the unconstrained cubic
  polynomial.

## 6. Current Embedded-Wall Normal-Derivative Closure

The cut-wall geometry precomputes a dedicated one-dimensional normal
functional for each axis-aligned wall patch. It uses:

- the Dirichlet value at the wall quadrature point;
- two distinct, well-separated inward control-volume samples;
- their coordinates relative to the wall.

The production projected-flux path now performs the following steps:

1. Evaluate the cubic polynomial and gradient at the wall quadrature point.
2. Evaluate inward reconstructed samples at the same tangential location.
3. Apply the wall-plus-inward normal derivative functional.
4. Replace only the coordinate-normal gradient component.
5. Preserve both cubic tangential components.
6. Apply the perpendicular projector, metric, Jacobian, and area weights.
7. Integrate the face flux and scatter it to the owning control volume.

In compact notation,

```text
grad_applied = grad_cubic
             + e_a (D_wall,a u - e_a . grad_cubic),
```

where `a` is the coordinate direction normal to the axis-aligned embedded
wall.

This repair has been wired into the production irregular projected-flux path
and the focused diagnostic path. It has not yet been validated by the next
`N=40` run.

There is an important order concern. A quadratic one-sided functional can be
second-order accurate for a face derivative, but a finite-volume second
derivative divides a face-flux difference by a cell length. At a boundary,
where opposite-face error cancellation is weaker, second-order face-gradient
accuracy may not guarantee second-order pointwise operator accuracy. Meeting
the requested `1.8+` Linf order may require a cubic or constrained Hermite
normal functional with moment-aware control-volume samples.

The restored closure is therefore the immediate, evidence-based repair, but
not assumed to be the final high-order boundary method.

## 7. Conservative Face Ownership

A conservative interior face must produce one physical flux and apply equal
and opposite integrated contributions:

```text
R_minus += F_f / V_minus
R_plus  -= F_f / V_plus.
```

Earlier implementations mixed:

- dense structured regular-face fluxes;
- sparse regular-face contribution rows;
- cut-wall boundary fluxes;
- aggregate source-to-target routing.

This created concrete failure modes:

- a structured face was closed but only one owner received a sparse flux;
- a sparse row sampled inactive or merged-source storage;
- dense and compact paths both contributed to one physical face;
- opposite contributions used separately reconstructed fluxes;
- an aggregate target received flux divided by an inconsistent volume.

The unified representation now classifies each physical face into one
exclusive path:

- ordinary full-fluid face: dense structured path;
- full face whose complete stencil touches compact geometry: transition row;
- partial open face: irregular quadrature row;
- embedded wall face: boundary quadrature row;
- face internal to one aggregate: omitted.

Interior transition and irregular interfaces are represented once. The same
integrated flux is scattered with opposite signs to the two owners. This
separates physical embedded-wall fluxes from active-to-aggregate interior
fluxes.

## 8. Dense-to-Compact Transition Faces

A transition face can be geometrically full and regular while its structured
operator support touches an irregular, merged, or reconstruction-controlled
owner.

The dense projected-gradient formula needs more samples than the two cells
directly adjacent to the face. Reading logical storage blindly can introduce:

- inactive zeros;
- merged-source values;
- values associated with a different owner;
- aggregate values interpreted at the wrong location;
- stale local-shard periodic samples.

Transition metadata now stores the complete support of the corresponding
dense structured functional. Each support entry is either:

- a direct ordinary control-volume average; or
- a virtual regular-cell average reconstructed from an irregular owner.

For a virtual regular-cell average,

```text
U_virtual = U_i
          + g . d
          + 1/2 H : (M2_v + d d^T - M2_i)
          + 1/6 T : (M3_v + sym(d, M2_v) + d d d - M3_i).
```

The transition row then applies the same linear face functional used by the
dense structured operator. Consequently, an all-ordinary transition support
reduces exactly to the dense formula.

The current transition diagnostics show:

- zero invalid transition rows;
- complete support with at most 14 samples in the `N=40` case;
- source-storage poisoning independence in focused checks;
- exclusive dense/transition ownership;
- no conservation imbalance in the single-shard diagnostic.

This indicates that the earlier dense-to-compact wiring gap has been repaired
at the architectural level.

## 9. Remaining Parallel Scalar-Flux Error

The cubic reconstruction greatly improved first derivatives, but the `N=40`
parallel density-flux divergence changed very little.

At the worst aggregate target, the diagnostic found approximately

```text
numerical divergence              1.0086e-1
reference divergence              1.1613e-1
divergence with exact irregular   1.1612e-1
divergence with exact dense       1.0086e-1.
```

Thus the error is carried by attached irregular faces, not the dense faces.
However, transition functionals evaluated with exact support averages are
already close to their production values. This means the dominant residual is
not obviously a bad cubic coefficient or a remaining owner-routing bug.

Remaining possibilities include:

- ordinary second-order truncation with a large coarse-grid coefficient;
- face-value accuracy that is insufficient after divergence cancellation;
- inconsistency between the exact face reference and the projected
  finite-volume source;
- insufficient quadrature order for the metric-weighted scalar flux;
- loss of cancellation among multiple irregular faces of one aggregate;
- a boundary face value that must be treated as a face average rather than a
  pointwise quadrature trace.

The decisive next measurement is `N=40` versus `N=80` using the same merge and
shard policy. A factor near four supports second-order truncation. A factor
near two or a flat error indicates another consistency defect.

## 10. Perpendicular Laplacian Sensitivity

The perpendicular Laplacian is a divergence of a projected gradient:

```text
lap_perp(u) =
  (1 / J) d_i [J (g^ij - b^i b^j) d_j u].
```

Errors can enter through:

- reconstructed point gradients;
- wall-normal derivative closure;
- face-gradient interpolation;
- metric and magnetic projector evaluation;
- face quadrature;
- face ownership and sign;
- aggregate volume normalization.

The latest cubic run improved all-active L2 error compared with the quadratic
path, but worsened the worst cut-wall Linf error. Exact substitution then
showed:

- exact dense fluxes did not repair the worst cell;
- exact irregular fluxes recovered the reference operator;
- exact tangential data with the numerical normal derivative retained the
  error;
- the numerical tangential data with the exact normal derivative recovered
  the exact cut-wall flux.

This is unusually strong attribution. The next normal-closure run should show
whether the restored dedicated functional removes that spike. If it improves
the coarse result but does not reach the desired order, the normal functional
must be raised in order or enforced through a constrained multidimensional
reconstruction.

## 11. Regular Physical Radial Boundaries

The physical radial boundaries at `x_min=0.2` and `x_max=1.0` are ordinary
coordinate boundaries, not embedded cut walls.

They use a separate moment-aware closure built from:

- a Dirichlet face average;
- the first three inward `J`-weighted control-volume averages.

It evaluates normal derivatives at both:

- the physical face, for conservative projected fluxes;
- the first owner centroid, for Poisson bracket, curvature, and parallel
  gradient consumers.

Shifted-torus reproduction diagnostics recover constant-through-cubic radial
bases to approximately `1e-10`, so the functional construction itself is
algebraically correct.

`poisson_omega` nevertheless retains a large lower-radial-plane Linf error.
Its diagnostic shows contributions from both radial and tangential gradient
components. This issue is distinct from the embedded-wall normal derivative.
Possible causes include:

- finite-volume versus pointwise tangential gradient semantics at the first
  owner plane;
- projection of the analytic Poisson-bracket reference;
- cancellation between cross-product components;
- boundary-local first-order truncation hidden by a higher global L2 order.

The physical radial boundary should remain on its specialized structured and
moment-aware path. Embedded cut-wall machinery should not replace it.

## 12. Halo and Sharding Requirements

The current design permits:

- shard-local aggregate ownership;
- owned reconstruction targets;
- exchanged remote values and geometric moments;
- exchanged cubic polynomial coefficients;
- mirrored transition rows at cross-shard regular interfaces.

It does not permit a merged source to map to an owner on another shard.

Any cross-shard reconstruction sample must use prepared halo or exchanged
metadata. Raw `jnp.roll` on a local shard is invalid because it wraps within
the shard rather than across the global periodic domain.

The single-shard tests establish local mathematical behavior. The same
`N=40,80` operator tests must then run with `1 1 4` sharding to verify:

- matching reconstruction masks;
- valid remote transition samples;
- identical mirrored face fluxes;
- no shard-boundary loss of order;
- no accidental cross-shard aggregate ownership.

## 13. MMS Projection and Error Norms

The exact state, phi, source, and time derivative must be projected using the
same physical fluid control volumes used by the operator:

```text
U_exact,i = (1 / V_i) integral_CV_i J u_exact dxi.
```

Comparing an aggregate average against the exact solution at an original cell
center introduces a false MMS residual. Diagnostics therefore report both:

- the exact aggregate average;
- the exact point value at the aggregate centroid.

The difference is expected to be `O(h^2)` for a smooth field, but it must not
be silently interpreted as operator error.

Primary convergence norms should be aggregate-volume weighted and evaluated
only on active owners. Merged sources are storage, not independent degrees of
freedom. Useful categories include:

- all active owners;
- bulk owners;
- one-wall owners;
- multi-wall owners;
- aggregate targets;
- retained cut cells;
- reconstruction rows;
- first and second dense-to-compact layers;
- physical radial owner planes;
- remote interfaces.

L2 and Linf communicate different behavior. A small set of boundary cells can
converge slowly in Linf while contributing little to volume-weighted L2. The
project's final target explicitly requires approximately second-order behavior
in both, so boundary-local consistency cannot be ignored.

## 14. SOL Wall Physics Versus the Dirichlet MMS

The current manufactured-solution test uses exact Dirichlet values at embedded
walls because they provide a controlled verification problem.

A physical SOL wall does not generally impose arbitrary Dirichlet data for
every evolved field. Depending on the model, relevant conditions may include:

- Bohm or Bohm-Chodura sheath constraints;
- logical-sheath closures;
- nonlinear Robin conditions;
- prescribed particle, momentum, or heat fluxes;
- current closure and floating-potential conditions;
- characteristic boundary conditions;
- recycling and neutral source models.

Two questions must remain separate:

1. Is the embedded-boundary discretization mathematically consistent for a
   known Dirichlet MMS problem?
2. Is Dirichlet the physically appropriate SOL wall model?

The first question must be answered before more realistic sheath or flux
conditions can be trusted. The geometry and conservative face ownership
should remain unchanged when the boundary functional changes.

The long-term boundary API should support Dirichlet, Neumann, normal flux,
no-flux, Robin, characteristic, and nonlinear sheath functionals without
changing the control-volume topology.

## 15. Current Status

| Subsystem | Current status | Remaining concern |
| --- | --- | --- |
| Aggregate ownership | Local, idempotent, conservative | `N=100` topology change needs a later audit |
| Aggregate geometry | Volume, centroid, `M2`, and `M3` available | Validate cut-volume quadrature order |
| Cubic reconstruction | Rank 19 with no `N=40` fallbacks | Derivative quality remains geometry dependent |
| First parallel gradients | Strongly improved near walls | Measure `N=40` to `N=80` order |
| Dense/compact ownership | Exclusive face paths established | Confirm multi-shard equivalence |
| Transition virtual averages | Moment-aware cubic | Scalar divergence order remains unknown |
| Cut-wall Dirichlet trace | Accurate in diagnostics | Trace accuracy does not ensure derivative accuracy |
| Cut-wall normal derivative | Dedicated closure restored | Re-run and measure convergence |
| Perpendicular Laplacian | Largest error isolated to wall-normal flux | Validate closure; possibly raise its order |
| Regular radial closure | Cubic reproduction passes | Lower-plane Poisson Linf remains |
| Phi GMRES solve | Temporarily skipped | Revisit after forward operator consistency |
| Full RK/MMS convergence | Not ready | Operator convergence must pass first |

## 16. Immediate Validation Sequence

The next focused run is

```bash
python test_fci_cutwall_shifted_torus_4field.py \
  --resolutions 40 \
  --shard-counts 1 1 1 \
  --deactivate-center-in-solid \
  --operator-convergence-only \
  --skip-operator-phi-solve \
  --debug-operator-failures \
  2>&1 | tee shifted_torus_operator_cubic_normal_closure_n40.txt
```

Expected evidence from that run:

- cut-wall diagnostics report `closure valid=True`;
- `applied_grad` differs from `raw_grad` only in the wall-normal coordinate;
- the applied normal derivative is closer to the exact derivative;
- the worst projected cut-wall flux spike decreases;
- transition and tangential-gradient accuracy do not regress.

After that:

1. Run `N=40,80` on one shard to measure spatial order.
2. If the wall-normal error remains first-order, replace the quadratic normal
   functional with a higher-order moment or constrained Hermite functional.
3. Diagnose the parallel scalar-flux order separately using the `N=40,80`
   ratio.
4. Repeat `N=40,80` with `1 1 4` sharding.
5. Re-enable and validate the phi solve.
6. Only then run the full RK MMS convergence sweep.

## 17. Literature Search Map

### Embedded-boundary finite-volume methods

Search terms:

- embedded boundary finite-volume method;
- Cartesian cut-cell method;
- sharp-interface cut-cell discretization;
- high-order cut-cell elliptic operator;
- embedded boundary anisotropic diffusion;
- conservative finite-volume internal obstacle.

Questions to compare:

- How are cut-cell averages and moments represented?
- How are face gradients reconstructed at Dirichlet walls?
- Is boundary data enforced strongly, weakly, or through flux equations?
- What local and global convergence orders are proven or measured?

### Small-cell stabilization

Search terms:

- cut-cell small-cell problem;
- cell merging and agglomeration;
- conservative flux redistribution;
- state redistribution method;
- cut-cell CFL stabilization;
- volume-weighted conservative remapping.

Questions to compare:

- Does the method merge geometry, state, fluxes, or updates?
- Does it preserve conservation and positivity?
- Is the aggregate map decomposition independent?
- What happens to high-order moments after merging?

### Reconstruction from cell averages

Search terms:

- moment-fitting finite-volume reconstruction;
- polynomial reconstruction from cell averages;
- constrained least-squares cut-cell reconstruction;
- generalized moving least squares embedded boundary;
- polynomial-preserving recovery;
- rank-revealing stencil selection;
- WENO reconstruction on cut cells.

Questions to compare:

- Are control-volume moments included explicitly?
- How are one-sided sample clouds conditioned?
- How are wall equations weighted relative to cell-average equations?
- Are derivative constraints included directly?

### Boundary-normal derivative construction

Search terms:

- high-order normal derivative from Dirichlet data;
- Hermite reconstruction embedded boundary;
- one-sided finite-volume boundary derivative;
- superconvergent boundary flux;
- Dirichlet-to-Neumann reconstruction;
- boundary truncation error finite-volume Laplacian;
- constrained polynomial normal derivative;
- Nitsche embedded boundary finite volume;
- summation-by-parts SAT cut-cell boundary.

Questions to compare:

- Is the normal derivative an independent functional or taken from the bulk
  polynomial?
- What derivative order is required for second-order boundary-cell Linf
  accuracy?
- Are wall values treated pointwise or as face averages?
- Can several wall patches constrain one control volume without overfitting?

### Conservative interface coupling

Search terms:

- conservative mortar flux;
- compact-to-structured interface flux;
- hybrid finite-volume interface reconstruction;
- mimetic finite difference cut cells;
- compatible discretization discrete Green identity;
- single-valued numerical flux nonconforming interface.

Questions to compare:

- Is one flux evaluated per interface?
- How are nonmatching reconstruction spaces coupled?
- How is equal-and-opposite scatter guaranteed across partitions?

### Alternative embedded-interface methods

Search terms:

- ghost-fluid method;
- immersed interface method;
- sharp immersed boundary method;
- cut finite element method;
- discontinuous Galerkin embedded boundary;
- hybridizable DG embedded geometry;
- Nitsche unfitted finite element;
- Brinkman penalization plasma wall.

These methods offer different tradeoffs between conservation, geometric
complexity, conditioning, and ease of imposing nonlinear wall physics.

### SOL-specific boundary conditions

Search terms:

- Bohm sheath boundary condition fluid SOL;
- Bohm-Chodura boundary condition;
- logical sheath boundary condition;
- floating sheath potential fluid model;
- SOL particle and heat flux boundary;
- sheath current closure finite volume;
- recycling boundary condition edge plasma;
- embedded wall plasma fluid solver.

Questions to compare:

- Which evolved variables receive value, flux, or characteristic conditions?
- How is the sheath condition coupled to potential inversion?
- Are wall fluxes evaluated pointwise or integrated over wall faces?
- How are oblique magnetic fields and wall normals combined?

## 18. Long-Term Design Principle

Every embedded operator should satisfy one common contract:

```text
control-volume average
  -> moment-aware reconstruction
  -> one unique physical face flux
  -> equal-and-opposite conservative scatter
  -> division by the same physical aggregate volume.
```

Boundary conditions should enter as boundary equations or flux functionals,
not as arbitrary values stored in solid cells.

Dense structured formulas remain the fast regular-grid specialization. The
compact machinery must reduce to those formulas on regular geometry and add
only the information required by cut cells, aggregate owners, partial faces,
and physical walls.

The central unresolved mathematical question is how best to construct a
high-order, stable wall-normal derivative from finite-volume averages and
Dirichlet wall data while preserving conservative projected fluxes. The
current dedicated normal functional directly addresses the diagnosed failure;
the next convergence runs will determine whether its order is sufficient or a
constrained higher-order boundary reconstruction is required.
