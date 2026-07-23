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
For the latest completed sweep, measured orders, suspected failure locations,
and prioritized next steps, see
[cutwall_current_progress.md](cutwall_current_progress.md).
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

Several concrete wiring bugs have been repaired. Compact conservative fluxes
now use direct cubic moment-fitted functionals for the complete integrated
projected, parallel-value, and parallel-gradient fluxes. This removes the
assumption that every reconstructed Cartesian gradient component must be
individually accurate at the wall. Cubic polynomials remain in use for
cell-gradient consumers such as parallel first derivatives, Poisson brackets,
curvature, and nonlinear product averages.

The remaining challenge is measured approximation quality, not missing flux
wiring. A one-shard `N=6` audit verified valid direct rows and improved the
perpendicular-Laplacian error, but it did not contain any merged sources and
still showed large errors in `poisson_omega`, the perpendicular Laplacian, and
the electron-parallel full RHS. The subsequent agglomeration-enabled
`N=10,14` operator sweep with projected exact phi is summarized in
[cutwall_current_progress.md](cutwall_current_progress.md).

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

The canonical topology permits a source to merge across one face into a
directly adjacent shard, including a periodic seam. Prepared owner halos
supply its value; integrated source and compact-face residuals return to the
canonical owner through reverse face-halo accumulation. Edge- and
corner-routed remote aggregates remain unsupported.

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

## 6. Direct Embedded-Wall Flux Functionals

The dedicated one-dimensional wall-normal patch was useful diagnostically,
but it is no longer the production compact-flux algorithm. The current method
targets the complete face-integrated functional during geometry preprocessing:

```text
G_perp[u] = integral_f J a . P_perp grad(u) dA
G_par[u]  = integral_f J (a . b) u dA
G_bb[u]   = integral_f J a . (b b) grad(u) dA.
```

For every cubic basis mode, the target is evaluated with the stored face
quadrature, metric, magnetic field, projector, and oriented area covector. The
moment matrix contains aggregate-average, remote-average, and Dirichlet trace
observations. Weighted SVD then produces direct observation weights for each
integrated flux.

At runtime, `build_local_control_volume_field_closure` performs bounded
owned/halo/boundary gathers and three weighted sums. It does not construct a
face gradient or solve a reconstruction system. Because the target is the
physical scalar flux itself, normal/tangential coupling from `P_perp` is
preserved without requiring a separately fitted normal derivative.

This change addresses the diagnosed weakness more directly than increasing
the polynomial degree or tuning a one-dimensional normal stencil. It still
requires a convergence measurement: polynomial reproduction proves algebraic
consistency on the fitted space, but not the size or asymptotic order of the
smooth nonpolynomial MMS error on translated and agglomerated geometries.

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

The dense structured formula can read invalid storage when its support crosses
the compact band. Reading logical storage blindly can introduce:

- inactive zeros;
- merged-source values;
- values associated with a different owner;
- aggregate values interpreted at the wrong location;
- stale local-shard periodic samples.

The current topology closes that dense face and assigns one canonical compact
face evaluator. Its direct cubic functional gathers actual aggregate averages,
remote averages, and boundary observations; it does not manufacture virtual
regular-cell point samples. The resulting integrated flux is scattered to the
minus owner, the local plus owner when present, or the exact remote residual
halo destination.

Focused tests establish unique global face IDs, decomposition-independent
functional weights, source-storage poisoning independence, and conservative
reverse halo accumulation. The production operator still needs the planned
multi-resolution and multi-shard convergence measurements.

## 9. Current Parallel Scalar-Flux Evidence

The direct parallel-value functional is now active. In the one-shard `N=6`
audit, parallel density-flux divergence had approximately

```text
all-active volume L2     2.997e-2
all-active Linf          5.502e-2
invalid functional rows  0.
```

That run had no merged sources and only one resolution, so it cannot establish
the agglomerated order. Remaining possibilities include ordinary coarse-grid
truncation, insufficient metric-weighted face quadrature, imperfect
cancellation among multiple compact faces, or inconsistency in the analytic
finite-volume reference. The `N=10,14` sweep is the next discriminating
measurement.

## 10. Perpendicular Laplacian Sensitivity

The perpendicular Laplacian is a divergence of a projected gradient:

```text
lap_perp(u) =
  (1 / J) d_i [J (g^ij - b^i b^j) d_j u].
```

Errors can enter through:

- direct functional observation selection and weights;
- metric and magnetic projector evaluation;
- face quadrature;
- face ownership and sign;
- aggregate volume normalization.

The direct projected functional materially improved the one-shard `N=6`
baseline relative to the preceding reconstructed-gradient path:

```text
                              previous        direct functional
all-active volume L2          about 6.12e-1   1.717e-1
all-active Linf               about 3.55      1.342
invalid functional rows       n/a             0
```

The remaining error is still too large to claim success. Because the direct
target already contains normal/tangential projector coupling, the next action
is not to restore a separate wall-normal patch. First measure `N=10,14` order
and localize the worst categories. If order remains deficient, inspect
functional observation coverage, face quadrature, weight amplification, and
the exact finite-volume reference before increasing polynomial degree.

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

- aggregate ownership on the same or one directly adjacent shard;
- owned reconstruction targets;
- exchanged remote values and geometric moments;
- direct functional gathers from prepared face halos;
- one canonical cross-shard face evaluator;
- reverse face-halo accumulation of remote residuals.

It does not permit edge- or corner-routed remote aggregate ownership.

Any cross-shard reconstruction sample must use prepared halo or exchanged
metadata. Raw `jnp.roll` on a local shard is invalid because it wraps within
the shard rather than across the global periodic domain.

The single-shard tests establish local mathematical behavior. After the
`N=10,14` one-shard sweep is understood, the same resolutions must run with a
compatible decomposed layout to verify:

- matching reconstruction masks;
- valid remote functional observations;
- one evaluator and one remote residual destination per shared face;
- no shard-boundary loss of order;
- conservative cross-shard aggregate accumulation.

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
| Aggregate ownership | Global, direct, idempotent; nonzero merges demonstrated at `N=10,14,18` | Topology changes between resolutions and affects the worst aggregate |
| Aggregate geometry | Volume, centroid, `M2`, and `M3` available | Validate cut-volume quadrature order |
| Cubic reconstruction | Required on every shifted-torus active row; no fallback in the tested grids | One-wall owners have many cell equations but few boundary equations |
| Direct compact functionals | Projected, parallel-value, and parallel-gradient wired and audited face by face | Broad, weakly localized fits produce inaccurate individual fluxes despite full rank |
| Dense/compact ownership | Exclusive global face paths established | Confirm cluster multi-shard equivalence |
| Cross-shard residuals | Reverse face-halo accumulation implemented | Run decomposed operator convergence |
| Parallel density flux | Product-average input and conservative scatter cleared as first defects | Direct tangential face functional remains nonconvergent |
| Perpendicular Laplacian | Experimental two-owner shared flux materially improves `N=10,14` | Three-grid all-active order is `1.575`; one-wall aggregate failure remains at `N=18` |
| Regular radial closure | Cubic reproduction passes | Lower-plane Poisson Linf remains |
| Phi GMRES solve | Implemented but intentionally skipped | Re-enable after forward operator consistency |
| Full RK/MMS convergence | Not ready | Operator convergence must pass first |

## 16. Immediate Validation Sequence

The previous `N=10,14` forward sweep and the follow-up `N=18` perpendicular
diagnostic have completed. They did not pass the convergence gates. The
current experiment record, exact values, and command-line controls are in
[cutwall_current_progress.md](cutwall_current_progress.md).

Work is paused before a production method change. When it resumes:

1. Implement one decomposition-safe radial-interior face flux from the two
   adjacent owner Taylor reconstructions.
2. Include boundary equations from all relevant boundary-containing
   neighbors, including a remote-boundary data path.
3. Use a polynomial-order-aware distance decay and controlled/adaptive support
   instead of a broad inverse-square fit or an arbitrary boundary multiplier.
4. Preserve face-level comparisons of direct, minus-owner, plus-owner, final
   shared, and exact integrated flux.
5. Repeat only `perp_laplacian_phi` at `N=10,14,18` and require monotone
   all-active and wall-category errors.
6. Repair `parallel_density_flux_divergence`, then continue through the regular
   radial operators, projected-exact-phi full RHS, phi solve, decomposed
   equivalence, and the full time-dependent MMS test.

## 17. Literature Search Map

### Completed primary-source review

The review supports the core finite-volume direction but not the present
direct-functional neighborhood policy:

- [Devendran et al. (2017)](https://escholarship.org/uc/item/9b97g2dg)
  demonstrate fourth-order Cartesian embedded-boundary Poisson stencils using
  weighted least squares and examine operator stability.
- [Overton-Katz et al. (2023)](https://arxiv.org/pdf/2209.02840) use
  overdetermined moment-based reconstructions, add boundary-condition
  equations from boundary-containing neighbors, and use an inverse-fifth-power
  distance weight for fourth-order stencils.
- [Thacher, Johansen, and Martin
  (2023)](https://escholarship.org/uc/item/69t7h4bx) use SVD-based local Taylor
  fits with weights `(1 + distance)^-(P+1)` and create the unique conservative
  face flux by averaging the two neighboring polynomial fluxes.
- [Colella and Graves
  (2011)](https://www.osti.gov/biblio/21499787) provide established
  second-order evidence for Cartesian cut-cell elliptic flux matching on
  nontrivial geometries.

The current moment-aware control volumes, integrated flux targets, boundary
equations, and unique conservative face records are therefore well motivated.
The experiments and literature both point to the same next refinement:
localize the support more strongly, include complete neighboring boundary
information, and construct the interior face flux symmetrically from adjacent
reconstructions. The radius-1 rank failure does not mean the coarse grids are
intrinsically unusable; successful high-order methods also use broader
boundary stencils. The issue is allowing enough equations for rank while
making distant observations decay strongly enough that they do not control
the local flux.

The topic map below is retained for future extensions, especially small-cell
time integration and physical sheath conditions.

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
  -> moment-aware cell reconstruction and/or direct face functional
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

The central unresolved mathematical question is whether the direct cubic
integrated functionals, together with the remaining polynomial cell-gradient
operators, deliver the required smooth-field order on translated,
agglomerated, and decomposed geometries. The next convergence runs determine
that empirically. Any further boundary refinement should target the failed
physical functional and its observation coverage, rather than reintroducing an
unconstrained point-gradient patch by default.
