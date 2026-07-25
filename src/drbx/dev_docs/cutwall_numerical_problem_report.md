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
wiring. The accepted perpendicular closure is now the default and only
control-volume perpendicular path: eligible active faces with plus or remote
owners use the symmetric two-owner polynomial flux when both owners pass the
radial-interior rule, including partial faces; eligible cut-wall faces use the
minus/fluid-owner polynomial; and the first two global radial owner layers
retain direct-functional fallback. It preserves the canonical face record and
conservative scatter. The fixed-box N=18,22,26 experiment gives a three-grid all-active
volume-L2 slope of `2.985`, but only `1.269` in Linf; the wall categories are
nonmonotone. Bulk errors are near second order, while changes in cut-wall and
agglomeration topology correlate with oscillatory extrema. That fixed-box
sequence alone does not establish isolated perpendicular convergence. The
later translated-topology campaign below does establish the isolated scalar
gate on a controlled O(h) phase sequence. Exact measurements and the rejected
diagnostics are summarized in
[cutwall_current_progress.md](cutwall_current_progress.md).

The retained path was subsequently extended to `N=40,60,80` with
`shard_counts=(1,1,4)`, where it gives all-active `1.991546/1.962795`, bulk
`1.998658/1.858362`, and remote-interface `1.450711/1.424796` in L2/Linf.
This validates the perpendicular operator and sharding, not projected full
RHS, phi inversion convergence, or full time MMS.

Production perpendicular diffusion, `Ti` reconstruction, and the `phi`
inverse now build and pass owner polynomials; control-volume perpendicular
calls reject a missing polynomial. The former perpendicular CLI switches and
their diagnostic/manual path were removed. Generic direct face functionals
remain because parallel/gradient closures and radial fallback still need them.

## Completed Isolated-Operator Campaign

The earlier fixed-grid results below are retained as history. The completed
campaign added absolute `box_translation=(dx,dtheta,dz)` and the CLI option
`--box-translation DX DTHETA DZ`, together with resolution-scaled
`--box-cell-translation FX FTHETA FZETA`. The effective translation is

```text
box_translation + (FX * dx_cell, FTHETA * dtheta_cell, FZETA * dz_cell).
```

These controls are threaded through geometry, exact and source projection,
convergence, and runtime initialization. Reports now include raw and aggregate
volume ratios relative to the median positive active raw volume, maximum
received source/member counts, and p95/max projected face-weight norms;
`top_error` includes the corresponding volume/source/member metadata.

The fixed-grid topology sequence oscillates and has an N=26 Linf rebound. A
quarter-cell translated sequence improves the fine trend. The cleanest result
is the half-cell `(0,0.5,0.5)` sequence. Because this phase is resolution
scaled, it is an O(h) topology ensemble and not proof of convergence for one
fixed physical geometry.

The half-cell all-active isolated results are:

| Operator | Resolutions | L2 order | Linf order |
| --- | --- | ---: | ---: |
| `perp_laplacian_phi` | N=40,60,80 retained fine path | 1.991546 | 1.962795 |
| `parallel_density_flux_divergence` | N=18,22,26 | 1.899879 | 1.928475 |
| `poisson_omega` | N=18,22,26 | 2.010197 | 2.236330 |
| `poisson_v_electron` | N=18,22,26 | 2.254952 | 3.144134 |
| `curvature_phi` | N=18,22,26 | 1.843095 | 2.226615 |
| `grad_parallel_v_electron` | N=26,30,34 fine window | 1.812328 | 1.982034 |
| `grad_parallel_phi` | N=26,30,34 fine window | 1.811031 | 1.939895 |

The parallel-density conservative signed sum is zero and invalid quadrature is
zero. `grad_parallel_density`, `grad_parallel_v_ion`, `poisson_density`,
`poisson_v_ion`, and `curvature_density` are zero-target exact/roundoff cases.
The direct regular-grid gradient baseline on N=18,22,26 was approximately
second order (`~1.99`).

The projected-exact-phi full-RHS result is not yet a clean pass:

| Component | L2 order | Linf order | N=18/N=22/N=26 L2 errors |
| --- | ---: | ---: | --- |
| density | 1.877320 | 1.955059 | `.02091865/.01441512/.01048588` |
| omega | 1.567201 | 1.482046 | `.9997461/.7420204/.5612542` |
| v_ion | exact | exact | zero-target exact cancellation |
| v_electron | 1.602884 | 1.787992 | `11.48058/8.426111/6.362756` |

Density passes the 1.8 gate. Omega misses, and electron misses clearly in L2
and narrowly in Linf. At the tested `t=0` stage, the exact omega and electron
time derivatives are zero, so these components deliberately test cancellation
between the discrete operator sum and the projected source. The electron
residual is amplified by `mi_over_me=1836` multiplying
`grad_parallel_phi`. Its adjacent L2 orders improve from `1.541` to `1.681`;
omega improves from `1.486` to `1.671`. These trends are consistent with the
delayed fine-window asymptotics of the isolated parallel gradients. They point
first to pre-asymptotic, coefficient-amplified cancellation, not a newly
identified production discretization error.

The stop/go decision is to accept the isolated scalar stage on this controlled
translated sequence but pause phi inversion and time convergence. The next
bounded diagnostic should report each full-RHS component and its cancellation:
omega (Poisson, parallel-gradient difference, curvature), electron (Poisson,
phi parallel gradient, pressure gradient), and the comparison between the sum
of independently projected analytic terms and projection of their pointwise
sum. A full-RHS-only path should then repeat the N=26,30,34 fine window.
Less cancellation-sensitive stage times or MMS amplitudes/fields are useful as
secondary conditioning checks before code fixes, but do not replace the
zero-target consistency test.

All reported runs used one shard; real multi-device validation remains
unavailable. N=34 geometry preprocessing takes approximately 330 s, so
caching/reuse is a priority. No invalid reconstruction rows occurred.

## Resumed Status Through N=26

The retained one-shard `perp_laplacian_phi` configuration uses agglomeration,
projected exact phi, the decomposition-safe two-owner projected flux,
cut-wall owner-polynomial flux, `1/d^4` reconstruction row weighting, and
boundary equation scale 10. The fine-grid geometry reports were:

| Quantity | `N=22` | `N=26` |
|---|---:|---:|
| Active aggregate owners | 10364 | 17162 |
| Merged sources | 44 | 54 |
| Aggregate targets | 44 | 54 |
| Irregular faces | 3794 | 5045 |
| Interior/partial/cut-wall faces | 2390/860/544 | 3145/1176/724 |
| Cubic reconstruction rows | 2412 | 3168 |
| Maximum reported condition number | 32.50649 | 34.98056 |
| Invalid reconstruction rows | 0 | 0 |

The exact N=22/N=26 error record is:

| Resolution | Category | Volume L2 | Linf |
|---:|---|---:|---:|
| 22 | all active | 0.04013236 | 0.5914156 |
| 22 | bulk | 0.03794186 | 0.3317934 |
| 22 | one wall | 0.08846575 | 0.5914156 |
| 22 | multi-wall | 0.05378290 | 0.2835862 |
| 22 | aggregate target | 0.07214926 | 0.1775413 |
| 22 | retained cut cell | 0.08196992 | 0.5914156 |
| 26 | all active | 0.03428972 | 1.301826 |
| 26 | bulk | 0.02732468 | 0.2429065 |
| 26 | one wall | 0.1352501 | 1.301826 |
| 26 | multi-wall | 0.09147248 | 0.5119297 |
| 26 | aggregate target | 0.2597501 | 1.301826 |
| 26 | retained cut cell | 0.084212 | 0.8028512 |

The adjacent N=22 to N=26 orders are `0.942/-4.723` for all-active L2/Linf
and `1.965/1.867` for bulk L2/Linf. The three-grid N=18,22,26 all-active
orders are `2.985/1.269`. These results do not support an isolated convergence
claim: the bulk is near second order, but the cut-wall and aggregate-target
categories are nonmonotone and the all-active Linf worsens on the finest pair.

At N=18, audited reconstruction rows had rank 19 with condition numbers
`14.4384` and `24.8176`. Their dimensionless first-derivative coefficient
sizes were O(1) after the expected grid-spacing derivative scaling, so the
audit found no evidence of rank loss or pathological coefficient amplification.

Four local diagnostics were tested and reverted: neighborhood-complete wall
observations; a nearest-48 direct-functional target with adaptive expansion;
reducing the radial owner-flux guard from 2 to 1; and choosing a farther
interior owner polynomial. Their exact N=10 results and failure modes are in
the companion progress log. The committed code remains `cf1d2ad7`; none of
these experiments should be treated as production changes.

## Fine-Grid Perpendicular Confirmation: N=40,60,80

The complete retained perpendicular configuration was run on the controlled
half-cell sequence through `N=80` with `shard_counts=(1,1,4)`: agglomeration,
the consolidated owner-polynomial closure, `1/d^4` reconstruction-row
localization, and boundary equation scale 10 were all enabled. The result is:

| Resolution | All-active L2 | All-active Linf | Bulk L2 | Bulk Linf |
|---:|---:|---:|---:|---:|
| 40 | `1.242690e-2` | `1.995061e-1` | `1.173657e-2` | `1.083002e-1` |
| 60 | `5.689690e-3` | `1.902729e-1` | `5.216219e-3` | `5.068009e-2` |
| 80 | `3.115134e-3` | `4.678020e-2` | `2.934457e-3` | `2.988912e-2` |

The fitted all-active orders are `1.991546/1.962795` in volume L2/Linf.
Bulk orders are `1.998658/1.858362`, one-wall orders are
`1.963498/2.054307`, and multi-wall orders are `2.530420/2.354154`.
Every reconstruction row remained cubic and valid. Maximum reconstruction
condition numbers were `28.83/39.04/28.74`, and maximum normalized projected
functional weight norms were `3.34/3.93/4.73`.

This is strong evidence for the retained perpendicular operator path: one
canonical conservative interior flux averaged from the two adjacent owner
polynomials on eligible radial-interior active faces (including partial
faces), a fluid-owner polynomial flux on eligible embedded cut-wall faces,
direct-functional fallback on the first two global radial owner layers, and
localized owner reconstruction. It clears the requested all-active `1.8`
fine-grid gate and validates sharding for this isolated operator.

There are four qualifications. First, all-active adjacent Linf orders are
`0.117` and `4.877`; the three-grid fit is near second order because the
finest maximum drops sharply. Second, agglomeration occurs only at `N=60`
(`360` sources), so the multi-wall and aggregate categories do not describe
one smoothly varying topology. Third, the reconstruction-row L2 category fits
only `1.499804`. Fourth, this result does not validate the projected full RHS,
phi inversion convergence, or full time MMS. The remote-interface orders are
L2 `1.450711` and Linf `1.424796`.

The completed output reports geometry elapsed times of `68.397 s`, `156.276 s`,
and `330.598 s` for `N=40,60,80`. Local-bundle payload/bundle totals were
`2.673/27.295 s`, `9.339/63.001 s`, and `22.931/139.311 s`.

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
| Aggregate ownership | Global, direct, idempotent; nonzero merges demonstrated through `N=26` | Topology changes between resolutions and affects the worst aggregate |
| Aggregate geometry | Volume, centroid, `M2`, and `M3` available | Validate cut-volume quadrature order |
| Cubic reconstruction | Required on every shifted-torus active row; no fallback in the tested grids | One-wall owners have many cell equations but few boundary equations |
| Direct compact functionals | Projected, parallel-value, and parallel-gradient wired and audited face by face | Broad, weakly localized fits produce inaccurate individual fluxes despite full rank |
| Dense/compact ownership | Exclusive global face paths established | Confirm cluster multi-shard equivalence |
| Cross-shard residuals | Reverse face-halo accumulation implemented; isolated perpendicular operator convergence completed with `shard_counts=(1,1,4)` | Full-RHS sharding and one-vs-four equivalence remain unvalidated |
| Geometry preprocessing | Batched coordinate-face quadrature, compact functional rows, reconstruction fallback, and canonical-only decomposed local-bundle lowering; aggregate moment construction is no longer quadratic; optional face-functional NPZ cache; resolution payloads are explicitly released between sweep grids | The downstream `N=100` run still appears to exceed the 14-GiB host envelope |
| Parallel density flux | Half-cell isolated all-active order `1.899879/1.928475`; signed sum and invalid quadrature are zero | Full-RHS coupling still requires component cancellation audit |
| Perpendicular Laplacian | Retained half-cell path at `N=40,60,80` with `shard_counts=(1,1,4)` gives all-active `1.991546/1.962795`, bulk `1.998658/1.858362`, and remote-interface `1.450711/1.424796`, with no invalid rows | Adjacent Linf and agglomeration topology are nonmonotone; this validates the isolated operator and sharding only |
| Isolated scalar operators | All advertised nondegenerate scalar components meet the 1.8 all-active gate on the controlled sequence | Full-RHS coupling and broader sharding coverage remain |
| Projected full RHS | Density passes at `1.877320/1.955059`; ion is exact | Omega `1.567201/1.482046` and electron `1.602884/1.787992` miss |
| Regular radial closure | Cubic reproduction passes | Lower-plane Poisson Linf remains |
| Phi GMRES solve | Implemented but intentionally paused | Re-enable after full-RHS cancellation is understood |
| Full RK/MMS convergence | Not ready | Do not start until projected full RHS passes |

## 16. Immediate Validation Sequence

The fine half-cell result establishes the retained perpendicular formulation
as the production default: localized owner reconstruction followed by one
shared conservative owner-polynomial face flux. Repeat one additional
controlled wall phase before broader acceptance.

The next steps are:

1. run one affordable alternate wall-phase regression;
2. lock radial, parallel, and gradient operator regressions;
3. rerun projected-exact-phi full RHS and isolate omega/electron cancellation
   and order failures;
4. establish one-vs-four full-RHS equivalence;
5. then measure phi-inverse convergence;
6. only then run full four-field time MMS.

Only after these steps should the accepted perpendicular closure be routed
through the phi inverse and into the full time-dependent MMS test.

## 16.1 Sharding Handoff: Completed Infrastructure and Evidence

The earlier one-shard-only qualification is now superseded for the isolated
perpendicular operator. The shifted-torus retained path has been executed
with a real `1x1x4` decomposition. This validates sharding of the isolated
`perp_laplacian_phi` operator only; it does not validate the full four-field
RHS, the algebraic phi solve, or full time-dependent MMS convergence.

The completed sharding path provides boundary-observation source-shard
metadata with global all-gather, canonical whole-domain face lowering,
canonical whole-domain reconstruction lowering, and canonical dense/compact
face and reconstruction-target masks. Decomposed radius-2 reconstruction
support requires `halo_width >= 3`. Owner aggregation and owner-field
expansion are production distributed operations, including remote source
averages and halo exchange, and remote residual contributions are reverse
scattered to their canonical aggregate owners.

Two bugs found during decomposition equivalence testing are fixed. A local
periodic-support candidate could wrongly close a valid dense face; canonical
global dense/compact masks now prevent that. Separately, a same-ID local
compact row was retained instead of replaced by the canonical whole-domain
row. Face `10488` exposed the error: the local row targeted `(8,13,4)` while
the canonical row targeted `(8,14,4)`.

The explicit N=16 diagnostic gives operator maximum difference
`4.440892098500626e-15` and compact signed-sum difference `6.94e-18` between
one shard and `1x1x4`. The retained runs have identical counts, error
summaries, and top-error record, with zero invalid reconstruction rows. Only
the decomposed reporting adds `remote_interface` count `134`, which records
remote participation and does not represent additional physical faces.

The isolated sharding handoff completed the 40/60/80 operator sweep. The
reproducible command was:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
.venv/bin/python tests/test_fci_cutwall_shifted_torus_4field.py \
  --operator-convergence-only \
  --resolutions 40 60 80 \
  --shard-counts 1 1 4 \
  --halo-width 3 \
  --operators perp_laplacian_phi \
  --enable-agglomeration \
  --box-cell-translation 0 0.5 0.5 \
  --reconstruction-distance-row-weight-exponent 4 \
  --reconstruction-boundary-weight-scale 10 \
  --minimum-order 1.8 \
  --geometry-cache-dir ~/.cache/drbx/cutwall_geometry \
  --skip-runtime-info \
  2>&1 | tee ~/shifted_torus_perp_laplacian_n40_n60_n80_shards_1x1x4.txt
```

The output is
`/home/exouser/shifted_torus_perp_laplacian_n40_n60_n80_shards_1x1x4.txt`.
It reports all-active errors `1.242690e-02`, `5.689690e-03`, and
`3.115134e-03`, final orders L2 `1.991546` and Linf `1.962795`, bulk orders
L2 `1.998658` and Linf `1.858362`, and remote-interface orders L2 `1.450711`
and Linf `1.424796`. This validates the perpendicular operator and sharding,
not projected full RHS, phi inversion convergence, or full time MMS.

## 16.2 Sharded preprocessing optimization

The decomposed local-bundle path now directly lowers the canonical whole-domain
reconstruction instead of calling per-shard
`precompute_local_moment_reconstruction`. Remote reconstruction sample-cloud
construction was removed as part of the same path. Legacy and one-shard
fitting remains unchanged, and local face discovery still remains in the
decomposed path. Thus the optimization removes the discarded per-shard SVD
and sample-cloud work without changing the canonical whole-domain geometry or
the one-shard fitting behavior.

The numeric-only face-functional cache covers only the global compiled face
functionals. It does not cache all geometry or topology. The redundant local
SVD work is now gone from decomposed local-bundle preparation despite that
cache scope.

Independent cache-hit N=40 `1x1x4` timing:

```text
global phase:                 23.028 s
local geometry:                1.745 s
per-shard payload total:       2.667 s
bundle total:                 26.722 s
local bundle phase:           29.390 s
padding/stacking:              9.134 s
total prepared geometry:      67.156 s
operator compile+run:          2.028 s
invalid reconstruction rows:       0
```

The optimization agent's earlier run measured `30.026 s` for local bundle
lowering; the independent run measured `29.390 s`. This is a reduction from
the previous `584.190 s` phase by approximately `19.9x`. The N=16 one-shard
versus `1x1x4` operator difference remains `4.44e-15`.

These measurements establish the preprocessing optimization and complement
the completed 40/60/80 fine-grid sharded convergence result above.

The previous `N=34` preprocessing measurement was roughly `330 s`. The
geometry path has since been optimized and must be remeasured: an old sweep
spent `845.624 s` in the `N=80` raw-topology phase, while the optimized
isolated `N=80` topology build takes `29.41 s` including process startup and
uses approximately `1.9 GB` peak resident memory. The improvement comes from
batched coordinate-face clipping/quadrature and removal of a quadratic
aggregate-ID rescan. Global compact functionals and fallback reconstruction
rows are also batched. An optional numeric-only face-functional cache is
available through `--geometry-cache-dir`; it does not bypass raw topology.
Any sweep process started before these edits must be restarted to load them.

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
The experiments and literature still point to localized support, complete
neighboring boundary information, and one symmetric interior-face flux. The
last item is now the default perpendicular control-volume path. The
exponent-4 experiment confirms that stronger
localization can repair global L2 behavior, but also shows that distance decay
alone does not cure every one-sided wall reconstruction or
regular-boundary-adjacent direct functional. The radius-1 rank failure does
not mean the coarse grids are intrinsically unusable; successful high-order
methods also use broader boundary stencils. The issue is allowing enough
equations for rank while selecting and weighting them so distant observations
do not control the local flux.

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
