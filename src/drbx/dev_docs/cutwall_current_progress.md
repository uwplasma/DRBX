# Cut-Wall Current Progress, Observations, and Next Steps

## Status Snapshot

This document records the current implementation and validation status of the
agglomerated embedded-control-volume path. It is the time-sensitive companion
to the durable infrastructure documents:

- [cutwall_agglomeration_ls_call_chain.md](cutwall_agglomeration_ls_call_chain.md)
  describes the implemented runtime and geometry call chain.
- [embedded_control_volume_cell_cases.md](embedded_control_volume_cell_cases.md)
  defines the supported cell and face cases.
- [fci_main_integration_compatibility.md](fci_main_integration_compatibility.md)
  records integration constraints with the main FCI solver.
- [cutwall_numerical_problem_report.md](cutwall_numerical_problem_report.md)
  explains the numerical problem and the design rationale.

Snapshot date: 2026-07-23.

Repository baseline at the time of this snapshot: `0caa74b8`, plus the current
worktree changes summarized below.

## Executive Summary

The structural migration to agglomerated finite volumes and direct
moment-fitted compact-face fluxes is substantially complete. The current
one-shard shifted-torus fixture has:

- physical aggregate-volume-average unknowns;
- aggregate volume, centroid, second moment, and third moment;
- direct owner maps for merged sources;
- one canonical record for each compact physical face;
- cubic moment-aware cell reconstruction on every required irregular owner;
- direct cubic functionals for integrated projected, parallel-value, and
  parallel-gradient compact-face fluxes;
- conservative face scatter into aggregate residuals;
- projected exact `phi` support for isolating spatial operators;
- fixed-shape JAX data and sharding-compatible lowering.

The first agglomeration-enabled `N=10,14` operator sweep completed after
adding per-operator executable and device-output cleanup. It did not satisfy
the requested minimum order of 1.8.

Two coarse resolutions are not enough to identify a trustworthy asymptotic
order, especially because the embedded box intersects a different set of
logical cells and cut fractions at each resolution. Nevertheless, the sweep
contains a strong localized warning: the perpendicular Laplacian maximum
error grows at multi-wall aggregate targets while nearby dense cells converge
near second order. That behavior should be diagnosed before simply running a
larger full sweep.

All five Phase A items are now implemented and checked in the current worktree:
geometry-phase timing, one-shard unsplit-bundle reuse, reuse of the generated
global functional records, inactive-centroid metric sanitation, and targeted
operator selection. Focused `N=6` coverage verifies that the one-shard result
is exactly the captured unsplit bundle and that global functional compilation
occurs once; the sanitation test covers inactive and nonfinite centroid
placeholders. The one-shard `N=6` build took `18.1 s`, versus `23.9 s` for the
decomposed build.

The resulting one-shard preprocessing times are:

```text
N=10: 46.0 s  (previously 92.514 s)
N=14: 73.5 s  (previously 144.647 s)
```

This materially improves iteration time without changing the numerical
method. The remaining geometry cost is dominated by global direct-functional
construction, not duplicate one-shard lowering.

Follow-up diagnostics have now isolated the perpendicular failure more
precisely. Disabling agglomeration does not restore convergence, exact
manufactured product averages do not repair the parallel face fluxes, and the
compact signed sums close correctly. The main defect is the accuracy and
locality of the fitted face flux itself.

The strongest experimental improvement came from constructing one
conservative interior-face flux by averaging the two adjacent owner-polynomial
fluxes, while using the owner polynomial on embedded cut-wall faces. That
diagnostic has now been moved into a default-neutral native helper that uses
the already exchanged remote-owner polynomial gradient on cross-shard faces.
The canonical face record and equal-and-opposite scatter are unchanged.
Focused eager and JIT tests cover local, remote, cut-wall, invalid-row, and
radial-boundary cases. Actual multi-device execution remains untested because
this host exposes only one JAX CPU device.

The first three-grid shared-flux result still failed: its all-active fitted
volume-L2 order was `1.575`, with a topology-dependent one-wall aggregate
failure at `N=18`. A second experiment added stronger dimensionless distance
decay to selected compact owner reconstructions and excluded the first two
global radial owner layers from both that weighting and the owner-flux
replacement. With a `1/d^4` WLS row multiplier and boundary scale 10, the
all-active errors became:

```text
N=10: volume L2 = 4.388113e-1, Linf = 3.511981
N=14: volume L2 = 2.153701e-1, Linf = 2.643948
N=18: volume L2 = 1.007078e-1, Linf = 1.956872
```

The fitted all-active orders are `2.483` in volume L2 and `0.987` in Linf.
This is a meaningful global-L2 improvement, but it is not an accepted
isolated convergence result: one-wall and aggregate-target errors reverse
between `N=14` and `N=18`, and the finest-grid maximum is a non-wall compact
reconstruction owner. The parallel-density flux also remains unresolved.

Work is paused at this diagnostic checkpoint. All alternative flux and weight
paths added in this phase are opt-in and default-neutral.

## 1. Latest Validation Configuration

The latest completed isolated perpendicular sweep used:

```text
resolutions                 N=10,14,18
shard counts                1,1,1
agglomeration               enabled
operator-only mode          enabled
phi algebraic solve         skipped
selected operator           perp_laplacian_phi
two-owner projected flux    enabled
cut-wall owner flux         enabled
distance row exponent       4
boundary equation scale     10
minimum requested order     1.8
```

The earlier all-operator `N=10,14` sweep used projected exact phi for the full
RHS. Disabling CUDA command buffers was initially necessary because that sweep
exhausted accelerator memory while instantiating a later compiled operator.
The convergence harness now also:

- converts completed operator outputs to NumPy host arrays;
- explicitly deletes the compiled scalar kernel and its device outputs;
- calls `jax.clear_caches()` between scalar operators;
- runs Python garbage collection between scalar operators.

These changes allowed all scalar operators, the projected-exact-phi full RHS,
and both resolutions to complete.

## 2. Geometry and Reconstruction Results

The latest geometry summary was:

| Quantity | `N=10` | `N=14` | `N=18` |
|---|---:|---:|---:|
| Active aggregate owners | 952 | 2648 | 5636 |
| Merged sources | 48 | 96 | 36 |
| Aggregate targets | 48 | 96 | 36 |
| Irregular faces | 898 | 1522 | 2860 |
| Interior compact faces | 622 | 1058 | 1812 |
| Partial faces | 140 | 252 | 632 |
| Cut-wall faces | 136 | 212 | 416 |
| Cubic reconstruction rows | 566 | 956 | 1844 |
| Quadratic fallbacks | 0 | 0 | 0 |
| Linear fallbacks | 0 | 0 | 0 |
| Maximum reported condition number | `5.336e4` | `1.404e5` | `3.085e5` |

Positive conclusions:

- Agglomeration is active at all three resolutions.
- Every required reconstruction row remains cubic.
- No active reconstruction row is invalid.
- No runtime reconstruction fallback is being used to hide a rank failure.
- The scalar operator kernels report zero invalid reconstruction rows.

Items to monitor:

- The maximum condition number increases by approximately 2.6 between the two
  geometries.
- The current summary does not report the worst direct functional weight norm,
  reproduction residual, face ID, aggregate volume fraction, or normal
  coverage.
- The one-shard sweep contains no remote interface owners and therefore does
  not validate the cross-shard compact-face path.

## 3. Interpretation Limits of the Current Orders

An order calculated from only `N=10` and `N=14` is a two-point slope:

```text
p = log(error_10 / error_14) / log(14 / 10).
```

The denominator is small, so modest nonmonotonic changes produce a large
change in the reported order. Embedded geometry adds further variation:

- different cut fractions occur at each resolution;
- a different number of sources are agglomerated;
- the identity of the worst cell can change;
- the number of multi-wall cells changes;
- Linf is sensitive to one unusually shaped aggregate.

The present slope is therefore diagnostic, not a final convergence claim.
Three or more resolutions are required after the build cost and targeted
operator diagnostics are improved.

Several operators are reported as exact because the current MMS makes their
continuum target zero to floating-point precision. These include the tested
density and ion-parallel cases for several first-derivative, bracket, and
curvature functionals. Such results check algebraic cancellation but do not
provide a meaningful nonzero convergence test for those operators.

## 4. Current Operator Observations

The most useful all-active orders from `N=10` to `N=14` are:

| Operator | Volume L2 order | Linf order | Current interpretation |
|---|---:|---:|---|
| `grad_parallel_phi` | 1.770 | 1.490 | Plausibly pre-asymptotic and close to target |
| `grad_parallel_v_electron` | 1.499 | 0.006 | L2 improves; maximum radial-boundary error is flat |
| `parallel_density_flux_divergence` | 0.790 | -0.361 | Compact multi-wall maximum error grows |
| `poisson_omega` | 1.219 | 0.960 | Improves slowly; lower radial boundary dominates Linf |
| `poisson_v_electron` | -0.278 | -2.037 | Error grows, especially at the lower radial boundary |
| `curvature_phi` | 1.903 | 0.061 | L2 is promising; one radial-boundary maximum is flat |
| `perp_laplacian_phi` | 0.831 | -0.852 | Dense region improves; multi-wall aggregate error grows |
| `full_rhs_density` | 0.965 | 0.385 | Inherits compact/wall operator errors |
| `full_rhs_omega` | 0.945 | -0.803 | Linf grows |
| `full_rhs_v_electron_parallel` | 1.604 | 1.357 | Improving, but not yet at target |

The full-RHS ion-parallel result is at floating-point roundoff. Source
round-trip diagnostics also confirm that the ion source projection is
consistent. This is useful plumbing evidence but not a nonzero spatial-order
test.

## 5. Strongest Localized Numerical Signal

The clearest remaining embedded-wall failure is
`perp_laplacian_phi`.

All-active errors:

```text
N=10: volume L2 = 5.416e-1, Linf = 4.690
N=14: volume L2 = 4.094e-1, Linf = 6.246
```

The dense region behaves much better:

```text
bulk L2 order              1.708
dense compact distance 1   2.072
dense compact distance 2   2.509
dense far                  2.374
```

The failing categories are:

```text
one-wall L2 order          -0.839
multi-wall L2 order        -0.487
aggregate-target L2 order  -0.054
multi-wall Linf order      -0.852
aggregate-target Linf      -0.852
```

At both resolutions, the largest error is located at an aggregate target
with:

```text
embedded cut-wall faces    3
irregular faces            12
reconstruction rows        1
regular radial boundary    false
```

This localization is important. The regular radial boundaries converge well
for the perpendicular Laplacian, while the multi-face embedded aggregate does
not. The immediate suspect is therefore not the dense perpendicular operator.
It is the compact projected-flux/divergence closure or its use on a
multi-member, multi-wall aggregate.

`parallel_density_flux_divergence` shows a related pattern. Its `N=14` worst
cell is also an aggregate target with three cut-wall faces and twelve
irregular faces. This suggests that the common compact face gather, face
ownership, or aggregate divergence path should be inspected before treating
the two operator failures as unrelated.

## 6. Suspected Numerical Problems

The following are hypotheses to test, not established causes.

### 6.0 Face-audit result: the direct functional is the first bad stage

The targeted `N=10,14` audit is recorded in
`shifted_torus_targeted_face_audit_n10_n14.txt`. It selected only
`parallel_density_flux_divergence` and `perp_laplacian_phi`; their all-active
volume-L2/Linf two-point orders were respectively `0.788/-0.361` and
`0.847/-0.852`.

The original process loaded before the audit target was restricted from the
global worst cell to the worst aggregate target. Its corrected `N=10`
parallel aggregate audit is preserved separately in
`shifted_torus_parallel_face_audit_n10_corrected.txt`.

For each worst aggregate, the numerical compact signed sum equals the actual
integrated residual to machine precision. The exact compact sum differs from
the independently projected reference only by the reported dense remainder.
Thus face ownership, scatter signs, aggregate-volume division, and the MMS
control-volume reference are not the first defect.

The first bad quantity is the individual direct-functional flux. The dominant
perpendicular failures are x-normal interior and cut-wall functional fluxes
with incorrect sign or magnitude; their signed contributions drive the
multi-wall aggregate error. The parallel-density flux also has large
tangential face errors whose cancellation is wrong. These rows have full
cubic rank and small reproduction residuals, so polynomial reproduction alone
does not establish physical flux accuracy.

### 6.1 Multi-face aggregate divergence

A direct face functional may reproduce its target polynomial correctly while
the final aggregate divergence is still wrong because:

- one or more physical faces are missing, duplicated, or oriented
  inconsistently;
- compact and dense face contributions overlap;
- a face flux is scattered to the wrong aggregate owner;
- a merged source or aggregate volume is used inconsistently;
- several individually large face errors fail to cancel at the aggregate
  level;
- the MMS reference is projected over a different control volume.

The completed worst-aggregate audit now clears this class as the first defect
for the two targeted operators: numerical compact sums close to their actual
integrated residuals, and exact compact sums close to the MMS reference. Keep
these checks as invariants while repairing the individual functional fluxes.

### 6.2 Functional conditioning and coefficient amplification

All direct rows are algebraically valid, but rank alone is insufficient.
The next diagnostic must record:

- polynomial order and rank;
- scaled condition number;
- reproduction residual;
- normalized projected-flux weight norm;
- maximum absolute normalized coefficient;
- aggregate volume fraction;
- face area fraction;
- number and type of Dirichlet observations.

The maximum reconstruction condition number rises from `5.336e4` to
`1.404e5`. The direct functional condition and weight norms may identify a
small set of geometrically weak faces even when polynomial reproduction
passes.

### 6.3 Boundary observation coverage

Global direct functional records restrict boundary observations to wall rows
owned by the evaluator aggregate. A multi-wall aggregate may therefore have a
different balance of volume-average and boundary equations than intended.
Verify that every relevant wall patch contributes the correct Dirichlet
quadrature data exactly once.

### 6.4 Regular radial-boundary closure

Several non-perpendicular operators have their largest error at the regular
lower radial boundary:

- `grad_parallel_v_electron`;
- `poisson_omega`;
- `poisson_v_electron`;
- `curvature_phi`.

This is separate from the multi-wall perpendicular-Laplacian signal. The
regular radial moment closure, ghost closure, and reference projection should
be tested independently so a regular-boundary defect is not attributed to the
embedded cut wall.

### 6.5 MMS coverage

Zero-target manufactured fields leave several operator paths untested. Once
the current nonzero failures are understood, add rotated/nontrivial fields
whose:

- parallel derivative is nonzero;
- Poisson bracket is nonzero;
- curvature drive is nonzero;
- wall trace varies tangentially;
- projected normal flux is nonzero on oblique and multi-wall faces.

## 7. Geometry Build-Time Status

Previous measured preprocessing time:

```text
N=10:  92.514 s for  898 irregular faces and 566 reconstruction rows
N=14: 144.647 s for 1522 irregular faces and 956 reconstruction rows
```

The growth broadly follows the irregular wall band rather than the full cell
count, which is desirable. The absolute constant is not acceptable for an
iterative development test.

The previous `_build_stacked_embedded_control_volume_geometry` path performed:

1. Global raw moments, face measures, and aggregate topology.
2. A complete unsplit global embedded bundle.
3. Internal compilation of global cubic face-functional records while
   building that bundle.
4. A second explicit compilation of those global records.
5. Construction of equivalent local geometry and cell data.
6. Another complete local embedded bundle.
7. Padding and stacking of the local bundles.

For `shard_counts=(1,1,1)`, the unsplit global bundle and the only local bundle
describe the same partition. The current path directly stacks the unsplit
bundle and carries the generated functional-record dictionary forward, which
removes both duplicates.

The global topology builder also loops in Python over every coordinate face,
constructs its open rectangles, evaluates quadrature, and evaluates the
shifted-torus metric separately. That path should eventually be batched.

### Remaining build optimization priorities

1. Batch the coordinate-face measure calculation.
2. Add an optional persistent geometry cache keyed by all geometry-affecting
   inputs after the build path is deterministic and validated.

## 8. Known Diagnostic Noise

### Inactive centroid warnings

The geometry builder evaluates centroid metric data over the full storage
array. Inactive solid and merged-source slots contain the placeholder
centroid `(0,0,0)`, which produces divide-by-zero warnings even though the
physical radial domain begins at `x_min=0.2`.

This is fixed: inactive or nonfinite centroid slots are replaced by an
in-domain reference coordinate before metric and curvature evaluation. The
replacement is masked from active-owner physics and is unit-tested.

### Empty categories

In the one-shard run, `remote_interface` has count zero. Its printed L2 and
Linf values are `nan` because the category is empty, not because an active
operator result is invalid.

### Relative errors for zero references

When the exact reference norm is zero, machine-precision absolute errors are
printed with enormous relative errors. Acceptance and diagnosis must use the
absolute error or mark the result exact/degenerate.

## 9. Phase Status

### Phase A: make iteration affordable

Completed: geometry phase timings, the one-shard bundle-reuse fast path,
single global functional compilation, inactive-centroid sanitation, and the
targeted `--operators` CLI option. Focused `N=6` tests cover the first four;
the operator-selection test validates accepted and rejected names without a
geometry build.

### Phase B: diagnose direct-functional accuracy

Completed far enough to identify the next design question:

- exact compact sums and the MMS reference agree, clearing conservative
  scatter, aggregate-volume division, and reference projection as the first
  defect;
- disabling agglomeration does not restore convergence;
- exact analytic product averages do not repair the parallel flux;
- bad direct functionals can draw on more than 100 cell averages while
  receiving few or no wall equations;
- symmetric two-owner polynomial fluxes materially improve the perpendicular
  operator, but the current diagnostic implementation is not robust at every
  regular radial boundary or every one-wall aggregate.

The decomposition-safe symmetric physical-face flux is now implemented as an
opt-in native helper. The remaining method question is controlled
reconstruction locality and boundary accuracy. A global boundary multiplier
alone does not fix the topology-sensitive rows, and a strong distance law
improves global L2 while leaving category and Linf failures.

### Phase C: isolate regular radial-boundary defects

Still pending. Run only:

```text
grad_parallel_v_electron
poisson_omega
poisson_v_electron
curvature_phi
```

Compare radial-boundary owners with radial-interior owners and test the
regular radial closure independently of the embedded box.

### Phase D: establish convergence

Two three-resolution perpendicular diagnostics have been run, but neither
passes the acceptance gate. When work resumes:

1. repeat `N=10,14,18` for `perp_laplacian_phi`;
2. require monotone all-active, one-wall, multi-wall, aggregate-target, and
   retained-cut-cell reduction before accepting a fitted order;
3. repair and repeat `parallel_density_flux_divergence`;
4. run the projected-exact-phi full RHS only after isolated operators pass;
5. re-enable the algebraic phi solve;
6. run a decomposed case to validate remote compact faces and reverse
   residual accumulation;
7. finally run the time-dependent four-field shifted-torus MMS test.

## 10. Acceptance Gates

The cut-wall operator path is ready for the final four-field convergence test
when:

- every required compact row remains valid and cubic;
- polynomial reproduction residuals are near floating-point tolerance;
- normalized direct-functional weights remain within an explicit bound;
- one physical face maps to one canonical flux record;
- shared interior fluxes cancel conservatively;
- multi-wall aggregate errors decrease monotonically;
- three or more resolutions support at least the requested 1.8 all-active
  volume-L2 and Linf order for each nondegenerate operator;
- regular radial-boundary and embedded-wall failures are independently
  resolved;
- the projected-exact-phi full RHS passes before the phi inversion is added;
- the decomposed result agrees with the one-shard result;
- geometry preprocessing is fast enough to make repeated validation
  practical, or a validated persistent cache is available.

## 11. Follow-up Direct-Functional Experiments

### 11.1 Controls that ruled out earlier hypotheses

Agglomeration-disabled `N=10,14` tests remained nonconvergent:

| Operator | All-active L2 order | All-active Linf order | Multi-wall L2 order |
|---|---:|---:|---:|
| Parallel density flux divergence | `0.867` | `0.041` | `-0.279` |
| Perpendicular Laplacian | `0.705` | `-0.474` | `-0.284` |

The defect is therefore intrinsic to the compact closure rather than created
only by merged-cell ownership.

For the bad parallel faces, the runtime covariance-corrected product and the
exact manufactured control-volume product average produced the same fitted
flux to the shown precision. Examples are:

```text
face 1711: fitted 7.665046e-05, exact flux 1.820220e-03
face 1806: fitted 5.280481e-04, exact flux 2.435709e-03
```

The product-average input is not the first parallel-density defect; the face
functional maps accurate input averages to an inaccurate flux.

### 11.2 Observation coverage and locality

Representative rows expose a large support and weak boundary influence:

| Face | Role | Cell equations | Dirichlet equations | Cell-weight L1 | Dirichlet-weight L1 |
|---|---|---:|---:|---:|---:|
| `1711` | parallel interior | 150 | 0 | not recorded | `0` |
| `1806` | parallel interior | 134 | 0 | not recorded | `0` |
| `284` | perpendicular interior x-face | 105 | 0 | `10.77` | `0` |
| `-20000800006` | perpendicular cut-wall x-face | 105 | 12 | `6.381` | `1.166` |

For face `284`, the direct fitted flux was `8.90e-2` versus the exact
`1.656e-2`. For cut-wall face `-20000800006`, it was `-6.42e-2` versus the
exact `+1.234e-2`.

Reducing the cell radius from 2 to 1 made the `N=10` cubic face system rank
deficient (`18/20`). Thus the present cubic basis genuinely needs more than a
radius-1 support on this coarse geometry. The concern is not that radius 2 is
automatically invalid; it is that a necessary broad candidate neighborhood is
only weakly localized and can draw appreciable weight from physically distant
cells.

Increasing the direct-functional boundary equation weight by 10 did not fix
the method. The perpendicular all-active error changed from
`0.54449/4.690` to `0.66554/5.339` in volume L2/Linf. Including wall
observations from all local compact-face owners improved the multi-wall error
from `1.466/4.690` to `0.878/2.693`, but worsened the all-active result to
`0.57113/6.909`. This confirms that missing neighboring-owner boundary data is
real, but not sufficient by itself.

### 11.3 Owner-polynomial and symmetric-face diagnostics

At the `N=10` worst aggregate face `284`:

```text
direct functional          +0.0890
minus-owner polynomial     +0.1563
plus-owner polynomial      -0.1269
two-owner average          +0.0147
exact                      +0.01656
```

Neither one-sided polynomial was reliable alone, but their average nearly
recovered the exact physical face flux. Applying two-owner averaging at every
face was not viable: regular radial-boundary errors grew above `1e2`.
Restricting it to radial-interior two-owner faces avoided that failure.

The best tested experimental split was:

- average the two owner-polynomial projected fluxes on radial-interior
  two-owner faces;
- use the owner-polynomial flux on embedded cut-wall faces;
- retain the direct path on regular radial boundaries.

It produced:

| Resolution | Category | Volume L2 | Linf |
|---:|---|---:|---:|
| 10 | all active | `0.444420` | `3.782884` |
| 10 | one wall | `0.766496` | `1.764649` |
| 10 | multi wall | `0.833655` | `2.315240` |
| 10 | aggregate target | `0.826535` | `2.315240` |
| 10 | retained cut cell | `0.687384` | `1.552876` |
| 14 | all active | `0.240544` | `4.284419` |
| 14 | one wall | `0.472573` | `1.205924` |
| 14 | multi wall | `0.451669` | `1.514102` |
| 14 | aggregate target | `0.481845` | `1.514102` |
| 14 | retained cut cell | `0.215238` | `0.361580` |
| 18 | all active | `0.177572` | `3.430412` |
| 18 | one wall | `0.632235` | `3.430412` |
| 18 | multi wall | `0.179591` | `0.599786` |
| 18 | aggregate target | `1.308957` | `3.430412` |
| 18 | retained cut cell | `0.322684` | `2.080289` |

The `N=10,14` all-active volume-L2 order is `1.824`, but the three-grid
regression is:

| Category | Volume-L2 order | Linf order |
|---|---:|---:|
| all active | `1.575` | `0.137` |
| multi wall | `2.568` | `2.241` |
| aggregate target | `-0.651` | `-0.563` |
| retained cut cell | `1.405` | `-0.232` |

The multi-wall category now converges. The `N=18` failure is concentrated in
one-wall aggregates and coincides with a topology change: aggregate count
falls from 96 at `N=14` to 36 at `N=18`. The worst aggregate has only four
Dirichlet samples, and its owner-polynomial cut-wall flux is
`-2.591e-2` versus the exact `-2.695e-3`.

Using the direct cut-wall functional instead does not solve the problem. At
`N=18` it worsens the all-active error to `0.21070/4.35758`, and individual
direct cut-wall fluxes can have the wrong sign. The owner-polynomial cut-wall
path is better overall, but its one-wall reconstruction remains
underconstrained by boundary information.

### 11.4 Boundary weighting in owner reconstruction

A one-wall cubic owner reconstruction can contain up to 48 cell-average
equations but only four Dirichlet equations. A default-neutral diagnostic
scales only those Dirichlet equations:

| Boundary scale | `N=18` all L2/Linf | One-wall L2/Linf | Multi-wall L2/Linf | Aggregate L2/Linf |
|---:|---|---|---|---|
| 1 | `0.17757 / 3.43041` | `0.63223 / 3.43041` | `0.17959 / 0.59979` | `1.30896 / 3.43041` |
| 10 | `0.16920 / 3.03060` | `0.55036 / 3.03060` | `0.14694 / 0.57952` | `1.22736 / 3.03060` |
| 100 | `0.17270 / 2.98781` | `0.54700 / 2.98781` | `0.19357 / 0.73147` | `1.24581 / 2.98781` |

The response saturates. Boundary weighting is directionally helpful but does
not remove the topology-sensitive one-wall aggregate error. A later
three-resolution scale-10 sweep gave:

| Resolution | All L2/Linf | One-wall L2/Linf | Multi-wall L2/Linf | Aggregate L2/Linf |
|---:|---|---|---|---|
| 10 | `0.46922 / 4.43032` | `0.85870 / 2.11224` | `0.95824 / 2.45551` | `0.94644 / 2.45551` |
| 14 | `0.21446 / 4.09232` | `0.33306 / 1.16648` | `0.26139 / 0.78586` | `0.30762 / 1.16648` |
| 18 | `0.16920 / 3.03060` | `0.55036 / 3.03060` | `0.14694 / 0.57952` | `1.22736 / 3.03060` |

The fitted all-active volume-L2 order is `1.768`, still below the requested
`1.8`, and the one-wall and aggregate categories reverse on the finest grid.
Boundary scale 10 is therefore not a fix.

### 11.5 Diagnostic code state

The committed baseline contains default-neutral controls for:

- exact product-average face auditing;
- observation counts and weight splits;
- face-functional boundary weight scale;
- all-local-owner boundary observations, one shard only;
- face-functional cell radius;
- reconstruction boundary equation weight scale.

The current worktree additionally contains:

- the native
  `replace_local_control_volume_projected_flux_with_owner_polynomials`
  helper;
- local-local and local-remote two-owner projected-flux averaging;
- cut-wall minus-owner projected flux;
- invalid selected-row propagation without silent fallback;
- global-radial eligibility that excludes the first two owner layers at each
  regular boundary;
- an opt-in reconstruction distance-row exponent and a target mask restricted
  to the compact reconstruction neighborhood away from those radial layers;
- CLI validation and plumbing for the distance exponent;
- focused eager/JIT tests for the helper, including the remote-owner branch.

All defaults remain unchanged. The helper no longer has a one-shard software
restriction because it consumes the polynomial remote-face gradient already
exchanged by the existing lowering path. This host has only one JAX device,
so decomposition safety is covered structurally and by the remote-branch unit
test, not yet by a real multi-shard convergence run.

### 11.6 Localized owner-reconstruction experiment

The distance experiment applies a dimensionless row multiplier to selected
compact owner reconstructions. Exponent zero reproduces the legacy `1/d`
weighted-least-squares row multiplier exactly. A positive exponent `e` uses
`1/max(d,1)^e` for selected rows. The unit floor avoids singularly
overweighting observations inside one local cell width.

Three variants were tried:

1. Applying exponent 4 to every selected reconstruction row improved wall
   categories but created a large non-wall reconstruction error next to the
   regular radial boundary.
2. Applying exponent 4 only to exact wall owners destroyed cancellation
   between differently weighted adjacent owner polynomials and severely
   worsened the multi-wall category.
3. Applying exponent 4 to the complete compact reconstruction neighborhood,
   while excluding the first two global radial owner layers, preserved the
   shared weighting context and avoided the worst boundary-layer damage.

For variant 3 with boundary scale 10:

| Resolution | Category | Volume L2 | Linf |
|---:|---|---:|---:|
| 10 | all active | `0.438811` | `3.511981` |
| 10 | one wall | `0.247087` | `0.712083` |
| 10 | multi wall | `0.937001` | `3.237150` |
| 10 | aggregate target | `0.802880` | `3.237150` |
| 10 | retained cut cell | `0.258182` | `0.416787` |
| 14 | all active | `0.215370` | `2.643948` |
| 14 | one wall | `0.153163` | `0.461210` |
| 14 | multi wall | `0.281706` | `0.952191` |
| 14 | aggregate target | `0.217621` | `0.952191` |
| 14 | retained cut cell | `0.198419` | `0.417714` |
| 18 | all active | `0.100708` | `1.956872` |
| 18 | one wall | `0.191200` | `1.180011` |
| 18 | multi wall | `0.127367` | `0.603588` |
| 18 | aggregate target | `0.411046` | `1.180011` |
| 18 | retained cut cell | `0.110576` | `0.702874` |

Three-grid fitted orders are:

| Category | Volume-L2 order | Linf order |
|---|---:|---:|
| all active | `2.483` | `0.987` |
| bulk | `1.726` | `1.508` |
| one wall | `0.490` | `-0.741` |
| multi wall | `3.405` | `2.900` |
| aggregate target | `1.290` | `1.822` |
| retained cut cell | `1.406` | `-0.841` |

The `N=18` face audit separates the remaining failures:

- The worst active owner `(2,14,2)` has no embedded wall face. Its dominant
  radial face is outside the shared-owner replacement eligibility and uses
  the direct functional. The numerical integrated residual is
  `1.102662e-2` versus `2.102979e-2` reference.
- The worst aggregate `(3,14,2)` is dominated by its cut-wall owner-polynomial
  flux: `1.618864e-2` numerical versus `6.946984e-3` analytic. Its compact
  signed sum closes, so scatter and volume division are not the first defect.

A milder exponent 2 was then checked at `N=10,14`. It gave all-active L2
`0.457714 -> 0.233861` (order `1.996`) but Linf
`3.511981 -> 3.439366` (order `0.062`). The same non-wall compact
reconstruction outlier persisted, and the N=10 wall categories were worse
than exponent 4. No exponent-2 `N=18` run is warranted from this checkpoint.

## 12. Primary-Literature Check

The literature supports the overall architecture:

- Devendran, Graves, Johansen, and Ligocki use weighted least squares for
  fourth-order Cartesian embedded-boundary Poisson stencils and validate both
  convergence and operator stability:
  [CAMCoS 12 (2017)](https://escholarship.org/uc/item/9b97g2dg).
- Overton-Katz et al. reconstruct face fluxes from control-volume moments with
  overdetermined weighted least squares, add physical boundary conditions as
  equations whenever a neighboring cell contains boundary, and use an
  inverse-fifth-power distance weight for fourth-order stencils:
  [SIAM J. Sci. Comput. 45 (2023)](https://arxiv.org/pdf/2209.02840).
- Thacher, Johansen, and Martin build cell-centered Taylor reconstructions
  constrained by interface or boundary data, use SVD and distance weights
  proportional to `(1 + distance)^-(P+1)`, and enforce conservation by
  averaging the neighboring Taylor-polynomial fluxes into one shared face
  flux:
  [J. Comput. Phys. 491 (2023)](https://escholarship.org/uc/item/69t7h4bx).
- Established second-order Cartesian cut-cell methods also impose interface
  matching through boundary flux approximations and demonstrate convergence
  on nontrivial geometries:
  [Colella and Graves, JCP 230 (2011)](https://www.osti.gov/biblio/21499787).

The comparison gives the following judgment.

**The project is on the right mathematical track, but the present direct face
fit is too global and too weakly localized.** Moment-aware volume averages,
boundary equations, integrated face fluxes, and one conservative physical
face record are all literature-aligned. The successful two-owner diagnostic
is especially significant because a recent high-order conservative method
uses exactly this neighboring-polynomial averaging pattern.

The important discrepancies are:

1. The current direct face fit can use 105--165 cell observations with only
   inverse-distance-squared localization. The cited high-order methods use a
   neighborhood only as large as needed for rank/robustness and make weights
   decay faster than the highest polynomial growth, for example inverse fifth
   power for a fourth-order fit.
2. Boundary observations are currently evaluator-owner restricted. The
   literature includes boundary equations from every boundary-containing
   neighbor in the reconstruction.
3. Tuning an arbitrary boundary multiplier is not a substitute for a unified,
   nondimensional distance and equation-type weighting law.
4. A radius-1 failure does not imply the resolution is unusable. Published
   fourth-order methods use radius-3 neighborhoods near boundaries. The
   requirement is enough equations with controlled locality, not the smallest
   possible stencil.
5. Cut cells commonly dominate Linf error even in successful methods. That
   explains why Linf is the hardest norm, but it does not relax the current
   acceptance gate.
6. High-order embedded-boundary elliptic methods can remain stable without
   merging. Combined with the agglomeration-disabled result, this reinforces
   that agglomeration is not the first error source here.

## 13. Paused Handoff

No full convergence test should be started from this checkpoint. The isolated
perpendicular operator still fails its three-grid all-active and Linf gates,
and the parallel-density operator has not been repaired.

When work resumes:

1. Keep the native decomposition-safe shared-flux helper and its canonical
   scatter invariant; validate it on real multiple devices when available.
2. Diagnose the regular-boundary-adjacent direct-functional row separately
   from the embedded cut-wall owner reconstruction.
3. Include boundary equations from every relevant boundary-containing
   neighbor, including a defined remote exchange/lowering path.
4. Replace the broad fixed support with the smallest full-rank support chosen
   adaptively or by controlled shells; retain the dimensionless distance law
   and record reconstruction/functional weight norms.
5. Investigate why the `N=18` cut-wall owner polynomial overpredicts the
   dominant wall flux despite full cubic rank and low reproduction residual.
6. Preserve the direct-functional and owner-polynomial face audit so exact,
   direct, minus-owner, plus-owner, and final shared flux can be compared on
   the same face.
7. Rerun only `perp_laplacian_phi` at `N=10,14,18`. Proceed to the parallel
   flux only after all-active, one-wall, multi-wall, aggregate, and retained
   categories reduce monotonically.
8. Continue with the regular radial-boundary operators, projected-exact-phi
   full RHS, phi inversion, decomposed equivalence, and finally the full
   time-dependent four-field MMS test.

The next agent should not increase polynomial degree, restore the removed
wall-normal point-gradient patch, or treat either the scale-10 or exponent-4
global-L2 result as a fix. The current evidence points to two distinct
accuracy defects: a regular-boundary-adjacent direct-functional row and an
embedded cut-wall owner reconstruction. Conservative face construction is no
longer the missing implementation step.
