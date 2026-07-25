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

Snapshot date: 2026-07-24.

Repository baseline at the time of this snapshot: `cf1d2ad7` (`add
decomposition-safe owner flux diagnostics`). The resumed investigation below
was performed from that commit; the alternative diagnostics described here
were temporary experiments and were reverted.

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
the requested minimum order of 1.8. The later controlled translated-topology
campaign described below did establish the isolated scalar gate; this opening
history explains why that additional control was required.

Two coarse resolutions are not enough to identify a trustworthy asymptotic
order, especially because the embedded box intersects a different set of
logical cells and cut fractions at each resolution. Nevertheless, the sweep
contains a strong localized warning: the perpendicular Laplacian maximum
error grows at multi-wall aggregate targets while nearby dense cells converge
near second order. That behavior should be diagnosed before simply running a
larger full sweep.

All five Phase A items are implemented and checked in the committed baseline:
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

The accepted perpendicular closure is now the default and only control-volume
perpendicular path. Eligible active faces with plus or remote owners use the
symmetric two-owner polynomial flux when both owners pass the radial-interior
rule, including partial faces. Eligible cut-wall faces use the minus/fluid-owner
polynomial. The first two global radial owner layers retain the direct-
functional fallback. The canonical face record and equal-and-opposite scatter
are unchanged. Focused eager and JIT tests cover local, remote, cut-wall,
invalid-row, and radial-boundary cases.

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

The resumed N=22 and N=26 measurements show a continued fine-grid decrease in
all-active volume L2 and near-second-order bulk behavior, but the wall-category
errors and all-active maximum are not monotone. This is consistent with a
topology-sensitive pre-asymptotic regime, not an accepted isolated convergence
result. All alternative diagnostics tried in the resumed investigation were
reverted; the committed implementation remains `cf1d2ad7`.

The retained path has now been extended to `N=40,60,80` with
`shard_counts=(1,1,4)` on the controlled half-cell sequence. It gives
all-active orders `1.991546/1.962795` and bulk orders `1.998658/1.858362`,
with no invalid reconstruction rows. This validates the perpendicular
operator and sharding, but does not validate the projected full RHS, phi
inversion convergence, or full time MMS.

Production perpendicular diffusion, `Ti` reconstruction, and the `phi`
inverse now build and pass owner polynomials; control-volume perpendicular
calls reject a missing polynomial. The former
`--perp-use-two-owner-polynomial-flux` and
`--perp-use-cutwall-owner-polynomial-flux` flags and their diagnostic/manual
path were removed. Generic direct face functionals remain because the
parallel/gradient closures and radial fallback still need them.

## Completed Isolated-Operator Campaign and Current Decision

The preceding paragraphs preserve the fixed-grid and early diagnostic history.
The subsequent campaign added explicit translation controls and completed a
controlled translated-topology ensemble. The geometry, exact/source
projection, convergence, and runtime paths now accept:

```text
--box-translation DX DTHETA DZ
--box-cell-translation FX FTHETA FZETA
```

`box_translation` is an absolute `(dx, dtheta, dz)` offset. The optional
resolution-scaled control is applied as

```text
effective_translation = box_translation
                       + (FX * dx_cell, FTHETA * dtheta_cell,
                          FZETA * dz_cell)
```

where the cell widths are the current logical widths in the three coordinate
directions. The controls are threaded consistently through geometry creation,
exact and source projection, operator convergence, and runtime initialization.
The resulting geometry report includes topology descriptors: raw and aggregate
volume ratios relative to the median positive active raw volume, maximum
received source/member counts, and p95/max norms of the projected face weights.
`top_error` records the relevant volume and source/member metadata alongside
the error.

The fixed-box topology sequence oscillates and has an `N=26` Linf rebound. A
quarter-cell translated sequence improves the fine-grid trend, while the
half-cell sequence `(0, 0.5, 0.5)` is the cleanest controlled ensemble. This
half-cell phase is an O(h) translation sequence: it is a deliberate way to
sample changing cut topology at each resolution, not proof of convergence for
one fixed physical geometry.

On the half-cell sequence, the isolated all-active results are:

| Operator | Resolutions | L2 order | Linf order |
| --- | --- | ---: | ---: |
| `perp_laplacian_phi` | N=40,60,80 retained fine path | 1.991546 | 1.962795 |
| `parallel_density_flux_divergence` | N=18,22,26 | 1.899879 | 1.928475 |
| `poisson_omega` | N=18,22,26 | 2.010197 | 2.236330 |
| `poisson_v_electron` | N=18,22,26 | 2.254952 | 3.144134 |
| `curvature_phi` | N=18,22,26 | 1.843095 | 2.226615 |
| `grad_parallel_v_electron` | N=26,30,34 fine window | 1.812328 | 1.982034 |
| `grad_parallel_phi` | N=26,30,34 fine window | 1.811031 | 1.939895 |

The parallel-density conservative signed sum is exactly zero and invalid
quadrature is zero. `grad_parallel_density`, `grad_parallel_v_ion`,
`poisson_density`, `poisson_v_ion`, and `curvature_density` are zero-target
exact/roundoff cases rather than nonzero order tests. The direct regular-grid
`grad` baseline on N=18,22,26 was approximately second order (`~1.99`), which
is consistent with the fine-window cut-wall results.

The projected-exact-phi full-RHS run was also completed on N=18,22,26. Its
all-active orders are:

| Component | L2 order | Linf order | Interpretation |
| --- | ---: | ---: | --- |
| full RHS density | 1.877320 | 1.955059 | passes the 1.8 gate |
| full RHS omega | 1.567201 | 1.482046 | misses |
| full RHS v_electron | 1.602884 | 1.787992 | misses narrowly in Linf and clearly in L2 |
| full RHS v_ion | exact | exact | zero-target/exact cancellation |

Representative full-RHS L2 errors for density, omega, and electron were,
respectively, `.02091865/.01441512/.01048588`,
`.9997461/.7420204/.5612542`, and `11.48058/8.426111/6.362756` at
N=18/22/26. Since the isolated scalar pieces pass, the omega and electron
results require a cancellation-aware interpretation. At the tested `t=0`
stage, the exact omega and electron time derivatives are zero, so these rows
deliberately test cancellation between the discrete operator sum and the
projected source. The electron residual is especially sensitive because
`mi_over_me=1836` multiplies `grad_parallel_phi`. Its adjacent L2 orders
improve from `1.541` on N=18/22 to `1.681` on N=22/26; omega similarly
improves from `1.486` to `1.671`. This matches the delayed fine-window
asymptotics already seen in the isolated parallel gradients. The first
hypothesis is therefore pre-asymptotic, coefficient-amplified cancellation,
not a new cut-wall flux defect. These measurements do not by themselves prove
that the production discretization is wrong.

The current stop/go decision is therefore: the isolated scalar stage is
established on a controlled translated sequence, but the project is not ready
to proceed to phi inversion or time convergence. The next bounded diagnostic
must report each full-RHS component's discrete error and cancellation:

- omega: Poisson, parallel-gradient difference, and curvature terms;
- electron: Poisson, phi parallel-gradient, and pressure-gradient terms;
- independently projected analytic terms summed together versus projection of
  their pointwise sum.

Before changing production discretization, add a full-RHS-only/term-resolved
mode and repeat the same N=26,30,34 fine window used to establish the two slow
parallel gradients. Less cancellation-sensitive stage times or MMS
amplitudes/fields are secondary conditioning checks, not replacements for the
zero-target consistency test. All reported runs used one shard; real
multi-device validation remains unavailable. N=34 geometry preprocessing is
approximately 330 s, so caching and geometry reuse are now priorities. No
invalid reconstruction rows occurred in the reported runs.

## Resumed Isolated Perpendicular Investigation: N=18, 22, 26

The retained configuration was the decomposition-safe two-owner projected
flux, cut-wall owner-polynomial flux, `1/d^4` reconstruction row weighting,
boundary equation scale 10, one shard, agglomeration enabled, projected exact
phi, and `perp_laplacian_phi` only. The fine-grid geometry summaries were:

| Quantity | `N=22` | `N=26` |
|---|---:|---:|
| Active aggregate owners | 10364 | 17162 |
| Merged sources | 44 | 54 |
| Aggregate targets | 44 | 54 |
| Irregular faces | 3794 | 5045 |
| Interior compact faces | 2390 | 3145 |
| Partial faces | 860 | 1176 |
| Cut-wall faces | 544 | 724 |
| Cubic reconstruction rows | 2412 | 3168 |
| Maximum reported condition number | 32.50649 | 34.98056 |
| Invalid reconstruction rows | 0 | 0 |

The volume-weighted errors were:

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

The N=22 to N=26 adjacent orders were `0.942/-4.723` for all-active L2/Linf
and `1.965/1.867` for bulk L2/Linf. The three-grid N=18,22,26 all-active
orders were `2.985` in volume L2 and `1.269` in Linf. The wall categories are
nonmonotone, so these slopes do not establish convergence. The bulk signal is
near second order, while cut-wall and agglomeration topology controls the
oscillatory extrema. The N=18 spike is therefore not sufficient evidence that
the functional samples values from an unacceptably distant region, and the
N=22 decrease does not remove the need for matched-topology testing.

### Fine-grid retained-path confirmation: N=40,60,80

The controlled half-cell sequence was repeated at `N=40,60,80` using the
complete retained perpendicular configuration:

```text
agglomeration                               enabled
box-cell translation                       (0, 0.5, 0.5)
accepted consolidated perpendicular closure enabled
reconstruction distance row exponent       4
reconstruction boundary equation scale     10
selected operator                          perp_laplacian_phi
```

Every required reconstruction remained cubic and valid. The main errors were:

| Resolution | Category | Volume L2 | Linf |
|---:|---|---:|---:|
| 40 | all active | `1.242690e-2` | `1.995061e-1` |
| 60 | all active | `5.689690e-3` | `1.902729e-1` |
| 80 | all active | `3.115134e-3` | `4.678020e-2` |
| 40 | bulk | `1.173657e-2` | `1.083002e-1` |
| 60 | bulk | `5.216219e-3` | `5.068009e-2` |
| 80 | bulk | `2.934457e-3` | `2.988912e-2` |
| 40 | one wall | `2.370849e-2` | `1.995061e-1` |
| 60 | one wall | `1.377665e-2` | `1.902729e-1` |
| 80 | one wall | `5.896897e-3` | `4.370982e-2` |
| 40 | multi-wall | `1.513551e-2` | `7.216366e-2` |
| 60 | multi-wall | `2.702957e-2` | `1.206391e-1` |
| 80 | multi-wall | `2.160301e-3` | `1.183225e-2` |

The three-grid fitted orders were:

| Category | Volume-L2 order | Linf order |
|---|---:|---:|
| all active | `1.991546` | `1.962795` |
| bulk | `1.998658` | `1.858362` |
| one wall | `1.963498` | `2.054307` |
| multi-wall | `2.530420` | `2.354154` |
| reconstruction row | `1.499804` | `1.962795` |
| retained cut cell | `1.960667` | `2.054307` |
| radial lower owner | `1.953705` | `2.017633` |
| radial upper owner | `1.863010` | `1.821438` |

This passes the requested `1.8` all-active gate in both norms. It is strong
fine-grid evidence that the accepted perpendicular compact-face path is one
unique conservative physical flux formed from adjacent owner polynomials on
eligible radial-interior faces, including partial faces, with the fluid-owner
polynomial on eligible cut-wall faces and a strongly localized owner
reconstruction. The first two global radial owner layers use the direct-
functional fallback.

The result must not be overinterpreted:

- The all-active adjacent L2 orders are stable (`1.927` then `2.094`), but
  adjacent Linf orders are `0.117` then `4.877`. The fitted Linf order is good
  because the `N=80` maximum falls sharply, not because every pair is smoothly
  asymptotic.
- Agglomeration is topology dependent: the sequence has `0`, `360`, and `0`
  merged sources at `N=40,60,80`. The multi-wall error increases at `N=60`
  before dropping at `N=80`, and aggregate-target accuracy cannot be fitted
  from one populated grid.
- The reconstruction-row L2 category fits only `1.500`, although its error
  improves on both adjacent pairs and reaches `2.054` on `N=60 -> 80`.
- The worst cell changes from a one-wall reconstruction owner at `N=40,60`
  to a non-wall compact reconstruction owner with one irregular face at
  `N=80`. The maximum nevertheless falls from `1.902729e-1` to
  `4.678020e-2`.
- The completed `shard_counts=(1,1,4)` run executes the remote-owner path and
  confirms the isolated operator's sharding behavior. It does not validate
  the projected full RHS, phi inversion convergence, or full time MMS.

The completed output
`/home/exouser/shifted_torus_perp_laplacian_n40_n60_n80_shards_1x1x4.txt`
also reports geometry elapsed times of `68.397 s`, `156.276 s`, and
`330.598 s` for `N=40,60,80`. Local-bundle totals were payload/bundle
`2.673/27.295 s`, `9.339/63.001 s`, and `22.931/139.311 s`, respectively.

Functional conditioning remained benign: maximum reconstruction condition
numbers were `28.83/39.04/28.74`, no invalid reconstruction rows occurred,
and maximum normalized projected-functional weight norms were
`3.34/3.93/4.73`.

### N=18 reconstruction-row audit

Two representative N=18 rows both had rank 19 and benign scaled conditioning:

```text
target (2,14,2), row 110: condition 14.4384; 48 cell observations, 0 BC
target (3,14,2), row 208: condition 24.8176; 48 cell observations, 4 BC
```

The largest raw transformed derivative coefficients were expected from the
`h^-1`, `h^-2`, and `h^-3` derivative scaling. After multiplying by the local
grid spacing, the dominant first-derivative coefficients were O(1). This audit
does not support rank loss or coefficient amplification as the explanation for
the N=18 outlier.

### Rejected diagnostics (all reverted)

The following experiments were run against N=10 and then removed. They are
recorded to prevent repeating them as presumed fixes:

- Neighborhood-complete wall observations: all-active `0.4679744/3.521068`
  with boundary scale 10; one-wall `0.6004812/1.588156`; multi-wall
  `1.137921/3.521068`; aggregate `1.033229/3.521068`. The expanded wall
  observation set worsened the relevant global and wall errors.
- Nearest-48 direct-functional target observations with adaptive expansion:
  all-active `0.6027166/5.200490`; multi-wall `1.169427/3.605770`; aggregate
  `0.9803488/3.605770`; retained `0.5234641/1.168551`. Truncating the direct
  rows did not provide a uniform improvement.
- Radial owner-flux guard 2 to 1: all-active `0.4531168/3.396870`; one-wall
  `0.2470874/0.7120828`; multi-wall `1.305913/3.201480`; aggregate
  `1.042473/3.201480`; retained `0.9927974/2.320360`. The shared interior
  flux extension, rather than only cut-wall eligibility, caused degradation.
- Farther-interior owner choice: all-active `0.4826284/7.061519`; multi-wall
  `0.5800003/1.678479`; aggregate `0.4185170/1.678479`; retained
  `0.7545753/1.590821`. Face audits found wrong-sign owner-polynomial fluxes
  at regular-boundary-adjacent faces, so this heuristic was rejected.

The code and tests for all four diagnostics were reverted. No production code
change from these experiments is present beyond `cf1d2ad7`.

## 1. Earlier Validation Configuration

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

The same configuration was subsequently extended one resolution at a time to
`N=22` and `N=26`; their exact results are recorded in the resumed
investigation section above.

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

The global topology builder previously looped in Python over every coordinate
face, constructed its open rectangles, and evaluated quadrature separately.
At `N=80` this means

```text
(81 * 80 * 80) + (80 * 81 * 80) + (80 * 80 * 81)
  = 1,555,200 coordinate faces.
```

An operator sweep started before the optimization measured
`845.624 s` in `raw moments and global topology`. The timing was consistent
with cubic scaling of the per-face Python work.

The following geometry optimizations are now implemented:

1. Coordinate-face clipping, quadrature-point generation, and shifted-torus
   metric evaluation run in fixed-size NumPy batches. The decomposition is
   regression-tested against the previous scalar face quadrature.
2. Global aggregate moments use a bulk copy for one-cell aggregates and
   recompute only owners that actually receive merge sources. The old code
   scanned the complete aggregate-ID array once per active owner, making this
   part quadratic in the global cell count.
3. Compact face candidates, nearest-periodic-image wrapping, cubic
   control-volume basis rows, and boundary trace rows are formed in batches.
4. The radius-3 reconstruction fallback forms all candidate offsets and cubic
   design rows in batches instead of scalar nested Python loops.
5. `--geometry-cache-dir PATH` enables a versioned, numeric-only NPZ cache for
   compiled global face-functional records. The cache key includes all
   geometry and functional settings. Cache hit/miss equivalence is tested
   exactly; this cache does not yet bypass raw moments or global topology.

Measured results on this host:

```text
isolated global topology, N=40:  4.31 s wall, including process/import
isolated global topology, N=80: 29.41 s wall, including process/import
N=80 peak resident memory:      approximately 1.9 GB

complete one-shard N=10 geometry:
  previous baseline: 46.0 s
  optimized cold:      9.1 s
  face-functional cache hit: 5.2 s
```

The isolated `N=80` comparison is about a 29-fold reduction from the observed
old `845.624 s` phase. A Python process already running the sweep retains the
old functions and must be stopped and restarted to use these changes.

### Remaining build optimization priorities

1. Reprofile the full `N=80` build after restart; the coordinate-face record
   emission is still a Python loop over approximately 1.5 million retained
   faces, although it is no longer the dominant 845-second path.
2. Consider caching the deterministic raw topology and reconstruction payload
   only if repeated fine-grid runs remain too expensive after measurement.
3. Keep the cache schema explicit and invalidate it whenever functional
   semantics change.

### Resolution-level memory cleanup

The first `N=40,80,100` operator sweep completed `N=80` but stopped during
the `N=100` geometry build immediately after cell lowering, without a Python
traceback. The host has `14 GiB` RAM and no swap. The convergence loop retained
the previous resolution's host geometry, device geometry, exact fields,
invariants, and diagnostic cell arrays while evaluating the next geometry
builder's right-hand side. The `N=80` payload therefore overlapped the
allocation peak of the `N=100` face/reconstruction/functional phase.

All operator-run exit paths now explicitly delete resolution-owned host and
device payloads, clear JAX compilation caches, and run Python garbage
collection before continuing. This covers targeted-operator mode,
projected-phi/full-RHS mode, and the phi-solve path. The time-dependent
four-field convergence loop performs the same cleanup after each completed
resolution. Runs print

```text
Released shifted_torus resolution payloads: N=..., collected=...
```

before beginning the next resolution. A real two-resolution targeted run
confirmed that the cleanup message appears before the next geometry build,
and a full-RHS/phi-skipped `N=6` run completed through the same cleanup path.

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

The symmetric physical-face flux is now the production perpendicular closure.
The remaining method questions are controlled reconstruction locality and
boundary accuracy in the other operators. A global boundary multiplier alone
does not fix topology-sensitive rows, and a strong distance law improves
global L2 while leaving category and Linf failures.

### Phase C: lock radial/parallel/gradient regressions

The regular radial fallback remains intentional for the first two global
radial owner layers. Lock the radial, parallel, and gradient operator
regressions before broadening the accepted perpendicular campaign. The
earlier targeted checks were:

```text
grad_parallel_v_electron
poisson_omega
poisson_v_electron
curvature_phi
```

Compare radial-boundary owners with radial-interior owners and test the
regular radial closure independently of the embedded box.

### Phase D: establish convergence

The isolated scalar stage and perpendicular sharding validation are complete
on the controlled half-cell translated sequence. The projected-exact-phi full
RHS is not yet accepted because its omega and electron components miss the
order gate. Do not re-enable the algebraic phi solve or begin time convergence
until the bounded full-RHS cancellation diagnostic is understood.

## 10. Acceptance Gates

The cut-wall operator path is ready for the final four-field convergence test
when:

- every required compact row remains valid and cubic;
- polynomial reproduction residuals are near floating-point tolerance;
- normalized direct-functional weights remain within an explicit bound;
- one physical face maps to one canonical flux record;
- shared interior fluxes cancel conservatively;
- the controlled translated-sequence scalar tests support at least the
  requested 1.8 all-active volume-L2 and Linf order for each nondegenerate
  operator;
- the projected full-RHS component cancellations are understood and the
  omega/electron component orders meet the acceptance target;
- regular radial-boundary and embedded-wall failures are independently
  resolved;
- the projected-exact-phi full RHS passes before the phi inversion is added;
- the decomposed result agrees with the one-shard result;
- geometry preprocessing is fast enough to make repeated validation
  practical, or a validated persistent cache is available.

The present isolated scalar gate is met on the half-cell sequence. The full
four-field gate is not met: projected full-RHS omega and electron remain below
the target, so phi inversion and time convergence remain intentionally paused.

## Working-Tree and Execution Notes

The translation controls, topology diagnostics, convergence harness changes,
and documentation updates are currently uncommitted. Git identity is still
unset in this workspace, so a commit cannot be created until an author
identity is configured. The isolated validation campaign includes the
completed `1x1x4` run; the full-RHS and time-dependent paths remain
unvalidated. Reported runs had no invalid reconstruction rows. Caching and
reuse remain practical priorities before broader sweeps.

## Sharding Handoff: Isolated Perpendicular Operator

The sharding infrastructure has now been exercised with an actual `1x1x4`
layout on the shifted-torus retained `perp_laplacian_phi` path. This supersedes
the earlier statement above that multi-device behavior was unvalidated. The
validation scope is deliberately limited: it covers the isolated operator,
not the full four-field RHS, algebraic phi solve, or time-dependent MMS
convergence.

The completed decomposition work includes:

- boundary-observation source-shard metadata and global all-gather, so a
  boundary observation is evaluated from its owning source shard rather than
  from a shard-local periodic wrap;
- canonical whole-domain face lowering and canonical whole-domain cubic
  reconstruction lowering on every evaluator shard;
- canonical dense/compact face masks and target masks, with one global record
  per physical face and decomposition-independent row ordering;
- decomposed radius-2 support requiring `halo_width >= 3`;
- production distributed owner aggregation and owner-field expansion,
  including remote source averages and halo exchange;
- reverse-scattered remote residual contributions back to their canonical
  aggregate owners.

Two decomposition bugs were found and fixed. First, a local periodic-support
candidate could incorrectly close a valid dense face at a shard boundary;
global dense/compact canonicalization now prevents that. Second, a same-ID
local compact row could survive instead of being replaced by the canonical
whole-domain row. For face `10488`, the retained local row targeted
`(8,13,4)` while the canonical row correctly targeted `(8,14,4)`.

The explicit N=16 one-shard versus `1x1x4` diagnostic now reports only
roundoff-level decomposition differences: operator maximum difference
`4.440892098500626e-15` and compact signed-sum difference `6.94e-18`.
The retained one-shard and `1x1x4` runs have identical active/face/
reconstruction counts, error summaries, and top-error record, with zero
invalid reconstruction rows. The decomposed report additionally identifies
`remote_interface` count `134`; that is a reporting category, not an extra
physical face or an operator discrepancy.

The completed isolated 40/60/80 operator sweep with four devices used:

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
Its all-active errors are `1.242690e-02`, `5.689690e-03`, and
`3.115134e-03`; final orders are L2 `1.991546` and Linf `1.962795`.
Bulk orders are L2 `1.998658` and Linf `1.858362`; remote-interface orders
are L2 `1.450711` and Linf `1.424796`. This validates the isolated
perpendicular operator and sharding only.

### Sharded preprocessing optimization

The decomposed local-bundle path has now been optimized. Decomposed local
bundles directly lower the canonical whole-domain reconstruction instead of
calling per-shard `precompute_local_moment_reconstruction`. Construction of
the remote reconstruction sample cloud was also removed. The legacy and
one-shard fitting path remains unchanged; local face discovery still remains
in the sharded path.

The optimization removes the redundant per-shard SVD/reconstruction fitting
work while preserving the canonical row ordering, face payloads, target
masks, and lowered reconstruction data. The numeric face-functional cache
still covers only the global compiled face functionals; it does not cache all
geometry or topology. The former redundant local SVD work is nevertheless no
longer performed on the decomposed path.

An independent cache-hit N=40 `1x1x4` run measured:

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
lowering; the independent run confirmed `29.390 s`, versus the previous
`584.190 s` for that phase, an approximately `19.9x` reduction. The N=16
one-shard versus `1x1x4` operator difference remains `4.44e-15`.

This is an isolated N=40 timing and an N=16 decomposition-equivalence
check; the completed 40/60/80 fine-grid sharded result is recorded above.

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

Commit `cf1d2ad7` additionally contains:

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

The former perpendicular switches and diagnostic/manual path were removed.
The owner-polynomial closure is now the default and only control-volume
perpendicular path. The helper consumes the polynomial remote-face gradient
already exchanged by the existing lowering path; a missing polynomial is a
hard error in a control-volume perpendicular call. Generic direct face
functionals remain for parallel/gradient closures and the first two radial
fallback layers.

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

## 13. Current Handoff

The retained perpendicular path is now confirmed on the controlled half-cell
sequence through `N=80` and the `shard_counts=(1,1,4)` run: its all-active
orders are `1.991546/1.962795`, with bulk `1.998658/1.858362`, no invalid
reconstruction rows, and benign conditioning. The primary compact projected-
flux path is owner reconstruction followed by one shared conservative
physical-face flux, using both adjacent owner polynomials on eligible
radial-interior active faces (including partial faces) and the fluid-owner
polynomial on eligible cut-wall faces. The first two global radial owner
layers retain direct-functional fallback.

The `1/d^4` reconstruction localization, boundary weight 10, agglomeration,
and generic face-audit infrastructure remain part of the validated
configuration. The completed isolated run includes remote-owner sharding;
another wall phase is still required before broader acceptance.

Do not start the full time-dependent four-field test yet:
projected-exact-phi full-RHS omega and electron miss the order gate despite
the isolated scalar pieces passing.

Next steps:

1. Run one affordable alternate wall-phase regression.
2. Lock radial, parallel, and gradient operator regressions.
3. Rerun projected-exact-phi full RHS and isolate omega/electron
   cancellation and order failures.
4. Establish one-vs-four full-RHS equivalence.
5. Then measure phi-inverse convergence.
6. Only then run full four-field time MMS.

Do not repeat the four reverted local diagnostics or treat the successful
isolated sequence as proof for a single fixed physical cut geometry.
