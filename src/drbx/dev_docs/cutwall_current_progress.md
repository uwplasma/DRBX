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

Snapshot date: 2026-07-22.

Repository baseline at the time of this snapshot: `f2a50f60`, plus the current
worktree change that releases each compiled operator kernel and its device
outputs during the convergence sweep.

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

Geometry preprocessing is also too expensive for the current problem size:

```text
N=10:  92.514 s
N=14: 144.647 s
```

The single-shard builder currently repeats major global and local bundle work
and compiles global face-functional records more than once. Removing that
duplication is the next infrastructure priority.

## 1. Latest Validation Configuration

The completed sweep used:

```text
resolutions                 N=10,14
shard counts                1,1,1
agglomeration               enabled
operator-only mode          enabled
phi algebraic solve         skipped
full-RHS phi                projected exact phi
minimum requested order     1.8
CUDA command buffers        disabled
```

Disabling CUDA command buffers was initially necessary because the sweep
exhausted accelerator memory while instantiating a later compiled operator.
The convergence harness now also:

- converts completed operator outputs to NumPy host arrays;
- explicitly deletes the compiled scalar kernel and its device outputs;
- calls `jax.clear_caches()` between scalar operators;
- runs Python garbage collection between scalar operators.

These changes allowed all scalar operators, the projected-exact-phi full RHS,
and both resolutions to complete.

## 2. Geometry and Reconstruction Results

The geometry summary was:

| Quantity | `N=10` | `N=14` |
|---|---:|---:|
| Active aggregate owners | 952 | 2648 |
| Merged sources | 48 | 96 |
| Aggregate targets | 48 | 96 |
| Irregular faces | 898 | 1522 |
| Interior compact faces | 622 | 1058 |
| Partial faces | 140 | 252 |
| Cut-wall faces | 136 | 212 |
| Cubic reconstruction rows | 566 | 956 |
| Quadratic fallbacks | 0 | 0 |
| Linear fallbacks | 0 | 0 |
| Maximum reported condition number | `5.336e4` | `1.404e5` |

Positive conclusions:

- Agglomeration is active at both resolutions.
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

The worst aggregate should be audited face by face. For every physical face,
print its global face ID, kind, signed area measure, exact integrated flux,
numerical integrated flux, residual contribution, and final owner.

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

## 7. Geometry Build-Time Problem

Measured preprocessing time:

```text
N=10:  92.514 s for  898 irregular faces and 566 reconstruction rows
N=14: 144.647 s for 1522 irregular faces and 956 reconstruction rows
```

The growth broadly follows the irregular wall band rather than the full cell
count, which is desirable. The absolute constant is not acceptable for an
iterative development test.

The current `_build_stacked_embedded_control_volume_geometry` path performs:

1. Global raw moments, face measures, and aggregate topology.
2. A complete unsplit global embedded bundle.
3. Internal compilation of global cubic face-functional records while
   building that bundle.
4. A second explicit compilation of those global records.
5. Construction of equivalent local geometry and cell data.
6. Another complete local embedded bundle.
7. Padding and stacking of the local bundles.

For `shard_counts=(1,1,1)`, the unsplit global bundle and the only local bundle
describe the same partition. Rebuilding both is unnecessary.

The global topology builder also loops in Python over every coordinate face,
constructs its open rectangles, evaluates quadrature, and evaluates the
shifted-torus metric separately. That path should eventually be batched.

### Build optimization priorities

1. Add phase timers before changing algorithms:

   ```text
   raw moments
   global face measures
   global topology/agglomeration
   global face discovery
   cell reconstruction precompute
   global direct functional fitting
   local lowering
   regular boundary closure
   padding and stacking
   ```

2. Add a one-shard fast path that directly stacks and returns the already
   constructed unsplit bundle.
3. Refactor global functional compilation so generated records are returned
   and reused rather than recomputed.
4. Batch the coordinate-face measure calculation.
5. Add an optional persistent geometry cache keyed by all geometry-affecting
   inputs after the build path is deterministic and validated.

The first two changes should remove major duplicated work without changing
the numerical method.

## 8. Known Diagnostic Noise

### Inactive centroid warnings

The geometry builder evaluates centroid metric data over the full storage
array. Inactive solid and merged-source slots contain the placeholder
centroid `(0,0,0)`, which produces divide-by-zero warnings even though the
physical radial domain begins at `x_min=0.2`.

Use a valid reference coordinate for inactive entries before evaluating
centroid metric and curvature arrays. This is a data sanitation fix and
should not alter any active-owner result.

### Empty categories

In the one-shard run, `remote_interface` has count zero. Its printed L2 and
Linf values are `nan` because the category is empty, not because an active
operator result is invalid.

### Relative errors for zero references

When the exact reference norm is zero, machine-precision absolute errors are
printed with enormous relative errors. Acceptance and diagnosis must use the
absolute error or mark the result exact/degenerate.

## 9. Prioritized Next Steps

### Phase A: make iteration affordable

1. Add geometry phase timings.
2. Add the one-shard bundle-reuse fast path.
3. Remove duplicate global functional compilation.
4. Sanitize inactive centroid metric evaluation.
5. Add an operator-selection CLI option to the sweep.

### Phase B: isolate the multi-wall aggregate defect

1. Reproduce only:

   ```text
   perp_laplacian_phi
   parallel_density_flux_divergence
   ```

2. For the maximum-error aggregate, dump every contributing face.
3. Compare numerical and exact integrated flux per face.
4. Verify one unique face ID and one signed scatter per physical face.
5. Verify the sum is divided by the same aggregate volume used by the MMS
   projection.
6. Report direct-functional condition numbers, reproduction residuals, and
   normalized weight norms for those faces.
7. Repeat with agglomeration disabled to separate the cut-face closure from
   aggregate ownership/divergence.

### Phase C: isolate regular radial-boundary defects

Run only:

```text
grad_parallel_v_electron
poisson_omega
poisson_v_electron
curvature_phi
```

Compare radial-boundary owners with radial-interior owners and test the
regular radial closure independently of the embedded box.

### Phase D: establish convergence

After Phases A-C:

1. Run at least three resolutions, initially `N=10,14,18`.
2. Require monotone error reduction before interpreting a fitted order.
3. Inspect all-active, bulk, one-wall, multi-wall, aggregate-target, retained
   cut-cell, and radial-boundary categories separately.
4. Run the projected-exact-phi full RHS only after the isolated operators are
   understood.
5. Re-enable the algebraic phi solve after the spatial operator sweep passes.
6. Run a decomposed case to validate remote compact faces and reverse
   residual accumulation.
7. Finally run the time-dependent four-field shifted-torus MMS convergence
   test.

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

## 11. Immediate Handoff

The next agent should not begin by increasing polynomial degree or restoring
the removed wall-normal gradient patch.

The immediate work package is:

1. instrument geometry build phases;
2. remove single-shard duplicate bundle and functional construction;
3. add targeted operator selection;
4. audit the worst `perp_laplacian_phi` aggregate face by face;
5. determine whether the first bad quantity is an individual face flux, its
   signed scatter, aggregate-volume division, or the MMS reference.

Only after that audit should a third resolution be used to decide whether the
remaining behavior is pre-asymptotic or a closure defect.
