# Geometry Metric Evaluator Call Chain

This note documents the current wall-fitted geometry pipeline for the
`D^2 x S^1` mesh path. The goal is to keep the geometry layer explicit:
`B` is evaluated by a dedicated field evaluator, the scalar mesh coordinate
`eta` is fitted from that field, the wall provides the boundary geometry, the
MMPDE solver relaxes the structured mesh, and the final `MetricEvaluator`
turns the solved node positions into a smooth interpolation object.

The relevant top-level call chain is:

```text
tests/test_MetricEvaluator.py:_hsx_cli
  -> build_hsx_metric_plot(...)
    -> bfield_evaluator_from_makegrid(...)
    -> scalar_potential_evaluator_from_bfield(...)
    -> build_metric_evaluator(...)
       -> build_wall_fitted_initial_mesh(...)
          -> WallEvaluator.constant_eta_boundary_curve(...)
       -> solve_mmpde(...)
       -> MetricEvaluator(...)
```

The same chain applies when the geometry layer is used from another driver:

1. load a MAKEGRID field bundle into a `BFieldEvaluator`
2. fit the normalized mesh coordinate `eta` from that field
3. build a wall-fitted initial `D^2 x S^1` mesh
4. relax the mesh with the MMPDE solver
5. fit the final smooth metric object from the solved nodes

## Coordinate and topology convention

The current geometry path uses a nonperiodic logical square in the first two
coordinates and a periodic toroidal direction in the third:

- `u in [0, 1]`
- `v in [0, 1]`
- `eta` is endpoint-exclusive and spans one field period

The physical embedding is a Cartesian surface/volume map. The final metric
object stores a smooth representation of that embedding and its derivatives.

## 1. `BFieldEvaluator`

File: [src/drbx/geometry/Bfield_evaluator.py](../geometry/Bfield_evaluator.py)

The `BFieldEvaluator` interface is the low-level magnetic-field layer. Its
concrete MAKEGRID implementation is `ComponentSplineBFieldEvaluator`.

What it does:

- evaluates the magnetic field in cylindrical or Cartesian coordinates
- preserves the source grids `R`, `phi`, `Z`, the field-period count `nfp`,
  and the current weights used to assemble the bundle
- provides a fast vectorized field evaluator for arbitrary query points

How it works:

- the MAKEGRID file is read on its native cylindrical grid
- the combined coil-group field is stored as component arrays on
  `(phi, Z, R, 3)`
- `phi` is treated as periodic over one field period
- `R` and `Z` are interpolated with local spline machinery
- `evaluate_cartesian()` converts Cartesian query points to cylindrical
  coordinates, evaluates the field there, and returns Cartesian components

Current handling is explicit:

- scaled `S` files interpret the supplied `currents` as amperes
- raw `R` files interpret the supplied `currents` as dimensionless
  multipliers of the currents already baked into the component arrays

This evaluator is only responsible for returning a smooth `B(x)`. It does
not know anything about `eta`, the wall, or the MMPDE solve.

## 2. `ScalarPotentialEvaluator`

File: [src/drbx/geometry/ScalarPotential_evaluator.py](../geometry/ScalarPotential_evaluator.py)

The scalar-potential layer fits a smooth analytic representation of the
normalized mesh coordinate `eta` from the magnetic field.

What it does:

- fits a scalar potential `Phi` in least squares from sampled `B`
- returns the normalized periodic mesh coordinate `eta`
- returns `grad(eta)` in cylindrical or Cartesian form
- optionally fits the secular `I` term if a reference axis is provided

How it works:

- the fit minimizes a weighted projection error of the form
  `|grad Phi - B|^2`
- the fitted potential is split into a secular part and a periodic remainder
  `Phi = I*theta_ref + G*(phi - phi0) + Phi_tilde`
- `Phi_tilde` is represented by a Chebyshev basis in `R` and `Z` and a
  Fourier basis in `phi`
- the public `evaluate_*` methods return normalized `eta`, not the magnetic
  potential itself
- `wrapped=True` is opt-in and is only used when a caller explicitly wants a
  periodic representative of `eta`

Important implementation point:

- the evaluator is analytic after the fit
- no nodal interpolation of `eta` is used downstream
- `gradient_cartesian()` and `gradient_cylindrical()` are computed from the
  fitted representation, not by finite differences

In the current wall-fitted pipeline this is the object the MMPDE solver
projects onto when it enforces a fixed `eta` surface on each toroidal layer.

## 3. `WallEvaluator`

File: [src/drbx/geometry/WallEvaluator.py](../geometry/WallEvaluator.py)

The wall evaluator provides the geometric boundary used to seed the mesh.

What it does:

- parses Kisslinger / FLARE wall input
- evaluates the wall surface periodically in the toroidal angle
- provides wall containment and nearest-point utilities
- provides a toroidally varying reference axis used by the scalar-potential
  fit when needed
- produces a constant-`eta` wall contour for each toroidal plane

How it works:

- the wall file is parsed into toroidal planes and poloidal contours
- each channel is turned into a periodic tensor-product spline
- `constant_eta_boundary_curve()` brackets a requested `eta` value on the
  wall over one field period and refines the crossing with vectorized
  bisection
- `reference_axis()` returns the centerline and its toroidal derivatives
  for the optional `I` term in the scalar-potential fit

This object is the only geometry-specific source of the wall shape. The mesh
solver itself never reads the wall file directly.

## 4. `build_wall_fitted_initial_mesh`

File: [src/drbx/geometry/MetricEvaluator.py](../geometry/MetricEvaluator.py)

This helper constructs the initial Cartesian mesh before the MMPDE relaxes
it.

What it does:

- asks the wall for a closed constant-`eta` boundary curve on each toroidal
  plane
- maps that contour onto the perimeter of the logical square
- fills the interior with a discrete harmonic extension

How it works:

- the perimeter of the `u-v` square is assigned in a fixed order
  `bottom -> right -> top -> left`
- the wall contour is re-ordered to preserve the expected orientation of the
  Cartesian `D^2 x S^1` chart
- the interior nodes are solved by a 5-point Laplacian
- the result is a positive-radius Cartesian seed mesh of shape
  `(nu, nv, neta, 3)`

This step is not the final solve. It only produces a stable starting point
for the MMPDE relaxation.

## 5. `solve_mmpde`

File: [src/drbx/geometry/solve_MMPDE.py](../geometry/solve_MMPDE.py)

This is the actual mesh relaxation engine.

What it does:

- relaxes a structured `D^2 x S^1` mesh
- keeps the logical `u` and `v` edges nonperiodic
- keeps the toroidal `eta` direction periodic
- backtracks whenever a candidate step loses Jacobian positivity or fails to
  reduce the frozen-monitor energy

How it works:

- the objective is dimensionless and combines a normalized frozen-monitor
  Dirichlet edge regularizer with four cell objectives: metric alignment,
  metric-volume equidistribution, a positive-volume barrier, and neighboring
  log-volume smoothness
- the cell monitor is formed from frozen nodal monitors; the reference
  ``vbar`` is the initial mean metric volume and remains fixed throughout the
  solve
- a monitor can be either constant or callable; callable monitors are sampled
  once per iteration and held fixed during its line search
- the periodic toroidal identification is handled through a rigid-image
  callback, including the exact rotational pullback at the eta seam
- the projector is applied to every candidate iterate before acceptance
- the first descent trial is capped by
  ``maximum_step_fraction`` times the initial median physical cell-edge
  length, then ordinary backtracking continues
- boundary nodes are fixed by mask, the candidate Jacobian positivity check
  remains a hard feasibility constraint, and the toroidal seam is checked
  explicitly
- the solver terminates when the free-node update is small or when the
  iteration budget is exhausted
- `MMPDEResult.component_energy_history` records raw per-component histories
  plus the weighted total, which makes relaxation regressions visible without
  reconstructing the objective from the mesh

The solver is intentionally geometry-agnostic. It does not know about the
wall, the field, or the scalar potential. It only sees:

- a starting mesh
- a monitor
- a projector
- a periodic image map
- a fixed-node mask

The current implementation keeps boundary nodes fixed.  A future constraint
mechanism could allow wall nodes to slide tangentially along an exact wall
curve while preserving the wall constraint; that is separate from, and not
currently implemented by, the composite objective.

## 6. `MetricEvaluator`

File: [src/drbx/geometry/MetricEvaluator.py](../geometry/MetricEvaluator.py)

This is the final smooth geometry object returned to downstream consumers.

What it does:

- evaluates the smooth embedding `x(u, v, eta)`
- computes the Jacobian matrix and determinant
- builds covariant and contravariant metric tensors
- evaluates magnetic fields in metric coordinates
- provides cell-center and open-boundary-face sampling helpers

How it works:

- the solved node positions are treated as samples on a structured grid
- each Cartesian channel is decomposed into a Fourier series in `eta`
  whose coefficients are splines in `(u, v)`
- the reconstructed position is recovered from the cylindrical radius,
  toroidal phase correction, and vertical coordinate
- `jacobian_matrix()` is assembled by differentiating the fitted channels
- `evaluate()` then computes `J`, `g_cov`, `g_contra`, and the inverse
  consistency residual
- `evaluate_magnetic_field()` maps a Cartesian `B` field through the
  Jacobian so the field can be reported in logical and covariant forms

The class also exposes the sampling used to validate the fit:

- `sample_cell_centers()` checks interior finite-volume cells
- `sample_open_boundary_faces()` checks the non-corner boundary faces

Those checks are what protect the final accepted fit from a negative or
nearly singular Jacobian at the wall-fitted corners.

## Final acceptance rule

The mesh build is only accepted when all of the following are true:

- the wall contour closes and matches the field-period seam
- the scalar-potential projection converges to the requested `eta` levels
- the MMPDE solve converges or produces an acceptable projected candidate
- the final sampled Jacobian is positive at cell centers and open boundary
  faces
- the smooth `MetricEvaluator` can be constructed from the accepted nodes

## Practical debugging order

When something looks wrong, the fastest way to isolate it is usually:

1. check the raw `BFieldEvaluator`
2. check the fitted `ScalarPotentialEvaluator`
3. check the wall contour and the initial mesh seed
4. check the MMPDE result before metric smoothing
5. check the final `MetricEvaluator` samples and Jacobian sign

That order matches the amount of structure added at each stage, so it is the
cleanest way to find where the geometry first becomes inconsistent.
