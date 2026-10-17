# Geometry and Metric Pipeline

The geometry stack converts MAKEGRID magnetic data and vessel data into one
continuous `MetricEvaluator`, then samples it onto the PDE grid. The unified
factory supports two topologies with different construction algorithms.

## Main data flow

```text
MAKEGRID file -> BFieldEvaluator -------------------------+
                                                        |
                                                        v
                     ScalarPotentialEvaluator (eta): Chebyshev-Fourier (default)
                                                     or local B-spline
                                                        |
vessel file -> WallEvaluator ----------------------------+
                                                        |
                                                        v
                                  build_metric_evaluator(topology=...)
                                                        |
                                                        v
                         continuous MetricEvaluator (fitted planes, default)
                           [toroidal + spline eta: ExactPlaneMetricEvaluator,
                            planes are exact level sets of eta]
                                                        |
                                                        v
                                  FciGeometry3D sampled on PDE cells/faces
```

The main implementation is
[`geometry/MetricEvaluator.py`](../geometry/MetricEvaluator.py). The HSX
producer orchestration is
[`geometry/hsx_fci_builder.py`](../geometry/hsx_fci_builder.py)
(`build_hsx_metric_evaluator`, `build_hsx_fci_geometry`), driven by
[`generate_hsx_fci_geometry.py`](../../../generate_hsx_fci_geometry.py).
The eta and plane selection is `eta_representation="chebyshev_fourier" |
"spline"` (with `spline_eta_options`) and `plane_construction="fitted" |
"exact"`; `exact` requires the spline eta and the toroidal topology.

## Shared input evaluators

### Magnetic field

`BFieldEvaluator` interpolates MAKEGRID cylindrical field components and
returns Cartesian or cylindrical magnetic data. Current weighting depends on
the MAKEGRID representation: scaled component files consume supplied coil
currents, while raw component files use dimensionless multipliers for currents
already represented in the data.

### Scalar potential and toroidal coordinate

`ScalarPotentialEvaluator` fits the scalar-potential coordinate used as
`eta`. It carries a field-period `period` and, when available, `nfp`.
`build_metric_evaluator` rejects inconsistent period and field-period
metadata. Both representations return unwrapped, quasi-periodic eta from
`evaluate_*(..., wrapped=False)` (`eta(phi + period) = eta(phi) + period`)
and `eta mod period` with `wrapped=True`; Cartesian queries use the
principal branch `atan2(y, x)`.

`chebyshev_fourier` (default,
[`ScalarPotential_evaluator.py`](../geometry/ScalarPotential_evaluator.py)):
`Phi = I theta_ref + G (phi - phi0) + Phi~`, `Phi~` a global Chebyshev
(R, Z) x Fourier (phi) series fitted by dense least squares on a node grid
inside the vessel. The driver option `--axis-core-radius` only excludes the
singular reference-axis neighborhood from this scalar fit; it is not an
operator-side axis treatment or evolved core region.

`spline`
([`spline_scalar_potential.py`](../geometry/spline_scalar_potential.py)):
`chi = G (phi - phi0) + Phi~`, `I = 0`, `Phi~` a tensor product of uniform
B-splines of odd degree (quintic default) in `(R, Z)`, periodic in `phi`
over one field period; `eta = phi - phi0 + Phi~/G`.
`spline_scalar_potential_evaluator_from_bfield(bfield | loader, wall, ...)`
minimizes `|grad(chi) - B|^2 / Bs^2` (cylindrical volume weights,
`Bs = median |B|`) on the Gauss points of knot cells whose wall
pseudo-radius is `<= rho_data` (1.02), plus `lam h^2 |Lap Phi~|^2` on
interior cells only (all `(p+1)^3` functions active; penalizing cells with
inactive functions biases the fit). `G` is an unknown. The operator is
matrix-free and sum-factorised in JAX (dense R/Z GEMMs, a banded periodic phi
stage), column-scaled by the exact `diag(A^T A)`, and solved by JAX LSMR
warm-started from a coarse fit (2 cm, 45 phi cells, 600 iterations) through
exact dyadic B-spline prolongation. The fine solve stops on the LSMR
tolerances, `maxiter`, or an intrinsic monitor: every 100 iterations the
tan-angle misfit `angle(grad chi, B)` RMS and max on a fixed subsample of the
fit's own data points, stopping when both change by less than 0.2 % after at
least 200 iterations. The knot box is `spline_knot_box`: Z centred on 0 with
a half-width snapped up to a multiple of `max(1 cm, h_coarse/2)`, `R_lo`
independent of the cell width, phi knots from `phi0 = 0`; the fit refuses any
other box. This box makes the fit stellarator-symmetric to round-off and lets
coarse and fine boxes nest. A loader argument is called only for the
right-hand-side assembly and released before the solves (the HSX builder
reloads MAKEGRID afterwards), so the field table is not resident during
LSMR. The evaluator rejects queries outside the knot box (`R` and `Z` padded
2 cm beyond the vessel). `JaxSplinePotential` is its PyTree runtime, which
also solves `eta_hat(R, phi, Z) = eta` for `phi` by fixed-iteration Newton
with implicit derivatives (`jax.lax.custom_root`), exact under nested
`jacfwd` to third order, also with respect to the spline coefficients.

### Wall

`WallEvaluator` parses the vessel representation, evaluates wall points and
derivatives, and supplies constant-eta wall curves through the eta evaluator
(`constant_eta_boundary_curve` brackets the unwrapped eta from
`evaluate_cylindrical(..., wrapped=False)` along each wall poloidal line and
bisects). The toroidal map is always built on the selected eta's own wall
curves, so `u = 1` follows that eta's level curves on the vessel.
It is separate from the final metric evaluator because wall evaluation and
volume-map evaluation have different domains and representations.

## Unified metric factory

`build_metric_evaluator(..., topology="square" | "toroidal")` is the only
public metric factory. Both branches return `MetricEvaluator`; callers do not
select a topology-specific evaluator class.

### Square topology

The logical domain is `[0,1]^2 x S1`, with coordinates `(u,v,eta)`.

1. `build_wall_fitted_initial_mesh` samples a wall contour on every
   endpoint-exclusive eta plane.
2. The contour is mapped to the perimeter of the logical square.
3. A discrete harmonic extension fills the interior.
4. `solve_mmpde` optionally moves interior nodes while fixed boundary nodes
   remain on the wall.
5. The final sampled Cartesian map is represented by periodic tensor-product
   splines.

The eta seam is a rotational field-period seam. Square topology has four
logical perimeter faces representing one physical vessel wall and no
coordinate axis.

Relevant driver controls are `--fit-sample-shape`,
`--metric-spline-degree`, and `--mmpde-iterations`.

### Toroidal topology

The logical domain is the polar chart `D2 x S1`, with coordinates
`(u,theta,eta)`:

- `u=0` is the collapsed magnetic axis;
- `u=1` is the physical vessel wall;
- theta and eta are periodic;
- theta samples are endpoint-exclusive over `2*pi`;
- metric eta samples are per field period.

The builder samples and phase-aligns wall curves, selects or constructs an
axis curve, initializes a disk-to-wall map, optionally projects interior eta,
and fits an axis-regular Fourier-Zernike representation. The basis enforces
the radial scaling required for smooth polar modes. The resulting evaluator
is periodic over its field-period representation and is used to materialize
the full-torus PDE grid.

Relevant driver controls are:

- `--metric-mesh-shape NU NTHETA NETA_PER_PERIOD`;
- `--metric-radial-degree`;
- `--metric-poloidal-modes`;
- `--metric-toroidal-modes`;
- `--eta-projection-iterations`;
- `--eta-representation`, `--spline-eta-options`, `--plane-construction`.

#### Exact level-set planes

With `plane_construction="exact"` the builder wraps the fitted toroidal map
in `ExactPlaneMetricEvaluator`
([`exact_plane_metric.py`](../geometry/exact_plane_metric.py)). The map keeps
the base map's `R(u, theta, eta)` and `Z(u, theta, eta)` and replaces its
fitted angle by the solution of `eta_hat(R, phi, Z) = eta` at every
evaluation point (fixed Newton iterations from the base map's
`phi = eta + delta_phi`, about 1e-4 rad off). `eta_hat` increases with `phi`
in the vessel, so each level set is a graph over `(R, Z)`. The Jacobian
follows by implicit differentiation,
`phi_q = (delta_{q,eta} - eta_R R_q - eta_Z Z_q) / eta_phi`, so the plane
normal `X_u x X_theta` is parallel to `grad(eta_hat)` to round-off. The
evaluator is a `MetricEvaluator` subclass with the same public evaluation
semantics (`evaluate`, `position`, `jacobian_matrix`, `sample*`,
`evaluate_regularized`, magnetic-field projection) and is computed by the JAX
runtime `JaxExactPlaneMetricEvaluator` on fixed padded blocks. It holds no
fitted channels of its own (`_channels` is empty): code that reaches into the
Fourier--Zernike channels for a fast path (for example a plane sampler)
falls back to `evaluate` instead of silently evaluating the base map.
`JaxMetricEvaluator.from_metric_evaluator` and `from_cache_payload` return
the exact-plane runtime for such a metric; its `position_and_jacobian`,
`evaluate`, `evaluate_regularized` and `jacobian_matrix` are pointwise with
fixed shapes and differentiable to third order (curvature and divergence
identities through nested `jacfwd`), and evaluate beyond `u = 1` (e.g. the
eta-filter table's `u = 1.02`) inside the spline knot box.

`--metric-mesh-shape` controls nodes used to construct the continuous map.
`--resolution` controls PDE cell counts after that fit. They are distinct
because fitting and PDE discretization are separate operations, although the
spectral mode count must remain resolvable by the chosen sampling and PDE
grids.

## Sampling and caches

The continuous evaluator supplies position, Jacobian, covariant and
contravariant metrics, and regularized axis frames. Geometry assembly samples
these quantities at cells and faces. The toroidal PDE grid spans the full
`2*pi` torus even though the continuous map is fitted from one field period.

The driver caches evaluated metric data and validates cache metadata before
reuse. Toroidal angular-RLP geometry has a separate cache keyed by the metric
cache identity, grid faces, and selected angular profile.

Non-default eta/plane selections add `eta_representation`, `spline_eta`
(the resolved spline options) and `plane_construction` to the metric-cache
spec; the defaults add nothing, so their cache keys are unchanged, and specs
without these keys are compared as the defaults. An exact-plane metric's
payload is the base map's payload plus `plane_construction =
"exact_level_set"`, the Newton iteration count and the spline potential
under `exact_plane_eta_`, so a cache hit restores the complete map (the
builder context does not carry eta). `MetricEvaluator.from_cache_payload`
and `JaxMetricEvaluator.from_cache_payload` dispatch on that flag.

## Required invariants

For either topology:

- Cartesian positions and metric tensors are finite;
- the signed Jacobian has the expected positive orientation;
- eta period and `nfp` agree;
- periodic seams close with the required rotation;
- sampled field and metric array shapes match the PDE grid.

Additional toroidal invariants:

- every point at `u=0` collapses to the axis independently of theta;
- regularized metric data remain finite as `u -> 0`;
- only `u=1` is a physical radial wall;
- theta and eta endpoint identification is periodic.

Exact level-set planes additionally require:

- `|eta_hat(X(u, theta, eta)) - eta|` at round-off (evaluations raise above
  1e-9, i.e. for a non-converged solve);
- `d eta_hat / d phi > 0` along every solve;
- every evaluated point inside the spline knot box;
- `R` and `Z` identical to the base map's channels.

## Validation

- [`tests/test_MetricEvaluator.py`](../../../tests/test_MetricEvaluator.py)
  covers the unified factory and square map.
- [`tests/test_MetricEvaluator_toroidal.py`](../../../tests/test_MetricEvaluator_toroidal.py)
  covers axis collapse, Fourier-Zernike regularity, caches, and toroidal
  sampling.
- [`tests/test_toroidal_metric_builder.py`](../../../tests/test_toroidal_metric_builder.py)
  covers wall alignment, initialization, eta projection, the spline eta's
  wall level curves feeding the toroidal builder, and the eta/plane
  selection checks of the HSX builders.
- [`tests/test_spline_scalar_potential.py`](../../../tests/test_spline_scalar_potential.py)
  covers the B-spline basis, the least-squares operator (adjoint, column
  norms), synthetic harmonic recovery, stellarator symmetry, the knot-box
  guard, prolongation, the phi solve and its implicit derivatives, payloads,
  and `wrapped=` semantics.
- [`tests/test_exact_plane_metric.py`](../../../tests/test_exact_plane_metric.py)
  covers level-set exactness, the implicit Jacobian, NumPy/JAX parity,
  nested `jacfwd` to third order, bitwise batch independence of the autodiff
  curvature, `u = 1.02`, payload round trips and dispatch; a slow test builds
  the real-HSX spline/exact metric end to end when the inputs are present.
- [`tests/test_metric_cache_identity.py`](../../../tests/test_metric_cache_identity.py)
  covers cache identities, including legacy specs without the eta/plane keys.
- [`tests/test_simulate_hsx_blob_toroidal_geometry.py`](../../../tests/test_simulate_hsx_blob_toroidal_geometry.py)
  covers the driver topology contract.

## Experimental compact toroidal magnetic interpolation

`bfield_evaluator_from_makegrid(..., toroidal_method="compact_c2")` selects
an opt-in six-plane toroidal Hermite interpolant. `compact_c3` selects an
eight-plane alternative. The default remains `spline`; neither compact path
is wired into the canonical geometry producer or production driver defaults.
Both require `method="cubic"` and `extrapolate=False`.

For a native interval `[phi_i, phi_{i+1}]`, C2 constructs a quintic from the
values and first/second derivative estimates at both endpoints. Each nodal
jet uses the same centered five-plane polynomial differentiation formula in
both adjacent intervals. The combined support is `i-2,...,i+3`. C3 uses
first through third derivatives from seven-plane nodal formulas, a septic
interval polynomial, and support `i-3,...,i+4`. The plane indices wrap across
the field-period seam. No toroidal prefilter is applied; preparation filters
each plane only in R/Z, retaining the previous reflected cubic R/Z behavior.
C2/C3 refer to **toroidal** derivative continuity; the full tensor interpolant
retains C2 continuity in R/Z. These linear interpolants reproduce native
samples, but do not enforce `div B = 0` or monotonicity.

The host evaluator and `JaxComponentSplineBFieldEvaluator` use the same
compact weight contract. The JAX state stores the toroidal method as static
PyTree metadata, so compilation distinguishes the coefficient interpretation.
Any external persisted geometry/cache identity must also include this option
before using it outside an explicit diagnostic. Canonical traces generated
with a different evaluator cannot silently be reused as this evaluator's
traces. Plasma reconstruction and toroidal magnetic interpolation are
separate operations.

Verification is in `tests/test_compact_toroidal_bfield.py`: shared jets,
periodicity, source locality, native-plane/RZ parity, smooth refinement,
MAKEGRID current conventions, and JAX value/derivative agreement. These
algebraic tests do not establish physical HSX field accuracy between samples.
The compact candidates require real-HSX qualification before default adoption.
Hermite interpolation and differentiation of local interpolating polynomials
are standard constructions; see [NIST DLMF interpolation](https://dlmf.nist.gov/3.3.vi)
and [numerical differentiation](https://dlmf.nist.gov/3.4.i).
