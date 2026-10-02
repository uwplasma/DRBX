# Geometry and Metric Pipeline

The geometry stack converts MAKEGRID magnetic data and vessel data into one
continuous `MetricEvaluator`, then samples it onto the PDE grid. The unified
factory supports two topologies with different construction algorithms.

## Main data flow

```text
MAKEGRID file -> BFieldEvaluator -------------------------+
                                                        |
                                                        v
                                      ScalarPotentialEvaluator (eta)
                                                        |
vessel file -> WallEvaluator ----------------------------+
                                                        |
                                                        v
                                  build_metric_evaluator(topology=...)
                                                        |
                                                        v
                                         continuous MetricEvaluator
                                                        |
                                                        v
                                  FciGeometry3D sampled on PDE cells/faces
```

The main implementation is
[`geometry/MetricEvaluator.py`](../geometry/MetricEvaluator.py). The HSX
orchestration lives in
[`simulate_hsx_blob.py`](../../../simulate_hsx_blob.py).

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
metadata.

The fit samples cylindrical space. The driver option `--axis-core-radius`
only excludes the singular reference-axis neighborhood from this scalar fit;
it is not an operator-side axis treatment or evolved core region.

### Wall

`WallEvaluator` parses the vessel representation, evaluates wall points and
derivatives, and supplies constant-eta wall curves through the eta evaluator.
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
- `--eta-projection-iterations`.

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

## Validation

- [`tests/test_MetricEvaluator.py`](../../../tests/test_MetricEvaluator.py)
  covers the unified factory and square map.
- [`tests/test_MetricEvaluator_toroidal.py`](../../../tests/test_MetricEvaluator_toroidal.py)
  covers axis collapse, Fourier-Zernike regularity, caches, and toroidal
  sampling.
- [`tests/test_toroidal_metric_builder.py`](../../../tests/test_toroidal_metric_builder.py)
  covers wall alignment, initialization, and eta projection.
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
