# Axis-Regular Angular RLP

Radius-dependent angular agglomeration is the production state and solver
architecture for toroidal topology. It removes the polar small-cell time-step
restriction without introducing a second near-axis polynomial representation.

RLP denotes the operator composition

```text
owner state --P--> fine polar state --A_f--> fine residual --R--> owner residual
```

or, algebraically,

```text
A_owner = R A_f P.
```

## Angular owner profile

For radial ring `i`, `q_i` is the number of adjacent theta cells represented
by one owner value. The profile obeys:

- `q_0 = Ntheta`, so the innermost ring has one owner per eta plane;
- every `q_i` is a positive divisor of `Ntheta`;
- `q_i` is non-increasing with radius;
- `q_{i+1}` divides `q_i`.

The metric-aware selector compares physical radial and angular widths and
chooses the least-coarsened nested profile that satisfies the width criterion.
Composite `Ntheta` values are supported when they provide a useful divisor
chain. The user selects PDE resolution; the geometry layer owns profile
selection.

Host construction is in
[`geometry/fci_geometry.py`](../geometry/fci_geometry.py) and
[`geometry/fci_control_volumes.py`](../geometry/fci_control_volumes.py):

- `metric_aware_angular_group_profile` selects the profile;
- `build_polar_angular_agglomeration_geometry` builds the owner map and
  physical volume moments;
- `build_metric_aware_polar_angular_agglomeration_geometry` handles selection
  and cache reuse.

Explicit profiles are a geometry-API testing facility, not a simulation CLI
option.

## Owner geometry

The host payload contains:

- the fine-cell-to-owner map;
- active-owner and merged-source masks;
- raw physical cell volumes and chart moments;
- aggregate physical volumes and chart moments;
- the angular profile and coordinate periods.

Native lowering in
[`native/fci_angular_agglomeration.py`](../native/fci_angular_agglomeration.py)
packs this data into `LocalEmbeddedControlVolumeGeometry3D`. Production RLP
lowering intentionally has:

- ordinary fine-grid regular faces;
- no irregular face rows;
- no moment-fitted face functionals;
- no Cartesian core reconstruction.

RLP currently requires one local domain/device for toroidal simulations.

## Prolongation and restriction

Prolongation is piecewise-constant owner injection. If fine cell `c` belongs
to owner `o(c)`,

```text
(P x)_c = x_o(c).
```

All fine cells in one angular aggregate therefore carry the same materialized
value during one operator evaluation.

Restriction is the physical-volume average

```text
(R y)_o = sum[c in o] V_c y_c / V_o,
V_o = sum[c in o] V_c.
```

Alias storage slots remain exactly zero in evolved and Krylov vectors.
`expand_local_control_volume_owner_field` implements `P`, and
`aggregate_local_control_volume_average` implements `R`.

This makes restriction the volume-weighted adjoint of injection up to the
owner mass matrix. Conservation and the Krylov inner product therefore use
the same physical-volume measure.

## Fine-grid operator contract

RLP does not repair an invalid fine-grid operator. `A_f` must already be a
valid operator on the materialized polar grid. For the scalar fields in the
seven-field state, the required axis behavior is:

- half-turn scalar halo continuation,
  `f(-u,theta,eta) = f(u,theta+pi,eta)`;
- classification of the lower radial side as an internal axis, not a wall;
- zero conservative flux through the collapsed lower radial face;
- signed-radius reflection plus a theta half-turn for FCI traces crossing the
  axis;
- finite, axis-regular metric and face coefficients.

There is no polynomial near-axis reconstruction, mode projection, compact
axis face fit, or separately evolved core state.

The production Poisson bracket uses shared compatible face fluxes and the
production curvature path is conservative. Physical-wall traces are applied
only at `u=1`; the axis never receives a physical-wall closure.

## Owner-space phi inversion

`LocalPerpLaplacianInverseSolver.solve_rlp_owner` solves directly in the
owner unknown space. GMRES uses:

- the active-owner mask;
- aggregate physical volumes for means, norms, and compatibility projection;
- an owner-space operator formed by `R A_f P`;
- the nested radial-tree `line-u` preconditioner when requested.

The tree preconditioner assembles conductances from the same ordinary fine
faces used by `A_f`. Owner-internal faces cancel. Distinct-owner radial
subfaces are summed into one child-parent edge. No compact-face fit is read by
the preconditioner.

## Driver contract

For `simulate_hsx_blob.py --topology toroidal`, the driver automatically:

1. builds the continuous toroidal metric;
2. selects and caches the angular profile;
3. lowers one RLP geometry;
4. volume-averages the initial state into owners;
5. evolves owner-only state;
6. materializes fine-grid fields only for operators and output.

The current toroidal requirements are:

- `--shard-counts 1 1 1`;
- `--poisson-bracket-scheme compatible-flux`;
- `--curvature-scheme conservative`;
- `--gmres-preconditioner none` or `line-u`.

Coordinate and traced-FCI parallel operator families remain selectable. There
is no fallback to full-grid toroidal phi, fixed-ring topology, compact angular
faces, or a Cartesian core.

## Validation

- [`tests/test_polar_angular_agglomeration_geometry.py`](../../../tests/test_polar_angular_agglomeration_geometry.py)
  validates profiles, owner maps, moments, and volume conservation.
- [`tests/test_fci_projected_fine_grid_control_volume.py`](../../../tests/test_fci_projected_fine_grid_control_volume.py)
  validates `R A_f P` behavior.
- [`tests/test_fci_angular_agglomeration_tree_preconditioner.py`](../../../tests/test_fci_angular_agglomeration_tree_preconditioner.py)
  compares the tree solve with dense owner systems.
- [`tests/test_fci_gmres_control_volume_owner_space.py`](../../../tests/test_fci_gmres_control_volume_owner_space.py)
  validates owner-space GMRES and physical-volume norms.
- [`tests/test_simulate_hsx_blob_angular_agglomeration.py`](../../../tests/test_simulate_hsx_blob_angular_agglomeration.py)
  validates automatic driver selection and the absence of legacy controls.
