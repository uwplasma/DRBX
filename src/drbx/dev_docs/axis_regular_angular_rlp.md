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

The native payload is shardable in eta. Each eta shard reconstructs the same
radial/theta owner profile locally and receives only its local raw and
aggregate physical-volume fields; moment arrays are not communicated because
the projected fine-grid RLP path does not consume them.

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

### Eta-only decomposition

Production RLP is decomposed only in `eta` (the third logical axis). The
production contract is

```text
--shard-counts 1 1 Seta
```

for every topology, including square topology. `x`/radial and `theta`
remain replicated within each device. Toroidal RLP is compatible with this
decomposition because an angular owner aggregate changes only radial and
theta membership: each aggregate is confined to one eta plane and therefore
cannot cross an eta shard boundary. No x/theta RLP decomposition is
supported, and there is no fallback to a different state representation or
single-device path when the contract is violated.

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

For `simulate_hsx_blob.py`, the driver automatically:

1. builds the continuous toroidal metric;
2. selects and caches the angular profile;
3. builds the host owner/volume geometry;
4. shards the two-volume-field payload and assembles eta-local RLP geometry
   inside each compiled kernel;
5. volume-averages the initial state into owners;
6. evolves owner-only state;
7. materializes fine-grid fields only for operators and output.

The production sharding requirement for all topologies is:

- `--shard-counts 1 1 Seta`, with `Seta >= 1` and `Neta` divisible by
  `Seta`;

The current toroidal operator requirements are:
- `--poisson-bracket-scheme compatible-flux`;
- `--curvature-scheme conservative`;
- `--gmres-preconditioner none` or `line-u`.

Coordinate and traced-FCI parallel operator families remain selectable. P/R
are local because an owner aggregate is eta-plane confined. Global means,
norms, compatibility projections, and GMRES convergence reductions still
use cross-device reductions over the eta (`z`) axis. The line-u preconditioner
contains local radial trees; its eta-face diagonal contributions include both
sides of a local slab, including faces incident on a slab interface. There is
no x/theta RLP decomposition and no fallback to full-grid toroidal phi,
fixed-ring topology, compact angular faces, or a Cartesian core.

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
