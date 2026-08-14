# FCI Simulation Architecture

[`simulate_hsx_blob.py`](../../../../simulate_hsx_blob.py) constructs HSX
geometry and advances the seven-field electrostatic Boussinesq model through
the local/sharded FCI stack. This document records the currently selectable
runtime architecture.

## Driver pipeline

```text
MAKEGRID + vessel
  -> magnetic, eta, wall, and metric evaluators
  -> global FciGeometry3D sampled on the PDE grid
  -> optional traced forward/backward FCI maps
  -> ShardedFciGeometry3D and local halo topology
  -> automatic angular RLP geometry for toroidal topology
  -> LocalFciDrbEBRhs
  -> RK4, ARK2 IMEX, or SBDF2 IMEX advance
  -> history and atomic snapshot NPZ files
```

The native model is
[`native/fci_drb_EB_rhs.py`](../native/fci_drb_EB_rhs.py). Geometry lowering
is in [`native/fci_sharding.py`](../native/fci_sharding.py), and scalar halo
rules are in [`native/fci_halo.py`](../native/fci_halo.py).

## State

`FciDrbEBState` stores:

```text
density, phi, Te, Ti, Vi, Ve, vorticity
```

Phi is algebraic: the perpendicular polarization equation reconstructs it
from vorticity and ion-temperature terms. On square topology the inversion
uses the ordinary owned grid. On toroidal topology it uses the angular-RLP
owner space.

## Operator families

| Subsystem | Selectable schemes | Production toroidal requirement |
|---|---|---|
| Poisson bracket | `direct`, `compatible-flux` | `compatible-flux` |
| Curvature | `direct`, `conservative`, `disabled` | `conservative` |
| Parallel derivatives | `coordinate`, `fci` | Either, with valid FCI maps required for `fci` |
| Phi preconditioner | `none`, `jacobi`, line variants | `none` or angular-tree `line-u` |

The compatible Poisson bracket uses antisymmetrized shared-face fluxes and
includes the RHS `1/B` factor. Conservative curvature uses shared face
coefficients. Operator-specific physical-wall traces are supplied when an
operator needs a scalar value or flux.

The coordinate parallel family evaluates conservative coordinate-space face
fluxes. The FCI family traces to adjacent eta planes and evaluates parallel
operators from forward/backward mapped data. FCI map construction supports
the signed-radius axis crossing used by toroidal topology.

## Boundary layers

Boundary handling has three separate responsibilities:

1. **Topology fill** handles periodic theta/eta seams and polar half-turn
   continuation. These are identifications, not physical BCs.
2. **Physical ghost fill** enforces primitive Dirichlet or Neumann data at the
   vessel wall.
3. **Operator trace/flux closure** converts the physics-level candidate wall
   state into the quantity required by a particular conservative operator.

The lower toroidal radial side is an internal axis. It receives half-turn
scalar continuation and zero collapsed-face conservative flux; it never
receives a vessel-wall condition. The physical radial wall is the upper
`u=1` face.

Parallel velocity wall candidates are selectable as `dirichlet-zero`,
`neumann`, or the diagnostic `bohm` state. Parallel inflow closure is
selectable as `central`, `local-characteristic`, or
`equilibrium-characteristic`. Conservative curvature closure is selectable
as `central` or `upwind-equilibrium`.

These characteristic closures are wall-flux policies. They do not replace
the interior discretization or the angular-RLP axis treatment.

## Toroidal production contract

Selecting `--topology toroidal` requires:

- an explicit `--metric-mesh-shape`;
- even `Ntheta`;
- one device/subdomain (`--shard-counts 1 1 1`);
- compatible-flux Poisson bracket;
- conservative curvature;
- `none` or `line-u` phi preconditioning.

The driver automatically creates the metric-aware angular owner profile. It
does not expose a user-selectable axis treatment, core polynomial degree,
observation/target rings, fixed-ring topology, compact angular operator, or
alternate phi state space.

See [Axis-regular angular RLP](axis_regular_angular_rlp.md) for the owner
operator and solver details.

## Time integration

- `rk4` advances the complete explicit RHS and performs the algebraic phi
  inversion at each stage.
- `ark2-imex` uses an ARK2 scheme with a Newton-FGMRES solve for the selected
  stiff electron/acoustic-polarization block.
- `imex-bdf2` uses one classical RK4 startup step, then fixed-step SBDF2 with
  one Newton-FGMRES solve per step.

The coupled Newton preconditioners are separate from the standalone phi
preconditioner. Runtime diagnostics report nonlinear and GMRES convergence,
state ranges, positivity, optional term fields, and grid-scale indicators.

## Output and restart

History output stores logical coordinates, Cartesian cell positions, state
history, topology metadata, metric settings, operator selections, solver
space, and angular owner metadata. Scheduled snapshots are written atomically
and can optionally include Ve RHS term fields. Restart accepts either a
snapshot or a selected frame from a history file.

RLP state is owner-sparse during evolution. Output materialization prolongs
owner values to all fine cells so visualization receives a complete field.

## Extension rules

- A new toroidal operator must be valid as a fine polar operator under
  half-turn fills and lower-axis zero flux before it is wrapped in `R A_f P`.
- A new conservative wall term must define its operator-level boundary trace
  or flux; primitive field BCs are not a universal flux closure.
- A new FCI parallel operator must define mapped endpoint behavior at physical
  wall hits and axis crossings.
- Do not add a second toroidal state representation or silent fallback around
  owner-space RLP.

## Validation

- [`tests/test_simulate_hsx_blob_fci_driver_wiring.py`](../../../tests/test_simulate_hsx_blob_fci_driver_wiring.py)
  covers driver/map plumbing.
- [`tests/test_simulate_hsx_blob_toroidal_geometry.py`](../../../tests/test_simulate_hsx_blob_toroidal_geometry.py)
  covers toroidal production constraints.
- [`tests/test_fci_drb_eb_operator_boundaries.py`](../../../tests/test_fci_drb_eb_operator_boundaries.py)
  and [`tests/test_fci_operator_boundary_fluxes.py`](../../../tests/test_fci_operator_boundary_fluxes.py)
  cover operator-level wall traces.
- [`tests/test_fci_drb_eb_parallel_operator_scheme.py`](../../../tests/test_fci_drb_eb_parallel_operator_scheme.py)
  covers coordinate/FCI selection.
- [`tests/test_simulate_hsx_blob_imex_cli.py`](../../../tests/test_simulate_hsx_blob_imex_cli.py)
  covers integrator and solver wiring.
