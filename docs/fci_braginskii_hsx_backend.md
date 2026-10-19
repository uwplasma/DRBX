# HSX FCI Braginskii backend (drbx.fci_braginskii)

`drbx.fci_braginskii` is a compact FCI drift-reduced Braginskii plasma backend
for HSX. It is the plasma model of the 2D_fci line at
`d340f4d638e0d9495546d1fd2e2c66c1c206704a`, vendored into its own namespace
(`drbx.fci_braginskii.native`, `drbx.fci_braginskii.geometry`) and independent
of the `drbx.native` FCI modules. The code was reduced to the one production
configuration below; with `rho_star = 1` and `implicit_current_phi_pair = false` the
compiled advance and its fields are identical to the base commit. The
defaults add two changes on top of that: a consistent rho* normalization and
an implicit current/potential pair (see Normalization below). Tracing is
about ten times faster because host-side validation no longer builds JAX's
jaxpr-walking error messages (`drbx.fci_braginskii._host_guards`).

The backend consumes a precomputed geometry artifact. The geometry builder
that produces it (field-line tracing, metric fitting, RLP construction) is a
separate subpackage, `drbx.fci_braginskii.geometry_build`, run through
the `[geometry]` deck table. It is the 2D_fci producer at
`6c2b005d81d3`, the commit that built the canonical 32-cubed bundle, with only
its imports changed. It keeps its own copies of the geometry and boundary
modules it shares with the backend, so building geometry cannot change the
backend.

## Fixed configuration

- Geometry: toroidal FCI geometry with the radius-dependent angular (RLP)
  topology, read from a schema-v1 artifact.
- Time advance: IMEX-SSP222. Material fluxes on all physical wall legs are
  treated by local backward Euler inside each stage. By default, each
  implicit stage also does one linearized backward-Euler coupled potential
  solve treating mu*grad_par(phi) and the homogeneous parallel current
  divergence implicitly, with every other coefficient frozen at the stage
  state; the wall current lift stays explicit.
  `implicit_current_phi_pair = false` restores the explicit pair.
- Parallel transport: FCI parallel operator, production-path characteristic
  material fluxes with support-core pairing and characteristic-SAT
  boundaries, physical-boundary-state characteristic wall law.
- E x B advection: material-scalar third-order upwind bracket (compatible
  centred bracket for vorticity).
- Boundaries: Bohm sheath on parallel legs that end on the vessel (default;
  `wall_closure = "zero-flow"` restores Vi = Ve = 0 face traces), physical-normal Neumann
  ghosts for density and temperatures, Dirichlet potential and vorticity.
- Potential solve: SOLVAX FGMRES with the line-u preconditioner.

The wall closure is a Bohm sheath entrance (see Wall closure below). An
optional diffusive neutral model (off by default) is described under Neutrals
below. No presheath model is included; recycling is optional (see Recycling below).

## Normalization

Lengths in the geometry are in metres (the artifact is SI: major radius
0.87-1.53 m). |B| is in tesla (B_ref = 1 T). n, Te, Ti are in reference
units, and time is in L_ref/c_s with L_ref = 1 m. rho* = rho_s / L_ref,
rho_s = sqrt(Te m_i) / (e B_ref); E x B and curvature drift terms scale with
rho*, and the polarization relation scales with rho*^2. The default rho* is
5e-4 (hydrogen, Te ~ 24 eV, B = 1 T; rho* scales as sqrt(Te)/B, e.g. 4.6e-4
at 20 eV). With these values c_s ~ 4.8e4 m/s, so one time unit is about 21
microseconds and t = 0.15 is about 3 microseconds. The base commit used
rho* only as a divisor of the E x B terms; this is the same operator at
rho* = 1.

## Wall closure

Every parallel leg whose FCI endpoint is a physical wall (forward legs with
`b.n_w > 0`, backward legs with `b.n_w < 0`, `n_w` the outward wall normal)
gets a complete wall state `(n, Te, Ti, Vi, Ve)` that the existing wall path
consumes: the per-stage local backward Euler of the wall legs, the explicit
residual and the wall current exported to the vorticity equation. Code:
`native/fci_wall_closure.py`, `LocalFciDrbEBRhs.sheath`.

`wall_closure = "bohm"` (default), with `s = sign(b.n_w)` (+1 forward, -1
backward) and velocities in `c_s0 = sqrt(Te0/m_i)`:

- Te, Ti: the physical-normal Neumann wall traces (unchanged).
- Bohm entrance: `Vi_w = s max(s Vi, c_s)`, `c_s = sqrt(Te_w + tau Ti_w)`
  (isothermal electrons and ions).
- Floating wall, local ambipolar form: `Ve_w = Vi_w`, so the current through
  every wall leg is zero. Zero net current per connected wall region (which
  allows local currents through the wall) is not implemented yet.
- Entrance density from the ion momentum flux over the half leg,
  `n_w (Vi_w^2 + c_s^2) = n (Vi^2 + Te + tau Ti)`; for a stagnant owner this
  is the isothermal two-point result `n_t = n_u/2`. The Neumann density trace
  is not used: with it the conservative momentum row below is unstable.
- One flux per quantity. With the sheath, the bulk parallel density and
  thermal-energy rows are written in flux form with the weighted support pair
  (`LocalFciDrbEBRhs._sheath_conservative_parallel_rates`): `D`, the weighted
  negative adjoint of the constant-exact gradient `G`, satisfies
  `sum_i V_i D(f)_i = 0` exactly, and

  ```
  dn/dt   = -D(n Ve)
  dW_e/dt = -D(5/2 pe Ve - 0.71 Te j) + Ve G(pe) - 0.71 j G(Te)
  dW_i/dt = -D(5/2 pi Vi)             + Vi G(pi) + 5/2 Ti D(j)
  ```

  (`W = 3/2 p`, `j = n (Vi - Ve)`; the same five-field equations as the
  characteristic matrix, regrouped, including the `div b` terms). The first
  terms only move particles and energy between rows; the second are the
  parallel work and current exchange, reported as volumetric sources `S_e`,
  `S_i`. The characteristic material rows are not used for n, Te and Ti here:
  on FCI maps a row's legs end on interpolated points, not on its neighbours'
  legs, so their fluxes do not telescope, and at HSX wall cells the mismatch
  was about 40% of the wall loss.
- The wall legs then carry the wall flux alone: the (n, Te, Ti) rows of a leg
  are `-s F_w / L` (`sheath_wall_leg_residual(..., conservative_bulk=True)`),
  with the ion velocity (`Ve_w = Vi_w`): particles `n Vi`, energies
  `gamma n T Vi`. The Vi row is in flux form too (see Sheath potential and
  the momentum row below); its leg is `-s (F_w - p_c)/L`. These rows replace the
  characteristic rows in the local backward-Euler block, with their own exact
  local Jacobian (traces frozen); the Ve row keeps the characteristic law.
- Flux-tube length `L` (`sheath_tube_lengths`): the length of the owner's
  interior leg, so the wall flux crosses the cross-section `A = V / L` that
  the bulk gives the same row (both legs on walls: `L = dx_minus +
  dx_plus`). The wall-leg length `dx_wall` is the distance to the wall, often
  a fraction of the cell (half a cell for a wall at the cell face in 1D), and
  `V / dx_wall` overstated the HSX particle loss by a factor of 2.7.
- Heat transmission convention: `gamma_e` and `gamma_i` are total Stangeby
  coefficients at M = 1, `q_e = gamma_e n Te c_s`, `q_i = gamma_i n Ti c_s`
  (energy in `n0 Te0 c_s0`, the ion one multiplied by tau). The wall removes
  exactly `gamma n T c_s` from the thermal energy and nothing is counted
  twice: in the flux form the bulk enthalpy flux ends on the owner row, and
  in the characteristic form (`conservative_bulk=False`, used by the 1D
  checks) the leg replaces the centre enthalpy `(5/2) n T V` by the wall flux. The momentum flux is continuous over
  the half leg, so the flow is not re-accelerated from the temperatures: lost
  ions take the owner's kinetic energy `Vi^2/2` with them through the Vi row.
  Defaults: `gamma_e = 2/(1-d) - 0.5 ln[2 pi (m_e/m_i)(1 + tau)/(1-d)^2]`
  (4.49 for hydrogen with `d = secondary_electron_emission = 0`) and
  `gamma_i = 2.5`.
- Incidence angle: with `sin(alpha) = |b.n_w|`, the wall surface area of a
  leg is `A_perp / sin(alpha)`, so integrated losses do not depend on alpha;
  only surface flux densities (`q sin(alpha)` on the vessel) do. The artifact
  has no vessel normals at the FCI endpoints, so surface densities are not
  reported yet.

Flux contract: `wall_face_fluxes(wall_state, s, tau, gamma_e, gamma_i,
owner_velocity)` returns, per unit cross-section normal to b and for the wall
state actually applied, the outward particle flux `n V s`, the momentum flux
`n V^2 + n (Te + tau Ti)`, `q_e`, `q_i`, the flow kinetic energy
`(Vi_owner^2/2) n V s` and the current `n (Vi - Ve) s`. The wall-leg rows
remove exactly these. `LocalFciDrbEBRhs.wall_flux_ledger` integrates them
over all wall legs with `A = V_cell / L` and returns the particle, thermal and
ion kinetic inventories on the same cell measure and the sources `S_e`,
`S_i`. `LocalFciDrbEBRhs.wall_step_ledger` gives what one IMEX step actually
applied (`next = current + dt (implicit + explicit)`): the wall part is the
implicit increment of the wall-leg block, the rest is the explicit rate, and
`dN = other - wall`, `dW = other - wall + (3/2) dn dT` hold to round-off.
With the sheath on, `drbx run` prints a `[wall]` line (contract at the state)
and a `[wall-step]` line (applied over the step) per step and stores both as
`wall_ledger` (`wall_ledger_columns`) in the history file.

`wall_closure = "zero-flow"` keeps the historical Vi = Ve = 0 wall traces;
the advance is then bitwise the code without the closure
(`scripts/fci_gate.py record OUT --hsx-set wall_closure='"zero-flow"'`).

Deck keys (and `--` options): `wall_closure`, `electron_heat_transmission`
(0 selects the floating-wall value), `ion_heat_transmission` (>= 2.5),
`secondary_electron_emission`, `sheath_potential` (`"ambipolar"` or
`"current"`).

Checks (`tests/fci_braginskii/test_fci_wall_closure.py`): the wall-leg rows
remove exactly the contract fluxes (algebraic identity); on a 1D flux tube
driven by the production material operator with a uniform source, the steady
state has a sonic target, `n_t = n_u/2` within 3%, the particle ledger closed
to 2e-3 and heating = thermal sheath losses + flow kinetic energy + source
dissipation to 1%. The isothermal profile converges to the exact solution at
an observed order of about 0.55 (10, 20, 40 cells), limited by the
square-root singularity at the sonic target. No manufactured-solution test
with sheath boundaries exists yet.

HSX (32-cubed bundle, defaults, 20 steps to t = 0.015, about 77 GMRES
iterations per solve, 4.6 s per step): 1848 wall legs; the contract losses go
from 0.205 to 0.187 particles, 0.919 to 0.757 (electron) and 0.511 to 0.452
(ion) energy per unit time, the wall current is zero, n stays in [0.857, 1.049]
and the particle inventory falls by 0.40%. Ledger per step:

- Inventory change against the applied fluxes: `dN - (other - wall)` and the
  energy equivalents are 1e-12 of the wall loss or less.
- Particles: `other`, everything that is not the wall, is the perpendicular
  transport through the radial boundary and grows to 9e-5 of the wall loss;
  the parallel bulk contributes nothing (`sum V D(n Ve)` is 1e-18).
- Energy: `other` is 0.25% of the wall loss and follows the parallel work
  sources `S_e dt`, `S_i dt` (plus perpendicular transport).
- The applied wall flux equals the contract flux of the step states to first
  order in dt: 6e-4 (particles), 3e-3 (electron energy), 1e-3 (ion energy) at
  dt = 7.5e-4, and 1.4e-4, 7e-4, 2.6e-4 at dt/4.

Before this form (the ion-velocity legs with a centre flux over `dx_wall`
against the characteristic bulk), the inventory fell 1.38 times faster than
the contract loss (energy 1.36 times), at any dt.

### Sheath potential and the momentum row (PR 1c)

`sheath_potential` selects the electron wall closure:

- `"ambipolar"` (default): `Ve_w = Vi_w`, zero current on every wall leg, phi
  wall boundary Dirichlet 0 (the form above).
- `"current"` (experimental): the sheath-limited current is a boundary
  condition of the implicit current/phi solve
  (`solve_implicit_current_phi_pair`). Per wall leg, along +b,

  ```
  j_w(phi) = s n_w c_s [1 - exp(Lambda - phi/Te_w)]
  Lambda   = ln[(1 - d) / sqrt(2 pi (m_e/m_i)(1 + Ti_w/Te_w))]
           = 0.5 ln[m_i / (2 pi m_e (1 + tau Ti_w/Te_w))] + ln(1 - d)
  ```

  with the Bohm entrance `n_w`, `c_s`, the local `tau Ti_w/Te_w` and the
  secondary-emission yield `d` (`secondary_electron_emission`). Hydrogen:
  2.84 at `Ti = 0`, 2.49 at `Ti = Te`. (PR 1b quoted 2.84 for `Ti = Te`;
  that is the `Ti = 0` value.) `phi` is the owner-cell potential, so the wall
  is a conducting wall at `phi = 0` and `j_w = 0` at the floating potential
  `phi = Lambda Te_w`. The closure is local per leg; per-region floating
  (zero net current over each connected wall region with a free wall
  potential, a Lagrange multiplier in the solve) is not implemented.
  Linearized about the stage potential `phi_b`, `j_w = [j_w(phi_b) - J
  phi_b] + J phi` with `J = dj_w/dphi = s n_w c_s exp(Lambda - phi_b/Te)/Te`.
  Writing `W(e)` for the divergence of the wall endpoint current `e` alone,
  the affine part `h kappa W(j_w(phi_b) - J phi_b)` goes into the predicted
  vorticity (right-hand side) and the homogeneous part `(h/rho*^2) P kappa
  W(J E phi)` into the potential operator, where it is a positive diagonal
  wall term. The characteristic wall-current lift of the explicit vorticity
  RHS is zero in this mode (no second wall current), and the mode requires
  `implicit_current_phi_pair`. The exponent is capped at 40. The heat
  transmission `gamma_e` and the wall particle flux (`n_w Vi_w`, ions) are
  unchanged: the electron particle flux `n_w Vi_w - s j_w` is not yet used
  in the density row.

Momentum row in flux form (both modes): `d(n Vi)/dt = -D(n Vi^2 + p) + p
D(1) + Vi D(j)`, `p = n (Te + tau Ti)` (the characteristic `dVi/dt = -Vi
G(Vi) - G(p)/n` regrouped with `dn = -D(n Ve)`). The wall leg is `-s (F_w -
p_c)/L`: the wall momentum flux `n Vi_w^2 + p_w` less the leg's share `s
p_c/L` of the well-balanced mirror force (`div b = D(1) + sum_legs s/L`, so a
uniform static pressure is in equilibrium). Ledger: `dP/dt =
-momentum_wall + momentum_leg_pressure + momentum_source` plus non-parallel
terms (`wall_flux_ledger`), and the applied-step columns
`step_momentum_{wall,other,product}`.

HSX, 32-cubed, dt = 7.5e-4, 100 steps (t = 0.075):

| | ambipolar (default) | current |
|---|---|---|
| Steps | 100, finite | fails at step 47 (t = 0.0345); step 49 before the electron-flux fix |
| GMRES iterations/solve | 76 | 85-89 |
| Wall current (contract) | 0 | -2.27 at t = 0, -0.80 at t = 0.030 (electron-dominated, decaying) |
| n range at end | [0.530, 1.162] | [0.157, 1.611] at step 46 |
| Particle loss rate | 0.205 to 0.137 | 0.205 to 0.180 at t = 0.030 |
| Applied-step imbalance (n, W_e, W_i, P) | <= 1e-12 each | <= 1e-12 (n, W); P 3e-9 |

The `"current"` failure: wall-adjacent cells get pulled to `phi ~ Lambda Te`
and their parallel neighbours do not, so phi becomes grid-scale along b near
the radial edge (owner cells alternating 3 and -2, max 13.7). With `mu =
1836` and inertial electrons in the implicit stage, `Ve` reaches 1e3, and
`-D(n Ve)` empties a wall cell (n down to 0.02). The 1D flux-tube check
(`test_flux_tube_floating_sheath_potential_drop`: Ohm's-law current, cell
capacitance, linearized sheath walls, the production affine/homogeneous
split) reaches `phi_w - phi_entrance = -Lambda Te` and zero current to 1e-10
at dt = 10, where the explicit wall current diverges. That model has
Boltzmann electrons (resistive Ohm's law), not the inertial `Ve` of the
backend.

In `"current"` mode the wall leg now carries the electron flux: the density
row, the electron energy flux `gamma_e Te (n_w Vi_w - j_w)` and the
density-change terms of the Te, Ti and Vi rows use `n Ve = n Vi - j`, the ion
rows keep `n_w Vi_w`, and the transmission follows the sheath drop,
`gamma_e = 2/(1-d) + max(phi/Te, 0)` (the Stangeby value at `phi = Lambda
Te`). The 0.71 Te j thermoelectric term belongs to the bulk heat flux and is
not part of the wall leg, which uses the sheath transmission for the total
electron energy flux. With a fixed floating `gamma_e` the run failed at step
36, with `gamma_e(phi)` at step 47, so the energy leg matters but is not the
root cause: Te still collapses (to about 0) in wall-adjacent cells, where
`phi` alternates in sign at grid scale on the radial edge and `Ve` reaches
+-40. The cure for that is the wall-cell electron force (`G(phi)` against the
sheath endpoint), not another flux term, so the default stays `"ambipolar"`.

## Geometry artifact

`geometry` names a directory with `manifest.json` (schema
`drbx.fci_simulation_geometry`, version 1). The reader
(`drbx.fci_braginskii.geometry.fci_simulation_geometry`) loads the base
geometry, centre maps, RLP topology and cell positions, and ignores the
artifact's newer overlap, vertex-reconstruction, endpoint-field and curvature
edge one-form components. Resolution, topology and producer metadata come from
the artifact.

The canonical 32-cubed bundle uses 64 trace substeps and a 64-cubed
coordinate-fit sample, unlike the historical 48-cubed, four-substep runs, so it
tests the old plasma model on canonical geometry rather than replaying a
historical run.

The bundle is not in the repository; obtain it from the maintainers and place
it at `artifacts/geometry/hsx_fci_32x32x32` (`artifacts/` is ignored by git).
The manifest records a SHA-256 for every file. To check a copy:

```bash
python -c "from drbx.fci_braginskii.geometry_build.fci_simulation_geometry import audit_fci_simulation_geometry_checksums as a; print(a('artifacts/geometry/hsx_fci_32x32x32')['valid'])"
```

## Building geometry

The `[geometry]` deck table (`build_hsx_simulation_geometry`) builds a bundle from the HSX vacuum field and
vessel. The default build needs only files shipped with DRBX (see "Building
from coils" below). The mgrid route needs the MAKEGRID file, which is not in
the repository and is obtained from the maintainers; `makegrid_path` and
`vessel_path` override the paths.

| File | Size | SHA-256 |
|---|---|---|
| `mgrid_res2p5cm_180pln.nc` (MAKEGRID vacuum field) | 5,831,797,532 bytes | `45dd643a4ae3930386b8f8bff7a789c796a0d5174b5209e1202b5bfd4ac87e44` |
| `vessel_hsx_flare.txt` (vessel outline) | 909,289 bytes | `65d794b7131e35dc72b554e5b6ae3c97974a5e11a7d936ebedc66451749d1ff3` |

The canonical bundle was built on Perlmutter with a vessel file of the same
length but a different SHA-256 (`b5ad2a36...`), so a rebuild from these inputs
is close to the canonical bundle but not identical to it (see below). The
default coil
currents are the QHS configuration (10722 A in the first six MAKEGRID coil
groups, the main coils, and 0 A in the other six); `makegrid_currents`
changes them.

The canonical 32-cubed settings:

```toml
[geometry]
resolution = [32, 32, 32]
metric_mesh_shape = [32, 32, 32]
fit_sample_shape = [64, 64, 64]
toroidal_modes = 10
metric_toroidal_modes = 3
include_curvature_edge_one_form = true
metric_cache_dir = "/path/to/build/metric_cache"
cache_dir = "/path/to/build"
```

The builder runs in stages (metric fit and cell maps, vertex-trace atlas,
angular owner geometry, owner-boundary overlap, artifact write). Completed
stages are checkpointed in the output's sibling
`.NAME.producer-checkpoints/` directory, so an interrupted build resumes.
The bundle is written atomically after its structural checks pass. Field-line
tracing uses JAX by default and shards the trajectory batch across all local
devices (`trace_device_count`, `trace_batch_size`); `--trace-backend numpy`
is the reference implementation.

On an 8-core M1 the canonical 32-cubed build takes about 17 minutes with a
3.7 GB peak; field-line tracing of the cell-centre maps takes about 3.5 minutes
and the owner-overlap stage about 4.5 minutes. The overlap stage grows quickly
with resolution: on Perlmutter it peaked at 29 GiB for 48-cubed and 65 GiB for
64-cubed. A 32-cubed rebuild from the inputs above differs from the canonical
bundle by 1.4 to 3.5 mm in cell positions and by up to 0.7% in |B|, most likely
because of the vessel file; the backend runs on either.

### Building from coils (default)

The HSX coil file ships with DRBX at `examples/inputs/hsx/coils.hsx`
(MAKEGRID format, 12 coil groups, 4 field periods; SHA-256
`0bb6cf70cf1696489ec6b67bfadc0a60b128fec12f8cfdf95aa2982482eb13fe`), the
vessel outline at `examples/inputs/hsx/vessel_hsx_flare.txt` (SHA-256 above),
and
`[geometry]` uses it by default: B is evaluated from these
filaments (the ones the mgrid was made from) with the exact straight-segment
Biot--Savart law, at the QHS currents
(`10722,10722,10722,10722,10722,10722,0,0,0,0,0,0`) unless
`makegrid_currents` says otherwise. `coils_path` selects another coil
file; `makegrid_path` switches back to an mgrid field. In Python,
`build_hsx_fci_geometry(bfield_provider=...)` accepts any
`drbx.geometry.field_provider.MagneticFieldProvider` (ESSOS coils,
`FilamentCoilField`), so the field is a differentiable function of the coil
parameters.

By default `coils` does not evaluate Biot--Savart at every trace step. It
tabulates the coils once, in memory, on a uniform `(R, phi, Z)` grid over one
field period (`coil_table_shape`, default `[96, 144, 96]`;
stellarator symmetry halves the evaluations) and interpolates the table with
the same cubic B-spline the mgrid path uses (`CoilFieldTable` in
`provider_bfield.py`). The table and its spline coefficients are JAX arrays
computed from the coil parameters, so derivatives still reach the coil
currents and nodes. At the default shape the table differs from direct coil
evaluation by a median 1e-7 (p99 5e-6) relative error at the bundle cell
centres, below the coil-vs-mgrid difference, and costs about 2 minutes on a
laptop. `coil_table_shape = None` (Python) evaluates the coils at every step instead (about 25x
slower tracing).

## Running

From the repository root:

```toml
[fci_braginskii]
geometry = "artifacts/geometry/hsx_fci_32x32x32"
save_every = 10
checkpoint_every = 50
diagnostic_every = 10
filament_cache_dir = "/path/to/run/initialization_cache"
output = "/path/to/run/history.npz"
```

```bash
drbx run deck.toml
```

`drbx run examples/inputs/hsx_fci_blob.toml` runs a deck (`drbx inspect` prints
the resolved configuration); omitted keys take the defaults and unknown keys
are errors. It calls `drbx.fci_braginskii.run.run`, print one
progress line per step (step, time, timings, ETA) and write the history to
`output`.

The defaults run to t = 0.15 in 200 steps (dt = 7.5e-4) at rho* = 5e-4 with
the implicit current/potential pair. With the explicit pair this rho* limits
dt to about 1e-4. With the implicit pair, 32-cubed runs are stable at dt =
1.5e-3 (tested to t = 0.3) and fail at 5e-3, where the explicit electron
parallel terms set the limit. Runs at dt = 1.5e-3, 7.5e-4 and 3.75e-4 to
t = 0.15 converge at about order 1.5 in dt. At the default step the
estimated error is 0.1% or less (relative L2) for density and temperatures,
and the blob centroid is converged to 1e-5 m. The potential is 2 to 5% off
in amplitude and about 13% off in relative L2. That error is smooth and sits
in the bulk around the filament, not at the walls. Ve behaves the same way,
and vorticity converges more slowly. Where the potential matters, use more
steps: 400 steps (dt = 3.75e-4) bring the potential error to about 5%.

Measured temporal order (HSX 32-cubed, t = 0.003, dt = 7.5e-4 down to dt/8,
tight GMRES tolerances; investigation in PR #28):

| Field | Order | Cause of the reduction |
|---|---|---|
| vorticity | 1.8-1.9 | none (design order) |
| n, Te, Ti, Vi | 1.4-1.6 | one Gauss-Seidel pass between the wall backward-Euler block and the current/potential pair per implicit stage; converging the stage gives about 1.8 at roughly twice the GMRES cost |
| phi | 1.0-1.2 | stiff electron relaxation (mi/me = 1836): phi acts as an algebraic multiplier and the stage-order-1 implicit scheme loses order; with mi/me = 100 phi converges at 1.6-1.9 and with 10 at 1.9-2.0 |
| Ve | 1.3-1.6 | same mechanism as phi |

The order is not set by the IMEX tableau (SSP2, ARS(2,2,2) and a stiffly
accurate stage-order-2 ARK2 give the same orders), by GMRES tolerance, by the
wall closure (no clipping or flux fallback fires) or by the initial state.
Second-order phi at the physical mass ratio needs a higher-stage-order implicit
treatment of the current/potential block; until then, refine dt for
potential-sensitive observables as above.

The driver imports `drbx` from this repository's `src`; `DRBX_SOURCE_ROOT`
overrides that. Set `DRBX_CACHE_DIR` to a writable, run-local JAX compilation
cache so repeated runs skip compilation.

Remaining options cover time and output cadence, snapshots and restarts, the
initial filament (`--blob-*`, `density_amplitude`), physical parameters
(`tau`, `rho_star`, `mi_over_me`, `perp_diffusion`,
`parallel_diffusion` (default 0), `electron_collision_frequency`) and GMRES
tolerances; see [Deck Reference](decks.md).

`advance_execution` selects how each step runs:

- `staged-compiled` (default): four reusable kernels (implicit stage with
  potential solve, explicit right-hand side, standalone potential, stage
  diagnostics). Fastest on the N32 case with both cold and warm caches.
- `compiled`: one fused IMEX step.
- `eager`: no outer `jax.jit`; slow, but Python-level debugging and printing
  work inside the right-hand side.

Phase timing, which reports the operator and GMRES shares of each step, is on
by default; `no_phase_timing` disables it. At N32 with the defaults the
potential solves take a little over half of each step (about 90 GMRES
iterations per solve on average, four solves per step: two coupled
current/potential solves and two standalone potential solves), and a step
takes about 16 s on an 8-core M1.

Multi-device runs decompose only the toroidal (eta) direction:
`shard_counts = [1, 1, N]`, with N dividing the eta resolution; other splits are
rejected. On CPU, emulate N devices with
`XLA_FLAGS=--xla_force_host_platform_device_count=N` (the driver does not read
`DRBX_HOST_DEVICE_COUNT`). Every flag in `XLA_FLAGS` needs its `--` prefix: a
bare token such as `intra_op_parallelism_threads=4` stops XLA from parsing the
flags after it, and the run falls back to one device.

## Example script

`examples/stellarator/hsx_fci_blob_render.py` runs the same driver and renders
the result. Its parameters are module-level constants at the top of the file
(geometry bundle, final time, steps, save cadence, output directory). From the
repository root, with the bundle at `artifacts/geometry/hsx_fci_32x32x32`:

```bash
PYTHONPATH=src python examples/stellarator/hsx_fci_blob_render.py
```

The driver prints the geometry and model setup, then one progress line per
step (progress bar, step, time, step/operator/GMRES timings, ETA, density
range) plus per-step field diagnostics, and finally where the history was
written. Everything goes to `output/hsx_fci_blob/`: `history.npz`,
`hsx_fci_blob.png` (3-D context and R-Z sections) and `hsx_fci_blob.gif`.
The default is a quick 5-step preset to t = 0.00375 (about two minutes on an
M1 laptop including compilation), not the converged case; the figures in
`docs/media/hsx_fci_blob.*` use FINAL_TIME = 0.15, NUM_STEPS = 200,
SAVE_EVERY = 10. Set `RUN_SIMULATION = False` to re-render an existing history.

## Neutrals

`neutrals = true` (deck key, `--neutrals`) adds one evolved field, the
atomic neutral density `N`, on the plasma owner cells. Code:
`native/fci_neutrals.py`; with `neutrals = false` (default) no neutral code
runs and the advance is bitwise unchanged (`scripts/fci_gate.py`).

Model choices:

- **Diffusive, prescribed temperature.** At HSX parameters a fluid neutral
  momentum equation adds stiff fields and a sound-speed limit without a
  closure that the diffusive model lacks. The neutral temperature is fixed,
  `T_n = neutral_temperature_ev` (default 3 eV, Franck-Condon atoms), and the
  neutral flow is the diffusive drift. The energy and momentum that the
  reactions hand to the neutrals beyond `3/2 T_n` per atom go to a neutral
  reservoir that is reported (`neutral_reservoir_energy`,
  `neutral_reservoir_momentum`), so the energy budget stays closed.
- **Isotropic diffusion with the full metric.** Neutrals are not magnetized:
  `dN/dt = (1/J) d_i (J D g^ij d_j N)`, a finite-volume flux on every fine
  logical face (two-point normal term plus averaged cross derivatives,
  face metric from the artifact), summed over the angular aggregates and
  divided by the owner volume. No FCI map is used. Faces inside an aggregate
  cancel, so `sum V dN/dt` is zero to round-off. The axis and the vessel
  (`u = 1`) faces carry no flux (reflecting wall) until recycling supplies a
  wall influx (PR 3; `wall_influx` argument).
- **Diffusivity.** `D = v_th^2 / (nu_cx + nu_iz)`, `v_th^2 = T_n/m_i`,
  `nu_cx = n <sigma v>_cx(T_i + T_n)`, `nu_iz = n <sigma v>_iz(T_e, n_e)`, with
  the mean free path `lambda = v_th/nu` capped at `neutral_mfp_max` (default
  5 cm): `D = v_th min(lambda, lambda_max)`. The cap is a flux limiter for
  the low-density regions where `lambda` exceeds the device; where it is
  active the diffusive closure is not valid (see the Knudsen map).
- **Reactions.** `drbx.native.neutrals.compute_hydrogen_reaction_sources`
  (AMJUEL H.4/H.10 2.1.5 and 2.1.8, H.2 3.1.8; January 2020 report, the
  same code and data as the 1D model): ionisation, recombination and charge
  exchange, receiver mixing heat, CX frictional heat, and the electron
  energy loss `<sigma v E>` with the 13.6 eV recombination add-back
  (`energy_loss_coefficient`). Rates use `n0 = neutral_reference_density_m3`
  (default 5e18 m^-3; the plasma model itself is normalized) and
  `Te0 = (rho* e B_ref L_ref)^2 / (m_i e)` (23.9 eV at rho* = 5e-4). The
  physical ion temperature is `tau Ti`. Inputs outside the fit range
  (`T` in [0.1, 1e4] eV, `n_e` in [1e14, 1e22] m^-3) are clamped by the
  evaluator; the step counts the owner cells outside the range for `T_e`,
  `n_e` and the CX temperature and prints them.
- **Source hook.** `apply_volumetric_source(n, Vi, Te, Ti, source, dt)`
  takes a `VolumetricSource` (particles, parallel momentum, ion and electron
  thermal energy per volume) and updates the conserved variables `n`,
  `n Vi`, `3/2 tau n Ti + n Vi^2/2` and `3/2 n Te` (the ion row also gets the
  kinetic work `Vi S_M - Vi^2 S_n/2`), then recovers the primitives. Ve and
  the vorticity are not changed (electron momentum transfer is O(m_e/m_i)).
  Recycling and any external source can use the same hook.

Time stepping: Lie operator split after each IMEX-SSP222 step: the reaction
sources are advanced explicitly (`neutral_reaction_substeps`, default 1;
the step prints `max dt nu`, which was 4.5e-3 on HSX), then the diffusion
is backward Euler (preconditioned GMRES, Jacobi), followed by the
conservative update `N_new = N + dt L N*`, so the neutral inventory is exact
regardless of the GMRES residual. The split is first order in dt, like the
wall ledger. It does not touch `evaluate_stage` or the implicit current/phi
solve. The potential is not re-solved after the split: the next step's
stage potentials are reconstructed from the updated state.

Ledger: each step prints `[neutrals]` (event counts `sum V I dt`, `R`, `C`,
the electron energy loss, the reservoir energy, the neutral inventory, the
nuclei inventory `sum V (n + N)`, and the split identities: relative change
of the nuclei and `(dE_plasma + dE_neutral + sinks)/sinks` over the split).
With the sheath, the `[wall-step]` imbalance removes the split's inventory
change. The history stores `neutral_density` (per saved frame),
`neutral_ionisation_rate` (last step), `neutral_ledger` and its column
names, and the Knudsen fields below.

Knudsen map: `Kn_cell = lambda / h` (`h` the smallest physical cell size,
`sqrt(g_ii) d xi_i`) and `Kn_grad = lambda / min(L_N, L_n)` with
`L_X = X / |grad X|` (full metric), stored as `neutral_knudsen_cell`,
`neutral_knudsen_gradient`, `neutral_mean_free_path`. The diffusive closure
is valid where `Kn_grad << 1`. The run prints the fractions of cells below
0.1, between 0.1 and 1, and above 1.

Deck keys (and `--` options): `neutrals`, `neutral_reference_density_m3`,
`neutral_temperature_ev`, `neutral_mfp_max`, `neutral_initial_density`
(at the wall, units of n0), `neutral_initial_decay_length` (initial profile
`N0 exp(-(1-u)/L)` in the radial label, 0 = uniform),
`neutral_reaction_substeps`.

Checks (`tests/fci_braginskii/test_fci_neutrals.py`, on a sheared slab with a
constant non-orthogonal metric and an annulus with a varying metric and a
cross term):

- constant `N` with a strongly varying `D` is an exact null of the
  diffusion, and stays uniform through the implicit step;
- `sum V dN/dt = 0` (reflecting wall) and the implicit step changes the
  inventory by exactly the supplied wall influx;
- with a diagonal metric the operator is symmetric negative semi-definite;
- the reaction hook: per-cell energy identity of the transfer model
  (`Q_i + Vi S_M - Vi^2 S_n/2 + Q_n = 0` with stationary neutrals), exact
  particle, momentum and energy updates;
- closed box (five steps, three reaction substeps): nuclei to 1e-14 and
  energy (plasma thermal + kinetic + neutral thermal + electron loss +
  reservoir) to 1e-12 of the sinks;
- pure ionisation at fixed `Te` follows the logistic solution (handoff 6.3)
  within the spread of the density-dependent AMJUEL coefficient;
- 1D neutral penetration `D N'' = nu N` with a wall influx on the sheared
  slab against `N = Gamma cosh(k s) / (D k sinh(k L))`, `k^2 = nu/D`: error
  below 2e-3 at 64 cells, second order (ratio > 3.5 from 32 to 64);
- slow, HSX bundle: two steps with sheath and neutrals, split and wall
  ledgers close.

HSX (32-cubed bundle, sheath on, defaults plus `neutrals = true`: n0 =
5e18 m^-3, T_n = 3 eV, initial `N = 0.01 exp(-(1-u)/0.1)`, 60 steps to t =
0.045, 4.4 s per step, GMRES 76 iterations, the neutral step about 0.1 s):

- Stable: n in [0.66, 1.08], `N > 0` everywhere (minimum 1.2e-4 at the
  end), GMRES relres 1e-8 at every solve.
- Split identities at every step: nuclei 3e-16, energy 9e-11 of the sinks.
  The wall-step imbalance (split removed) stays below 1e-11 of the wall
  loss (measured: particles 2.0e-12, W_i 1.2e-12, W_e 9e-13, P 2.7e-13 over
  100 steps with neutrals and recycling). The bar is 1e-11 rather than
  1e-12 because the imbalance is a difference of O(1) inventories, so it
  sits at round-off of the inventory, not of the wall term. No rate input left the AMJUEL range.
- The neutrals spread inward (`sqrt(D t)` about 3 cm): the radial mean of `N`
  goes from 3.4e-3 (edge) / 0 (axis) to 2.4e-3 / 1.6e-4. The neutral
  inventory falls 10% (ionisation 2.2e-6 to 2.5e-6 per step against CX
  2.8e-6 and recombination 1.6e-9).
- Ionisation is edge-localized: 28% of it in the outer 4 of 32 radial cells
  and 54% in the outer 8.
- Neutral energy sinks (electron loss 1.5e-4, reservoir 2.4e-4 over the
  run) are below 1% of the sheath energy loss; the plasma evolution is still
  sheath-dominated.
- Knudsen map: the CX/ionisation mean free path is 5.5 to 14 cm, larger than
  the 5 cm cap almost everywhere, so `D` is the flux-limited value. Against
  the profile scale lengths, no cell is diffusive (`Kn_grad < 0.1`), 37% are
  transitional and 63% kinetic (`Kn_grad >= 1`); `lambda/h` is 6 to 185. At
  HSX edge densities the diffusive closure is therefore **not** valid: the
  model is a mean-free-path-limited reduced closure, useful for the ledger,
  the coupling and recycling, not a quantitative neutral prediction. Since
  `lambda ~ 1/n`, `Kn_grad < 0.1` would need roughly ten times the density
  at these profiles (an estimate, not run); a kinetic reference (EIRENE) is
  the validation path (handoff 6.5).

Not included: restart (the checkpoint has no neutral field; `restart_from`
with neutrals is rejected), recycling (see Recycling below), molecules, neutral momentum,
a neutral energy equation, and kinetic neutrals.

## Recycling

`recycling = true` (deck key, `--recycling`) returns a fraction `R` of the ion
particles lost at the sheath as neutral atoms. It needs `neutrals = true` and
the Bohm wall closure (anything else is rejected at validation). Code:
`native/fci_neutrals.py` (`RecyclingParameters`, `recycled_neutral_source`,
`RECYCLING_LEDGER_COLUMNS`); with `recycling = false` (default) the run
traces the same code as without it.

- **Source.** `S = R x (-implicit.n)` of the IMEX step: the wall ion loss
  actually applied to each wall owner cell. Under the sheath the implicit
  density row is the area-weighted wall-leg flux alone, the increment
  `wall_step_ledger` reports. The atoms are deposited in the same owner
  cells (not as a radial boundary influx of the diffusion), and enter the
  conservative backward-Euler update as `volume_source`, so
  `sum V dN = dt sum V S` exactly. The same amount is added to the
  `wall_influx` column of the neutral ledger.
- **Energy.** `recycling_energy = "franck-condon"` (default, 3 eV per atom,
  `recycling_franck_condon_energy_ev`) or `"wall"` (`3/2 T_wall`,
  `recycling_wall_temperature_ev`). The `3/2 T_n` part of the recycled
  energy is accounted in the split's energy check; the remainder goes to
  the reservoir.
- **Ledger.** Each step prints a `[recycling]` line and stores
  `recycling_ledger`: wall ion loss, recycled particles, recycled energy,
  the `3/2 T_n` part, reservoir remainder, nuclei, d(nuclei) and the expected
  `step_particle_other - (1-R) wall`. The particle and energy checks of the
  split subtract the recycled particles and their `3/2 T_n`.
- Deck keys: `recycling`, `recycling_coefficient` (default 0.99),
  `recycling_energy`, `recycling_franck_condon_energy_ev`,
  `recycling_wall_temperature_ev`.

Checks (`tests/fci_braginskii/test_fci_recycling.py`): closed box at R = 1
and 0.9 on the slab and the annulus (`d(n+N) = -(1-R) wall` to 1e-13, energy
identity with the recycled `3/2 T_n`, off equals the unchanged split); the
1D flux-tube high-recycling trend (target density rises and target
temperature falls for R = 0, 0.5, 0.8, with `(1-R) wall` equal to the core
source within 1%; the flux-tube relaxation time grows like `1/(1-R)`, so the
test integrates `30/(1-R)` periods); and, with the HSX bundle, a 3-step
ledger at R = 1 and 0.9 that also confirms that the owner cell volumes of
`wall_step_ledger` equal the neutral grid volumes.

Recycling in a neutrals-only model is a closure for the particle and energy
ledger and the coupling; the neutral model keeps the validity limits stated
under Neutrals (diffusive closure outside its Knudsen range).

## Known limitations

- The largest relative GMRES residual in the first N32 step is about 8e-4 at
  the default tolerances, above the 5e-5 acceptance tolerance; the driver logs
  it as `gmres-relres`.
- Under the physical-boundary-state wall law, an isolated stiff two-wall row
  (the historical hotspot) is admissible, but its selected parallel material
  residual is not sign-restoring on its own.
- A short successful evolution does not qualify wall budgets, MMS accuracy or
  long-time stability.
- With the default Bohm sheath the floating wall is local (zero current per
  leg) and the phi wall boundary is Dirichlet zero. The sheath-current wall
  (`sheath_potential = "current"`) is experimental and fails on HSX at step 47
  (see Wall closure).

## Tests

```bash
DRBX_TEST_GEOMETRY_BUNDLE=/path/to/hsx_fci_32x32x32 python -m pytest -q tests/fci_braginskii
XLA_FLAGS=--xla_force_host_platform_device_count=4 DRBX_HOST_DEVICE_COUNT=4 \
  python -m pytest -q tests/fci_braginskii
```

The second run also covers the multi-device halo exchange. The builder tests
live in `tests/fci_braginskii/geometry_build` and use synthetic or mocked
inputs; the few that read the real field or vessel run only when
`DRBX_HSX_MGRID` and `DRBX_HSX_VESSEL` point at those files.
