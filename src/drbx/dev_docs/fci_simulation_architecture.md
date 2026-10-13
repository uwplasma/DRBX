# FCI Simulation Architecture

[`generate_hsx_fci_geometry.py`](../../../generate_hsx_fci_geometry.py)
constructs and qualifies HSX geometry.  [`simulate_hsx_blob.py`](../../../simulate_hsx_blob.py)
only loads an explicitly selected geometry artifact and advances the
seven-field electrostatic Boussinesq model through the local/sharded FCI
stack. This document records the currently selectable runtime architecture.

## Driver pipeline

```text
producer: MAKEGRID + vessel + requested resolution
  -> magnetic, eta, wall, and metric evaluators
  -> global FciGeometry3D sampled on the PDE grid
  -> compiled, trajectory-batch-sharded 64-substep cell-center maps
     and complete raw transverse-vertex atlas
  -> polar owner topology and trace-free lean owner-boundary overlap graph
  -> producer qualification and atomic directory artifact
consumer: explicit --geometry directory
  -> ShardedFciGeometry3D and local halo topology (eta-only production sharding)
  -> LocalFciDrbEBRhs
  -> classical four-stage RK4 advance
  -> history and atomic snapshot NPZ files
```

The consumer never fits metrics, traces field lines, builds owner overlap,
searches caches, regenerates missing arrays, or invokes producer validation.
It trusts the selected artifact; choosing the correct qualified artifact is
the caller's responsibility. Checksums in the manifest are audit metadata,
not a discovery or fallback mechanism.

The native model is
[`native/fci_drb_EB_rhs.py`](../native/fci_drb_EB_rhs.py). Geometry lowering
is in [`native/fci_sharding.py`](../native/fci_sharding.py), and scalar halo
rules are in [`native/fci_halo.py`](../native/fci_halo.py).

## Te-only conduction experiment

The workspace driver [`run_hsx_sanity.py`](../../../run_hsx_sanity.py) has a
separate heat-spot experiment that advances only the electron-temperature
parallel conduction lane. Its `--conduction-reconstruction` option defaults
to `legacy`, which keeps the mapped fine-cell sampler. The opt-in
`aggregate-quadratic` value prepares a host-side degree-two evaluator with 32
initial donors and adaptive expansion through 96 donors, then supplies that
prepared data to the native FCI diffusion runtime. On eligible ordinary
two-sided interior rows, the runtime replaces the source and endpoint field
values; native coefficient, connection-length, and physical-wall row handling
remains in force elsewhere. This option is accepted only by the `heat-spot`
conduction experiment with angular RLP enabled; the seven-field model and
`--no-rlp` path reject it. Geometry and reconstruction metadata are written
with the run output so the experimental choice is reproducible.

The selectors `conservative-quadratic` and
`conservative-quadratic-volume` use the same quadratic H prolongation before
with fixed degree two and same-eta donor fits. The first returns the fine RHS
with H's mass adjoint; the second uses ordinary volume restriction as a local
control. H reproduces the configured owner averages, but neither selector
claims native FCI mass conservation or monotonicity for the complete mapped
operator. These are conditional conduction experiments and do not activate H
in unrelated production operators.

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
| Phi preconditioner | `none`, `jacobi`, line variants | `none`, owner `jacobi`, or angular-tree `line-u` |

The compatible Poisson bracket uses antisymmetrized shared-face fluxes and
includes the RHS `1/B` factor. Conservative curvature uses shared face
coefficients. Operator-specific physical-wall traces are supplied when an
operator needs a scalar value or flux.

On toroidal HSX geometry, the optional curvature edge one-form is selected at
geometry-production time and stored in the artifact. The producer samples the
continuous covariant one-form `b/B` once on each unique logical edge before
applying the same compatible incidence curl. This path constructs
the full-torus face coefficients on the host, packs both faces of every cell,
and then lowers those invariant channels to eta shards; shared shard-interface
faces therefore remain identical. Axis projection and physical-boundary edge
patching are unchanged, and the resulting `Q = J K/B` retains discrete
`div(curl)=0` closure to roundoff.

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

The production driver uses eta-only sharding for every topology. The valid
decomposition is:

```text
--shard-counts 1 1 Seta
```

where `Seta >= 1` and the eta resolution is divisible by `Seta`. Radial and
poloidal (`x` and `theta`) coordinates are not sharded. This is a production
contract, not merely a current test limitation.

The artifact producer requires:

- an explicit resolution and output directory;
- even `Ntheta`;
- compatible-flux Poisson bracket;
- conservative curvature;
- `none`, owner-`jacobi`, or `line-u` phi preconditioning.

The producer creates the metric-aware angular owner profile. In
toroidal RLP, each owner aggregate is confined to a single eta plane, so the
owner prolongation and physical-volume restriction remain local on an eta
shard. Global reductions for owner-space means, compatibility, norms, and
GMRES residuals still span all eta shards. The `line-u` preconditioner keeps
local radial trees; its eta-face diagonal assembly includes contributions from
both faces of every local eta slab, including the faces at slab interfaces.
There is no x/theta RLP decomposition and no fallback to a single-device
geometry, a full-grid toroidal phi solve, fixed-ring topology, compact angular
faces, or a Cartesian core. The driver does not expose a user-selectable axis
treatment, core polynomial degree, observation/target rings, fixed-ring
topology, compact angular operator, or alternate phi state space.

See [Axis-regular angular RLP](axis_regular_angular_rlp.md) for the owner
operator and solver details.

## Time integration

- `rk4` advances the complete explicit RHS and performs the algebraic phi
  inversion at each stage.

The driver accepts only RK4. Runtime diagnostics report GMRES convergence,
state ranges, positivity, optional term fields, and grid-scale indicators.

The high-level `--flux-framework production-split` selector records and
exports the production curvature and parallel-material wiring. The curvature
block uses one live canonical-face characteristic matrix on every active
coordinate face and an equilibrium exterior state on physical coordinate
boundaries. The parallel-material block likewise uses one live canonical-face
matrix on ordinary and wall-ending FCI legs; at a wall it characteristic-
projects the operator's primitive wall trace with the same interior matrix
used by the one-sided fluctuation. These effective wall closures are recorded
separately from the bypassed
legacy scalar closure selectors. The production framework requires the
compatible FCI/owner-space contract described above; `legacy` remains the
default for reproducibility of existing runs.

## Frozen MMS diagnostic contract

`simulate_hsx_mms.py` evaluates exact-phi, source-paired, and reconstructed-
phi frozen states through the production stage operator. When MMS
counterfactual controls are enabled, all three evaluations reuse one compiled
augmented stage graph and differ only in their state and source inputs. This is
recorded as `shared-mms-audited-stage-graph-v1` in the aggregate artifact.
Keeping one graph shape is a qualification requirement: separate lean and
augmented specializations of a very large XLA graph are not assumed to be
numerically interchangeable.

The report checks both the graph-contract marker and the algebraic identity
`R(q,s) - R(q,0) - s`. Runs made with `--skip-counterfactuals` retain a shared
lean graph for exploratory use, but they do not satisfy the frozen spatial
qualification gate.

## Output and restart

History output stores logical coordinates, Cartesian cell positions, state
history, topology metadata, metric settings, operator selections, solver
space, and angular owner metadata. Scheduled snapshots are written atomically
and can optionally include Ve RHS term fields. Restart accepts either a
snapshot or a selected frame from a history file.

RLP state is owner-sparse during evolution. Output materialization prolongs
owner values to all fine cells so visualization receives a complete field.

## Nodal SBP perpendicular stack

The nodal SBP perpendicular stack is the perpendicular right-hand side of the P path: summation-by-parts operators on the family-A
nodes of the shared nodal layout, with SAT wall rows. It is a **reusable library with a verified model adapter**. It is not
selected by `simulate_hsx_blob.py`, `simulate_hsx_mms.py` or the Stage-7 selectors, which keep the owner-based state; wiring the
drivers is deferred to the P+Q nodal model (see "Deferred" below). The evolved MMS of P10 certified it (harness
`scripts/p10_evolved_mms/`, commit 555c1b76).

### Module map

| Layer | Modules (`src/drbx/`) |
|---|---|
| Layout | `geometry/nodal_layout.py`, `geometry/nodal_families.py` (`build_family_a_layout`: core, ring levels, one wall) |
| SBP building blocks | `geometry/sbp_core.py`, `sbp_operators.py`, `sbp_laplacian.py`, `sbp_laplacian_assembly.py`; `stencils/nodal_plan.py` (`NodalMetric`, `NodalPlan`, `load_nodal_metric`) |
| Operators | `native/fci_perpendicular_sbp_{ops,boundary,norms,bracket,curvature,dissipation,laplacian,laplacian_solve}.py`, `fci_perpendicular_plane_preconditioner.py` (core-Schur CG preconditioner) |
| Runtime RHS | `native/fci_nodal_perpendicular_rhs.py`: `NodalPerpendicularContext`, `NodalPerpendicularOptions`, `nodal_perpendicular_rhs`, `solve_potential` (jitted twins `*_jit`), `NodalWallData`, `bracket_rule_inflow` |
| Geometry bundle | `native/fci_nodal_geometry_bundle.py`: `assemble_nodal_bundle`, `save_nodal_bundle`, `load_nodal_bundle`, `NodalGeometryBundle` |
| Model adapter | `native/fci_nodal_perpendicular_model.py`: `NodalState`, `StageInfo`, `NODAL_PERPENDICULAR_DEFAULTS`, `nodal_stage_rhs`, `make_nodal_rhs` |
| Time stepping | `native/fci_time_integrator.py` (`Rk4Stepper`) |
| Sharding | `native/fci_perpendicular_sbp_sharding.py` (eta-sharded RHS and CG; halo 4 planes) |

### Data flow

1. **Layout.** `build_family_a_layout(n)` fixes the nodes `(u, theta, eta)`, the ring levels, the core and the wall points.
2. **Metric extraction** (host, once per geometry and arm; `scripts/p09_sbp_bracket/extract_metric.py`,
   `scripts/p09_sbp_laplacian/extract_metric.py`). Two files on the same nodes: the bracket/curvature metric `(h, |J|, B, K)` and
   the Laplacian metric `(A = |J| P_perp, d_i A^{ij}, |J|, g^{uj})` at the nodes and wall points.
3. **Bundle.** `assemble_nodal_bundle` joins them: it checks the arm identity, `P`, `n_eta`, nodes, wall points, `|J|` and `Hp` to
   tolerance, builds the `NodalPlan` and `LaplacianPlan`, the core-Schur preconditioner and the curvature flux `|J| K / B`, and
   records an identity (`bundle_sha256` over the geometry identity, layout and plan shas, metric identities and the stored arrays'
   sha). `save_nodal_bundle` / `load_nodal_bundle` round-trip it. A bundle is a pytree: pass it to `jit` as an argument.
4. **Context.** `bundle.ctx` is the `NodalPerpendicularContext(plan, lplan, prec, curvature_flux)`.
5. **Stage RHS.** `make_nodal_rhs(context, params, options, terms=..., phi_mode=..., wall_fn=..., source_fn=..., phi_fn=...,
   sigma_fn=...)` returns an `Rk4Stepper` `rhs_fn(state, t, carry) -> (rhs_state, carry, StageInfo)`. Per stage it evaluates
   `wall_fn`, the optional source, then either takes a prescribed `phi` or solves `L psi = (Omega - sigma) / rho*^2` by
   preconditioned CG (`psi = phi + tau n Ti`, Dirichlet `psi` data, warm start = the carry, which is the `psi` of the last solve),
   and sums the selected terms (`bracket`, `curvature`, `diffusion`). `nodal_stage_rhs` is the same stage as a pure function of
   `(context, params, state, t, carry)`; `rhs_fn.unbound` exposes it. Compile strategy: the geometry and the parameters are jit
   arguments (no geometry constants in the HLO) and `t` is traced, so one compilation serves every stage time.
6. **Time stepping.** `Rk4Stepper(rhs_fn)` advances `NodalState(n, Te, Ti, omega)`; the four `StageInfo` report CG iterations,
   residual, convergence, state minima and finiteness.

Normalization: single-length (`rho_star = rho_s0 / L_ref`): the bracket and the curvature carry `rho*`, the polarization is
`Omega = rho*^2 lap_perp(phi + tau p_i)`, diffusion is unchanged. The scaling acts on outputs and on the solve right-hand side,
never on the operators.

### Accepted defaults

`NODAL_PERPENDICULAR_DEFAULTS` (frozen record in `fci_nodal_perpendicular_model.py`; `.options(diffusion_kinds=..., phi_mode=...)`
builds the `NodalPerpendicularOptions`). A test pins it against the P10 harness configuration.

| Selector | Value |
|---|---|
| Layout family | A (one wall) |
| Fields | `density, Te, Ti, vorticity` |
| Bracket | split form with inflow SAT, level-face upwind SAT and face-jump dissipation (always on); `bracket_c_kappa = 1` scales the core shell damping |
| Curvature | centred split form, no jump dissipation (`curvature_c_kappa = 0`), `closed_form` absolute value |
| Neumann rows | physical normal derivative (`neumann_mode = "physical"`); `laplacian_c_kappa = 1` |
| Polarization variable | `psi = "phi_plus_tau_pi"` (`psi = phi + tau n Ti`), `tau = 1` |
| ψ solve | CG, `rtol = 1e-11`, `maxit = 50`, `core_schur` preconditioner with float32 factors |

Run-dependent entries are not defaults: the boundary kind per field (`diffusion_kinds`), `phi_mode`, `rho_star` and `D`.

### Verified scope and limits (P10, filtered arm)

- Global N-R orders are at least 2.0 on both resolution intervals (32-48, 48-64) for E1 (diffusion), E2 (bracket and curvature,
  prescribed `phi`, NNN-D only) and E3 (coupled with the `psi` solve) with the patterns DDDD and NNN-D (lowest 2.01); O-R is at
  most 3.6% of N-R.
- Neumann rows cover `n`, `Te` and `Ti` only; `Omega` and `psi` are Dirichlet with manufactured, time-dependent trace data.
- One manufactured family at physical `rho* = 4.5e-4`, `T = 1/30 t_ref`: this is a consistency test, **stability is not certified**.
- The raw (unfiltered metric) arm is reported only: it stalls over 48-64 where its t = 0 truncation stalls (E1 through the raw
  Laplacian, E3 through the `psi` solve).
- Single wall (family A); the unsharded RHS wraps eta periodically on one device (the sharded twin splits eta, never x or theta).
- Wall data are the caller's: the MMS supplies manufactured values; a production caller applies `bracket_rule_inflow` (Neumann-kind
  fields take their own wall trace, Dirichlet-kind keep the datum). The curvature and bracket walls are inflow SATs on incoming
  characteristics, never a Neumann closure.
- Automatic differentiation: forward mode through the RHS is verified against finite differences for prescribed `phi` and for the
  `psi` solve from a cold start. The CG is a bare `lax.while_loop`: reverse mode through the solve fails (the JAX `while_loop`
  error), and a warm start that already solves the system has a zero `psi` tangent. Prescribed-`phi` RHS is reverse-differentiable.
  `solve_dirichlet_implicit` is the differentiable solve; the RHS does not use it.
- Identity: layout and plan shas hash LAPACK-derived float bytes and differ by ulps between platforms. `load_nodal_bundle` keeps
  the stored identity and records this platform's shas under `identity["rebuilt"]`; a differing stored sha is never a rejection.
  What is verified bit for bit is the stored data (`arrays_sha256`) and the stored identity's own `bundle_sha256`.
- Replay across platforms (Linux campaign, macOS): the saved filtered N32 bundle loads and the t = 0 `tau` and a 10-step E3
  evolution reproduce the campaign snapshots (state to 9e-16, `psi` to 2e-13, `tau` to 3e-13 to 7e-11 depending on the operator:
  the diffusion `tau` is an ulp-sensitive `H^-1` second-derivative difference).

### Cost (campaign receipts, filtered and raw arms, both sources)

Perlmutter CPU node, 32 concurrent workers with 2 threads each. A unit is one run (50 / 62 / 71 RK4 steps at N32 / 48 / 64, six
snapshots with diagnostics, the compile of the step programs).

| N | mode | mean wall per unit (s) | peak RSS (GiB) |
|---|---|---|---|
| 32 | E1 diffusion | 69 | 0.78 |
| 32 | E2 hyperbolic | 140 | 1.18 |
| 32 | E3 coupled | 255 | 1.68 |
| 48 | E1 diffusion | 205 | 1.20 |
| 48 | E2 hyperbolic | 370 | 1.65 |
| 48 | E3 coupled | 717 | 2.29 |
| 64 | E1 diffusion | 288 | 2.03 |
| 64 | E2 hyperbolic | 583 | 2.61 |
| 64 | E3 coupled | 1525 | 3.79 |

Local N32 E3 (macOS, 2 threads, one process): lowering 5.2 s plus compile 1.0 s for one RK4 step program, then 0.25 s per step;
peak RSS 1.15 GiB for the whole replay.

### Deferred

Production-driver wiring (`simulate_hsx_mms.py`, `simulate_hsx_blob.py`, Stage-7 selectors) is not done: those drivers use the
owner-based state. It is part of the P+Q nodal model, where the parallel operators share this stack's norm `H` and the exchange
`E`, `R = H^-1 E^T m` described under "Extension rules". Until then the module map above is the supported entry surface.

## Extension rules

- A new toroidal operator must be valid as a fine polar operator under
  half-turn fills and lower-axis zero flux before it is wrapped in `R A_f P`.
- A new conservative wall term must define its operator-level boundary trace
  or flux; primitive field BCs are not a universal flux closure.
- A new FCI parallel operator must define mapped endpoint behavior at physical
  wall hits and axis crossings.
- Do not add a second toroidal state representation or silent fallback around
  owner-space RLP. The one exception is the nodal SBP perpendicular path: its
  state is point values at the family-A nodes, it is selected explicitly, and
  it exchanges with the parallel operators only through the plane-local
  raw-grid transfers `E` and `R = H^-1 E^T m`, with the pair cell mass `m`
  so that the parallel pairs stay adjoint in the nodal norm `H`.
- New production sharding paths must preserve the eta-only contract. Any
  communication required by a new operator must be explicit; P/R cannot be
  made nonlocal by splitting x or theta.

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
- [`tests/test_fci_time_integrator.py`](../../../tests/test_fci_time_integrator.py)
  covers the RK4 integrator and stage-output reduction.
