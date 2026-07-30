# Solvers and Design Decisions

This page documents the numerical solvers `drbx` actually uses — with the
concrete parameters and file locations — and the design rules the codebase
follows. The governing equations these solvers advance are on
[Models and Governing Equations](models_and_equations.md).

## Perpendicular-Laplacian inversion (SOLVAX FGMRES)

The sharded FCI lanes invert the conservative perpendicular Laplacian through
`LocalPerpLaplacianInverseSolver` in
[`native/fci_operators.py`](../src/drbx/native/fci_operators.py). Its Krylov
adapter and diagnostics live in
[`native/fci_gmres.py`](../src/drbx/native/fci_gmres.py).

- `solvax.gmres` supplies restarted flexible GMRES, CGS2 Arnoldi
  orthogonalization, incremental Givens rotations, and right
  preconditioning.
- DRBX supplies a custom inner product whose local owned-cell dot product is
  reduced over every `shard_map` mesh axis. The SOLVAX default inner product
  must not be used for a partitioned field because it would stop on a
  shard-local residual.
- `maxiter` remains a total Krylov-step budget. The adapter chooses an exact
  restart-cycle divisor and maps that budget to SOLVAX `max_restarts`.
- Nonzero physical boundary data are lifted before the solve. FGMRES sees the
  homogeneous correction operator, and the lift is added back afterward.
- The adapter recomputes the global true residual and retains separate target
  and acceptance tolerances, active-cell masking, optional weighted
  mean-zero projection, regularization, and finiteness diagnostics.

The regular-grid solver currently supports these local right
preconditioners:

- `none`;
- geometry-aware point `jacobi`;
- tridiagonal `line-u`;
- tridiagonal `line-v`;
- additive `line-uv`.

The bands retain the axis-normal part of the metric projector and omit mixed
metric couplings. With `(1,1,N)` sharding, complete `u` and `v` lines are
local, so these line solves remain shard-compatible. A cyclic `eta` line
solve is intentionally not offered: an eta-sharded local cyclic solve would
invert independent shard segments rather than the global periodic line.

Distributed multigrid is not implemented in this path yet. A correct version
must keep halo exchange in every coarse matvec and provide a distributed
coarse solve; applying a stock local hierarchy independently on each shard
would change the global operator.

The full EB RK4 advance, including four potential reconstructions, is lowered
as one jitted `shard_map` program. The driver returns each stage's iteration
count, true relative residual, acceptance flag, and failure flag, and aborts
before continuing from an unaccepted inversion.

## solvax structured solves

[`solvax`](https://github.com/uwplasma/SOLVAX) is the reusable structured
solver library extracted from this code family. `drbx` uses two pieces:

- **Spectral Fourier–Helmholtz elliptic solve** — the electrostatic vorticity
  deck lane ([`native/vorticity.py`](../src/drbx/native/vorticity.py))
  builds a `FourierHelmholtzOperator` from the metric payload
  (`build_fourier_helmholtz_operator`) and inverts its potential with
  `solve_fourier_helmholtz`: FFT in the periodic direction, a tridiagonal
  solve per Fourier mode in the bounded direction.
- **Tridiagonal (Thomas) solves** — the implicit pieces of the 1-D neutral
  models: neutral parallel diffusion in
  [`neutrals/recycling_sol_model.py`](../src/drbx/native/neutrals/recycling_sol_model.py)
  and both the neutral diffusion and the implicit Spitzer conduction
  \(\kappa \sim T^{5/2}\) in
  [`neutrals/detachment_sol_model.py`](../src/drbx/native/neutrals/detachment_sol_model.py)
  call `solvax.tridiagonal_solve` (which lowers to
  `jax.lax.linalg.tridiagonal_solve`), making the stiff parabolic terms
  unconditionally stable while staying differentiable.

## Spectral Poisson solve in Hasegawa-Wakatani

The HW flagship needs no iterative solver at all: on the doubly periodic
grid the vorticity relation \(\zeta = \nabla_\perp^2\phi\) inverts
algebraically in Fourier space, \(\hat\phi_k = -\hat\zeta_k/k^2\)
(`potential_from_vorticity` in
[`native/hasegawa_wakatani.py`](../src/drbx/native/hasegawa_wakatani.py)),
and the Poisson bracket is evaluated pseudo-spectrally with 2/3-rule
dealiasing. This is why the closed-field-line turbulence lane is the fastest
model in the package.

## Time integration

- The FCI models advance with classical **RK4**
  ([`native/fci_time_integrator.py`](../src/drbx/native/fci_time_integrator.py)).
  `rk4_step` is model-agnostic over any `FciModelState` pytree and threads a
  `carry` (e.g. the warm-start \(\phi\)) plus an opaque per-stage `aux`
  payload (solver diagnostics, stage timings) through the four stage calls
  without the RK4 core knowing about them.
- The HW flagship uses its own fixed-step RK4 inside `jax.lax` loops
  (`hw_run`), fully jitted.
- The 1-D neutral models use operator splitting: explicit hyperbolic
  transport + implicit tridiagonal diffusion/conduction + per-cell implicit
  (self-limiting) stiff sources per step.
- The compact deck lanes use an exact matrix-exponential propagator
  (diffusion) and adaptive Dormand–Prince (`odeint`, electrostatic
  vorticity).

## Multi-device parallelization (`shard_map` halo exchange)

[`native/fci_sharding.py`](../src/drbx/native/fci_sharding.py) promotes the
FCI stack to multi-device execution:

- `make_shard_mesh` builds a three-axis `jax.sharding.Mesh`;
- `build_local_fci_geometries` splits the global `FciGeometry3D` into
  per-shard geometry bundles partitioned with `PartitionSpec("x", "y", "z")`;
- inside `shard_map`, `assemble_local_fci_geometry` reassembles each shard's
  `LocalFciGeometry3D` by halo exchange plus periodic topology fill, and
  `make_sharded_2field_step` returns a jitted RK4 step in which **every stage
  refreshes the state halos before evaluating the RHS on local geometry**.

The sharded step is **bit-exact** against the single-device step
(`tests/test_fci_sharded_2field.py`, including a forced 4-device run), so
sharding changes where the work runs, never the result. Measured strong
scaling and the current GPU status live on
[Performance and Differentiability](performance_and_differentiability.md).

## Geometry by autodiff of the embedding

Every analytic geometry (rotating ellipse, island divertor, …) supplies only
its embedding map \(u = (x,\theta,\zeta) \mapsto (X,Y,Z)\);
[`geometry/embedding.py`](../src/drbx/geometry/embedding.py) computes the
covariant metric exactly as the Gram matrix of the embedding Jacobian with
`jax.jacfwd` — \(g_{ij} = \partial_i \mathbf{X} \cdot \partial_j \mathbf{X}\),
\(J = \sqrt{\det g}\), \(g^{ij} = (g_{ij})^{-1}\) — instead of hand-derived
metric formulas. Because the metric is built by autodiff, it is itself
**differentiable with respect to the shape parameters** (the shape-gradient
gate is the supported sharded full-EB MMS gate
(`tests/test_mms_shifted_torus_EB_sharded.py`). Imported geometries (ESSOS
coils/VMEC, VMEC-extender field grids, VMEX equilibria) enter through the
adapters in
[`geometry/essos_import.py`](../src/drbx/geometry/essos_import.py),
[`geometry/vmec_extender_import.py`](../src/drbx/geometry/vmec_extender_import.py),
and [`geometry/vmex_import.py`](../src/drbx/geometry/vmex_import.py).

### The VMEX adapter

[`geometry/vmex_import.py`](../src/drbx/geometry/vmex_import.py)
(new in July 2026) imports [VMEX](https://github.com/uwplasma/VMEX)
from an external checkout (`DRBX_VMEX_ROOT`, default
a local checkout) the same way the ESSOS adapter does, and adds the pieces
`drbx` examples need on top of a loaded `wout_*.nc` equilibrium:
`vmex_runtime_available`, `load_vmex_wout`, `vmex_wout_summary`
(nfp, aspect ratio, iota profile, \(B_0\)),
`evaluate_vmex_surface_field` (\(B^\theta\), \(B^\phi\), \(|B|\) on
half-mesh surfaces from the Nyquist tables), `trace_vmex_field_lines`
(a JAX RK4 tracer in \((s,\theta,\phi)\): since \(B^s = 0\) a line stays on
its surface and obeys \(d\theta/d\phi = B^\theta/B^\phi\)),
`traced_rotational_transform`, and the cylindrical mappings
`vmex_surface_rz` / `vmex_boundary_rz`. The examples are
`examples/geometry-3D/vmex/closed_field_lines.py` (traced iota matches the
wout `iotaf` profile to ~1e-6) and
`examples/geometry-3D/vmex/closed_open_field_lines.py` (ESSOS coil field
with the VMEC last-closed-flux-surface overlay). The adapter is lazy and
optional: `drbx` imports cleanly without VMEX installed.

## Why FCI

Field-aligned coordinate systems degenerate where the field-aligned
coordinate does (X-points, islands, stochastic regions, magnetic axes). The
flux-coordinate-independent approach (Hariri & Ottaviani, *CPC* 184, 2419
(2013)) keeps the mesh an ordinary cylindrical/logical grid — **no
field-aligned coordinate, hence no coordinate singularities** — and builds
parallel operators by tracing field lines between neighboring toroidal planes
and interpolating (`geometry/fci_maps.py`, `geometry/fci_geometry.py`).
Perpendicular operators stay local on the plane with the full metric. This is
what lets one operator stack serve tokamaks, rotating-ellipse and
island-divertor stellarators, and imported coil/VMEC fields, with open field
lines handled by endpoint masks feeding the Bohm sheath closure rather than by
special coordinates.

## Design rules

The codebase follows a small set of deliberate rules:

- **Pure-`jnp` hot paths.** Every RHS, operator, and solver kernel is pure
  `jax.numpy` on explicit inputs — `jit`/`grad`/`vmap`-transparent by
  construction. Solver diagnostics remain device arrays inside the compiled
  advance and are transferred to the host only at progress/output boundaries.
- **Host syncs only at boundaries.** `float(...)`, `block_until_ready`,
  printing, plotting, and file I/O happen in the driver scripts and
  validation harnesses, not inside kernels. The phi-solver fast path exists
  precisely to keep the RK4 hot loop free of device round-trips.
- **Pytree dataclasses.** Model states (`Fci4FieldState`, `FciDrbState`,
  …), parameter bundles, boundary payloads, local geometry, and SOLVAX
  configuration/diagnostics are frozen dataclasses registered with
  `jax.tree_util.register_pytree_node_class`, so whole model configurations
  pass through `jit`, `grad`, and `shard_map` as ordinary arguments and
  static metadata lives in `aux_data`.
- **Build once, solve many.** Stencil builders, face projectors, BC payloads,
  curvature coefficients, and invariant boundary characteristic projectors
  are constructed once per geometry and reused every stage. The local
  preconditioner bands are geometry-only arrays lowered with the compiled
  solve; only fields and dynamic BC values change per call.
- **TOML decks at the user boundary.** The CLI (`drbx inspect/run`,
  [`cli.py`](../src/drbx/cli.py) →
  [`native/deck_runner.py`](../src/drbx/native/deck_runner.py)) parses TOML
  decks, dispatches to the native models, and serializes JSON/NPZ artifacts —
  NumPy/SciPy and file I/O live here, outside the differentiable core.
- **Examples are flat pedagogical scripts**: imports → a PARAMETERS block →
  explicit setup → a run loop with progress prints → plotting, so a reader
  can see every physics and numerics choice in one file (see the
  [tutorials](tutorial_hasegawa_wakatani.md)).
