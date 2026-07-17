# Solvers and Design Decisions

This page documents the numerical solvers `drbx` actually uses — with the
concrete parameters and file locations — and the design rules the codebase
follows. The governing equations these solvers advance are on
[Models and Governing Equations](models_and_equations.md).

## Perpendicular-Laplacian inversion (solvax GMRES)

The 4-field and DRB lanes must invert the conservative perpendicular
Laplacian \(-\nabla_\perp\!\cdot\!\nabla_\perp \phi = -\omega\) once per RK4
stage. The solver is `PerpLaplacianInverseSolver` in
[`native/fci_operators.py`](../src/drbx/native/fci_operators.py):

- the operator is a matrix-free matvec wrapping the conservative stencil
  (`perp_laplacian_conservative_op` on a `ConservativeStencilBuilder`
  payload), solved with `solvax.gmres` (restarted flexible GMRES, fully
  jit-able) using `rtol = atol = tol` (default `1e-6`) on the **true**
  residual, a Krylov cycle of `min(restart, maxiter)`, and a total iteration
  budget of `maxiter`; the multigrid V-cycle enters as a right
  preconditioner;
- nonzero regular-face and cut-wall boundary values are **lifted out of the
  operator** (a one-time boundary-source application) so the GMRES matvec
  stays linear; Dirichlet lifts get a homogeneous correction solve;
- nullspace handling is explicit: optional weighted mean-zero projection
  (`project_mean_zero`), a pinned point (`pin_point`/`pin_value`), or a Tikhonov
  `regularization_epsilon` — mutually checked against the preconditioner;
- the solve closures are built **once per geometry/BC payload** and cached as
  jitted functions; the stage-dependent RHS, warm-start guess, and boundary
  values remain dynamic arguments, and the previous stage's \(\phi\) is passed
  back as `phi_guess` to warm-start the next inversion.

### Fast path vs diagnostic path

The solver has two call paths (new 2026-07-17):

- **Diagnostic path** (`check_residual=True`, the default, or
  `return_diagnostics=True`): after the GMRES solve it recomputes the true
  residual, compatibility ratios, and finiteness flags, converts them to
  Python floats (host syncs), and raises if the relative residual exceeds
  `10 x tol`. This is the validation/debugging path — those host-synced
  `float(...)` conversions force a device round-trip per stage.
- **Fast path** (`check_residual=False` and no `return_diagnostics`):
  `_solve_fast_impl` returns \(\phi\) only — one boundary-source application
  plus the GMRES solve, **no diagnostic matvecs and no host syncs** — so it is
  safe to call from inside `jit`-compiled stepping code. Two companion fixes
  landed with it: the GMRES solve now honors the requested
  `phi_inversion_tol` (it was previously hardcoded to `rtol=atol=1e-6`,
  over-solving each stage for looser-tolerance models), and the FCI RHS
  assemblies (`compute_2field_rhs`, `compute_4field_*`) return
  `timings=None` by default so they are sync-free and jittable —
  `with_diagnostics=True` restores the host-synced stage-timings and
  phi-diagnostics payload for the validation harnesses.

On the fast path the **entire 4-field RK4 step** — four RHS evaluations
including their GMRES phi inversions — compiles as one jit program in
`drbx.native.stellarator_turbulence.run_stellarator_turbulence` (a
one-time ~9 s compile). On the `(24, 32, 8)` rotating-ellipse case this took
the single-CPU step from 1.200 s (eager, host-synced diagnostics) to
0.623 s (1.9x); the measurement and its context are on
[Performance and Differentiability](performance_and_differentiability.md).

### Multigrid V-cycle preconditioner

For larger grids the GMRES solve takes an optional geometric-multigrid
preconditioner: `build_perp_laplacian_mg_hierarchy` coarsens the geometry,
face BCs, and face metrics by factors of two into a
`PerpLaplacianMgHierarchy` of `PerpLaplacianMgLevel`s, applied as a V-cycle
(`pre_smooth=2`, `post_smooth=2`) with a Chebyshev smoother by default
(`smoother="chebyshev"`, order 2; weighted-Jacobi `omega_jacobi=0.65` is the
alternative). The hierarchy is a pytree, so it passes through `jit`
unchanged.

When the coarsest level has at most `direct_coarse_size=512` cells, the dense
coarse operator is assembled **once at build time** and LU-factorized with
`jax.scipy.linalg.lu_factor`; each V-cycle then does a cheap triangular
`lu_solve` instead of smoothing (or re-factorizing) at the coarsest level
(new in July 2026, stored as `coarse_lu_and_piv` on the hierarchy).
Cut-wall payloads are not coarsened (`NotImplementedError`), and hierarchies
exclude pinned rows and Tikhonov regularization by construction. Gate:
`tests/test_multigrid_preconditioner.py`.

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
gate is `tests/test_rotating_ellipse_fci.py`). Imported geometries (ESSOS
coils/VMEC, VMEC-extender field grids, vmec_jax equilibria) enter through the
adapters in
[`geometry/essos_import.py`](../src/drbx/geometry/essos_import.py),
[`geometry/vmec_extender_import.py`](../src/drbx/geometry/vmec_extender_import.py),
and [`geometry/vmec_jax_import.py`](../src/drbx/geometry/vmec_jax_import.py).

### The vmec_jax adapter

[`geometry/vmec_jax_import.py`](../src/drbx/geometry/vmec_jax_import.py)
(new in July 2026) imports [vmec_jax](https://github.com/rogeriojorge/vmec_jax)
from an external checkout (`DRBX_VMEC_JAX_ROOT`, default
`~/local/vmec_jax`) the same way the ESSOS adapter does, and adds the pieces
`drbx` examples need on top of a loaded `wout_*.nc` equilibrium:
`vmec_jax_runtime_available`, `load_vmec_jax_wout`, `vmec_jax_wout_summary`
(nfp, aspect ratio, iota profile, \(B_0\)),
`evaluate_vmec_jax_surface_field` (\(B^\theta\), \(B^\phi\), \(|B|\) on
half-mesh surfaces from the Nyquist tables), `trace_vmec_jax_field_lines`
(a JAX RK4 tracer in \((s,\theta,\phi)\): since \(B^s = 0\) a line stays on
its surface and obeys \(d\theta/d\phi = B^\theta/B^\phi\)),
`traced_rotational_transform`, and the cylindrical mappings
`vmec_jax_surface_rz` / `vmec_jax_boundary_rz`. The examples are
`examples/geometry-3D/vmec-jax/closed_field_lines.py` (traced iota matches the
wout `iotaf` profile to ~1e-6) and
`examples/geometry-3D/vmec-jax/closed_open_field_lines.py` (ESSOS coil field
with the VMEC last-closed-flux-surface overlay). The adapter is lazy and
optional: `drbx` imports cleanly without vmec_jax installed.

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
  construction. Optional diagnostics that need host values (timings, residual
  checks) are opt-in flags (`with_diagnostics`, `check_residual`,
  `return_diagnostics`), never the default inside stepping loops.
- **Host syncs only at boundaries.** `float(...)`, `block_until_ready`,
  printing, plotting, and file I/O happen in the driver scripts and
  validation harnesses, not inside kernels. The phi-solver fast path exists
  precisely to keep the RK4 hot loop free of device round-trips.
- **Pytree dataclasses.** Model states (`Fci4FieldState`, `FciDrbState`,
  …), parameter bundles, boundary payloads, and even the multigrid hierarchy
  are frozen dataclasses registered with
  `jax.tree_util.register_pytree_node_class`, so whole model configurations
  pass through `jit`, `grad`, and `shard_map` as ordinary arguments and
  static metadata lives in `aux_data`.
- **Build once, solve many.** Stencil builders, face projectors, BC payloads,
  curvature coefficients, the GMRES solve closures, and the MG hierarchy
  (with its prefactored coarse LU) are constructed once per geometry and
  reused every stage; only fields and dynamic BC values change per call.
- **TOML decks at the user boundary.** The CLI (`drbx inspect/run`,
  [`cli.py`](../src/drbx/cli.py) →
  [`native/deck_runner.py`](../src/drbx/native/deck_runner.py)) parses TOML
  decks, dispatches to the native models, and serializes JSON/NPZ artifacts —
  NumPy/SciPy and file I/O live here, outside the differentiable core.
- **Examples are flat pedagogical scripts**: imports → a PARAMETERS block →
  explicit setup → a run loop with progress prints → plotting, so a reader
  can see every physics and numerics choice in one file (see the
  [tutorials](tutorial_hasegawa_wakatani.md)).
