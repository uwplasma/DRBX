# Options & Toggles

This page collects the most important **physics and numerics toggles** exposed by
the unified DRB system. All options are designed to be **subsets of the same core RHS**.

---

## Physics Toggles (`[system]`)

- `electromagnetic_on`: enable parallel magnetic fluctuations (`psi` evolution).
- `hot_ion_on`: evolve ion temperature and ion parallel flow.
- `boussinesq_on`: use constant density in polarization (turn off for non‑Boussinesq).
- `sheath_on`: enable sheath boundary closures.
- `neutrals_on`: enable neutral interaction terms.
- `linear_on`: drop nonlinear ExB advection terms (linearized dynamics).

---

## Geometry Options

Geometry is selected via `[geometry]` + a geometry‑specific block.

- `kind = "slab"`: Cartesian shear‑slab.
- `kind = "salpha"`: analytic s‑alpha (ballooning) equilibrium.
- `kind = "miller"`: analytic Miller equilibrium.
- `kind = "axisymmetric_file"`: axisymmetric coefficients loaded from file.
- `kind = "fci"`: 3D flux‑coordinate independent geometry from maps.
- `kind = "line"`: 1D field‑aligned / flux‑tube.

All geometries feed the **same coefficient interface** (curvature, `dpar_factor`,
metric scalings), so the core RHS remains unchanged.

---

## Boundary Conditions (`[bc]`)

- `bc_x`, `bc_y`: periodic, Neumann, or Dirichlet in perpendicular directions.
- `bc_z`: field‑aligned BCs (periodic, sheath, relaxation).
- `bc_enforce_nu_*`: enforcement rates for relaxation‑style BCs.

Region‑policy BCs can be configured via `[boundary_policy]` to apply different
BCs in core/SOL/divertor windows without splitting the equations.

---

## Term Scheduling

Use `term_schedule` to select explicit term ordering, or `term_schedule_preset`
for minimal preset schedules:

- `preset_linear`: parallel + curvature + drive + diffusion (no nonlinear advection)
- `preset_nonlinear`: adds ExB advection to `preset_linear`
- `preset_min`: advection + parallel + curvature + diffusion (no drive)

---

## Numerics (`[numerics]`)

- `poisson_solver`: `spectral` (periodic) or `cg_fd` (non‑periodic).
- `poisson_preconditioner`: `jacobi` or `fd_fft`.
- `poisson_warm_start`: reuse previous `phi` as CG initial guess.
- `poisson_track_iters`: record CG iteration stats.
- `parallel_z_mode`: `vmap` (fast, more memory) or `scan` (lower memory).

---

## Time Integrators (`[time]`)

- `method = "rk4_scan"`: fixed‑step RK4 scan (JIT‑compiled).
- `method = "rk4_imex_strang"`: Strang split with implicit diffusion/parallel update.
- `method = "diffrax"`: adaptive Diffrax solvers (e.g., `dopri8`).

All integrators are differentiable; use `remat` or `scan_remat` for long‑run
memory control.
