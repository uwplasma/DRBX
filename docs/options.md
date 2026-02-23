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

## SOL & Sheath Closures (`[physics]`)

- `sol_on`: enable SOL masks and open/closed field‑line logic.
- `sol_parallel_loss_on`: enable Bohm‑like parallel loss sink terms.
- `sol_sheath_phi_on`: enable sheath‑current damping in the vorticity equation.
- `sol_sheath_phi_model`: `linear` or `exp` (linear is recommended for implicit updates).
- `sol_sheath_phi_implicit`: operator‑split implicit update for sheath current (robust for long runs).
- `sol_sheath_phi_implicit_solver`: `gmres` (default) or `cg`.
- `sol_sheath_phi_implicit_rtol/atol/maxiter/restart`: linear solve tolerances.

When `sol_sheath_phi_implicit=true`, the explicit term is disabled and the
implicit update is applied by IMEX time integrators (e.g. `rk4_imex_strang`).

### Sheath Boundary Models (`[closures].sheath`)

The core sheath closure is configured via `sheath_bc_on=true` and
`sheath_bc_model`. Available models include:

- `simple`: linearized Bohm-style relaxation with optional particle/energy damping.
- `loizu_linear`: Loizu-style linearized sheath model (target-aware).
- `bohm_current`: Bohm condition with current-balance relaxation **without**
  direct particle damping (density loss occurs via parallel fluxes). This
  mirrors common SOL implementations where the sheath sets flow and potential,
  rather than imposing a direct sink.

The target values for the ion and electron flows follow Bohm-style conditions:

```
v_i >= c_s,  v_e ~ c_s - phi
```

where `c_s = sqrt(1 + tau_i)` in normalized units. Energy sinks are controlled
via `sheath_gamma_e` and `sheath_gamma_i`, and direct particle damping can be
enabled with `sheath_loss_on=true`.

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
- `exb_y_scale`: scale the poloidal (`y`) component of ExB advection
  (`exb_y_scale=0` disables y-advection to mimic `poloidal_flows=false` in BOUT++).
- `parallel_limiter`: slope limiter applied to open-field parallel derivatives
  (`none`, `minmod`, `mc`).
- `parallel_flux_conservative`: use conservative parallel fluxes for `n` and `p`
  (e.g., `-∂‖(n v‖)` and `-∂‖(p v‖)`), with limiter/Lax flux when open-field.

### Parallel Flux Scheme (Open-Field)

When `parallel_flux_conservative=true` and `open_field_line=true`, the solver
uses a **finite-volume Lax flux** with slope-limited reconstruction along the
parallel coordinate:

```
F_{i+1/2} = 0.5 (f_L v_L + f_R v_R) + 0.5 a_max (f_L - f_R)
```

where `a_max = max(|v_L|, |v_R|)` and `f_L`, `f_R` are limited states obtained
with `parallel_limiter = "minmod"` or `"mc"`. The divergence is then
`(F_{i+1/2} - F_{i-1/2}) / Δz`.

---

## Time Integrators (`[time]`)

- `method = "rk4_scan"`: fixed‑step RK4 scan (JIT‑compiled).
- `method = "rk4_imex_strang"`: Strang split with implicit diffusion/parallel update.
- `method = "diffrax"`: adaptive Diffrax solvers (e.g., `dopri8`).

All integrators are differentiable; use `remat` or `scan_remat` for long‑run
memory control.
