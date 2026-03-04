# Options & Toggles

This page collects the most important **physics and numerics toggles** exposed by
the unified DRB system. All options are designed to be **subsets of the same core RHS**.

---

## Engine Selection (top-level)

- `engine = "unified"`: current broad unified core (default).
- `engine = "drb_fv"`: strict finite-volume Hermes-alignment rewrite path.

Accepted aliases in input parsing:
- `engine = "fv_drb"`
- `engine = "drb-fv"`

CLI listing:
- `jaxdrb --list-engines`

---

## Physics Toggles (`[system]`)

- `electromagnetic_on`: enable parallel magnetic fluctuations (`psi` evolution).
- `hot_ion_on`: evolve ion temperature and ion parallel flow.
- `boussinesq_on`: use constant density in polarization (turn off for non‑Boussinesq).
- `sheath_on`: enable sheath boundary closures.
- `neutrals_on`: enable neutral interaction terms.
- `linear_on`: drop nonlinear ExB advection terms (linearized dynamics).
- `diamagnetic_on`: enable Hermes‑style diamagnetic drift terms.
- `diamagnetic_polarisation_on`: add ion pressure contribution to polarization operator.
- `drive_from_equilibrium_on`: compute background‑gradient drive from equilibrium profiles.

### Diamagnetic Drift (`[physics]`)

The diamagnetic drift uses the curvature vector (Curl(b/B)) from `curv_x/curv_y`
and adds drift terms to density, pressure/temperature, and parallel momentum:

- `diamag_form`: blend factor between divergence form (1) and gradient form (0).
- `diamag_form_profile`: optional profile (`x`, `1-x`, `x*(1-x)`).
- `diamag_density_model`: which species density to use (`electron`, `ion`, `none`).
- `diamag_bndry_flux`: allow diamagnetic flux through boundaries.

The diamagnetic polarization correction adds an ion‑pressure term to the
vorticity operator:

```
ω = ∇·(n ∇φ) + ∇·(B^{-2} ∇p_i)
```

Enable with `diamagnetic_polarisation_on=true` and scale with
`diamagnetic_polarisation_scale`.

### Equilibrium-Profile Drives (`[physics]`)

When `drive_from_equilibrium_on=true`, the density and temperature drives are
derived from equilibrium profiles instead of prescribed constants:

```
ω_n(x)  = -∂_x ln n_0(x)
ω_T(x)  = -∂_x ln T_0(x)
```

The drive terms are then `-ω_n ∂_y φ` and `-ω_T ∂_y φ`, optionally masked by
open/closed field regions. For SOL runs, the equilibrium profiles are derived
from `sol_n_core/sol_n_sol` and `sol_Te_core/sol_Te_sol`.

Additional controls:

- `drive_equilibrium_mode`: `auto`, `sol`, or `constant`.
- `drive_equilibrium_n0/Te0/Ti0`: constant equilibrium values (used when
  `drive_equilibrium_mode="constant"`).

---

## SOL & Sheath Closures (`[physics]`)

- `sol_on`: enable SOL masks and open/closed field‑line logic.
- `sol_parallel_loss_on`: enable Bohm‑like parallel loss sink terms.
- `sol_sheath_phi_on`: enable sheath‑current damping in the vorticity equation.
- `sol_sheath_phi_dissipation_on`: alignment switch for the explicit/implicit
  sheath-current vorticity dissipation path.
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

## Braginskii Closures (`[transport]`)

Enable Braginskii collision terms for heat exchange, friction, and classical
cross‑field diffusion:

- `braginskii_heat_exchange_on`: electron‑ion heat exchange.
- `braginskii_friction_on`: parallel friction between species.
- `braginskii_frictional_heating_on`: add frictional heating contributions.
- `classical_diffusion_on`: classical diffusion (density, momentum, temperature).
- `braginskii_nu_ei`, `braginskii_nu_ii`: collision frequencies (normalized).
- `braginskii_friction_coeff`: friction coefficient (default 0.51).
- `classical_diffusion_custom_D`, `classical_diffusion_custom_kappa_e/i`: override
  classical diffusivities; set to a non‑negative value to force a constant.

These terms are included in the default term schedule and are treated as stiff
updates for IMEX splitting.

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

For `axisymmetric_file`, coefficient files may additionally include metric cell
sizes:
- `metric_dx` (or `dx` fallback): radial cell size used in ExB FV transport.
- `metric_dy` (or `dy` fallback): field-line cell size used in metric DDY terms.
- `metric_dz` (or `dz` fallback): binormal/toroidal cell size used in X-Z ExB FV terms.

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
- `exb_flux_scheme`: flux-form perpendicular stencil selector in field-aligned
  geometry. `centered` uses centered divergence; `hermes_fromm` uses
  Hermes/BOUT-like Fromm-upwind X-Z transport; `hermes_xppm` uses MC-limited
  XPPM-style X-Z transport.
- `exb_bndry_flux`: allow/disallow radial boundary exchange in flux-form ExB
  (Hermes/BOUT `bndry_flux` role in `Div_n_bxGrad_f_B_XPPM`).
- `exb_poloidal_flows`: enable metric-coupled field-aligned ExB transport
  branch (Hermes/BOUT X-Y poloidal-flow contribution) in
  `FieldAlignedGeometryAdapter.exb_flux_divergence()`.
- `exb_poloidal_scale`: scalar multiplier on the metric-coupled poloidal ExB
  branch. Use `1.0` for equation-level alignment scans and vary only for
  controlled calibration sweeps.
- `exb_poloidal_x_scale`: multiplier on the X-flux branch of metric-coupled
  poloidal ExB transport.
- `exb_poloidal_y_scale`: multiplier on the Y-flux branch of metric-coupled
  poloidal ExB transport.
- `exb_poloidal_ddy_scheme`: index-space derivative used in the X-flux of the
  metric-coupled poloidal ExB branch. `face` keeps the legacy face-gradient
  form; `c2` uses a centered 2nd-order DDY-style stencil for Hermes/BOUT
  alignment studies.
- `exb_copy_grad_x_boundary`: when `true`, copy first interior phi-gradient
  values onto non-periodic x boundaries in metric-coupled ExB X-Y transport
  (Hermes/BOUT guard-cell-compatible behavior).
- `neumann_boundary_average_y`: Hermes/BOUT-compatible Neumann boundary mode
  for perpendicular operators. When enabled, Neumann x-ghost values are
  averaged over the perpendicular-y index before ghost padding
  (equivalent role to `neumann_boundary_average_z=true` in BOUT inputs).
- `parallel_limiter`: slope limiter applied to open-field parallel derivatives
  (`none`, `minmod`, `mc`).
- `parallel_transform`: `none` or `shifted`. `shifted` enables
  field-aligned parallel transforms using `z_shift` from geometry coefficients.
- `parallel_shift_interp`: interpolation used by `parallel_transform="shifted"`.
  `linear` is default; `spectral` applies FFT phase-shift interpolation.
- `parallel_flux_scheme`: open-field conservative flux (`rusanov` or `lax`).
- `parallel_flux_conservative`: use conservative parallel fluxes for `n` and `p`
  (e.g., `-∂‖(n v‖)` and `-∂‖(p v‖)`), with limiter/Lax flux when open-field.
- `parallel_pressure_model`: named pressure-transport closure for conservative
  parallel energy transport. `hermes_vgradp` maps to
  `-(5/3)∂‖(p v‖) + (2/3) v‖ ∂‖p`; `hermes_pdivv` maps to `-∂‖(p v‖)`;
  `custom` uses the coefficients below directly.
- `parallel_pressure_flux_coeff`: multiplier on conservative parallel pressure
  transport (`-∂‖(p v‖)` term). Used for Hermes early-time alignment calibration.
- `parallel_pressure_work_coeff`: optional `v‖∂‖p` add-on in pressure transport.
- `parallel_use_sheath_targets`: in open-field + sheath runs, replace boundary
  face `v‖` by Bohm/sheath targets in conservative parallel fluxes (Hermes-style
  boundary-flux alignment mode).
- `parallel_sheath_flux_mode`: `replace_boundary` (legacy) or `boundary_flux`
  (apply sheath targets on boundary face fluxes only).
- `phi_dissipation_on`: alignment switch for `phi_par_dissipation` in vorticity.
- `core_vorticity_damping_on`: alignment switch for core vorticity damping
  equivalents (`mu_lin_omega`, `mu_zonal_omega`).

DRB-FV specific numerics (`engine = "drb_fv"`):
- `fv_limiter`: limiter for parallel FV reconstruction (`mc`, `minmod`, `none`).
- `fv_poisson_solver`: `spectral_xy` (FFT inversion of `∇⊥²φ=ω`) or `identity` (debug).
- `parallel_pressure_flux_coeff`: coefficient on conservative pressure flux.
- `parallel_pressure_work_coeff`: coefficient on `v_parallel * d_parallel(p)`.
- `vorticity_parallel_coeff`: coefficient on parallel-current vorticity coupling.
- `curvature_coeff`: coefficient on curvature-driven vorticity source.

DRB-FV specific geometry ingestion:
- `geometry.coeff_path`: optional metric bundle used by the rewrite path.
  Currently supported fields are `J`, `curv_x`/`bxcv`, `gxx`, `gxy`, `gyy`,
  and `dpar_factor`.
- If `coeff_path` provides `nx`, `ny`, or `z`, those are used to infer the
  alignment grid unless the config already specifies the same values.
- Config/file dimension mismatches are treated as errors.

DRB-FV specific term toggles (`[terms]`):
- `parallel_on`
- `curvature_on`
- `sheath_on`

DRB-FV sheath coupling uses:
- `geometry.open_field_line = true`
- `[closures.sheath]` values:
  - `sheath_bc_on`
  - `sheath_loss_on`
  - `sheath_bohm_velocity_on`
  - `sheath_energy_on`
  - `sheath_gamma_e`
  - `sheath_current_closure_coeff`
- `[numerics]` value:
  - `sheath_relax_coeff`

### Sheath Options

- `sheath_energy_model`: electron sheath-energy closure. `relaxation` keeps the
  older local damping form; `hermes_flux` applies a boundary heat-flux source
  using the Hermes sheath formula and geometry-dependent face-to-volume scaling.
- `sheath_energy_flux_scale`: scalar multiplier on the `hermes_flux` energy
  source.
- `sheath_secondary_electron_coef`: effective secondary-electron coefficient
  used in the sheath transmission and electron outflow formulas.
- `sheath_wall_potential`: wall potential used in the sheath exponential.
- `sheath_floor_potential`: if `true`, floor the sheath potential at the wall
  potential before evaluating electron losses.
- `sheath_electron_adiabatic`: electron ratio of specific heats used in the
  extra sheath-energy flux.
- `sheath_ion_adiabatic`: ion ratio of specific heats used in Bohm/Loizu ion
  sheath targets.

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
