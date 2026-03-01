# Inputs & Outputs

This page describes the **configuration schema** and the **output files** emitted by
`jax_drb` runs. The solver is configured through TOML and writes NumPy `.npz` files
containing diagnostics and snapshots.

---

## Input Structure (TOML)

Typical input files are organized into the following sections:

- `[system]`: primary toggles (ES/EM, hot/cold ions, Boussinesq, sheath, neutrals).
- `[geometry]`: grid sizes and analytic geometry parameters.
- `[geometry_*]`: geometry‑specific blocks (e.g., `geometry.salpha`, `geometry.axisymmetric`).
- `[physics]`: drive parameters, curvature scaling, resistivity, normalization‑free coefficients.
- `[transport]`: diffusion, hyperdiffusion, and linear damping rates.
- `[closures]`: SOL closures, sheath settings, neutral coupling, edge relaxation.
- `[bc]`: perpendicular and parallel BC types (periodic/Neumann/Dirichlet) and enforcement rates.
- `[initial]`: initial profiles, noise, and mixmode perturbations.
- `[numerics]`: Poisson solver selection, preconditioners, tolerances, operator options.
- `[time]`: integrator choice, step size, save frequency, and diagnostics.
- `[normalization]`: optional physical‑to‑normalized conversion block.
- `[geometry_physical]`, `[physics_physical]`, `[transport_physical]`, `[closures_physical]`,
  `[initial_physical]`, `[bc_physical]`: physical units converted into normalized values when
  normalization is enabled.

The CLI expects a single TOML file:

```bash
jaxdrb /path/to/input.toml --run --output /path/to/output.npz
```

## Parameter Listing (Full)

This section enumerates the full configuration keys for each block.

### `[physics]`

- `nonlinear_on`
- `hot_ion_on`
- `em_on`
- `neutrals_on`
- `boussinesq`
- `non_boussinesq_perturbed_density_on`
- `log_n`
- `log_Te`
- `omega_n`
- `omega_Te`
- `omega_Ti`
- `omega_drive_mask`
- `drive_from_equilibrium_on`
- `drive_equilibrium_mode`
- `drive_equilibrium_n0`
- `drive_equilibrium_Te0`
- `drive_equilibrium_Ti0`
- `kappa`
- `kappa_profile`
- `kappa_theta0`
- `alpha`
- `alpha_nonzonal_only`
- `kpar`
- `eta`
- `eta_par`
- `me_hat`
- `alpha_Te_ohm`
- `alpha_Ti`
- `alpha_Ti_ohm`
- `beta`
- `Dpsi`
- `tau_i`
- `curvature_on`
- `curvature_coeff`
- `curvature_model`
- `curvature_theta_scale`
- `curvature_scale`
- `curvature_Te_coeff`
- `diamagnetic_on`
- `diamag_form`
- `diamag_form_profile`
- `diamag_density_model`
- `diamag_bndry_flux`
- `diamagnetic_polarisation_on`
- `diamagnetic_polarisation_scale`
- `n0`
- `n0_min`
- `n0_max`
- `k2_min`
- `kperp2_min`
- `log_n_clip`
- `log_Te_clip`
- `source_on`
- `source_profile`
- `source_x_mode`
- `source_only_in_core`
- `source_n0`
- `source_Te0`
- `source_Te_is_pressure`
- `source_x0`
- `source_y0`
- `source_width_x`
- `source_width_y`

### `[transport]`

- `Dn`
- `DOmega`
- `Dvpar`
- `DTe`
- `DTi`
- `Dpsi`
- `chi_par`
- `chi_par_Te`
- `chi_par_Ti`
- `nu_par_e`
- `nu_par_i`
- `nu_sink_n`
- `nu_sink_Te`
- `nu_sink_vpar`
- `Dn4`
- `DOmega4`
- `DTe4`
- `DTi4`
- `Dpsi4`
- `nu4_n`
- `nu4_omega`
- `mu_zonal_omega`
- `mu_lin_n`
- `mu_lin_omega`
- `mu_lin_vpar_e`
- `mu_lin_vpar_i`
- `mu_lin_Te`
- `vort_par_dissipation`
- `phi_par_dissipation`
- `phi_dissipation_on`
- `core_vorticity_damping_on`
- `braginskii_on`
- `braginskii_state_dependent_on`
- `braginskii_eta_on`
- `braginskii_kappa_e_on`
- `braginskii_kappa_i_on`
- `braginskii_visc_e_on`
- `braginskii_visc_i_on`
- `braginskii_Tref`
- `braginskii_T_floor`
- `braginskii_T_smooth`
- `braginskii_heat_exchange_on`
- `braginskii_friction_on`
- `braginskii_frictional_heating_on`
- `classical_diffusion_on`
- `braginskii_nu_ei`
- `braginskii_nu_ii`
- `braginskii_nu_floor`
- `braginskii_friction_coeff`
- `classical_diffusion_custom_D`
- `classical_diffusion_custom_kappa_e`
- `classical_diffusion_custom_kappa_i`

### `[closures].sheath`

- `sheath_on`
- `sheath_nu_factor`
- `sheath_bc_on`
- `sheath_bc_nu_factor`
- `sheath_bc_model`
- `sheath_bc_linearized`
- `sheath_lambda`
- `sheath_Te_floor`
- `sheath_heat_on`
- `sheath_gamma_auto`
- `sheath_gamma_e`
- `sheath_gamma_i`
- `sheath_see_on`
- `sheath_see_yield`
- `sheath_end_damp_on`
- `sheath_loss_on`
- `sheath_loss_nu_factor`
- `sheath_nu_mom`
- `sheath_nu_particle`
- `sheath_nu_energy`
- `sheath_delta`
- `sheath_cos2`
- `sheath_bc_model_fci`

### `[closures].sol`

- `sol_on`
- `sol_xs`
- `sol_width`
- `sol_open_left`
- `sol_mask_y_taper`
- `sol_n_core`
- `sol_n_sol`
- `sol_Te_core`
- `sol_Te_sol`
- `sol_relax_core`
- `sol_relax_open`
- `sol_sink_open_n`
- `sol_sink_open_Te`
- `sol_sink_open_omega`
- `sol_sink_open_omega_mode`
- `sol_sink_open_vpar`
- `sol_nonlinear_open_scale`
- `sol_n_floor`
- `sol_Te_floor`
- `sol_source_n0`
- `sol_source_Te0`
- `sol_source_xs`
- `sol_source_width`
- `sol_source2_n0`
- `sol_source2_Te0`
- `sol_source2_xs`
- `sol_source2_width`
- `sol_source_mask`
- `sol_source_y_taper`
- `sol_parallel_loss_on`
- `sol_parallel_loss_model`
- `sol_parallel_loss_q`
- `sol_parallel_loss_coeff`
- `sol_parallel_loss_lambda`
- `sol_parallel_loss_Te_floor`
- `sol_parallel_loss_vpar_on`
- `sol_parallel_loss_omega_on`
- `sol_sheath_omega_on`
- `sol_sheath_omega_coeff`
- `sol_sheath_phi_on`
- `sol_sheath_phi_model`
- `sol_sheath_phi_lambda`
- `sol_sheath_phi_coeff`
- `sol_sheath_phi_dissipation_on`
- `sol_sheath_phi_Te_floor`
- `sol_sheath_phi_clip`
- `sol_sheath_phi_implicit`
- `sol_sheath_phi_implicit_solver`
- `sol_sheath_phi_implicit_rtol`
- `sol_sheath_phi_implicit_atol`
- `sol_sheath_phi_implicit_maxiter`
- `sol_sheath_phi_implicit_restart`
- `sol_edge_relax_on`
- `sol_edge_relax_nu`
- `sol_edge_n_right`
- `sol_edge_Te_right`
- `sol_edge_relax_apply_y`
- `sol_omega_bc_dirichlet_on`
- `sol_omega_bc_value`
- `sol_omega_bc_nu`
- `sol_omega_bc_apply_y`
- `sol_vpar_bc_dirichlet_on`
- `sol_vpar_bc_value`
- `sol_vpar_bc_nu`
- `sol_phi_bc_on`
- `sol_phi_bc_lambda`

### `[closures].neutrals`

- `enabled`
- `Dn0`
- `n_background`
- `n_floor`
- `N_floor`
- `nu_ion`
- `nu_rec`
- `S0`
- `nu_sink`
- `nu_cx_omega`

### `[closures].line_bcs`

- `enabled`
- `n`
- `omega`
- `vpar_e`
- `vpar_i`
- `Te`
- `Ti`
- `psi`

### `[numerics]`

- `bracket`
- `bracket_zero_mean`
- `exb_scale`
- `exb_y_scale`
- `perp_operator`
- `parallel_z_mode`
- `parallel_limiter`
- `parallel_flux_conservative`
- `parallel_use_sheath_targets`
- `poisson_scale`
- `poisson_metric_on`
- `poisson`
- `poisson_force_spectral_when_periodic`
- `poisson_force_fd_fft_when_nonperiodic`
- `poisson_preconditioner`
- `poisson_cg_maxiter`
- `poisson_cg_tol`
- `poisson_cg_atol`
- `poisson_gauge_epsilon`
- `poisson_maxiter`
- `poisson_tol`
- `dealias_on`
- `polarization_cg_maxiter`
- `polarization_cg_tol`
- `polarization_cg_atol`
- `polarization_preconditioner`
- `polarization_precond_shift`
- `use_target_aware_dpar`
- `target_scheme`
- `operator_split_on`
- `operator_conservative_on`
- `operator_source_on`
- `operator_dissipative_on`
- `term_schedule`
- `term_schedule_preset`
- `phi_relax_in_rhs`

### `[bc]`

- `region_bc_on`
- `bc_enforce_nu`
- `bc_enforce_nu_n`
- `bc_enforce_nu_omega`
- `bc_enforce_nu_vpar_e`
- `bc_enforce_nu_vpar_i`
- `bc_enforce_nu_Te`
- `bc_enforce_nu_Ti`
- `bc_enforce_nu_psi`
- `bc_enforce_nu_phi`
- `bc_n`
- `bc_omega`
- `bc_vpar_e`
- `bc_vpar_i`
- `bc_Te`
- `bc_Ti`
- `bc_psi`
- `bc_phi`
- `perp_bc`
- `perp_bc_nu`

### Initial mixmode keys

`[initial]` supports two mixmode paths:

- `n_profile = "gaussian_mixmode"` with `mixmode_amp`, `mixmode_terms`, `mixmode_mode`
  to build a deterministic density perturbation from the selected profile.
- `n_mixmode_amp`, `n_mixmode_terms`, `n_mixmode_mode`, `n_mixmode_seed` to overlay
  deterministic mixmode perturbations on top of any base density profile (for example,
  a linear equilibrium profile plus a Hermes-style `x-z` perturbation).

### Pressure-consistent temperature initialization

For Hermes-equivalent initialization, you can define density and pressure profiles
independently and derive temperature from `Te = p / n`:

- `p_profile`: pressure profile (`linear_x`, `parabolic_x`, `gaussian_x`).
- `p_profile_*`: coefficients (`offset/slope/xref`, `a0/a1/a2/xref`, or `amp/width/x0`).
- `Te_profile = "from_pressure"` (or `Te_pressure_consistent = true`): force
  temperature initialization to use `p_profile / n_profile`, including deterministic
  mixmode perturbations applied to `n`.

---

## Output File (`.npz`)

When `--output` is provided (or when `return_numpy = true`), diagnostics are saved into
a NumPy archive. Common keys include:

- `times`: saved diagnostic times (1D array).
- `t`: final time (float).
- `snapshot_n`, `snapshot_Te`, `snapshot_Ti`, `snapshot_omega`, `snapshot_phi`,
  `snapshot_vpar_e`, `snapshot_vpar_i`, `snapshot_psi`: final‑time snapshots.
- `snapshots_*`: time series of saved fields when `time.save_fields = true` and
  `time.snapshot_fields` lists the field. These arrays are shaped `(nsave, ...)`.
- `rms_n`, `rms_Te`, `rms_omega`, `rms_phi`: RMS time series for scalar diagnostics.
- `point_n`, `point_Te`, `point_phi`: time series at a fixed probe index.

Additional arrays may be present when `trace_stats = true` or `trace_enstrophy = true`.
All outputs are normalized unless `normalization.enabled = true`, in which case the
normalization block maps physical inputs into those normalized units.

---

## Tips

- Use `time.diag_mode = "basic"` to skip Poisson solves in diagnostics when only RMS values
  are needed.
- Set `time.diag_phi_use_guess = true` to reuse a carried `phi` and avoid extra Poisson work.
- For long runs, use `time.remat = true` to reduce memory and keep differentiability.
