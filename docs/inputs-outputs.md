# Inputs & Outputs

This page is the full argument reference for the CLI and Python API. Defaults reflect the
current code. If you add new knobs, update this page in the same commit.

## CLI argument reference

### `jaxdrb-scan` (linear 1D ky scan)

| Argument | Default | Notes |
| --- | --- | --- |
| `--model` | `DEFAULT_MODEL` | Model registry key from `jaxdrb.models.registry` |
| `--geom` | required | `slab`, `slab-open`, `tabulated`, `tokamak`, `tokamak-open`, `salpha`, `salpha-open` |
| `--geom-file` | `None` | Required when `--geom tabulated` |
| `--nl` | `64` | Field-line grid points |
| `--length` | `2π` | Field-line length |
| `--shat` | `0.8` | Magnetic shear |
| `--q` | `1.4` | Safety factor |
| `--R0` | `1.0` | Major radius |
| `--epsilon` | `0.18` | Inverse aspect ratio |
| `--alpha` | `0.0` | Ballooning parameter |
| `--curvature0` | `0.0` | Constant curvature override |
| `--omega-n` | `0.8` | Density-gradient drive |
| `--omega-Te` | `0.0` | Te-gradient drive |
| `--omega-Ti` | `0.0` | Ti-gradient drive |
| `--eta` | `1.0` | Resistivity-like coefficient |
| `--me-hat` | `0.05` | Electron inertia knob |
| `--beta` | `0.0` | EM beta |
| `--tau-i` | `0.0` | Ti/Te |
| `--no-curvature` | `False` | Disable curvature drive |
| `--no-boussinesq` | `False` | Use non-Boussinesq polarization |
| `--non-boussinesq-perturbed-density` | `False` | Use `n0 + Re[n]` in polarization |
| `--Dn` | `0.01` | Perpendicular diffusion on `n` |
| `--DOmega` | `0.01` | Perpendicular diffusion on `Omega` |
| `--DTe` | `0.01` | Perpendicular diffusion on `Te` |
| `--DTi` | `0.01` | Perpendicular diffusion on `Ti` |
| `--Dpsi` | `0.0` | Perpendicular diffusion on `psi` |
| `--kperp2-min` | `1e-6` | Polarization safety floor |
| `--line-bc` | `none` | `none`, `dirichlet`, `neumann` |
| `--line-bc-value` | `0.0` | Dirichlet value for `--line-bc` |
| `--line-bc-grad` | `0.0` | Neumann gradient for `--line-bc` |
| `--line-bc-nu` | `0.0` | BC relaxation rate |
| `--chi-par-Te` | `0.0` | Parallel Te conduction |
| `--chi-par-Ti` | `0.0` | Parallel Ti conduction |
| `--nu-par-e` | `0.0` | Parallel electron flow diffusion |
| `--nu-par-i` | `0.0` | Parallel ion flow diffusion |
| `--nu-sink-n` | `0.0` | Volumetric sink on `n` |
| `--nu-sink-Te` | `0.0` | Volumetric sink on `Te` |
| `--nu-sink-vpar` | `0.0` | Volumetric sink on `vpar` |
| `--braginskii` | `False` | Enable equilibrium-based Braginskii scalings |
| `--braginskii-state-dependent` | `False` | Evaluate Braginskii on evolving fields |
| `--braginskii-Tref` | `1.0` | Reference temperature |
| `--braginskii-T-floor` | `1e-3` | Temperature floor |
| `--braginskii-T-smooth` | `1e-3` | Temperature smoothing |
| `--no-braginskii-eta` | `False` | Disable η scaling |
| `--no-braginskii-kappa-e` | `False` | Disable χ||,e scaling |
| `--no-braginskii-kappa-i` | `False` | Disable χ||,i scaling |
| `--no-braginskii-visc-e` | `False` | Disable ν||,e scaling |
| `--no-braginskii-visc-i` | `False` | Disable ν||,i scaling |
| `--eq-n0` | `1.0` | Equilibrium density |
| `--eq-Te0` | `1.0` | Equilibrium temperature |
| `--sheath` | `False` | Alias for `--sheath-bc` |
| `--no-sheath-bc` | `False` | Disable MPSE BCs for open geometries |
| `--sheath-bc` | `False` | Enable MPSE BCs |
| `--sheath-bc-model` | `simple` | `simple` or `loizu2012` |
| `--sheath-bc-nu-factor` | `1.0` | BC relaxation rate factor |
| `--sheath-cos2` | `1.0` | Loizu vorticity BC `cos^2` factor |
| `--sheath-lambda` | `3.28` | Floating potential constant |
| `--sheath-delta` | `0.0` | Ion transmission correction |
| `--sheath-loss` | `False` | Enable volumetric end-loss proxy |
| `--sheath-loss-nu-factor` | `1.0` | Multiplier for end-loss rate |
| `--sheath-end-damp` | `True` | Boundary-localized damping toggle |
| `--sheath-heat` | `False` | Enable sheath heat transmission |
| `--sheath-gamma-auto` | `True` | Auto set `gamma_e` |
| `--sheath-gamma-e` | `0.0` | Electron heat transmission factor |
| `--sheath-gamma-i` | `3.5` | Ion heat transmission factor |
| `--sheath-see` | `False` | Secondary electron emission (SEE) |
| `--sheath-see-yield` | `0.0` | SEE yield δ |
| `--ky-min` | required | Minimum ky |
| `--ky-max` | required | Maximum ky |
| `--nky` | `32` | Number of ky points |
| `--kx` | `0.0` | Single kx |
| `--out` | required | Output directory |
| `--arnoldi-m` | `40` | Arnoldi Krylov dimension |
| `--arnoldi-max-m` | `None` | Optional cap on Arnoldi m |
| `--arnoldi-tol` | `1e-3` | Arnoldi relative residual tolerance |
| `--nev` | `6` | Number of Ritz values |
| `--tmax` | `30.0` | IVP integration end time |
| `--dt0` | `0.01` | IVP initial step size |
| `--nsave` | `200` | IVP samples |
| `--seed` | `0` | RNG seed |
| `--no-initial-value` | `False` | Skip IVP growth-rate estimate |

### `jaxdrb-scan2d` (linear kx-ky scan)

| Argument | Default | Notes |
| --- | --- | --- |
| `--model` | `DEFAULT_MODEL` | Model registry key from `jaxdrb.models.registry` |
| `--geom` | required | `slab`, `slab-open`, `tabulated`, `tokamak`, `tokamak-open`, `salpha`, `salpha-open` |
| `--geom-file` | `None` | Required when `--geom tabulated` |
| `--nl` | `64` | Field-line grid points |
| `--length` | `2π` | Field-line length |
| `--shat` | `0.8` | Magnetic shear |
| `--q` | `1.4` | Safety factor |
| `--R0` | `1.0` | Major radius |
| `--epsilon` | `0.18` | Inverse aspect ratio |
| `--alpha` | `0.0` | Ballooning parameter |
| `--curvature0` | `0.0` | Constant curvature override |
| `--omega-n` | `0.8` | Density-gradient drive |
| `--omega-Te` | `0.0` | Te-gradient drive |
| `--omega-Ti` | `0.0` | Ti-gradient drive |
| `--eta` | `1.0` | Resistivity-like coefficient |
| `--me-hat` | `0.05` | Electron inertia knob |
| `--beta` | `0.0` | EM beta |
| `--tau-i` | `0.0` | Ti/Te |
| `--no-curvature` | `False` | Disable curvature drive |
| `--no-boussinesq` | `False` | Use non-Boussinesq polarization |
| `--non-boussinesq-perturbed-density` | `False` | Use `n0 + Re[n]` in polarization |
| `--Dn` | `0.01` | Perpendicular diffusion on `n` |
| `--DOmega` | `0.01` | Perpendicular diffusion on `Omega` |
| `--DTe` | `0.01` | Perpendicular diffusion on `Te` |
| `--DTi` | `0.01` | Perpendicular diffusion on `Ti` |
| `--Dpsi` | `0.0` | Perpendicular diffusion on `psi` |
| `--kperp2-min` | `1e-6` | Polarization safety floor |
| `--chi-par-Te` | `0.0` | Parallel Te conduction |
| `--chi-par-Ti` | `0.0` | Parallel Ti conduction |
| `--nu-par-e` | `0.0` | Parallel electron flow diffusion |
| `--nu-par-i` | `0.0` | Parallel ion flow diffusion |
| `--nu-sink-n` | `0.0` | Volumetric sink on `n` |
| `--nu-sink-Te` | `0.0` | Volumetric sink on `Te` |
| `--nu-sink-vpar` | `0.0` | Volumetric sink on `vpar` |
| `--braginskii` | `False` | Enable equilibrium-based Braginskii scalings |
| `--braginskii-state-dependent` | `False` | Evaluate Braginskii on evolving fields |
| `--braginskii-Tref` | `1.0` | Reference temperature |
| `--braginskii-T-floor` | `1e-3` | Temperature floor |
| `--braginskii-T-smooth` | `1e-3` | Temperature smoothing |
| `--no-braginskii-eta` | `False` | Disable η scaling |
| `--no-braginskii-kappa-e` | `False` | Disable χ||,e scaling |
| `--no-braginskii-kappa-i` | `False` | Disable χ||,i scaling |
| `--no-braginskii-visc-e` | `False` | Disable ν||,e scaling |
| `--no-braginskii-visc-i` | `False` | Disable ν||,i scaling |
| `--eq-n0` | `1.0` | Equilibrium density |
| `--eq-Te0` | `1.0` | Equilibrium temperature |
| `--line-bc` | `none` | `none`, `dirichlet`, `neumann` |
| `--line-bc-value` | `0.0` | Dirichlet value for `--line-bc` |
| `--line-bc-grad` | `0.0` | Neumann gradient for `--line-bc` |
| `--line-bc-nu` | `0.0` | BC relaxation rate |
| `--sheath` | `False` | Alias for `--sheath-bc` |
| `--no-sheath-bc` | `False` | Disable MPSE BCs for open geometries |
| `--sheath-bc` | `False` | Enable MPSE BCs |
| `--sheath-bc-nu-factor` | `1.0` | BC relaxation rate factor |
| `--sheath-lambda` | `3.28` | Floating potential constant |
| `--sheath-delta` | `0.0` | Ion transmission correction |
| `--sheath-loss` | `False` | Enable volumetric end-loss proxy |
| `--sheath-loss-nu-factor` | `1.0` | Multiplier for end-loss rate |
| `--sheath-end-damp` | `True` | Boundary-localized damping toggle |
| `--sheath-heat` | `False` | Enable sheath heat transmission |
| `--sheath-gamma-auto` | `True` | Auto set `gamma_e` |
| `--sheath-gamma-e` | `0.0` | Electron heat transmission factor |
| `--sheath-gamma-i` | `3.5` | Ion heat transmission factor |
| `--sheath-see` | `False` | Secondary electron emission (SEE) |
| `--sheath-see-yield` | `0.0` | SEE yield δ |
| `--ky-min` | required | Minimum ky |
| `--ky-max` | required | Maximum ky |
| `--nky` | `32` | Number of ky points |
| `--kx-min` | required | Minimum kx |
| `--kx-max` | required | Maximum kx |
| `--nkx` | `33` | Number of kx points |
| `--out` | required | Output directory |
| `--arnoldi-m` | `40` | Arnoldi Krylov dimension |
| `--arnoldi-max-m` | `None` | Optional cap on Arnoldi m |
| `--arnoldi-tol` | `1e-3` | Arnoldi relative residual tolerance |
| `--nev` | `6` | Number of Ritz values |
| `--seed` | `0` | RNG seed |

### `jaxdrb-hw2d` (nonlinear 2D HW)

| Argument | Default | Notes |
| --- | --- | --- |
| `--nx` | `96` | Grid points in x |
| `--ny` | `96` | Grid points in y |
| `--Lx` | `2π` | Domain size x |
| `--Ly` | `2π` | Domain size y |
| `--dt` | `0.05` | Base time step |
| `--tmax` | `40.0` | Final time |
| `--save-stride` | `20` | Save every N steps |
| `--solver` | `tsit5` | Diffrax solver name |
| `--fixed-step` | `False` | Disable adaptive stepping |
| `--rtol` | `1e-5` | Diffrax relative tolerance |
| `--atol` | `1e-8` | Diffrax absolute tolerance |
| `--max-steps` | `300000` | Max Diffrax steps |
| `--progress` | `False` | Progress meter |
| `--seed` | `0` | RNG seed |
| `--amp` | `1e-3` | Initial-condition amplitude |
| `--kappa` | `1.0` | Background-gradient drive |
| `--alpha` | `0.5` | Adiabaticity / coupling |
| `--Dn` | `2e-4` | Density diffusion |
| `--DOmega` | `2e-4` | Vorticity diffusion |
| `--bracket` | `spectral` | `spectral`, `arakawa`, `centered` |
| `--poisson` | `spectral` | `spectral`, `cg_fd` |
| `--no-dealias` | `False` | Disable dealiasing |
| `--bc-x` | `periodic` | `periodic`, `dirichlet`, `neumann` |
| `--bc-y` | `periodic` | `periodic`, `dirichlet`, `neumann` |
| `--bc-value-x` | `0.0` | Dirichlet value at x boundary |
| `--bc-value-y` | `0.0` | Dirichlet value at y boundary |
| `--bc-grad-x` | `0.0` | Neumann gradient at x boundary |
| `--bc-grad-y` | `0.0` | Neumann gradient at y boundary |
| `--bc-enforce-nu` | `0.0` | Boundary relaxation rate |
| `--neutrals` | `False` | Enable neutral coupling |
| `--Dn0` | `1e-3` | Neutral diffusion |
| `--nu-ion` | `0.2` | Ionization rate |
| `--nu-rec` | `0.02` | Recombination rate |
| `--n-background` | `1.0` | Background density for neutral rates |
| `--neutral-source` | `0.0` | Uniform neutral source |
| `--neutral-sink` | `0.0` | Uniform neutral sink |
| `--nu-cx-omega` | `0.0` | Charge-exchange vorticity drag |
| `--out` | `out_hw2d_cli` | Output directory |

Nonlinear DRB2D and FCI/3D examples use per-script `argparse` in `examples/`; run `python <script> --help` for the full list.
This page documents the Python parameter classes that those examples use.

## Python API reference

### Geometry inputs

`jaxdrb` operates on a field line coordinate `l` (periodic by default). Geometry providers include
`SlabGeometry`, `CircularTokamakGeometry`, and `SAlphaGeometry`, plus open-field-line variants.
Tabulated geometries use `.npz` files with keys `l`, `gxx`, `gxy`, `gyy` and optional `curv_x`, `curv_y`,
`dpar_factor`, `B`.

### `Grid2D.make`

| Name | Default | Notes |
| --- | --- | --- |
| `nx` | required | Grid points in x |
| `ny` | required | Grid points in y |
| `Lx` | required | Domain length x |
| `Ly` | required | Domain length y |
| `dealias` | `True` | 2/3-rule dealias mask (periodic only) |
| `bc_x` | `periodic` | `periodic`, `dirichlet`, `neumann` |
| `bc_y` | `periodic` | `periodic`, `dirichlet`, `neumann` |
| `bc_value_x` | `0.0` | Dirichlet value x |
| `bc_value_y` | `0.0` | Dirichlet value y |
| `bc_grad_x` | `0.0` | Neumann gradient x |
| `bc_grad_y` | `0.0` | Neumann gradient y |

### Diffrax integrators

`diffeqsolve(rhs, y0, t0, t1, dt0, save_ts, solver, adaptive, rtol, atol, max_steps, progress)`

| Name | Default | Notes |
| --- | --- | --- |
| `rhs` | required | `rhs(t, y) -> dy/dt` |
| `y0` | required | Initial state |
| `t0` | required | Start time |
| `t1` | required | End time |
| `dt0` | required | Initial step size |
| `save_ts` | `None` | Save at specific times |
| `solver` | `tsit5` | One of `tsit5`, `dopri5`, `dopri8`, `euler`, `implicit_euler`, `kvaerno3`, `kvaerno4`, `kvaerno5`, `kencarp3`, `kencarp4`, `kencarp5` |
| `adaptive` | `True` | Use adaptive PID control |
| `rtol` | `1e-5` | Relative tolerance |
| `atol` | `1e-8` | Absolute tolerance |
| `max_steps` | `200000` | Max internal steps |
| `progress` | `None` | Defaults to `JAXDRB_PROGRESS` env var |

`diffeqsolve_fixed_steps(rhs, y0, t0, dt, nsteps, solver, save_every, max_steps, progress)`

| Name | Default | Notes |
| --- | --- | --- |
| `rhs` | required | `rhs(t, y) -> dy/dt` |
| `y0` | required | Initial state |
| `t0` | required | Start time |
| `dt` | required | Fixed step size |
| `nsteps` | required | Number of steps |
| `solver` | `dopri5` | Diffrax solver name |
| `save_every` | `1` | Save cadence in steps |
| `max_steps` | `None` | Optional cap on internal steps |
| `progress` | `None` | Defaults to `JAXDRB_PROGRESS` env var |

### `NeutralParams`

| Name | Default | Notes |
| --- | --- | --- |
| `enabled` | `False` | Toggle neutral coupling |
| `Dn0` | `0.0` | Neutral diffusion |
| `n_background` | `1.0` | Background density used in rates |
| `n_floor` | `1e-6` | Density floor |
| `N_floor` | `1e-6` | Neutral floor |
| `nu_ion` | `0.0` | Ionization rate |
| `nu_rec` | `0.0` | Recombination rate |
| `S0` | `0.0` | Uniform neutral source |
| `nu_sink` | `0.0` | Uniform neutral sink |
| `nu_cx_omega` | `0.0` | Charge-exchange vorticity drag |

### `HW2DParams`

| Name | Default | Notes |
| --- | --- | --- |
| `kappa` | `1.0` | Background-gradient drive |
| `alpha` | `1.0` | Adiabaticity / resistive coupling |
| `Dn` | `1e-3` | Density diffusion |
| `DOmega` | `1e-3` | Vorticity diffusion |
| `nu4_n` | `0.0` | Hyperdiffusion on `n` |
| `nu4_omega` | `0.0` | Hyperdiffusion on `omega` |
| `bracket` | `arakawa` | `spectral`, `arakawa`, `centered` |
| `poisson` | `spectral` | `spectral`, `cg_fd` |
| `dealias_on` | `True` | Dealiasing for spectral bracket |
| `k2_min` | `1e-12` | Poisson k2 floor |
| `bc_enforce_nu` | `0.0` | Boundary relaxation rate |
| `alpha_nonzonal_only` | `False` | Apply coupling only to ky≠0 |
| `neutrals` | `NeutralParams()` | Neutral coupling parameters |

### `DRB2DParams`

| Name | Default | Notes |
| --- | --- | --- |
| `omega_n` | `0.0` | Density-gradient drive |
| `omega_Te` | `0.0` | Te-gradient drive |
| `omega_drive_mask` | `all` | `all`, `closed`, `open` |
| `kpar` | `0.0` | Constant k_par coupling |
| `eta` | `0.0` | Resistivity-like coupling |
| `me_hat` | `0.2` | Electron inertia knob |
| `curvature_on` | `False` | Enable curvature drive |
| `curvature_coeff` | `0.0` | Curvature coefficient |
| `curvature_model` | `slab` | `slab` or tokamak-like |
| `curvature_theta_scale` | `None` | Optional poloidal scaling |
| `curvature_scale` | `None` | Optional scaling override |
| `boussinesq` | `True` | Boussinesq polarization |
| `n0` | `1.0` | Reference density |
| `n0_min` | `1e-6` | Density floor |
| `n0_max` | `None` | Density cap |
| `non_boussinesq_perturbed_density_on` | `False` | Use `n0 + Re[n]` |
| `log_n` | `False` | Evolve `ln n` |
| `log_Te` | `False` | Evolve `ln Te` |
| `log_n_clip` | `50.0` | Clamp for log variables |
| `log_Te_clip` | `50.0` | Clamp for log variables |
| `Dn` | `0.0` | Density diffusion |
| `DOmega` | `0.0` | Vorticity diffusion |
| `DTe` | `0.0` | Te diffusion |
| `Dn4` | `0.0` | Hyperdiffusion on `n` |
| `DOmega4` | `0.0` | Hyperdiffusion on `omega` |
| `DTe4` | `0.0` | Hyperdiffusion on `Te` |
| `mu_zonal_omega` | `0.0` | Zonal vorticity drag |
| `mu_lin_n` | `0.0` | Linear damping on `n` |
| `mu_lin_omega` | `0.0` | Linear damping on `omega` |
| `mu_lin_vpar_e` | `0.0` | Linear damping on `vpar_e` |
| `mu_lin_vpar_i` | `0.0` | Linear damping on `vpar_i` |
| `mu_lin_Te` | `0.0` | Linear damping on `Te` |
| `bracket` | `arakawa` | `spectral`, `arakawa`, `centered` |
| `bracket_zero_mean` | `False` | Enforce zero-mean bracket |
| `exb_scale` | `1.0` | Scale for ExB advection |
| `poisson` | `spectral` | `spectral`, `cg_fd`, `mixed_fft` |
| `poisson_preconditioner` | `auto` | `auto`, `spectral`, `jacobi`, `none` |
| `poisson_cg_maxiter` | `300` | Poisson CG max iterations |
| `poisson_cg_tol` | `1e-8` | Poisson CG tol |
| `poisson_cg_atol` | `0.0` | Poisson CG atol |
| `poisson_gauge_epsilon` | `None` | Gauge lifting epsilon |
| `dealias_on` | `True` | Dealiasing for spectral bracket |
| `k2_min` | `1e-12` | Poisson k2 floor |
| `bc_enforce_nu` | `0.0` | Boundary relaxation rate |
| `polarization_cg_maxiter` | `400` | Non-Bouss CG max iterations |
| `polarization_cg_tol` | `1e-8` | Non-Bouss CG tol |
| `polarization_cg_atol` | `0.0` | Non-Bouss CG atol |
| `polarization_preconditioner` | `auto` | `auto`, `spectral`, `jacobi`, `none` |
| `polarization_precond_shift` | `1e-12` | SPD preconditioner shift |
| `alpha_Te_ohm` | `1.71` | Thermal force coefficient |
| `operator_split_on` | `False` | Enable operator split |
| `operator_conservative_on` | `True` | Include conservative part |
| `operator_source_on` | `True` | Include source part |
| `operator_dissipative_on` | `True` | Include dissipative part |
| `neutrals` | `NeutralParams()` | Neutral coupling parameters |
| `sol_on` | `False` | Enable closed→open mask |
| `sol_xs` | `0.0` | LCFS location |
| `sol_width` | `0.05` | Transition width |
| `sol_open_left` | `False` | Open region on left if True |
| `sol_mask_y_taper` | `0.0` | Optional y taper |
| `sol_n_core` | `1.0` | Target core density |
| `sol_n_sol` | `0.2` | Target SOL density |
| `sol_Te_core` | `1.0` | Target core Te |
| `sol_Te_sol` | `0.2` | Target SOL Te |
| `sol_relax_core` | `0.2` | Relaxation rate in core |
| `sol_relax_open` | `0.6` | Relaxation rate in open region |
| `sol_sink_open_n` | `0.0` | Open-region sink on n |
| `sol_sink_open_Te` | `0.0` | Open-region sink on Te |
| `sol_sink_open_omega` | `0.0` | Open-region sink on omega |
| `sol_sink_open_omega_mode` | `local` | `local` or `global` |
| `sol_sink_open_vpar` | `0.0` | Open-region sink on vpar |
| `sol_nonlinear_open_scale` | `1.0` | Scale ExB in open region |
| `sol_n_floor` | `0.0` | n floor for SOL losses |
| `sol_Te_floor` | `0.0` | Te floor for SOL losses |
| `sol_source_n0` | `0.0` | Primary n source amplitude |
| `sol_source_Te0` | `0.0` | Primary Te source amplitude |
| `sol_source_xs` | `0.0` | Primary source center |
| `sol_source_width` | `1.0` | Primary source width |
| `sol_source2_n0` | `0.0` | Secondary n source amplitude |
| `sol_source2_Te0` | `0.0` | Secondary Te source amplitude |
| `sol_source2_xs` | `0.0` | Secondary source center |
| `sol_source2_width` | `1.0` | Secondary source width |
| `sol_source_mask` | `all` | `all`, `closed`, `open` |
| `sol_source_y_taper` | `0.0` | Optional y taper for sources |
| `sol_parallel_loss_on` | `False` | Enable parallel-loss closure |
| `sol_parallel_loss_model` | `bohm` | `bohm`, `bohm_exp`, `bohm_linear` |
| `sol_parallel_loss_q` | `4.0` | Connection length factor |
| `sol_parallel_loss_coeff` | `1.0` | Loss prefactor |
| `sol_parallel_loss_lambda` | `3.0` | Floating potential constant |
| `sol_parallel_loss_Te_floor` | `1e-6` | Te floor for losses |
| `sol_parallel_loss_vpar_on` | `False` | Loss term on vpar |
| `sol_parallel_loss_omega_on` | `False` | Loss term on omega |
| `sol_sheath_omega_on` | `False` | Sheath omega clamp |
| `sol_sheath_omega_coeff` | `1.0` | Sheath omega coefficient |
| `sol_sheath_phi_on` | `False` | Sheath phi clamp |
| `sol_sheath_phi_model` | `exp` | `exp` or `linear` |
| `sol_sheath_phi_lambda` | `3.0` | Floating potential constant |
| `sol_sheath_phi_coeff` | `1.0` | Sheath phi coefficient |
| `sol_sheath_phi_Te_floor` | `1e-6` | Te floor for phi clamp |
| `sol_sheath_phi_clip` | `10.0` | Clip for phi clamp |
| `sol_gbs_bc_on` | `False` | GBS-style radial BC relaxation |
| `sol_gbs_bc_nu` | `0.0` | GBS BC relaxation rate |
| `sol_gbs_n_right` | `0.1` | Right boundary n target |
| `sol_gbs_Te_right` | `0.1` | Right boundary Te target |
| `sol_gbs_apply_y` | `True` | Apply GBS BC in y |
| `sol_omega_bc_dirichlet_on` | `False` | Dirichlet omega BC |
| `sol_omega_bc_value` | `0.0` | Dirichlet omega value |
| `sol_omega_bc_nu` | `1.0` | Dirichlet omega rate |
| `sol_omega_bc_apply_y` | `False` | Apply omega BC in y |
| `sol_vpar_bc_dirichlet_on` | `False` | Dirichlet vpar BC |
| `sol_vpar_bc_value` | `0.0` | Dirichlet vpar value |
| `sol_vpar_bc_nu` | `1.0` | Dirichlet vpar rate |
| `sol_phi_bc_on` | `False` | Phi clamp BC |
| `sol_phi_bc_lambda` | `3.0` | Phi clamp constant |

### `DRB2DHotIonParams`

| Name | Default | Notes |
| --- | --- | --- |
| `omega_n` | `0.0` | Density-gradient drive |
| `omega_Te` | `0.0` | Te-gradient drive |
| `omega_Ti` | `0.0` | Ti-gradient drive |
| `kpar` | `0.0` | Constant k_par coupling |
| `eta` | `0.0` | Resistivity-like coupling |
| `me_hat` | `0.2` | Electron inertia knob |
| `tau_i` | `1.0` | Ti/Te |
| `alpha_Te_ohm` | `1.71` | Thermal force coefficient |
| `alpha_Ti` | `1.0` | Ti pressure coupling |
| `curvature_on` | `False` | Curvature drive |
| `curvature_coeff` | `0.0` | Curvature coefficient |
| `boussinesq` | `True` | Boussinesq polarization |
| `n0` | `1.0` | Reference density |
| `n0_min` | `1e-6` | Density floor |
| `non_boussinesq_perturbed_density_on` | `False` | Use `n0 + Re[n]` |
| `Dn` | `0.0` | Density diffusion |
| `DOmega` | `0.0` | Vorticity diffusion |
| `DTe` | `0.0` | Te diffusion |
| `DTi` | `0.0` | Ti diffusion |
| `Dn4` | `0.0` | Hyperdiffusion on `n` |
| `DOmega4` | `0.0` | Hyperdiffusion on `omega` |
| `DTe4` | `0.0` | Hyperdiffusion on `Te` |
| `DTi4` | `0.0` | Hyperdiffusion on `Ti` |
| `mu_zonal_omega` | `0.0` | Zonal vorticity drag |
| `bracket` | `arakawa` | `spectral`, `arakawa`, `centered` |
| `poisson` | `spectral` | `spectral`, `cg_fd` |
| `dealias_on` | `True` | Dealiasing |
| `k2_min` | `1e-12` | Poisson k2 floor |
| `bc_enforce_nu` | `0.0` | Boundary relaxation rate |
| `polarization_cg_maxiter` | `400` | Non-Bouss CG max iterations |
| `polarization_cg_tol` | `1e-8` | Non-Bouss CG tol |
| `polarization_cg_atol` | `0.0` | Non-Bouss CG atol |
| `operator_split_on` | `False` | Enable operator split |
| `operator_conservative_on` | `True` | Include conservative part |
| `operator_source_on` | `True` | Include source part |
| `operator_dissipative_on` | `True` | Include dissipative part |

### `DRB2DEMParams`

| Name | Default | Notes |
| --- | --- | --- |
| `omega_n` | `0.0` | Density-gradient drive |
| `omega_Te` | `0.0` | Te-gradient drive |
| `kpar` | `0.0` | Constant k_par coupling |
| `eta` | `0.0` | Resistivity-like coupling |
| `me_hat` | `0.2` | Electron inertia knob |
| `beta` | `0.0` | EM beta |
| `Dpsi` | `0.0` | Psi diffusion |
| `curvature_on` | `False` | Curvature drive |
| `curvature_coeff` | `0.0` | Curvature coefficient |
| `boussinesq` | `True` | Boussinesq polarization |
| `n0` | `1.0` | Reference density |
| `n0_min` | `1e-6` | Density floor |
| `non_boussinesq_perturbed_density_on` | `False` | Use `n0 + Re[n]` |
| `Dn` | `0.0` | Density diffusion |
| `DOmega` | `0.0` | Vorticity diffusion |
| `DTe` | `0.0` | Te diffusion |
| `Dn4` | `0.0` | Hyperdiffusion on `n` |
| `DOmega4` | `0.0` | Hyperdiffusion on `omega` |
| `DTe4` | `0.0` | Hyperdiffusion on `Te` |
| `Dpsi4` | `0.0` | Hyperdiffusion on `psi` |
| `mu_zonal_omega` | `0.0` | Zonal vorticity drag |
| `bracket` | `arakawa` | `spectral`, `arakawa`, `centered` |
| `poisson` | `spectral` | `spectral`, `cg_fd` |
| `dealias_on` | `True` | Dealiasing |
| `k2_min` | `1e-12` | Poisson k2 floor |
| `bc_enforce_nu` | `0.0` | Boundary relaxation rate |
| `polarization_cg_maxiter` | `400` | Non-Bouss CG max iterations |
| `polarization_cg_tol` | `1e-8` | Non-Bouss CG tol |
| `polarization_cg_atol` | `0.0` | Non-Bouss CG atol |
| `alpha_Te_ohm` | `1.71` | Thermal force coefficient |
| `operator_split_on` | `False` | Enable operator split |
| `operator_conservative_on` | `True` | Include conservative part |
| `operator_source_on` | `True` | Include source part |
| `operator_dissipative_on` | `True` | Include dissipative part |

### `DRBParams` (field-line models)

| Name | Default | Notes |
| --- | --- | --- |
| `omega_n` | `0.5` | Density-gradient drive |
| `omega_Te` | `0.0` | Te-gradient drive |
| `omega_Ti` | `0.0` | Ti-gradient drive |
| `eta` | `1.0` | Resistivity-like coefficient |
| `me_hat` | `1e-3` | Electron inertia knob |
| `beta` | `0.0` | EM beta |
| `Dpsi` | `0.0` | Psi diffusion |
| `tau_i` | `0.0` | Ti/Te |
| `DTi` | `0.02` | Ti diffusion |
| `curvature_on` | `True` | Curvature drive |
| `Dn` | `0.02` | Density diffusion |
| `DOmega` | `0.02` | Vorticity diffusion |
| `DTe` | `0.02` | Te diffusion |
| `chi_par_Te` | `0.0` | Parallel Te conduction |
| `chi_par_Ti` | `0.0` | Parallel Ti conduction |
| `nu_par_e` | `0.0` | Parallel e flow diffusion |
| `nu_par_i` | `0.0` | Parallel i flow diffusion |
| `nu_sink_n` | `0.0` | Volumetric sink on `n` |
| `nu_sink_Te` | `0.0` | Volumetric sink on `Te` |
| `nu_sink_vpar` | `0.0` | Volumetric sink on vpar |
| `kperp2_min` | `1e-6` | Polarization safety floor |
| `boussinesq` | `True` | Boussinesq polarization |
| `n0_min` | `1e-6` | Density floor |
| `non_boussinesq_perturbed_density_on` | `False` | Use `n0 + Re[n]` |
| `alpha_Te_ohm` | `1.71` | Thermal force coefficient |
| `braginskii_on` | `False` | Enable Braginskii scalings |
| `braginskii_state_dependent_on` | `False` | State-dependent Braginskii |
| `braginskii_eta_on` | `True` | Enable η scaling |
| `braginskii_kappa_e_on` | `True` | Enable χ||,e scaling |
| `braginskii_kappa_i_on` | `True` | Enable χ||,i scaling |
| `braginskii_visc_e_on` | `True` | Enable ν||,e scaling |
| `braginskii_visc_i_on` | `True` | Enable ν||,i scaling |
| `braginskii_Tref` | `1.0` | Reference temperature |
| `braginskii_T_floor` | `1e-3` | Temperature floor |
| `braginskii_T_smooth` | `1e-3` | Temperature smoothing |
| `sheath_bc_on` | `True` | Enable MPSE BCs |
| `sheath_bc_nu_factor` | `1.0` | BC relaxation factor |
| `sheath_bc_model` | `0` | `0` simple, `1` Loizu 2012 |
| `sheath_cos2` | `1.0` | Loizu vorticity BC factor |
| `sheath_bc_linearized` | `True` | Linearized MPSE |
| `sheath_lambda` | `3.28` | Floating potential constant |
| `sheath_delta` | `0.0` | Ion transmission correction |
| `sheath_Te_floor` | `1e-6` | Te floor for sheath |
| `sheath_heat_on` | `False` | Enable sheath heat transmission |
| `sheath_gamma_auto` | `True` | Auto set `gamma_e` |
| `sheath_gamma_e` | `0.0` | Electron heat transmission factor |
| `sheath_gamma_i` | `3.5` | Ion heat transmission factor |
| `sheath_see_on` | `False` | SEE toggle |
| `sheath_see_yield` | `0.0` | SEE yield |
| `sheath_end_damp_on` | `True` | Boundary-localized damping |
| `sheath_loss_on` | `False` | Volumetric end-loss proxy |
| `sheath_loss_nu_factor` | `1.0` | End-loss multiplier |
| `sheath_on` | `False` | Deprecated alias |
| `sheath_nu_factor` | `1.0` | Deprecated alias |
| `line_bcs` | `LineBCs.disabled()` | User BCs on `l` |
| `operator_split_on` | `False` | Enable operator split |
| `operator_conservative_on` | `True` | Include conservative part |
| `operator_source_on` | `True` | Include source part |
| `operator_dissipative_on` | `True` | Include dissipative part |

### `LineBCs`

| Name | Default | Notes |
| --- | --- | --- |
| `enabled` | `False` | Toggle user BCs |
| `n` | periodic | `BC1D` for density |
| `omega` | periodic | `BC1D` for vorticity |
| `vpar_e` | periodic | `BC1D` for electron parallel flow |
| `vpar_i` | periodic | `BC1D` for ion parallel flow |
| `Te` | periodic | `BC1D` for electron temperature |
| `Ti` | periodic | `BC1D` for ion temperature |
| `psi` | periodic | `BC1D` for EM potential |

## FCI and 3D parameter reference

### `ZPlaneFCIConfig`

| Name | Default | Notes |
| --- | --- | --- |
| `x0` | required | Perpendicular origin (x) |
| `y0` | required | Perpendicular origin (y) |
| `dx` | required | Perpendicular spacing (x) |
| `dy` | required | Perpendicular spacing (y) |
| `nx` | required | Grid points x |
| `ny` | required | Grid points y |
| `z0` | required | First plane z |
| `dz` | required | Plane spacing |
| `nz` | required | Number of planes |
| `periodic_z` | `False` | Periodic z planes |
| `open_field_line` | `False` | Encode plate hits in maps |
| `cell_centered` | `False` | Cell-centered plates |

### `EssosToroidalFCIConfig`

| Name | Default | Notes |
| --- | --- | --- |
| `R0` | required | Plane origin R |
| `Z0` | required | Plane origin Z |
| `dR` | required | Plane spacing R |
| `dZ` | required | Plane spacing Z |
| `nR` | required | Plane points R |
| `nZ` | required | Plane points Z |
| `phi0` | required | First toroidal angle |
| `dphi` | required | Toroidal spacing |
| `nphi` | required | Number of planes |
| `periodic_R` | `False` | Periodic R plane |
| `periodic_Z` | `False` | Periodic Z plane |
| `periodic_phi` | `True` | Periodic toroidal planes |
| `open_field_line` | `True` | Encode plate hits in maps |
| `cell_centered` | `True` | Cell-centered plates |
| `R_min` | `None` | Target bounding box min R |
| `R_max` | `None` | Target bounding box max R |
| `Z_min` | `None` | Target bounding box min Z |
| `Z_max` | `None` | Target bounding box max Z |

### `FCISlabGrid.make`

| Name | Default | Notes |
| --- | --- | --- |
| `nx` | required | Perpendicular grid points x |
| `ny` | required | Perpendicular grid points y |
| `nz` | required | Number of planes |
| `Lx` | required | Domain length x |
| `Ly` | required | Domain length y |
| `Lz` | required | Parallel length |
| `Bx` | required | Magnetic field x |
| `By` | required | Magnetic field y |
| `Bz` | required | Magnetic field z |
| `open_field_line` | `True` | Use open field lines |
| `cell_centered` | `False` | Cell-centered planes |

### `FCISlabGrid.from_maps`

| Name | Default | Notes |
| --- | --- | --- |
| `x0` | required | Plane origin x |
| `y0` | required | Plane origin y |
| `dx` | required | Plane spacing x |
| `dy` | required | Plane spacing y |
| `nx` | required | Plane points x |
| `ny` | required | Plane points y |
| `l` | required | 1D parallel coordinate array |
| `map_fwd` | required | Forward FCI map |
| `map_bwd` | required | Backward FCI map |
| `open_field_line` | required | Open-field-line flag |
| `cell_centered` | required | Cell-centered planes |
| `Bx` | `0.0` | Magnetic field x |
| `By` | `0.0` | Magnetic field y |
| `Bz` | `1.0` | Magnetic field z |
| `sheath_mask` | `None` | Optional sheath mask |
| `sheath_sign` | `None` | Optional sheath sign |

### `FCISlabParams`

| Name | Default | Notes |
| --- | --- | --- |
| `nu_par` | `0.0` | Parallel diffusion |
| `sheath_nu` | `0.0` | Sheath damping |
| `open_field_line` | `None` | Override grid open flag |

### `FCIDRB3DParams`

| Name | Default | Notes |
| --- | --- | --- |
| `kappa` | `0.0` | Background-gradient drive |
| `alpha` | `0.0` | Adiabaticity / coupling |
| `kpar` | `0.0` | Parallel coupling |
| `Dn` | `0.0` | Density diffusion |
| `DOmega` | `0.0` | Vorticity diffusion |
| `bracket` | `arakawa` | `arakawa`, `centered` |
| `poisson` | `spectral` | `spectral`, `fd_cg` |
| `boussinesq` | `True` | Boussinesq polarization |
| `non_boussinesq_perturbed_density_on` | `True` | Use `n0 + Re[n]` |
| `n0` | `1.0` | Reference density |
| `n0_min` | `1e-6` | Density floor |
| `poisson_preconditioner` | `spectral` | `spectral`, `jacobi`, `none` |
| `poisson_maxiter` | `400` | Poisson CG max iterations |
| `poisson_tol` | `1e-10` | Poisson CG tol |
| `dealias_on` | `False` | Dealiasing |
| `k2_min` | `1e-12` | Poisson k2 floor |
| `sheath_nu` | `0.0` | Sheath damping |

### `FCIDRB3DFullParams`

| Name | Default | Notes |
| --- | --- | --- |
| `omega_n` | `0.0` | Density-gradient drive |
| `omega_Te` | `0.0` | Te-gradient drive |
| `kappa` | `0.0` | Curvature drive amplitude |
| `kappa_profile` | `constant` | `constant`, `cosine` |
| `kappa_theta0` | `0.0` | Phase for cosine profile |
| `alpha` | `0.0` | Adiabaticity / coupling |
| `eta_par` | `0.0` | Resistive coupling |
| `me_hat` | `1.0` | Electron inertia knob |
| `alpha_Te_ohm` | `1.0` | Thermal force coefficient |
| `alpha_Ti_ohm` | `0.0` | Ti thermal force coefficient |
| `Dn` | `0.0` | Density diffusion |
| `DOmega` | `0.0` | Vorticity diffusion |
| `Dvpar` | `0.0` | Parallel flow diffusion |
| `DTe` | `0.0` | Te diffusion |
| `chi_par` | `0.0` | Parallel heat conduction |
| `DTi` | `0.0` | Ti diffusion |
| `Dpsi` | `0.0` | Psi diffusion |
| `hot_ion_on` | `False` | Enable hot-ion terms |
| `tau_i` | `1.0` | Ti/Te |
| `omega_Ti` | `0.0` | Ti-gradient drive |
| `em_on` | `False` | Electromagnetic toggle |
| `beta` | `0.0` | EM beta |
| `neutrals_on` | `False` | Neutral coupling toggle |
| `neutrals` | `NeutralParams()` | Neutral parameters |
| `bracket` | `arakawa` | `arakawa`, `centered` |
| `perp_operator` | `spectral` | `spectral`, `fd`, `fv` |
| `perp_bc` | `BC2D.periodic()` | Perpendicular BCs |
| `perp_bc_nu` | `0.0` | Boundary relaxation rate |
| `use_target_aware_dpar` | `True` | Target-aware parallel derivative |
| `target_scheme` | `appendix_b` | Target scheme name |
| `boussinesq` | `True` | Boussinesq polarization |
| `non_boussinesq_perturbed_density_on` | `True` | Use `n0 + Re[n]` |
| `n0` | `1.0` | Reference density |
| `n0_min` | `1e-6` | Density floor |
| `poisson_preconditioner` | `spectral` | `spectral`, `jacobi`, `none` |
| `poisson_maxiter` | `400` | Poisson CG max iterations |
| `poisson_tol` | `1e-10` | Poisson CG tol |
| `k2_min` | `1e-12` | Poisson k2 floor |
| `sheath_on` | `False` | Enable sheath terms |
| `sheath_nu_mom` | `0.0` | Sheath momentum damping |
| `sheath_nu_particle` | `0.0` | Sheath particle damping |
| `sheath_nu_energy` | `0.0` | Sheath energy damping |
| `sheath_gamma_e` | `3.5` | Sheath electron heat transmission |
| `sheath_gamma_i` | `3.5` | Sheath ion heat transmission |
| `sheath_delta` | `0.0` | Ion transmission correction |
| `sheath_cos2` | `1.0` | Loizu vorticity BC factor |
| `sheath_bc_model` | `simple` | `simple`, `loizu_linear` |
| `operator_split_on` | `False` | Enable operator split |
| `operator_conservative_on` | `True` | Include conservative part |
| `operator_source_on` | `True` | Include source part |
| `operator_dissipative_on` | `True` | Include dissipative part |

## Outputs

### Linear scan outputs

Linear scans write `results.npz` and `params.json` plus plots such as `gamma_vs_ky.png` and
`eigs_spectrum.png`. The `.npz` file contains `ky`, `gamma`, `omega`, `eigs`, and Arnoldi
diagnostics.

### Nonlinear 2D outputs

Most 2D examples write:

- `movie.gif` for quick visual regression,
- `panel.png` or `*_panel.png` for diagnostic summaries,
- `state.npz` or `results.npz` with time series and fields,
- `params.json` with the full run configuration.

### FCI/3D outputs

FCI/3D examples typically save:

- `snapshot_*.png` or `movie.gif` for plane diagnostics,
- `*_metrics.json` when target-hit or budget diagnostics are computed,
- `params.json` with map and model settings.

### SOL-width workflow outputs

The SOL width proxy used in Halpern-style gradient-removal workflows is:

`max(gamma, 0) / ky`.

The fixed-point estimate returns `Lp`, `ky_star`, `gamma_over_ky_star`, and a `history` array.

## Environment variables

- `JAXDRB_FAST=1` reduces resolution in some examples.
- `JAXDRB_PROGRESS=0` disables Diffrax progress output.

## Reproducibility tips

- Always save `params.json` next to `results.npz`.
- When reporting SOL-width proxies, use the normalized `max(gamma,0)/(ky cs)` convention.
