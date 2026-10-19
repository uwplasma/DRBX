# Deck Reference

`drbx run deck.toml` runs every DRBX model from a TOML file with the same configuration objects the Python API uses: each table is the dataclass (or function signature) of the Python call, so there is one set of defaults. This page documents every table, every key and every output. A test fails when this page and the code disagree.

```bash
drbx inspect deck.toml   # resolved configuration, nothing is run
drbx run deck.toml       # run; add --output-dir DIR or --case-name NAME to override [output]
```

## Rules

- A deck names its model with `[model] backend = "<name>"`. Decks without `backend` use the BOUT-style native format (diffusion, periodic fluid, electrostatic vorticity; see [Inputs And Outputs](input_output_reference.md)).
- Unknown tables and keys are errors that list the valid ones. Required keys are marked `required`; every other key may be omitted.
- Relative paths resolve against the deck file. `[output] directory` defaults to `./output/<case_name>`.
- Integers are accepted where a float is expected; lists are TOML arrays; `path` values are strings.
- `[runtime]` and `[output]` exist for every backend except `fci_braginskii`, which names its own history file (`output`) and has no `--output-dir`, `--case-name` or `--precision`.
- Every backend except `fci_braginskii` writes `<case_name>_summary.json` (`case_name`, `backend`, the resolved `config`, and `results`) and `<case_name>_arrays.npz`.
- Time-dependent reduced models write `state_*` arrays; pass that file as `[run] restart_from` to continue.

## Python versus deck

| Python entry point | Deck `backend` | Example deck |
| --- | --- | --- |
| `hw_run`, `hw_run_flux_history` | `hasegawa_wakatani` | [`examples/inputs/hasegawa_wakatani.toml`](../examples/inputs/hasegawa_wakatani.toml) |
| `sol_flux_tube_run`, `compute_fci_sheath_recycling` | `sol_flux_tube` | [`examples/inputs/sol_flux_tube.toml`](../examples/inputs/sol_flux_tube.toml) |
| `sol_recycling_run` | `sol_recycling` | [`examples/inputs/sol_recycling.toml`](../examples/inputs/sol_recycling.toml) |
| `detachment_sol_run`, `detachment_diagnostics`, `detachment_ledger` | `detachment_sd1d` | [`examples/inputs/detachment_sd1d.toml`](../examples/inputs/detachment_sd1d.toml) |
| `run_stellarator_turbulence` with `build_rotating_ellipse_geometry`, `build_shifted_torus_geometry`, `build_island_divertor_geometry` | `stellarator_turbulence` | [`examples/inputs/stellarator_turbulence.toml`](../examples/inputs/stellarator_turbulence.toml) |
| `build_context`, `differentiability_report` | `fci_differentiable` | [`examples/inputs/fci_differentiable.toml`](../examples/inputs/fci_differentiable.toml) |
| `HsxBlobConfig`, `drbx.fci_braginskii.run.run` | `fci_braginskii` | [`examples/inputs/hsx_fci_blob.toml`](../examples/inputs/hsx_fci_blob.toml) |

BOUT-style decks (diffusion, periodic fluid MMS, electrostatic vorticity, restart) are unchanged. Not available from a deck, because they are user-written objective or plotting code rather than a model: the Hasegawa-Wakatani, diffusion and detachment optimization scripts (`examples/tokamak/hasegawa_wakatani_optimization.py`, `examples/autodiff_diffusion_*.py`, `examples/autodiff/detachment_control.py`), imported-geometry builders that need external files (ESSOS coils, VMEC/VMEX, VMEC-extender), and the movie and figure scripts. Their model runs are covered by the backends above; the Hasegawa-Wakatani gradient is available through `[autodiff]`.

## Hasegawa-Wakatani drift-wave turbulence

`[model] backend = "hasegawa_wakatani"`

Two-field resistive drift-wave model on a doubly periodic pseudo-spectral box (Python: `drbx.native.hasegawa_wakatani`, `examples/tokamak/drift_wave_turbulence.py`). Units are the usual normalized ones (lengths in rho_s, times in L_n/c_s).

### Tables

#### `[hasegawa_wakatani]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `adiabaticity` | `float` | `1.0` | 1/time | alpha: parallel electron response coupling density and potential. |
| `gradient` | `float` | `1.0` | 1/length | kappa: background density-gradient drive. |
| `hyperviscosity` | `float` | `0.001` | length^4/time | nu: Laplacian-squared damping of the grid scale. |
| `friction` | `float` | `0.0` | 1/time | mu: linear drag absorbing the inverse cascade. |

#### `[grid]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `n` | `int` | `96` | cells | Points per side of the periodic `n x n` box. |
| `length` | `float` | `50.26548245743669` | rho_s | Box side length. |

#### `[run]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `dt` | `float` | `0.005` | time | RK4 time step. |
| `steps_per_block` | `int` | `800` | steps | RK4 steps between diagnostics; one entry of `time`, `energy`, `particle_flux` per block. |
| `blocks` | `int` | `40` | - | Number of blocks; total time is `blocks * steps_per_block * dt`. |
| `seed` | `int` | `0` | - | RNG seed of the initial noise (Hasegawa-Wakatani) or of the seeded mode phases (turbulence). |
| `seed_amplitude` | `float` | `0.05` | - | RMS of the initial real-space noise field. |
| `save_frames` | `bool` | `False` | - | Also write `vorticity_frames` (one frame per block). |
| `restart_from` | `Path | None` | `None` | path | `*_arrays.npz` of an earlier run of the same model; continues from its `state_*` arrays and time. |

#### `[autodiff]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `parameter` | `str` | `required` | - | Hasegawa-Wakatani parameter to differentiate: `adiabaticity`, `gradient`, `hyperviscosity` or `friction`. The table is optional; omit it to skip the gradient. |
| `fd_step` | `float` | `0.001` | relative | Central finite-difference step as a fraction of the parameter value (floor 1), used to check the autodiff gradient. |

#### `[runtime]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `precision` | `str` | `'float64'` | - | `float64` or `float32` floating-point precision for the whole run. |

#### `[output]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `directory` | `Path | None` | `None` | path | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `case_name` | `str | None` | `None` | - | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

### Outputs

`<case_name>_arrays.npz`:

| Name | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `time` | `(blocks,)` | time | Time at the end of each block. |
| `energy` | `(blocks,)` | normalized | Fluctuation energy `sum(k^2 |phi_k|^2 + |n_k|^2) / n^4`. |
| `particle_flux` | `(blocks,)` | normalized | Domain-averaged radial E x B particle flux. |
| `vorticity` | `(n, n)` | normalized | Final real-space vorticity. |
| `density` | `(n, n)` | normalized | Final real-space density fluctuation. |
| `vorticity_frames` | `(blocks, n, n)` | normalized | Vorticity after each block (only with `save_frames`). |
| `state_zeta_hat` | `(n, n) complex` | normalized | Final spectral vorticity (restart state). |
| `state_n_hat` | `(n, n) complex` | normalized | Final spectral density (restart state). |
| `state_time` | `()` | time | Final time (restart state). |

`<case_name>_summary.json` `results` keys: `final_time`, `final_energy`, `final_particle_flux`, `mean_particle_flux_second_half`, `autodiff` (only with `[autodiff]`: parameter, value, gradient, finite_difference, relative_error).

## Open-field-line SOL flux tube

`[model] backend = "sol_flux_tube"`

Isothermal density and parallel-momentum Euler system with Bohm sheaths at both targets, plus the FCI sheath/recycling accounting (Python: `drbx.native.sol_flux_tube`, `compute_fci_sheath_recycling`, `examples/sol/open_sol_flux_tube.py`). The target diagnostic uses the `[run]` temperatures.

### Tables

#### `[sol_flux_tube]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `sound_speed` | `float` | `1.0` | c_s | Normalized isothermal sound speed; sets the Bohm outflow. |
| `source_amplitude` | `float` | `0.02` | n0 c_s/L | Peak of the Gaussian upstream particle source. |
| `source_width` | `float` | `4.0` | L | Parallel 1/e width of the source. |
| `density_floor` | `float` | `1e-06` | n0 | Positivity floor applied after each step. |

#### `[open_slab]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `shape` | `tuple[int, int, int]` | `required` | cells | `[nx, ny, nz]` grid; `nz` cells along the field between the targets. |
| `radial_extent` | `float` | `1.0` | L | Radial size of the flux tube. |
| `binormal_extent` | `float` | `1.0` | L | Binormal size of the flux tube. |
| `parallel_length` | `float` | `40.0` | L | Connection length between the two target plates. |

#### `[sheath]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `recycling_fraction` | `float` | `0.98` | - | Fraction of the target ion flux returned as neutrals (0 = perfect absorber). |
| `electron_sheath_transmission` | `float` | `5.0` | - | Electron sheath heat-transmission coefficient. |
| `ion_sheath_transmission` | `float` | `3.5` | - | Ion sheath heat-transmission coefficient. |
| `recycled_neutral_energy` | `float` | `0.03` | T_norm | Energy carried by each recycled neutral. |
| `ion_mass` | `float` | `1.0` | m_p | Ion mass in the Bohm sound speed. |
| `density_floor` | `float` | `1e-12` | n0 | Density floor inside the sheath closure. |
| `temperature_floor` | `float` | `1e-12` | T_norm | Temperature floor inside the sheath closure. |

#### `[run]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `cfl` | `float` | `0.4` | - | Time step is `cfl * dz / (c_s + 1)`. |
| `steps` | `int` | `60000` | steps | Total time steps (SOL, recycling), or RK4 steps (turbulence). |
| `report_every` | `int` | `10000` | steps | Steps per chunk; the steady-state residual is recorded after each chunk. |
| `electron_temperature` | `float` | `0.5` | T_norm | Uniform electron temperature of the target sheath diagnostic. |
| `ion_temperature` | `float` | `0.5` | T_norm | Uniform ion temperature of the target sheath diagnostic. |
| `restart_from` | `Path | None` | `None` | path | `*_arrays.npz` of an earlier run of the same model; continues from its `state_*` arrays and time. |

#### `[runtime]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `precision` | `str` | `'float64'` | - | `float64` or `float32` floating-point precision for the whole run. |

#### `[output]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `directory` | `Path | None` | `None` | path | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `case_name` | `str | None` | `None` | - | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

### Outputs

`<case_name>_arrays.npz`:

| Name | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `z` | `(nz,)` | L | Parallel cell centers. |
| `density` | `(nz,)` | n0 | Final density along the tube. |
| `momentum` | `(nz,)` | n0 c_s | Final parallel momentum. |
| `mach` | `(nz,)` | c_s | Parallel Mach number `m / (n c_s)`; expect -1 and +1 at the targets. |
| `source` | `(nz,)` | n0 c_s/L | Upstream particle source. |
| `report_step` | `(n_reports,)` | steps | Cumulative step count of each chunk. |
| `residual` | `(n_reports,)` | n0 c_s/L | Steady-state residual `max|dn/dt|` per chunk. |
| `ion_particle_loss` | `(nz,)` | normalized | Sheath ion loss per cell (non-zero at the targets). |
| `target_heat_load` | `(nz,)` | normalized | Target heat load per cell. |
| `recycled_particle_source` | `(nz,)` | normalized | Recycled-neutral source per cell. |
| `state_density` | `(nx, ny, nz)` | n0 | Restart state. |
| `state_momentum` | `(nx, ny, nz)` | n0 c_s | Restart state. |
| `state_time` | `()` | L/c_s | Restart state time. |

`<case_name>_summary.json` `results` keys: `mach_at_targets`, `n_target_over_upstream` (two-point prediction 0.5), `total_target_ion_loss`, `total_recycled_source`, `total_target_heat_load`, `particle_recycling_residual`, `current_balance_residual`, `final_time`.

## Recycling SOL with neutrals

`[model] backend = "sol_recycling"`

Plasma plus diffusive neutral on a flux tube with a prescribed temperature profile and AMJUEL ionization, recombination and charge-exchange rates (Python: `drbx.native.neutrals`, `examples/sol/recycling_sol.py`). The reference run uses `[sol_recycling] upstream_density`; each `[scan]` value is a fresh run.

### Tables

#### `[sol_recycling]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `parallel_length` | `float` | `30.0` | m | Connection length, stagnation point to target (sets the atomic-rate scale). |
| `upstream_density` | `float` | `1.0` | N_norm | Upstream ion density pinned at `z = 0`. |
| `recycling_fraction` | `float` | `1.0` | - | Fraction R of the target ion flux returned as neutrals. |
| `neutral_diffusion` | `float` | `8.0` | - | Neutral parallel diffusion enhancement (kinetic-transport proxy). |
| `neutral_temperature` | `float` | `0.04` | T_norm | Neutral temperature (0.04 is about 2 eV). |
| `ion_mass` | `float` | `2.0` | m_p | Ion mass (2 = deuterium). |
| `density_floor` | `float` | `0.0001` | N_norm | Density floor. |

#### `[sol_recycling.normalization]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `Nnorm` | `float` | `1e+19` | m^-3 | Density normalization. |
| `Tnorm` | `float` | `100.0` | eV | Temperature normalization. |
| `Bnorm` | `float` | `1.0` | T | Magnetic-field normalization. |
| `ion_mass` | `float` | `2.0` | m_p | Ion mass used by the normalization. |

#### `[profile]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `nz` | `int` | `200` | cells | Parallel cells, stagnation point to target. |
| `upstream_ev` | `float` | `30.0` | eV | Prescribed upstream temperature. |
| `target_ev` | `float` | `1.5` | eV | Prescribed target temperature (quadratic profile between the two). |
| `neutral_seed` | `float` | `0.05` | N_norm | Initial trace neutral density. |

#### `[run]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `dt` | `float | None` | `None` | L/c_s | Time step; default `0.1 / nz` (CFL 0.3 against 3 c_s) when omitted. |
| `steps` | `int` | `60000` | steps | Total time steps (SOL, recycling), or RK4 steps (turbulence). |
| `report_every` | `int` | `15000` | steps | Steps per chunk; the steady-state residual is recorded after each chunk. |
| `restart_from` | `Path | None` | `None` | path | `*_arrays.npz` of an earlier run of the same model; continues from its `state_*` arrays and time. |

#### `[scan]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `upstream_density` | `tuple[float, ...]` | `[]` | see text | Extra steady states, one per value (recycling: `N_norm`; detachment: m^-3). Detachment reuses the previous solution as the next guess. Empty: none. |

#### `[runtime]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `precision` | `str` | `'float64'` | - | `float64` or `float32` floating-point precision for the whole run. |

#### `[output]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `directory` | `Path | None` | `None` | path | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `case_name` | `str | None` | `None` | - | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

### Outputs

`<case_name>_arrays.npz`:

| Name | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `z_over_L` | `(nz,)` | - | Parallel coordinate, target at 1. |
| `temperature` | `(nz,)` | T_norm | Prescribed temperature profile. |
| `ion_density` | `(nz,)` | N_norm | Final ion density. |
| `ion_momentum` | `(nz,)` | N_norm m_p c_s | Final ion momentum. |
| `neutral_density` | `(nz,)` | N_norm | Final neutral density. |
| `mach` | `(nz,)` | local c_s | Ion Mach number. |
| `scan_upstream_density` | `(n_scan,)` | N_norm | Scan inputs. |
| `scan_target_mach` | `(n_scan,)` | local c_s | Target Mach number per scan point. |
| `scan_target_ion_flux` | `(n_scan,)` | normalized | Target ion flux per scan point. |
| `state_ion_density` | `(nz,)` | N_norm | Restart state. |
| `state_ion_momentum` | `(nz,)` | N_norm m_p c_s | Restart state. |
| `state_neutral_density` | `(nz,)` | N_norm | Restart state. |
| `state_time` | `()` | L/c_s | Restart state time. |

`<case_name>_summary.json` `results` keys: `target_mach`, `target_ion_flux`, `dt`, `final_time`, `neutral_cushion_ratio` (neutral density at target over midpoint).

## SD1D 1-D detachment

`[model] backend = "detachment_sd1d"`

Plasma-neutral leg solved to steady state by pseudo-transient Newton continuation, matched to SD1D (Python: `drbx.native.neutrals.detachment_sol_run`, `examples/benchmarks/b6_detachment_sd1d.py`). With `[scan]`, one steady state per upstream density, each starting from the previous one.

### Tables

#### `[detachment_sd1d]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `ny` | `int` | `800` | cells | Nonuniform cells along the leg. |
| `length` | `float` | `30.0` | m | Total leg length. |
| `length_xpt` | `float` | `10.0` | m | Length of the source region and position of the area kink (X-point). |
| `area_expansion` | `float` | `2.0` | - | Flux-tube area ratio between the X-point and the target. |
| `dymin` | `float` | `0.1` | - | Smallest cell size (grid refinement toward the target). |
| `upstream_density` | `float` | `3e+19` | m^-3 | Density held at the upstream cell. |
| `power_flux` | `float` | `50000000.0` | W/m^2 | Power flux injected over `length_xpt`. |
| `particle_flux` | `float` | `4e+23` | m^-2 s^-1 | Particle-source flux (shape of the source). |
| `ionisation_energy` | `float` | `13.6` | eV | Energy lost per ionisation. |
| `recycling_fraction` | `float` | `0.99` | - | Target recycling fraction. |
| `neutral_diffusion` | `float` | `10.0` | - | Neutral diffusion multiplier (`dneut`). |
| `neutral_temperature_floor` | `float` | `0.5` | eV | Neutral temperature floor. |
| `recycled_temperature` | `float` | `3.5` | eV | Franck-Condon temperature of recycled neutrals. |
| `max_mean_free_path` | `float` | `0.1` | m | Cap on the neutral mean free path. |
| `sheath_gamma` | `float` | `6.0` | - | Sheath heat-transmission coefficient. |
| `charge_exchange` | `bool` | `True` | - | Include charge-exchange friction. |
| `recombination` | `bool` | `True` | - | Include recombination. |
| `Nnorm` | `float` | `1e+20` | m^-3 | Density normalization. |
| `Tnorm` | `float` | `100.0` | eV | Temperature normalization. |
| `Bnorm` | `float` | `1.0` | T | Magnetic-field normalization. |
| `AA` | `float` | `2.0` | m_p | Ion mass number. |

#### `[run]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `source_scale` | `float` | `0.1` | - | Initial multiplier of the proportional-integral particle source. |
| `tol` | `float` | `1e-08` | - | Stop when the scaled steady-state residual falls below this. |
| `max_iter` | `int` | `400` | - | Maximum pseudo-transient Newton iterations. |
| `initial_dt` | `float` | `100.0` | 1/Omega_ci | Initial backward-Euler step (grown by the residual ratio). |
| `newton_switch` | `float` | `1e-05` | - | Residual below which the step is a full Newton step. |

#### `[scan]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `upstream_density` | `tuple[float, ...]` | `[]` | see text | Extra steady states, one per value (recycling: `N_norm`; detachment: m^-3). Detachment reuses the previous solution as the next guess. Empty: none. |

#### `[runtime]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `precision` | `str` | `'float64'` | - | `float64` or `float32` floating-point precision for the whole run. |

#### `[output]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `directory` | `Path | None` | `None` | path | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `case_name` | `str | None` | `None` | - | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

### Outputs

`<case_name>_arrays.npz`:

| Name | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `upstream_density`, `target_temperature_ev`, `target_ion_flux_area`, `target_density`, `target_mach`, `upstream_temperature_ev`, `residual`, `iterations`, `converged`, `particle_imbalance`, `power_imbalance` | `(n_points,)` | m^-3, eV, m^-2 s^-1, m^-3, -, eV, -, -, -, -, - | One entry per steady state (a single point without `[scan]`). |
| `profile_ion_density`, `profile_ion_momentum`, `profile_plasma_pressure`, `profile_neutral_density`, `profile_neutral_momentum`, `profile_neutral_pressure` | `(ny,)` | normalized (SD1D) | Fields of the last steady state. |

`<case_name>_summary.json` `results` keys: `points` (one dict per steady state with the per-point values above).

## Four-field FCI turbulence (stellarator, tokamak, island divertor)

`[model] backend = "stellarator_turbulence"`

Density, vorticity, ion and electron parallel velocity with a GMRES potential inversion on a flux-coordinate-independent geometry (Python: `drbx.native.stellarator_turbulence`, `examples/stellarator/stellarator_turbulence.py`). Exactly one geometry table selects the configuration: `[rotating_ellipse]` (stellarator, optionally with a limiter SOL), `[shifted_torus]` (tokamak-like) or `[island_divertor]`.

### Tables

Exactly one of `[rotating_ellipse]`, `[shifted_torus]`, `[island_divertor]`; `[run]` is required.

#### `[rotating_ellipse]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `shape` | `tuple[int, int, int]` | `required` | cells | `[nx, ny, nz]` grid (radial, poloidal, toroidal). |
| `r0` | `float` | `3.0` | L | Major-radius offset of the torus. |
| `x_min` | `float` | `0.2` | L | Inner radial (minor-radius label) bound. |
| `x_max` | `float` | `1.0` | L | Outer radial bound. |
| `elongation` | `float` | `0.35` | - | Ellipse deformation delta in [0, 1); aspect ratio `(1+delta)/(1-delta)`. |
| `n_field_periods` | `int` | `1` | - | Ellipse rotations per toroidal turn (field periods). |
| `iota` | `float` | `0.9` | - | Rotational transform of the field. |
| `c_phi` | `float` | `3.0` | - | Toroidal-field scaling constant. |
| `map_substeps` | `int` | `8` | - | Field-line tracer substeps per toroidal cell. |
| `limiter_radius` | `float | None` | `None` | L | Toroidal limiter at zeta=0 opening every line with `x > limiter_radius` into a sheath-drained SOL; omit for all-closed lines. |

#### `[shifted_torus]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `shape` | `tuple[int, int, int]` | `required` | cells | `[nx, ny, nz]` grid (radial, poloidal, toroidal). |
| `x_min` | `float` | `0.15` | L | Inner radial (minor-radius label) bound. |
| `x_max` | `float` | `1.0` | L | Outer radial bound. |
| `r0` | `float` | `3.0` | L | Major-radius offset of the torus. |
| `alpha_value` | `float` | `0.25` | - | Shafranov-like shift amplitude. |
| `iota` | `float` | `1.1` | - | Rotational transform of the field. |
| `c_phi` | `float` | `3.0` | - | Toroidal-field scaling constant. |
| `sigma` | `float` | `0.0` | - | Poloidal shear; non-zero makes the metric non-axisymmetric. |

#### `[island_divertor]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `shape` | `tuple[int, int, int]` | `required` | cells | `[nx, ny, nz]` grid (radial, poloidal, toroidal). |
| `r0` | `float` | `3.0` | L | Major-radius offset of the torus. |
| `elongation` | `float` | `0.35` | - | Ellipse deformation delta in [0, 1); aspect ratio `(1+delta)/(1-delta)`. |
| `n_field_periods` | `int` | `1` | - | Ellipse rotations per toroidal turn (field periods). |
| `c_phi` | `float` | `3.0` | - | Toroidal-field scaling constant. |
| `map_substeps` | `int` | `8` | - | Field-line tracer substeps per toroidal cell. |
| `open_field_line_masks` | `bool` | `False` | - | Mark open endpoints by multi-transit field-line tracing. |
| `mask_max_transits` | `int` | `30` | transits | Maximum toroidal transits traced when classifying open lines. |

#### `[island_divertor.field]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `x_min` | `float` | `0.2` | L | Inner radial bound. |
| `x_max` | `float` | `1.0` | L | Outer radial bound. |
| `iota_axis` | `float` | `0.55` | - | Rotational transform at `x_min` (linear profile). |
| `iota_edge` | `float` | `0.82` | - | Rotational transform at `x_max`. |
| `resonances` | `tuple[tuple[int, int, float], ...]` | `[[3, 2, 0.005], [4, 3, 0.005], [5, 4, 0.005]]` | - | `[[m, n, epsilon], ...]`: island chains at `iota = n/m` with relative width epsilon. |

#### `[run]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `steps` | `int` | `required` | steps | Number of RK4 steps (required). |
| `dt` | `float` | `required` | time | RK4 time step (required). |
| `amplitude` | `float` | `0.08` | - | Peak relative density perturbation of the seeded multi-mode state. |
| `seed` | `int` | `0` | - | RNG seed of the initial noise (Hasegawa-Wakatani) or of the seeded mode phases (turbulence). |
| `sheath_sink` | `bool` | `False` | - | Drain the open (SOL) endpoints with the Bohm sheath sink each step. |
| `frame_stride` | `int` | `1` | steps | Record a frame every this many steps (the last step is always recorded). |

#### `[runtime]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `precision` | `str` | `'float64'` | - | `float64` or `float32` floating-point precision for the whole run. |

#### `[output]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `directory` | `Path | None` | `None` | path | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `case_name` | `str | None` | `None` | - | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

### Outputs

`<case_name>_arrays.npz`:

| Name | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `density_frames` | `(n_frames, nx, ny, nz) float32` | n0 | Density at the recorded steps (frame 0 is the initial state). |
| `omega_frames` | `(n_frames, nx, ny, nz) float32` | normalized | Vorticity. |
| `times` | `(n_frames,)` | time | Time of each frame. |
| `particle_content` | `(n_frames,)` | normalized | Jacobian-weighted total density. |
| `target_flux` | `(n_frames,)` | normalized | Total sheath ion loss (0 unless `sheath_sink`). |

`<case_name>_summary.json` `results` keys: `geometry`, `shape`, `n_frames`, `initial_particle_content`, `final_particle_content`, `total_target_flux`, `finite`.

## Differentiable FCI flux tube

`[model] backend = "fci_differentiable"`

Two-field shifted-torus FCI rollout whose gradient with respect to the initial amplitude is computed by reverse-mode autodiff and compared with a central finite difference (Python: `drbx.native.fci_differentiable_case`, `examples/stellarator/fci_differentiable.py`).

### Tables

#### `[fci_differentiable]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `shape` | `tuple[int, int, int]` | `[16, 16, 8]` | cells | `[nx, ny, nz]` grid. |
| `sigma` | `float` | `0.6` | - | Poloidal shear of the shifted torus. |
| `rho_star` | `float` | `1.0` | - | Drift scale rho_*. |
| `x_min` | `float` | `0.15` | L | Inner radial bound. |
| `x_max` | `float` | `1.0` | L | Outer radial bound. |
| `r0` | `float` | `3.0` | L | Major-radius offset. |
| `alpha_value` | `float` | `0.25` | - | Shafranov-like shift. |
| `iota` | `float` | `1.1` | - | Rotational transform. |
| `c_phi` | `float` | `3.0` | - | Toroidal-field scaling. |

#### `[run]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `amp0` | `float` | `0.1` | - | Initial perturbation amplitude at which the gradient is evaluated. |
| `n_steps` | `int` | `24` | steps | Rollout steps. |
| `dt` | `float` | `0.001` | time | Rollout time step. |
| `fd_step` | `float` | `1e-05` | - | Central finite-difference step in `amp0`. |

#### `[runtime]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `precision` | `str` | `'float64'` | - | `float64` or `float32` floating-point precision for the whole run. |

#### `[output]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `directory` | `Path | None` | `None` | path | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `case_name` | `str | None` | `None` | - | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

### Outputs

`<case_name>_arrays.npz`:

| Name | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `density` | `(nx, ny, nz)` | n0 | Final density. |
| `v_parallel` | `(nx, ny, nz)` | c_s | Final parallel velocity. |

`<case_name>_summary.json` `results` keys: `grad` (autodiff d objective / d amp0), `fd` (finite difference), `rel_error`, `fd_step`, `amp0`, `n_steps`, `dt`, `objective_value` (density variance), `finite`, `density_max`, `v_parallel_max`.

## HSX FCI drift-reduced Braginskii (`fci_braginskii`)

`[model] backend = "fci_braginskii"`

Seven-field electrostatic Boussinesq drift-reduced Braginskii model on the HSX toroidal FCI geometry artifact, advanced with IMEX-SSP222 and an FGMRES potential solve (Python: `drbx.fci_braginskii.run.HsxBlobConfig`/`run`, `simulate_hsx_blob.py`, see [HSX FCI Braginskii Backend](fci_braginskii_hsx_backend.md)). Every key is also a `simulate_hsx_blob.py` option with dashes. The geometry artifact is produced by `generate_hsx_fci_geometry.py`.

### Tables

A single table, `[fci_braginskii]`.

#### `[fci_braginskii]`

| Key | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `geometry` | `Path | None` | `None` | path | Directory of the precomputed FCI simulation geometry artifact (sets topology, resolution, owner profile). Required. |
| `shard_counts` | `tuple[int, int, int]` | `[1, 1, 1]` | - | JAX decomposition `[su, sv, seta]`; only eta decomposition is supported, so the first two are 1. |
| `halo_width` | `int` | `2` | cells | Halo width exchanged between shards. |
| `filament_cache_dir` | `Path` | `<repo>/.hsx_filament_cache` | path | Directory of reusable initial-filament field-line label caches. |
| `no_filament_cache` | `bool` | `False` | - | Disable loading and writing filament caches. |
| `rebuild_filament_cache` | `bool` | `False` | - | Ignore a matching cache and replace it after tracing. |
| `final_time` | `float` | `0.15` | normalized time | Final simulation time. |
| `num_steps` | `int` | `200` | steps | Equal IMEX-SSP222 steps to reach `final_time`. |
| `save_every` | `int` | `1` | steps | History is stored every this many steps. |
| `checkpoint_every` | `int` | `0` | steps | Atomic restartable checkpoint every N steps; 0 disables. |
| `snapshot_times` | `tuple[float, ...]` | `[]` | normalized time | Times at which snapshot files are written (first step at or after each). |
| `snapshot_dir` | `Path | None` | `None` | path | Directory for snapshot files; default the output directory. |
| `restart_from` | `Path | None` | `None` | path | Snapshot or history NPZ to restart from. |
| `restart_frame` | `int` | `-1` | - | Frame loaded from a history NPZ; ignored for a snapshot. |
| `diagnostic_every` | `int` | `0` | steps | Print min/max diagnostics for every field every N steps; 0 disables. |
| `blob_initialization` | `str` | `'field-aligned'` | - | `field-aligned` (backtraced finite filament) or `logical` (eta-copied Gaussian). |
| `density_amplitude` | `float` | `0.05` | n0 | Relative density amplitude of the blob. |
| `temperature_amplitude` | `float` | `0.0` | T0 | Electron-temperature perturbation (legacy logical initialization only). |
| `blob_center` | `tuple[float, float]` | `[0.65, 0.5]` | logical u, v | Blob center `[u, v]`. |
| `blob_width` | `float` | `0.1` | logical u, v | Gaussian width of the blob. |
| `blob_reference_eta` | `float` | `3.141592653589793` | rad | Toroidal reference plane of the filament. |
| `blob_parallel_half_length` | `float` | `3.141592653589793` | rad | Half-length of the cos^2 filament envelope in eta; at most pi. |
| `fieldline_substeps_per_plane` | `int` | `4` | - | RK4 tracing substeps per plane spacing for the initial labels. |
| `toroidal_perturbation_amplitude` | `float` | `0.0` | - | Relative cosine modulation (legacy logical blob only). |
| `toroidal_perturbation_mode` | `int` | `1` | - | Full-torus integer mode number of the perturbation. |
| `toroidal_perturbation_phase` | `float` | `0.0` | rad | Initial toroidal perturbation phase. |
| `tau` | `float` | `1.0` | - | Ion-to-electron temperature ratio. |
| `rho_star` | `float` | `0.0005` | - | rho_s / L_ref (L_ref = 1 m): scales E x B and curvature drifts by rho_*, polarization by rho_*^2. |
| `mi_over_me` | `float` | `1836.0` | - | Ion-to-electron mass ratio. |
| `perp_diffusion` | `float` | `1e-05` | normalized | Perpendicular diffusion coefficient. |
| `parallel_diffusion` | `float` | `0.0` | normalized | Parallel diffusion coefficient. |
| `electron_collision_frequency` | `float` | `0.0` | normalized | Electron collision frequency. |
| `implicit_current_phi_pair` | `bool` | `True` | - | Treat `mu grad_par(phi)` and the parallel current divergence implicitly through a coupled potential solve (default). |
| `advance_execution` | `str` | `'staged-compiled'` | - | `compiled` (one fused executable), `staged-compiled` (reusable kernels) or `eager`. |
| `gmres_acceptance_tolerance` | `float` | `5e-05` | - | Residual accepted after GMRES stops. |
| `gmres_target_tolerance` | `float` | `1e-08` | - | Residual target that stops GMRES. |
| `gmres_max_iterations` | `int` | `500` | - | Maximum GMRES iterations. |
| `gmres_restart` | `int` | `100` | - | GMRES restart length (capped at `gmres_max_iterations`). |
| `gmres_residual_correction_steps` | `int` | `1` | - | True-residual correction solves tried when GMRES would be rejected. |
| `gmres_preconditioner` | `str` | `'line-u'` | - | Right preconditioner of the potential solves: `line-u` (production per-eta radial-tree line solve) or `none`. |
| `no_phase_timing` | `bool` | `False` | - | Disable in-executable timing markers. |
| `geometry_only` | `bool` | `False` | - | Stop after loading and lowering the geometry. |
| `output` | `Path` | `<repo>/hsx_blob_history.npz` | path | History NPZ written at the end of the run. |

### Outputs

Written to `[fci_braginskii] output` (default `<repo>/hsx_blob_history.npz`), plus snapshot and checkpoint files. Shapes use `nu, nv, neta` = the geometry artifact's radial, poloidal and toroidal cell counts; `n_saved` is the number of stored times.

| Name | Shape | Units | Meaning |
| --- | --- | --- | --- |
| `times` | `(n_saved,)` | normalized time | Stored times, starting with the initial state. |
| `density`, `phi`, `Te`, `Ti`, `Vi`, `Ve`, `vorticity` | `(n_saved, nu, nv, neta)` | normalized | Field histories in `history_dtype` (float32). |
| `u`, `v`, `theta`, `eta` | `(nu,)`, `(nv,)`, `(nv,)`, `(neta,)` | logical | Cell-center coordinates (`theta` equals `v`). |
| `jacobian`, `Bmag` | `(nu, nv, neta)` | geometry units | Cell Jacobian and field strength. |
| `B_contravariant`, `cartesian` | `(nu, nv, neta, 3)` | geometry units | Contravariant field and cell positions in metres. |
| `owner_active`, `owner_aggregate_volume` | cell arrays | bool, volume | Angular-agglomeration owner masks and volumes (RLP geometries only). |
| `nfp`, `simulated_field_periods`, `toroidal_extent`, `shard_counts`, `periodic_axes`, `axis_regular_axes`, `topology`, `coordinate_names_json`, `logical_extents`, `metric_mesh_shape`, `metric_radial_degree`, `metric_poloidal_modes`, `metric_toroidal_modes`, `eta_projection_iterations`, `parallel_operator_scheme`, `fci_trace_substeps`, `history_dtype` | scalars or short vectors | - | Geometry and run bookkeeping copied from the artifact. |
| `run_metadata_json` | scalar string | - | JSON of the resolved run, including `gmres_preconditioner` and all solver settings. |

Snapshot files (`<output stem>.snapshot_t<time>.npz`, `<output stem>.checkpoint_step<step>.npz`, `<output stem>.failure_step<step>.npz`) hold one frame: the seven fields without the time axis, plus `time`, `requested_snapshot_time`, `step`, the metric arrays `metric_*`, `jacobian`, `Bmag`, `B_contravariant`, the parallel-coefficient diagnostics, `wall_ghost_states`, `polarization_residual`, `grid_scale_diagnostics` and `run_metadata_json`. Any of them, or a history file, can be passed as `restart_from`.

