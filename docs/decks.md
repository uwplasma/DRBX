# Deck Reference

`drbx run deck.toml` runs every DRBX model from a TOML file. A deck names its model with `[model] backend = "<name>"`; every other table is the Python configuration dataclass or function signature of that model, so there is one set of defaults. A test fails when this page and the code disagree.

```bash
drbx inspect deck.toml   # print the resolved configuration
drbx run deck.toml       # run; --output-dir / --case-name override [output]
```

- Unknown tables and keys are errors that list the valid ones; `required` keys have no default. Relative paths resolve against the deck. Decks without `backend` are the BOUT-style format ([Inputs And Outputs](input_output_reference.md)).
- Every backend except `fci_braginskii` accepts `[runtime] precision` (`float64`/`float32`) and `[output] directory` (default `./output/<case_name>`), `case_name` (default: deck name), and writes `<case_name>_summary.json` (`case_name`, `backend`, resolved `config`, `results`) and `<case_name>_arrays.npz`.
- Example decks: `examples/inputs/<backend>.toml` (`hsx_fci_blob.toml` for `fci_braginskii`).
- Not available from a deck: optimization and inverse-design scripts (`hasegawa_wakatani_optimization.py`, `autodiff_diffusion_*.py`, `detachment_control.py`; the Hasegawa-Wakatani gradient is `[autodiff]`), VMEC/VMEX/VMEC-extender imports, and figure/movie scripts.

## Example decks

Every deck in `examples/inputs/` is commented key by key (what to change, units, typical ranges) and runs in under a minute except the HSX one (about 4 min, mostly the one-time geometry build). Run one with `drbx run examples/inputs/<deck>.toml`; the results go to the deck's `[output] directory`. The decks are run by the test suite.

| Deck | Case | Time | What you get |
| --- | --- | --- | --- |
| [`hasegawa_wakatani.toml`](../examples/inputs/hasegawa_wakatani.toml) | Drift-wave turbulence, autodiff gradient | 4 s | energy and flux histories, final fields; `d<flux>/d(gradient)` agrees with finite differences to 1e-6 |
| [`four_field_tokamak.toml`](../examples/inputs/four_field_tokamak.toml) | Four-field FCI, tokamak-like torus | 5 s | 5 density/vorticity frames, particle content |
| [`stellarator_turbulence.toml`](../examples/inputs/stellarator_turbulence.toml) | Four-field FCI, rotating-ellipse stellarator with limiter SOL | 20 s | frames, particle content, sheath ion loss |
| [`island_divertor.toml`](../examples/inputs/island_divertor.toml) | Four-field FCI, island divertor edge | 12 s | frames, particle content; open lines drain through the sheath |
| [`hsx_fci_blob.toml`](../examples/inputs/hsx_fci_blob.toml) | HSX from the shipped coils, density filament | 4 min | geometry built and cached, then a seven-field history `.npz` |
| [`sol_flux_tube.toml`](../examples/inputs/sol_flux_tube.toml) | Open SOL flux tube with Bohm sheath | 2 s | Mach profile (about 0.9 at the targets after 6000 steps), n_target/n_up about 0.49, sheath and recycling totals |
| [`sol_recycling.toml`](../examples/inputs/sol_recycling.toml) | Recycling SOL with neutrals, detachment-onset scan | 2 s | target Mach 1.45 at n_up = 4, falling to 0.61 at n_up = 8 |
| [`detachment_sd1d.toml`](../examples/inputs/detachment_sd1d.toml) | SD1D detachment, density scan | 30 s | target temperature 15.5 eV at 2e19 m^-3, 7.2 eV at 3e19 m^-3 |
| [`fci_differentiable.toml`](../examples/inputs/fci_differentiable.toml) | Autodiff versus finite difference through an FCI rollout | 12 s | gradient and finite difference agree to about 1e-10 |
| [`restartable_diffusion.toml`](../examples/inputs/restartable_diffusion.toml) | Restart (BOUT-style deck) | 5 s | summary, arrays and a restart file; `--restart-in` continues it |

The Hasegawa-Wakatani deck in full:

```toml
# Hasegawa-Wakatani drift-wave turbulence on a periodic 2-D box (about 10 s).
#   drbx run examples/inputs/hasegawa_wakatani.toml
# Writes <output>/hasegawa_wakatani_{summary.json,arrays.npz}.  Every key is
# optional; the values here are the small, fast ones.  Lengths are in rho_s.
# All keys: docs/decks.md.

[model]
backend = "hasegawa_wakatani"

[hasegawa_wakatani]
adiabaticity = 1.0      # alpha, parallel coupling.  0.1 gives streamers; 2+ weak transport.
gradient = 1.0          # kappa, density-gradient drive of the instability (0.3 - 3).
hyperviscosity = 1.0e-2 # nu, damps the grid scale; raise it if the run blows up.
friction = 3.0e-2       # mu, large-scale drag; 0 lets the inverse cascade pile up.

[grid]
n = 32                  # points per side (32 here; 96-256 for real turbulence)
length = 25.1327        # box side, 8 * 2*pi rho_s; sets the smallest wavenumber 2*pi/length

[run]
dt = 5.0e-3             # RK4 step; stiff in alpha / k_min^2, halve it if unstable
steps_per_block = 200   # steps between diagnostics (one time/energy/flux sample each)
blocks = 4              # total time = blocks * steps_per_block * dt
seed = 0                # RNG seed of the initial noise
seed_amplitude = 5.0e-2 # rms of the initial noise
save_frames = true      # also store the vorticity after every block (for movies)

# Optional.  d<flux>/d(parameter) by autodiff, checked against finite differences.
[autodiff]
parameter = "gradient"  # any [hasegawa_wakatani] key
fd_step = 1.0e-3        # relative step of the finite-difference check

[output]
directory = "../../output/hasegawa_wakatani"   # relative to this file
```

```text
$ drbx run examples/inputs/hasegawa_wakatani.toml
[hasegawa_wakatani] block 1/4 t=1 energy=1.7477e-04 flux=+2.9898e-05
[hasegawa_wakatani] block 4/4 t=4 energy=1.5738e-04 flux=+1.1052e-05
[hasegawa_wakatani] d<flux>/dgradient = +3.4848e-05 (finite difference +3.4848e-05)
[hasegawa_wakatani] wrote output/hasegawa_wakatani/hasegawa_wakatani_summary.json and ..._arrays.npz
```

```python
import json, numpy as np
results = json.load(open("output/hasegawa_wakatani/hasegawa_wakatani_summary.json"))["results"]
energy = np.load("output/hasegawa_wakatani/hasegawa_wakatani_arrays.npz")["energy"]   # (blocks,)
```

## Hasegawa-Wakatani drift-wave turbulence

`[model] backend = "hasegawa_wakatani"`

Two-field resistive drift-wave model on a periodic pseudo-spectral box (`hw_run`; `examples/tokamak/drift_wave_turbulence.py`). Lengths in rho_s.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `hasegawa_wakatani.adiabaticity` | `float` | `1.0` | alpha: parallel electron response coupling density and potential. (1/time) |
| `hasegawa_wakatani.gradient` | `float` | `1.0` | kappa: background density-gradient drive. (1/length) |
| `hasegawa_wakatani.hyperviscosity` | `float` | `0.001` | nu: Laplacian-squared damping of the grid scale. (length^4/time) |
| `hasegawa_wakatani.friction` | `float` | `0.0` | mu: linear drag absorbing the inverse cascade. (1/time) |
| `grid.n` | `int` | `96` | Points per side of the periodic `n x n` box. (cells) |
| `grid.length` | `float` | `50.26548245743669` | Box side length. (rho_s) |
| `run.dt` | `float` | `0.005` | RK4 time step. (time) |
| `run.steps_per_block` | `int` | `800` | RK4 steps between diagnostics; one entry of `time`, `energy`, `particle_flux` per block. (steps) |
| `run.blocks` | `int` | `40` | Number of blocks; total time is `blocks * steps_per_block * dt`. |
| `run.seed` | `int` | `0` | RNG seed of the initial noise (Hasegawa-Wakatani) or of the seeded mode phases (turbulence). |
| `run.seed_amplitude` | `float` | `0.05` | RMS of the initial real-space noise field. |
| `run.save_frames` | `bool` | `False` | Also write `vorticity_frames` (one frame per block). |
| `autodiff.parameter` | `str` | `required` | Hasegawa-Wakatani parameter to differentiate: `adiabaticity`, `gradient`, `hyperviscosity` or `friction`. The table is optional; omit it to skip the gradient. |
| `autodiff.fd_step` | `float` | `0.001` | Central finite-difference step as a fraction of the parameter value (floor 1), used to check the autodiff gradient. (relative) |
| `runtime.precision` | `str` | `'float64'` | `float64` or `float32` floating-point precision for the whole run. |
| `output.directory` | `Path | None` | `None` | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `output.case_name` | `str | None` | `None` | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

**Outputs**

| Name | Shape | Meaning |
| --- | --- | --- |
| `time` | `(blocks,)` | Time at the end of each block. |
| `energy` | `(blocks,)` | Fluctuation energy `sum(k^2 |phi_k|^2 + |n_k|^2) / n^4`. |
| `particle_flux` | `(blocks,)` | Domain-averaged radial E x B flux. |
| `vorticity`, `density` | `(n, n)` | Final real-space fields. |
| `vorticity_frames` | `(blocks, n, n)` | Vorticity after each block (only with `save_frames`). |

`results` keys: `final_time`, `final_energy`, `final_particle_flux`, `mean_particle_flux_second_half`, `autodiff` (only with `[autodiff]`: parameter, value, gradient, finite_difference, relative_error).

## Open-field-line SOL flux tube

`[model] backend = "sol_flux_tube"`

Isothermal Euler system with Bohm sheaths at both targets plus the FCI sheath/recycling accounting (`sol_flux_tube_run`, `compute_fci_sheath_recycling`; `examples/sol/open_sol_flux_tube.py`).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `sol_flux_tube.sound_speed` | `float` | `1.0` | Normalized isothermal sound speed; sets the Bohm outflow. (c_s) |
| `sol_flux_tube.source_amplitude` | `float` | `0.02` | Peak of the Gaussian upstream particle source. (n0 c_s/L) |
| `sol_flux_tube.source_width` | `float` | `4.0` | Parallel 1/e width of the source. (L) |
| `sol_flux_tube.density_floor` | `float` | `1e-06` | Positivity floor applied after each step. (n0) |
| `open_slab.shape` | `tuple[int, int, int]` | `required` | `[nx, ny, nz]` grid; `nz` cells along the field between the targets. (cells) |
| `open_slab.radial_extent` | `float` | `1.0` | Radial size of the flux tube. (L) |
| `open_slab.binormal_extent` | `float` | `1.0` | Binormal size of the flux tube. (L) |
| `open_slab.parallel_length` | `float` | `40.0` | Connection length between the two target plates. (L) |
| `sheath.recycling_fraction` | `float` | `0.98` | Fraction of the target ion flux returned as neutrals (0 = perfect absorber). |
| `sheath.electron_sheath_transmission` | `float` | `5.0` | Electron sheath heat-transmission coefficient. |
| `sheath.ion_sheath_transmission` | `float` | `3.5` | Ion sheath heat-transmission coefficient. |
| `sheath.recycled_neutral_energy` | `float` | `0.03` | Energy carried by each recycled neutral. (T_norm) |
| `sheath.ion_mass` | `float` | `1.0` | Ion mass in the Bohm sound speed. (m_p) |
| `sheath.density_floor` | `float` | `1e-12` | Density floor inside the sheath closure. (n0) |
| `sheath.temperature_floor` | `float` | `1e-12` | Temperature floor inside the sheath closure. (T_norm) |
| `run.cfl` | `float` | `0.4` | Time step is `cfl * dz / (c_s + 1)`. |
| `run.steps` | `int` | `60000` | Total time steps (SOL, recycling), or RK4 steps (turbulence). (steps) |
| `run.report_every` | `int` | `10000` | Steps per chunk; the steady-state residual is recorded after each chunk. (steps) |
| `run.electron_temperature` | `float` | `0.5` | Uniform electron temperature of the target sheath diagnostic. (T_norm) |
| `run.ion_temperature` | `float` | `0.5` | Uniform ion temperature of the target sheath diagnostic. (T_norm) |
| `runtime.precision` | `str` | `'float64'` | `float64` or `float32` floating-point precision for the whole run. |
| `output.directory` | `Path | None` | `None` | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `output.case_name` | `str | None` | `None` | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

**Outputs**

| Name | Shape | Meaning |
| --- | --- | --- |
| `z`, `density`, `momentum`, `mach`, `source` | `(nz,)` | Parallel profiles of the relaxed state; `mach` is -1 and +1 at the targets. |
| `report_step`, `residual` | `(n_reports,)` | Step count and steady-state residual `max|dn/dt|` per chunk. |
| `ion_particle_loss`, `target_heat_load`, `recycled_particle_source` | `(nz,)` | Sheath accounting per cell (non-zero at the targets). |

`results` keys: `mach_at_targets`, `n_target_over_upstream` (two-point prediction 0.5), `total_ion_particle_loss`, `total_recycled_particle_source`, `total_target_heat_load`, `particle_recycling_residual`, `current_balance_residual`, `final_time`.

## Recycling SOL with neutrals

`[model] backend = "sol_recycling"`

Plasma plus diffusive neutral with AMJUEL rates and a prescribed temperature profile (`sol_recycling_run`; `examples/sol/recycling_sol.py`). The reference run uses `upstream_density`; each `[scan]` value is a fresh run.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `sol_recycling.parallel_length` | `float` | `30.0` | Connection length, stagnation point to target (sets the atomic-rate scale). (m) |
| `sol_recycling.upstream_density` | `float` | `1.0` | Upstream ion density pinned at `z = 0`. (N_norm) |
| `sol_recycling.recycling_fraction` | `float` | `1.0` | Fraction R of the target ion flux returned as neutrals. |
| `sol_recycling.neutral_diffusion` | `float` | `8.0` | Neutral parallel diffusion enhancement (kinetic-transport proxy). |
| `sol_recycling.neutral_temperature` | `float` | `0.04` | Neutral temperature (0.04 is about 2 eV). (T_norm) |
| `sol_recycling.ion_mass` | `float` | `2.0` | Ion mass (2 = deuterium). (m_p) |
| `sol_recycling.density_floor` | `float` | `0.0001` | Density floor. (N_norm) |
| `sol_recycling.normalization.Nnorm` | `float` | `1e+19` | Density normalization. (m^-3) |
| `sol_recycling.normalization.Tnorm` | `float` | `100.0` | Temperature normalization. (eV) |
| `sol_recycling.normalization.Bnorm` | `float` | `1.0` | Magnetic-field normalization. (T) |
| `sol_recycling.normalization.ion_mass` | `float` | `2.0` | Ion mass used by the normalization. (m_p) |
| `profile.nz` | `int` | `200` | Parallel cells, stagnation point to target. (cells) |
| `profile.upstream_ev` | `float` | `30.0` | Prescribed upstream temperature. (eV) |
| `profile.target_ev` | `float` | `1.5` | Prescribed target temperature (quadratic profile between the two). (eV) |
| `profile.neutral_seed` | `float` | `0.05` | Initial trace neutral density. (N_norm) |
| `run.dt` | `float | None` | `None` | Time step; default `0.1 / nz` (CFL 0.3 against 3 c_s) when omitted. (L/c_s) |
| `run.steps` | `int` | `60000` | Total time steps (SOL, recycling), or RK4 steps (turbulence). (steps) |
| `run.report_every` | `int` | `15000` | Steps per chunk; the steady-state residual is recorded after each chunk. (steps) |
| `scan.upstream_density` | `tuple[float, ...]` | `[]` | Extra steady states, one per value (recycling: `N_norm`; detachment: m^-3). Detachment reuses the previous solution as the next guess. Empty: none. |
| `runtime.precision` | `str` | `'float64'` | `float64` or `float32` floating-point precision for the whole run. |
| `output.directory` | `Path | None` | `None` | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `output.case_name` | `str | None` | `None` | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

**Outputs**

| Name | Shape | Meaning |
| --- | --- | --- |
| `z_over_L`, `temperature`, `mach`, `ion_density`, `ion_momentum`, `neutral_density` | `(nz,)` | Final profiles (target at `z_over_L = 1`). |
| `scan_upstream_density`, `scan_target_mach`, `scan_target_ion_flux` | `(n_scan,)` | Detachment-onset scan. |

`results` keys: `target_mach`, `target_ion_flux`, `dt`, `neutral_cushion_ratio` (neutral density at the target over the midpoint).

## SD1D 1-D detachment

`[model] backend = "detachment_sd1d"`

Plasma-neutral leg solved to steady state by pseudo-transient Newton continuation, matched to SD1D (`detachment_sol_run`; `examples/benchmarks/b6_detachment_sd1d.py`). Scan points start from the previous solution.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `detachment_sd1d.ny` | `int` | `800` | Nonuniform cells along the leg. (cells) |
| `detachment_sd1d.length` | `float` | `30.0` | Total leg length. (m) |
| `detachment_sd1d.length_xpt` | `float` | `10.0` | Length of the source region and position of the area kink (X-point). (m) |
| `detachment_sd1d.area_expansion` | `float` | `2.0` | Flux-tube area ratio between the X-point and the target. |
| `detachment_sd1d.dymin` | `float` | `0.1` | Smallest cell size (grid refinement toward the target). |
| `detachment_sd1d.upstream_density` | `float` | `3e+19` | Density held at the upstream cell. (m^-3) |
| `detachment_sd1d.power_flux` | `float` | `50000000.0` | Power flux injected over `length_xpt`. (W/m^2) |
| `detachment_sd1d.particle_flux` | `float` | `4e+23` | Particle-source flux (shape of the source). (m^-2 s^-1) |
| `detachment_sd1d.ionisation_energy` | `float` | `13.6` | Energy lost per ionisation. (eV) |
| `detachment_sd1d.recycling_fraction` | `float` | `0.99` | Target recycling fraction. |
| `detachment_sd1d.neutral_diffusion` | `float` | `10.0` | Neutral diffusion multiplier (`dneut`). |
| `detachment_sd1d.neutral_temperature_floor` | `float` | `0.5` | Neutral temperature floor. (eV) |
| `detachment_sd1d.recycled_temperature` | `float` | `3.5` | Franck-Condon temperature of recycled neutrals. (eV) |
| `detachment_sd1d.max_mean_free_path` | `float` | `0.1` | Cap on the neutral mean free path. (m) |
| `detachment_sd1d.sheath_gamma` | `float` | `6.0` | Sheath heat-transmission coefficient. |
| `detachment_sd1d.charge_exchange` | `bool` | `True` | Include charge-exchange friction. |
| `detachment_sd1d.recombination` | `bool` | `True` | Include recombination. |
| `detachment_sd1d.Nnorm` | `float` | `1e+20` | Density normalization. (m^-3) |
| `detachment_sd1d.Tnorm` | `float` | `100.0` | Temperature normalization. (eV) |
| `detachment_sd1d.Bnorm` | `float` | `1.0` | Magnetic-field normalization. (T) |
| `detachment_sd1d.AA` | `float` | `2.0` | Ion mass number. (m_p) |
| `scan.upstream_density` | `tuple[float, ...]` | `[]` | Extra steady states, one per value (recycling: `N_norm`; detachment: m^-3). Detachment reuses the previous solution as the next guess. Empty: none. |
| `run.source_scale` | `float` | `0.1` | Initial multiplier of the proportional-integral particle source. |
| `run.tol` | `float` | `1e-08` | Stop when the scaled steady-state residual falls below this. |
| `run.max_iter` | `int` | `400` | Maximum pseudo-transient Newton iterations. |
| `run.initial_dt` | `float` | `100.0` | Initial backward-Euler step (grown by the residual ratio). (1/Omega_ci) |
| `run.newton_switch` | `float` | `1e-05` | Residual below which the step is a full Newton step. |
| `runtime.precision` | `str` | `'float64'` | `float64` or `float32` floating-point precision for the whole run. |
| `output.directory` | `Path | None` | `None` | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `output.case_name` | `str | None` | `None` | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

**Outputs**

| Name | Shape | Meaning |
| --- | --- | --- |
| `upstream_density`, `target_temperature_ev`, `target_ion_flux_area`, `target_density`, `target_mach`, `upstream_temperature_ev`, `residual`, `iterations`, `converged`, `particle_imbalance`, `power_imbalance` | `(n_points,)` | Per steady state: m^-3, eV, m^-2 s^-1, m^-3, -, eV, -, -, -, -, -. |
| `profile_ion_density`, `profile_ion_momentum`, `profile_plasma_pressure`, `profile_neutral_density`, `profile_neutral_momentum`, `profile_neutral_pressure` | `(ny,)` | SD1D-normalized fields of the last steady state. |

`results` keys: `points` (one dict per steady state with the values above).

## Four-field FCI turbulence

`[model] backend = "stellarator_turbulence"`

Density, vorticity and parallel velocities with a GMRES potential inversion on an FCI geometry (`run_stellarator_turbulence`; `examples/stellarator/stellarator_turbulence.py`). Exactly one geometry table: `[rotating_ellipse]` (stellarator, optional limiter SOL), `[shifted_torus]` (tokamak-like), `[island_divertor]` or `[essos]` (coil field lines).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `rotating_ellipse.shape` | `tuple[int, int, int]` | `required` | `[nx, ny, nz]` grid (radial, poloidal, toroidal). (cells) |
| `rotating_ellipse.r0` | `float` | `3.0` | Major-radius offset of the torus. (L) |
| `rotating_ellipse.x_min` | `float` | `0.2` | Inner radial (minor-radius label) bound. (L) |
| `rotating_ellipse.x_max` | `float` | `1.0` | Outer radial bound. (L) |
| `rotating_ellipse.elongation` | `float` | `0.35` | Ellipse deformation delta in [0, 1); aspect ratio `(1+delta)/(1-delta)`. |
| `rotating_ellipse.n_field_periods` | `int` | `1` | Ellipse rotations per toroidal turn (field periods). |
| `rotating_ellipse.iota` | `float` | `0.9` | Rotational transform of the field. |
| `rotating_ellipse.c_phi` | `float` | `3.0` | Toroidal-field scaling constant. |
| `rotating_ellipse.map_substeps` | `int` | `8` | Field-line tracer substeps per toroidal cell. |
| `rotating_ellipse.limiter_radius` | `float | None` | `None` | Toroidal limiter at zeta=0 opening every line with `x > limiter_radius` into a sheath-drained SOL; omit for all-closed lines. (L) |
| `shifted_torus.shape` | `tuple[int, int, int]` | `required` | `[nx, ny, nz]` grid (radial, poloidal, toroidal). (cells) |
| `shifted_torus.x_min` | `float` | `0.15` | Inner radial (minor-radius label) bound. (L) |
| `shifted_torus.x_max` | `float` | `1.0` | Outer radial bound. (L) |
| `shifted_torus.r0` | `float` | `3.0` | Major-radius offset of the torus. (L) |
| `shifted_torus.alpha_value` | `float` | `0.25` | Shafranov-like shift amplitude. |
| `shifted_torus.iota` | `float` | `1.1` | Rotational transform of the field. |
| `shifted_torus.c_phi` | `float` | `3.0` | Toroidal-field scaling constant. |
| `shifted_torus.sigma` | `float` | `0.0` | Poloidal shear; non-zero makes the metric non-axisymmetric. |
| `island_divertor.shape` | `tuple[int, int, int]` | `required` | `[nx, ny, nz]` grid (radial, poloidal, toroidal). (cells) |
| `island_divertor.r0` | `float` | `3.0` | Major-radius offset of the torus. (L) |
| `island_divertor.elongation` | `float` | `0.35` | Ellipse deformation delta in [0, 1); aspect ratio `(1+delta)/(1-delta)`. |
| `island_divertor.n_field_periods` | `int` | `1` | Ellipse rotations per toroidal turn (field periods). |
| `island_divertor.c_phi` | `float` | `3.0` | Toroidal-field scaling constant. |
| `island_divertor.map_substeps` | `int` | `8` | Field-line tracer substeps per toroidal cell. |
| `island_divertor.open_field_line_masks` | `bool` | `False` | Mark open endpoints by multi-transit field-line tracing. |
| `island_divertor.mask_max_transits` | `int` | `30` | Maximum toroidal transits traced when classifying open lines. (transits) |
| `island_divertor.field.x_min` | `float` | `0.2` | Inner radial bound. (L) |
| `island_divertor.field.x_max` | `float` | `1.0` | Outer radial bound. (L) |
| `island_divertor.field.iota_axis` | `float` | `0.55` | Rotational transform at `x_min` (linear profile). |
| `island_divertor.field.iota_edge` | `float` | `0.82` | Rotational transform at `x_max`. |
| `island_divertor.field.resonances` | `tuple[tuple[int, int, float], ...]` | `[[3, 2, 0.005], [4, 3, 0.005], [5, 4, 0.005]]` | `[[m, n, epsilon], ...]`: island chains at `iota = n/m` with relative width epsilon. |
| `essos.coil_json_path` | `str | Path | None` | `None` | ESSOS coil JSON; default the bundled Landreman-Paul QA coils. |
| `essos.vmec_wout_path` | `str | Path | None` | `None` | VMEC `wout` file of the seed surfaces. Not bundled: pass it or set `DRBX_ESSOS_ROOT`. |
| `essos.map_source` | `str` | `'coil'` | `coil` (trace the coil field, keep open lines), `vmec` (closed surface-preserving maps) or `hybrid`. |
| `essos.nx` | `int` | `6` | Radial cells. (cells) |
| `essos.ny` | `int` | `8` | Toroidal-angle cells. (cells) |
| `essos.nz` | `int` | `16` | Poloidal-angle cells. (cells) |
| `essos.rho_min` | `float` | `0.1` | Inner normalized radius. |
| `essos.rho_max` | `float` | `0.46` | Outer normalized radius. |
| `essos.maxtime` | `float` | `140.0` | Integration time of each traced line. (ESSOS units) |
| `essos.times_to_trace` | `int` | `768` | Samples per traced line. (samples) |
| `essos.trace_tolerance` | `float` | `1e-08` | Integrator tolerance. |
| `essos.limiter_rho` | `float | None` | `None` | Opens every surface beyond this rho into a sheath-drained SOL; omit for all closed. |
| `run.steps` | `int` | `required` | Number of RK4 steps (required). (steps) |
| `run.dt` | `float` | `required` | RK4 time step (required). (time) |
| `run.amplitude` | `float` | `0.08` | Peak relative density perturbation of the seeded multi-mode state. |
| `run.seed` | `int` | `0` | RNG seed of the initial noise (Hasegawa-Wakatani) or of the seeded mode phases (turbulence). |
| `run.sheath_sink` | `bool` | `False` | Drain the open (SOL) endpoints with the Bohm sheath sink each step. |
| `run.frame_stride` | `int` | `1` | Record a frame every this many steps (the last step is always recorded). (steps) |
| `runtime.precision` | `str` | `'float64'` | `float64` or `float32` floating-point precision for the whole run. |
| `output.directory` | `Path | None` | `None` | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `output.case_name` | `str | None` | `None` | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

`run.steps` and `run.dt` are required.

**Outputs**

| Name | Shape | Meaning |
| --- | --- | --- |
| `density_frames`, `omega_frames` | `(n_frames, nx, ny, nz) float32` | Fields at the recorded steps (frame 0 is the initial state). |
| `times`, `particle_content`, `target_flux` | `(n_frames,)` | Time, Jacobian-weighted total density, total sheath ion loss (0 unless `sheath_sink`). |

`results` keys: `geometry`, `shape`, `n_frames`, `initial_particle_content`, `final_particle_content`, `total_target_flux`, `finite`.

## Differentiable FCI flux tube

`[model] backend = "fci_differentiable"`

Two-field shifted-torus rollout: reverse-mode gradient in the initial amplitude against a central finite difference (`differentiability_report`; `examples/stellarator/fci_differentiable.py`).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `fci_differentiable.shape` | `tuple[int, int, int]` | `[16, 16, 8]` | `[nx, ny, nz]` grid. (cells) |
| `fci_differentiable.sigma` | `float` | `0.6` | Poloidal shear of the shifted torus. |
| `fci_differentiable.rho_star` | `float` | `1.0` | Drift scale rho_*. |
| `fci_differentiable.x_min` | `float` | `0.15` | Inner radial bound. (L) |
| `fci_differentiable.x_max` | `float` | `1.0` | Outer radial bound. (L) |
| `fci_differentiable.r0` | `float` | `3.0` | Major-radius offset. (L) |
| `fci_differentiable.alpha_value` | `float` | `0.25` | Shafranov-like shift. |
| `fci_differentiable.iota` | `float` | `1.1` | Rotational transform. |
| `fci_differentiable.c_phi` | `float` | `3.0` | Toroidal-field scaling. |
| `run.amp0` | `float` | `0.1` | Initial perturbation amplitude at which the gradient is evaluated. |
| `run.n_steps` | `int` | `24` | Rollout steps. (steps) |
| `run.dt` | `float` | `0.001` | Rollout time step. (time) |
| `run.fd_step` | `float` | `1e-05` | Central finite-difference step in `amp0`. |
| `runtime.precision` | `str` | `'float64'` | `float64` or `float32` floating-point precision for the whole run. |
| `output.directory` | `Path | None` | `None` | Directory for the two result files. Default `./output/<case_name>`; relative paths resolve against the deck file. |
| `output.case_name` | `str | None` | `None` | File prefix. Default: the deck file name without `.toml`. `--case-name` overrides. |

**Outputs**

| Name | Shape | Meaning |
| --- | --- | --- |
| `density`, `v_parallel` | `(nx, ny, nz)` | Final fields. |

`results` keys: `grad` (autodiff), `fd` (finite difference), `rel_error`, `fd_step`, `amp0`, `n_steps`, `dt`, `objective_value`, `finite`, `density_max`, `v_parallel_max`.

## HSX FCI drift-reduced Braginskii

`[model] backend = "fci_braginskii"`

Seven-field electrostatic Boussinesq model on the HSX toroidal FCI geometry, IMEX-SSP222 with an FGMRES potential solve (`HsxBlobConfig`; [HSX backend](fci_braginskii_hsx_backend.md)). With a `[geometry]` table the geometry is built from the shipped HSX coils and vessel (`build_hsx_simulation_geometry`) and cached.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `fci_braginskii.geometry` | `Path | None` | `None` | Directory of a prebuilt geometry artifact. Omit it and give a `[geometry]` table to build one from the shipped coils. |
| `fci_braginskii.shard_counts` | `tuple[int, int, int]` | `[1, 1, 1]` | JAX decomposition `[su, sv, seta]`; only eta decomposition is supported, so the first two are 1. |
| `fci_braginskii.halo_width` | `int` | `2` | Halo width exchanged between shards. (cells) |
| `fci_braginskii.filament_cache_dir` | `Path` | `<repo>/.hsx_filament_cache` | Directory of reusable initial-filament field-line label caches. |
| `fci_braginskii.no_filament_cache` | `bool` | `False` | Disable loading and writing filament caches. |
| `fci_braginskii.rebuild_filament_cache` | `bool` | `False` | Ignore a matching cache and replace it after tracing. |
| `fci_braginskii.final_time` | `float` | `0.15` | Final simulation time. (normalized time) |
| `fci_braginskii.num_steps` | `int` | `200` | Equal IMEX-SSP222 steps to reach `final_time`. (steps) |
| `fci_braginskii.save_every` | `int` | `1` | History is stored every this many steps. (steps) |
| `fci_braginskii.checkpoint_every` | `int` | `0` | Atomic restartable checkpoint every N steps; 0 disables. (steps) |
| `fci_braginskii.snapshot_times` | `tuple[float, ...]` | `[]` | Times at which snapshot files are written (first step at or after each). (normalized time) |
| `fci_braginskii.snapshot_dir` | `Path | None` | `None` | Directory for snapshot files; default the output directory. |
| `fci_braginskii.restart_from` | `Path | None` | `None` | Snapshot or history NPZ to restart from. |
| `fci_braginskii.restart_frame` | `int` | `-1` | Frame loaded from a history NPZ; ignored for a snapshot. |
| `fci_braginskii.diagnostic_every` | `int` | `0` | Print min/max diagnostics for every field every N steps; 0 disables. (steps) |
| `fci_braginskii.blob_initialization` | `str` | `'field-aligned'` | `field-aligned` (backtraced finite filament) or `logical` (eta-copied Gaussian). |
| `fci_braginskii.density_amplitude` | `float` | `0.05` | Relative density amplitude of the blob. (n0) |
| `fci_braginskii.temperature_amplitude` | `float` | `0.0` | Electron-temperature perturbation (legacy logical initialization only). (T0) |
| `fci_braginskii.blob_center` | `tuple[float, float]` | `[0.65, 0.5]` | Blob center `[u, v]`. (logical u, v) |
| `fci_braginskii.blob_width` | `float` | `0.1` | Gaussian width of the blob. (logical u, v) |
| `fci_braginskii.blob_reference_eta` | `float` | `3.141592653589793` | Toroidal reference plane of the filament. (rad) |
| `fci_braginskii.blob_parallel_half_length` | `float` | `3.141592653589793` | Half-length of the cos^2 filament envelope in eta; at most pi. (rad) |
| `fci_braginskii.fieldline_substeps_per_plane` | `int` | `4` | RK4 tracing substeps per plane spacing for the initial labels. |
| `fci_braginskii.toroidal_perturbation_amplitude` | `float` | `0.0` | Relative cosine modulation (legacy logical blob only). |
| `fci_braginskii.toroidal_perturbation_mode` | `int` | `1` | Full-torus integer mode number of the perturbation. |
| `fci_braginskii.toroidal_perturbation_phase` | `float` | `0.0` | Initial toroidal perturbation phase. (rad) |
| `fci_braginskii.tau` | `float` | `1.0` | Ion-to-electron temperature ratio. |
| `fci_braginskii.rho_star` | `float` | `0.0005` | rho_s / L_ref (L_ref = 1 m): scales E x B and curvature drifts by rho_*, polarization by rho_*^2. |
| `fci_braginskii.mi_over_me` | `float` | `1836.0` | Ion-to-electron mass ratio. |
| `fci_braginskii.perp_diffusion` | `float` | `1e-05` | Perpendicular diffusion coefficient. (normalized) |
| `fci_braginskii.parallel_diffusion` | `float` | `0.0` | Parallel diffusion coefficient. (normalized) |
| `fci_braginskii.electron_collision_frequency` | `float` | `0.0` | Electron collision frequency. (normalized) |
| `fci_braginskii.implicit_current_phi_pair` | `bool` | `True` | Treat `mu grad_par(phi)` and the parallel current divergence implicitly through a coupled potential solve (default). |
| `fci_braginskii.advance_execution` | `str` | `'staged-compiled'` | `compiled` (one fused executable), `staged-compiled` (reusable kernels) or `eager`. |
| `fci_braginskii.gmres_acceptance_tolerance` | `float` | `5e-05` | Residual accepted after GMRES stops. |
| `fci_braginskii.gmres_target_tolerance` | `float` | `1e-08` | Residual target that stops GMRES. |
| `fci_braginskii.gmres_max_iterations` | `int` | `500` | Maximum GMRES iterations. |
| `fci_braginskii.gmres_restart` | `int` | `100` | GMRES restart length (capped at `gmres_max_iterations`). |
| `fci_braginskii.gmres_residual_correction_steps` | `int` | `1` | True-residual correction solves tried when GMRES would be rejected. |
| `fci_braginskii.gmres_preconditioner` | `str` | `'line-u'` | Right preconditioner of the potential solves: `line-u` (production per-eta radial-tree line solve) or `none`. |
| `fci_braginskii.no_phase_timing` | `bool` | `False` | Disable in-executable timing markers. |
| `fci_braginskii.geometry_only` | `bool` | `False` | Stop after loading and lowering the geometry. |
| `fci_braginskii.output` | `Path` | `<repo>/hsx_blob_history.npz` | History NPZ written at the end of the run. |
| `geometry.resolution` | `tuple[int, int, int]` | `required` | Toroidal cell counts `[nu, ntheta, neta]` (ntheta even). Required. (cells) |
| `geometry.makegrid_path` | `Path | None` | `None` | MAKEGRID vacuum-field file; when set it replaces the coils. |
| `geometry.vessel_path` | `Path | None` | `<repo>/vessel_hsx_flare.txt` | Vessel/wall boundary file. Default: the vessel shipped in `examples/inputs/hsx`. |
| `geometry.fit_sample_shape` | `tuple[int, int, int]` | `[8, 9, 8]` | Field-fit sample grid `[nr, nphi, nz]`. (points) |
| `geometry.radial_degree` | `int` | `3` | Radial polynomial degree of the flux-surface fit. |
| `geometry.vertical_degree` | `int` | `3` | Vertical polynomial degree of the fit. |
| `geometry.toroidal_modes` | `int` | `2` | Toroidal Fourier modes of the fit. |
| `geometry.metric_spline_degree` | `int` | `1` | Spline degree of the metric. |
| `geometry.mmpde_iterations` | `int` | `0` | Moving-mesh smoothing iterations. |
| `geometry.axis_core_radius` | `float` | `0.03` | Radius of the axis core region. (normalized) |
| `geometry.reference_magnetic_field` | `float | None` | `None` | Reference field; default taken from the field. (T) |
| `geometry.makegrid_currents` | `Sequence[float] | None` | `None` | One current per coil group; default QHS (six main groups at 10722 A, six auxiliary off). (A) |
| `geometry.metric_mesh_shape` | `tuple[int, int, int] | None` | `None` | Metric mesh `[nu, ntheta, neta_period]`; default `resolution`. (cells) |
| `geometry.metric_radial_degree` | `int` | `17` | Radial degree of the metric fit. |
| `geometry.metric_poloidal_modes` | `int` | `15` | Poloidal modes of the metric fit. |
| `geometry.metric_toroidal_modes` | `int` | `16` | Toroidal modes of the metric fit. |
| `geometry.eta_projection_iterations` | `int` | `0` | Eta-projection iterations. |
| `geometry.metric_cache_dir` | `Path | None` | `None` | Metric cache directory. |
| `geometry.map_cache_path` | `Path | None` | `None` | Independent FCI map cache/checkpoint. |
| `geometry.rebuild_metric_cache` | `bool` | `False` | Ignore a cached metric and rebuild. |
| `geometry.include_curvature_edge_one_form` | `bool` | `False` | Store the curvature edge one-form. |
| `geometry.trace_backend` | `str` | `'jax'` | `jax` (compiled, batch-sharded) or `numpy` (reference). |
| `geometry.trace_batch_size` | `int` | `2048` | Compiled trace batch size. (trajectories) |
| `geometry.trace_device_count` | `int | None` | `None` | Devices sharding the trace; default all. (devices) |
| `geometry.coils_path` | `Path | None` | `<repo>/coils.hsx` | MAKEGRID coils file (B from the coil filaments). Default: the HSX coils shipped in `examples/inputs/hsx`. |
| `geometry.coil_table_shape` | `tuple[int, int, int] | None` | `[96, 144, 96]` | In-memory coil field table `[nr, nphi, nz]` per field period. (points) |
| `geometry.cache_dir` | `Path` | `<repo>/hsx_geometry` | The built artifact is stored (and reused when the table is unchanged) under `cache_dir/hsx_<res>_<hash>`. |

**Outputs**

Written to `[fci_braginskii] output` (the `.npz` suffix is added; shapes below are from the 16^3 example deck, `n_saved` = steps/`save_every` + 1 = 5), plus snapshot/checkpoint files. `rc` = `[nu, nv, neta]`.

| Name | Shape | Meaning |
| --- | --- | --- |
| `times` | `(n_saved,)` | Stored normalized times, starting with the initial state. |
| `density`, `phi`, `Te`, `Ti`, `Vi`, `Ve`, `vorticity` | `(n_saved, 16, 16, 16)` float32 | Field histories. |
| `u`, `v`, `theta`, `eta` | `(nu,)`, `(nv,)`, `(nv,)`, `(neta,)` | Cell-center logical coordinates (`theta` equals `v`). |
| `jacobian`, `Bmag`, `owner_aggregate_volume`, `owner_active` | `(16, 16, 16)` | Cell Jacobian, field strength, owner volumes and mask. |
| `B_contravariant`, `cartesian` | `(16, 16, 16, 3)` | Contravariant field and cell positions. |
| `shard_counts`, `periodic_axes`, `axis_regular_axes`, `metric_mesh_shape` | `(3,)` | Decomposition, axis flags, metric mesh. |
| `logical_extents` | `(3, 2)` | Logical-coordinate extents. |
| `nfp`, `simulated_field_periods`, `toroidal_extent`, `metric_radial_degree`, `metric_poloidal_modes`, `metric_toroidal_modes`, `eta_projection_iterations`, `fci_trace_substeps` | `()` | Geometry bookkeeping. |
| `topology`, `coordinate_names_json`, `parallel_operator_scheme`, `history_dtype` | `()` string | Geometry and storage labels. |
| `run_metadata_json` | `()` string | JSON of the resolved run, including all solver settings. |

Snapshot and checkpoint files (`<stem>.snapshot_t<time>.npz`, `<stem>.checkpoint_step<step>.npz`, `<stem>.failure_step<step>.npz`) hold one frame: the seven fields without the time axis, `time`, `requested_snapshot_time`, `step`, the `metric_*` arrays, `jacobian`, `Bmag`, `B_contravariant`, parallel-coefficient diagnostics, `wall_ghost_states`, `polarization_residual`, `grid_scale_diagnostics` and `run_metadata_json`. Any of them, or a history file, is valid for `restart_from`.

