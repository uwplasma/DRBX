# Deck Reference

`drbx run deck.toml` runs every DRBX model from a TOML file. A deck names its model with `[model] backend = "<name>"`; every other table is the Python configuration dataclass or function signature of that model, so there is one set of defaults. A test fails when this page and the code disagree.

```bash
drbx inspect deck.toml   # print the resolved configuration
drbx run deck.toml       # run; --output-dir / --case-name override [output]
drbx run deck.toml --set run.blocks=2 --set 'hasegawa_wakatani.gradient=1.5'   # override any key (TOML value syntax, repeatable)
```

The same code path is available from Python; keyword arguments are `table={"key": value}` overrides, and the return value is the summary payload plus the `paths` written:

```python
import drbx

out = drbx.run("examples/inputs/hasegawa_wakatani.toml", output_dir="out",
               run={"blocks": 2}, hasegawa_wakatani={"gradient": 1.5})
print(out["results"], out["paths"])
```

`--set` and keyword overrides are validated by the same schema, so unknown keys are errors. (`--set` applies to `[model] backend` decks, not BOUT-style ones.)

- Unknown tables and keys are errors that list the valid ones; `required` keys have no default. Relative paths resolve against the deck. Decks without `backend` are the BOUT-style format ([Inputs And Outputs](input_output_reference.md)).
- The reduced-model backends accept `[runtime] precision` (`float64`/`float32`) and `[output] directory` (default `./output/<case_name>`), `case_name` (default: deck name), and write `<case_name>_summary.json` (`case_name`, `backend`, resolved `config`, `results`) and `<case_name>_arrays.npz`.
- Example decks are in `examples/inputs/`.
- Not available from a deck: optimization and inverse-design scripts (`hasegawa_wakatani_optimization.py`, `autodiff_diffusion_*.py`, `detachment_control.py`; the Hasegawa-Wakatani gradient is `[autodiff]`), VMEC/VMEX/VMEC-extender imports, and figure/movie scripts.

## Example decks

Most example decks run in under a minute. Reduced-model decks use `drbx run` and write to the deck's `[output] directory`. The test suite exercises those decks. The nodal example uses `drbx build` with separate device inputs before `drbx run`.

| Deck | Case | Time | What you get |
| --- | --- | --- | --- |
| [`hasegawa_wakatani.toml`](../examples/inputs/hasegawa_wakatani.toml) | Drift-wave turbulence, autodiff gradient | 4 s | energy and flux histories, final fields; `d<flux>/d(gradient)` agrees with finite differences to 1e-6 |
| [`four_field_tokamak.toml`](../examples/inputs/four_field_tokamak.toml) | Four-field FCI, tokamak-like torus | 5 s | 5 density/vorticity frames, particle content |
| [`stellarator_turbulence.toml`](../examples/inputs/stellarator_turbulence.toml) | Four-field FCI, rotating-ellipse stellarator with limiter SOL | 20 s | frames, particle content, sheath ion loss |
| [`electromagnetic_turbulence.toml`](../examples/inputs/electromagnetic_turbulence.toml) | Four-field FCI, tokamak-like torus with A_par, flutter and electron inertia | 20 s | frames, `apar_frames`, `energies`, particle content |
| [`island_divertor.toml`](../examples/inputs/island_divertor.toml) | Four-field FCI, island divertor edge | 12 s | frames, particle content; open lines drain through the sheath |
| [`sol_flux_tube.toml`](../examples/inputs/sol_flux_tube.toml) | Open SOL flux tube with Bohm sheath | 2 s | Mach profile (about 0.9 at the targets after 6000 steps), n_target/n_up about 0.49, sheath and recycling totals |
| [`sol_recycling.toml`](../examples/inputs/sol_recycling.toml) | Recycling SOL with neutrals, detachment-onset scan | 2 s | target Mach 1.45 at n_up = 4, falling to 0.61 at n_up = 8 |
| [`detachment_sd1d.toml`](../examples/inputs/detachment_sd1d.toml) | SD1D detachment, density scan | 30 s | target temperature 15.5 eV at 2e19 m^-3, 7.2 eV at 3e19 m^-3 |
| [`fci_differentiable.toml`](../examples/inputs/fci_differentiable.toml) | Autodiff versus finite difference through an FCI rollout | 12 s | gradient and finite difference agree to about 1e-10 |
| [`restartable_diffusion.toml`](../examples/inputs/restartable_diffusion.toml) | Restart (BOUT-style deck) | 5 s | summary, arrays and a restart file; `--restart-in` continues it |
| [`fci_nodal_mms.toml`](../examples/inputs/fci_nodal_mms.toml) | N16 HSX manufactured solution | Separate case build | Receipts, final nodal state and optional static RHS |

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

Plasma plus a diffusive neutral (the SD1D closure) with AMJUEL rates and a prescribed temperature profile (`sol_recycling_run`; `examples/sol/recycling_sol.py`). The reference run uses `upstream_density`; each `[scan]` value is a fresh run.

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
| `detachment_sd1d.sheath_momentum_flux` | `str` | `'minmod'` | Target-face ion momentum flux: `minmod` (SD1D) or `face` (`N V_out^2`). |
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

Density, vorticity and parallel velocities with a GMRES potential inversion on an FCI geometry (`run_stellarator_turbulence`; `examples/stellarator/stellarator_turbulence.py`). Exactly one geometry table: `[rotating_ellipse]` (stellarator, optional limiter SOL), `[shifted_torus]` (tokamak-like), `[island_divertor]` or `[essos]` (coil field lines). `[physics] electromagnetic = true` adds A_par, magnetic flutter and electron inertia and then also writes the arrays `apar_frames` (A_par per frame) and `energies` (E x B, magnetic, electron and ion kinetic energy per frame) and the `results` key `final_energy_components`.

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
| `physics.electromagnetic` | `bool` | `False` | Add A_par, magnetic flutter and electron inertia; `v_electron_parallel` then evolves the canonical momentum `v_e - (mi/me) A_par`. Off: the electrostatic model, unchanged. |
| `physics.beta` | `float` | `0.001` | `mu0 n T / B^2` in the model's normalization; Alfven speed `B / sqrt(beta n)`. Positive when `electromagnetic`. |
| `physics.mi_over_me` | `float` | `1836.0` | Ion-to-electron mass ratio (sets the electron skin depth `d_e^2 = 1 / (beta mi/me n)`). |
| `physics.reference_density` | `float` | `1.0` | `n_ref` of the fixed-coefficient Ampere (Helmholtz) operator; density variations about it are iterated. |
| `physics.ampere_iterations` | `int` | `2` | Defect-correction passes for the density variation (one pass is exact for uniform density). |
| `physics.ampere_tol` | `float` | `1e-08` | Relative residual of each Ampere solve. |
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

## Nodal FCI manufactured cases

`[model] backend = "fci_nodal"`

Build a receipt-bound case and run a manufactured solution from the publication.
The example is `examples/inputs/fci_nodal_mms.toml`. See [FCI nodal manufactured solutions](fci_nodal.md).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `fci_nodal.case_dir` | `Path` | `required` | Build products, receipts and MMS artifacts. |
| `fci_nodal.resolution` | `int` | `required` | Nodal resolution and total number of eta planes. |
| `fci_nodal.symmetry` | `str` | `required` | `nfp` replicates the source period. `none` builds the full turn. |
| `fci_nodal.field_periods` | `int` | `required` | Number of physical field periods, shared by the filter and map. |
| `fci_nodal.initial_state` | `Path | None` | `None` | A supplied path is refused with a missing physical-state loader and layout validation error. |
| `fci_nodal.owners` | `int` | `1` | Number of owner slabs for the build and MMS. Each slab requires at least four eta planes. |
| `fci_nodal.threads` | `int` | `2` | Positive library thread limit per process for build and MMS stages other than polygon and geometry, which use one thread. |
| `geometry.makegrid_path` | `Path` | `required` | MAKEGRID magnetic field archive. |
| `geometry.vessel_path` | `Path` | `required` | Material vessel data. |
| `geometry.makegrid_currents` | `tuple[float, ...]` | `required` | Currents in the archive's coil-group order. |
| `geometry.rings` | `int` | `required` | Number of rings in the polygon band. |
| `geometry.resolution` | `tuple[int, int, int]` | `required` | Dense metric output grid. |
| `geometry.workers` | `int` | `1` | Local geometry build workers. |
| `field_filter.max_harmonic_per_period` | `int` | `required` | Retained field-table harmonic cutoff. |
| `field_filter.samples_per_period` | `int` | `required` | Samples of each column per field period. |
| `field_filter.table_grid` | `tuple[int, int, float]` | `required` | Radial samples, poloidal samples and maximum logical radius. |
| `field_filter.filter_map` | `bool` | `False` | Also low-pass the solved map in eta for filtered verification geometry. |
| `parallel.workers` | `int` | `1` | Local workers for indexed parallel construction. |
| `mms.field_profile` | `str` | `'wall_flat_temperature'` | Manufactured field profile `wall_flat_temperature` or `prescribed_wall_heat_flux`. |
| `mms.wall_law` | `str` | `'floating_wall'` | Wall law `conducting_wall` or `floating_wall`. |
| `mms.max_steps` | `int | None` | `None` | Positive step cap. With the cap omitted, advance to the native final milestone. |
| `mms.stop_after_first` | `bool` | `False` | Use the first native milestone at its supported resolution. |
| `mms.capture_static_rhs` | `bool` | `False` | Record static raw and manufactured RHS arrays. |
| `mms.launch_mode` | `str` | `'local'` | Owner MMS launch. `local` uses one process, `per_owner` uses one process per owner on one host, and `scheduler` delegates task placement. |
| `mms.rank_rss_limit_bytes` | `int | None` | `None` | Optional positive RSS limit per process. Requires a ranked mode. |
| `mms.run_timeout_seconds` | `float | None` | `None` | Optional positive, finite wall-clock timeout. Requires a ranked mode. |
| `mms.rank_command` | `tuple[str, ...] | None` | `None` | Required in scheduler mode. Supply a TOML list of nonempty strings. The backend appends the rank entry and invokes the command once. |
| `mms.rank_index_env` | `str | None` | `None` | Scheduler-only environment variable naming the zero-based owner index. When omitted, the launcher uses `SLURM_PROCID`. |
| `mms.collectives_interface` | `str | None` | `None` | Scheduler-only CPU collectives interface. Omit it to use the collectives library's default. |
| `mms.segment_wall_seconds` | `float | None` | `None` | Wall-time budget for one local single-owner invocation, with time reserved for another step. |
| `mms.segment_steps` | `int | None` | `None` | Accepted-step budget for one local single-owner invocation. |
| `mms.checkpoint_interval_seconds` | `float | None` | `None` | Checkpoint interval for local single-owner runs. The effective default is 600 seconds. |
| `mms.face_record_chunk` | `int | None` | `None` | Positive number of face records per scan step in the manufactured advance. When omitted, use the value stored in the built operators. |
| `mms.device` | `str` | `'cpu'` | Advance on CPU or select `gpu` for the first visible GPU, with local launch and one owner. Supply a GPU-enabled JAX installation. |

With scheduler launch, run the driver outside a scheduler step on the host where
task 0 will run. The command starts exactly one task per owner, forwards the
appended argv to every task, and waits for all tasks. The case path and Python
environment must be available on every host. See the
[nodal backend guide](fci_nodal.md) for launch evidence.

**Outputs** in the case driver artifacts.

| Array | Meaning |
| --- | --- |
| `U` | Final nodal state in `nodal_state.npy`. |
| `rhs`, `rhs_manufactured` | Optional static RHS arrays in `static_rhs.npz`. |

`results` keys: `time`, `steps`, `dt`, `positivity_bound`, `field_profile` in `result.json`.
