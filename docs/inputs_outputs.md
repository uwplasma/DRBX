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

### Initial mixmode keys

`[initial]` supports two mixmode paths:

- `n_profile = "gaussian_mixmode"` with `mixmode_amp`, `mixmode_terms`, `mixmode_mode`
  to build a deterministic density perturbation from the selected profile.
- `n_mixmode_amp`, `n_mixmode_terms`, `n_mixmode_mode`, `n_mixmode_seed` to overlay
  deterministic mixmode perturbations on top of any base density profile (for example,
  a linear equilibrium profile plus a Hermes-style `x-z` perturbation).

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
