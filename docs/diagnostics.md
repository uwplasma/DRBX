# Diagnostics

`jax_drb` now includes a shared benchmark diagnostics layer used for both
`jax_drb` outputs and Hermes dump outputs.

Code paths:
- `src/jaxdrb/benchmarking/schema.py`
- `src/jaxdrb/benchmarking/diagnostics.py`
- `tools/build_benchmark_bundle.py`
- `tools/plot_benchmark_panel.py`

## Shared Schema (Normalized + SI)

A benchmark bundle stores:
- normalized time (`times_norm`)
- SI time (`times_si`)
- normalization constants (`Nnorm`, `Tnorm_eV`, `Bnorm_T`, `m_i_amu`, `Z_i`)
- diagnostics channels
- last snapshots / fluctuation snapshots

Normalization-derived reference scales:

\[
 c_{s0} = \sqrt{\frac{e T_{e0}}{m_i}}, \qquad
 \Omega_{ci} = \frac{Z_i e B_0}{m_i}, \qquad
 \rho_{s0} = \frac{c_{s0}}{\Omega_{ci}}.
\]

These are encoded in `BenchmarkNormalization` for explicit parity checks.

## Implemented Shared Diagnostics

All diagnostics are available from `src/jaxdrb/benchmarking/diagnostics.py`.

### Fluctuation RMS

For equilibrium field `f_eq` (default first snapshot):

\[
 f' = f - f_{eq}, \qquad
 \mathrm{RMS}(f') = \sqrt{\langle (f')^2 \rangle}.
\]

APIs:
- `compute_fluctuation_rms`

### Spectra

- Frequency spectrum from Welch-like windowed FFT of probe traces:
  `compute_frequency_psd`
- Binormal spectrum from FFT along the chosen `y` axis:
  `compute_ky_psd`

### PDFs

- Histogram-based PDFs of fluctuation fields:
  `compute_pdf`

### Cross-Coherence and Phase

From cross-spectrum `S_xy(f)`:

\[
 C_{xy}(f) = \frac{|S_{xy}|^2}{S_{xx}S_{yy}}, \qquad
 \phi_{xy}(f) = \arg(S_{xy}).
\]

API:
- `compute_cross_coherence_phase`

### Radial Flux and Target Fluxes

- Particle flux profile using
  \(\Gamma_r = \langle n v_{E,r}\rangle\),
  \(v_{E,r} = -\partial_y\phi / B\):
  `compute_radial_particle_flux_profile`
- Target particle/heat flux proxies:
  `compute_target_fluxes`

### Finite-Run Gate

Shared finite-run gate used by scanning/staging workflows:
- finite checks for all RMS channels
- growth-factor gate
- absolute-peak gate

API:
- `finite_run_gate`

## Hermes-Compatible Workflow

1. Build Hermes bundle:

```bash
cd <repo>
PYTHONPATH=src python tools/build_benchmark_bundle.py \
  --code hermes \
  --input <hermes-data-dir> \
  --output <run-dir>/bundle_hermes_short.npz \
  --geometry tokamak_open_field
```

2. Build jax_drb bundle:

```bash
cd <repo>
PYTHONPATH=src python tools/build_benchmark_bundle.py \
  --code jax \
  --input <run-dir>/jax_short.npz \
  --config examples/open_field_line/input_tokamak_bxcv_benchmark_alignment.toml \
  --output <run-dir>/bundle_jax_short.npz \
  --geometry tokamak_open_field
```

3. Generate canonical side-by-side panel:

```bash
cd <repo>
PYTHONPATH=src python tools/plot_benchmark_panel.py \
  --hermes <run-dir>/bundle_hermes_short.npz \
  --jax <run-dir>/bundle_jax_short.npz \
  --out docs/figures/tokamak_sol_benchmark_panel.png \
  --summary-csv docs/figures/tokamak_sol_benchmark_panel.csv
```

## Literature Anchors

- Hermes numerics and BC docs:
  `external/hermes-3/docs/sphinx/solver_numerics.rst`,
  `external/hermes-3/docs/sphinx/boundary_conditions.rst`,
  `external/hermes-3/docs/sphinx/equations.rst`
- SOL turbulence and diagnostics references:
  `2303.12131v2.pdf`,
  `Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf`,
  `Stegmeir_2018_Plasma_Phys._Control._Fusion_60_035005.pdf`
