# Drift-Wave Benchmark

This benchmark page turns the committed short-window drift-wave baseline into a reproducible validation artifact. The inputs remain small on purpose: one active radial cell, periodic parallel and binormal directions, fixed ion/electron temperatures, quasineutral density closure, and electrostatic vorticity coupling.

The diagnostic figure below is generated directly from the committed array baseline and the same benchmark-analysis code exercised by the regression suite.

![Drift-wave benchmark diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_diagnostics.png)

The parity figure below compares the current native short-window output against the committed reference baseline, using the same portable array payload schema as the automated regressions.

![Drift-wave short-window parity](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_parity.png)

## Locked Scalars

- `sigma_parallel / omega_* = 1.0542443560`
- measured `gamma / omega_* = 0.2747889979`
- analytic `gamma / omega_* = 0.2861001064`
- measured `omega / omega_* = 0.2322431514`
- analytic `omega / omega_* = 0.2286359364`

The measured values are within a few percent of the analytic finite-electron-mass dispersion root, which is the right standard for this short-window benchmark: the transient stays in the linear regime, but it still exercises the full discrete grid, guard handling, normalization, and electrostatic closure path.

## Current Native Parity Envelope

The current native transient is benchmark-close and field-close, but not yet under a single strict all-field tolerance. The committed parity report records:

- reference `gamma / omega_* = 0.2747889979`, native `= 0.2754510776`
- reference `omega / omega_* = 0.2322431514`, native `= 0.2292267529`
- max `|Ni|` and `|Ne|` error `= 1.4721e-3`
- max `|NVe|` error `= 1.7009e-4`
- max `|Vort|` error `= 2.1405e-2`
- max `|phi|` error `= 4.3146e-4`

This is the correct summary status for the current milestone: the reduced short-window branch is stable, benchmark-consistent, and documented quantitatively while the remaining long-window field drift is tightened.

## Benchmark Definitions

The analysis module computes

- `omega_* = k_z T_e (1 / L_n) / B`
- `sigma_parallel = (k_y / k_z)^2 Omega_ci Omega_ce / (0.51 nu_ei)`

and then solves the finite-electron-mass cubic for the fastest-growing mode,

```text
(omega_* / (0.51 nu_ei)) x^3 + i x^2 - (sigma_parallel / omega_*) x + sigma_parallel / omega_* = 0
```

with `x = omega / omega_* + i gamma / omega_*`.

The measured growth rate comes from the tail slope of `log(n_rms)`, and the measured frequency comes from a tracked density-peak phase speed along the periodic binormal direction.

## Reproduction

The committed JSON and figure were generated with:

```bash
PYTHONPATH=src python -m jax_drb analyze-drift-wave \
  /path/to/curated/drift_wave/BOUT.inp \
  references/baselines/reference_arrays/drift_wave_short_window.npz \
  --json-out docs/data/drift_wave_short_window_analysis.json \
  --plot-out https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_diagnostics.png
```

Artifacts:

- analysis JSON: [docs/data/drift_wave_short_window_analysis.json](docs/data/drift_wave_short_window_analysis.json)
- diagnostic figure: [https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_diagnostics.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_diagnostics.png)
- parity JSON: [docs/data/drift_wave_short_window_parity.json](docs/data/drift_wave_short_window_parity.json)
- parity figure: [https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_parity.png](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_parity.png)

The CLI command is source-neutral: it operates on any compatible input file plus any portable drift-wave array payload with `Ni` time history and normalization metadata.

The native/reference parity report was generated with:

```bash
PYTHONPATH=src python -m jax_drb compare-drift-wave \
  /path/to/curated/drift_wave/BOUT.inp \
  references/baselines/reference_arrays/drift_wave_short_window.npz \
  /tmp/jax_drb_drift_wave_short_window_native.npz \
  --json-out docs/data/drift_wave_short_window_parity.json \
  --plot-out https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__drift_wave_short_window_parity.png
```
