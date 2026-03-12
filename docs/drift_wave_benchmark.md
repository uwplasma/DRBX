# Drift-Wave Benchmark

This benchmark page turns the committed short-window drift-wave baseline into a reproducible validation artifact. The inputs remain small on purpose: one active radial cell, periodic parallel and binormal directions, fixed ion/electron temperatures, quasineutral density closure, and electrostatic vorticity coupling.

The diagnostic figure below is generated directly from the committed array baseline and the same benchmark-analysis code exercised by the regression suite.

![Drift-wave benchmark diagnostics](images/drift_wave_short_window_diagnostics.png)

## Locked Scalars

- `sigma_parallel / omega_* = 1.0542443560`
- measured `gamma / omega_* = 0.2747889979`
- analytic `gamma / omega_* = 0.2861001064`
- measured `omega / omega_* = 0.2322431514`
- analytic `omega / omega_* = 0.2286359364`

The measured values are within a few percent of the analytic finite-electron-mass dispersion root, which is the right standard for this short-window benchmark: the transient stays in the linear regime, but it still exercises the full discrete grid, guard handling, normalization, and electrostatic closure path.

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
  --plot-out docs/images/drift_wave_short_window_diagnostics.png
```

Artifacts:

- analysis JSON: [docs/data/drift_wave_short_window_analysis.json](/Users/rogerio/local/jax_drb/docs/data/drift_wave_short_window_analysis.json)
- diagnostic figure: [docs/images/drift_wave_short_window_diagnostics.png](/Users/rogerio/local/jax_drb/docs/images/drift_wave_short_window_diagnostics.png)

The CLI command is source-neutral: it operates on any compatible input file plus any portable drift-wave array payload with `Ni` time history and normalization metadata.
