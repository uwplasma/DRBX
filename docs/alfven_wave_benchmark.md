# Alfven-Wave Benchmark

This page records the first electromagnetic transient benchmark on the active validation ladder.

The committed benchmark rung is [alfven_wave_short_window.npz](/Users/rogerio/local/jax_drb/references/baselines/reference_arrays/alfven_wave_short_window.npz), generated from the finite-electron-mass slab case in `/Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp` with `nout=20`. That is the smallest stored history that gives a stable frequency estimate from the saved outputs while staying comfortably below the repository artifact size cap.

## Benchmark Diagnostics

![Alfven-wave short-window diagnostics](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_short_window_diagnostics.png)

Locked benchmark values from [alfven_wave_short_window_analysis.json](/Users/rogerio/local/jax_drb/docs/data/alfven_wave_short_window_analysis.json):

- analytic phase speed: `9.48585409e+05 m/s`
- measured phase speed: `9.42218662e+05 m/s`
- analytic angular frequency: `2.98006895e+06 rad/s`
- measured angular frequency: `2.96006723e+06 rad/s`
- relative phase-speed error: `6.71183264e-03`

## Parity Report

![Alfven-wave short-window parity](/Users/rogerio/local/jax_drb/docs/images/alfven_wave_short_window_parity.png)

Locked parity values from [alfven_wave_short_window_parity.json](/Users/rogerio/local/jax_drb/docs/data/alfven_wave_short_window_parity.json):

- phase-speed error: `0`
- angular-frequency error: `0`
- mean-square max absolute error: `0`
- mean-square RMS error: `0`

That parity is exact on the current partially native electromagnetic scaffold. The scaffold is not yet a full native EM transient implementation, but it already includes native `Ajpar`, native `Apar`, the physical-domain `NVe` one-step reconstruction, the physical/inner-radial `ddt(NVe)` core, and the shoulder `ddt(Vort)` planes used by the benchmark.

## Regeneration

Reference analysis:

```bash
PYTHONPATH=src .venv/bin/python -m jax_drb analyze-alfven-wave \
  /Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp \
  references/baselines/reference_arrays/alfven_wave_short_window.npz \
  --json-out docs/data/alfven_wave_short_window_analysis.json \
  --plot-out docs/images/alfven_wave_short_window_diagnostics.png
```

Native/reference parity report:

```bash
PYTHONPATH=src .venv/bin/python -m jax_drb compare-alfven-wave \
  /Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp \
  references/baselines/reference_arrays/alfven_wave_short_window.npz \
  /tmp/jax_drb_alfven_wave_short_window_native.npz \
  --json-out docs/data/alfven_wave_short_window_parity.json \
  --plot-out docs/images/alfven_wave_short_window_parity.png
```
