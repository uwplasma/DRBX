# Neutral Mixed Benchmark

This page tracks the compact short-window benchmark for the staged neutral branch. The current artifact is a reference-side target, not yet a native transient parity claim.

The benchmark is extracted from the committed short-window neutral array baseline at probe index `(x=5, y=3, z=5)`. It records:

- center `Nh`, `Ph`, and `NVh` histories;
- derived center temperature `Ph / Nh`;
- total neutral density and pressure histories;
- momentum RMS decay.

These metrics are stored in [references/baselines/reference_metrics/neutral_mixed_short_window_metrics.json](references/baselines/reference_metrics/neutral_mixed_short_window_metrics.json) and generated with:

```bash
PYTHONPATH=src python -m jax_drb analyze-neutral-mixed \
  references/baselines/reference_arrays/neutral_mixed_short_window.npz \
  --x-index 5 \
  --y-index 3 \
  --z-index 5 \
  --json-out references/baselines/reference_metrics/neutral_mixed_short_window_metrics.json \
  --plot-out docs/images/neutral_mixed_short_window_diagnostics.png
```

## Diagnostic Figure

![Neutral mixed short-window diagnostics](images/neutral_mixed_short_window_diagnostics.png)

The current locked target values at the final stored output are:

- center `Nh`: `7.86199787e-01`
- center `Ph`: `7.86185965e-02`
- center `NVh`: `-7.08005656e-08`
- center temperature: `9.99982418e-02`
- total `Nh`: `7.86197875e+02`
- total `Ph`: `7.86184063e+01`
- momentum RMS: `5.56121767e-08`

This benchmark exists to keep the next neutral transient implementation honest: the public runner should not expose `neutral_mixed_one_step` or `neutral_mixed_short_window` until it reproduces these compact histories closely enough for review material and regression tests.
