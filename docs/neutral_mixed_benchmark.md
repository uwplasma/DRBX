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

This benchmark exists to keep the next neutral transient implementation honest. The public runner now executes both `neutral_mixed_one_step` and `neutral_mixed_short_window`, and the latest bounded one-step compare is materially tighter after fixing the neutral density wall-guard reconstruction to match the reference wall extrapolation:

- center `Nh` max-abs error: about `8.03e-3`
- center `Ph` max-abs error: about `5.66e-4`
- center `NVh` max-abs error: about `8.60e-4`
- center temperature max-abs error: about `2.91e-4`
- momentum RMS max-abs error: about `1.71e-3`

That is good enough to lock an operational one-step gate. The same native runner path now also finishes a bounded full short-window metric gate on the matrix-free path inside the ten-minute validation policy, with roughly:

- `center Nh ≈ 8.03e-3`
- `center Ph ≈ 6.47e-4`
- `center NVh ≈ 8.60e-4`
- `center T ≈ 2.91e-4`
- `total Nh ≈ 3.24e-1`
- `total Ph ≈ 2.89e-2`
- `momentum RMS ≈ 1.71e-3`

That keeps the heavier transient under bounded native regression on both centerline and total-history metrics. The remaining neutral hardening task is now the broader full-array short-window field surface, not the existence of any global short-window metric gate at all.
