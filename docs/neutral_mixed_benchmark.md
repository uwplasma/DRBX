# Neutral Mixed Benchmark

This page tracks the compact short-window benchmark for the staged neutral branch. The original short-window artifact remains a reference-side target, while the current one-step and substep/hybrid diagnostics provide bounded native parity evidence for the term-level implementation.

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
  --plot-out https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__neutral_mixed_short_window_diagnostics.png
```

## Diagnostic Figure

![Neutral mixed short-window diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__images__neutral_mixed_short_window_diagnostics.png)

The current locked target values at the final stored output are:

- center `Nh`: `7.86199787e-01`
- center `Ph`: `7.86185965e-02`
- center `NVh`: `-7.08005656e-08`
- center temperature: `9.99982418e-02`
- total `Nh`: `7.86197875e+02`
- total `Ph`: `7.86184063e+01`
- momentum RMS: `5.56121767e-08`

This benchmark keeps the neutral transient implementation honest across two levels: the short-window reference history protects the expected observable scale, and the one-step/substep diagnostics localize native parity to the pressure-gradient, viscosity, boundary, and state-history pieces that feed `NVh`. The public runner now executes both `neutral_mixed_one_step` and `neutral_mixed_short_window`, and the latest bounded one-step compare is materially tighter after fixing connected-y guard reconstruction and promoting the one-step default to eight internal BDF substeps:

- center `Nh` max-abs error: about `5.39e-5`
- center `Ph` max-abs error: about `5.90e-6`
- center `NVh` max-abs error: about `4.24e-6`
- center temperature max-abs error: about `6.38e-7`
- momentum RMS max-abs error: about `7.74e-10`

That is good enough to promote the one-step gate from a loose operational probe to a much tighter bounded native check. The same native runner path still keeps the short-window default at four internal substeps and finishes a bounded full short-window metric gate on the matrix-free path inside the ten-minute validation policy, with roughly:

- `center Nh ≈ 8.03e-3`
- `center Ph ≈ 6.47e-4`
- `center NVh ≈ 8.60e-4`
- `center T ≈ 2.91e-4`
- `total Nh ≈ 3.24e-1`
- `total Ph ≈ 2.89e-2`
- `momentum RMS ≈ 1.71e-3`

That keeps the heavier transient under bounded native regression on both centerline and total-history metrics, and the same trimmed active-domain `Nh`/`Ph`/`NVh` surface now also clears a bounded full-array short-window field gate (`Nh max|Δ| ≈ 1.27e-2`, `Ph max|Δ| ≈ 1.21e-3`, `NVh max|Δ| ≈ 3.37e-3`). The remaining neutral work is therefore no longer about whether a short-window field surface exists, but only whether a broader standalone claim needs wider field coverage than this committed trimmed-domain gate.
