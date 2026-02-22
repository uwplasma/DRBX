# Figures & Diagnostics

This page showcases **representative nonlinear DRB results** and highlights
diagnostic outputs available in `jax_drb`.

## Nonlinear Snapshot Panel

![Nonlinear DRB panel](figures/nonlinear_panel.png)

The panel shows mid‑plane snapshots of key fields from a nonlinear s‑alpha run:
`n`, `phi`, `omega`, and `Te`. The image can be regenerated with
`tools/plot_nonlinear_panel.py`.

## RMS Time Series

![Nonlinear RMS time series](figures/nonlinear_rms_timeseries.png)

The RMS traces highlight transient growth and saturation behavior. Use these to
validate stability windows, time‑stepping, and dissipation choices. Regenerate
with `tools/plot_rms_timeseries.py`.
