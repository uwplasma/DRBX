# Figures & Diagnostics

This page showcases **representative nonlinear DRB results** and highlights
diagnostic outputs available in `jax_drb`.

The plotting scripts in `tools/` call internal diagnostics utilities under
`jaxdrb.diagnostics` (spectra, PDFs, and zonal averages) so all figures remain
fully reproducible without external code.

## Nonlinear Snapshot Panel

![Nonlinear DRB panel](figures/nonlinear_panel.png)

The panel shows mid‑plane snapshots of key fields from a nonlinear plane run with
tokamak‑style curvature drive: `n`, `phi`, `omega`, and `Te`. By default we plot
**fluctuations** (zonal‑mean subtracted for `n`/`Te`, global‑mean subtracted for
`phi`/`omega`) to highlight nonlinear structure.

Regenerate it with:

```bash
python examples/plane_nonlinear/run.py --make-figures --make-movies
```

## RMS Time Series

![Nonlinear RMS time series](figures/nonlinear_rms_timeseries.png)

The RMS traces highlight transient growth and saturation behavior. Use these to
validate stability windows, time‑stepping, and dissipation choices. The same
example command above regenerates them.

## Zonal Profiles

![Zonal profiles](figures/nonlinear_zonal_profile.png)

Zonal averages highlight self‑organized flow structure and large‑scale shear.

## Zonal Flow

![Zonal flow](figures/nonlinear_zonal_flow.png)

Time‑averaged zonal flow (`v_{E,y}`) computed from the zonal mean of `phi`.

## Spectra

![Isotropic spectra](figures/nonlinear_spectrum.png)

The isotropic spectra are computed using internal `jax_drb` diagnostics.

## PDFs

![PDFs](figures/nonlinear_pdfs.png)

PDFs are computed from fluctuation fields (mean‑subtracted) to characterize
intermittency.

## Blob Movie

![Blob movie](figures/blob_movie.gif)

This short GIF is generated from the saved snapshots in the public example.

## Open Field‑Line Example

![Open field-line poloidal equilibrium](figures/open_field_poloidal_eq.png)

Poloidal visualization (circular cross‑section) of the open/closed SOL mask and
equilibrium profile used in the open‑field‑line example.

![Open field-line poloidal fluctuation](figures/open_field_poloidal_fluct.png)

Fluctuation snapshot overlaid on the equilibrium profile to highlight open vs
closed‑field structure.

![Open field-line movie](figures/open_field_movie.gif)

## Field‑Aligned 3D Example

![3D slices](figures/three_d_slices.png)

3D slice views (xy/xz/yz) from a field‑aligned s‑alpha example.

![3D movie](figures/three_d_movie.gif)

![3D RMS](figures/three_d_rms_timeseries.png)
