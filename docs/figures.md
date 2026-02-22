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

## Energy Conservation

![Energy error](figures/energy_error.png)

Relative energy error for an advection‑only conservation check (`examples/conservation_check/`).

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

### Poloidal + Toroidal Conventions

Tokamak diagnostics are commonly shown in the **poloidal \((R,Z)\) plane**, where
a vertical slice through the torus exposes the magnetic cross‑section and flux
surfaces. The toroidal angle \(\phi\) is the azimuthal coordinate around the
symmetry axis. We follow this convention by:

- Rendering poloidal cuts in \((R,Z)\), and
- Plotting toroidal cuts as annuli in \((R,\phi)\) so the “donut hole” remains
  visible.

The last closed flux surface (LCFS / separatrix) is overlaid as a **dashed
circle**, consistent with common presentation in edge‑turbulence literature. See
the coordinate definitions and poloidal cross‑section convention in
[ASCOT5’s coordinate notes](https://ascot4fusion.github.io/ascot5/main/theory/coordinates.html),
and example separatrix overlays in tokamak edge turbulence figures (e.g.
[Pan et al. 2018, Entropy](https://www.mdpi.com/1099-4300/20/4/227)).

## Field‑Aligned 3D Example

![3D toroidal cuts](figures/three_d_toroidal.png)

Poloidal + toroidal cuts (mapped onto a circular cross‑section for the poloidal
plane, and **polar annuli** for the toroidal cuts) from a field‑aligned s‑alpha
example. We show **outboard** and **inboard** toroidal cuts side‑by‑side to
highlight ballooning‑like asymmetry.

![3D movie](figures/three_d_toroidal_movie.gif)

![3D RMS](figures/three_d_rms_timeseries.png)
