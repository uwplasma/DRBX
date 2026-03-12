# Validation Gallery

This page collects the first curated parity figures from the active validation ladder. Each figure is generated from the same committed baseline artifacts used by the regression harness, so the visuals and the automated checks stay in sync.

## Diffusion Short Window

![Diffusion short-window parity](images/diffusion_short_window_parity.png)

What this locks down:

- structured mesh reconstruction;
- metric normalization on the transport path;
- Neumann guard handling;
- repeated output scheduling over a short transient.

## Electrostatic Vorticity Short Window

![Vorticity short-window parity](images/vorticity_short_window_parity.png)

What this locks down:

- discrete X-Z XPPM advection;
- Boussinesq potential inversion;
- repeated electrostatic output parity for both `Vort` and `phi`.

## Coupled Drift-Wave One Step

![Drift-wave one-step parity](images/drift_wave_one_step_parity.png)

What this locks down:

- coupled density, electron momentum, vorticity, and potential output;
- quasineutral electron closure;
- fixed-temperature electron pressure;
- trimmed active-cell comparisons for the first 2D density-vorticity benchmark.

## Drift-Wave Short-Window Benchmark

![Drift-wave short-window diagnostics](images/drift_wave_short_window_diagnostics.png)

What this locks down:

- benchmark postprocessing on the committed short-window array baseline;
- measured growth-rate and frequency extraction from the periodic density history;
- analytic finite-electron-mass dispersion evaluation from the same normalization and geometry scalars used by the run;
- documentation-ready reviewer figures backed by automated regression tests.

## Regeneration

These figures are generated from the committed baseline arrays plus native case runs. The current gallery uses:

- `diffusion_short_window`
- `vorticity_short_window`
- `drift_wave_one_step`
- `drift_wave_short_window`

The next gallery pass should add:

- periodic 1D fluid short-window figures;
- native drift-wave short-window parity figures once the longer transient path is locked;
- benchmark and validation plots for open-field-line cases as those stages land.
