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

## Regeneration

These figures are generated from the committed baseline arrays plus native case runs. The current gallery uses:

- `diffusion_short_window`
- `vorticity_short_window`
- `drift_wave_one_step`

The next gallery pass should add:

- periodic 1D fluid short-window figures;
- longer electrostatic drift-wave transient figures once the native transient path is locked;
- benchmark and validation plots for open-field-line cases as those stages land.
