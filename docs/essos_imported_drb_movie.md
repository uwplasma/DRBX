# ESSOS Imported QA-Coil DRB Movie

This page documents the first movie-grade non-axisymmetric DRB transient run
on externally traced Landreman-Paul QA coil field-line maps. The magnetic
field and field-line integration are supplied by ESSOS, then `jax_drb`
imports trajectories seeded on a scaled VMEC QA flux-surface shell as
fixed-layout FCI maps and advances the JAX-native `FciDrbState` with sheath
losses, target recycling, neutral
reaction/diffusion, charge exchange, metric-weighted vorticity diffusion, a
compact potential solve, and reduced nonlinear/interchange forcing used only
for this visualization gate.

This is intentionally labeled as a reduced QA-coil transient movie, not yet as
a promoted long-time turbulence validation. Promotion to a headline turbulence
claim still requires longer nonlinear runs, grid refinement, and comparison
against an external 3D edge/SOL reference. The value of this gate is that it
connects the imported coil-field geometry to the same fixed-layout JAXDRB state
used by the differentiability and PyTree RHS validation path, while enforcing
physics checks on the rendered artifact.

The movie geometry uses the VMEC Fourier surface from
`wout_LandremanPaul2021_QA_reactorScale_lowres.nc`, scaled and translated onto
the ESSOS coil-field axis. The current default movie is no longer just a
renderer-side interpolation of a coarse transient. It advances a heavier
near-boundary physics grid (`7 x 14 x 40`, `rho = 0.20 ... 0.92`) and then
renders the resulting field on a smooth VMEC surface for visual continuity.
The transient also seeds deterministic multi-mode perturbations so the visible
structure is supported by the evolved state rather than by post-processing
noise. The camera is fixed across frames, the GIF is quantized with a shared
no-dither palette, and the report includes a frame-by-frame audit of bounding
box and RMS changes.

The companion [field-line/VMEC registration gate](essos_vmec_fieldline_surface.md)
shows that the imported coil field traces finite Poincare points but does not
remain on a single scaled VMEC seed surface over the long trace. The movie
therefore treats the VMEC Fourier surface as the geometric boundary and the
imported coil traces as the FCI map source, without claiming exact
closed-surface confinement for the coil field.

Regenerate the campaign with:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py
```

## Current Gate

The current public report passes the following checks:

- endpoint fraction on the imported FCI maps: about `0.78`;
- magnetic-field modulation from the coil field: about `1.42`;
- non-axisymmetric major-radius RMS of the scaled VMEC QA surface: about
  `0.116`;
- ion-density fluctuation RMS grows from about `2.6e-2` to `7.9e-2`;
- compact potential residual: about `6.3e-2`;
- target recycling and zero-current sheath residuals are at roundoff;
- neutral particle and momentum residuals are at roundoff;
- the final toroidal/poloidal spectrum has nontrivial mode content, including
  a high toroidal index selected by the seeded near-boundary transient;
- the GIF audit passes with a fixed camera, `24` frames, and only two
  detected frame bounding boxes;
- the radial-flux proxy remains finite and nonzero during the transient.

![ESSOS imported QA-coil DRB diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_diagnostics.png)

![ESSOS imported QA-coil DRB snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_snapshots.png)

![ESSOS imported QA-coil DRB poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_poster.png)

![ESSOS imported QA-coil DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__movies__essos_imported_drb_movie_campaign.gif)

## Artifact Files

- `docs/data/essos_imported_drb_movie_artifacts/data/essos_imported_drb_movie_campaign.json`
- `docs/data/essos_imported_drb_movie_artifacts/data/essos_imported_drb_movie_campaign.npz`
- `docs/data/essos_imported_drb_movie_artifacts/images/essos_imported_drb_movie_campaign_diagnostics.png`
- `docs/data/essos_imported_drb_movie_artifacts/images/essos_imported_drb_movie_campaign_snapshots.png`
- `docs/data/essos_imported_drb_movie_artifacts/images/essos_imported_drb_movie_campaign_poster.png`
- `docs/data/essos_imported_drb_movie_artifacts/movies/essos_imported_drb_movie_campaign.gif`
