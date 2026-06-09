# ESSOS Imported QA-Coil DRB Movie

This page documents the first movie-grade non-axisymmetric DRB transient run
on imported Landreman-Paul QA FCI maps. The geometry adapter now supports
three map sources: coil-traced maps, VMEC-coordinate maps, and a hybrid map
that uses VMEC-coordinate interpolation with coil endpoint masks. `jax_drb`
advances the JAX-native `FciDrbState` on those fixed-layout maps with sheath
losses where endpoints are present, target recycling, neutral
reaction/diffusion, charge exchange, metric-weighted vorticity diffusion, a
compact potential solve, and reduced nonlinear/interchange forcing used only
for this visualization gate. The movie advection wrapper now calls the same
tested logical \(E\times B\) bracket helper used by the pedagogical
stellarator vorticity example, but the normalization and forcing amplitudes
remain movie-gate controls rather than calibrated transport coefficients.

This is intentionally labeled as a reduced QA-coil transient movie, not yet as
a promoted long-time turbulence validation. Promotion to a headline turbulence
claim still requires longer nonlinear runs, grid refinement, and comparison
against an external 3D edge/SOL reference. The value of this gate is that it
connects imported non-axisymmetric geometry to the same fixed-layout JAXDRB
state used by the differentiability and PyTree RHS validation path, while
enforcing physics checks on the rendered artifact.

The physics grid is seeded from the VMEC Fourier surface in
`wout_LandremanPaul2021_QA_reactorScale_lowres.nc`, scaled and translated onto
the ESSOS coil-field axis before tracing the FCI maps. The rendered boundary is
the raw VMEC Fourier surface, so the movie shape follows the same convention as
`vmec_jax --plot` while the reduced state remains indexed on the traced
fixed-layout physics grid. The current default movie advances a heavier
near-boundary physics grid (`8 x 28 x 80`, `rho = 0.20 ... 0.92`) rather than
relying on a renderer-only high-resolution interpolation of a coarse transient.
The transient also seeds deterministic multi-mode perturbations so the visible
structure is supported by the evolved state rather than by post-processing
noise. Set `MAP_SOURCE = "coil"`, `"vmec"`, or `"hybrid"` near the top of
`examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py` to
switch between the open coil trace, the closed VMEC-coordinate control, and the
hybrid open-field SOL bridge. The camera is fixed across frames, the GIF is
quantized with a shared no-dither palette, and the report includes a
frame-by-frame audit of bounding box and RMS changes.

The companion [field-line/VMEC registration gate](essos_vmec_fieldline_surface.md)
shows that the imported coil field traces finite Poincare points but does not
remain on a single scaled VMEC seed surface over the long trace. The movie
therefore treats the VMEC Fourier surface as the geometric boundary and the
imported coil traces as the FCI map source, without claiming exact
closed-surface confinement for the coil field. The same page also includes the
VMEC-coordinate control trace, which verifies that the rendered QA boundary is
surface-preserving for the VMEC equilibrium field itself.

The published figures and movie are restored by
`python scripts/fetch_example_artifacts.py --skip-baselines`. Regenerating the
default coil-traced campaign from the external coil geometry is a developer
workflow and requires the geometry source checkout:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py
```

For the hybrid bridge, set `MAP_SOURCE = "hybrid"` and `OUTPUT_ROOT =
Path("docs/data/essos_imported_drb_movie_hybrid_artifacts")`; for the
closed-field control, set `MAP_SOURCE = "vmec"` and use a separate output root.

## Current Gate

The current public report passes the following checks:

- endpoint fraction on the imported FCI maps: about `0.72`;
- magnetic-field modulation from the coil field: about `1.42`;
- non-axisymmetric major-radius RMS of the scaled VMEC QA surface: about
  `0.116`;
- ion-density fluctuation RMS grows from about `2.7e-2` to `8.7e-2`;
- compact potential residual: about `2.3e-1`;
- target recycling and zero-current sheath residuals are at roundoff;
- neutral particle and momentum residuals are at roundoff;
- the final toroidal/poloidal spectrum has nontrivial mode content, including
  a high toroidal index selected by the seeded near-boundary transient;
- the GIF audit passes with a fixed camera, `24` frames, and only two
  detected frame bounding boxes; the median inter-frame RMS is about `3.2` on
  an 8-bit RGB scale;
- the radial-flux proxy remains finite and nonzero during the transient.

![ESSOS imported QA-coil DRB diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_diagnostics.png)

![ESSOS imported QA-coil DRB snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_snapshots.png)

![ESSOS imported QA-coil DRB poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_poster.png)

![ESSOS imported QA-coil DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__movies__essos_imported_drb_movie_campaign.gif)

The promoted `hybrid` movie uses VMEC-coordinate map locations with
coil-derived endpoint masks. It passes the same closure checks while reducing
the compact potential residual to about `1.0e-2` on a `7 x 24 x 64` grid. The
report records endpoint fraction about `0.74`, magnetic-field modulation about
`1.43`, ion-density fluctuation RMS growth from about `2.6e-2` to `6.0e-2`,
particle recycling residual about `1.1e-15`, neutral particle residual about
`1.2e-18`, and a fixed-camera GIF audit with `24` frames.

![ESSOS imported QA-hybrid DRB diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_hybrid_artifacts__images__essos_imported_drb_movie_hybrid_campaign_diagnostics.png)

![ESSOS imported QA-hybrid DRB snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_hybrid_artifacts__images__essos_imported_drb_movie_hybrid_campaign_snapshots.png)

![ESSOS imported QA-hybrid DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_hybrid_artifacts__movies__essos_imported_drb_movie_hybrid_campaign.gif)

## Artifact Files

- `docs/data/essos_imported_drb_movie_artifacts/data/essos_imported_drb_movie_campaign.json`
- `docs/data/essos_imported_drb_movie_artifacts/data/essos_imported_drb_movie_campaign.npz`
- `docs/data/essos_imported_drb_movie_artifacts/images/essos_imported_drb_movie_campaign_diagnostics.png`
- `docs/data/essos_imported_drb_movie_artifacts/images/essos_imported_drb_movie_campaign_snapshots.png`
- `docs/data/essos_imported_drb_movie_artifacts/images/essos_imported_drb_movie_campaign_poster.png`
- `docs/data/essos_imported_drb_movie_artifacts/movies/essos_imported_drb_movie_campaign.gif`
- `docs/data/essos_imported_drb_movie_hybrid_artifacts/data/essos_imported_drb_movie_hybrid_campaign.json`
- `docs/data/essos_imported_drb_movie_hybrid_artifacts/data/essos_imported_drb_movie_hybrid_campaign.npz`
- `docs/data/essos_imported_drb_movie_hybrid_artifacts/images/essos_imported_drb_movie_hybrid_campaign_diagnostics.png`
- `docs/data/essos_imported_drb_movie_hybrid_artifacts/images/essos_imported_drb_movie_hybrid_campaign_snapshots.png`
- `docs/data/essos_imported_drb_movie_hybrid_artifacts/images/essos_imported_drb_movie_hybrid_campaign_poster.png`
- `docs/data/essos_imported_drb_movie_hybrid_artifacts/movies/essos_imported_drb_movie_hybrid_campaign.gif`
