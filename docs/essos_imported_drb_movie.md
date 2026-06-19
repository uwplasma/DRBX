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

To run a lightweight grid/time movie-promotion sweep before rendering any
large artifacts, use the report-only refinement campaign:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_refinement_campaign.py
```

This campaign executes the reduced imported-field transient at the configured
`GRID_SHAPES` and `TIME_DT_VALUES`, writes one JSON report per run, and then
builds the same summary described below. It intentionally does not write GIF,
PNG, or NPZ files. Increase `GRID_SHAPES`, `TIMES_TO_TRACE`, `FRAMES`, and
`SUBSTEPS_PER_FRAME` only after the compact report-only path proves the workflow
on your machine. The checked-in compact campaign summary is:

- `docs/data/essos_imported_drb_movie_refinement_campaign_artifacts/data/essos_imported_drb_movie_refinement_campaign_summary.json`

It is intentionally negative evidence: the compact default is useful for
workflow QA, but fails publication promotion because the grid-refinement metrics
and spectral-resolution gates are not yet stable.

After regenerating two or more same-map-source movie reports at different grid
sizes and two or more reports at different effective frame timesteps, summarize
the refinement evidence without committing heavyweight media:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_refinement_summary.py
```

Edit `GRID_REPORT_JSON_PATHS` and `TIME_REPORT_JSON_PATHS` in that script to
point at the regenerated report JSON files. The summary compares
`final_fluctuation_rms`, `max_fluctuation_rms`, `radial_flux_abs_mean`,
`radial_flux_rms`, `low_mode_spectral_power_fraction`, and
`final_potential_residual_l2`, plus spectral-centroid mode-index and edge-band
metrics. It checks that the grid or timestep ordering is meaningful,
requires consistent map source labels, and rejects under-resolved reports whose
low-mode window covers the available grid or whose edge-band spectral power is
too large. The default edge-band ceiling is `0.85`, so compact exploratory
movies can remain useful visual QA while still failing publication promotion
when the spectrum is crowded near the grid edge. Relative changes are computed
with metric-specific denominator floors; for example,
`final_potential_residual_l2` uses a `1e-10` floor so roundoff-level changes in
an already-converged elliptic solve do not dominate the movie-refinement
decision. A residual-only failure is not interpreted as turbulence-grid
evidence by itself. In that case the report recommends rerunning the same
grid/time pair with a larger `POTENTIAL_ITERATIONS` budget, or inspecting the
metric-weighted CG conditioning, before spending wall time on a larger movie
grid. The report-only campaign exposes `POTENTIAL_ITERATIONS` and
`POTENTIAL_REGULARIZATION`, and each movie report records the values as
`potential_iterations` and `potential_regularization` so solver-budget changes
are auditable. The signed
`radial_flux_proxy` remains in each report as a cancellation and symmetry
diagnostic, but refinement promotion uses magnitude and RMS radial-flux
statistics because a domain-averaged signed flux can change sign when inward
and outward turbulent transport nearly cancel. The refinement summary also
exports `failed_metric_reports`, `dominant_failed_metrics`, and
`refinement_recommendations`, so a failed campaign identifies whether the next
run should prioritize radial transport convergence, toroidal/poloidal spectral
placement, spectral edge-band occupancy, or elliptic residual conditioning. The
same JSON includes `next_campaign_suggestion`, a deterministic planning aid
that proposes the next `GRID_SHAPES` and effective frame timestep values from
the dominant blockers. Treat that suggestion as a campaign input only: it is
not validation evidence until the regenerated summary passes the grid, time,
and spectral-resolution gates. The
checked-in compact campaign intentionally remains negative evidence: it has
enough grid and timestep reports to exercise the gate, but it is not
publication-ready until the grid-refinement metrics and spectral-resolution
diagnostics are stable.

Each movie report also records spectral resolution diagnostics:
`spectral_poloidal_mode_count`, `spectral_toroidal_mode_count`,
`spectral_centroid_poloidal_index`, `spectral_centroid_toroidal_index`,
`spectral_centroid_poloidal_fraction`,
`spectral_centroid_toroidal_fraction`,
`spectral_edge_band_power_fraction`, and `low_mode_window_covers_grid`. These
fields make coarse-grid failures easier to interpret. In particular, the
low-mode fraction is not a resolution claim when the low-mode window covers the
entire available grid; a publication-grade run must show stable scalar
fluctuation statistics together with stable spectral-centroid mode indices and
resolved spectral content away from the Nyquist/edge band. The normalized
centroid fractions are reserved for underresolution screening because they
change when the Nyquist range changes, even if the physical Fourier-mode
centroid is unchanged.

Before using restored release assets as fresh publication evidence, run the
clean-clone schema audit:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_artifact_schema_audit.py
```

The audit only reads committed JSON reports. It does not rerun the external
geometry import or the transient, but it catches reports that predate the
current map-source, GIF-audit, and validation metadata. As of the June 18,
2026 regeneration pass, the coil and hybrid movie JSON reports both match the
current schema. The report separates `passed` from `publication_ready`:
`passed=true` means the reduced transient, closures, and GIF artifact satisfy
the current movie QA checks, while `publication_ready=false` remains in force
until the associated connection-length summary is promotion-ready and the movie
itself passes grid-refinement, time-refinement, and long-time statistical
gates.

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

The preferred current `hybrid` showcase uses VMEC-coordinate map locations with
coil-derived endpoint masks. It passes the same closure checks while reducing
the compact potential residual to about `1.0e-2` on a `7 x 24 x 64` grid. The
report records endpoint fraction about `0.74`, magnetic-field modulation about
`1.43`, ion-density fluctuation RMS growth from about `2.6e-2` to `6.0e-2`,
particle recycling residual about `1.1e-15`, neutral particle residual about
`1.2e-18`, and a fixed-camera GIF audit with `24` frames. Its
`movie_evidence_role` is
`movie_showcase_connection_control_pending_grid_time_refinement`, reflecting
that the live hybrid connection-length control has passed but the turbulence
movie still needs grid/time refinement before being used as publication
evidence.

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
- `docs/data/essos_imported_drb_movie_refinement_artifacts/data/essos_imported_drb_movie_refinement_summary.json`
