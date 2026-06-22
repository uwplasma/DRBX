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
For the full hybrid promotion sequence, prefer the top-level workflow ledger:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/hybrid_open_sol_demo.py
```

That script writes a self-contained dry-run contract by default. Live media
generation is deliberately the last optional stage, after the same map has
passed FCI/source-profile accounting, hybrid parallel-step refinement,
stationarity, and grid/time movie checks. A hybrid GIF should stay diagnostic
unless the workflow summary reports `promotion_ready=true` and the frames have
been visually reviewed for camera stability, non-axisymmetric geometry, radial
open-field structure, readable color scale, and physical time annotation.
The workflow exposes `STATIONARITY_PRESET = "quick"` for a bounded live smoke
test of the stationarity plumbing. A quick run can pass internally, but the
workflow records `quick_stationarity_preset_not_promotion_evidence` and keeps
`promotion_ready=false`; use `STATIONARITY_PRESET = "promotion"` plus the
grid/time and visual-QA gates below before using media as README or paper
evidence.

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

The current committed summary recommends the next heavier report-only
publication-candidate sweep rather than another renderer-only interpolation:
`GRID_SHAPES = ((4, 6, 12), (8, 12, 24))`,
`TIME_SHAPE = (8, 12, 24)`, effective frame timestep values `0.004` and
`0.002`, and `potential_iterations = 3072`. The example exposes those settings
through `build_publication_candidate_refinement_settings()`. Passing that
heavier sweep is the next required gate before using the imported-field
turbulence movie as publication evidence.

That heavier candidate has now been run and committed as:

- `docs/data/essos_imported_drb_movie_refinement_publication_artifacts/data/essos_imported_drb_movie_refinement_publication_summary.json`

It is still negative publication evidence, but it narrows the blocker. The
`8 x 12 x 24` time-refinement pair passes the scalar and spectral gates
(`max_relative_metric_change = 0.066`), while the grid-refinement pair still
fails due to radial-flux sensitivity, poloidal spectral-centroid motion, and
edge-band spectral power (`max_relative_metric_change = 0.94`). The next
publication-candidate sweep should therefore compare `8 x 12 x 24` with
`16 x 24 x 48` while reusing the same effective frame timestep pair.

The follow-on `16 x 24 x 48` candidate is committed as:

- `docs/data/essos_imported_drb_movie_refinement_16x_candidate_artifacts/data/essos_imported_drb_movie_refinement_16x_candidate_summary.json`

This second candidate closes the time-refinement gate again and removes the
radial-flux and spectral edge-band blockers. After the refinement gate was
corrected to compare normalized spectral-centroid fractions rather than raw
mode indices, the remaining failed metric is the toroidal centroid fraction.
The next targeted sweep should therefore compare `16 x 24 x 48` with
`16 x 24 x 96`, or use the later high-poloidal Jacobi evidence below when the
goal is a stronger same-machine movie gate.

That targeted poloidal sweep is committed as:

- `docs/data/essos_imported_drb_movie_refinement_poloidal_candidate_artifacts/data/essos_imported_drb_movie_refinement_poloidal_candidate_summary.json`

It is useful failure evidence rather than a promotion. The higher-poloidal
`16 x 48 x 48` report exposes an elliptic-potential residual instability
(`final_potential_residual_l2 ~ 8e22`) and nonfinite smaller-step diagnostics,
so the next technical step is not a larger movie grid. The next run should
first repeat the same `16 x 24 x 48 -> 16 x 48 x 48` grid pair with the
potential budget increased to `6144` iterations and, if needed, a stronger
potential preconditioner before any `16 x 96 x 48` escalation.

The follow-up potential-solve checks show that raw iterations alone are not the
right fix, while Jacobi preconditioning is. The unpreconditioned `6144`
iteration rerun remains blocked by `final_potential_residual_l2`, but the
Jacobi-preconditioned `3072` iteration rerun closes the potential-residual,
grid-refinement, and time-refinement blockers:

- `docs/data/essos_imported_drb_movie_refinement_poloidal_6144_artifacts/data/essos_imported_drb_movie_refinement_poloidal_6144_summary.json`
- `docs/data/essos_imported_drb_movie_refinement_poloidal_jacobi_artifacts/data/essos_imported_drb_movie_refinement_poloidal_jacobi_summary.json`
- `docs/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_artifacts/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_summary.json`

The `16 x 48 x 48 -> 16 x 96 x 48` Jacobi candidate is the strongest
checked-in report-only gate from this lane: both grid and time refinement pass
with no potential-residual blocker. It still does not by itself prove a final
publication movie, because the next evidence should be a polished long-window
movie generated with the same Jacobi potential solve and accompanied by
stationarity statistics.

The long-window stationarity gate is JSON-only and uses the same high-resolution
Jacobi solver settings without writing GIF, PNG, or NPZ media:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_stationarity_campaign.py
```

The committed stationarity report is:

- `docs/data/essos_imported_drb_movie_stationarity_jacobi_artifacts/data/essos_imported_drb_movie_stationarity_jacobi.json`

That report passes on the `16 x 96 x 48` hybrid grid with `frames = 12`,
`substeps_per_frame = 3`, `dt = 0.002`, and
`potential_preconditioner = "jacobi"`. The tail fluctuation-RMS drift is
`0.15`, the ion-density drift is `7.6e-3`, the neutral-density drift is
`3.8e-2`, the vorticity-RMS drift is `8.3e-3`, and the tail potential residual
stays below `3.4e-11`. The next media step should therefore generate a polished
GIF and diagnostics images with these settings, QA the frames visually, and
release-host the media rather than committing it to git.

That media step has been run locally with the same high-resolution Jacobi
settings. The generated bundle is kept under ignored `artifacts/` so the
repository remains lightweight, and a small tracked manifest records the file
sizes, checksums, image dimensions, frame count, and visual-QA decision:

- `docs/data/essos_imported_drb_movie_stationarity_jacobi_media_manifest.json`

The local QA pass inspected the poster, diagnostics page, FCI-plane snapshots,
and a fixed-camera GIF contact sheet. The camera is stable, the opened sector
shows the non-axisymmetric Landreman-Paul QA geometry, and the density
fluctuations evolve smoothly across the 12-frame window. The uncropped bundle
is suitable as docs/example validation media after release-hosting, but the raw
poster and GIF retain excess title whitespace. The media-run JSON itself also
remains conservative unless it is paired with the committed refinement and
stationarity reports above.

The final README/docs display assets are release-hosted compact crops of that
same QA'd media bundle. The crop is display-only: it removes excess title and
caption whitespace from the GIF/poster while preserving the fixed camera,
opened sector, colorbar, and evolved state.

![ESSOS imported QA-hybrid high-resolution diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__images__diagnostics.png)

![ESSOS imported QA-hybrid high-resolution snapshots](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__images__snapshots.png)

![ESSOS imported QA-hybrid high-resolution poster](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__images__poster_compact.png)

![ESSOS imported QA-hybrid high-resolution movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__movies__movie_compact.gif)

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
`POTENTIAL_REGULARIZATION`, plus the opt-in `POTENTIAL_PRECONDITIONER` setting
for the metric-weighted CG inversion. Each movie report records the values as
`potential_iterations`, `potential_regularization`, and
`potential_preconditioner` so solver-budget and preconditioner changes are
auditable. The example also exposes `REUSE_EXISTING_REPORTS`; when this is true,
an existing report is reused only if its recorded grid, timestep, geometry,
transient, and potential-solver metadata match the requested case. This makes
larger sweeps restartable while preventing stale JSON files from being treated
as new validation evidence. When `final_potential_residual_l2` blocks a
refinement summary, the
same JSON reports `current_potential_iterations` and
`recommended_potential_iterations` in `next_campaign_suggestion`; this keeps
elliptic-solver budget sweeps explicit instead of silently promoting a larger
movie grid. For the current hybrid `(16,24,96)` report-only probe, increasing
the unpreconditioned budget from `1536` to `3072` iterations reduced the final
potential residual to `3.8e-11` without materially changing the radial-flux or
spectral metrics; this supports explicit residual-budget scheduling rather than
changing the default compact campaign. The signed
`radial_flux_proxy` remains in each report as a cancellation and symmetry
diagnostic, but refinement promotion uses magnitude and RMS radial-flux
statistics because a domain-averaged signed flux can change sign when inward
and outward turbulent transport nearly cancel. Failed metrics that land within
five percent of the declared tolerance are also copied to
`near_tolerance_failed_metric_reports`; the gate still fails, but the summary
recommends repeating or extending the same transient before making an expensive
grid jump. This distinction is important for the current high-grid hybrid
campaign, where the `3072`-iteration residual budget passes the potential and
time gates, while `radial_flux_abs_mean` and `radial_flux_rms` miss the grid
gate only marginally. A follow-up `frames=8` report-only repeat did not remove
that sensitivity and introduced a toroidal spectral-centroid time failure, so
the next promotion attempt should use true grid refinement and a smaller
effective frame timestep rather than only increasing the potential-solver or
movie-window budget. The refinement summary also
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
diagnostics are stable. The next committed candidate settings are deliberately
heavier than the default quick check and should be run only when the goal is to
advance the 3D imported-field movie toward publication evidence.

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
- `docs/data/essos_imported_drb_movie_stationarity_jacobi_media/data/report.json`
- `docs/data/essos_imported_drb_movie_stationarity_jacobi_media/images/diagnostics.png`
- `docs/data/essos_imported_drb_movie_stationarity_jacobi_media/images/snapshots.png`
- `docs/data/essos_imported_drb_movie_stationarity_jacobi_media/images/poster_compact.png`
- `docs/data/essos_imported_drb_movie_stationarity_jacobi_media/movies/movie_compact.gif`
- `docs/data/essos_imported_drb_movie_refinement_artifacts/data/essos_imported_drb_movie_refinement_summary.json`
