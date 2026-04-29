# ESSOS Imported QA-Coil DRB Movie

This page documents the first movie-grade non-axisymmetric DRB transient run
on externally traced Landreman-Paul QA coil field-line maps. The magnetic
field and field-line integration are supplied by ESSOS, then `jax_drb`
imports the annular trajectories as fixed-layout FCI maps and advances the
JAX-native `FciDrbState` with sheath losses, target recycling, neutral
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

Regenerate the campaign with:

```bash
JAX_DRB_ESSOS_ROOT=/path/to/ESSOS \
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py
```

## Current Gate

The current public report passes the following checks:

- endpoint fraction on the imported FCI maps: about `0.82`;
- magnetic-field modulation from the coil field: about `9.47`;
- ion-density fluctuation RMS grows from about `1.3e-2` to `6.6e-2`;
- compact potential residual: about `9.5e-11`;
- target recycling and zero-current sheath residuals are at roundoff;
- neutral particle and momentum residuals are at roundoff;
- the final toroidal/poloidal spectrum has nontrivial mode content and a
  low-mode power fraction of about `0.33`;
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
