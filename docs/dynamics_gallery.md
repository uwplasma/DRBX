# Dynamics Gallery

This page gathers the main visual dynamics packages advertised by the README
and docs. The movies are release-backed so the repository remains lightweight.
Use this page to find the command that regenerates each figure, the source
module that implements the plotting, and the validation status that bounds the
claim.

Restore media first:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines
```

## Diverted Tokamak Turbulence

![Diverted tokamak turbulence](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

Regenerate the movie and profile package:

```bash
PYTHONPATH=src python examples/diverted_tokamak_movie_demo.py
PYTHONPATH=src python examples/diverted_tokamak_profile_analysis_demo.py
```

| Item | Details |
| --- | --- |
| Outputs | GIF, poster, snapshots, radial profile, target lineouts, time traces |
| Source | [`src/jax_drb/validation/diverted_tokamak_movie.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/diverted_tokamak_movie.py) |
| Example | [`examples/diverted_tokamak_movie_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/diverted_tokamak_movie_demo.py) |
| Status | benchmark-backed visualization from release-restored arrays |
| Related docs | [Diverted Tokamak Movie](diverted_tokamak_movie_demo.md) |

The visual package is intended for quick physics QA: coherent edge structures,
wall/divertor context, target-adjacent activity, and profile diagnostics should
be inspectable without a live external reference run.

## Synthetic Stellarator FCI Reduced SOL

![Stellarator SOL 3D movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__movies__stellarator_sol_showcase.gif)

Run the full self-contained educational workflow:

```bash
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/geometry_plotting_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/linear_mode_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py
```

| Item | Details |
| --- | --- |
| Outputs | geometry plots, linear-mode snapshots, nonlinear diagnostics, 3D poster, GIF, profile analysis |
| Source | [`src/jax_drb/validation/stellarator_sol_showcase.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/stellarator_sol_showcase.py), [`src/jax_drb/native/fci_drb_rhs.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/fci_drb_rhs.py) |
| Examples | [`examples/geometry-3D/stellarator-fci`](https://github.com/uwplasma/jax_drb/tree/main/examples/geometry-3D/stellarator-fci) |
| Status | self-contained reduced non-axisymmetric SOL demonstration with separate operator validation gates |
| Related docs | [Stellarator Examples](stellarator_examples.md), [Stellarator FCI Validation](stellarator_fci_validation.md), [Connection Length](connection_length.md) |

This workflow is the best starting point for users who want a clean-clone 3D
stellarator example. The nonlinear movie is intentionally compact. The
vorticity/bracket example shows the more physics-backed nonlinear coupling
through the tested logical `E x B` bracket and potential/vorticity solve.

## Imported QA Coil, VMEC, And Hybrid Geometry

![ESSOS imported QA-hybrid DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__movies__movie_compact.gif)

Inspect release-backed evidence and run default self-contained contracts:

```bash
PYTHONPATH=src python examples/geometry-3D/essos-field-lines/direct_coil_open_sol_demo.py
PYTHONPATH=src python examples/geometry-3D/essos-field-lines/vmec_closed_field_demo.py
PYTHONPATH=src python examples/geometry-3D/essos-field-lines/hybrid_open_sol_demo.py
PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py
PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_drb_movie_refinement_summary.py
```

Developer regeneration from live external geometry is opt-in and documented on
the imported-geometry pages.

| Item | Details |
| --- | --- |
| Outputs | connection-length refinement reports, endpoint/source ledgers, FCI maps, movie QA, diagnostics, snapshots |
| Source | [`src/jax_drb/geometry/essos_import.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/geometry/essos_import.py), [`src/jax_drb/validation/essos_imported_drb_movie_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/essos_imported_drb_movie_campaign.py) |
| Examples | [`examples/geometry-3D/essos-field-lines`](https://github.com/uwplasma/jax_drb/tree/main/examples/geometry-3D/essos-field-lines) |
| Status | compact release-backed vacuum bridge and diagnostic contracts; full finite-beta and long-window device-scale turbulence are deferred |
| Related docs | [ESSOS Field-Line Import](essos_fieldline_import.md), [ESSOS Imported FCI Validation](essos_imported_fci_validation.md), [ESSOS Imported DRB Movie](essos_imported_drb_movie.md) |

The imported-geometry lane is split deliberately. Users can inspect restored
figures and run dry-run or release-backed contracts from a clean clone.
Developers can opt into live field-line regeneration when refreshing the
validation evidence.

## Reading The Gallery Scientifically

The visual hierarchy is:

| Level | Interpretation |
| --- | --- |
| tutorial movie | shows how to run and inspect a workflow |
| validation movie | backed by a campaign report, thresholds, and tests |
| benchmark-backed visualization | uses restored benchmark/reference arrays and documented plotting |
| scaffold/control movie | validates plotting, geometry, or selected-field wiring, but is not a full physics result |

Use [Validation Gallery](validation_gallery.md) for the full figure set,
[Research-Grade Validation Matrix](research_grade_validation_matrix.md) for
promotion status, and [Feature Reference](feature_reference.md) to trace each
movie to inputs, outputs, source code, and tests.
