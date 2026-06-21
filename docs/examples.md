# Examples And Artifact Map

The examples are grouped by intent. User-facing examples are ordinary Python
scripts with parameters near the top of the file. Campaign examples generate
JSON/NPZ/PNG/GIF artifacts that are also used by tests and documentation.

## Quick Start

Large generated example outputs are release-backed so that cloning the private
repository stays fast. Authenticate with the GitHub CLI or a token, then restore
the media and baseline bundles before running the movie and validation examples:

```bash
gh auth login --hostname github.com
python scripts/fetch_example_artifacts.py
```

Without the GitHub CLI, set `GH_TOKEN` or `GITHUB_TOKEN` to a token with access
to `uwplasma/jax_drb`. The fetch script restores PNG/GIF/NPZ docs media under
`docs/data/` and heavy reference baselines under `references/baselines/`. Use
`python scripts/fetch_example_artifacts.py --skip-baselines` when you only need
the README/docs movies, figures, and user-facing self-contained example
payloads. The expected release URLs are tracked in
`docs/release_artifacts_manifest.json`.

After this step, the user-facing commands listed below are self-contained: they
use JAXDRB source code plus release-backed arrays/media and do not require users
to download external plasma codes. Some scripts under `examples/engineering/`
and selected-field regeneration directories are developer/live-reference
workflows; those scripts say so in their command-line help and require an
explicit `--reference-root` or local geometry checkout when they are rerun from
first principles.

To exercise the main self-contained tutorial and movie surface from a fresh
checkout, run:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines
PYTHONPATH=src python examples/restartable_diffusion_tutorial.py --quiet
PYTHONPATH=src python examples/autodiff_diffusion_sensitivity_demo.py
PYTHONPATH=src python examples/autodiff_diffusion_uncertainty_demo.py
PYTHONPATH=src python examples/autodiff_diffusion_inverse_design_demo.py
PYTHONPATH=src python examples/diverted_tokamak_movie_demo.py
PYTHONPATH=src python examples/diverted_tokamak_profile_analysis_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/geometry_plotting_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/linear_mode_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py
PYTHONPATH=src python examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py
```

Imported external-geometry regeneration scripts are developer workflows. Their
published arrays and movies are release-backed, but rerunning those adapters
from coils or external field-line traces requires a local geometry checkout and
is documented separately on the imported-geometry pages. Users who only want to
run the README/docs stellarator examples should use the `stellarator-fci`
commands above after `fetch_example_artifacts.py --skip-baselines`.

| Example | What it teaches |
| --- | --- |
| [`examples/inputs/restartable_diffusion.toml`](https://github.com/uwplasma/jax_drb/blob/main/examples/inputs/restartable_diffusion.toml) | Small native TOML deck with restartable output. |
| [`examples/restartable_diffusion_tutorial.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/restartable_diffusion_tutorial.py) | End-to-end run, restart, NPZ reading, and plotting workflow. |
| [`examples/diffusion_precision_benchmark.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/diffusion_precision_benchmark.py) | Float32/float64 runtime comparison on the compact diffusion lane. |

Representative output:

![Restartable diffusion density surface](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__restartable_diffusion_demo_artifacts__images__restartable_diffusion_density_surface.png)

## Differentiability

| Example | What it teaches |
| --- | --- |
| [`examples/autodiff_diffusion_sensitivity_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/autodiff_diffusion_sensitivity_demo.py) | `jax.grad` sensitivity against finite differences. |
| [`examples/autodiff_diffusion_uncertainty_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/autodiff_diffusion_uncertainty_demo.py) | Covariance pushforward and vectorized Monte Carlo. |
| [`examples/autodiff_diffusion_inverse_design_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/autodiff_diffusion_inverse_design_demo.py) | Gradient-based inverse-design loop. |
| [`examples/strong_scaling_diffusion_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/strong_scaling_diffusion_demo.py) | Fixed-work CPU/GPU process-group scaling demo. |

Representative output:

![Autodiff diffusion uncertainty](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__autodiff_diffusion_uncertainty_artifacts__images__autodiff_diffusion_uncertainty.png)

## 3D Stellarator Geometry

These scripts follow the SIMSOPT-style pattern: edit constants near the top,
run the file, inspect the output directory.

| Example | What it writes |
| --- | --- |
| [`geometry_plotting_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/geometry_plotting_demo.py) | Non-axisymmetric geometry, metric, connection-length, and curvature-map figure. |
| [`linear_mode_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/linear_mode_demo.py) | Linear FCI mode history plus diagnostics and snapshots. |
| [`vorticity_bracket_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/vorticity_bracket_demo.py) | Physics-backed nonlinear coupling through the vorticity/potential solve and logical \(E\times B\) bracket. |
| [`nonlinear_turbulence_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py) | Compact nonlinear reduced SOL history, diagnostics, snapshots, 3D poster, and GIF movie. |
| [`turbulent_profile_analysis_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py) | Radial fluctuation, RMS, transport-proxy, connection-length, and energy-trace analysis from the nonlinear SOL history. |
| [`validation_campaign_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/validation_campaign_demo.py) | Full promoted synthetic stellarator FCI validation bundle. |

Representative output:

![Stellarator SOL diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__images__stellarator_sol_showcase_diagnostics.png)

The detailed guide is [Stellarator Examples](stellarator_examples.md).

## Imported 3D Geometry

The VMEC-extender entry is a self-contained synthetic validation example that
writes its own compact NetCDF field grids before running the import and SOL
verification gates. The ESSOS imported-geometry entries document developer
regeneration of external-geometry artifacts; users can inspect the published
figures and movies after `fetch_example_artifacts.py`, while rerunning those
adapters requires the geometry source checkout named by the page.

| Example | What it teaches |
| --- | --- |
| [`examples/geometry-3D/vmec-extender/imported_field_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/vmec-extender/imported_field_demo.py) | VMEC-extender field-grid import and compact SOL verification gate. |
| [`examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py) | External QA field-line import into portable arrays. |
| [`examples/geometry-3D/essos-field-lines/direct_coil_open_sol_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/direct_coil_open_sol_demo.py) | Direct-coil open-SOL promotion workflow: dry-run contract by default, with opt-in live FCI, source/profile, connection-length, endpoint-label, stationarity, and diagnostic media gates. |
| [`examples/geometry-3D/essos-field-lines/direct_coil_closed_field_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/direct_coil_closed_field_demo.py) | Direct-coil closed/near-closed return-map control plus refinement gate: self-contained by default, with opt-in live ESSOS tracing and no target/sheath semantics. |
| [`examples/geometry-3D/essos-field-lines/vmec_closed_field_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/vmec_closed_field_demo.py) | VMEC closed-field control: dry-run contracts by default, with opt-in live periodic FCI/operator gates plus reduced transient, profile, spectrum, and GIF artifacts. |
| [`examples/geometry-3D/essos-field-lines/imported_fci_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_fci_campaign.py) | Imported FCI maps with sheath/recycling and neutral validation. |
| [`examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py) | Self-contained nested connection-length refinement gate for imported-field promotion. |
| [`examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py) | Imported fixed-layout PyTree/JVP RHS gate. |
| [`examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py) | Reduced imported QA DRB movie with open endpoints where present. |
| [`examples/geometry-3D/essos-field-lines/imported_drb_movie_refinement_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_drb_movie_refinement_campaign.py) | Report-only grid/time transient sweep for imported DRB movie promotion, without writing GIF/NPZ media. |
| [`examples/geometry-3D/essos-field-lines/imported_drb_movie_refinement_summary.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_drb_movie_refinement_summary.py) | Report-only grid/time refinement summary for imported DRB movie promotion. |
| [`examples/geometry-3D/essos-field-lines/imported_drb_movie_stationarity_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_drb_movie_stationarity_campaign.py) | JSON-only long-window stationarity gate for the high-resolution Jacobi imported-field movie settings. |
| [`examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py) | Field-line/surface registration and Poincare diagnostics. |

Representative output:

![ESSOS imported QA-hybrid DRB movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__movies__movie_compact.gif)

## Tokamak Geometry And Movies

| Example | What it teaches |
| --- | --- |
| [`examples/diverted_tokamak_movie_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/diverted_tokamak_movie_demo.py) | Full-domain diverted tokamak movie from benchmark output. |
| [`examples/diverted_tokamak_profile_analysis_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/diverted_tokamak_profile_analysis_demo.py) | Radial profiles, target lineouts, time traces, and final diverted-domain field analysis from release-backed arrays. |
| [`examples/tokamak-3D/tcv-x21/scaffold_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/tokamak-3D/tcv-x21/scaffold_demo.py) | TCV-X21 scaffold package. |
| [`examples/tokamak-3D/tcv-x21/selected_field_parity_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/tokamak-3D/tcv-x21/selected_field_parity_demo.py) | Compact selected-field parity package. |
| [`examples/tokamak-3D/tcv-x21/toroidal_movie_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/tokamak-3D/tcv-x21/toroidal_movie_demo.py) | Toroidal 3D movie from scaffold arrays. |

Representative output:

![Diverted tokamak movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

## Validation Campaigns

The engineering examples in
[`examples/engineering/`](https://github.com/uwplasma/jax_drb/tree/main/examples/engineering)
regenerate publication-grade validation packages:

| Family | Documentation |
| --- | --- |
| MMS and open-field operators | [Fluid 1D MMS](fluid_1d_mms_convergence.md), [Open-Field Operator Campaign](open_field_operator_campaign.md) |
| Reactions, collisions, neutrals, target recycling | [Reactions And Collisions](reactions_collisions_campaign.md), [Collision Closure](collision_closure_campaign.md), [Neutral Parallel Diffusion](neutral_parallel_diffusion_campaign.md), [Target Recycling](target_recycling_campaign.md) |
| Reference comparison | [Hermes Comparison Gallery](hermes_comparison_gallery.md), with live reruns reserved for developer validation. |
| Performance | [Local CPU Scaling](local_cpu_scaling_campaign.md), [Native 3D Runtime](native_3d_runtime_campaign.md), [JAX Native Profile Audit](jax_native_profile_audit.md) |
| Control and detachment | [Controller Feedback](controller_feedback_campaign.md), [Temperature Feedback](temperature_feedback_campaign.md), [Detachment Controller](detachment_controller_campaign.md) |

All validation figures are collected in [Validation Gallery](validation_gallery.md).

Reference-comparison and live-rerun commands are intentionally separated from
the user examples. They are used to refresh baselines, parity reports, and
publication-grade validation figures, and they can require heavier local
reference data or a local reference checkout. The release-backed baseline
bundles let users and CI run cached validation checks without those live
dependencies.
