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
the README/docs movies and figures.

After this step, the user-facing examples and movies are self-contained: they
use JAXDRB source code plus release-backed arrays/media and do not require users
to download external plasma codes. Live reference-code reruns remain available
as developer validation campaigns, but they are not part of the normal examples
workflow.

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
| [`nonlinear_turbulence_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py) | Compact nonlinear reduced SOL history, diagnostics, snapshots, 3D poster, and GIF movie. |
| [`turbulent_profile_analysis_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py) | Radial fluctuation, RMS, transport-proxy, connection-length, and energy-trace analysis from the nonlinear SOL history. |
| [`validation_campaign_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/validation_campaign_demo.py) | Full promoted synthetic stellarator FCI validation bundle. |

Representative output:

![Stellarator SOL diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__stellarator_fci_validation_artifacts__showcase__images__stellarator_sol_showcase_diagnostics.png)

The detailed guide is [Stellarator Examples](stellarator_examples.md).

## Imported 3D Geometry

| Example | What it teaches |
| --- | --- |
| [`examples/geometry-3D/vmec-extender/imported_field_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/vmec-extender/imported_field_demo.py) | VMEC-extender field-grid import and compact SOL smoke gate. |
| [`examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py) | External QA field-line import into portable arrays. |
| [`examples/geometry-3D/essos-field-lines/imported_fci_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_fci_campaign.py) | Imported FCI maps with sheath/recycling and neutral validation. |
| [`examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py) | Imported fixed-layout PyTree/JVP RHS gate. |
| [`examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py) | Reduced imported QA DRB movie with open endpoints where present. |
| [`examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py) | Field-line/surface registration and Poincare diagnostics. |

Representative output:

![ESSOS imported QA-coil DRB diagnostics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_artifacts__images__essos_imported_drb_movie_campaign_diagnostics.png)

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
