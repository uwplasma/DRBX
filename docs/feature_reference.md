# Feature Reference

This page is the high-level map from user goals to examples, inputs, outputs,
source code, validation tests, and release-backed media. It is intended to be
the first page to read after installation when deciding what JAXDRB can run
today and which scripts or modules to inspect.

Large movies, figures, and NPZ payloads are not tracked in git. They are linked
from GitHub Releases and can be restored locally with:

```bash
python scripts/fetch_example_artifacts.py
```

## Visual Overview

Representative release-backed outputs:

![Diverted tokamak dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__diverted_tokamak_turbulence_artifacts__movies__diverted_tokamak_turbulence.gif)

![Toroidal tokamak dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__movies__tokamak_tcv_x21_toroidal.gif)

![Imported QA stellarator dynamics](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_drb_movie_stationarity_jacobi_media__movies__movie_compact.gif)

## Capability Map

| Capability | User entry point | Main inputs | Main outputs | Implementation | Validation and tests |
| --- | --- | --- | --- | --- | --- |
| TOML runtime and CLI | [`jax_drb run`](native_runtime_cli.md), [`examples/restartable_diffusion_tutorial.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/restartable_diffusion_tutorial.py) | TOML deck with `[time]`, `[runtime]`, `[mesh]`, `[solver]`, `[model]`, `[species.*]`, `[fields.*]`, `[output]`, `[restart]` | summary JSON, arrays NPZ, restart NPZ, run-log JSON | [`src/jax_drb/cli.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/cli.py), [`src/jax_drb/native/deck_runner.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/deck_runner.py), [`src/jax_drb/runtime/output.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/runtime/output.py) | [`tests/test_cli_run.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_cli_run.py), [`tests/test_restartable_diffusion_tutorial.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_restartable_diffusion_tutorial.py) |
| Restartable diffusion and MMS | [`scripts/run_fluid_1d_mms_convergence.py`](https://github.com/uwplasma/jax_drb/blob/main/scripts/run_fluid_1d_mms_convergence.py), [Fluid 1D MMS](fluid_1d_mms_convergence.md) | analytic manufactured fields, grid levels, diffusion/transport coefficients | convergence JSON/NPZ/PNG, observed-order table | [`src/jax_drb/native/fluid_1d.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/fluid_1d.py), [`src/jax_drb/validation/fluid_1d_mms_convergence.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/fluid_1d_mms_convergence.py) | [`tests/test_native_fluid_1d.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_native_fluid_1d.py), [`tests/test_mms_convergence.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_mms_convergence.py) |
| Electrostatic vorticity | [`jax_drb run`](native_runtime_cli.md) | compact grid, vorticity initial field, electrostatic potential closure | vorticity and potential history arrays | [`src/jax_drb/native/vorticity.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/vorticity.py) | [`tests/test_native_vorticity.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_native_vorticity.py) |
| Structured elliptic solver (solvax) | [Physics Models](physics_models.md), [Performance And Differentiability](performance_and_differentiability.md) | metric weights, periodic/bounded axes, right-hand side | inverted potential field, differentiable through the solve | [`solvax.elliptic`](https://github.com/uwplasma/SOLVAX), [`src/jax_drb/native/vorticity.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/vorticity.py) | [`tests/test_native_vorticity.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_native_vorticity.py) |
| Differentiability, UQ, and inverse design | [`examples/autodiff_diffusion_sensitivity_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/autodiff_diffusion_sensitivity_demo.py), [`examples/autodiff_diffusion_uncertainty_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/autodiff_diffusion_uncertainty_demo.py), [`examples/autodiff_diffusion_inverse_design_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/autodiff_diffusion_inverse_design_demo.py) | differentiable scalar objectives, covariance, design parameters | sensitivity plots, covariance pushforward plots, optimization traces | [`src/jax_drb/validation/autodiff_diffusion.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/autodiff_diffusion.py), [`src/jax_drb/validation/autodiff_diffusion_uncertainty.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/autodiff_diffusion_uncertainty.py) | [`tests/test_validation_autodiff_diffusion.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_validation_autodiff_diffusion.py), [`tests/test_validation_autodiff_diffusion_uncertainty.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_validation_autodiff_diffusion_uncertainty.py) |
| Diverted tokamak movie and profiles | [`examples/diverted_tokamak_movie_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/diverted_tokamak_movie_demo.py), [`examples/diverted_tokamak_profile_analysis_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/diverted_tokamak_profile_analysis_demo.py) | release-backed tokamak mesh/field arrays | GIF, poster, snapshots, radial profiles, target lineouts | [`src/jax_drb/validation/diverted_tokamak_movie.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/diverted_tokamak_movie.py) | [`tests/test_validation_diverted_tokamak_movie.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_validation_diverted_tokamak_movie.py) |
| Synthetic stellarator FCI and reduced SOL | [`examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py), [Stellarator Examples](stellarator_examples.md) | analytic non-axisymmetric geometry constants, FCI maps, reduced turbulence settings | geometry plots, linear mode, nonlinear GIF, profile analysis | [`src/jax_drb/geometry/stellarator.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/geometry/stellarator.py), [`src/jax_drb/native/fci.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/fci.py), [`src/jax_drb/native/fci_drb_rhs.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/native/fci_drb_rhs.py), [`src/jax_drb/validation/stellarator_sol_showcase.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/stellarator_sol_showcase.py) | [`tests/test_geometry_fci_maps.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_geometry_fci_maps.py), [`tests/test_validation_stellarator_fci_campaigns.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_validation_stellarator_fci_campaigns.py) |
| Imported coil, VMEC, and hybrid geometry | [ESSOS Imported FCI Validation](essos_imported_fci_validation.md) | imported field-line arrays, VMEC map coordinates, endpoint masks, connection-length reports | FCI/source/refinement reports, Poincare figures, movie QA artifacts | [`src/jax_drb/geometry/essos_import.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/geometry/essos_import.py), [`src/jax_drb/validation/essos_imported_fci_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/essos_imported_fci_campaign.py), [`src/jax_drb/validation/essos_imported_drb_movie_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/essos_imported_drb_movie_campaign.py) | [`tests/test_essos_fieldline_import.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_essos_fieldline_import.py), [`tests/test_validation_stellarator_fci_campaigns.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_validation_stellarator_fci_campaigns.py) |
| VMEC-extender field-grid import | [`examples/geometry-3D/vmec-extender/imported_field_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/geometry-3D/vmec-extender/imported_field_demo.py), [VMEC Extender Edge Fields](vmec_extender_edge_fields.md) | compact NetCDF field grids with physical `phi`, cylindrical field components, metadata | imported-field report, FCI maps, compact SOL smoke package | [`src/jax_drb/geometry/vmec_extender_import.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/geometry/vmec_extender_import.py), [`src/jax_drb/validation/vmec_extender_edge_field_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/validation/vmec_extender_edge_field_campaign.py) | [`tests/test_vmec_extender_import.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_vmec_extender_import.py), [`tests/test_validation_vmec_extender_edge_field_campaign.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_validation_vmec_extender_edge_field_campaign.py) |
| Performance and scaling | [`examples/strong_scaling_diffusion_demo.py`](https://github.com/uwplasma/jax_drb/blob/main/examples/strong_scaling_diffusion_demo.py) | fixed workload, batch size, CPU/GPU selector, profiler output directory | scaling plot, RSS/profile JSON, JAX trace/profile summaries | [`src/jax_drb/runtime/performance.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/runtime/performance.py) | [`tests/test_runtime_performance.py`](https://github.com/uwplasma/jax_drb/blob/main/tests/test_runtime_performance.py) |
| Release and artifact management | [Release Packaging](release_packaging.md) | release artifact bundle, GitHub token, optional cache directory | restored docs media, wheel/sdist, footprint audit | [`scripts/fetch_example_artifacts.py`](https://github.com/uwplasma/jax_drb/blob/main/scripts/fetch_example_artifacts.py), [`scripts/audit_repository_footprint.py`](https://github.com/uwplasma/jax_drb/blob/main/scripts/audit_repository_footprint.py) | [Release Packaging](release_packaging.md) |

## Runbook By User Goal

| Goal | Commands | Expected artifacts |
| --- | --- | --- |
| Install and run a compact native case | `pip install jax-drb`; `jax_drb inspect examples/inputs/restartable_diffusion.toml`; `jax_drb run examples/inputs/restartable_diffusion.toml --verbose` | `output/.../*_summary.json`, `*_arrays.npz`, `*_restart.npz`, `*_run_log.json` |
| Restore figures and movies without bloating git | `python scripts/fetch_example_artifacts.py` | `docs/data/**` PNG/GIF/NPZ media restored from the release bundle |
| Regenerate the public restart tutorial | `PYTHONPATH=src python examples/restartable_diffusion_tutorial.py --quiet` | density snapshots, density surface, restart-consistency plot, run logs |
| Demonstrate differentiability | run the three `examples/autodiff_diffusion_*_demo.py` scripts | sensitivity, uncertainty, and inverse-design figures plus JSON/NPZ reports |
| Make diverted tokamak movie/profile plots | restore artifacts, then run `examples/diverted_tokamak_movie_demo.py` and `examples/diverted_tokamak_profile_analysis_demo.py` | GIF, poster, snapshots, target/profile/time-trace PNG |
| Run compact stellarator turbulence | run `geometry_plotting_demo.py`, `linear_mode_demo.py`, `vorticity_bracket_demo.py`, `nonlinear_turbulence_demo.py`, and `turbulent_profile_analysis_demo.py` | geometry maps, nonlinear GIF/poster, radial fluctuation and transport-proxy plots |
| Audit imported coil/VMEC/hybrid geometry | run `direct_coil_open_sol_demo.py`, `vmec_closed_field_demo.py`, or `hybrid_open_sol_demo.py` in dry-run mode | machine-readable workflow summaries and promotion-boundary ledgers |
| Run validation campaigns for docs/paper figures | run selected scripts under `examples/benchmarks/` and `examples/geometry-3D/` | JSON/NPZ/PNG validation packages used by the documentation gallery |
| Run release checks | `python scripts/run_fast_research_checks.py`; `pytest -q -m "not slow"`; `mkdocs build --strict --clean` | local pass/fail evidence for docs, tests, and build surfaces |

## Input Surfaces

JAXDRB has three public input styles:

| Input style | Used by | Details |
| --- | --- | --- |
| TOML decks | `jax_drb run`, `run_input_case` | See [Input And Output Reference](input_output_reference.md) for `[time]`, `[runtime]`, `[mesh]`, `[model]`, `[species.*]`, `[fields.*]`, `[output]`, and `[restart]`. |
| Python example constants | most scripts in `examples/` | Scripts follow the SIMSOPT-style pattern: edit constants near the top, run the file, inspect the output directory. |
| Release-backed artifacts | movies, galleries, cached example media | Restored by `scripts/fetch_example_artifacts.py`; avoids large tracked media and keeps clone size small. |

## Output Surfaces

| Output | Format | Purpose |
| --- | --- | --- |
| Summary report | JSON | scalar metadata, capability tier, runtime configuration, variable summaries |
| Arrays | NPZ | field histories and derived arrays for plotting and analysis |
| Restart payload | NPZ | resume native runs without starting from the initial condition |
| Run log | JSON | ordered runtime events, progress, timing, sanitized paths, artifact provenance |
| Validation package | JSON/NPZ/PNG/GIF | publication-grade figures and machine-readable metrics |
| Movie package | GIF/PNG/NPZ/JSON | dynamics movie, poster, snapshots, diagnostics, visual-QA metadata |
| Release artifact bundle | ZIP | large generated media stored outside git history |

## Source-Code Reading Path

For users who want to understand or modify the implementation, read in this
order:

1. [Native runtime CLI](native_runtime_cli.md) and [`src/jax_drb/cli.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/cli.py).
2. [Input And Output Reference](input_output_reference.md) and [`src/jax_drb/runtime/output.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jax_drb/runtime/output.py).
3. [Physics Models](physics_models.md) and [Equation To Code Map](equation_to_code_map.md).
4. [Code Structure](code_structure.md), especially the FCI and geometry sections.
5. [Validation Gallery](validation_gallery.md).
6. [Performance And Differentiability](performance_and_differentiability.md) before changing solver backends or claiming speedups.

## Claim Boundaries

The docs use conservative labels because the code mixes stable release
surfaces with active research lanes:

| Label | Meaning |
| --- | --- |
| `native_exact` | native JAX model with exact or roundoff-level agreement on its validation target |
| `native_operational` | native and useful, with bounded documented residuals |
| `self-contained` | clean-clone runnable after installing JAXDRB and, when needed, restoring release artifacts |
| `developer/geometry-input` | regenerates source data from heavier local geometry inputs |
| `opt-in research gate` | tested enough for development and evidence collection, but not a stable default claim |

When a page advertises a movie or figure, the corresponding example command,
artifact path, source module, and validation status should be traceable through
this page, [Examples And Artifact Map](examples.md), or
[Validation Gallery](validation_gallery.md).
