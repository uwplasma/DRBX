# Equation To Code Map

This page maps the main model terms in the documentation to the modules that
implement and test them. It is intended as a developer and reviewer index: use
[physics_models.md](physics_models.md) for the derivation-level description,
[code_structure.md](code_structure.md) for package organization, and the pages
listed here for validation artifacts.

## Core Drift-Reduced Operators

| Model term | Implementation | Validation or tests |
| --- | --- | --- |
| Parallel gradient, divergence, and structured metric factors | [`native/mesh.py`](../src/jax_drb/native/mesh.py), [`native/metrics.py`](../src/jax_drb/native/metrics.py) | [`tests/test_native_mesh.py`](../tests/test_native_mesh.py), [`tests/test_native_metrics.py`](../tests/test_native_metrics.py) |
| Density, pressure, and vorticity transport on compact native decks | [`native/fluid_1d.py`](../src/jax_drb/native/fluid_1d.py), [`native/transport.py`](../src/jax_drb/native/transport.py), [`native/vorticity.py`](../src/jax_drb/native/vorticity.py) | [Fluid 1D MMS Convergence](fluid_1d_mms_convergence.md), [`tests/test_native_fluid_1d.py`](../tests/test_native_fluid_1d.py), [`tests/test_native_transport.py`](../tests/test_native_transport.py), [`tests/test_native_vorticity.py`](../tests/test_native_vorticity.py) |
| Elliptic potential/vorticity solve, including Boussinesq and non-Boussinesq FCI polarization gates | [`solver/elliptic.py`](../src/jax_drb/solver/elliptic.py), [`native/fci_vorticity.py`](../src/jax_drb/native/fci_vorticity.py) | [`tests/test_solver_elliptic.py`](../tests/test_solver_elliptic.py), [Stellarator FCI Validation](stellarator_fci_validation.md) |
| Implicit residuals, sparse Jacobians, JVP Jacobian actions, and linearized Newton solves | [`solver/implicit.py`](../src/jax_drb/solver/implicit.py) | [`tests/test_solver_implicit.py`](../tests/test_solver_implicit.py) |

## Non-Axisymmetric And 3D Geometry

| Model term | Implementation | Validation or tests |
| --- | --- | --- |
| Field-line-following interpolation, metric-weighted operators, and 3D selected-field surfaces | [`native/fci.py`](../src/jax_drb/native/fci.py), [`native/fci_drb_rhs.py`](../src/jax_drb/native/fci_drb_rhs.py) | [Stellarator FCI Validation](stellarator_fci_validation.md) |
| FCI sheath/recycling, neutral, and vorticity closure gates | [`native/fci_sheath_recycling.py`](../src/jax_drb/native/fci_sheath_recycling.py), [`native/fci_neutral.py`](../src/jax_drb/native/fci_neutral.py), [`native/fci_vorticity.py`](../src/jax_drb/native/fci_vorticity.py) | [Stellarator FCI Validation](stellarator_fci_validation.md), [`tests/test_validation_stellarator_fci_campaigns.py`](../tests/test_validation_stellarator_fci_campaigns.py) |
| Imported field-line and surface geometry adapters | [`geometry/essos_import.py`](../src/jax_drb/geometry/essos_import.py), [`geometry/vmec_extender_import.py`](../src/jax_drb/geometry/vmec_extender_import.py), [`validation/essos_imported_fci_campaign.py`](../src/jax_drb/validation/essos_imported_fci_campaign.py), [`validation/vmec_extender_edge_field_campaign.py`](../src/jax_drb/validation/vmec_extender_edge_field_campaign.py) | [ESSOS Field-Line Import](essos_fieldline_import.md), [ESSOS Imported FCI Validation](essos_imported_fci_validation.md), [VMEC Extender Edge Fields](vmec_extender_edge_fields.md) |
| 3D movies, posters, profile analysis, and validation-gallery plots | [`validation/stellarator_sol_showcase.py`](../src/jax_drb/validation/stellarator_sol_showcase.py), [`validation/essos_imported_drb_movie_campaign.py`](../src/jax_drb/validation/essos_imported_drb_movie_campaign.py), [`validation/diverted_tokamak_movie.py`](../src/jax_drb/validation/diverted_tokamak_movie.py), [`validation/publication_plotting.py`](../src/jax_drb/validation/publication_plotting.py) | [ESSOS Imported FCI Validation](essos_imported_fci_validation.md), [Validation Gallery](validation_gallery.md) |

## Runtime, I/O, And Artifacts

| Surface | Implementation | Validation or tests |
| --- | --- | --- |
| TOML/input parsing, precision defaults, output manifests, restart metadata, and live progress estimates | [`config`](../src/jax_drb/config), [`runtime`](../src/jax_drb/runtime), [`native/deck_runner.py`](../src/jax_drb/native/deck_runner.py) | [Inputs And Outputs](input_output_reference.md), [Native Runtime CLI](native_runtime_cli.md), [Restartable Diffusion Tutorial](restartable_diffusion_tutorial.md), [`tests/test_cli_run.py`](../tests/test_cli_run.py), [`tests/test_restartable_diffusion_tutorial.py`](../tests/test_restartable_diffusion_tutorial.py) |
| Portable run summaries and array payloads | [`native/deck_runner.py`](../src/jax_drb/native/deck_runner.py) | [Native Runtime CLI](native_runtime_cli.md), [`tests/test_cli_run.py`](../tests/test_cli_run.py) |
| Self-contained examples, release-backed media, and artifact restoration | [`examples`](../examples), [`scripts/fetch_example_artifacts.py`](../scripts/fetch_example_artifacts.py) | [Examples And Artifacts](examples.md), [`tests/docs/examples/test_self_contained_example_smoke.py`](../tests/docs/examples/test_self_contained_example_smoke.py) |

## Current Gaps

The most important open implementation gaps remain deliberately separated from
the validated equations above:

- Multi-device GPU speedup is not a release claim until the device-level
  identity check, the real-kernel agreement gate, and a committed timing summary
  all pass.
