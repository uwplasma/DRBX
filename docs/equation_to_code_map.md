# Equation To Code Map

This page maps the main model terms in the documentation to the modules that
implement and test them. It is intended as a developer and reviewer index: use
[physics_models.md](physics_models.md) for the derivation-level description,
[code_structure.md](code_structure.md) for package organization, and the pages
listed here for validation artifacts.

## Core Drift-Reduced Operators

| Model term | Implementation | Validation or tests |
| --- | --- | --- |
| Parallel gradient, divergence, and open-field metric factors | [`native/open_field.py`](../src/jax_drb/native/open_field.py), [`native/mesh.py`](../src/jax_drb/native/mesh.py), [`native/metrics.py`](../src/jax_drb/native/metrics.py) | [Open-Field Operator Campaign](open_field_operator_campaign.md), [`tests/test_native_open_field.py`](../tests/test_native_open_field.py), [`tests/test_validation_open_field_operator_campaign.py`](../tests/test_validation_open_field_operator_campaign.py) |
| Density, pressure, and vorticity transport on compact native decks | [`native/fluid_1d.py`](../src/jax_drb/native/fluid_1d.py), [`native/vorticity.py`](../src/jax_drb/native/vorticity.py), [`native/drift_wave.py`](../src/jax_drb/native/drift_wave.py) | [Fluid 1D MMS Convergence](fluid_1d_mms_convergence.md), [Drift-Wave Benchmark](drift_wave_benchmark.md), [`tests/test_native_fluid_1d.py`](../tests/test_native_fluid_1d.py), [`tests/test_native_vorticity.py`](../tests/test_native_vorticity.py) |
| Elliptic potential/vorticity solve | [`solver/elliptic.py`](../src/jax_drb/solver/elliptic.py), [`native/fci_vorticity.py`](../src/jax_drb/native/fci_vorticity.py) | [`tests/test_solver_elliptic.py`](../tests/test_solver_elliptic.py), [Stellarator FCI Validation](stellarator_fci_validation.md) |
| Implicit residuals, sparse Jacobians, JVP Jacobian actions, and linearized Newton solves | [`solver/implicit.py`](../src/jax_drb/solver/implicit.py), [`native/recycling_fixed_residual.py`](../src/jax_drb/native/recycling_fixed_residual.py) | [Implicit Solver Profile Audit](implicit_solver_profile_audit.md), [Performance And Differentiability](performance_and_differentiability.md), [`tests/test_solver_implicit.py`](../tests/test_solver_implicit.py), [`tests/test_native_recycling_fixed_residual.py`](../tests/test_native_recycling_fixed_residual.py) |

## Recycling, Atomic Physics, And Boundaries

| Model term | Implementation | Validation or tests |
| --- | --- | --- |
| Evolving recycling fields, active layout, fixed PyTree state, and BDF residual seams | [`native/recycling_fields.py`](../src/jax_drb/native/recycling_fields.py), [`native/recycling_layout.py`](../src/jax_drb/native/recycling_layout.py), [`native/recycling_fixed_residual.py`](../src/jax_drb/native/recycling_fixed_residual.py), [`native/recycling_1d.py`](../src/jax_drb/native/recycling_1d.py) | [Profiling Runtime](profiling_runtime.md), [`tests/test_native_recycling_fields.py`](../tests/test_native_recycling_fields.py), [`tests/test_native_recycling_layout.py`](../tests/test_native_recycling_layout.py), [`tests/test_native_recycling_history.py`](../tests/test_native_recycling_history.py) |
| Atomic ionization, recombination, charge-exchange rates, and radiated-power fits | [`native/recycling_atomic.py`](../src/jax_drb/native/recycling_atomic.py), packaged data in [`data/atomic_rates`](../src/jax_drb/data/atomic_rates) | [Atomic Rate Differentiability](atomic_rate_differentiability_campaign.md), [`tests/test_native_recycling_atomic.py`](../tests/test_native_recycling_atomic.py) |
| Reaction source assembly for density, momentum, and pressure equations | [`native/recycling_reactions.py`](../src/jax_drb/native/recycling_reactions.py), [`native/recycling_source_accumulation.py`](../src/jax_drb/native/recycling_source_accumulation.py) | [Reactions And Collisions](reactions_collisions_campaign.md), [`tests/test_native_recycling_reactions.py`](../tests/test_native_recycling_reactions.py), [`tests/test_native_recycling_source_accumulation.py`](../tests/test_native_recycling_source_accumulation.py) |
| Collision frequencies, friction, heat exchange, and viscosity inputs | [`native/recycling_collisions.py`](../src/jax_drb/native/recycling_collisions.py), [`native/recycling_collision_closure.py`](../src/jax_drb/native/recycling_collision_closure.py) | [Collision Closure](collision_closure_campaign.md), [Reactions And Collisions](reactions_collisions_campaign.md), [`tests/test_native_recycling_collisions.py`](../tests/test_native_recycling_collisions.py), [`tests/test_native_recycling_collision_closure.py`](../tests/test_native_recycling_collision_closure.py) |
| Neutral parallel diffusion and mixed neutral transport | [`native/recycling_neutral_diffusion.py`](../src/jax_drb/native/recycling_neutral_diffusion.py), [`native/neutral_mixed.py`](../src/jax_drb/native/neutral_mixed.py), [`native/neutral_mixed_operators.py`](../src/jax_drb/native/neutral_mixed_operators.py) | [Neutral Parallel Diffusion](neutral_parallel_diffusion_campaign.md), [Neutral Mixed Term Balance](neutral_mixed_term_balance_campaign.md), [Neutral Mixed Boundary](neutral_mixed_boundary_campaign.md), [`tests/test_native_recycling_neutral_diffusion.py`](../tests/test_native_recycling_neutral_diffusion.py), [`tests/test_native_neutral_mixed_operators.py`](../tests/test_native_neutral_mixed_operators.py) |
| Sheath, target recycling, and guard-cell boundary updates | [`native/open_field.py`](../src/jax_drb/native/open_field.py), [`native/recycling_boundaries.py`](../src/jax_drb/native/recycling_boundaries.py), [`native/recycling_targets.py`](../src/jax_drb/native/recycling_targets.py), [`native/neutral_mixed_boundaries.py`](../src/jax_drb/native/neutral_mixed_boundaries.py) | [Target Recycling](target_recycling_campaign.md), [Neutral Mixed Boundary](neutral_mixed_boundary_campaign.md), [`tests/test_native_recycling_boundaries.py`](../tests/test_native_recycling_boundaries.py), [`tests/test_native_recycling_targets.py`](../tests/test_native_recycling_targets.py) |
| Density, temperature, detachment, impurity, and recycling controllers | [`native/recycling_feedback.py`](../src/jax_drb/native/recycling_feedback.py), [`validation/controller_feedback_campaign.py`](../src/jax_drb/validation/controller_feedback_campaign.py), [`validation/temperature_feedback_campaign.py`](../src/jax_drb/validation/temperature_feedback_campaign.py), [`validation/detachment_controller_campaign.py`](../src/jax_drb/validation/detachment_controller_campaign.py), [`validation/impurity_radiation_campaign.py`](../src/jax_drb/validation/impurity_radiation_campaign.py) | [Controller Feedback](controller_feedback_campaign.md), [Temperature Feedback](temperature_feedback_campaign.md), [Detachment Controller](detachment_controller_campaign.md), [Impurity Radiation](impurity_radiation_campaign.md) |

## Non-Axisymmetric And 3D Geometry

| Model term | Implementation | Validation or tests |
| --- | --- | --- |
| Field-line-following interpolation, metric-weighted operators, and 3D selected-field surfaces | [`native/fci.py`](../src/jax_drb/native/fci.py), [`native/fci_drb_rhs.py`](../src/jax_drb/native/fci_drb_rhs.py), [`validation/geometry_selected_field.py`](../src/jax_drb/validation/geometry_selected_field.py) | [Stellarator FCI Validation](stellarator_fci_validation.md), [Native 3D Convergence](native_3d_convergence_campaign.md), [Native 3D Runtime](native_3d_runtime_campaign.md) |
| FCI sheath/recycling, neutral, and vorticity closure gates | [`native/fci_sheath_recycling.py`](../src/jax_drb/native/fci_sheath_recycling.py), [`native/fci_neutral.py`](../src/jax_drb/native/fci_neutral.py), [`native/fci_vorticity.py`](../src/jax_drb/native/fci_vorticity.py) | [Stellarator FCI Validation](stellarator_fci_validation.md), [`tests/test_validation_stellarator_fci_campaigns.py`](../tests/test_validation_stellarator_fci_campaigns.py) |
| Imported field-line and surface geometry adapters | [`geometry/essos_import.py`](../src/jax_drb/geometry/essos_import.py), [`geometry/vmec_extender_import.py`](../src/jax_drb/geometry/vmec_extender_import.py), [`validation/essos_imported_fci_campaign.py`](../src/jax_drb/validation/essos_imported_fci_campaign.py), [`validation/vmec_extender_edge_field_campaign.py`](../src/jax_drb/validation/vmec_extender_edge_field_campaign.py) | [ESSOS Field-Line Import](essos_fieldline_import.md), [ESSOS Imported FCI Validation](essos_imported_fci_validation.md), [VMEC Extender Edge Fields](vmec_extender_edge_fields.md) |
| 3D movies, posters, profile analysis, and validation-gallery plots | [`validation/stellarator_sol_showcase.py`](../src/jax_drb/validation/stellarator_sol_showcase.py), [`validation/essos_imported_drb_movie_campaign.py`](../src/jax_drb/validation/essos_imported_drb_movie_campaign.py), [`validation/diverted_tokamak_movie.py`](../src/jax_drb/validation/diverted_tokamak_movie.py), [`validation/publication_plotting.py`](../src/jax_drb/validation/publication_plotting.py) | [Dynamics Gallery](dynamics_gallery.md), [ESSOS Imported DRB Movie](essos_imported_drb_movie.md), [Diverted Tokamak Movie](diverted_tokamak_movie_demo.md), [Validation Gallery](validation_gallery.md) |

## Runtime, I/O, And Parity Artifacts

| Surface | Implementation | Validation or tests |
| --- | --- | --- |
| TOML/input parsing, precision defaults, output manifests, restart metadata, and live progress estimates | [`config`](../src/jax_drb/config), [`runtime`](../src/jax_drb/runtime), [`native/runner.py`](../src/jax_drb/native/runner.py), [`native/recycling_progress.py`](../src/jax_drb/native/recycling_progress.py) | [Inputs And Outputs](input_output_reference.md), [Native Runtime CLI](native_runtime_cli.md), [Restartable Diffusion Tutorial](restartable_diffusion_tutorial.md), [`tests/test_cli_run.py`](../tests/test_cli_run.py), [`tests/test_restartable_diffusion_tutorial.py`](../tests/test_restartable_diffusion_tutorial.py) |
| Portable summaries, array payloads, diff reports, and cached validation baselines | [`parity/portable.py`](../src/jax_drb/parity/portable.py), [`parity/arrays.py`](../src/jax_drb/parity/arrays.py), [`parity/compare.py`](../src/jax_drb/parity/compare.py), [`parity/diff.py`](../src/jax_drb/parity/diff.py) | [Parity Harness](parity_harness.md), [Parity Matrix](parity_matrix.md), [`tests/test_parity_portable.py`](../tests/test_parity_portable.py), [`tests/test_parity_arrays.py`](../tests/test_parity_arrays.py), [`tests/test_parity_diff.py`](../tests/test_parity_diff.py) |
| Self-contained examples, release-backed media, and artifact restoration | [`examples`](../examples), [`scripts/fetch_example_artifacts.py`](../scripts/fetch_example_artifacts.py), [`docs/release_artifacts_manifest.json`](release_artifacts_manifest.json) | [Examples And Artifacts](examples.md), [Example Status Matrix](example_status_matrix.md), [`tests/docs/examples/test_self_contained_example_smoke.py`](../tests/docs/examples/test_self_contained_example_smoke.py), [`tests/test_release_surface.py`](../tests/test_release_surface.py) |

## Current Gaps

The most important open implementation gaps remain deliberately separated from
the validated equations above:

- Full output-window recycling still uses the stable host/SciPy BDF default;
  JAX-linearized and sparse-JVP routes are opt-in until long-window parity and
  runtime gates pass.
- The direct mixed-neutral `NVh` source formulas match written diagnostics at
  roundoff; the remaining visible error is target-band state/history and
  boundary sequencing.
- Multi-device GPU speedup is not a release claim until the device-level
  identity check, the real-kernel parity gate, and a committed timing summary
  all pass.
