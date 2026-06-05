# Example Status Matrix

This page records what each public example is expected to do from a fresh
checkout. The goal is to separate self-contained user examples from heavier
developer validation commands, while keeping the README and docs reproducible.

## Setup

Most examples use release-backed arrays, figures, and movies instead of large
binary files in git history. Restore those artifacts first:

```bash
python scripts/fetch_example_artifacts.py --skip-baselines
```

Use the full artifact restore when running cached parity or baseline checks:

```bash
python scripts/fetch_example_artifacts.py
```

The self-contained examples below use JAXDRB source code plus release-backed
artifacts. They do not require users to install or run external plasma codes.
Developer/live-reference examples are kept in the tree because they regenerate
publication packages, but they require explicit local reference inputs.

## Self-Contained User Examples

| Example | Status | Main output | Notes |
| --- | --- | --- | --- |
| [`examples/restartable_diffusion_tutorial.py`](../examples/restartable_diffusion_tutorial.py) | self-contained | restart logs, NPZ output, PNG/GIF media | Exercises runtime output, restart, live progress metadata, and plotting. |
| [`examples/autodiff_diffusion_sensitivity_demo.py`](../examples/autodiff_diffusion_sensitivity_demo.py) | self-contained | sensitivity JSON/PNG | Demonstrates `jax.grad` against finite differences. |
| [`examples/autodiff_diffusion_uncertainty_demo.py`](../examples/autodiff_diffusion_uncertainty_demo.py) | self-contained | uncertainty JSON/PNG | Demonstrates covariance pushforward and vectorized Monte Carlo. |
| [`examples/autodiff_diffusion_inverse_design_demo.py`](../examples/autodiff_diffusion_inverse_design_demo.py) | self-contained | optimization JSON/PNG | Demonstrates gradient-based inverse design. |
| [`examples/strong_scaling_diffusion_demo.py`](../examples/strong_scaling_diffusion_demo.py) | self-contained | scaling JSON/PNG | Supports CPU process-group, CPU host-device, and optional GPU modes. |
| [`examples/diverted_tokamak_movie_demo.py`](../examples/diverted_tokamak_movie_demo.py) | self-contained after artifact restore | diverted tokamak GIF/poster/snapshots | Uses release-backed arrays so users do not need a local reference run. |
| [`examples/diverted_tokamak_profile_analysis_demo.py`](../examples/diverted_tokamak_profile_analysis_demo.py) | self-contained after artifact restore | target/profile/time-trace PNG | Analyzes the same diverted tokamak output used by the movie. |
| [`examples/geometry-3D/stellarator-fci/geometry_plotting_demo.py`](../examples/geometry-3D/stellarator-fci/geometry_plotting_demo.py) | self-contained | geometry, metric, connection-length figure | Uses the native synthetic FCI geometry lane. |
| [`examples/geometry-3D/stellarator-fci/linear_mode_demo.py`](../examples/geometry-3D/stellarator-fci/linear_mode_demo.py) | self-contained | linear-mode diagnostics and snapshots | Compact linear 3D stellarator workflow. |
| [`examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py`](../examples/geometry-3D/stellarator-fci/nonlinear_turbulence_demo.py) | self-contained | nonlinear diagnostics, poster, GIF | Compact reduced nonlinear SOL turbulence demonstration. |
| [`examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py`](../examples/geometry-3D/stellarator-fci/turbulent_profile_analysis_demo.py) | self-contained after nonlinear demo output | profile-analysis PNG | Computes radial fluctuation, RMS, transport-proxy, connection-length, and energy-trace diagnostics. |

## Validation And Publication Campaign Examples

These examples regenerate validation packages used by the docs and paper. They
are still useful to users, but they are not all meant to run in seconds.

| Family | Example | Status | Documentation |
| --- | --- | --- | --- |
| MMS and operators | [`examples/engineering/fluid_1d_mms_convergence_demo.py`](../examples/engineering/fluid_1d_mms_convergence_demo.py), [`examples/engineering/open_field_operator_campaign_demo.py`](../examples/engineering/open_field_operator_campaign_demo.py) | self-contained validation | [Fluid 1D MMS Convergence](fluid_1d_mms_convergence.md), [Open-Field Operator Campaign](open_field_operator_campaign.md) |
| Atomic and collision closures | [`examples/engineering/atomic_rate_differentiability_campaign_demo.py`](../examples/engineering/atomic_rate_differentiability_campaign_demo.py), [`examples/engineering/reactions_collisions_campaign_demo.py`](../examples/engineering/reactions_collisions_campaign_demo.py), [`examples/engineering/collision_closure_campaign_demo.py`](../examples/engineering/collision_closure_campaign_demo.py) | self-contained validation | [Atomic Rate Differentiability](atomic_rate_differentiability_campaign.md), [Reactions And Collisions](reactions_collisions_campaign.md), [Collision Closure](collision_closure_campaign.md) |
| Neutrals and target recycling | [`examples/engineering/neutral_parallel_diffusion_campaign_demo.py`](../examples/engineering/neutral_parallel_diffusion_campaign_demo.py), [`examples/engineering/neutral_mixed_boundary_campaign_demo.py`](../examples/engineering/neutral_mixed_boundary_campaign_demo.py), [`examples/engineering/neutral_mixed_term_balance_campaign_demo.py`](../examples/engineering/neutral_mixed_term_balance_campaign_demo.py), [`examples/engineering/target_recycling_campaign_demo.py`](../examples/engineering/target_recycling_campaign_demo.py) | self-contained or cached-reference validation | [Neutral Parallel Diffusion](neutral_parallel_diffusion_campaign.md), [Neutral Mixed Boundary](neutral_mixed_boundary_campaign.md), [Neutral Mixed Term Balance](neutral_mixed_term_balance_campaign.md), [Target Recycling](target_recycling_campaign.md) |
| Control and radiation | [`examples/engineering/controller_feedback_campaign_demo.py`](../examples/engineering/controller_feedback_campaign_demo.py), [`examples/engineering/temperature_feedback_campaign_demo.py`](../examples/engineering/temperature_feedback_campaign_demo.py), [`examples/engineering/detachment_controller_campaign_demo.py`](../examples/engineering/detachment_controller_campaign_demo.py), [`examples/engineering/impurity_radiation_campaign_demo.py`](../examples/engineering/impurity_radiation_campaign_demo.py) | self-contained validation | [Controller Feedback](controller_feedback_campaign.md), [Temperature Feedback](temperature_feedback_campaign.md), [Detachment Controller](detachment_controller_campaign.md), [Impurity Radiation](impurity_radiation_campaign.md) |
| Performance | [`examples/engineering/local_cpu_scaling_campaign_demo.py`](../examples/engineering/local_cpu_scaling_campaign_demo.py), [`examples/engineering/native_3d_runtime_campaign_demo.py`](../examples/engineering/native_3d_runtime_campaign_demo.py), [`examples/engineering/jax_native_profile_audit_demo.py`](../examples/engineering/jax_native_profile_audit_demo.py) | local profiling validation | [Local CPU Scaling](local_cpu_scaling_campaign.md), [Native 3D Runtime](native_3d_runtime_campaign.md), [JAX Native Profile Audit](jax_native_profile_audit.md) |
| Reference comparison | [`examples/engineering/hermes_comparison_summary_demo.py`](../examples/engineering/hermes_comparison_summary_demo.py), [`examples/engineering/hermes_live_rerun_campaign_demo.py`](../examples/engineering/hermes_live_rerun_campaign_demo.py), [`examples/engineering/hermes_capability_audit_demo.py`](../examples/engineering/hermes_capability_audit_demo.py) | developer/live-reference | [Hermes Comparison Gallery](hermes_comparison_gallery.md), [Hermes Live Rerun Campaign](hermes_live_rerun_campaign.md), [Hermes Capability Audit](hermes_capability_audit.md) |

## Imported Geometry Examples

Imported-geometry examples are split into published artifact use and developer
regeneration. Users can view restored figures and movies after
`fetch_example_artifacts.py --skip-baselines`; rerunning the import adapters
requires the local geometry source named by each page.

| Example | Status | Main output | Documentation |
| --- | --- | --- | --- |
| [`examples/geometry-3D/vmec-extender/imported_field_demo.py`](../examples/geometry-3D/vmec-extender/imported_field_demo.py) | developer regeneration | imported edge-field arrays and SOL smoke diagnostics | [VMEC Extender Edge Fields](vmec_extender_edge_fields.md) |
| [`examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py`](../examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py) | developer regeneration | field-line import JSON/PNG | [ESSOS Field-Line Import](essos_fieldline_import.md) |
| [`examples/geometry-3D/essos-field-lines/imported_fci_campaign.py`](../examples/geometry-3D/essos-field-lines/imported_fci_campaign.py) | developer regeneration | imported FCI validation package | [ESSOS Imported FCI Validation](essos_imported_fci_validation.md) |
| [`examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py`](../examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py) | developer regeneration | imported fixed-layout PyTree/JVP gate | [ESSOS Imported PyTree Validation](essos_imported_pytree_validation.md) |
| [`examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py`](../examples/geometry-3D/essos-field-lines/imported_drb_movie_campaign.py) | developer regeneration | imported QA DRB movie, poster, snapshots, diagnostics | [ESSOS Imported DRB Movie](essos_imported_drb_movie.md) |
| [`examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py`](../examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py) | developer regeneration | field-line/surface registration and Poincare diagnostics | [ESSOS VMEC Field-Line Surface](essos_vmec_fieldline_surface.md) |

## Profiling Commands

The source-kernel GPU profiler is self-contained:

```bash
PYTHONPATH=src python scripts/profile_atomic_rate_throughput_gate.py \
  --output-dir tmp/profiles/atomic_rate_throughput_gate
```

The fixed-layout D/T/He recycling residual/JVP profiler is also self-contained
by default through the fixture decks:

```bash
PYTHONPATH=src python scripts/profile_recycling_batched_jvp_gate.py \
  --case dthe \
  --rhs-backend fixed_full_field_array \
  --override mesh:ny=100 \
  --batch-sizes 1,4,16,64 \
  --timed-runs 3 \
  --output-dir tmp/profiles/recycling_dthe_batched_jvp_gate
```

Pass `--reference-root` for a full local reference suite or `--input-path` for
a single staged deck. The default fixture path is intended for reproducibility
and regression testing; full reference campaigns remain developer validation.
`fixed_full_field_array` is the default release-facing backend;
`--rhs-backend active_array` is an opt-in migration seam for active-field RHS
kernels, and `--rhs-backend host_bridge` remains available only for
bridge-comparison tasks.

## Maintenance Gates

These commands keep the matrix honest:

```bash
PYTHONPATH=src pytest -q tests/docs/examples/test_self_contained_example_smoke.py
PYTHONPATH=src pytest -q tests/test_release_surface.py
mkdocs build --strict --clean
```
