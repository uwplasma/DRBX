from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


PUBLIC_RELEASE_FILES = (
    REPO_ROOT / ".readthedocs.yaml",
    REPO_ROOT / "mkdocs.yml",
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "index.md",
    REPO_ROOT / "docs" / "installation.md",
    REPO_ROOT / "docs" / "input_output_reference.md",
    REPO_ROOT / "docs" / "examples.md",
    REPO_ROOT / "docs" / "native_runtime_cli.md",
    REPO_ROOT / "docs" / "restartable_diffusion_tutorial.md",
    REPO_ROOT / "docs" / "validation_gallery.md",
    REPO_ROOT / "docs" / "physics_models.md",
    REPO_ROOT / "docs" / "profiling_runtime.md",
    REPO_ROOT / "docs" / "autodiff_and_scaling_examples.md",
    REPO_ROOT / "docs" / "research_directions.md",
    REPO_ROOT / "docs" / "runtime_gap_remediation.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_scaffold_demo.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_selected_field_demo.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_toroidal_movie_demo.md",
    REPO_ROOT / "docs" / "tokamak_native_selected_field_demo.md",
    REPO_ROOT / "docs" / "traced_field_line_scaffold_demo.md",
    REPO_ROOT / "docs" / "traced_field_line_selected_field_demo.md",
    REPO_ROOT / "docs" / "stellarator_vmec_scaffold_demo.md",
    REPO_ROOT / "docs" / "stellarator_vmec_selected_field_demo.md",
    REPO_ROOT / "docs" / "stellarator_vmec_native_selected_field_demo.md",
    REPO_ROOT / "docs" / "essos_fieldline_import.md",
    REPO_ROOT / "docs" / "essos_imported_fci_validation.md",
    REPO_ROOT / "docs" / "essos_imported_pytree_validation.md",
    REPO_ROOT / "docs" / "essos_imported_drb_movie.md",
    REPO_ROOT / "docs" / "essos_vmec_fieldline_surface.md",
    REPO_ROOT / "docs" / "vmec_extender_edge_fields.md",
    REPO_ROOT / "docs" / "stellarator_examples.md",
    REPO_ROOT / "docs" / "stellarator_fci_validation.md",
    REPO_ROOT / "docs" / "non_axisymmetric_stellarator_sol_plan.md",
    REPO_ROOT / "docs" / "traced_field_line_native_selected_field_demo.md",
    REPO_ROOT / "docs" / "reactions_collisions_campaign.md",
    REPO_ROOT / "docs" / "impurity_radiation_campaign.md",
    REPO_ROOT / "docs" / "controller_feedback_campaign.md",
    REPO_ROOT / "docs" / "temperature_feedback_campaign.md",
    REPO_ROOT / "docs" / "detachment_controller_campaign.md",
    REPO_ROOT / "docs" / "autodiff_diffusion_uncertainty_demo.md",
    REPO_ROOT / "docs" / "closeout_coverage.md",
    REPO_ROOT / "docs" / "release_packaging.md",
    REPO_ROOT / "docs" / "release_notes_1_0_2.md",
    REPO_ROOT / "docs" / "release_notes_1_0_1.md",
    REPO_ROOT / "docs" / "release_notes_1_0_0.md",
    REPO_ROOT / "docs" / "native_3d_runtime_campaign.md",
    REPO_ROOT / "docs" / "native_3d_convergence_campaign.md",
    REPO_ROOT / "docs" / "jax_native_profile_audit.md",
    REPO_ROOT / "docs" / "local_cpu_scaling_campaign.md",
    REPO_ROOT / "docs" / "research_campaigns.md",
    REPO_ROOT / "docs" / "hermes_comparison_gallery.md",
    REPO_ROOT / "docs" / "dynamics_gallery.md",
    REPO_ROOT / "docs" / "hermes_capability_audit.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_validation_methodology.md",
    REPO_ROOT / "docs" / "fluid_1d_mms_convergence.md",
    REPO_ROOT / "docs" / "open_field_operator_campaign.md",
    REPO_ROOT / "docs" / "hermes_live_rerun_campaign.md",
    REPO_ROOT / "docs" / "neutral_parallel_diffusion_campaign.md",
    REPO_ROOT / "docs" / "neutral_mixed_boundary_campaign.md",
    REPO_ROOT / "docs" / "collision_closure_campaign.md",
    REPO_ROOT / "docs" / "tokamak_anomalous_diffusion_campaign.md",
    REPO_ROOT / "docs" / "target_recycling_campaign.md",
    REPO_ROOT / "examples" / "alfven_wave_meeting_demo.py",
    REPO_ROOT / "examples" / "blob2d_meeting_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tcv-x21" / "scaffold_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tcv-x21" / "selected_field_parity_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tcv-x21" / "toroidal_movie_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tokamak-native" / "selected_field_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "traced-field-line" / "scaffold_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "traced-field-line" / "selected_field_parity_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "traced-field-line" / "native_selected_field_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-vmec" / "scaffold_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-vmec" / "selected_field_parity_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-vmec" / "native_selected_field_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-fci" / "geometry_plotting_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-fci" / "linear_mode_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-fci" / "nonlinear_turbulence_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-fci" / "validation_campaign_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "landreman_paul_qa_import.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_fci_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_pytree_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_drb_movie_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "vmec_fieldline_surface_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "vmec-extender" / "imported_field_demo.py",
    REPO_ROOT / "examples" / "engineering" / "native_3d_runtime_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "native_3d_convergence_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "jax_native_profile_audit_demo.py",
    REPO_ROOT / "examples" / "engineering" / "local_cpu_scaling_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "hermes_comparison_summary_demo.py",
    REPO_ROOT / "examples" / "engineering" / "hermes_capability_audit_demo.py",
    REPO_ROOT / "examples" / "engineering" / "fluid_1d_mms_convergence_demo.py",
    REPO_ROOT / "examples" / "engineering" / "open_field_operator_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "hermes_live_rerun_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "neutral_parallel_diffusion_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "neutral_mixed_boundary_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "collision_closure_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "tokamak_anomalous_diffusion_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "target_recycling_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "reactions_collisions_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "impurity_radiation_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "controller_feedback_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "temperature_feedback_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "detachment_controller_campaign_demo.py",
    REPO_ROOT / "examples" / "autodiff_diffusion_uncertainty_demo.py",
    REPO_ROOT / "scripts" / "profile_curated_case.py",
    REPO_ROOT / "scripts" / "profile_recycling_batched_jvp_gate.py",
    REPO_ROOT / "scripts" / "profile_atomic_rate_throughput_gate.py",
    REPO_ROOT / "scripts" / "run_research_campaign_bundle.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_tcv_x21_scaffold.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_tcv_x21_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_tcv_x21_toroidal_movie.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_native_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "traced_field_line_scaffold.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "traced_field_line_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "traced_field_line_native_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "stellarator_vmec_scaffold.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "stellarator_vmec_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "stellarator_vmec_native_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "geometry" / "essos_import.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "essos_fieldline_import_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "essos_imported_fci_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "essos_imported_pytree_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "essos_imported_drb_movie_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "essos_vmec_fieldline_surface_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "geometry" / "vmec_extender_import.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "vmec_extender_edge_field_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "vmec_extender_sol_smoke_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "native_3d_runtime_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "native_3d_convergence_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "jax_native_profile_audit.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "local_cpu_scaling_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "recycling_batched_jvp_profile.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "hermes_comparison_summary.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "hermes_capability_audit.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "fluid_1d_mms_convergence.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "open_field_operator_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "hermes_live_rerun_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "neutral_parallel_diffusion_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "neutral_mixed_boundary_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "collision_closure_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_anomalous_diffusion_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "target_recycling_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "reactions_collisions_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "impurity_radiation_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "controller_feedback_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "temperature_feedback_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "detachment_controller_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "autodiff_diffusion_uncertainty.py",
    REPO_ROOT / ".github" / "workflows" / "coverage.yml",
    REPO_ROOT / ".github" / "workflows" / "docs.yml",
    REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml",
    REPO_ROOT / ".github" / "workflows" / "research-campaigns.yml",
    REPO_ROOT / ".github" / "workflows" / "test.yml",
)

PUBLIC_RUN_LOGS = (
    REPO_ROOT / "docs" / "data" / "restartable_diffusion_demo_artifacts" / "run_first" / "restartable_diffusion_run_log.json",
    REPO_ROOT / "docs" / "data" / "restartable_diffusion_demo_artifacts" / "run_full" / "restartable_diffusion_full_run_log.json",
    REPO_ROOT / "docs" / "data" / "restartable_diffusion_demo_artifacts" / "run_resumed" / "restartable_diffusion_resumed_run_log.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_scaffold_artifacts" / "data" / "tokamak_tcv_x21_scaffold_manifest.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_scaffold_artifacts" / "data" / "traced_field_line_scaffold_manifest.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_manifest.json",
)

PUBLIC_JSON_ARTIFACTS = (
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_scaffold_artifacts" / "data" / "tokamak_tcv_x21_scaffold_input_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_scaffold_artifacts" / "data" / "tokamak_tcv_x21_scaffold_benchmark_data_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_scaffold_artifacts" / "data" / "tokamak_tcv_x21_scaffold_validation_contract.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_scaffold_artifacts" / "data" / "tokamak_tcv_x21_scaffold_observable_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_selected_field_artifacts" / "data" / "tokamak_tcv_x21_selected_field_parity.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_selected_field_artifacts" / "data" / "tokamak_tcv_x21_selected_field_parity_observable_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_selected_field_artifacts" / "data" / "tokamak_tcv_x21_selected_field_parity_benchmark_data_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_toroidal_movie_artifacts" / "data" / "tokamak_tcv_x21_toroidal_summary.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_artifacts" / "data" / "tokamak_native_selected_field.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_artifacts" / "data" / "tokamak_native_selected_field_comparison.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_artifacts" / "data" / "tokamak_native_selected_field_observable_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_artifacts" / "data" / "tokamak_native_selected_field_runtime_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_short_window_artifacts" / "data" / "tokamak_native_selected_field_short_window.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_short_window_artifacts" / "data" / "tokamak_native_selected_field_short_window_comparison.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_short_window_artifacts" / "data" / "tokamak_native_selected_field_short_window_observable_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_short_window_artifacts" / "data" / "tokamak_native_selected_field_short_window_runtime_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_scaffold_artifacts" / "data" / "traced_field_line_scaffold_input_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_scaffold_artifacts" / "data" / "traced_field_line_scaffold_validation_contract.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_scaffold_artifacts" / "data" / "traced_field_line_scaffold_observable_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_scaffold_artifacts" / "data" / "traced_field_line_scaffold_line_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_scaffold_artifacts" / "data" / "traced_field_line_scaffold_slice_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_selected_field_artifacts" / "data" / "traced_field_line_selected_field_parity_observable_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_selected_field_artifacts" / "data" / "traced_field_line_selected_field_parity.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_selected_field_artifacts" / "data" / "traced_field_line_selected_field_parity_source_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_native_selected_field_artifacts" / "data" / "traced_field_line_native_selected_field.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_native_selected_field_artifacts" / "data" / "traced_field_line_native_selected_field_comparison.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_native_selected_field_artifacts" / "data" / "traced_field_line_native_selected_field_observable_report.json",
    REPO_ROOT / "docs" / "data" / "traced_field_line_native_selected_field_artifacts" / "data" / "traced_field_line_native_selected_field_runtime_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_input_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_validation_contract.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_profile_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_surface_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_observable_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_selected_field_artifacts" / "data" / "stellarator_vmec_selected_field_parity.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_selected_field_artifacts" / "data" / "stellarator_vmec_selected_field_parity_observable_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_selected_field_artifacts" / "data" / "stellarator_vmec_selected_field_parity_source_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_native_selected_field_artifacts" / "data" / "stellarator_vmec_native_selected_field.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_native_selected_field_artifacts" / "data" / "stellarator_vmec_native_selected_field_comparison.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_native_selected_field_artifacts" / "data" / "stellarator_vmec_native_selected_field_observable_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_native_selected_field_artifacts" / "data" / "stellarator_vmec_native_selected_field_runtime_report.json",
    REPO_ROOT / "docs" / "data" / "essos_fieldline_import_artifacts" / "data" / "essos_landreman_paul_qa_fieldline_import.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_fci_artifacts" / "data" / "essos_imported_fci_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_fci_vmec_artifacts" / "data" / "essos_imported_fci_vmec_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_fci_hybrid_artifacts" / "data" / "essos_imported_fci_hybrid_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_pytree_artifacts" / "data" / "essos_imported_pytree_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_pytree_vmec_artifacts" / "data" / "essos_imported_pytree_vmec_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_pytree_hybrid_artifacts" / "data" / "essos_imported_pytree_hybrid_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_drb_movie_artifacts" / "data" / "essos_imported_drb_movie_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_imported_drb_movie_hybrid_artifacts" / "data" / "essos_imported_drb_movie_hybrid_campaign.json",
    REPO_ROOT / "docs" / "data" / "essos_vmec_fieldline_surface_artifacts" / "data" / "essos_vmec_fieldline_surface_campaign.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "essos_vmec_equilibrium_fieldline_surface_artifacts"
    / "data"
    / "essos_vmec_equilibrium_fieldline_surface_campaign.json",
    REPO_ROOT / "docs" / "data" / "native_3d_runtime_campaign_artifacts" / "data" / "native_3d_runtime_campaign.json",
    REPO_ROOT / "docs" / "data" / "native_3d_convergence_campaign_artifacts" / "data" / "native_3d_convergence_campaign.json",
    REPO_ROOT / "docs" / "data" / "jax_native_profile_audit_artifacts" / "data" / "jax_native_profile_audit.json",
    REPO_ROOT / "docs" / "data" / "local_cpu_scaling_campaign_artifacts" / "data" / "local_cpu_scaling_campaign.json",
    REPO_ROOT / "docs" / "data" / "hermes_comparison_summary_artifacts" / "data" / "hermes_comparison_summary.json",
    REPO_ROOT / "docs" / "data" / "hermes_capability_audit.json",
    REPO_ROOT / "docs" / "data" / "fluid_1d_mms_convergence_artifacts" / "data" / "fluid_1d_mms_convergence.json",
    REPO_ROOT / "docs" / "data" / "open_field_operator_campaign_artifacts" / "data" / "open_field_operator_campaign.json",
    REPO_ROOT / "docs" / "data" / "hermes_live_rerun_campaign_artifacts" / "data" / "hermes_live_rerun_campaign.json",
    REPO_ROOT / "docs" / "data" / "neutral_parallel_diffusion_campaign_artifacts" / "data" / "neutral_parallel_diffusion_campaign.json",
    REPO_ROOT / "docs" / "data" / "neutral_mixed_boundary_campaign_artifacts" / "data" / "neutral_mixed_boundary_campaign.json",
    REPO_ROOT / "docs" / "data" / "collision_closure_campaign_artifacts" / "data" / "collision_closure_campaign.json",
    REPO_ROOT / "docs" / "data" / "tokamak_anomalous_diffusion_campaign_artifacts" / "data" / "tokamak_anomalous_diffusion_campaign.json",
    REPO_ROOT / "docs" / "data" / "target_recycling_campaign_artifacts" / "data" / "target_recycling_campaign.json",
    REPO_ROOT / "docs" / "data" / "reactions_collisions_campaign_artifacts" / "data" / "reactions_collisions_campaign.json",
    REPO_ROOT / "docs" / "data" / "impurity_radiation_campaign_artifacts" / "data" / "impurity_radiation_campaign.json",
    REPO_ROOT / "docs" / "data" / "controller_feedback_campaign_artifacts" / "data" / "controller_feedback_campaign.json",
    REPO_ROOT / "docs" / "data" / "temperature_feedback_campaign_artifacts" / "data" / "temperature_feedback_campaign.json",
    REPO_ROOT / "docs" / "data" / "detachment_controller_campaign_artifacts" / "data" / "detachment_controller_campaign.json",
    REPO_ROOT / "docs" / "data" / "autodiff_diffusion_uncertainty_artifacts" / "data" / "autodiff_diffusion_uncertainty_analysis.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "recycling_1d_jax_linearized_gate" / "profile_summary.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "recycling_dthe_one_step" / "profile_summary.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "recycling_dthe_jax_linearized_gate" / "profile_summary.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "recycling_dthe_jax_linearized_gate_gpu" / "profile_summary.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "recycling_dthe_jax_linearized_gate_gpu_warm" / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_jax_linearized_gate_ny100_dt1e4_cpu"
    / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_jax_linearized_gate_ny100_dt1e4_gpu"
    / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_jax_linearized_gate_ny200_dt1e4_cpu"
    / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_jax_linearized_gate_ny200_dt1e4_gpu"
    / "profile_summary.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "recycling_dthe_batched_jvp_gate_cpu" / "profile_summary.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "atomic_rate_throughput_gate_cpu" / "profile_summary.json",
    REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / "atomic_rate_throughput_gate_gpu" / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "stellarator_fci_validation_artifacts"
    / "pytree_drb"
    / "data"
    / "stellarator_drb_pytree_gpu_profile_summary.json",
)

SIMSOPT_STYLE_EXAMPLES = tuple(
    path for path in PUBLIC_RELEASE_FILES if "examples/geometry-3D/" in path.as_posix()
) + (
    REPO_ROOT / "examples" / "autodiff_diffusion_uncertainty_demo.py",
    REPO_ROOT / "examples" / "diverted_tokamak_movie_demo.py",
)


def test_public_release_surface_avoids_local_path_leaks() -> None:
    forbidden = ("/Users/", "local/hermes", "local/jax_drb")
    for path in PUBLIC_RELEASE_FILES:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


def test_public_docs_pages_avoid_local_path_leaks() -> None:
    forbidden = ("/Users/", "local/hermes", "local/jax_drb")
    for path in sorted((REPO_ROOT / "docs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


def test_public_docs_data_text_artifacts_avoid_local_path_leaks() -> None:
    forbidden = ("/Users/", "local/hermes", "local/jax_drb")
    text_suffixes = {".json", ".md", ".txt", ".toml", ".yaml", ".yml"}
    for path in sorted((REPO_ROOT / "docs" / "data").rglob("*")):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


def test_readthedocs_configuration_points_to_mkdocs_site() -> None:
    rtd_config = (REPO_ROOT / ".readthedocs.yaml").read_text(encoding="utf-8")
    mkdocs_config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "configuration: mkdocs.yml" in rtd_config
    assert 'python: "3.12"' in rtd_config
    assert "site_url: https://jax-drb.readthedocs.io/" in mkdocs_config
    assert "Installation: installation.md" in mkdocs_config
    assert "Inputs And Outputs: input_output_reference.md" in mkdocs_config
    assert "Examples And Artifacts: examples.md" in mkdocs_config


def test_simsopt_style_examples_have_top_level_parameters() -> None:
    forbidden_patterns = (
        re.compile(r"\bimport argparse\b"),
        re.compile(r"\bdef main\("),
        re.compile(r"if __name__ == [\"']__main__[\"']"),
    )
    for path in SIMSOPT_STYLE_EXAMPLES:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern.search(text) is None, f"{path} is not a top-level parameter script"


def test_public_release_surface_avoids_legacy_branding_in_user_docs() -> None:
    forbidden = ("Hermes-style", "Hermes-3 input deck", "BOUT++")
    for path in PUBLIC_RELEASE_FILES:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


def test_committed_demo_run_logs_use_sanitized_paths() -> None:
    for path in PUBLIC_RUN_LOGS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload, sort_keys=True)
        assert "/Users/" not in text
        if "run_configuration" in payload:
            assert payload["run_configuration"]["runtime"]["compilation_cache_dir"].startswith("~/")
            continue
        if "workdir" in payload:
            assert not str(payload["workdir"]).startswith("/")
        if "mesh_path" in payload:
            assert not str(payload["mesh_path"]).startswith("/")


def test_public_json_artifacts_use_sanitized_paths() -> None:
    for path in PUBLIC_JSON_ARTIFACTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload, sort_keys=True)
        assert "/Users/" not in text
