from __future__ import annotations

import json
import re
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


PUBLIC_RELEASE_FILES = (
    REPO_ROOT / "CITATION.cff",
    REPO_ROOT / "MANIFEST.in",
    REPO_ROOT / ".readthedocs.yaml",
    REPO_ROOT / "mkdocs.yml",
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "index.md",
    REPO_ROOT / "docs" / "installation.md",
    REPO_ROOT / "docs" / "input_output_reference.md",
    REPO_ROOT / "docs" / "examples.md",
    REPO_ROOT / "docs" / "example_status_matrix.md",
    REPO_ROOT / "docs" / "native_runtime_cli.md",
    REPO_ROOT / "docs" / "restartable_diffusion_tutorial.md",
    REPO_ROOT / "docs" / "validation_gallery.md",
    REPO_ROOT / "docs" / "physics_models.md",
    REPO_ROOT / "docs" / "equation_to_code_map.md",
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
    REPO_ROOT / "docs" / "connection_length.md",
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
    REPO_ROOT / "docs" / "release_notes_2_0_0_dev0.md",
    REPO_ROOT / "docs" / "release_notes_1_0_3.md",
    REPO_ROOT / "docs" / "release_notes_1_0_2.md",
    REPO_ROOT / "docs" / "release_notes_1_0_1.md",
    REPO_ROOT / "docs" / "release_notes_1_0_0.md",
    REPO_ROOT / "docs" / "native_3d_runtime_campaign.md",
    REPO_ROOT / "docs" / "native_3d_convergence_campaign.md",
    REPO_ROOT / "docs" / "local_cpu_scaling_campaign.md",
    REPO_ROOT / "docs" / "research_campaigns.md",
    REPO_ROOT / "docs" / "dynamics_gallery.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_validation_methodology.md",
    REPO_ROOT / "docs" / "fluid_1d_mms_convergence.md",
    REPO_ROOT / "docs" / "open_field_operator_campaign.md",
    REPO_ROOT / "docs" / "neutral_parallel_diffusion_campaign.md",
    REPO_ROOT / "docs" / "neutral_mixed_boundary_campaign.md",
    REPO_ROOT / "docs" / "neutral_mixed_term_balance_campaign.md",
    REPO_ROOT / "docs" / "collision_closure_campaign.md",
    REPO_ROOT / "docs" / "tokamak_anomalous_diffusion_campaign.md",
    REPO_ROOT / "docs" / "target_recycling_campaign.md",
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
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-fci" / "turbulent_profile_analysis_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-fci" / "validation_campaign_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "landreman_paul_qa_import.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "direct_coil_open_sol_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "hybrid_open_sol_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_fci_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_connection_length_refinement_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_pytree_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_drb_movie_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "vmec_fieldline_surface_campaign.py",
    REPO_ROOT / "examples" / "geometry-3D" / "vmec-extender" / "imported_field_demo.py",
    REPO_ROOT / "examples" / "engineering" / "native_3d_runtime_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "native_3d_convergence_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "local_cpu_scaling_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "fluid_1d_mms_convergence_demo.py",
    REPO_ROOT / "examples" / "engineering" / "open_field_operator_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "neutral_parallel_diffusion_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "neutral_mixed_boundary_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "neutral_mixed_term_balance_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "collision_closure_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "tokamak_anomalous_diffusion_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "target_recycling_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "reactions_collisions_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "impurity_radiation_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "controller_feedback_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "temperature_feedback_campaign_demo.py",
    REPO_ROOT / "examples" / "engineering" / "detachment_controller_campaign_demo.py",
    REPO_ROOT / "examples" / "autodiff_diffusion_uncertainty_demo.py",
    REPO_ROOT / "examples" / "diverted_tokamak_profile_analysis_demo.py",
    REPO_ROOT / "scripts" / "profile_curated_case.py",
    REPO_ROOT / "scripts" / "profile_recycling_batched_jvp_gate.py",
    REPO_ROOT / "scripts" / "profile_atomic_rate_throughput_gate.py",
    REPO_ROOT / "scripts" / "run_research_campaign_bundle.py",
    REPO_ROOT / "scripts" / "fetch_example_artifacts.py",
    REPO_ROOT / "scripts" / "audit_release_readiness.py",
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
    REPO_ROOT / "src" / "jax_drb" / "validation" / "local_cpu_scaling_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "recycling_batched_jvp_profile.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "fluid_1d_mms_convergence.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "open_field_operator_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "neutral_parallel_diffusion_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "neutral_mixed_boundary_campaign.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "neutral_mixed_term_balance_campaign.py",
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
    REPO_ROOT
    / "docs"
    / "data"
    / "neutral_mixed_term_balance_campaign_artifacts"
    / "data"
    / "neutral_mixed_term_balance_campaign.json",
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
    / "neutral_mixed_substep_hybrid_artifacts"
    / "data"
    / "neutral_mixed_substep_hybrid.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_1d_adaptive_bdf_jax_lineax_gate"
    / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_adaptive_bdf_trace_probe"
    / "profile_summary.json",
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
    / "recycling_dthe_jax_linearized_gate_ny100_dt1e4_lineax_cpu"
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

COMMITTED_GPU_PROFILE_SUMMARIES = (
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_jax_linearized_gate_gpu_current"
    / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_jax_linearized_gate_gpu"
    / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "recycling_dthe_jax_linearized_gate_gpu_warm"
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
    / "recycling_dthe_jax_linearized_gate_ny200_dt1e4_gpu"
    / "profile_summary.json",
    REPO_ROOT
    / "docs"
    / "data"
    / "runtime_profile_artifacts"
    / "atomic_rate_throughput_gate_gpu"
    / "profile_summary.json",
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
    REPO_ROOT / "examples" / "diverted_tokamak_profile_analysis_demo.py",
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


def test_user_examples_are_self_contained_by_default() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    examples_doc = (REPO_ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    diverted_demo = (REPO_ROOT / "examples" / "diverted_tokamak_movie_demo.py").read_text(
        encoding="utf-8"
    )
    normalized_readme = " ".join(readme.split())

    assert "Users do not need to install or download any" in normalized_readme
    assert "external plasma code to run those examples or the README/docs" in normalized_readme
    assert "Live reference-code reruns are developer validation tasks only" in normalized_readme
    assert "do not require users" in examples_doc
    assert "to download external plasma codes" in examples_doc
    assert "developer/live-reference" in examples_doc
    assert "from jax_drb.reference.paths import default_reference_root" not in diverted_demo
    assert "REFERENCE_ROOT: Path | None = None" in diverted_demo


def test_direct_coil_open_sol_default_contract_includes_media_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    runpy.run_path(
        str(
            REPO_ROOT
            / "examples"
            / "geometry-3D"
            / "essos-field-lines"
            / "direct_coil_open_sol_demo.py"
        )
    )

    summary_path = (
        tmp_path
        / "artifacts"
        / "essos_direct_coil_open_sol"
        / "data"
        / "essos_direct_coil_open_sol_workflow_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stage_by_name = {stage["stage"]: stage for stage in summary["stage_reports"]}

    assert summary["map_source"] == "coil"
    assert summary["settings"]["run_live_media_gate"] is False
    assert summary["settings"]["run_live_collocated_endpoint_label_refinement_gate"] is False
    assert (
        summary["settings"][
            "run_live_boundary_resolved_endpoint_label_refinement_gate"
        ]
        is False
    )
    assert summary["promotion_ready"] is False
    assert summary["near_term_closeout_status"] == "finalized_diagnostic_contract"
    assert "promoted_pure_coil_open_sol_movie" in summary["deferred_claims"]
    assert "finite_beta_vmec_extender_open_sol" in summary["deferred_claims"]
    assert "no_live_promotion_gates_ran" in summary["promotion_rejection_reasons"]
    assert any(
        blocker["stage"] == "direct_coil_diagnostic_turbulence_media"
        for blocker in summary["promotion_blocking_stages"]
    )
    assert any(
        "Set RUN_LIVE_MEDIA_GATE=True" in action
        for action in summary["next_actions"]
    )
    assert stage_by_name["direct_coil_source_profile_gate"]["status"] == "contract_only"
    assert "target-label, heat-load" in stage_by_name[
        "direct_coil_source_profile_gate"
    ]["next_action"]
    assert (
        stage_by_name["direct_coil_collocated_endpoint_label_refinement_gate"]["status"]
        == "diagnostic"
    )
    assert "non-collocated even-ratio seed grids" in stage_by_name[
        "direct_coil_collocated_endpoint_label_refinement_gate"
    ]["next_action"]
    assert (
        stage_by_name[
            "direct_coil_boundary_resolved_endpoint_label_refinement_gate"
        ]["status"]
        == "diagnostic"
    )
    assert "(7, 15, 27) -> (11, 25, 45)" in stage_by_name[
        "direct_coil_boundary_resolved_endpoint_label_refinement_gate"
    ]["next_action"]
    assert "boundary-excluded cells" in stage_by_name[
        "direct_coil_boundary_resolved_endpoint_label_refinement_gate"
    ]["next_action"]
    assert stage_by_name["direct_coil_diagnostic_turbulence_media"]["status"] == "skipped"
    assert "Set RUN_LIVE_MEDIA_GATE=True" in stage_by_name[
        "direct_coil_diagnostic_turbulence_media"
    ]["next_action"]


def test_direct_coil_open_sol_partial_live_stages_do_not_promote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    module = runpy.run_path(
        str(
            REPO_ROOT
            / "examples"
            / "geometry-3D"
            / "essos-field-lines"
            / "direct_coil_open_sol_demo.py"
        )
    )
    settings = module["build_settings"](
        output_root=tmp_path / "partial_live",
        case_label="partial_live",
        run_live_fci_gate=True,
    )
    summary_path = module["write_workflow_summary"](
        settings,
        [
            {
                "stage": "direct_coil_fci_endpoint_source_gate",
                "status": "ran",
                "promotion_ready": True,
            },
            {
                "stage": "direct_coil_source_profile_gate",
                "status": "ran",
                "promotion_ready": True,
            },
            {
                "stage": "direct_coil_endpoint_label_refinement_gate",
                "status": "skipped",
                "promotion_ready": False,
                "next_action": "run endpoint labels",
            },
            {
                "stage": "direct_coil_target_exit_length_refinement_gate",
                "status": "diagnostic",
                "promotion_ready": False,
                "next_action": "run target-exit diagnostics",
            },
        ],
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["promotion_ready"] is False
    assert summary["near_term_closeout_status"] == "live_evidence_incomplete"
    assert summary["promotion_rejection_reasons"] == [
        "skipped_stage_not_live_promotion_evidence"
    ]
    assert [
        blocker["stage"] for blocker in summary["promotion_blocking_stages"]
    ] == ["direct_coil_endpoint_label_refinement_gate"]
    assert [
        stage["stage"] for stage in summary["diagnostic_stages"]
    ] == ["direct_coil_target_exit_length_refinement_gate"]


def test_hybrid_open_sol_default_contract_includes_promotion_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    runpy.run_path(
        str(
            REPO_ROOT
            / "examples"
            / "geometry-3D"
            / "essos-field-lines"
            / "hybrid_open_sol_demo.py"
        )
    )

    summary_path = (
        tmp_path
        / "artifacts"
        / "essos_hybrid_open_sol"
        / "data"
        / "essos_hybrid_open_sol_workflow_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stage_by_name = {stage["stage"]: stage for stage in summary["stage_reports"]}

    assert summary["map_source"] == "hybrid"
    assert summary["connection_quantity"] == "parallel_step_per_toroidal_radian"
    assert summary["settings"]["run_live_media_gate"] is False
    assert summary["settings"]["run_release_evidence_audit"] is True
    assert summary["release_evidence_ready"] is True
    assert summary["release_evidence_stage_count"] == 1
    assert summary["promotion_ready"] is False
    assert (
        summary["near_term_closeout_status"]
        == "release_backed_compact_vacuum_bridge_ready"
    )
    assert "live_regenerated_hybrid_promotion_bundle" in summary["deferred_claims"]
    assert "finite_beta_vmec_extender_open_sol" in summary["deferred_claims"]
    assert "no_live_promotion_gates_ran" in summary["promotion_rejection_reasons"]
    assert any(
        blocker["stage"] == "hybrid_diagnostic_turbulence_media"
        for blocker in summary["promotion_blocking_stages"]
    )
    assert any(
        "Set RUN_LIVE_MEDIA_GATE=True" in action
        for action in summary["next_actions"]
    )
    assert stage_by_name["hybrid_fci_endpoint_source_gate"]["status"] == "contract_only"
    assert stage_by_name["hybrid_source_profile_gate"]["status"] == "contract_only"
    assert "target-label, heat-load" in stage_by_name[
        "hybrid_source_profile_gate"
    ]["next_action"]
    assert stage_by_name["hybrid_parallel_step_refinement_gate"]["status"] == "skipped"
    assert stage_by_name["hybrid_movie_grid_time_refinement_gate"]["status"] == "skipped"
    assert stage_by_name["hybrid_diagnostic_turbulence_media"]["status"] == "skipped"
    release_audit = stage_by_name["hybrid_release_evidence_audit"]
    assert release_audit["status"] == "release_evidence"
    assert release_audit["promotion_ready"] is True
    assert release_audit["promotion_rejection_reasons"] == []
    assert Path(release_audit["report_json_path"]).exists()


def test_hybrid_open_sol_partial_live_stages_do_not_promote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    module = runpy.run_path(
        str(
            REPO_ROOT
            / "examples"
            / "geometry-3D"
            / "essos-field-lines"
            / "hybrid_open_sol_demo.py"
        )
    )
    settings = module["build_settings"](
        output_root=tmp_path / "partial_hybrid_live",
        case_label="partial_hybrid_live",
        run_live_fci_gate=True,
    )
    summary_path = module["write_workflow_summary"](
        settings,
        [
            {
                "stage": "hybrid_fci_endpoint_source_gate",
                "status": "ran",
                "promotion_ready": True,
            },
            {
                "stage": "hybrid_source_profile_gate",
                "status": "ran",
                "promotion_ready": True,
            },
            {
                "stage": "hybrid_parallel_step_refinement_gate",
                "status": "skipped",
                "promotion_ready": False,
                "next_action": "run hybrid refinement",
            },
            {
                "stage": "hybrid_reduced_transient_stationarity_gate",
                "status": "skipped",
                "promotion_ready": False,
                "next_action": "run hybrid stationarity",
            },
            {
                "stage": "hybrid_movie_grid_time_refinement_gate",
                "status": "skipped",
                "promotion_ready": False,
                "next_action": "run hybrid movie refinement",
            },
            {
                "stage": "hybrid_diagnostic_turbulence_media",
                "status": "skipped",
                "promotion_ready": False,
                "next_action": "run hybrid media",
            },
        ],
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["promotion_ready"] is False
    assert summary["near_term_closeout_status"] == "live_evidence_incomplete"
    assert summary["promotion_rejection_reasons"] == [
        "skipped_stage_not_live_promotion_evidence"
    ]
    assert [
        blocker["stage"] for blocker in summary["promotion_blocking_stages"]
    ] == [
        "hybrid_parallel_step_refinement_gate",
        "hybrid_reduced_transient_stationarity_gate",
        "hybrid_movie_grid_time_refinement_gate",
        "hybrid_diagnostic_turbulence_media",
    ]
    assert summary["diagnostic_stages"] == []
    assert summary["next_actions"] == [
        "run hybrid refinement",
        "run hybrid stationarity",
        "run hybrid movie refinement",
        "run hybrid media",
    ]


def test_pypi_publish_workflow_ignores_artifact_releases() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")

    assert "release:" in workflow
    assert "types: [published]" in workflow
    assert "startsWith(github.event.release.tag_name, 'v')" in workflow


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


def test_committed_gpu_profile_summaries_report_gpu_execution() -> None:
    for path in COMMITTED_GPU_PROFILE_SUMMARIES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        backend = payload.get("backend", payload.get("default_backend"))
        devices = payload.get("devices", payload.get("local_devices", ()))
        device_text = json.dumps(devices, sort_keys=True).lower()

        assert backend == "gpu", f"{path} is not a committed GPU summary"
        assert devices, f"{path} does not record GPU devices"
        assert any(token in device_text for token in ("cuda", "gpu", "nvidia")), path

        if "recycling_dthe_jax_linearized_gate" in path.as_posix():
            profile = payload["profile"]
            assert profile["solver_mode"] == "jax_linearized"
            residual_ceiling = payload.get("gate_requirements", {}).get(
                "max_residual_inf_norm",
                profile["residual_tolerance"],
            )
            assert profile["residual_inf_norm"] <= residual_ceiling
        if "atomic_rate_throughput_gate_gpu" in path.as_posix():
            showcase = payload["differentiability_showcase"]
            assert payload["case"] == "atomic_rate_throughput_gate"
            assert showcase["sensitivity_relative_error"] <= 1e-8
            assert payload["pmap_requested"] is False
            assert payload["pmap_enabled"] is False
            assert payload["pmap_sanity_passed"] is None
        if "stellarator_drb_pytree_gpu_profile_summary" in path.as_posix():
            assert payload["campaign_passed"] is True
            assert payload["local_device_count"] >= 1


def test_atomic_rate_throughput_summaries_record_pmap_sanity_metadata() -> None:
    for suffix in ("atomic_rate_throughput_gate_cpu", "atomic_rate_throughput_gate_gpu"):
        path = REPO_ROOT / "docs" / "data" / "runtime_profile_artifacts" / suffix / "profile_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["case"] == "atomic_rate_throughput_gate"
        assert "pmap_requested" in payload
        assert "pmap_enabled" in payload
        assert "pmap_sanity_passed" in payload
        assert "pmap_skip_reason" in payload
        assert payload["pmap_requested"] is False
        assert payload["pmap_enabled"] is False
        for result in payload["results"]:
            assert result["pmap_device_count"] == 0
            assert result["pmap_parity_passed"] is None


def test_recycling_batched_jvp_summary_records_pmap_sanity_metadata() -> None:
    path = (
        REPO_ROOT
        / "docs"
        / "data"
        / "runtime_profile_artifacts"
        / "recycling_dthe_batched_jvp_gate_cpu"
        / "profile_summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["case"] == "recycling_batched_jvp_profile"
    assert payload["rhs_backend"] == "fixed_full_field_array"
    assert payload["pmap_requested"] is False
    assert payload["pmap_enabled"] is False
    assert payload["pmap_sanity_passed"] is None
    assert "pmap_skip_reason" in payload
    assert payload["differentiability"]["jvp_fd_relative_error"] < 1.0e-8
    for result in payload["batch_results"]:
        assert result["residual_batched_serial_max_abs_error"] <= 1.0e-18
        assert result["jvp_batched_serial_max_abs_error"] <= 1.0e-18
        assert result["pmap_device_count"] == 0
        assert result["pmap_jvp_seconds_median"] is None


def test_committed_jax_linearized_cpu_profiles_report_solver_status() -> None:
    profile_paths = (
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
        / "recycling_dthe_jax_linearized_gate_ny100_dt1e4_lineax_cpu"
        / "profile_summary.json",
    )

    for path in profile_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = payload["profile"]
        diagnostics = profile["diagnostics"]

        assert payload["backend"] == "cpu"
        assert profile["residual_inf_norm"] <= profile["residual_tolerance"]
        assert profile["linear_solver_success"] is True
        assert diagnostics["linear_solver_success"] is True
        assert profile["linear_solver_backend"] == diagnostics["linear_solver_backend"]
        assert profile["linear_solver_status"] == diagnostics["linear_solver_status"]
        assert profile["linear_solver_reported_iterations"] == diagnostics["linear_solver_reported_iterations"]
        assert "/Users/" not in json.dumps(payload, sort_keys=True)


def test_committed_adaptive_bdf_jax_lineax_profile_reports_controller_health() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "docs"
            / "data"
            / "runtime_profile_artifacts"
            / "recycling_1d_adaptive_bdf_jax_lineax_gate"
            / "profile_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["case"] == "recycling_1d_one_step"
    assert payload["diagnostics_only"] is True
    assert payload["adaptive_bdf_gate_errors"] == {
        "adaptive_bdf_jax_linearized": [],
        "adaptive_bdf_jax_linearized_lineax": [],
    }
    assert payload["mode_elapsed_seconds"]["adaptive_bdf_jax_linearized_lineax"] < payload["mode_elapsed_seconds"]["adaptive_bdf_jax_linearized"]

    expected_step_modes = {
        "adaptive_bdf_jax_linearized": "jax_linearized",
        "adaptive_bdf_jax_linearized_lineax": "jax_linearized_lineax",
    }
    for mode, step_mode in expected_step_modes.items():
        diagnostics = payload["mode_diagnostics"][mode]
        assert diagnostics["adaptive_bdf_step_solver_mode"] == step_mode
        assert diagnostics["adaptive_bdf_accepted_steps"] == 21
        assert diagnostics["adaptive_bdf_rejected_steps"] == 6
        assert diagnostics["adaptive_bdf_trial_solver_steps"] == 61
        assert diagnostics["adaptive_bdf_fixed_full_field_rhs_solver_steps"] == 61
        assert diagnostics["adaptive_bdf_minimum_dt_fallbacks"] == 0
        assert diagnostics["adaptive_bdf_unconverged_solver_steps"] == 0
        assert diagnostics["adaptive_bdf_max_accepted_error_ratio"] <= 0.95


def test_committed_dthe_adaptive_bdf_trace_probe_reports_blocker() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "docs"
            / "data"
            / "runtime_profile_artifacts"
            / "recycling_dthe_adaptive_bdf_trace_probe"
            / "profile_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["case"] == "recycling_dthe_one_step"
    assert payload["mode"] == "adaptive_bdf_jax_linearized"
    assert "timed out" in payload["gate_failure"]
    assert payload["completed_implicit_trials"] == 8
    assert payload["started_implicit_trials"] == 9
    assert payload["completed_error_estimates"] == 2
    assert payload["linear_solve_seconds_completed_trials"] > payload["jacobian_assembly_seconds_completed_trials"]
    assert payload["startup_error_ratios"][0] > 1.0e6
    assert payload["krylov_control_probe_10x10"]["average_completed_trial_elapsed_seconds"] > 20.0
    assert payload["lineax_backend_probe"]["linear_solver_failure_count"] == 5
    assert payload["sparse_jvp_backend_probe"]["completed_trial_counts_by_kind"]["bdf2_corrector"] == 37
    assert payload["sparse_jvp_backend_probe"]["jacobian_assembly_seconds_completed_trials"] > 150.0
    contributor_probe = payload["sparse_jvp_error_contributor_probe"]
    assert contributor_probe["completed_error_estimates"] == 17
    assert contributor_probe["last_dominant_contributors"][-1]["dominant"] == "NVd+"
    assert contributor_probe["last_top_fields"][0]["name"] == "NVd+"
    assert contributor_probe["last_top_fields"][1]["name"] == "NVt+"
    assert contributor_probe["last_top_feedback"][0]["rms_ratio"] < 1.0e-6
    momentum_floor_probe = payload["sparse_jvp_momentum_floor_probe"]
    assert momentum_floor_probe["momentum_atol_floor"] == 0.01
    assert momentum_floor_probe["error_ratios"][-1] < contributor_probe["last_error_ratios"][-1]
    assert momentum_floor_probe["dominant_contributors"][-1] == "Nd"
    component_floor_gate = payload["sparse_jvp_component_floor_gate"]
    assert component_floor_gate["adaptive_bdf_gate_errors"] == []
    assert component_floor_gate["accepted_steps"] == 8
    assert component_floor_gate["bdf2_accepted_steps"] == 7
    assert component_floor_gate["rejected_steps"] == 0
    assert component_floor_gate["unconverged_solver_steps"] == 0
    assert component_floor_gate["minimum_dt_fallbacks"] == 0
    assert component_floor_gate["max_accepted_error_ratio"] <= 0.95
    workspace_gate = payload["sparse_jvp_workspace_reuse_gate"]
    assert workspace_gate["adaptive_bdf_gate_errors"] == []
    assert workspace_gate["accepted_steps"] == component_floor_gate["accepted_steps"]
    assert workspace_gate["bdf2_accepted_steps"] == component_floor_gate["bdf2_accepted_steps"]
    assert workspace_gate["sparse_jvp_workspace_reuses"] == workspace_gate["trial_solver_steps"]
    assert workspace_gate["jvp_jacobian_prebuilt_direction_batch_uses"] == workspace_gate["trial_solver_steps"]
    assert workspace_gate["jvp_jacobian_tangent_build_seconds"] == 0.0
    assert workspace_gate["jvp_jacobian_linearize_seconds"] > workspace_gate["jvp_jacobian_device_execute_seconds"]
    assert workspace_gate["jvp_jacobian_device_execute_seconds"] > workspace_gate["jvp_jacobian_sparse_assembly_seconds"]
    assert workspace_gate["jvp_jacobian_host_transfer_seconds"] < 1.0e-2
    assert workspace_gate["max_accepted_error_ratio"] <= 0.95
    assert "/Users/" not in json.dumps(payload, sort_keys=True)


