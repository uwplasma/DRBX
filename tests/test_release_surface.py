from __future__ import annotations

import json

from conftest import REPO_ROOT


PUBLIC_RELEASE_FILES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "native_runtime_cli.md",
    REPO_ROOT / "docs" / "restartable_diffusion_tutorial.md",
    REPO_ROOT / "docs" / "validation_gallery.md",
    REPO_ROOT / "docs" / "physics_models.md",
    REPO_ROOT / "docs" / "research_directions.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_scaffold_demo.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_selected_field_demo.md",
    REPO_ROOT / "docs" / "tokamak_native_selected_field_demo.md",
    REPO_ROOT / "docs" / "traced_field_line_scaffold_demo.md",
    REPO_ROOT / "docs" / "traced_field_line_selected_field_demo.md",
    REPO_ROOT / "docs" / "stellarator_vmec_scaffold_demo.md",
    REPO_ROOT / "docs" / "stellarator_vmec_selected_field_demo.md",
    REPO_ROOT / "examples" / "alfven_wave_meeting_demo.py",
    REPO_ROOT / "examples" / "blob2d_meeting_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tcv-x21" / "scaffold_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tcv-x21" / "selected_field_parity_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tokamak-native" / "selected_field_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "traced-field-line" / "scaffold_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "traced-field-line" / "selected_field_parity_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-vmec" / "scaffold_demo.py",
    REPO_ROOT / "examples" / "geometry-3D" / "stellarator-vmec" / "selected_field_parity_demo.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_tcv_x21_scaffold.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_tcv_x21_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_native_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "traced_field_line_scaffold.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "traced_field_line_selected_field.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "stellarator_vmec_scaffold.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "stellarator_vmec_selected_field.py",
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
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_artifacts" / "data" / "tokamak_native_selected_field.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_artifacts" / "data" / "tokamak_native_selected_field_observable_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_artifacts" / "data" / "tokamak_native_selected_field_runtime_report.json",
    REPO_ROOT / "docs" / "data" / "tokamak_native_selected_field_short_window_artifacts" / "data" / "tokamak_native_selected_field_short_window.json",
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
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_input_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_validation_contract.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_profile_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_surface_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_scaffold_artifacts" / "data" / "stellarator_vmec_scaffold_observable_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_selected_field_artifacts" / "data" / "stellarator_vmec_selected_field_parity.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_selected_field_artifacts" / "data" / "stellarator_vmec_selected_field_parity_observable_report.json",
    REPO_ROOT / "docs" / "data" / "stellarator_vmec_selected_field_artifacts" / "data" / "stellarator_vmec_selected_field_parity_source_report.json",
)


def test_public_release_surface_avoids_local_path_leaks() -> None:
    forbidden = ("/Users/", "rogeriojorge", "local/hermes", "local/jax_drb")
    for path in PUBLIC_RELEASE_FILES:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


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
