from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import (
    FciMaps,
    build_synthetic_stellarator_geometry,
    identity_fci_maps,
)
from drbx.native.fci_vorticity import (
    apply_fci_vorticity_operator,
    solve_fci_vorticity_potential_cg,
)
from drbx.validation import (
    audit_essos_imported_artifact_report,
    audit_essos_imported_artifact_reports,
    audit_hybrid_open_sol_promotion_evidence,
    create_stellarator_fci_geometry_campaign_package,
    create_stellarator_fci_operator_campaign_package,
    create_stellarator_fci_suite_campaign_package,
    create_stellarator_drb_pytree_campaign_package,
    create_stellarator_metric_mms_campaign_package,
    create_stellarator_neutral_physics_campaign_package,
    create_stellarator_sheath_recycling_campaign_package,
    create_stellarator_sol_showcase_package,
    create_stellarator_vorticity_campaign_package,
)
from drbx.validation.essos_imported_fci_campaign import (
    _IMPORTED_FCI_DIAGNOSTIC_SCHEMA,
    _IMPORTED_FCI_REQUIRED_REPORT_FIELDS,
    build_essos_imported_connection_length_refinement_diagnostics,
    build_essos_imported_endpoint_label_refinement_diagnostics,
    create_essos_imported_endpoint_label_refinement_package,
    build_essos_imported_fci_map_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fci_vorticity_jacobi_preconditioner_reduces_fixed_budget_residual() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=10, ny=8, nz=18)
    phi_exact = (
        jnp.sin(2.0 * geometry.poloidal_angle - 3.0 * geometry.toroidal_angle)
        + 0.35 * geometry.radial * jnp.cos(
            3.0 * geometry.poloidal_angle + geometry.toroidal_angle
        )
    )
    density = (
        1.0
        + 0.20 * geometry.radial
        + 0.06 * jnp.cos(geometry.poloidal_angle - 2.0 * geometry.toroidal_angle)
    )
    vorticity = apply_fci_vorticity_operator(
        phi_exact,
        density,
        geometry.metric,
        regularization=0.5,
    )

    unpreconditioned = solve_fci_vorticity_potential_cg(
        vorticity,
        density,
        geometry.metric,
        iterations=10,
        regularization=0.5,
    )
    jacobi = solve_fci_vorticity_potential_cg(
        vorticity,
        density,
        geometry.metric,
        iterations=10,
        regularization=0.5,
        preconditioner="jacobi",
    )

    assert jacobi.preconditioner == "jacobi"
    assert float(jacobi.residual_l2) < 0.75 * float(unpreconditioned.residual_l2)


def test_fci_vorticity_boussinesq_non_boussinesq_limit_and_contrast() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=8, ny=6, nz=14)
    phi = (
        jnp.sin(jnp.pi * geometry.radial)
        * jnp.cos(2.0 * geometry.poloidal_angle - 3.0 * geometry.toroidal_angle)
        + 0.15
        * geometry.radial
        * jnp.sin(geometry.poloidal_angle + 2.0 * geometry.toroidal_angle)
    )
    variable_density = (
        1.0
        + 0.40 * geometry.radial
        + 0.12 * jnp.cos(geometry.poloidal_angle - geometry.toroidal_angle)
    )
    boussinesq = apply_fci_vorticity_operator(
        phi,
        variable_density,
        geometry.metric,
        boussinesq=True,
    )
    non_boussinesq = apply_fci_vorticity_operator(
        phi,
        variable_density,
        geometry.metric,
        boussinesq=False,
    )
    relative_difference = np.linalg.norm(np.asarray(non_boussinesq - boussinesq)) / max(
        np.linalg.norm(np.asarray(boussinesq)),
        1.0e-30,
    )

    constant_coefficient_density = 2.3 * jnp.square(geometry.metric.Bxy)
    constant_boussinesq = apply_fci_vorticity_operator(
        phi,
        constant_coefficient_density,
        geometry.metric,
        boussinesq=True,
    )
    constant_non_boussinesq = apply_fci_vorticity_operator(
        phi,
        constant_coefficient_density,
        geometry.metric,
        boussinesq=False,
    )

    assert relative_difference > 5.0e-2
    assert np.max(np.abs(np.asarray(constant_boussinesq - constant_non_boussinesq))) < 1.0e-10


def test_imported_fci_example_resolves_source_specific_artifact_defaults(capsys) -> None:
    module = _load_imported_fci_campaign_example()

    settings = module.build_run_settings(
        map_sources=("coil", "vmec", "hybrid"),
        nx=3,
        ny=4,
        nz=8,
    )

    assert [item.map_source for item in settings] == ["coil", "vmec", "hybrid"]
    assert [item.case_label for item in settings] == [
        "essos_imported_fci_campaign",
        "essos_imported_fci_vmec_campaign",
        "essos_imported_fci_hybrid_campaign",
    ]
    assert [str(item.output_root) for item in settings] == [
        "docs/data/essos_imported_fci_artifacts",
        "docs/data/essos_imported_fci_vmec_artifacts",
        "docs/data/essos_imported_fci_hybrid_artifacts",
    ]
    assert all((item.nx, item.ny, item.nz) == (3, 4, 8) for item in settings)
    assert all(item.require_connection_resolution is False for item in settings)

    custom_settings = module.build_run_settings(
        map_sources=("hybrid",),
        output_root=Path("tmp/hybrid"),
        case_label="custom",
    )
    module.run_resolved_campaigns(custom_settings, dry_run=True)
    captured = capsys.readouterr()
    assert "map_source=hybrid" in captured.out
    assert "output_root=tmp/hybrid" in captured.out
    assert "case_label=custom" in captured.out


def test_imported_fci_dry_run_artifact_schema_is_self_contained(tmp_path: Path, capsys) -> None:
    module = _load_imported_fci_campaign_example()
    output_root = tmp_path / "imported_fci_contract"

    settings = module.build_run_settings(
        map_sources=("hybrid",),
        output_root=output_root,
        case_label="custom_imported_fci",
        nx=3,
        ny=4,
        nz=8,
        require_connection_resolution=True,
    )
    module.run_resolved_campaigns(settings, dry_run=True, dry_run_artifacts=True)

    captured = capsys.readouterr()
    contract_path = output_root / "data" / "custom_imported_fci_dry_run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert "wrote dry-run contract" in captured.out
    assert contract["self_contained"] is True
    assert contract["requires_essos_runtime"] is False
    assert contract["live_run_requires_essos_runtime"] is True
    assert contract["map_source"] == "hybrid"
    assert contract["grid"]["shape"] == [3, 4, 8]
    assert contract["planned_artifacts"]["report_json"].endswith("custom_imported_fci.json")
    assert contract["planned_artifacts"]["dry_run_contract_json"].endswith("custom_imported_fci_dry_run_contract.json")
    assert "connection_length_diagnostics" in contract["required_report_fields"]
    assert "connection_length_resolution_diagnostics" in contract["required_report_fields"]
    assert "map_quality_diagnostics" in contract["required_report_fields"]
    assert "endpoint_length_diagnostics" in contract["required_report_fields"]
    assert "target_label_diagnostics" in contract["required_report_fields"]
    assert "connection_length_resolution_diagnostics" in contract["diagnostic_schema"]
    assert "map_quality_diagnostics" in contract["diagnostic_schema"]
    assert "endpoint_length_diagnostics" in contract["diagnostic_schema"]
    assert "target_label_diagnostics" in contract["diagnostic_schema"]
    assert "refinement_diagnostics" in contract["diagnostic_schema"]
    assert "consumed_map_diagnostics" in contract["diagnostic_schema"]
    assert "target_label_toroidal" in contract["required_array_keys"]
    assert "adjacent_step_toroidal" in contract["required_array_keys"]
    assert "target_exit_toroidal" in contract["required_array_keys"]
    assert contract["external_inputs"]["not_read_in_dry_run"] is True
    assert contract["strict_gates"]["require_connection_resolution"] is True
    assert contract["passed"] is True


def test_imported_artifact_schema_audit_accepts_current_fci_report(
    tmp_path: Path,
) -> None:
    report = {field: True for field in _IMPORTED_FCI_REQUIRED_REPORT_FIELDS}
    for parent, children in _IMPORTED_FCI_DIAGNOSTIC_SCHEMA.items():
        report[parent] = {child: True for child in children}
    report["passed"] = True
    report_path = tmp_path / "current_fci.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    audit = audit_essos_imported_artifact_report(report_path, artifact_kind="fci")

    assert audit["schema_passed"] is True
    assert audit["stale"] is False
    assert audit["missing_report_fields"] == []
    assert audit["missing_diagnostic_fields"] == {}


def test_imported_artifact_schema_audit_flags_stale_fci_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "stale_fci.json"
    report_path.write_text(json.dumps({"case": "stale", "passed": True}), encoding="utf-8")

    audit = audit_essos_imported_artifact_report(report_path, artifact_kind="fci")

    assert audit["schema_passed"] is False
    assert audit["stale"] is True
    assert "connection_length_diagnostics" in audit["missing_report_fields"]
    assert "connection_length_resolution_diagnostics" in audit[
        "missing_diagnostic_fields"
    ]


_SCHEMA_AUDIT_PROBE = (
    REPO_ROOT / "docs/data/essos_imported_fci_artifacts/data/essos_imported_fci_campaign.json"
)


def _require_local_artifacts(*paths: Path) -> None:
    """Skip when regenerable docs/data artifacts are absent (docs/data is gitignored)."""

    missing = [path.name for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"local artifacts not present (docs/data is gitignored): {missing}")


def test_imported_artifact_schema_audit_example_reports_committed_current_artifacts(
    capsys,
) -> None:
    _require_local_artifacts(_SCHEMA_AUDIT_PROBE)
    module = _load_imported_artifact_schema_audit_example()
    settings = module.build_audit_settings(require_all_current=False)

    summary = module.run_artifact_schema_audit(settings)
    output = capsys.readouterr().out

    assert summary["report_count"] == 5
    assert summary["stale_report_count"] == 0
    assert summary["schema_passed"] is True
    assert output.count("status=stale") == 0
    assert output.count("status=current, kind=fci") == 3
    assert output.count("status=current, kind=movie") == 2


def test_imported_artifact_schema_audit_example_requires_current_artifacts() -> None:
    _require_local_artifacts(_SCHEMA_AUDIT_PROBE)
    module = _load_imported_artifact_schema_audit_example()
    settings = module.build_audit_settings(require_all_current=True)

    summary = module.run_artifact_schema_audit(settings)

    assert summary["schema_passed"] is True
    assert summary["stale_report_count"] == 0


def test_hybrid_open_sol_promotion_evidence_audit_accepts_committed_bundle() -> None:
    _require_local_artifacts(
        REPO_ROOT / "docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json",
        REPO_ROOT / "docs/data/essos_imported_drb_movie_stationarity_jacobi_artifacts/data/essos_imported_drb_movie_stationarity_jacobi.json",
        REPO_ROOT / "docs/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_artifacts/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_summary.json",
        REPO_ROOT / "docs/data/essos_imported_drb_movie_stationarity_jacobi_media_manifest.json",
    )
    audit = audit_hybrid_open_sol_promotion_evidence(
        fci_report_json_path=REPO_ROOT
        / "docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json",
        stationarity_report_json_path=REPO_ROOT
        / "docs/data/essos_imported_drb_movie_stationarity_jacobi_artifacts/data/essos_imported_drb_movie_stationarity_jacobi.json",
        refinement_summary_json_path=REPO_ROOT
        / "docs/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_artifacts/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_summary.json",
        media_manifest_json_path=REPO_ROOT
        / "docs/data/essos_imported_drb_movie_stationarity_jacobi_media_manifest.json",
    )

    assert audit["diagnostic"] == "essos_hybrid_open_sol_promotion_evidence_audit"
    assert audit["map_source"] == "hybrid"
    assert audit["promotion_ready"] is True
    assert audit["promotion_rejection_reasons"] == []
    assert [stage["stage"] for stage in audit["stage_reports"]] == [
        "hybrid_fci_source_profile",
        "hybrid_stationarity",
        "hybrid_grid_time_refinement",
        "hybrid_media_manifest",
    ]
    assert all(stage["passed"] is True for stage in audit["stage_reports"])


def test_hybrid_open_sol_promotion_evidence_audit_rejects_stale_media_manifest(
    tmp_path: Path,
) -> None:
    media_manifest = {
        "map_source": "coil",
        "qa": {
            "visual_qa": "failed",
            "camera_stability": "passed",
            "non_axisymmetric_geometry_visible": False,
            "opened_radial_toroidal_sector_visible": True,
        },
        "release_assets": ["https://example.invalid/movie.gif"],
        "files": [{"path": "movies/movie.gif"}],
    }
    media_path = tmp_path / "bad_media_manifest.json"
    media_path.write_text(json.dumps(media_manifest), encoding="utf-8")

    audit = audit_hybrid_open_sol_promotion_evidence(
        fci_report_json_path=REPO_ROOT
        / "docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json",
        stationarity_report_json_path=REPO_ROOT
        / "docs/data/essos_imported_drb_movie_stationarity_jacobi_artifacts/data/essos_imported_drb_movie_stationarity_jacobi.json",
        refinement_summary_json_path=REPO_ROOT
        / "docs/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_artifacts/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_summary.json",
        media_manifest_json_path=media_path,
    )

    media_stage = audit["stage_reports"][-1]
    assert audit["promotion_ready"] is False
    assert media_stage["stage"] == "hybrid_media_manifest"
    assert media_stage["passed"] is False
    assert "media_map_source_not_hybrid" in media_stage["reasons"]
    assert "media_visual_qa_not_passed" in media_stage["reasons"]
    assert "media_non_axisymmetric_geometry_not_visible" in media_stage["reasons"]
    assert "media_release_asset_url_not_project_release" in media_stage["reasons"]
    assert "media_diagnostics_file_missing" in media_stage["reasons"]
    assert "media_contact_sheet_file_missing" in media_stage["reasons"]
    assert set(media_stage["reasons"]).issubset(
        set(audit["promotion_rejection_reasons"])
    )


def test_imported_connection_length_refinement_example_resolves_live_sources(
    tmp_path: Path,
) -> None:
    module = _load_connection_length_refinement_example()

    settings = module.build_run_settings(
        live_import=True,
        map_sources=("coil", "vmec", "hybrid"),
        output_root=tmp_path / "live_refinement",
        case_label="demo",
        live_level_shapes=((3, 4, 6), (6, 8, 12), (12, 16, 24)),
        live_convergence_threshold=0.11,
        live_linf_threshold=0.22,
        require_pass=False,
    )

    assert [item.map_source for item in settings] == ["coil", "vmec", "hybrid"]
    assert [item.case_label for item in settings] == [
        "demo_coil_live",
        "demo_vmec_live",
        "demo_hybrid_live",
    ]
    assert [item.connection_quantity for item in settings] == [
        "adjacent_step_length",
        "parallel_step_per_toroidal_radian",
        "parallel_step_per_toroidal_radian",
    ]
    assert all(item.live_import is True for item in settings)
    assert all(item.require_pass is False for item in settings)
    assert all(item.convergence_threshold == 0.11 for item in settings)
    assert all(item.linf_threshold == 0.22 for item in settings)
    assert module.resolve_connection_quantity("hybrid", "target-exit-length") == (
        "target_exit_length"
    )
    with pytest.raises(ValueError, match="Unsupported imported map_source"):
        module.resolve_connection_quantity("bad_source")

    entries = [
        {
            "case_label": item.case_label,
            "promotion_ready": item.map_source != "coil",
            "advisory_only": item.map_source == "coil",
            "evidence_role": (
                "negative_observed_order_control"
                if item.map_source == "coil"
                else "promotion_ready"
            ),
        }
        for item in settings
    ]
    summary_path = module.write_refinement_sweep_summary(settings, entries)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_path.name == "demo_summary.json"
    assert summary["report_count"] == 3
    assert summary["promotion_ready_count"] == 2
    assert summary["advisory_count"] == 1
    assert summary["negative_control_count"] == 1
    assert summary["all_promotion_ready"] is False


def test_imported_connection_length_refinement_example_runs_manufactured_gate(
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_connection_length_refinement_example()
    output_root = tmp_path / "manufactured_refinement"
    settings = module.build_run_settings(
        live_import=False,
        output_root=output_root,
        case_label="manufactured_gate",
        level_shapes=((4, 6, 8), (8, 12, 16), (16, 24, 32)),
    )

    assert len(settings) == 1
    assert settings[0].map_source == "manufactured"
    assert settings[0].connection_quantity == "manufactured"
    summary = module.run_resolved_campaigns(settings)
    output = capsys.readouterr().out

    report_path = output_root / "data" / "manufactured_gate.json"
    arrays_path = output_root / "data" / "manufactured_gate.npz"
    plot_path = output_root / "images" / "manufactured_gate.png"
    summary_path = output_root / "data" / "manufactured_gate_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["promotion_ready"] is True
    assert report["evidence_role"] == "promotion_ready"
    assert report["promotion_rejection_reasons"] == []
    assert summary["report_count"] == 1
    assert summary["promotion_ready_count"] == 1
    assert summary["all_promotion_ready"] is True
    assert summary["summary_json_path"] == str(summary_path)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["entries"][0]["case_label"] == "manufactured_gate"
    assert summary_payload["entries"][0]["promotion_ready"] is True
    assert arrays_path.exists()
    assert plot_path.exists()
    assert summary_path.exists()
    assert "wrote sweep summary" in output
    assert "connection-length refinement gate passed" in output


def test_imported_fci_map_diagnostics_verify_consumed_endpoint_masks() -> None:
    base_maps = identity_fci_maps(nx=3, ny=4, nz=8, dphi=0.25)
    forward_boundary = np.zeros(base_maps.shape, dtype=bool)
    backward_boundary = np.zeros(base_maps.shape, dtype=bool)
    forward_boundary[0, :, :] = True
    backward_boundary[-1, :, :] = True
    maps = FciMaps(
        forward_x=base_maps.forward_x,
        forward_z=base_maps.forward_z,
        backward_x=base_maps.backward_x,
        backward_z=base_maps.backward_z,
        forward_boundary=jnp.asarray(forward_boundary),
        backward_boundary=jnp.asarray(backward_boundary),
        dphi=base_maps.dphi,
    )
    endpoint_count = forward_boundary.astype(np.float64) + backward_boundary.astype(np.float64)
    connection_length = np.linspace(0.0, 1.0, num=int(np.prod(base_maps.shape)), dtype=np.float64).reshape(base_maps.shape)
    adjacent_step_length = np.ones(base_maps.shape, dtype=np.float64)
    target_exit_length = np.where(endpoint_count > 0.0, 2.0, np.nan)

    diagnostics = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=connection_length,
        adjacent_step_length=adjacent_step_length,
        target_exit_length=target_exit_length,
        forward_target_exit_length=np.where(forward_boundary, 2.1, np.nan),
        backward_target_exit_length=np.where(backward_boundary, 2.2, np.nan),
        endpoint_count=endpoint_count,
        map_source="hybrid",
    )

    assert diagnostics["passed"] is True
    assert diagnostics["connection_length_diagnostics"]["finite_fraction"] == 1.0
    assert diagnostics["connection_length_diagnostics"]["nonnegative_fraction"] == 1.0
    assert diagnostics["map_quality_diagnostics"]["recommended_next_action"]
    assert diagnostics["refinement_diagnostics"]["shape"] == [3, 4, 8]
    assert diagnostics["consumed_map_diagnostics"]["endpoint_count_matches_boundary_masks"] is True
    assert diagnostics["consumed_map_diagnostics"]["orphan_endpoint_fraction"] == 0.0
    assert diagnostics["consumed_map_diagnostics"]["unconsumed_boundary_fraction"] == 0.0
    assert diagnostics["endpoint_length_diagnostics"]["passed"] is True
    assert diagnostics["endpoint_length_diagnostics"]["target_exit_finite_endpoint_fraction"] == 1.0
    assert diagnostics["endpoint_length_diagnostics"]["target_exit_p95"] == 2.0
    assert sum(diagnostics["endpoint_length_diagnostics"]["target_exit_histogram_counts"]) == int(
        np.sum(endpoint_count > 0.0)
    )
    assert diagnostics["endpoint_length_diagnostics"]["forward_exit_finite_forward_boundary_fraction"] == 1.0
    assert diagnostics["endpoint_length_diagnostics"]["backward_exit_finite_backward_boundary_fraction"] == 1.0
    assert diagnostics["endpoint_length_diagnostics"]["forward_exit_nonnegative_finite_fraction"] == 1.0
    assert diagnostics["endpoint_length_diagnostics"]["backward_exit_nonnegative_finite_fraction"] == 1.0
    assert diagnostics["target_label_diagnostics"]["passed"] is True
    assert diagnostics["target_label_diagnostics"]["endpoint_count_matches_target_labels"] is True
    assert diagnostics["target_label_diagnostics"]["forward_only_cell_count"] > 0
    assert diagnostics["target_label_diagnostics"]["backward_only_cell_count"] > 0


def test_imported_fci_connection_length_resolution_diagnostics_are_advisory() -> None:
    base_maps = identity_fci_maps(nx=4, ny=5, nz=8, dphi=0.25)
    forward_boundary = np.zeros(base_maps.shape, dtype=bool)
    backward_boundary = np.zeros(base_maps.shape, dtype=bool)
    forward_boundary[0, :, :] = True
    backward_boundary[-1, :, :] = True
    maps = FciMaps(
        forward_x=base_maps.forward_x,
        forward_z=base_maps.forward_z,
        backward_x=base_maps.backward_x,
        backward_z=base_maps.backward_z,
        forward_boundary=jnp.asarray(forward_boundary),
        backward_boundary=jnp.asarray(backward_boundary),
        dphi=base_maps.dphi,
    )
    endpoint_count = forward_boundary.astype(np.float64) + backward_boundary.astype(np.float64)
    radial = np.linspace(0.0, 1.0, base_maps.shape[0], dtype=np.float64)[:, None, None]
    toroidal = np.linspace(0.0, 2.0 * np.pi, base_maps.shape[1], endpoint=False, dtype=np.float64)[None, :, None]
    poloidal = np.linspace(0.0, 2.0 * np.pi, base_maps.shape[2], endpoint=False, dtype=np.float64)[None, None, :]
    smooth_connection = 10.0 + 0.2 * radial + 0.05 * np.cos(toroidal) + 0.03 * np.sin(poloidal)
    checkerboard = np.indices(base_maps.shape).sum(axis=0) % 2
    rough_connection = 1.0 + 9.0 * checkerboard.astype(np.float64)
    adjacent_step_length = np.ones(base_maps.shape, dtype=np.float64)
    target_exit_length = np.where(endpoint_count > 0.0, 2.0, np.nan)

    smooth = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=smooth_connection,
        adjacent_step_length=adjacent_step_length,
        target_exit_length=target_exit_length,
        forward_target_exit_length=np.where(forward_boundary, 2.1, np.nan),
        backward_target_exit_length=np.where(backward_boundary, 2.2, np.nan),
        endpoint_count=endpoint_count,
        map_source="hybrid",
    )
    rough = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=rough_connection,
        adjacent_step_length=adjacent_step_length,
        target_exit_length=target_exit_length,
        forward_target_exit_length=np.where(forward_boundary, 2.1, np.nan),
        backward_target_exit_length=np.where(backward_boundary, 2.2, np.nan),
        endpoint_count=endpoint_count,
        map_source="hybrid",
    )

    smooth_resolution = smooth["connection_length_resolution_diagnostics"]
    rough_resolution = rough["connection_length_resolution_diagnostics"]
    assert smooth_resolution["passed"] is True
    assert smooth_resolution["underresolved_face_fraction"] == 0.0
    assert rough["passed"] is True
    assert rough_resolution["passed"] is False
    assert rough_resolution["normalized_face_jump_p95"] > smooth_resolution["normalized_face_jump_p95"]
    assert rough_resolution["underresolved_face_fraction"] > 0.5
    assert rough_resolution["dominant_rough_direction"] in {"radial", "toroidal", "poloidal"}
    assert rough_resolution["radial_underresolved_face_fraction"] is not None
    assert rough["map_quality_diagnostics"]["dominant_rough_direction"] == rough_resolution[
        "dominant_rough_direction"
    ]
    assert rough["map_quality_diagnostics"]["roughness_localization"] in {
        "distributed",
        "endpoint_touch_dominated",
        "interior_dominated",
    }

    strict_rough = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=rough_connection,
        adjacent_step_length=adjacent_step_length,
        target_exit_length=target_exit_length,
        forward_target_exit_length=np.where(forward_boundary, 2.1, np.nan),
        backward_target_exit_length=np.where(backward_boundary, 2.2, np.nan),
        endpoint_count=endpoint_count,
        map_source="hybrid",
        require_connection_resolution=True,
    )
    assert strict_rough["connection_length_resolution_required"] is True
    assert strict_rough["connection_length_resolution_passed"] is False
    assert strict_rough["passed"] is False

    endpoint_rough = np.array(smooth_connection, copy=True)
    endpoint_rough[forward_boundary | backward_boundary] += 12.0
    endpoint_localized = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=endpoint_rough,
        adjacent_step_length=adjacent_step_length,
        target_exit_length=target_exit_length,
        forward_target_exit_length=np.where(forward_boundary, 2.1, np.nan),
        backward_target_exit_length=np.where(backward_boundary, 2.2, np.nan),
        endpoint_count=endpoint_count,
        map_source="hybrid",
        require_connection_resolution=True,
    )
    assert (
        endpoint_localized["map_quality_diagnostics"]["roughness_localization"]
        == "endpoint_touch_dominated"
    )
    assert endpoint_localized["connection_length_resolution_passed"] is True
    assert (
        endpoint_localized["connection_length_resolution_diagnostics"][
            "endpoint_aware_passed"
        ]
        is True
    )
    assert (
        endpoint_localized["connection_length_resolution_diagnostics"][
            "interior_resolution_passed"
        ]
        is True
    )
    assert "endpoint-mask refinement" in endpoint_localized["map_quality_diagnostics"][
        "recommended_next_action"
    ]


def test_imported_fci_map_diagnostics_require_endpoint_lengths_for_open_fields() -> None:
    base_maps = identity_fci_maps(nx=3, ny=4, nz=8, dphi=0.25)
    forward_boundary = np.zeros(base_maps.shape, dtype=bool)
    backward_boundary = np.zeros(base_maps.shape, dtype=bool)
    forward_boundary[0, :, :] = True
    backward_boundary[-1, :, :] = True
    maps = FciMaps(
        forward_x=base_maps.forward_x,
        forward_z=base_maps.forward_z,
        backward_x=base_maps.backward_x,
        backward_z=base_maps.backward_z,
        forward_boundary=jnp.asarray(forward_boundary),
        backward_boundary=jnp.asarray(backward_boundary),
        dphi=base_maps.dphi,
    )
    endpoint_count = forward_boundary.astype(np.float64) + backward_boundary.astype(np.float64)
    connection_length = np.ones(base_maps.shape, dtype=np.float64)

    diagnostics = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=connection_length,
        adjacent_step_length=np.ones(base_maps.shape, dtype=np.float64),
        target_exit_length=np.full(base_maps.shape, np.nan, dtype=np.float64),
        endpoint_count=endpoint_count,
        map_source="coil",
    )

    assert diagnostics["endpoint_length_diagnostics"]["passed"] is False
    assert diagnostics["passed"] is False
    assert diagnostics["endpoint_length_diagnostics"]["target_exit_finite_endpoint_fraction"] == 0.0


def test_imported_fci_map_diagnostics_require_directional_endpoint_lengths() -> None:
    base_maps = identity_fci_maps(nx=3, ny=4, nz=8, dphi=0.25)
    forward_boundary = np.zeros(base_maps.shape, dtype=bool)
    backward_boundary = np.zeros(base_maps.shape, dtype=bool)
    forward_boundary[0, :, :] = True
    backward_boundary[-1, :, :] = True
    maps = FciMaps(
        forward_x=base_maps.forward_x,
        forward_z=base_maps.forward_z,
        backward_x=base_maps.backward_x,
        backward_z=base_maps.backward_z,
        forward_boundary=jnp.asarray(forward_boundary),
        backward_boundary=jnp.asarray(backward_boundary),
        dphi=base_maps.dphi,
    )
    endpoint_count = forward_boundary.astype(np.float64) + backward_boundary.astype(np.float64)

    diagnostics = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=np.ones(base_maps.shape, dtype=np.float64),
        adjacent_step_length=np.ones(base_maps.shape, dtype=np.float64),
        target_exit_length=np.where(endpoint_count > 0.0, 2.0, np.nan),
        forward_target_exit_length=np.where(forward_boundary, 2.1, np.nan),
        backward_target_exit_length=np.full(base_maps.shape, np.nan, dtype=np.float64),
        endpoint_count=endpoint_count,
        map_source="hybrid",
    )

    assert diagnostics["endpoint_length_diagnostics"]["passed"] is False
    assert diagnostics["endpoint_length_diagnostics"]["target_exit_finite_endpoint_fraction"] == 1.0
    assert diagnostics["endpoint_length_diagnostics"]["forward_exit_finite_forward_boundary_fraction"] == 1.0
    assert diagnostics["endpoint_length_diagnostics"]["backward_exit_finite_backward_boundary_fraction"] == 0.0
    assert diagnostics["passed"] is False


def test_imported_fci_map_diagnostics_reject_mismatched_target_labels() -> None:
    base_maps = identity_fci_maps(nx=3, ny=4, nz=8, dphi=0.25)
    forward_boundary = np.zeros(base_maps.shape, dtype=bool)
    backward_boundary = np.zeros(base_maps.shape, dtype=bool)
    forward_boundary[0, :, :] = True
    backward_boundary[-1, :, :] = True
    maps = FciMaps(
        forward_x=base_maps.forward_x,
        forward_z=base_maps.forward_z,
        backward_x=base_maps.backward_x,
        backward_z=base_maps.backward_z,
        forward_boundary=jnp.asarray(forward_boundary),
        backward_boundary=jnp.asarray(backward_boundary),
        dphi=base_maps.dphi,
    )
    endpoint_count = np.zeros(base_maps.shape, dtype=np.float64)
    connection_length = np.ones(base_maps.shape, dtype=np.float64)
    target_exit_length = np.where(forward_boundary | backward_boundary, 2.0, np.nan)

    diagnostics = build_essos_imported_fci_map_diagnostics(
        maps=maps,
        connection_length=connection_length,
        adjacent_step_length=np.ones(base_maps.shape, dtype=np.float64),
        target_exit_length=target_exit_length,
        endpoint_count=endpoint_count,
        map_source="hybrid",
    )

    assert diagnostics["target_label_diagnostics"]["passed"] is False
    assert diagnostics["target_label_diagnostics"]["endpoint_count_matches_target_labels"] is False
    assert diagnostics["passed"] is False


def test_imported_fci_connection_length_refinement_diagnostics_rank_nested_grids() -> None:
    def smooth_connection(shape: tuple[int, int, int]) -> np.ndarray:
        x = (np.arange(shape[0], dtype=np.float64) + 0.5) / shape[0]
        y = (np.arange(shape[1], dtype=np.float64) + 0.5) / shape[1]
        z = (np.arange(shape[2], dtype=np.float64) + 0.5) / shape[2]
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return 10.0 + 0.2 * xx + 0.03 * np.sin(2.0 * np.pi * yy) + 0.02 * np.cos(2.0 * np.pi * zz)

    high = smooth_connection((16, 16, 32))
    mid_truth = high.reshape(8, 2, 8, 2, 16, 2).mean(axis=(1, 3, 5))
    coarse_truth = mid_truth.reshape(4, 2, 4, 2, 8, 2).mean(axis=(1, 3, 5))
    coarse = coarse_truth + 0.02
    mid = mid_truth + 0.005

    report = build_essos_imported_connection_length_refinement_diagnostics(
        [coarse, mid, high],
        labels=["coarse", "medium", "fine"],
        convergence_threshold=0.01,
        linf_threshold=0.01,
    )

    assert report["passed"] is True
    assert report["promotion_ready"] is False
    assert report["advisory_only"] is True
    assert report["evidence_role"] == "advisory_only"
    assert report["promotion_rejection_reasons"] == ["observed_order_not_required"]
    assert report["pair_reports"][0]["normalized_rms_error"] > report["pair_reports"][1]["normalized_rms_error"]
    assert report["pair_reports"][0]["max_error_indices"] is not None
    assert report["pair_reports"][0]["max_error_normalized"] == report["pair_reports"][0]["normalized_linf_error"]
    assert report["observed_orders"][0]["observed_order"] > 1.0
    assert report["monotonic_rms_error_reduction"] is True
    assert report["monotonic_linf_error_reduction"] is True
    assert report["rms_error_reduction_factors"][0] > 1.0
    assert report["linf_error_reduction_factors"][0] > 1.0


def test_imported_fci_connection_length_refinement_marks_promotion_ready() -> None:
    high = np.ones((16, 16, 32), dtype=np.float64)
    mid_truth = high.reshape(8, 2, 8, 2, 16, 2).mean(axis=(1, 3, 5))
    coarse_truth = mid_truth.reshape(4, 2, 4, 2, 8, 2).mean(axis=(1, 3, 5))
    coarse = coarse_truth + 0.02
    mid = mid_truth + 0.005

    report = build_essos_imported_connection_length_refinement_diagnostics(
        [coarse, mid, high],
        labels=["coarse", "medium", "fine"],
        convergence_threshold=0.01,
        linf_threshold=0.01,
        minimum_observed_order=1.5,
        require_observed_order=True,
    )

    assert report["passed"] is True
    assert report["promotion_ready"] is True
    assert report["advisory_only"] is False
    assert report["evidence_role"] == "promotion_ready"
    assert report["promotion_rejection_reasons"] == []
    assert report["observed_order_passed"] is True


def test_imported_fci_connection_length_refinement_classifies_negative_order() -> None:
    high = np.ones((16, 16, 32), dtype=np.float64)
    mid_truth = high.reshape(8, 2, 8, 2, 16, 2).mean(axis=(1, 3, 5))
    coarse_truth = mid_truth.reshape(4, 2, 4, 2, 8, 2).mean(axis=(1, 3, 5))
    coarse = coarse_truth + 0.040
    mid = mid_truth + 0.019

    report = build_essos_imported_connection_length_refinement_diagnostics(
        [coarse, mid, high],
        labels=["coarse", "medium", "fine"],
        convergence_threshold=0.05,
        linf_threshold=0.05,
        minimum_observed_order=0.5,
        require_observed_order=True,
    )

    assert report["passed"] is False
    assert report["promotion_ready"] is False
    assert report["advisory_only"] is True
    assert report["evidence_role"] == "negative_observed_order_control"
    assert report["observed_order_passed"] is False
    assert "observed_order_below_threshold" in report["promotion_rejection_reasons"]
    assert report["monotonic_error_reduction_passed"] is True


def test_imported_fci_connection_length_refinement_requires_monotonic_error_reduction() -> None:
    high = np.ones((16, 16, 32), dtype=np.float64)
    mid_truth = high.reshape(8, 2, 8, 2, 16, 2).mean(axis=(1, 3, 5))
    coarse_truth = mid_truth.reshape(4, 2, 4, 2, 8, 2).mean(axis=(1, 3, 5))
    coarse = coarse_truth + 0.002
    mid = mid_truth + 0.05

    report = build_essos_imported_connection_length_refinement_diagnostics(
        [coarse, mid, high],
        labels=["coarse", "medium", "fine"],
        convergence_threshold=0.50,
        linf_threshold=0.50,
        minimum_observed_order=0.0,
    )

    assert report["passed"] is False
    assert report["promotion_ready"] is False
    assert report["monotonic_rms_error_reduction"] is False
    assert report["monotonic_linf_error_reduction"] is False
    assert report["monotonic_error_reduction_passed"] is False
    assert "nonmonotonic_error_reduction" in report["promotion_rejection_reasons"]


def test_imported_fci_connection_length_refinement_can_relax_finite_overlap() -> None:
    coarse = np.ones((4, 4, 8), dtype=np.float64)
    fine = np.ones((8, 8, 16), dtype=np.float64)
    coarse[::2, :, :] = np.nan

    strict = build_essos_imported_connection_length_refinement_diagnostics(
        [coarse, fine],
        labels=["coarse", "fine"],
        convergence_threshold=0.01,
        linf_threshold=0.01,
    )
    relaxed = build_essos_imported_connection_length_refinement_diagnostics(
        [coarse, fine],
        labels=["coarse", "fine"],
        convergence_threshold=0.01,
        linf_threshold=0.01,
        minimum_finite_pair_fraction=0.4,
    )

    assert strict["finite_pairs_passed"] is False
    assert strict["evidence_role"] == "invalid_finite_pairs"
    assert relaxed["finite_pairs_passed"] is True
    assert relaxed["minimum_finite_pair_fraction"] == 0.4
    assert relaxed["pair_reports"][0]["finite_fraction"] == 0.5
    assert relaxed["passed"] is True
    assert relaxed["promotion_ready"] is False
    assert relaxed["evidence_role"] == "advisory_no_observed_order"


def test_imported_fci_endpoint_label_refinement_marks_promotion_ready() -> None:
    coarse = np.zeros((2, 2, 2), dtype=np.int8)
    coarse[0, :, 0] = 1
    coarse[1, :, 1] = 2
    coarse[1, 1, 0] = 3
    medium = np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), 2, axis=2)
    fine = np.repeat(np.repeat(np.repeat(medium, 2, axis=0), 2, axis=1), 2, axis=2)

    report = build_essos_imported_endpoint_label_refinement_diagnostics(
        [coarse, medium, fine],
        labels=["coarse", "medium", "fine"],
        minimum_agreement_fraction=1.0,
        minimum_endpoint_agreement_fraction=1.0,
        require_three_levels=True,
    )

    assert report["passed"] is True
    assert report["promotion_ready"] is True
    assert report["evidence_role"] == "promotion_ready"
    assert report["pair_reports"][0]["agreement_fraction"] == 1.0
    assert report["pair_reports"][0]["endpoint_agreement_fraction"] == 1.0
    assert report["pair_reports"][0]["confusion_matrix"][1][1] > 0
    assert report["promotion_rejection_reasons"] == []


def test_imported_fci_endpoint_label_refinement_rejects_endpoint_instability() -> None:
    coarse = np.zeros((2, 2, 2), dtype=np.int8)
    coarse[0, :, 0] = 1
    coarse[1, :, 1] = 2
    medium = np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), 2, axis=2)
    medium[:2, :, :2] = 0

    report = build_essos_imported_endpoint_label_refinement_diagnostics(
        [coarse, medium],
        labels=["coarse", "medium"],
        minimum_agreement_fraction=0.95,
        minimum_endpoint_agreement_fraction=0.95,
        require_three_levels=False,
    )

    assert report["passed"] is False
    assert report["promotion_ready"] is False
    assert report["evidence_role"] == "endpoint_label_instability"
    assert "endpoint_label_agreement_below_threshold" in report["promotion_rejection_reasons"]
    assert report["pair_reports"][0]["endpoint_false_negative_fraction"] > 0.0
    assert report["pair_reports"][0]["confusion_matrix"][1][0] > 0


def test_imported_fci_endpoint_label_refinement_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_essos_imported_endpoint_label_refinement_package(
        output_root=tmp_path / "endpoint_labels",
        case_label="endpoint_labels",
        level_shapes=((4, 6, 8), (8, 12, 16), (16, 24, 32)),
        minimum_agreement_fraction=0.0,
        minimum_endpoint_agreement_fraction=0.0,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["diagnostics"]["diagnostic"] == "essos_imported_endpoint_label_refinement"
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    with np.load(artifacts.arrays_npz_path) as arrays:
        assert "level_0_target_label_toroidal" in arrays.files
        assert "pair_0_confusion_matrix" in arrays.files


def test_imported_fci_connection_length_refinement_rejects_bad_finite_fraction() -> None:
    with pytest.raises(ValueError, match="minimum_finite_pair_fraction"):
        build_essos_imported_connection_length_refinement_diagnostics(
            [np.ones((4, 4, 8), dtype=np.float64), np.ones((8, 8, 16), dtype=np.float64)],
            minimum_finite_pair_fraction=0.0,
        )


def test_imported_fci_connection_length_refinement_rejects_non_nested_grids() -> None:
    coarse = np.ones((4, 4, 8), dtype=np.float64)
    non_nested = np.ones((7, 8, 16), dtype=np.float64)

    with pytest.raises(ValueError, match="nested by integer ratios"):
        build_essos_imported_connection_length_refinement_diagnostics(
            [coarse, non_nested],
            labels=["coarse", "bad"],
        )


def _load_imported_fci_campaign_example():
    module_path = REPO_ROOT / "examples" / "geometry-3D" / "essos-field-lines" / "imported_fci_campaign.py"
    spec = importlib.util.spec_from_file_location("imported_fci_campaign_example", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_imported_artifact_schema_audit_example():
    module_path = (
        REPO_ROOT
        / "examples"
        / "geometry-3D"
        / "essos-field-lines"
        / "imported_artifact_schema_audit.py"
    )
    source = module_path.read_text(encoding="utf-8").replace(
        "RUN_EXAMPLE = True",
        "RUN_EXAMPLE = False",
        1,
    )
    spec = importlib.util.spec_from_loader(
        "imported_artifact_schema_audit_example",
        loader=None,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(module_path)
    sys.modules[spec.name] = module
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


def _load_connection_length_refinement_example():
    module_path = (
        REPO_ROOT
        / "examples"
        / "geometry-3D"
        / "essos-field-lines"
        / "imported_connection_length_refinement.py"
    )
    source = module_path.read_text(encoding="utf-8").replace(
        "RUN_EXAMPLE = True",
        "RUN_EXAMPLE = False",
        1,
    )
    spec = importlib.util.spec_from_loader(
        "imported_connection_length_refinement_example",
        loader=None,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(module_path)
    sys.modules[spec.name] = module
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


def test_stellarator_fci_geometry_campaign_generates_passing_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_fci_geometry_campaign_package(output_root=tmp_path / "geometry", nx=10, ny=8, nz=16)

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    assert report["map_diagnostics"]["radial_shift_linf_cells"] > 0.0


def test_stellarator_fci_operator_campaign_generates_passing_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_fci_operator_campaign_package(output_root=tmp_path / "operators")

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["interpolation_convergence_slope"] > 1.55
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_metric_mms_campaign_generates_passing_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_metric_mms_campaign_package(
        output_root=tmp_path / "metric_mms",
        resolutions=(12, 16, 24),
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["identity_mms_observed_order"] > 1.7
    assert report["synthetic_constant_residual_linf"] < 1.0e-12
    assert report["synthetic_cross_term_fraction"] > 1.0e-3
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_fci_suite_campaign_generates_multi_configuration_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_fci_suite_campaign_package(output_root=tmp_path / "suite", nx=10, ny=8, nz=16)

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert len(report["configuration_labels"]) == 3
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_sheath_recycling_campaign_generates_balance_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_sheath_recycling_campaign_package(
        output_root=tmp_path / "sheath_recycling",
        nx=10,
        ny=8,
        nz=16,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["total_particle_loss"] > 0.0
    assert report["particle_recycling_relative_error"] < 1.0e-12
    assert report["current_balance_relative_error"] < 1.0e-12
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_neutral_physics_campaign_generates_conservation_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_neutral_physics_campaign_package(
        output_root=tmp_path / "neutral_physics",
        nx=10,
        ny=8,
        nz=16,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["total_ionisation"] > 0.0
    assert report["total_charge_exchange"] > 0.0
    assert report["particle_reaction_relative_error"] < 1.0e-12
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_vorticity_campaign_generates_inversion_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_vorticity_campaign_package(
        output_root=tmp_path / "vorticity",
        nx=10,
        ny=8,
        nz=16,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["relative_l2_potential_error"] < 2.5e-2
    assert report["relative_residual_l2"] < 5.0e-3
    assert report["boussinesq_relative_l2_potential_error"] < 2.5e-2
    assert report["non_boussinesq_relative_l2_potential_error"] < 2.5e-2
    assert report["operator_difference_relative_l2"] > 5.0e-2
    assert report["constant_coefficient_operator_linf"] < 1.0e-8
    assert report["density_over_b_squared_contrast"] > 1.0
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_drb_pytree_campaign_generates_jvp_and_scaling_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_drb_pytree_campaign_package(
        output_root=tmp_path / "pytree_drb",
        nx=8,
        ny=6,
        nz=12,
        steps=3,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["jvp_relative_error"] < 5.0e-3
    assert report["boussinesq_gate_potential_boussinesq"] is True
    assert report["non_boussinesq_gate_potential_boussinesq"] is False
    assert report["boussinesq_non_boussinesq_potential_relative_l2"] > 1.0e-4
    assert report["boussinesq_non_boussinesq_rhs_state_linf"] < 1.0e-12
    assert report["non_boussinesq_jvp_relative_error"] < 5.0e-3
    assert report["density_over_b_squared_contrast"] > 1.0
    assert report["potential_feedback_strength"] > 0.0
    assert report["potential_feedback_plasma_rhs_linf"] > 1.0e-8
    assert report["potential_feedback_neutral_rhs_linf"] < 1.0e-12
    assert report["potential_feedback_jvp_relative_error"] < 5.0e-3
    assert report["vmap_serial_linf"] < 1.0e-8
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    with np.load(artifacts.arrays_npz_path) as arrays:
        assert "potential_feedback_plasma_rhs_difference_slice" in arrays.files


def test_stellarator_sol_showcase_generates_dynamics_movie_and_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_sol_showcase_package(output_root=tmp_path / "showcase", nx=10, ny=10, nz=18)

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["final_rms_fluctuation"] > 1.0e-3
    assert artifacts.snapshot_png_path.exists()
    assert artifacts.diagnostics_png_path.exists()
    assert artifacts.poster_png_path.exists()
    assert artifacts.movie_gif_path.exists()
