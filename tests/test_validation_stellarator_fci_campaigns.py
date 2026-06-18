from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.geometry import FciMaps, identity_fci_maps
from jax_drb.validation import (
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
from jax_drb.validation.essos_imported_fci_campaign import (
    build_essos_imported_connection_length_refinement_diagnostics,
    build_essos_imported_fci_map_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    assert "endpoint_length_diagnostics" in contract["required_report_fields"]
    assert "target_label_diagnostics" in contract["required_report_fields"]
    assert "connection_length_resolution_diagnostics" in contract["diagnostic_schema"]
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
    assert diagnostics["refinement_diagnostics"]["shape"] == [3, 4, 8]
    assert diagnostics["consumed_map_diagnostics"]["endpoint_count_matches_boundary_masks"] is True
    assert diagnostics["consumed_map_diagnostics"]["orphan_endpoint_fraction"] == 0.0
    assert diagnostics["consumed_map_diagnostics"]["unconsumed_boundary_fraction"] == 0.0
    assert diagnostics["endpoint_length_diagnostics"]["passed"] is True
    assert diagnostics["endpoint_length_diagnostics"]["target_exit_finite_endpoint_fraction"] == 1.0
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
    assert report["pair_reports"][0]["normalized_rms_error"] > report["pair_reports"][1]["normalized_rms_error"]
    assert report["observed_orders"][0]["observed_order"] > 1.0
    assert report["monotonic_rms_error_reduction"] is True
    assert report["monotonic_linf_error_reduction"] is True
    assert report["rms_error_reduction_factors"][0] > 1.0
    assert report["linf_error_reduction_factors"][0] > 1.0


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
    assert report["monotonic_rms_error_reduction"] is False
    assert report["monotonic_linf_error_reduction"] is False


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
    assert report["vmap_serial_linf"] < 1.0e-8
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_sol_showcase_generates_dynamics_movie_and_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_sol_showcase_package(output_root=tmp_path / "showcase", nx=10, ny=10, nz=18)

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["final_rms_fluctuation"] > 1.0e-3
    assert artifacts.snapshot_png_path.exists()
    assert artifacts.diagnostics_png_path.exists()
    assert artifacts.poster_png_path.exists()
    assert artifacts.movie_gif_path.exists()
