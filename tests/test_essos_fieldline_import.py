from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.geometry import (
    build_essos_imported_fci_geometry,
    essos_runtime_available,
    load_essos_field_line_bundle_npz,
    resolve_essos_landreman_qa_json,
    resolve_essos_landreman_qa_wout,
)
from jax_drb.validation import (
    build_essos_imported_connection_length_refinement_diagnostics,
    build_essos_imported_drb_movie_refinement_diagnostics,
    build_essos_imported_drb_movie_refinement_next_campaign,
    build_essos_imported_drb_movie_refinement_summary,
    build_live_essos_imported_connection_length_levels,
    create_essos_fieldline_import_package,
    create_essos_imported_connection_length_refinement_package,
    create_essos_imported_drb_movie_refinement_campaign_package,
    create_live_essos_imported_connection_length_refinement_package,
    classify_essos_imported_drb_movie_evidence,
    create_essos_imported_drb_movie_package,
    create_essos_imported_fci_campaign_package,
    create_essos_imported_pytree_campaign_package,
    create_essos_vmec_fieldline_surface_package,
)
import jax_drb.validation.essos_imported_fci_campaign as imported_fci_campaign
import jax_drb.validation.essos_imported_drb_movie_campaign as imported_movie_campaign

REPO_ROOT = Path(__file__).resolve().parents[1]


def _logical_coordinates(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    nx, ny, nz = shape
    rho = np.linspace(0.12, 0.34, nx)
    phi = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False)
    minor_radius, toroidal_angle, poloidal_angle = np.meshgrid(
        rho,
        phi,
        theta,
        indexing="ij",
    )
    return {
        "minor_radius": minor_radius,
        "toroidal_angle": toroidal_angle,
        "poloidal_angle": poloidal_angle,
    }


def _has_essos_landreman_runtime() -> bool:
    if os.environ.get("JAX_DRB_RUN_ESSOS_TESTS") != "1":
        return False
    try:
        resolve_essos_landreman_qa_json()
        resolve_essos_landreman_qa_wout()
    except FileNotFoundError:
        return False
    return essos_runtime_available()


def _movie_report(
    *,
    grid: tuple[int, int, int],
    dt: float = 1.0e-3,
    frames: int = 4,
    substeps_per_frame: int = 4,
    map_source: str = "hybrid",
    rho_min: float = 0.20,
    rho_max: float = 0.60,
    maxtime: float = 24.0,
    times_to_trace: int = 80,
    final_fluctuation_rms: float = 0.06,
    radial_flux_proxy: float = -2.0e-3,
    radial_flux_abs_mean: float | None = None,
    radial_flux_rms: float | None = None,
    low_mode_window_covers_grid: bool | None = None,
    spectral_edge_band_power_fraction: float = 0.08,
    final_potential_residual_l2: float = 0.02,
    potential_iterations: int = 768,
    potential_regularization: float = 5.0,
    potential_preconditioner: str | None = None,
) -> dict[str, object]:
    radial_flux_abs_mean = (
        abs(radial_flux_proxy) if radial_flux_abs_mean is None else radial_flux_abs_mean
    )
    radial_flux_rms = radial_flux_abs_mean if radial_flux_rms is None else radial_flux_rms
    poloidal_count = grid[1]
    toroidal_count = grid[2] // 2 + 1
    if low_mode_window_covers_grid is None:
        low_mode_window_covers_grid = bool(poloidal_count <= 4 and toroidal_count <= 6)
    poloidal_centroid = min(1.25, max(float(poloidal_count - 1), 0.0))
    toroidal_centroid = min(2.0, max(float(toroidal_count - 1), 0.0))
    return {
        "case": f"{map_source}_{grid[0]}x{grid[1]}x{grid[2]}_{dt:g}",
        "map_source": map_source,
        "movie_physics_grid": list(grid),
        "geometry": {
            "map_source": map_source,
            "nx": grid[0],
            "ny": grid[1],
            "nz": grid[2],
            "rho_min": rho_min,
            "rho_max": rho_max,
            "maxtime": maxtime,
            "times_to_trace": times_to_trace,
        },
        "dt": dt,
        "frames": frames,
        "substeps_per_frame": substeps_per_frame,
        "passed": True,
        "final_fluctuation_rms": final_fluctuation_rms,
        "max_fluctuation_rms": final_fluctuation_rms * 1.05,
        "radial_flux_proxy": radial_flux_proxy,
        "radial_flux_abs_mean": radial_flux_abs_mean,
        "radial_flux_rms": radial_flux_rms,
        "radial_flux_peak_abs": radial_flux_rms * 1.2,
        "radial_flux_cancellation_ratio": abs(radial_flux_proxy)
        / max(radial_flux_abs_mean, 1.0e-30),
        "radial_flux_positive_fraction": (
            0.5 if radial_flux_proxy == 0.0 else float(radial_flux_proxy > 0.0)
        ),
        "low_mode_spectral_power_fraction": 0.34,
        "spectral_poloidal_mode_count": poloidal_count,
        "spectral_toroidal_mode_count": toroidal_count,
        "spectral_centroid_poloidal_index": poloidal_centroid,
        "spectral_centroid_toroidal_index": toroidal_centroid,
        "spectral_centroid_poloidal_fraction": poloidal_centroid
        / max(poloidal_count - 1, 1),
        "spectral_centroid_toroidal_fraction": toroidal_centroid
        / max(toroidal_count - 1, 1),
        "spectral_edge_band_power_fraction": spectral_edge_band_power_fraction,
        "low_mode_window_covers_grid": low_mode_window_covers_grid,
        "final_potential_residual_l2": final_potential_residual_l2,
        "potential_iterations": potential_iterations,
        "potential_regularization": potential_regularization,
        "potential_preconditioner": potential_preconditioner,
    }


def test_essos_imported_drb_movie_evidence_classification_is_conservative() -> None:
    coil = classify_essos_imported_drb_movie_evidence("coil")
    hybrid = classify_essos_imported_drb_movie_evidence("hybrid")
    vmec = classify_essos_imported_drb_movie_evidence("vmec")

    assert coil["publication_ready"] is False
    assert coil["movie_evidence_role"] == "movie_showcase_pending_connection_grid_time_refinement"
    assert "coil_connection_length_refinement_not_promotion_ready" in coil[
        "movie_promotion_rejection_reasons"
    ]
    assert hybrid["publication_ready"] is False
    assert hybrid["movie_evidence_role"] == (
        "movie_showcase_connection_control_pending_grid_time_refinement"
    )
    assert "movie_grid_refinement_not_passed" in hybrid["movie_promotion_rejection_reasons"]
    assert vmec["publication_ready"] is False
    assert vmec["movie_evidence_role"] == (
        "closed_field_movie_control_pending_open_sol_endpoint_evidence"
    )
    assert "connection_length_refinement_summary_promotion_ready" in vmec[
        "required_publication_gates"
    ]


def test_essos_imported_drb_movie_refinement_summary_passes_matched_reports() -> None:
    grid_reports = (
        _movie_report(grid=(4, 8, 16), final_fluctuation_rms=0.058),
        _movie_report(grid=(8, 16, 32), final_fluctuation_rms=0.061),
    )
    time_reports = (
        _movie_report(grid=(8, 16, 32), dt=2.0e-3, substeps_per_frame=2),
        _movie_report(grid=(8, 16, 32), dt=1.0e-3, substeps_per_frame=2),
    )

    summary = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=grid_reports,
        time_reports=time_reports,
        relative_tolerance=0.20,
    )

    assert summary["publication_ready"] is True
    assert summary["grid_refinement_passed"] is True
    assert summary["time_refinement_passed"] is True
    assert summary["movie_promotion_rejection_reasons"] == []
    assert summary["grid_refinement_diagnostics"]["axis_values"] == [512.0, 4096.0]
    assert summary["time_refinement_diagnostics"]["axis_values"] == [0.004, 0.002]


def test_essos_imported_drb_movie_refinement_summary_rejects_unstable_metrics() -> None:
    grid_reports = (
        _movie_report(
            grid=(4, 8, 16),
            radial_flux_proxy=-2.0e-3,
            radial_flux_abs_mean=2.0e-3,
        ),
        _movie_report(
            grid=(8, 16, 32),
            radial_flux_proxy=2.0e-3,
            radial_flux_abs_mean=1.0e-4,
        ),
    )
    time_reports = (_movie_report(grid=(8, 16, 32)),)

    summary = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=grid_reports,
        time_reports=time_reports,
        relative_tolerance=0.20,
    )

    assert summary["publication_ready"] is False
    assert summary["grid_refinement_passed"] is False
    assert summary["time_refinement_passed"] is False
    assert "movie_grid_refinement_not_passed" in summary["movie_promotion_rejection_reasons"]
    assert "movie_time_refinement_not_passed" in summary["movie_promotion_rejection_reasons"]
    pair = summary["grid_refinement_diagnostics"]["pair_reports"][0]
    assert pair["radial_flux_proxy_sign_agreement"] is False
    assert pair["radial_flux_sign_passed"] is False
    assert pair["metric_reports"]["radial_flux_abs_mean"]["passed"] is False
    failed_metrics = summary["grid_refinement_diagnostics"]["failed_metric_reports"]
    assert failed_metrics[0]["metric"] in {"radial_flux_abs_mean", "radial_flux_rms"}
    assert failed_metrics[0]["reason"] == "radial_transport_not_grid_or_time_stable"
    assert (
        summary["grid_refinement_diagnostics"]["dominant_failed_metrics"]
        == failed_metrics[:5]
    )
    assert any(
        "radial transport" in recommendation
        for recommendation in summary["grid_refinement_diagnostics"][
            "refinement_recommendations"
        ]
    )
    suggestion = summary["next_campaign_suggestion"]
    assert suggestion["current_finest_grid"] == [8, 16, 32]
    assert suggestion["suggested_next_grid"] == [16, 24, 48]
    assert suggestion["recommended_time_effective_frame_dt_values"] == []
    assert any(
        "radial transport" in note for note in suggestion["recommendation_notes"]
    )


def test_essos_imported_drb_movie_refinement_summary_rejects_underresolved_spectrum() -> None:
    grid_reports = (
        _movie_report(grid=(3, 4, 8), low_mode_window_covers_grid=True),
        _movie_report(grid=(4, 6, 12), low_mode_window_covers_grid=False),
    )
    time_reports = (
        _movie_report(grid=(4, 6, 12), dt=2.0e-3, substeps_per_frame=2),
        _movie_report(grid=(4, 6, 12), dt=1.0e-3, substeps_per_frame=2),
    )

    summary = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=grid_reports,
        time_reports=time_reports,
        relative_tolerance=0.20,
    )

    assert summary["publication_ready"] is False
    assert summary["grid_refinement_passed"] is False
    assert summary["time_refinement_passed"] is True
    assert "movie_grid_spectral_resolution_not_passed" in summary[
        "movie_promotion_rejection_reasons"
    ]
    assert "low_mode_window_covers_grid" in summary["movie_promotion_rejection_reasons"]
    grid_diagnostics = summary["grid_refinement_diagnostics"]
    assert grid_diagnostics["spectral_resolution_passed"] is False
    assert grid_diagnostics["spectral_resolution_reports"][0]["passed"] is False
    assert grid_diagnostics["spectral_resolution_reports"][1]["passed"] is True
    assert any(
        "spectrum" in recommendation
        for recommendation in grid_diagnostics["refinement_recommendations"]
    )
    suggestion = build_essos_imported_drb_movie_refinement_next_campaign(
        summary,
        max_total_cells=10_000,
    )
    assert suggestion["current_finest_grid"] == [4, 6, 12]
    assert suggestion["suggested_next_grid"] == [5, 12, 24]
    assert suggestion["suggested_next_grid_cell_count"] == 1440
    assert suggestion["suggested_grid_fits_cell_budget"] is True
    assert suggestion["time_refinement_action"] == (
        "reuse_current_timestep_pair_after_grid_change"
    )
    assert any("spectral content" in note for note in suggestion["recommendation_notes"])


def test_essos_imported_drb_movie_refinement_uses_floor_for_tiny_potential_residual() -> None:
    grid_reports = (
        _movie_report(grid=(4, 8, 16), final_potential_residual_l2=2.0e-12),
        _movie_report(grid=(8, 16, 32), final_potential_residual_l2=5.0e-12),
    )

    diagnostics = build_essos_imported_drb_movie_refinement_diagnostics(
        grid_reports,
        refinement_axis="grid",
        relative_tolerance=0.20,
    )

    pair = diagnostics["pair_reports"][0]
    residual_report = pair["metric_reports"]["final_potential_residual_l2"]
    assert residual_report["denominator_floor"] == pytest.approx(1.0e-10)
    assert residual_report["relative_change"] == pytest.approx(0.03)
    assert residual_report["passed"] is True
    assert diagnostics["passed"] is True


def test_essos_imported_drb_movie_refinement_flags_residual_only_solver_budget() -> None:
    grid_reports = (
        _movie_report(grid=(8, 16, 32), final_potential_residual_l2=1.0e-4),
        _movie_report(grid=(16, 32, 64), final_potential_residual_l2=4.0e-4),
    )
    time_reports = (
        _movie_report(grid=(16, 32, 64), dt=2.0e-3, final_potential_residual_l2=4.0e-4),
        _movie_report(grid=(16, 32, 64), dt=1.0e-3, final_potential_residual_l2=4.0e-4),
    )

    summary = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=grid_reports,
        time_reports=time_reports,
        relative_tolerance=0.20,
    )

    assert summary["publication_ready"] is False
    suggestion = summary["next_campaign_suggestion"]
    assert suggestion["potential_solve_action"] == (
        "rerun_same_grid_time_pair_with_larger_potential_iterations"
    )
    assert suggestion["current_potential_iterations"] == 768
    assert suggestion["recommended_potential_iterations"] == 1536
    assert suggestion["suggested_next_grid"] is None
    assert suggestion["suggested_grid_shapes"] == []
    assert any(
        "larger potential_iterations" in note
        for note in suggestion["recommendation_notes"]
    )


def test_essos_imported_drb_movie_refinement_flags_near_tolerance_radial_flux() -> None:
    coarse = _movie_report(
        grid=(16, 24, 48),
        radial_flux_proxy=1.31,
        radial_flux_abs_mean=1.31,
        radial_flux_rms=1.31,
        final_potential_residual_l2=1.0e-11,
        low_mode_window_covers_grid=False,
    )
    fine = _movie_report(
        grid=(16, 24, 96),
        radial_flux_proxy=1.0,
        radial_flux_abs_mean=1.0,
        radial_flux_rms=1.0,
        final_potential_residual_l2=1.0e-11,
        low_mode_window_covers_grid=False,
    )

    summary = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=(coarse, fine),
        time_reports=(fine, {**fine, "dt": 5.0e-4}),
        relative_tolerance=0.30,
    )
    grid_diagnostics = summary["grid_refinement_diagnostics"]
    suggestion = summary["next_campaign_suggestion"]

    assert summary["grid_refinement_passed"] is False
    assert {report["metric"] for report in grid_diagnostics["failed_metric_reports"]} == {
        "radial_flux_abs_mean",
        "radial_flux_rms",
    }
    assert {
        report["metric"]
        for report in grid_diagnostics["near_tolerance_failed_metric_reports"]
    } == {"radial_flux_abs_mean", "radial_flux_rms"}
    assert all(
        bool(report["near_tolerance"])
        for report in grid_diagnostics["near_tolerance_failed_metric_reports"]
    )
    assert any(
        "marginally above" in recommendation
        for recommendation in grid_diagnostics["refinement_recommendations"]
    )
    assert any(
        "near-tolerance miss" in note
        for note in suggestion["recommendation_notes"]
    )
    assert suggestion["near_tolerance_grid_blockers"]


def test_essos_imported_drb_movie_refinement_uses_mode_index_not_fraction() -> None:
    coarse = _movie_report(grid=(8, 12, 24))
    fine = _movie_report(grid=(16, 24, 48))
    assert coarse["spectral_centroid_toroidal_fraction"] != fine[
        "spectral_centroid_toroidal_fraction"
    ]
    assert coarse["spectral_centroid_toroidal_index"] == fine[
        "spectral_centroid_toroidal_index"
    ]

    summary = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=(coarse, fine),
        time_reports=(
            _movie_report(grid=(16, 24, 48), dt=2.0e-3),
            _movie_report(grid=(16, 24, 48), dt=1.0e-3),
        ),
        relative_tolerance=0.30,
    )

    assert summary["publication_ready"] is True
    assert summary["grid_refinement_passed"] is True
    assert summary["next_campaign_suggestion"]["suggested_grid_shapes"] == []


def test_essos_imported_drb_movie_refinement_suggests_toroidal_only_refinement() -> None:
    coarse = _movie_report(grid=(8, 12, 24))
    fine = _movie_report(grid=(16, 24, 48))
    coarse["spectral_centroid_toroidal_index"] = 4.0
    fine["spectral_centroid_toroidal_index"] = 2.0

    summary = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=(coarse, fine),
        time_reports=(
            _movie_report(grid=(16, 24, 48), dt=2.0e-3),
            _movie_report(grid=(16, 24, 48), dt=1.0e-3),
        ),
        relative_tolerance=0.30,
    )

    suggestion = summary["next_campaign_suggestion"]
    assert suggestion["dominant_grid_blockers"][0]["metric"] == (
        "spectral_centroid_toroidal_index"
    )
    assert suggestion["suggested_grid_multiplier"] == [1.0, 1.0, 2.0]
    assert suggestion["suggested_grid_shapes"] == [[16, 24, 48], [16, 24, 96]]


def test_essos_imported_drb_movie_refinement_summary_package_reads_reports(tmp_path: Path) -> None:
    first = _movie_report(grid=(4, 8, 16), dt=2.0e-3, substeps_per_frame=2)
    second = _movie_report(grid=(8, 16, 32), dt=1.0e-3, substeps_per_frame=2)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    from jax_drb.validation import create_essos_imported_drb_movie_refinement_summary_package

    artifacts = create_essos_imported_drb_movie_refinement_summary_package(
        output_root=tmp_path / "summary",
        case_label="movie_refinement",
        grid_report_json_paths=(first_path, second_path),
        time_report_json_paths=(first_path, second_path),
        relative_tolerance=0.25,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["diagnostic"] == "essos_imported_drb_movie_refinement_summary"
    assert report["publication_ready"] is True
    assert artifacts.report_json_path.name == "movie_refinement.json"


def test_essos_imported_drb_movie_refinement_campaign_writes_report_only_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_movie_campaign(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        grid = (int(kwargs["nx"]), int(kwargs["ny"]), int(kwargs["nz"]))
        report = _movie_report(
            grid=grid,
            dt=float(kwargs["dt"]),
            frames=int(kwargs["frames"]),
            substeps_per_frame=int(kwargs["substeps_per_frame"]),
            map_source=str(kwargs["map_source"]),
            rho_min=float(kwargs["rho_min"]),
            rho_max=float(kwargs["rho_max"]),
            maxtime=float(kwargs["maxtime"]),
            times_to_trace=int(kwargs["times_to_trace"]),
            low_mode_window_covers_grid=False,
            spectral_edge_band_power_fraction=0.08,
            potential_iterations=int(kwargs["potential_iterations"]),
            potential_regularization=float(kwargs["potential_regularization"]),
            potential_preconditioner=kwargs["potential_preconditioner"],
        )
        return SimpleNamespace(report=report)

    monkeypatch.setattr(
        imported_movie_campaign,
        "build_essos_imported_drb_movie_campaign",
        fake_movie_campaign,
    )

    artifacts = create_essos_imported_drb_movie_refinement_campaign_package(
        output_root=tmp_path / "movie_refinement_campaign",
        case_label="campaign",
        grid_shapes=((4, 8, 16), (8, 16, 32)),
        time_shape=(8, 16, 32),
        time_dt_values=(2.0e-3, 1.0e-3),
        grid_dt=2.0e-3,
        potential_iterations=1536,
        potential_regularization=4.0,
        potential_preconditioner="jacobi",
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["publication_ready"] is True
    assert len(calls) == 3
    assert all(call["potential_iterations"] == 1536 for call in calls)
    assert all(call["potential_regularization"] == 4.0 for call in calls)
    assert all(call["potential_preconditioner"] == "jacobi" for call in calls)
    assert len(artifacts.grid_report_json_paths) == 2
    assert len(artifacts.time_report_json_paths) == 2
    assert artifacts.grid_report_json_paths[1] == artifacts.time_report_json_paths[0]
    assert all(path.exists() for path in artifacts.grid_report_json_paths)
    assert all(path.exists() for path in artifacts.time_report_json_paths)
    assert artifacts.report_json_path.name == "campaign_summary.json"


def test_essos_imported_drb_movie_refinement_campaign_reuses_matching_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_movie_campaign(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        grid = (int(kwargs["nx"]), int(kwargs["ny"]), int(kwargs["nz"]))
        return SimpleNamespace(
            report=_movie_report(
                grid=grid,
                dt=float(kwargs["dt"]),
                frames=int(kwargs["frames"]),
                substeps_per_frame=int(kwargs["substeps_per_frame"]),
                map_source=str(kwargs["map_source"]),
                rho_min=float(kwargs["rho_min"]),
                rho_max=float(kwargs["rho_max"]),
                maxtime=float(kwargs["maxtime"]),
                times_to_trace=int(kwargs["times_to_trace"]),
                low_mode_window_covers_grid=False,
                final_potential_residual_l2=1.0e-11,
                potential_iterations=int(kwargs["potential_iterations"]),
                potential_regularization=float(kwargs["potential_regularization"]),
                potential_preconditioner=kwargs["potential_preconditioner"],
            )
        )

    monkeypatch.setattr(
        imported_movie_campaign,
        "build_essos_imported_drb_movie_campaign",
        fake_movie_campaign,
    )

    output_root = tmp_path / "movie_refinement_campaign"
    create_essos_imported_drb_movie_refinement_campaign_package(
        output_root=output_root,
        case_label="campaign",
        grid_shapes=((4, 8, 16), (8, 16, 32)),
        time_shape=(8, 16, 32),
        time_dt_values=(2.0e-3, 1.0e-3),
        grid_dt=2.0e-3,
        potential_iterations=1536,
        potential_regularization=4.0,
        potential_preconditioner="jacobi",
        reuse_existing_reports=True,
    )
    assert len(calls) == 3

    create_essos_imported_drb_movie_refinement_campaign_package(
        output_root=output_root,
        case_label="campaign",
        grid_shapes=((4, 8, 16), (8, 16, 32)),
        time_shape=(8, 16, 32),
        time_dt_values=(2.0e-3, 1.0e-3),
        grid_dt=2.0e-3,
        potential_iterations=1536,
        potential_regularization=4.0,
        potential_preconditioner="jacobi",
        reuse_existing_reports=True,
    )
    assert len(calls) == 3

    create_essos_imported_drb_movie_refinement_campaign_package(
        output_root=output_root,
        case_label="campaign",
        grid_shapes=((4, 8, 16), (8, 16, 32)),
        time_shape=(8, 16, 32),
        time_dt_values=(2.0e-3, 1.0e-3),
        grid_dt=2.0e-3,
        potential_iterations=3072,
        potential_regularization=4.0,
        potential_preconditioner="jacobi",
        reuse_existing_reports=True,
    )
    assert len(calls) == 6


def test_imported_drb_movie_refinement_summary_example_runs_on_reports(tmp_path: Path) -> None:
    module = _load_imported_drb_movie_refinement_summary_example()
    first = _movie_report(grid=(4, 8, 16), dt=2.0e-3, substeps_per_frame=2)
    second = _movie_report(grid=(8, 16, 32), dt=1.0e-3, substeps_per_frame=2)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    settings = module.build_refinement_summary_settings(
        output_root=tmp_path / "movie_refinement_summary",
        case_label="example_summary",
        grid_report_json_paths=(first_path, second_path),
        time_report_json_paths=(first_path, second_path),
        relative_tolerance=0.25,
        require_publication_ready=True,
    )
    report = module.run_refinement_summary(settings)

    assert report["publication_ready"] is True
    assert (
        tmp_path
        / "movie_refinement_summary"
        / "data"
        / "example_summary.json"
    ).exists()


def test_imported_drb_movie_refinement_campaign_example_runs_with_fake_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_imported_drb_movie_refinement_campaign_example()

    def fake_create_campaign(**kwargs: object) -> SimpleNamespace:
        root = Path(kwargs["output_root"])
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        grid_paths = (data_dir / "grid0.json", data_dir / "grid1.json")
        time_paths = (data_dir / "time0.json", data_dir / "time1.json")
        for path in (*grid_paths, *time_paths):
            path.write_text(json.dumps(_movie_report(grid=(4, 8, 16))), encoding="utf-8")
        report_path = data_dir / f"{kwargs['case_label']}_summary.json"
        report_path.write_text(
            json.dumps(
                {
                    "publication_ready": True,
                    "grid_refinement_passed": True,
                    "time_refinement_passed": True,
                    "movie_promotion_rejection_reasons": [],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            report_json_path=report_path,
            grid_report_json_paths=grid_paths,
            time_report_json_paths=time_paths,
        )

    monkeypatch.setattr(
        module,
        "create_essos_imported_drb_movie_refinement_campaign_package",
        fake_create_campaign,
    )
    settings = module.build_refinement_campaign_settings(
        output_root=tmp_path / "movie_refinement_campaign",
        case_label="example_campaign",
        grid_shapes=((4, 8, 16), (8, 16, 32)),
        time_shape=(8, 16, 32),
        time_dt_values=(2.0e-3, 1.0e-3),
        require_publication_ready=True,
    )
    report = module.run_refinement_campaign(settings)

    assert report["publication_ready"] is True
    assert (
        tmp_path
        / "movie_refinement_campaign"
        / "data"
        / "example_campaign_summary.json"
    ).exists()


def test_imported_drb_movie_refinement_campaign_example_exposes_publication_candidate() -> None:
    module = _load_imported_drb_movie_refinement_campaign_example()

    settings = module.build_publication_candidate_refinement_settings()

    assert settings.grid_shapes == ((4, 6, 12), (8, 12, 24))
    assert settings.time_shape == (8, 12, 24)
    assert settings.time_dt_values == (2.0e-3, 1.0e-3)
    assert settings.potential_iterations == 3072
    assert settings.reuse_existing_reports is True
    assert str(settings.output_root).endswith(
        "essos_imported_drb_movie_refinement_publication_artifacts"
    )


def test_imported_drb_movie_strict_json_payload_replaces_nonfinite_values() -> None:
    payload = imported_movie_campaign._strict_json_payload(
        {
            "finite": np.float64(1.25),
            "nan": np.float64(np.nan),
            "inf": np.float64(np.inf),
            "nested": [np.float32(-np.inf), np.int64(3), np.bool_(True)],
        }
    )

    assert payload == {
        "finite": 1.25,
        "nan": None,
        "inf": None,
        "nested": [None, 3, True],
    }
    json.dumps(payload, allow_nan=False)


def test_committed_imported_drb_movie_refinement_summary_locks_current_blocker() -> None:
    report_path = (
        REPO_ROOT
        / "docs"
        / "data"
        / "essos_imported_drb_movie_refinement_campaign_artifacts"
        / "data"
        / "essos_imported_drb_movie_refinement_campaign_summary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suggestion = report["next_campaign_suggestion"]

    assert report["publication_ready"] is False
    assert report["grid_refinement_passed"] is False
    assert report["time_refinement_passed"] is False
    assert report["grid_refinement_diagnostics"]["report_count"] == 2
    assert report["time_refinement_diagnostics"]["report_count"] == 2
    assert "movie_grid_spectral_resolution_not_passed" in report[
        "movie_promotion_rejection_reasons"
    ]
    assert "movie_time_spectral_resolution_not_passed" in report[
        "movie_promotion_rejection_reasons"
    ]
    assert "spectral_edge_band_power_fraction_above_limit" in report[
        "movie_promotion_rejection_reasons"
    ]
    assert suggestion["suggested_grid_shapes"] == [[4, 6, 12], [8, 12, 24]]
    assert suggestion["suggested_next_grid_cell_count"] == 2304
    assert suggestion["recommended_time_effective_frame_dt_values"] == [
        0.004,
        0.002,
    ]
    assert suggestion["time_refinement_action"] == (
        "fix_grid_resolution_before_reducing_timestep"
    )
    assert suggestion["potential_solve_action"] == "no_potential_residual_blocker"


def test_committed_imported_drb_movie_refinement_publication_summary_locks_grid_blocker() -> None:
    report_path = (
        REPO_ROOT
        / "docs"
        / "data"
        / "essos_imported_drb_movie_refinement_publication_artifacts"
        / "data"
        / "essos_imported_drb_movie_refinement_publication_summary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suggestion = report["next_campaign_suggestion"]

    assert report["publication_ready"] is False
    assert report["grid_refinement_passed"] is False
    assert report["time_refinement_passed"] is True
    assert report["grid_refinement_diagnostics"]["report_count"] == 2
    assert report["time_refinement_diagnostics"]["report_count"] == 2
    assert report["grid_refinement_diagnostics"]["spectral_resolution_passed"] is False
    assert report["time_refinement_diagnostics"]["spectral_resolution_passed"] is True
    assert report["grid_refinement_diagnostics"]["max_relative_metric_change"] > 0.90
    assert report["time_refinement_diagnostics"]["max_relative_metric_change"] < 0.10
    assert report["movie_promotion_rejection_reasons"] == [
        "movie_grid_refinement_not_passed",
        "movie_grid_spectral_resolution_not_passed",
        "spectral_edge_band_power_fraction_above_limit",
    ]
    assert suggestion["suggested_grid_shapes"] == [[8, 12, 24], [16, 24, 48]]
    assert suggestion["suggested_next_grid_cell_count"] == 18432
    assert suggestion["recommended_time_effective_frame_dt_values"] == [
        0.004,
        0.002,
    ]
    assert suggestion["time_refinement_action"] == (
        "reuse_current_timestep_pair_after_grid_change"
    )


def test_committed_imported_drb_movie_refinement_16x_summary_narrows_grid_blocker() -> None:
    report_path = (
        REPO_ROOT
        / "docs"
        / "data"
        / "essos_imported_drb_movie_refinement_16x_candidate_artifacts"
        / "data"
        / "essos_imported_drb_movie_refinement_16x_candidate_summary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suggestion = report["next_campaign_suggestion"]
    dominant_grid_blockers = suggestion["dominant_grid_blockers"]

    assert report["publication_ready"] is False
    assert report["grid_refinement_passed"] is False
    assert report["time_refinement_passed"] is True
    assert report["movie_promotion_rejection_reasons"] == [
        "movie_grid_refinement_not_passed"
    ]
    assert report["grid_refinement_diagnostics"]["spectral_resolution_passed"] is True
    assert report["time_refinement_diagnostics"]["spectral_resolution_passed"] is True
    assert report["grid_refinement_diagnostics"]["max_relative_metric_change"] < 0.55
    assert report["time_refinement_diagnostics"]["max_relative_metric_change"] < 0.10
    assert [item["metric"] for item in dominant_grid_blockers] == [
        "spectral_centroid_poloidal_index"
    ]
    assert suggestion["suggested_grid_shapes"] == [[16, 24, 48], [16, 48, 48]]
    assert suggestion["suggested_next_grid_cell_count"] == 36864
    assert suggestion["time_refinement_action"] == (
        "reuse_current_timestep_pair_after_grid_change"
    )


def test_committed_imported_drb_movie_refinement_poloidal_summary_locks_residual_blocker() -> None:
    report_path = (
        REPO_ROOT
        / "docs"
        / "data"
        / "essos_imported_drb_movie_refinement_poloidal_candidate_artifacts"
        / "data"
        / "essos_imported_drb_movie_refinement_poloidal_candidate_summary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suggestion = report["next_campaign_suggestion"]
    reasons = report["movie_promotion_rejection_reasons"]

    assert report["publication_ready"] is False
    assert report["grid_refinement_passed"] is False
    assert report["time_refinement_passed"] is False
    assert "movie_grid_refinement_not_passed" in reasons
    assert "movie_time_refinement_not_passed" in reasons
    assert "movie_time_spectral_resolution_not_passed" in reasons
    assert report["grid_refinement_diagnostics"]["spectral_resolution_passed"] is True
    assert report["time_refinement_diagnostics"]["spectral_resolution_passed"] is False
    assert suggestion["recommended_potential_iterations"] == 6144
    assert suggestion["potential_solve_action"] == (
        "check_potential_solver_after_primary_physics_metric_refinement"
    )
    assert suggestion["suggested_grid_shapes"] == [[16, 48, 48], [16, 96, 48]]
    assert suggestion["time_refinement_action"] == (
        "halve_effective_frame_dt_after_grid_candidate"
    )


def test_committed_imported_drb_movie_refinement_poloidal_6144_summary_keeps_residual_blocker() -> None:
    report_path = (
        REPO_ROOT
        / "docs"
        / "data"
        / "essos_imported_drb_movie_refinement_poloidal_6144_artifacts"
        / "data"
        / "essos_imported_drb_movie_refinement_poloidal_6144_summary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suggestion = report["next_campaign_suggestion"]

    assert report["publication_ready"] is False
    assert report["grid_refinement_passed"] is False
    assert report["time_refinement_passed"] is False
    assert report["movie_promotion_rejection_reasons"] == [
        "movie_grid_refinement_not_passed",
        "movie_time_refinement_not_passed",
    ]
    assert suggestion["potential_solve_action"] == (
        "check_potential_solver_after_primary_physics_metric_refinement"
    )
    assert suggestion["recommended_potential_iterations"] == 12288
    assert [item["metric"] for item in suggestion["dominant_time_blockers"]] == [
        "final_potential_residual_l2"
    ]
    assert "final_potential_residual_l2" in [
        item["metric"] for item in suggestion["dominant_grid_blockers"]
    ]


def test_committed_imported_drb_movie_refinement_poloidal_jacobi_summary_closes_residual_blocker() -> None:
    report_path = (
        REPO_ROOT
        / "docs"
        / "data"
        / "essos_imported_drb_movie_refinement_poloidal_jacobi_artifacts"
        / "data"
        / "essos_imported_drb_movie_refinement_poloidal_jacobi_summary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suggestion = report["next_campaign_suggestion"]

    assert report["publication_ready"] is False
    assert report["grid_refinement_passed"] is False
    assert report["time_refinement_passed"] is True
    assert report["movie_promotion_rejection_reasons"] == [
        "movie_grid_refinement_not_passed"
    ]
    assert suggestion["potential_solve_action"] == "no_potential_residual_blocker"
    assert suggestion["dominant_time_blockers"] == []
    assert [item["metric"] for item in suggestion["dominant_grid_blockers"]] == [
        "spectral_centroid_poloidal_index"
    ]
    assert suggestion["suggested_grid_shapes"] == [[16, 48, 48], [16, 96, 48]]


def _load_imported_drb_movie_refinement_summary_example():
    module_path = (
        REPO_ROOT
        / "examples"
        / "geometry-3D"
        / "essos-field-lines"
        / "imported_drb_movie_refinement_summary.py"
    )
    source = module_path.read_text(encoding="utf-8").replace(
        "RUN_EXAMPLE = True",
        "RUN_EXAMPLE = False",
        1,
    )
    spec = importlib.util.spec_from_loader(
        "imported_drb_movie_refinement_summary_example",
        loader=None,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(module_path)
    sys.modules[spec.name] = module
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


def _load_imported_drb_movie_refinement_campaign_example():
    module_path = (
        REPO_ROOT
        / "examples"
        / "geometry-3D"
        / "essos-field-lines"
        / "imported_drb_movie_refinement_campaign.py"
    )
    source = module_path.read_text(encoding="utf-8").replace(
        "RUN_EXAMPLE = True",
        "RUN_EXAMPLE = False",
        1,
    )
    spec = importlib.util.spec_from_loader(
        "imported_drb_movie_refinement_campaign_example",
        loader=None,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(module_path)
    sys.modules[spec.name] = module
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


def test_imported_connection_length_refinement_campaign_is_self_contained(tmp_path: Path) -> None:
    artifacts = create_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "connection_length_refinement",
        level_shapes=((4, 6, 8), (8, 12, 16), (16, 24, 32)),
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    arrays = np.load(artifacts.arrays_npz_path)

    assert report["passed"] is True
    assert report["manufactured"] is True
    assert report["diagnostics"]["level_count"] == 3
    assert report["finest_normalized_rms_error"] < 0.02
    assert report["finest_normalized_linf_error"] < 0.05
    assert report["minimum_observed_order_actual"] > 1.5
    assert report["observed_order_required"] is False
    assert report["diagnostics"]["observed_order_available"] is True
    assert arrays["level_0"].shape == (4, 6, 8)
    assert arrays["level_2"].shape == (16, 24, 32)
    assert arrays["pair_normalized_rms_error"][1] < arrays["pair_normalized_rms_error"][0]
    assert artifacts.plot_png_path.exists()


def test_connection_length_refinement_can_use_coordinate_interpolation() -> None:
    coarse_coordinates = _logical_coordinates((3, 4, 8))
    fine_coordinates = _logical_coordinates((6, 8, 16))
    coarse = 10.0 + 2.0 * coarse_coordinates["minor_radius"]
    fine = 10.0 + 2.0 * fine_coordinates["minor_radius"]

    block_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )
    coordinate_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        coordinate_levels=(coarse_coordinates, fine_coordinates),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )

    assert block_report["restriction_method"] == "block_average"
    assert coordinate_report["restriction_method"] == "coordinate_interpolation"
    assert block_report["passed"] is False
    assert coordinate_report["passed"] is True
    assert coordinate_report["pair_reports"][0]["normalized_linf_error"] < 1.0e-14


def test_connection_length_refinement_can_require_observed_order_for_promotion(tmp_path: Path) -> None:
    coarse_coordinates = _logical_coordinates((3, 4, 8))
    fine_coordinates = _logical_coordinates((6, 8, 16))
    coarse = 10.0 + 2.0 * coarse_coordinates["minor_radius"]
    fine = 10.0 + 2.0 * fine_coordinates["minor_radius"]

    advisory_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        coordinate_levels=(coarse_coordinates, fine_coordinates),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )
    promotion_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        coordinate_levels=(coarse_coordinates, fine_coordinates),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
        require_observed_order=True,
    )
    packaged = create_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "test_connection_length_refinement_requires_order",
        level_shapes=((4, 6, 8), (8, 12, 16), (16, 24, 32)),
        require_observed_order=True,
    )
    packaged_report = json.loads(packaged.report_json_path.read_text(encoding="utf-8"))

    assert advisory_report["passed"] is True
    assert advisory_report["observed_order_available"] is False
    assert promotion_report["passed"] is False
    assert promotion_report["observed_order_required"] is True
    assert promotion_report["observed_order_available"] is False
    assert packaged_report["passed"] is True
    assert packaged_report["observed_order_required"] is True
    assert packaged_report["diagnostics"]["observed_order_available"] is True


def test_live_imported_connection_length_refinement_uses_geometry_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int, int, int]] = []

    def fake_geometry(**kwargs):
        nx = int(kwargs["nx"])
        ny = int(kwargs["ny"])
        nz = int(kwargs["nz"])
        map_source = str(kwargs["map_source"])
        calls.append((map_source, nx, ny, nz))
        coordinates = _logical_coordinates((nx, ny, nz))
        level = 12.5 + 0.1 * coordinates["minor_radius"]
        return SimpleNamespace(
            connection_length=level,
            maps=SimpleNamespace(dphi=2.0 * np.pi / float(ny)),
            minor_radius=coordinates["minor_radius"],
            toroidal_angle=coordinates["toroidal_angle"],
            poloidal_angle=coordinates["poloidal_angle"],
            metadata={
                "map_source": map_source,
                "shape": [nx, ny, nz],
                "geometry_family": "test_imported_geometry",
            },
        )

    monkeypatch.setattr(
        imported_fci_campaign,
        "build_essos_imported_fci_geometry",
        fake_geometry,
    )

    levels = build_live_essos_imported_connection_length_levels(
        map_source="hybrid",
        level_shapes=((2, 2, 4), (4, 4, 8)),
    )

    assert calls == [("hybrid", 2, 2, 4), ("hybrid", 4, 4, 8)]
    assert levels.labels == ("hybrid_2x2x4", "hybrid_4x4x8")
    assert levels.levels[0].shape == (2, 2, 4)
    assert levels.coordinates[0]["minor_radius"].shape == (2, 2, 4)
    assert levels.metadata[0]["geometry_family"] == "test_imported_geometry"

    artifacts = create_live_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "live_connection_length_refinement",
        map_source="hybrid",
        level_shapes=((2, 2, 4), (4, 4, 8)),
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["live_imported"] is True
    assert report["manufactured"] is False
    assert report["map_source"] == "hybrid"
    assert report["source"] == "live imported connection-length refinement gate"
    assert report["diagnostics"]["level_labels"] == [
        "hybrid_2x2x4",
        "hybrid_4x4x8",
    ]
    assert report["diagnostics"]["restriction_method"] == "coordinate_interpolation"
    assert report["passed"] is True
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_live_imported_connection_length_refinement_can_use_step_length_per_radian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_geometry(**kwargs):
        nx = int(kwargs["nx"])
        ny = int(kwargs["ny"])
        nz = int(kwargs["nz"])
        coordinates = _logical_coordinates((nx, ny, nz))
        dphi = 2.0 * np.pi / float(ny)
        level_per_radian = 3.0 + 0.2 * coordinates["minor_radius"]
        return SimpleNamespace(
            connection_length=99.0 + dphi * level_per_radian,
            adjacent_step_length=dphi * level_per_radian,
            target_exit_length=50.0 + coordinates["minor_radius"],
            maps=SimpleNamespace(dphi=dphi),
            minor_radius=coordinates["minor_radius"],
            toroidal_angle=coordinates["toroidal_angle"],
            poloidal_angle=coordinates["poloidal_angle"],
            metadata={
                "map_source": kwargs["map_source"],
                "shape": [nx, ny, nz],
                "geometry_family": "test_imported_geometry",
            },
        )

    monkeypatch.setattr(
        imported_fci_campaign,
        "build_essos_imported_fci_geometry",
        fake_geometry,
    )

    artifacts = create_live_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "live_connection_length_refinement_per_radian",
        map_source="vmec",
        connection_quantity="parallel_step_per_toroidal_radian",
        level_shapes=((2, 2, 4), (4, 4, 8)),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    arrays = np.load(artifacts.arrays_npz_path)

    assert report["connection_quantity"] == "parallel_step_per_toroidal_radian"
    assert report["diagnostics"]["restriction_method"] == "coordinate_interpolation"
    assert report["passed"] is True
    assert np.max(arrays["level_0"]) < 4.0


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_fieldline_import_generates_portable_artifacts(tmp_path: Path) -> None:
    artifacts = create_essos_fieldline_import_package(
        output_root=tmp_path / "essos_import",
        n_field_lines=3,
        times_to_trace=768,
        maxtime=150.0,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    bundle = load_essos_field_line_bundle_npz(artifacts.arrays_npz_path, metadata=report["metadata"])

    assert report["passed"] is True
    assert report["source"] == "ESSOS external field and field-line import"
    assert report["metadata"]["field_model"] == "essos.fields.BiotSavart"
    assert report["metadata"]["tracing_model"] == "essos.dynamics.Tracing(FieldLineAdaptative)"
    assert bundle.trajectories_xyz.shape == (3, 768, 3)
    assert bundle.field_sample_xyz.shape == bundle.field_sample_b_xyz.shape
    assert bundle.poincare_point_count == report["poincare_point_count"]
    assert bundle.poincare_point_count > 0
    assert np.all(np.isfinite(bundle.trajectories_xyz))
    assert np.all(np.isfinite(bundle.field_sample_b_xyz))
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_imported_fci_maps_feed_native_sheath_and_neutral_gates(tmp_path: Path) -> None:
    geometry = build_essos_imported_fci_geometry(
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
    )

    assert geometry.shape == (3, 4, 6)
    assert geometry.metadata["geometry_family"] == "essos_imported_vmec_qa_fci"
    assert geometry.metadata["coordinate_model"] == "scaled_vmec_fourier_flux_surfaces"
    assert geometry.metadata["surface_nonaxisymmetric_major_rms"] > 1.0e-2
    assert np.all(np.isfinite(np.asarray(geometry.magnetic_field_magnitude)))
    assert np.all(np.isfinite(np.asarray(geometry.connection_length)))
    assert geometry.adjacent_step_length is not None
    assert geometry.target_exit_length is not None
    assert geometry.forward_target_exit_length is not None
    assert geometry.backward_target_exit_length is not None
    assert np.any(np.isfinite(np.asarray(geometry.adjacent_step_length)))
    assert np.any(np.isfinite(np.asarray(geometry.target_exit_length)))
    assert np.any(np.isfinite(np.asarray(geometry.forward_target_exit_length)))
    assert np.any(np.isfinite(np.asarray(geometry.backward_target_exit_length)))
    assert 0.05 < float(np.mean(np.asarray(geometry.maps.forward_boundary, dtype=bool))) < 0.95
    assert 0.05 < float(np.mean(np.asarray(geometry.maps.backward_boundary, dtype=bool))) < 0.95

    artifacts = create_essos_imported_fci_campaign_package(
        output_root=tmp_path / "essos_imported_fci",
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS-imported field-line maps with jax_drb FCI closures"
    assert report["endpoint_length_diagnostics"]["passed"] is True
    assert report["target_label_diagnostics"]["passed"] is True
    assert report["target_label_diagnostics"]["endpoint_count_matches_target_labels"] is True
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    arrays = np.load(artifacts.arrays_npz_path)
    assert "target_label_toroidal" in arrays.files
    assert "target_exit_toroidal" in arrays.files
    assert "adjacent_step_toroidal" in arrays.files
    assert np.max(arrays["target_label_toroidal"]) > 0.0
    assert np.any(np.isfinite(arrays["target_exit_toroidal"]))


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil/VMEC inputs are not available")
def test_essos_imported_fci_map_sources_expose_coil_vmec_and_hybrid_semantics() -> None:
    geometries = {
        source: build_essos_imported_fci_geometry(
            map_source=source,
            nx=3,
            ny=4,
            nz=6,
            rho_min=0.12,
            rho_max=0.34,
            maxtime=40.0,
            times_to_trace=160,
        )
        for source in ("coil", "vmec", "hybrid")
    }

    for source, geometry in geometries.items():
        assert geometry.shape == (3, 4, 6)
        assert geometry.metadata["map_source"] == source
        assert np.all(np.isfinite(np.asarray(geometry.maps.forward_x)))
        assert np.all(np.isfinite(np.asarray(geometry.maps.forward_z)))
        assert np.all(np.isfinite(np.asarray(geometry.maps.backward_x)))
        assert np.all(np.isfinite(np.asarray(geometry.maps.backward_z)))
        assert np.all(np.isfinite(np.asarray(geometry.magnetic_field_magnitude)))
        assert np.all(np.isfinite(np.asarray(geometry.connection_length)))
        assert geometry.metadata["surface_nonaxisymmetric_major_rms"] > 1.0e-2

    coil = geometries["coil"]
    vmec = geometries["vmec"]
    hybrid = geometries["hybrid"]
    coil_forward_fraction = float(np.mean(np.asarray(coil.maps.forward_boundary, dtype=bool)))
    coil_backward_fraction = float(np.mean(np.asarray(coil.maps.backward_boundary, dtype=bool)))
    vmec_forward_fraction = float(np.mean(np.asarray(vmec.maps.forward_boundary, dtype=bool)))
    vmec_backward_fraction = float(np.mean(np.asarray(vmec.maps.backward_boundary, dtype=bool)))
    hybrid_forward_fraction = float(np.mean(np.asarray(hybrid.maps.forward_boundary, dtype=bool)))
    hybrid_backward_fraction = float(np.mean(np.asarray(hybrid.maps.backward_boundary, dtype=bool)))

    assert 0.05 < coil_forward_fraction < 0.95
    assert 0.05 < coil_backward_fraction < 0.95
    assert vmec_forward_fraction == 0.0
    assert vmec_backward_fraction == 0.0
    assert hybrid_forward_fraction == coil_forward_fraction
    assert hybrid_backward_fraction == coil_backward_fraction
    assert np.allclose(np.asarray(hybrid.maps.forward_x), np.asarray(vmec.maps.forward_x))
    assert np.allclose(np.asarray(hybrid.maps.forward_z), np.asarray(vmec.maps.forward_z))
    assert not np.allclose(np.asarray(coil.maps.forward_z), np.asarray(vmec.maps.forward_z))
    assert float(np.min(np.asarray(vmec.connection_length))) > 0.0
    assert float(np.max(np.asarray(vmec.magnetic_field_magnitude)) / np.min(np.asarray(vmec.magnetic_field_magnitude))) > 1.01


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_fieldline_poincare_quantifies_scaled_vmec_surface_registration(tmp_path: Path) -> None:
    artifacts = create_essos_vmec_fieldline_surface_package(
        output_root=tmp_path / "essos_vmec_fieldline_surface",
        rho_min=0.20,
        rho_max=0.82,
        n_surfaces=3,
        ntheta_surface=96,
        maxtime=180.0,
        times_to_trace=768,
        sections=(0.0, float(np.pi / 2.0)),
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS coil field-line tracing compared with scaled Landreman-Paul QA VMEC surfaces"
    assert report["surface_nonaxisymmetric_major_rms"] > 5.0e-2
    assert report["poincare_point_count"] >= 2 * report["n_surfaces"]
    assert np.isfinite(report["same_surface_distance_normalized_p95"])
    assert np.isfinite(report["nearest_surface_distance_normalized_p95"])
    assert isinstance(report["fieldline_surface_match_passed"], bool)
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA VMEC wout are not available")
def test_essos_vmec_fieldline_poincare_preserves_scaled_vmec_surfaces(tmp_path: Path) -> None:
    artifacts = create_essos_vmec_fieldline_surface_package(
        output_root=tmp_path / "essos_vmec_equilibrium_fieldline_surface",
        field_source="vmec",
        rho_min=0.20,
        rho_max=0.82,
        n_surfaces=3,
        ntheta_surface=96,
        maxtime=180.0,
        times_to_trace=768,
        sections=(0.0, float(np.pi / 2.0)),
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["field_source"] == "vmec"
    assert report["fieldline_surface_match_passed"] is True
    assert report["surface_nonaxisymmetric_major_rms"] > 5.0e-2
    assert report["poincare_point_count"] >= 2 * report["n_surfaces"]
    assert report["fieldline_s_drift_max"] < 1.0e-7
    assert report["same_surface_distance_normalized_p95"] < 5.0e-2
    assert report["nearest_surface_distance_normalized_p95"] < 5.0e-2
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_imported_maps_feed_pytree_jvp_rhs_gate(tmp_path: Path) -> None:
    artifacts = create_essos_imported_pytree_campaign_package(
        output_root=tmp_path / "essos_imported_pytree",
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
        steps=3,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS-imported field-line maps with JAXDRB fixed-layout PyTree RHS"
    assert report["jvp_relative_error"] < 1.0e-2
    assert report["vmap_serial_linf"] < 1.0e-6
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_imported_maps_generate_drb_movie_gate(tmp_path: Path) -> None:
    artifacts = create_essos_imported_drb_movie_package(
        output_root=tmp_path / "essos_imported_drb_movie",
        nx=3,
        ny=4,
        nz=8,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=32.0,
        times_to_trace=120,
        frames=4,
        substeps_per_frame=2,
        dt=2.0e-3,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS-imported Landreman-Paul QA coil FCI maps with JAXDRB fixed-layout DRB transient"
    assert report["publication_ready"] is False
    assert report["movie_evidence_role"] == "movie_showcase_pending_connection_grid_time_refinement"
    assert "coil_connection_length_refinement_not_promotion_ready" in report[
        "movie_promotion_rejection_reasons"
    ]
    assert report["final_potential_residual_l2"] < 5.0
    assert report["final_fluctuation_rms"] > 1.0e-4
    assert report["radial_flux_abs_mean"] > 1.0e-8
    assert report["radial_flux_rms"] >= report["radial_flux_abs_mean"]
    assert report["radial_flux_peak_abs"] >= report["radial_flux_abs_mean"]
    assert 0.0 <= report["radial_flux_cancellation_ratio"] <= 1.0
    assert 0.0 <= report["radial_flux_positive_fraction"] <= 1.0
    assert report["spectral_poloidal_mode_count"] == 4
    assert report["spectral_toroidal_mode_count"] == 5
    assert report["spectral_centroid_poloidal_index"] >= 0.0
    assert report["spectral_centroid_toroidal_index"] >= 0.0
    assert 0.0 <= report["spectral_centroid_poloidal_fraction"] <= 1.0
    assert 0.0 <= report["spectral_centroid_toroidal_fraction"] <= 1.0
    assert 0.0 <= report["spectral_edge_band_power_fraction"] <= 1.0
    assert report["low_mode_window_covers_grid"] is True
    assert report["particle_recycling_relative_error"] < 1.0e-10
    assert report["neutral_particle_relative_error"] < 1.0e-10
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.snapshot_png_path.exists()
    assert artifacts.diagnostics_png_path.exists()
    assert artifacts.poster_png_path.exists()
    assert artifacts.movie_gif_path.exists()
