from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import (
    create_stellarator_fci_geometry_campaign_package,
    create_stellarator_fci_operator_campaign_package,
    create_stellarator_fci_suite_campaign_package,
    create_stellarator_drb_pytree_campaign_package,
    create_stellarator_neutral_physics_campaign_package,
    create_stellarator_sheath_recycling_campaign_package,
    create_stellarator_sol_showcase_package,
    create_stellarator_vorticity_campaign_package,
)


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
