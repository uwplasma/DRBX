from __future__ import annotations

import json
from pathlib import Path

from drbx.validation import (
    create_stellarator_fci_geometry_campaign_package,
    create_stellarator_fci_suite_campaign_package,
    create_stellarator_sheath_recycling_campaign_package,
)


def test_stellarator_fci_geometry_campaign_generates_passing_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_fci_geometry_campaign_package(
        output_root=tmp_path / "geometry", nx=10, ny=8, nz=16
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    assert report["map_diagnostics"]["radial_shift_linf_cells"] > 0.0


def test_stellarator_fci_suite_campaign_generates_multi_configuration_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_fci_suite_campaign_package(
        output_root=tmp_path / "suite", nx=10, ny=8, nz=16
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert len(report["configuration_labels"]) == 3
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_sheath_recycling_campaign_generates_balance_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_sheath_recycling_campaign_package(
        output_root=tmp_path / "sheath_recycling", nx=10, ny=8, nz=16
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["total_particle_loss"] > 0.0
    assert report["particle_recycling_relative_error"] < 1.0e-12
    assert report["current_balance_relative_error"] < 1.0e-12
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
