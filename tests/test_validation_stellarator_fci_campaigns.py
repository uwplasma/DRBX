from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

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


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_imported_fci_example_cli_resolves_source_specific_artifact_defaults(capsys) -> None:
    module = _load_imported_fci_campaign_example()

    args = module.parse_args(["--all-map-sources", "--dry-run", "--nx", "3", "--ny", "4", "--nz", "8"])
    settings = module.build_run_settings(args)

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

    assert module.main(["--map-source", "hybrid", "--dry-run", "--output-root", "tmp/hybrid", "--case-label", "custom"]) == 0
    captured = capsys.readouterr()
    assert "map_source=hybrid" in captured.out
    assert "output_root=tmp/hybrid" in captured.out
    assert "case_label=custom" in captured.out


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
