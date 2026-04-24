from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import (
    build_implicit_solver_profile_audit_report,
    create_implicit_solver_profile_audit_package,
)


def test_build_implicit_solver_profile_audit_report_validates_planned_jacobian() -> None:
    report = build_implicit_solver_profile_audit_report(
        active_shape=(4, 1, 6),
        field_count=2,
        repeats=1,
    )

    assert report["case"] == "implicit_solver_profile_audit"
    assert report["max_jacobian_abs_diff"] == 0.0
    assert report["jacobian_build_seconds"]["planned_serial_mean"] >= 0.0
    assert report["newton"]["residual_inf_norm"] < 1.0e-9
    assert report["newton"]["solution_max_abs_error"] < 1.0e-8
    assert report["newton"]["residual_evaluation_count"] >= 1
    assert report["newton"]["jacobian_refresh_count"] >= 1


def test_create_implicit_solver_profile_audit_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_implicit_solver_profile_audit_package(output_root=tmp_path / "artifacts")

    assert artifacts.report_json_path.exists()
    assert artifacts.report_plot_png_path.exists()
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert payload["case"] == "implicit_solver_profile_audit"
    assert payload["newton"]["fallback_used"] is False
