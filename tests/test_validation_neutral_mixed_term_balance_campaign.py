from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import (
    build_neutral_mixed_term_balance_campaign_report,
    create_neutral_mixed_term_balance_campaign_package,
    save_neutral_mixed_term_balance_campaign_plot,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_ARRAYS = _REPO_ROOT / "references" / "baselines" / "reference_arrays" / "neutral_mixed_one_step.npz"


def _write_neutral_mixed_input(path: Path) -> Path:
    path.write_text(
        """
nout = 15
timestep = 20

[mesh]
nx = 10
ny = 10
nz = 10

dx = 1e-3
dy = 1e-3
dz = 1e-3

yn = y / (2π)
zn = z / (2π)

J = 1

[solver]
mxstep = 1000

[model]
components = h

[h]
type = neutral_mixed

[Nh]
function = exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2 - (mesh:zn - 0.5)^2)

[Ph]
function = 0.1 * Nh:function
""",
        encoding="utf-8",
    )
    return path


def test_build_neutral_mixed_term_balance_campaign_report_has_named_terms(tmp_path: Path) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")

    report = build_neutral_mixed_term_balance_campaign_report(
        input_path=input_path,
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=_REFERENCE_ARRAYS,
    )

    assert report["case_name"] == "neutral_mixed_one_step"
    assert report["field"] == "NVh"
    assert report["final_momentum_error"]["max_abs"] == 0.0
    reference_terms = report["reference_balance"]["lineouts"]
    assert "parallel_inertia" in reference_terms
    assert "pressure_gradient" in reference_terms
    assert "residual_rate" in reference_terms
    assert report["reference_balance"]["term_metrics"]["residual_rate"]["max_abs"] >= 0.0


def test_create_neutral_mixed_term_balance_campaign_package_writes_outputs(tmp_path: Path) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")

    artifacts = create_neutral_mixed_term_balance_campaign_package(
        output_root=tmp_path / "artifacts",
        reference_root=None,
        input_path=input_path,
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=_REFERENCE_ARRAYS,
    )

    report = build_neutral_mixed_term_balance_campaign_report(
        input_path=input_path,
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=_REFERENCE_ARRAYS,
    )
    plot = save_neutral_mixed_term_balance_campaign_plot(report, tmp_path / "plot.png")

    assert artifacts.report_json_path.exists()
    assert artifacts.report_npz_path.exists()
    assert artifacts.report_plot_png_path.exists()
    assert plot.exists()
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert payload["field"] == "NVh"
