from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation import (
    build_neutral_mixed_term_balance_campaign_report,
    create_neutral_mixed_term_balance_campaign_package,
    save_neutral_mixed_term_balance_campaign_plot,
    write_neutral_mixed_diagnostic_input,
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


def test_neutral_mixed_term_balance_report_can_ingest_hermes_diagnostic_netcdf(tmp_path: Path) -> None:
    from netCDF4 import Dataset

    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    diagnostic_path = tmp_path / "BOUT.dmp.0.nc"
    shape = (2, 10, 14, 10)
    with Dataset(diagnostic_path, "w") as dataset:
        dataset.createDimension("t", shape[0])
        dataset.createDimension("x", shape[1])
        dataset.createDimension("y", shape[2])
        dataset.createDimension("z", shape[3])
        for name, scale in {
            "ddt(NVh)": 1.0,
            "SNVh": 0.0,
            "SNVh_pressure_gradient": -4.0,
            "mfh_visc_par_ylow": 2.0,
            "mfh_visc_perp_xlow": 3.0,
        }.items():
            variable = dataset.createVariable(name, "f8", ("t", "x", "y", "z"))
            variable[:] = scale * np.ones(shape, dtype=np.float64)

    report = build_neutral_mixed_term_balance_campaign_report(
        input_path=input_path,
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=_REFERENCE_ARRAYS,
        hermes_diagnostic_nc=diagnostic_path,
    )

    diagnostics = report["hermes_diagnostic_outputs"]
    assert "ddt(NVh)" in diagnostics["variables_present"]
    assert "SNVh_pressure_gradient" in diagnostics["variables_present"]
    assert "mfh_visc_perp_ylow" in diagnostics["variables_missing"]
    assert diagnostics["field_metrics"]["SNVh_pressure_gradient"]["max_abs"] == 4.0
    assert diagnostics["field_metrics"]["mfh_visc_par_ylow"]["max_abs"] == 2.0
    reconstruction = diagnostics["matched_reconstructions"]["pressure_gradient"]
    assert reconstruction["field_metrics"]["max_abs"] >= 0.0
    assert len(reconstruction["lineout"]) == len(report["active_y_indices"])


def test_write_neutral_mixed_diagnostic_input_enables_hermes_outputs(tmp_path: Path) -> None:
    source = _write_neutral_mixed_input(tmp_path / "source.inp")
    target = write_neutral_mixed_diagnostic_input(source, tmp_path / "data" / "BOUT.inp")

    text = target.read_text(encoding="utf-8")
    assert "nout = 1" in text
    assert "[h]" in text
    assert "output_ddt = true" in text
    assert "diagnose = true" in text
