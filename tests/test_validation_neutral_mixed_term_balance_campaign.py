from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jax_drb.validation import (
    build_neutral_mixed_substep_hybrid_report,
    build_neutral_mixed_term_balance_campaign_report,
    create_neutral_mixed_term_balance_campaign_package,
    save_neutral_mixed_term_balance_campaign_plot,
    write_neutral_mixed_substep_hybrid_json,
    write_neutral_mixed_diagnostic_input,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_ARRAYS = _REPO_ROOT / "references" / "baselines" / "reference_arrays" / "neutral_mixed_one_step.npz"


def _write_synthetic_native_history_with_target_drift(path: Path) -> Path:
    with np.load(_REFERENCE_ARRAYS) as reference:
        metadata = str(reference["__metadata__"].item())
        density = np.asarray(reference["var__Nh"], dtype=np.float64).copy()
        pressure = np.asarray(reference["var__Ph"], dtype=np.float64).copy()
        momentum = np.asarray(reference["var__NVh"], dtype=np.float64).copy()

    final = -1
    x_index = 5
    z_index = 5
    lower_target_y = 0
    upper_target_y = density.shape[2] - 1
    density[final, x_index, lower_target_y, z_index] += 2.0e-2
    pressure[final, x_index, lower_target_y, z_index] += 3.0e-3
    momentum[final, x_index, upper_target_y, z_index] += 1.0e-3
    np.savez_compressed(
        path,
        __metadata__=metadata,
        var__Nh=density,
        var__Ph=pressure,
        var__NVh=momentum,
    )
    return path


def _synthetic_native_history_with_scaled_target_drift(scale: float) -> dict[str, object]:
    with np.load(_REFERENCE_ARRAYS) as reference:
        metadata = json.loads(str(reference["__metadata__"].item()))
        time_points = np.asarray(metadata["time_points"], dtype=np.float64)
        density = np.asarray(reference["var__Nh"], dtype=np.float64).copy()
        pressure = np.asarray(reference["var__Ph"], dtype=np.float64).copy()
        momentum = np.asarray(reference["var__NVh"], dtype=np.float64).copy()
    final = -1
    density[final, 5, 0, 5] += 2.0e-2 * float(scale)
    pressure[final, 5, 0, 5] += 3.0e-3 * float(scale)
    momentum[final, 5, -1, 5] += 1.0e-3 * float(scale)
    return {
        "time_points": time_points,
        "Nh": density,
        "Ph": pressure,
        "NVh": momentum,
    }


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
    assert report["active_x_indices"]
    assert report["final_momentum_error"]["max_abs"] == 0.0
    reference_terms = report["reference_balance"]["lineouts"]
    assert "parallel_inertia" in reference_terms
    assert "pressure_gradient" in reference_terms
    assert "residual_rate" in reference_terms
    assert "pressure_gradient" in report["term_delta"]["lineouts"]
    assert report["offender_register"]["native_minus_hermes_term_delta"][0]["target_adjacent_max_abs"] >= 0.0
    assert report["offender_register"]["dominant_residual_cells"]
    assert report["reference_balance"]["term_metrics"]["residual_rate"]["max_abs"] >= 0.0
    field_register = report["final_field_error_register"]
    assert field_register["target_y_indices"]
    assert field_register["ranked_by_target_adjacent_max_abs"][0]["field"] in {"Nh", "Ph", "NVh"}
    assert set(field_register["fields"]) == {"Nh", "Ph", "NVh"}
    assert field_register["fields"]["Nh"]["target_adjacent_max_abs"] >= 0.0
    assert len(field_register["fields"]["Ph"]["lineout"]) == len(report["active_y_indices"])
    state_driver_register = report["state_driver_register"]
    assert set(state_driver_register["state_rate_errors"]) == {"Nh", "Ph", "NVh"}
    assert state_driver_register["ranked_state_rate_errors"][0]["field"] in {"Nh", "Ph", "NVh"}
    assert "pressure_to_pressure_gradient" in state_driver_register["momentum_driver_deltas"]
    assert "momentum_to_parallel_viscosity" in state_driver_register["momentum_driver_deltas"]
    pressure_driver = state_driver_register["momentum_driver_deltas"]["pressure_to_pressure_gradient"]
    assert pressure_driver["field"] == "Ph"
    assert pressure_driver["term"] == "pressure_gradient"
    assert len(pressure_driver["term_delta_lineout"]) == len(report["active_y_indices"])


def test_neutral_mixed_term_balance_register_ranks_target_adjacent_state_drift(tmp_path: Path) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    native_arrays = _write_synthetic_native_history_with_target_drift(tmp_path / "native_target_drift.npz")

    report = build_neutral_mixed_term_balance_campaign_report(
        input_path=input_path,
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=native_arrays,
    )

    field_register = report["final_field_error_register"]
    ranked_fields = field_register["ranked_by_target_adjacent_max_abs"]
    assert ranked_fields[0]["field"] == "Nh"
    assert ranked_fields[0]["target_adjacent_max_abs"] == pytest.approx(2.0e-2, rel=1.0e-12, abs=1.0e-12)
    assert field_register["fields"]["Nh"]["interior_max_abs"] == 0.0
    assert field_register["fields"]["Ph"]["target_adjacent_max_abs"] == pytest.approx(3.0e-3, rel=1.0e-12, abs=1.0e-12)
    assert field_register["fields"]["NVh"]["target_adjacent_max_abs"] == pytest.approx(1.0e-3, rel=1.0e-12, abs=1.0e-12)

    state_register = report["state_driver_register"]
    ranked_state_rates = state_register["ranked_state_rate_errors"]
    assert ranked_state_rates[0]["field"] == "Nh"
    assert ranked_state_rates[0]["target_adjacent_max_abs"] == pytest.approx(1.0e-3, rel=1.0e-12, abs=1.0e-12)
    pressure_driver = state_register["momentum_driver_deltas"]["pressure_to_pressure_gradient"]
    assert pressure_driver["target_adjacent_max_abs"] > 0.0
    assert pressure_driver["interior_max_abs"] >= 0.0
    assert state_register["target_y_indices"] == field_register["target_y_indices"]


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
    assert "final_field_error_register" in payload
    with np.load(artifacts.report_npz_path) as arrays:
        assert "final_field_error_Nh_lineout" in arrays
        assert "final_field_error_Ph_lineout" in arrays
        assert "final_field_error_NVh_lineout" in arrays
        assert "state_rate_error_Nh_lineout" in arrays
        assert "state_driver_pressure_to_pressure_gradient_term_delta_lineout" in arrays
        assert "state_driver_momentum_to_parallel_viscosity_term_delta_lineout" in arrays


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
    comparison = diagnostics["direct_comparisons"]["SNVh_pressure_gradient"]
    assert "least_squares_scale_to_native_units" in comparison
    assert np.isfinite(comparison["least_squares_scale_to_native_units"])
    assert "scaled_difference_metrics" in comparison
    assert len(comparison["scaled_direct_lineout"]) == len(report["active_y_indices"])


def test_write_neutral_mixed_diagnostic_input_enables_hermes_outputs(tmp_path: Path) -> None:
    source = _write_neutral_mixed_input(tmp_path / "source.inp")
    target = write_neutral_mixed_diagnostic_input(source, tmp_path / "data" / "BOUT.inp")

    text = target.read_text(encoding="utf-8")
    assert "nout = 1" in text
    assert "[h]" in text
    assert "output_ddt = true" in text
    assert "diagnose = true" in text


def test_neutral_mixed_substep_hybrid_report_ranks_successes_and_failures(tmp_path: Path) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    report = build_neutral_mixed_substep_hybrid_report(
        reference_root=tmp_path / "missing_reference_root",
        input_path=input_path,
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_by_substep={
            1: _synthetic_native_history_with_scaled_target_drift(1.0),
            4: _synthetic_native_history_with_scaled_target_drift(0.25),
        },
        substeps=(1, 4, 8),
    )

    assert report["diagnostic"] == "neutral_mixed_substep_hybrid_state"
    assert report["requires_hermes"] is False
    assert report["best"]["internal_substeps"] == 4
    assert report["best"]["value"] < report["sweep_points"][0]["final_field_error_register"]["fields"]["NVh"]["max_abs"]
    failed = [point for point in report["sweep_points"] if point["status"] == "failed"]
    assert failed
    assert failed[0]["internal_substeps"] == 8
    assert failed[0]["error_type"] == "FileNotFoundError"

    successful = [point for point in report["sweep_points"] if point["status"] == "ok"]
    hybrid = successful[0]["hybrid_state_register"]
    assert set(hybrid["swaps"]) == {"Nh", "Ph", "NVh"}
    assert hybrid["ranked_by_pressure_gradient_target_adjacent_delta"][0]["swapped_field"] == "Ph"
    assert hybrid["ranked_by_parallel_viscosity_target_adjacent_delta"][0]["swapped_field"] == "NVh"
    assert successful[0]["series_errors"]["ranked_by_final_target_adjacent_max_abs"][0]["field"] == "Nh"

    path = write_neutral_mixed_substep_hybrid_json(report, tmp_path / "substeps.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["field"] == "NVh"
