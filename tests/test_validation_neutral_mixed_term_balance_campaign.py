from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.validation import (
    build_neutral_mixed_accepted_step_trace_parity_report,
    build_neutral_mixed_native_accepted_step_trace_report,
    build_neutral_mixed_substep_hybrid_report,
    build_neutral_mixed_term_balance_campaign_report,
    create_neutral_mixed_term_balance_campaign_package,
    run_neutral_mixed_hermes_accepted_step_trace,
    save_neutral_mixed_term_balance_campaign_plot,
    write_neutral_mixed_accepted_step_trace_input,
    write_neutral_mixed_accepted_step_trace_parity_json,
    write_neutral_mixed_native_accepted_step_trace_json,
    write_neutral_mixed_substep_hybrid_json,
    write_neutral_mixed_diagnostic_input,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_ARRAYS = (
    _REPO_ROOT
    / "references"
    / "baselines"
    / "reference_arrays"
    / "neutral_mixed_one_step.npz"
)
_COMMITTED_REPORT_JSON = (
    _REPO_ROOT
    / "docs"
    / "data"
    / "neutral_mixed_term_balance_campaign_artifacts"
    / "data"
    / "neutral_mixed_term_balance_campaign.json"
)


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


def _synthetic_native_history_with_scaled_target_drift(
    scale: float,
) -> dict[str, object]:
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


def test_build_neutral_mixed_term_balance_campaign_report_has_named_terms(
    tmp_path: Path,
) -> None:
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
    assert (
        report["offender_register"]["native_minus_hermes_term_delta"][0][
            "target_adjacent_max_abs"
        ]
        >= 0.0
    )
    assert report["offender_register"]["dominant_residual_cells"]
    assert (
        report["reference_balance"]["term_metrics"]["residual_rate"]["max_abs"] >= 0.0
    )
    field_register = report["final_field_error_register"]
    assert field_register["target_y_indices"]
    assert field_register["ranked_by_target_adjacent_max_abs"][0]["field"] in {
        "Nh",
        "Ph",
        "NVh",
    }
    assert set(field_register["fields"]) == {"Nh", "Ph", "NVh"}
    assert field_register["fields"]["Nh"]["target_adjacent_max_abs"] >= 0.0
    assert len(field_register["fields"]["Ph"]["lineout"]) == len(
        report["active_y_indices"]
    )
    state_driver_register = report["state_driver_register"]
    assert set(state_driver_register["state_rate_errors"]) == {"Nh", "Ph", "NVh"}
    assert state_driver_register["ranked_state_rate_errors"][0]["field"] in {
        "Nh",
        "Ph",
        "NVh",
    }
    assert (
        "pressure_to_pressure_gradient"
        in state_driver_register["momentum_driver_deltas"]
    )
    assert (
        "momentum_to_parallel_viscosity"
        in state_driver_register["momentum_driver_deltas"]
    )
    pressure_driver = state_driver_register["momentum_driver_deltas"][
        "pressure_to_pressure_gradient"
    ]
    assert pressure_driver["field"] == "Ph"
    assert pressure_driver["term"] == "pressure_gradient"
    assert len(pressure_driver["term_delta_lineout"]) == len(report["active_y_indices"])


def test_committed_neutral_mixed_term_balance_report_closes_direct_nvh_sources() -> (
    None
):
    report = json.loads(_COMMITTED_REPORT_JSON.read_text(encoding="utf-8"))

    diagnostics = report["hermes_diagnostic_outputs"]["direct_comparisons"]
    assert (
        diagnostics["SNVh_pressure_gradient"]["scaled_difference_metrics"]["max_abs"]
        < 3.0e-19
    )
    assert (
        diagnostics["SNVh_parallel_viscosity"]["scaled_difference_metrics"]["max_abs"]
        < 2.0e-18
    )
    assert (
        diagnostics["SNVh_perpendicular_viscosity"]["scaled_difference_metrics"][
            "max_abs"
        ]
        < 2.0e-22
    )

    state_register = report["state_driver_register"]
    ranked_state_rates = state_register["ranked_state_rate_errors"]
    assert [entry["field"] for entry in ranked_state_rates] == ["Nh", "Ph", "NVh"]
    assert ranked_state_rates[0]["target_adjacent_max_abs"] == pytest.approx(
        7.605456118353615e-06
    )
    assert ranked_state_rates[1]["target_adjacent_max_abs"] == pytest.approx(
        7.524079128004568e-07
    )
    assert ranked_state_rates[2]["target_adjacent_max_abs"] == pytest.approx(
        2.023199034449706e-07
    )

    ranked_drivers = state_register["ranked_momentum_driver_deltas"]
    assert [entry["driver"] for entry in ranked_drivers[:2]] == [
        "momentum_to_parallel_viscosity",
        "pressure_to_pressure_gradient",
    ]
    assert ranked_drivers[0]["target_adjacent_max_abs"] == pytest.approx(
        1.0011406404022939e-05
    )
    assert ranked_drivers[1]["target_adjacent_max_abs"] == pytest.approx(
        8.096712974357042e-06
    )


def test_neutral_mixed_term_balance_register_ranks_target_adjacent_state_drift(
    tmp_path: Path,
) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    native_arrays = _write_synthetic_native_history_with_target_drift(
        tmp_path / "native_target_drift.npz"
    )

    report = build_neutral_mixed_term_balance_campaign_report(
        input_path=input_path,
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=native_arrays,
    )

    field_register = report["final_field_error_register"]
    ranked_fields = field_register["ranked_by_target_adjacent_max_abs"]
    assert ranked_fields[0]["field"] == "Nh"
    assert ranked_fields[0]["target_adjacent_max_abs"] == pytest.approx(
        2.0e-2, rel=1.0e-12, abs=1.0e-12
    )
    assert field_register["fields"]["Nh"]["interior_max_abs"] == 0.0
    assert field_register["fields"]["Ph"]["target_adjacent_max_abs"] == pytest.approx(
        3.0e-3, rel=1.0e-12, abs=1.0e-12
    )
    assert field_register["fields"]["NVh"]["target_adjacent_max_abs"] == pytest.approx(
        1.0e-3, rel=1.0e-12, abs=1.0e-12
    )

    state_register = report["state_driver_register"]
    ranked_state_rates = state_register["ranked_state_rate_errors"]
    assert ranked_state_rates[0]["field"] == "Nh"
    assert ranked_state_rates[0]["target_adjacent_max_abs"] == pytest.approx(
        1.0e-3, rel=1.0e-12, abs=1.0e-12
    )
    pressure_driver = state_register["momentum_driver_deltas"][
        "pressure_to_pressure_gradient"
    ]
    assert pressure_driver["target_adjacent_max_abs"] > 0.0
    assert pressure_driver["interior_max_abs"] >= 0.0
    assert state_register["target_y_indices"] == field_register["target_y_indices"]


def test_create_neutral_mixed_term_balance_campaign_package_writes_outputs(
    tmp_path: Path,
) -> None:
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
        assert (
            "state_driver_momentum_to_parallel_viscosity_term_delta_lineout" in arrays
        )


def test_neutral_mixed_term_balance_report_can_ingest_hermes_diagnostic_netcdf(
    tmp_path: Path,
) -> None:
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


def test_write_neutral_mixed_diagnostic_input_enables_hermes_outputs(
    tmp_path: Path,
) -> None:
    source = _write_neutral_mixed_input(tmp_path / "source.inp")
    target = write_neutral_mixed_diagnostic_input(
        source, tmp_path / "data" / "BOUT.inp"
    )

    text = target.read_text(encoding="utf-8")
    assert "nout = 1" in text
    assert "[h]" in text
    assert "output_ddt = true" in text
    assert "diagnose = true" in text


def test_write_neutral_mixed_accepted_step_trace_input_enables_monitor(
    tmp_path: Path,
) -> None:
    source = _write_neutral_mixed_input(tmp_path / "source.inp")
    trace_path = tmp_path / "trace" / "accepted.jsonl"
    target = write_neutral_mixed_accepted_step_trace_input(
        source,
        tmp_path / "data" / "BOUT.inp",
        trace_jsonl_path=trace_path,
        species="h",
    )

    text = target.read_text(encoding="utf-8")
    assert "nout = 1" in text
    assert "[solver]" in text
    assert "monitor_timestep = true" in text
    assert "[hermes]" in text
    assert "neutral_mixed_accepted_step_trace = true" in text
    assert f"neutral_mixed_accepted_step_trace_file = {trace_path.resolve()}" in text
    assert "neutral_mixed_accepted_step_trace_species = h" in text
    assert trace_path.parent.exists()


def test_neutral_mixed_accepted_step_reference_patch_documents_required_hook() -> None:
    patch_path = _REPO_ROOT / "docs" / (
        "hermes_neutral_mixed_accepted_step_trace_monitor.patch"
    )
    text = patch_path.read_text(encoding="utf-8")

    assert "load_vars(N_VGetArrayPointer(uvec));" in text
    assert "run_rhs(internal_time);" in text
    assert "int timestepMonitor(BoutReal simtime, BoutReal dt) override;" in text
    assert "Hermes::timestepMonitor(BoutReal simtime, BoutReal dt)" in text
    assert "neutral_mixed_accepted_step_trace" in text
    assert '\\"stages\\":{\\"post_accepted\\"' in text
    assert "json_field_payload(getNonFinal<Field3D>" in text
    for field_name in (
        "N\" + species",
        "P\" + species",
        "NV\" + species",
        "ddt(N\" + species + \")",
        "ddt(P\" + species + \")",
        "ddt(NV\" + species + \")",
        "SNV\" + species",
        "SNV\" + species + \"_pressure_gradient",
        "SNV\" + species + \"_parallel_viscosity",
        "SNV\" + species + \"_perpendicular_viscosity",
    ):
        assert field_name in text


def test_run_neutral_mixed_hermes_accepted_step_trace_returns_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root = tmp_path / "reference"
    source_dir = reference_root / "tests" / "integrated" / "neutral_mixed" / "data"
    source_dir.mkdir(parents=True)
    _write_neutral_mixed_input(source_dir / "BOUT.inp")
    binary = tmp_path / "hermes-3"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"

    def fake_run(command, **kwargs):
        assert command == [str(binary.resolve()), "-d", "data"]
        assert kwargs["cwd"] == (tmp_path / "work").resolve()
        trace_path.write_text(
            json.dumps(
                {
                    "diagnostic": "neutral_mixed_reference_accepted_step_trace",
                    "time": 0.0,
                    "stages": {
                        "post_accepted": {
                            name: {
                                "active_metrics": {"max_abs": 0.0, "rms": 0.0},
                                "target_adjacent_metrics": {
                                    "max_abs": 0.0,
                                    "rms": 0.0,
                                },
                                "guard_metrics": {"max_abs": 0.0, "rms": 0.0},
                                "sample_lineout_y_indices": [0],
                                "sample_lineout": [0.0],
                            }
                            for name in (
                                "Nh",
                                "Ph",
                                "NVh",
                                "ddt(Nh)",
                                "ddt(Ph)",
                                "ddt(NVh)",
                                "SNVh",
                                "SNVh_pressure_gradient",
                                "SNVh_parallel_viscosity",
                                "SNVh_perpendicular_viscosity",
                            )
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(
        "jax_drb.validation.neutral_mixed_term_balance_campaign.subprocess.run",
        fake_run,
    )

    result = run_neutral_mixed_hermes_accepted_step_trace(
        reference_root=reference_root,
        workdir=tmp_path / "work",
        hermes_binary=binary,
        trace_jsonl_path=trace_path,
    )

    assert result == trace_path.resolve()
    assert (tmp_path / "work" / "run.log").read_text(encoding="utf-8") == "ok"


def test_run_neutral_mixed_hermes_accepted_step_trace_requires_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root = tmp_path / "reference"
    source_dir = reference_root / "tests" / "integrated" / "neutral_mixed" / "data"
    source_dir.mkdir(parents=True)
    _write_neutral_mixed_input(source_dir / "BOUT.inp")
    binary = tmp_path / "hermes-3"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"

    def fake_run(command, **kwargs):
        trace_path.write_text(
            json.dumps(
                {
                    "diagnostic": "neutral_mixed_reference_accepted_step_trace",
                    "time": 0.0,
                    "stages": {"post_accepted": {"NVh": {}}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(
        "jax_drb.validation.neutral_mixed_term_balance_campaign.subprocess.run",
        fake_run,
    )

    with pytest.raises(ValueError, match="missing required diagnostics"):
        run_neutral_mixed_hermes_accepted_step_trace(
            reference_root=reference_root,
            workdir=tmp_path / "work",
            hermes_binary=binary,
            trace_jsonl_path=trace_path,
        )


def test_neutral_mixed_substep_hybrid_report_ranks_successes_and_failures(
    tmp_path: Path,
) -> None:
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
    assert (
        report["best"]["value"]
        < report["sweep_points"][0]["final_field_error_register"]["fields"]["NVh"][
            "max_abs"
        ]
    )
    failed = [point for point in report["sweep_points"] if point["status"] == "failed"]
    assert failed
    assert failed[0]["internal_substeps"] == 8
    assert failed[0]["error_type"] == "FileNotFoundError"

    successful = [point for point in report["sweep_points"] if point["status"] == "ok"]
    edge_trace = successful[0]["series_errors"]["fields"]["Nh"][
        "active_edge_history_trace"
    ]
    assert edge_trace["target_active_y_offsets"] == [0, 1, 8, 9]
    assert edge_trace["target_adjacent_max_abs_by_time"][0] == 0.0
    assert edge_trace["target_adjacent_max_abs_by_time"][-1] == pytest.approx(2.0e-2)
    hybrid = successful[0]["hybrid_state_register"]
    assert set(hybrid["swaps"]) == {"Nh", "Ph", "NVh"}
    assert (
        hybrid["ranked_by_pressure_gradient_target_adjacent_delta"][0]["swapped_field"]
        == "Ph"
    )
    assert (
        hybrid["ranked_by_parallel_viscosity_target_adjacent_delta"][0]["swapped_field"]
        == "NVh"
    )
    assert (
        successful[0]["series_errors"]["ranked_by_final_target_adjacent_max_abs"][0][
            "field"
        ]
        == "Nh"
    )

    path = write_neutral_mixed_substep_hybrid_json(report, tmp_path / "substeps.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["field"] == "NVh"


def test_neutral_mixed_native_accepted_step_trace_report_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")

    def fake_implicit_history(_config, **kwargs):
        mesh = kwargs["mesh"]
        shape = (3, mesh.nx, mesh.local_ny, mesh.nz)
        base = np.ones(shape, dtype=np.float64)
        assert kwargs["store_internal_substeps"] is True
        assert kwargs["internal_substeps"] == 2
        return SimpleNamespace(
            density_history=base[[0, -1]],
            pressure_history=2.0 * base[[0, -1]],
            momentum_history=3.0 * base[[0, -1]],
            accepted_step_time_points=np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            accepted_step_dt=np.asarray([0.0, 10.0, 10.0], dtype=np.float64),
            accepted_step_order=np.asarray([0, 1, 2], dtype=np.int32),
            accepted_step_density_history=base,
            accepted_step_pressure_history=2.0 * base,
            accepted_step_momentum_history=3.0 * base,
            accepted_step_residual_inf_norm=np.asarray(
                [0.0, 1.0e-10, 2.0e-10], dtype=np.float64
            ),
            accepted_step_nonlinear_iterations=np.asarray([0, 2, 3], dtype=np.int32),
        )

    monkeypatch.setattr(
        "jax_drb.validation.neutral_mixed_term_balance_campaign.advance_neutral_mixed_implicit_history",
        fake_implicit_history,
    )

    report = build_neutral_mixed_native_accepted_step_trace_report(
        input_path=input_path,
        internal_substeps=2,
        steps=1,
    )

    assert report["diagnostic"] == "neutral_mixed_native_accepted_step_trace"
    assert report["requires_hermes"] is False
    assert report["trace_point_count"] == 3
    assert report["target_y_indices"] == [2, 3, 10, 11]
    assert report["guard_y_indices"] == [0, 1, 12, 13]
    assert report["sample_y_indices"] == [0, 1, 2, 3, 10, 11, 12, 13]
    assert report["trace_points"][1]["solver_order"] == 1
    assert report["trace_points"][2]["solver_order"] == 2
    assert report["trace_points"][2]["fields"]["NVh"]["target_adjacent_metrics"][
        "max_abs"
    ] == pytest.approx(3.0)
    assert report["trace_points"][2]["fields"]["Nh"]["sample_lineout"] == pytest.approx(
        [1.0] * 8
    )

    path = write_neutral_mixed_native_accepted_step_trace_json(
        report, tmp_path / "native_trace.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["diagnostic"] == "neutral_mixed_native_accepted_step_trace"


def test_neutral_mixed_accepted_step_trace_parity_ingests_reference_jsonl(
    tmp_path: Path,
) -> None:
    native_trace = {
        "diagnostic": "neutral_mixed_native_accepted_step_trace",
        "trace_points": [
            {
                "index": 0,
                "time": 0.0,
                "dt": 0.0,
                "solver_order": 0,
                "stage": "post_accepted",
                "fields": {
                    "NVh": {
                        "active_metrics": {"max_abs": 2.0, "rms": 1.0},
                        "target_adjacent_metrics": {"max_abs": 4.0, "rms": 2.0},
                        "guard_metrics": {"max_abs": 1.0, "rms": 0.5},
                        "sample_lineout_y_indices": [0, 1],
                        "sample_lineout": [1.0, 2.0],
                    }
                },
            },
            {
                "index": 1,
                "time": 1.0e-5,
                "dt": 1.0e-5,
                "solver_order": 1,
                "stage": "post_accepted",
                "fields": {
                    "NVh": {
                        "active_metrics": {"max_abs": 3.0, "rms": 1.5},
                        "target_adjacent_metrics": {"max_abs": 6.0, "rms": 3.0},
                        "guard_metrics": {"max_abs": 1.5, "rms": 0.75},
                        "sample_lineout_y_indices": [0, 1],
                        "sample_lineout": [2.0, 4.0],
                    }
                },
            },
        ],
    }
    reference_records = [
        {
            "diagnostic": "neutral_mixed_reference_accepted_step_trace",
            "step_index": 0,
            "time": 0.0,
            "dt": 0.0,
            "solver": {"order": 0},
            "stages": {
                "post_accepted": {
                    "NVh": {
                        "active_metrics": {"max_abs": 2.5, "rms": 1.0},
                        "target_adjacent_metrics": {"max_abs": 5.0, "rms": 2.0},
                        "guard_metrics": {"max_abs": 1.25, "rms": 0.5},
                        "sample_lineout_y_indices": [0, 1],
                        "sample_lineout": [1.5, 1.0],
                    }
                }
            },
        },
        {
            "diagnostic": "neutral_mixed_reference_accepted_step_trace",
            "step_index": 1,
            "time": 1.0e-5,
            "dt": 1.0e-5,
            "solver": {"order": 1},
            "stages": {
                "post_accepted": {
                    "NVh": {
                        "active_metrics": {"max_abs": 1.0, "rms": 1.0},
                        "target_adjacent_metrics": {"max_abs": 2.0, "rms": 2.0},
                        "guard_metrics": {"max_abs": 0.5, "rms": 0.5},
                        "sample_lineout_y_indices": [0, 1],
                        "sample_lineout": [1.0, 1.0],
                    }
                }
            },
        },
    ]
    native_path = tmp_path / "native_trace.json"
    reference_path = tmp_path / "reference_trace.jsonl"
    native_path.write_text(json.dumps(native_trace), encoding="utf-8")
    reference_path.write_text(
        "\n".join(json.dumps(record) for record in reference_records) + "\n",
        encoding="utf-8",
    )

    report = build_neutral_mixed_accepted_step_trace_parity_report(
        native_trace_json=native_path,
        reference_trace_json=reference_path,
        time_tolerance=1.0e-12,
    )

    assert report["diagnostic"] == "neutral_mixed_accepted_step_trace_parity"
    assert report["matched_trace_point_count"] == 2
    assert report["fields"]["NVh"]["max_active_delta"] == pytest.approx(2.0)
    assert report["fields"]["NVh"]["max_target_adjacent_delta"] == pytest.approx(4.0)
    assert report["fields"]["NVh"]["max_guard_delta"] == pytest.approx(1.0)
    assert report["fields"]["NVh"]["max_sample_lineout_delta"] == pytest.approx(3.0)
    assert report["ranked_fields"][0]["field"] == "NVh"

    path = write_neutral_mixed_accepted_step_trace_parity_json(
        report, tmp_path / "trace_parity.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["diagnostic"] == "neutral_mixed_accepted_step_trace_parity"


def test_committed_neutral_mixed_substep_hybrid_artifact_tracks_substep_trend() -> None:
    path = (
        _REPO_ROOT
        / "docs"
        / "data"
        / "neutral_mixed_substep_hybrid_artifacts"
        / "data"
        / "neutral_mixed_substep_hybrid.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["diagnostic"] == "neutral_mixed_substep_hybrid_state"
    assert report["requires_hermes"] is False
    assert report["best"] == {
        "metric": "NVh_final_max_abs",
        "internal_substeps": 8,
        "value": pytest.approx(4.46733131195939e-6),
    }

    points = {
        int(point["internal_substeps"]): point for point in report["sweep_points"]
    }
    assert sorted(points) == [1, 2, 3, 4, 6, 8]
    assert points[3]["status"] == "failed"
    assert points[6]["status"] == "failed"
    assert points[3]["failure_vector"]["size"] == 1800
    assert points[3]["failure_vector"]["finite_fraction"] == pytest.approx(1.0)
    assert points[3]["failure_vector"]["max_abs"] == pytest.approx(8.526386727731725e-1)
    assert points[6]["failure_vector"]["size"] == 1800
    assert points[6]["failure_vector"]["finite_fraction"] == pytest.approx(1.0)
    assert points[6]["failure_vector"]["max_abs"] == pytest.approx(8.753496191572953e-1)

    successful_errors = [
        points[substeps]["final_field_error_register"]["fields"]["NVh"]["max_abs"]
        for substeps in (1, 2, 4, 8)
    ]
    assert successful_errors == sorted(successful_errors, reverse=True)
    assert successful_errors[0] == pytest.approx(8.840923081330086e-4)
    assert successful_errors[-1] == pytest.approx(4.46733131195939e-6)

    best_hybrid = points[8]["hybrid_state_register"]
    assert (
        best_hybrid["ranked_by_pressure_gradient_target_adjacent_delta"][0][
            "swapped_field"
        ]
        == "Ph"
    )
    assert (
        best_hybrid["ranked_by_parallel_viscosity_target_adjacent_delta"][0][
            "swapped_field"
        ]
        == "NVh"
    )

    best_trace = points[8]["series_errors"]["fields"]["Nh"]["active_edge_history_trace"]
    assert best_trace["target_active_y_offsets"] == [0, 1, 8, 9]
    assert best_trace["target_adjacent_max_abs_by_time"][0] == pytest.approx(
        0.0, abs=2.0e-16
    )
    assert best_trace["target_adjacent_max_abs_by_time"][-1] == pytest.approx(
        points[8]["series_errors"]["fields"]["Nh"]["final_target_adjacent_max_abs"]
    )

    assert "/Users/" not in json.dumps(report, sort_keys=True)
