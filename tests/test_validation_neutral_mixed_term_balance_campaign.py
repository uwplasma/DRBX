from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.validation.neutral_mixed_term_balance_campaign as term_module
from jax_drb.validation import (
    build_neutral_mixed_accepted_step_trace_parity_report,
    build_neutral_mixed_native_accepted_step_trace_report,
    build_neutral_mixed_reference_input_closure_report,
    build_neutral_mixed_substep_hybrid_report,
    build_neutral_mixed_term_balance_campaign_report,
    create_neutral_mixed_term_balance_campaign_package,
    run_neutral_mixed_hermes_accepted_step_trace,
    save_neutral_mixed_term_balance_campaign_plot,
    write_neutral_mixed_accepted_step_trace_input,
    write_neutral_mixed_accepted_step_trace_parity_json,
    write_neutral_mixed_native_accepted_step_trace_json,
    write_neutral_mixed_reference_input_closure_json,
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


def _write_neutral_mixed_input_closure_dump(
    input_path: Path,
    dump_path: Path,
    *,
    eta_perturbation: float = 0.0,
) -> Path:
    from netCDF4 import Dataset

    config = term_module.load_bout_input(input_path)
    run_config = term_module.RunConfiguration.from_config(config)
    mesh = term_module.build_structured_mesh(config, run_config)
    metrics = term_module.build_structured_metrics(config, run_config, mesh)
    scalars = term_module.resolved_dataset_scalars(run_config)
    state = term_module.initialize_neutral_mixed_state(config, section="h", mesh=mesh)
    prepared = term_module._prepare_neutral_mixed_state(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
    )
    eta = np.asarray(prepared.viscosity, dtype=np.float64).copy()
    if eta_perturbation:
        eta[mesh.xstart, mesh.ystart, 0] += float(eta_perturbation)
    fields = {
        "Nh": state.density,
        "Ph": state.pressure,
        "NVh": state.momentum,
        "Dnnh": prepared.diffusion,
        "Vh": prepared.velocity,
        "eta_h": eta,
    }
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(dump_path, "w") as dataset:
        dataset.createDimension("t", 1)
        dataset.createDimension("x", mesh.nx)
        dataset.createDimension("y", mesh.local_ny)
        dataset.createDimension("z", mesh.nz)
        for name, values in fields.items():
            variable = dataset.createVariable(name, "f8", ("t", "x", "y", "z"))
            variable[0, :, :, :] = np.asarray(values, dtype=np.float64)
    return dump_path


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


def test_neutral_mixed_reference_input_closure_report_matches_reference_dump(
    tmp_path: Path,
) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    diagnostic_path = _write_neutral_mixed_input_closure_dump(
        input_path,
        tmp_path / "BOUT.dmp.0.nc",
    )

    report = build_neutral_mixed_reference_input_closure_report(
        input_path=input_path,
        hermes_diagnostic_nc=diagnostic_path,
    )
    path = write_neutral_mixed_reference_input_closure_json(
        report,
        tmp_path / "input_closure.json",
    )

    assert report["diagnostic"] == "neutral_mixed_reference_input_closure"
    assert path.exists()
    assert [entry["field"] for entry in report["ranked_fields"]] == [
        "Dnnh",
        "Vh",
        "eta_h",
    ]
    for field in ("Dnnh", "Vh", "eta_h"):
        payload = report["fields"][field]
        assert payload["max_active_delta"] <= 1.0e-14
        assert payload["max_target_adjacent_delta"] <= 1.0e-14
        assert payload["max_guard_delta"] <= 1.0e-14
        assert payload["sample_lineout_delta"] == pytest.approx(
            [0.0] * len(report["sample_y_indices"]),
            abs=1.0e-14,
        )


def test_neutral_mixed_reference_input_closure_report_ranks_eta_perturbation(
    tmp_path: Path,
) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    diagnostic_path = _write_neutral_mixed_input_closure_dump(
        input_path,
        tmp_path / "BOUT.dmp.0.nc",
        eta_perturbation=1.0e-3,
    )

    report = build_neutral_mixed_reference_input_closure_report(
        input_path=input_path,
        hermes_diagnostic_nc=diagnostic_path,
    )

    assert report["ranked_fields"][0]["field"] == "eta_h"
    eta = report["fields"]["eta_h"]
    assert eta["max_target_adjacent_delta"] == pytest.approx(1.0e-3)
    assert eta["max_active_delta"] == pytest.approx(1.0e-3)
    assert report["fields"]["Dnnh"]["max_target_adjacent_delta"] == 0.0
    assert report["fields"]["Vh"]["max_target_adjacent_delta"] == 0.0


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
    assert 'options["neutral_mixed_accepted_step_trace"].setConditionallyUsed();' in text
    assert (
        'options["neutral_mixed_accepted_step_trace_file"].setConditionallyUsed();'
        in text
    )
    assert (
        'options["neutral_mixed_accepted_step_trace_species"].setConditionallyUsed();'
        in text
    )
    assert 'auto& species_state = state["species"][species];' in text
    assert "std::vector<int> sample_y_indices()" in text
    assert '\\"stages\\":{\\"post_accepted\\"' in text
    assert 'json_field_payload(get<Field3D>(species_state["density"]))' in text
    assert 'json_field_payload(get<Field3D>(species_state["pressure"]))' in text
    assert 'json_field_payload(get<Field3D>(species_state["momentum"]))' in text
    assert "json_field_payload(getNonFinal<Field3D>" in text
    assert "std::string json_optional_trace_field(" in text
    assert '"Tnlim" + species,' in text
    assert '"logPnlim" + species,' in text
    assert '"grad_logPnlim" + species,' in text
    assert '"Dnn" + species + "_raw",' in text
    assert '"Dnn" + species + "_flux_max",' in text
    assert '"Dnn" + species + "_flux_limited",' in text
    assert '"Dnn" + species + "_diffusion_limited",' in text
    assert '"Dnn" + species,' in text
    assert '"V" + species,' in text
    assert '"eta_" + species,' in text
    assert "Field3D Tnlim, grad_logPnlim;" in text
    assert "Field3D Dnn_raw, Dnn_flux_max;" in text
    assert "Dnn_raw = copy(Dnn);" in text
    assert "Dnn_flux_max =" in text
    assert "Dnn_flux_limited = copy(Dnn);" in text
    assert "Dnn_diffusion_limited = copy(Dnn);" in text
    assert "max_abs_index" in text
    assert "max_abs_value" in text
    assert "target_adjacent_shape" in text
    assert "target_adjacent_values" in text
    assert "guard_shape" in text
    assert "guard_values" in text
    assert 'state[std::string("eta_") + name]' in text
    assert 'state[std::string("grad_logPnlim") + name]' in text
    assert '      "N" + species,' not in text
    assert '      "P" + species,' not in text
    assert '      "NV" + species,' not in text
    for field_name in (
        "ddt(N\" + species + \")",
        "ddt(P\" + species + \")",
        "ddt(NV\" + species + \")",
        "SNV\" + species",
        "SNV\" + species + \"_pressure_gradient",
        "SNV\" + species + \"_parallel_viscosity",
        "SNV\" + species + \"_perpendicular_viscosity",
    ):
        assert field_name in text


def test_neutral_mixed_source_diagnostic_reference_patch_is_line_numbered() -> None:
    patch_path = (
        _REPO_ROOT / "docs" / "hermes_neutral_mixed_pressure_gradient_diagnostic.patch"
    )
    text = patch_path.read_text(encoding="utf-8")

    assert "diff --git a/include/neutral_mixed.hxx" in text
    assert "diff --git a/src/neutral_mixed.cxx" in text
    assert "@@ -76,6 +76,9 @@" in text
    assert "@@ -564,11 +564,12 @@" in text
    assert "\n@@\n" not in text
    for token in (
        "momentum_pressure_gradient_source = -Grad_par(Pn);",
        "momentum_parallel_viscosity_source = Div_par_K_Grad_par_mod",
        "momentum_perpendicular_viscosity_source =",
        "SNV\") + name + std::string(\"_pressure_gradient\")",
        "SNV\") + name + std::string(\"_parallel_viscosity\")",
        "SNV\") + name + std::string(\"_perpendicular_viscosity\")",
    ):
        assert token in text


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


def test_run_neutral_mixed_accepted_step_trace_auto_uses_patched_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root = tmp_path / "reference"
    source_dir = reference_root / "tests" / "integrated" / "neutral_mixed" / "data"
    source_dir.mkdir(parents=True)
    _write_neutral_mixed_input(source_dir / "BOUT.inp")
    binary = tmp_path / "patched" / "hermes-3"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    built: dict[str, Path] = {}

    def fake_build(root: Path):
        built["root"] = root
        return binary, tmp_path / "patched"

    def fake_run(command, **kwargs):
        assert command == [str(binary), "-d", "data"]
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
        term_module,
        "_build_patched_neutral_mixed_accepted_step_reference_binary",
        fake_build,
    )
    monkeypatch.setattr(term_module.subprocess, "run", fake_run)

    result = run_neutral_mixed_hermes_accepted_step_trace(
        reference_root=reference_root,
        workdir=tmp_path / "work",
        trace_jsonl_path=trace_path,
    )

    assert built["root"] == reference_root.resolve()
    assert result == trace_path.resolve()


def test_build_patched_neutral_mixed_reference_binary_uses_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_digest = term_module.hashlib.sha256(
        b"".join(
            (_REPO_ROOT / "docs" / name).read_bytes()
            for name in (
                "hermes_neutral_mixed_pressure_gradient_diagnostic.patch",
                "hermes_neutral_mixed_accepted_step_trace_monitor.patch",
            )
        )
    ).hexdigest()[:12]
    cache_root = (
        tmp_path
        / "jax_drb_neutral_mixed_accepted_step_reference"
        / f"deadbeef-{patch_digest}"
    )
    binary_path = cache_root / "build" / "hermes-3"
    binary_path.parent.mkdir(parents=True)
    binary_path.write_text("binary", encoding="utf-8")
    reference_root = tmp_path / "reference"
    reference_root.mkdir()

    monkeypatch.setattr(term_module, "_git_stdout", lambda root, *args: "deadbeef")
    monkeypatch.setattr(term_module.tempfile, "gettempdir", lambda: str(tmp_path))

    returned_binary, returned_cache = (
        term_module._build_patched_neutral_mixed_accepted_step_reference_binary(
            reference_root
        )
    )

    assert returned_binary == binary_path
    assert returned_cache == cache_root


def test_build_patched_neutral_mixed_reference_binary_builds_clean_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(term_module, "_git_stdout", lambda root, *args: "deadbeef")
    monkeypatch.setattr(term_module.tempfile, "gettempdir", lambda: str(tmp_path))

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:6] == ["git", "-C", str(reference_root), "worktree", "add", "--detach"]:
            Path(args[-2]).mkdir(parents=True, exist_ok=True)
        elif args[:2] == ["cmake", "--build"]:
            build_root = Path(args[2])
            build_root.mkdir(parents=True, exist_ok=True)
            (build_root / "hermes-3").write_text("binary", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(term_module.subprocess, "run", fake_run)

    binary, cache_root = (
        term_module._build_patched_neutral_mixed_accepted_step_reference_binary(
            reference_root
        )
    )

    assert binary == cache_root / "build" / "hermes-3"
    assert any(
        call[:6] == ["git", "-C", str(reference_root), "worktree", "add", "--detach"]
        for call in calls
    )
    assert any(
        call[:4] == ["git", "-C", str(cache_root / "src"), "submodule"]
        for call in calls
    )
    apply_calls = [
        call
        for call in calls
        if call[:4] == ["git", "-C", str(cache_root / "src"), "apply"]
    ]
    assert any(
        "--check" in call
        and "hermes_neutral_mixed_pressure_gradient_diagnostic.patch" in call[-1]
        for call in apply_calls
    )
    assert any(
        "--check" not in call
        and "hermes_neutral_mixed_pressure_gradient_diagnostic.patch" in call[-1]
        for call in apply_calls
    )
    split_root_apply_calls = [
        call
        for call in calls
        if call[:3] == ["git", "-C", str(cache_root / "src")]
        and call[3] == "apply"
        and "hermes_neutral_mixed_accepted_step_trace_monitor.patch" not in call[-1]
    ]
    split_solver_apply_calls = [
        call
        for call in calls
        if call[:3]
        == ["git", "-C", str(cache_root / "src" / "external" / "BOUT-dev")]
        and call[3] == "apply"
    ]
    assert any("--check" not in call for call in split_root_apply_calls)
    assert any("--check" not in call for call in split_solver_apply_calls)
    assert any(
        "--check" in call and "--reverse" not in call
        for call in split_solver_apply_calls
    )
    assert any(call[:2] == ["cmake", "-S"] for call in calls)
    assert any(call[:2] == ["cmake", "--build"] for call in calls)


def test_apply_reference_patch_skips_already_applied_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text("patch", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if "--reverse" in args and "--check" in args:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--check" in args:
            return SimpleNamespace(returncode=1, stdout="", stderr="already applied")
        raise AssertionError("already-applied patch should not be applied again")

    monkeypatch.setattr(term_module.subprocess, "run", fake_run)

    term_module._apply_reference_patch_if_needed(source_root, patch_path)

    assert calls == [
        ["git", "-C", str(source_root), "apply", "--check", str(patch_path)],
        [
            "git",
            "-C",
            str(source_root),
            "apply",
            "--reverse",
            "--check",
            str(patch_path),
        ],
    ]


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
        assert kwargs["accepted_step_time_points"] is None
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
    assert report["time_grid_source"] == "uniform_internal_substeps"
    assert report["reference_trace_json"] is None
    assert report["reference_trace_point_count"] == 0
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
    assert set(report["trace_points"][0]["fields"]) == {
        "Nh",
        "Ph",
        "NVh",
        "Tnlimh",
        "logPnlimh",
        "grad_logPnlimh",
        "Dnnh_raw",
        "Dnnh_flux_max",
        "Dnnh_flux_limited",
        "Dnnh_diffusion_limited",
        "Dnnh",
        "Vh",
        "eta_h",
        "ddt(Nh)",
        "ddt(Ph)",
        "ddt(NVh)",
        "SNVh",
        "SNVh_pressure_gradient",
        "SNVh_parallel_viscosity",
        "SNVh_perpendicular_viscosity",
    }
    fields = report["trace_points"][0]["fields"]
    assert fields["Dnnh_raw"]["active_metrics"]["max_abs"] > 0.0
    assert len(fields["Dnnh_raw"]["active_metrics"]["max_abs_index"]) == 3
    assert fields["Dnnh_raw"]["target_adjacent_shape"][1] == len(
        report["target_y_indices"]
    )
    assert len(fields["Dnnh_raw"]["target_adjacent_values"]) == int(
        np.prod(fields["Dnnh_raw"]["target_adjacent_shape"])
    )
    assert fields["Dnnh_raw"]["guard_shape"][1] == len(report["guard_y_indices"])
    assert len(fields["Dnnh_raw"]["guard_values"]) == int(
        np.prod(fields["Dnnh_raw"]["guard_shape"])
    )
    assert fields["Dnnh_diffusion_limited"]["active_metrics"]["max_abs"] > 0.0

    path = write_neutral_mixed_native_accepted_step_trace_json(
        report, tmp_path / "native_trace.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["diagnostic"] == "neutral_mixed_native_accepted_step_trace"


def test_neutral_mixed_native_accepted_step_trace_replays_reference_time_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    reference_trace = tmp_path / "reference_trace.jsonl"
    reference_trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "time": time_value,
                    "dt": dt_value,
                    "solver": {"order": order},
                    "stages": {
                        "post_accepted": {
                            "Nh": {
                                "active_metrics": {"max_abs": 1.0, "rms": 1.0},
                                "target_adjacent_metrics": {
                                    "max_abs": 1.0,
                                    "rms": 1.0,
                                },
                                "guard_metrics": {"max_abs": 1.0, "rms": 1.0},
                                "sample_lineout_y_indices": [],
                                "sample_lineout": [],
                            }
                        }
                    },
                }
            )
            for time_value, dt_value, order in (
                (5.0, 5.0, 1),
                (20.0, 15.0, 2),
            )
        ),
        encoding="utf-8",
    )

    def fake_implicit_history(_config, **kwargs):
        mesh = kwargs["mesh"]
        np.testing.assert_allclose(
            kwargs["accepted_step_time_points"],
            np.asarray([5.0, 20.0], dtype=np.float64),
        )
        shape = (3, mesh.nx, mesh.local_ny, mesh.nz)
        base = np.ones(shape, dtype=np.float64)
        return SimpleNamespace(
            density_history=base[[0, -1]],
            pressure_history=2.0 * base[[0, -1]],
            momentum_history=3.0 * base[[0, -1]],
            accepted_step_time_points=np.asarray([0.0, 5.0, 20.0], dtype=np.float64),
            accepted_step_dt=np.asarray([0.0, 5.0, 15.0], dtype=np.float64),
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
        reference_trace_json=reference_trace,
        time_tolerance=1.0e-9,
    )

    assert report["time_grid_source"] == "reference_accepted_steps"
    assert report["reference_trace_point_count"] == 2
    assert str(report["reference_trace_json"]).endswith("reference_trace.jsonl")
    assert report["time_points"] == pytest.approx([0.0, 5.0, 20.0])


def test_neutral_mixed_native_accepted_step_trace_rejects_reference_final_time_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _write_neutral_mixed_input(tmp_path / "BOUT.inp")
    reference_trace = tmp_path / "reference_trace.jsonl"
    reference_trace.write_text(
        json.dumps(
            {
                "time": 19.0,
                "dt": 19.0,
                "stages": {
                    "post_accepted": {
                        "Nh": {
                            "active_metrics": {"max_abs": 1.0, "rms": 1.0},
                            "target_adjacent_metrics": {"max_abs": 1.0, "rms": 1.0},
                            "guard_metrics": {"max_abs": 1.0, "rms": 1.0},
                            "sample_lineout_y_indices": [],
                            "sample_lineout": [],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("native solver should not run for invalid reference grid")

    monkeypatch.setattr(
        "jax_drb.validation.neutral_mixed_term_balance_campaign.advance_neutral_mixed_implicit_history",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="final time does not reach"):
        build_neutral_mixed_native_accepted_step_trace_report(
            input_path=input_path,
            steps=1,
            reference_trace_json=reference_trace,
            time_tolerance=1.0e-9,
        )


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
    assert report["parallel_viscosity_input_register"]["entries"] == []

    path = write_neutral_mixed_accepted_step_trace_parity_json(
        report, tmp_path / "trace_parity.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["diagnostic"] == "neutral_mixed_accepted_step_trace_parity"


def test_neutral_mixed_accepted_step_trace_parity_ranks_rhs_by_active_target(
    tmp_path: Path,
) -> None:
    def field(
        active: float, target: float, guard: float, sample: float | None = None
    ) -> dict[str, object]:
        sample_value = target if sample is None else sample
        return {
            "active_metrics": {"max_abs": active, "rms": active},
            "target_adjacent_metrics": {"max_abs": target, "rms": target},
            "guard_metrics": {"max_abs": guard, "rms": guard},
            "sample_lineout_y_indices": [0],
            "sample_lineout": [sample_value],
        }

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
                    "NVh": field(0.0, 0.0, 0.0),
                    "ddt(NVh)": field(0.0, 0.0, 1000.0, sample=1000.0),
                },
            },
            {
                "index": 1,
                "time": 1.0,
                "dt": 1.0,
                "solver_order": 1,
                "stage": "post_accepted",
                "fields": {
                    "NVh": field(0.0, 0.0, 0.0),
                    "ddt(NVh)": field(0.0, 0.0, 0.0),
                },
            }
        ],
    }
    reference_records = [
        {
            "diagnostic": "neutral_mixed_reference_accepted_step_trace",
            "step_index": 0,
            "time": 0.0,
            "dt": 0.0,
            "stages": {
                "post_accepted": {
                    "NVh": field(0.5, 0.5, 0.5),
                    "ddt(NVh)": field(0.1, 0.1, 0.0, sample=0.0),
                }
            },
        },
        {
            "diagnostic": "neutral_mixed_reference_accepted_step_trace",
            "step_index": 1,
            "time": 1.0,
            "dt": 1.0,
            "stages": {
                "post_accepted": {
                    "NVh": field(0.5, 0.5, 0.5),
                    "ddt(NVh)": field(0.2, 0.2, 0.0),
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

    assert report["ranked_fields"][0]["field"] == "NVh"
    assert report["fields"]["ddt(NVh)"]["comparison_scope"] == (
        "active_target_rhs_source"
    )
    assert report["fields"]["ddt(NVh)"]["max_guard_delta"] == pytest.approx(1000.0)
    assert report["fields"]["ddt(NVh)"]["max_sample_lineout_delta"] == pytest.approx(
        1000.0
    )
    assert report["fields"]["ddt(NVh)"]["max_target_adjacent_delta"] == pytest.approx(
        0.2
    )
    assert report["fields"]["ddt(NVh)"]["worst_time"] == pytest.approx(1.0)


def test_neutral_mixed_accepted_step_trace_parity_reports_viscosity_inputs(
    tmp_path: Path,
) -> None:
    def field(active: float, target: float, guard: float = 0.0) -> dict[str, object]:
        return {
            "active_metrics": {"max_abs": active, "rms": active},
            "target_adjacent_metrics": {"max_abs": target, "rms": target},
            "guard_metrics": {"max_abs": guard, "rms": guard},
            "target_adjacent_shape": [1, 1, 1],
            "target_adjacent_values": [target],
            "guard_shape": [1, 1, 1],
            "guard_values": [guard],
            "sample_lineout_y_indices": [0],
            "sample_lineout": [target],
        }

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
                    "SNVh_parallel_viscosity": field(0.0, 0.0),
                    "Nh": field(0.0, 0.0),
                    "Ph": field(0.0, 0.0),
                    "NVh": field(0.0, 0.0),
                    "Tnlimh": field(0.0, 0.0),
                    "logPnlimh": field(0.0, 0.0),
                    "grad_logPnlimh": field(0.0, 0.0),
                    "Dnnh_raw": field(0.0, 0.0),
                    "Dnnh_flux_max": field(0.0, 0.0),
                    "Dnnh_flux_limited": field(0.0, 0.0),
                    "Dnnh_diffusion_limited": field(0.0, 0.0),
                    "Dnnh": field(0.0, 0.0),
                    "Vh": field(0.0, 0.0),
                    "eta_h": field(0.0, 0.0),
                },
            }
        ],
    }
    reference_records = [
        {
            "diagnostic": "neutral_mixed_reference_accepted_step_trace",
            "step_index": 0,
            "time": 0.0,
            "dt": 0.0,
            "stages": {
                "post_accepted": {
                    "SNVh_parallel_viscosity": field(2.0, 5.0),
                    "Nh": field(0.1, 0.4),
                    "Ph": field(0.2, 0.1),
                    "NVh": field(0.3, 0.3),
                    "Tnlimh": field(0.01, 0.01),
                    "logPnlimh": field(0.02, 0.02),
                    "grad_logPnlimh": field(0.03, 0.03),
                    "Dnnh_raw": field(4.0, 4.0, 7.0),
                    "Dnnh_flux_max": field(0.6, 0.7),
                    "Dnnh_flux_limited": field(0.15, 0.2),
                    "Dnnh_diffusion_limited": field(0.1, 0.1),
                    "Dnnh": field(0.05, 0.15),
                    "Vh": field(0.25, 0.5),
                    "eta_h": field(0.1, 0.2),
                }
            },
        }
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

    register = report["parallel_viscosity_input_register"]
    assert register["missing_reference_input_fields"] == []
    entry = register["entries"][0]
    assert entry["source_field"] == "SNVh_parallel_viscosity"
    assert entry["input_fields_present"] is True
    assert entry["diagnosis"] == "input_drift_check_available"
    assert entry["diffusion_field"] == "Dnnh"
    assert entry["diffusion_error"]["max_target_adjacent_delta"] == pytest.approx(
        0.15
    )
    assert entry["missing_closure_input_fields"] == []
    assert entry["closure_input_fields_present"] is True
    assert entry["velocity_field"] == "Vh"
    assert entry["viscosity_field"] == "eta_h"
    assert set(entry["closure_input_errors"]) == {"Dnnh", "Vh", "eta_h"}
    assert entry["dominant_closure_input_field"] == "Vh"
    assert entry["max_closure_input_target_adjacent_delta"] == pytest.approx(0.5)
    assert entry["max_closure_input_target_adjacent_pointwise_delta"] == pytest.approx(
        0.5
    )
    assert entry["max_closure_input_active_delta"] == pytest.approx(0.25)
    assert entry["source_max_target_adjacent_delta"] == pytest.approx(5.0)
    assert entry["source_max_target_adjacent_pointwise_delta"] == pytest.approx(5.0)
    assert entry["max_input_target_adjacent_delta"] == pytest.approx(0.5)
    assert entry["max_input_target_adjacent_pointwise_delta"] == pytest.approx(0.5)
    assert entry["max_input_active_delta"] == pytest.approx(0.25)
    assert entry["state_input_fields"] == ["Nh", "Ph", "NVh"]
    assert entry["missing_state_input_fields"] == []
    assert entry["state_input_fields_present"] is True
    assert entry["dominant_state_input_field"] == "Nh"
    assert entry["max_state_input_target_adjacent_delta"] == pytest.approx(0.4)
    assert entry["max_state_input_target_adjacent_pointwise_delta"] == pytest.approx(
        0.4
    )
    assert entry["max_state_input_active_delta"] == pytest.approx(0.3)
    assert entry["viscosity_to_state_target_ratio"] == pytest.approx(0.5)
    assert entry["viscosity_to_state_target_pointwise_ratio"] == pytest.approx(0.5)
    assert entry["viscosity_to_state_active_ratio"] == pytest.approx(1.0 / 3.0)
    assert entry["diffusion_to_state_target_ratio"] == pytest.approx(0.375)
    assert entry["diffusion_to_state_target_pointwise_ratio"] == pytest.approx(0.375)
    assert entry["diffusion_to_state_active_ratio"] == pytest.approx(1.0 / 6.0)
    assert register["missing_reference_state_input_fields"] == []
    assert register["missing_reference_closure_input_fields"] == []
    ladder_register = report["neutral_diffusion_ladder_register"]
    assert ladder_register["missing_reference_ladder_fields"] == []
    ladder_entry = ladder_register["entries"][0]
    assert ladder_entry["section"] == "h"
    assert ladder_entry["diffusion_field"] == "Dnnh"
    assert ladder_entry["ladder_fields_present"] is True
    assert ladder_entry["diagnosis"] == "diffusion_ladder_check_available"
    assert ladder_entry["dominant_ladder_field"] == "Dnnh_raw"
    assert ladder_entry["max_ladder_target_adjacent_delta"] == pytest.approx(4.0)
    assert ladder_entry["ranked_ladder_errors"][0]["field"] == "Dnnh_raw"
    assert report["fields"]["Dnnh_raw"][
        "max_target_adjacent_pointwise_delta"
    ] == pytest.approx(4.0)
    assert (
        report["fields"]["Dnnh_raw"]["comparison_scope"]
        == "active_target_preboundary_diagnostic"
    )
    assert report["fields"]["Dnnh_raw"][
        "max_target_adjacent_pointwise_delta_worst_index"
    ]["local_index"] == [0, 0, 0]
    assert report["fields"]["Dnnh_raw"]["max_guard_pointwise_delta"] == pytest.approx(
        7.0
    )
    assert report["fields"]["Dnnh_raw"]["max_guard_pointwise_delta_worst_index"][
        "local_index"
    ] == [0, 0, 0]
    assert report["fields"]["Dnnh_raw"]["max_target_adjacent_delta_worst_index"][
        "native_index"
    ] == []


def test_neutral_mixed_accepted_step_trace_parity_reports_missing_viscosity_inputs(
    tmp_path: Path,
) -> None:
    def field(value: float) -> dict[str, object]:
        return {
            "active_metrics": {"max_abs": value, "rms": value},
            "target_adjacent_metrics": {"max_abs": value, "rms": value},
            "guard_metrics": {"max_abs": value, "rms": value},
            "target_adjacent_shape": [],
            "target_adjacent_values": [],
            "guard_shape": [],
            "guard_values": [],
            "sample_lineout_y_indices": [0],
            "sample_lineout": [value],
        }

    native_trace = {
        "diagnostic": "neutral_mixed_native_accepted_step_trace",
        "trace_points": [
            {
                "index": 0,
                "time": 0.0,
                "dt": 0.0,
                "solver_order": 0,
                "stage": "post_accepted",
                "fields": {"SNVh_parallel_viscosity": field(0.0)},
            }
        ],
    }
    reference_record = {
        "diagnostic": "neutral_mixed_reference_accepted_step_trace",
        "step_index": 0,
        "time": 0.0,
        "dt": 0.0,
        "stages": {"post_accepted": {"SNVh_parallel_viscosity": field(1.0)}},
    }
    native_path = tmp_path / "native_trace.json"
    reference_path = tmp_path / "reference_trace.jsonl"
    native_path.write_text(json.dumps(native_trace), encoding="utf-8")
    reference_path.write_text(json.dumps(reference_record) + "\n", encoding="utf-8")

    report = build_neutral_mixed_accepted_step_trace_parity_report(
        native_trace_json=native_path,
        reference_trace_json=reference_path,
        time_tolerance=1.0e-12,
    )

    register = report["parallel_viscosity_input_register"]
    assert register["missing_reference_input_fields"] == ["Vh", "eta_h"]
    entry = register["entries"][0]
    assert entry["input_fields_present"] is False
    assert entry["missing_input_fields"] == ["Vh", "eta_h"]
    assert entry["missing_closure_input_fields"] == ["Dnnh", "eta_h"]
    assert entry["closure_input_fields_present"] is False
    assert entry["closure_input_errors"] == {}
    assert entry["dominant_closure_input_field"] is None
    assert entry["state_input_fields_present"] is False
    assert entry["missing_state_input_fields"] == ["Nh", "Ph", "NVh"]
    assert entry["dominant_state_input_field"] is None
    assert entry["viscosity_to_state_target_ratio"] is None
    assert entry["diffusion_to_state_target_ratio"] is None
    assert entry["diagnosis"] == "reference_input_trace_missing"
    assert report["neutral_diffusion_ladder_register"]["entries"] == []


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
