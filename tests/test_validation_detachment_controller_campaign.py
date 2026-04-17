from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

from jax_drb.validation import create_detachment_controller_campaign_package
from jax_drb.validation.detachment_controller_campaign import (
    _build_detachment_controller_series,
    _calculate_gradient,
    _extract_scalar_series,
    _extract_spatial_series,
    _extract_time_points,
    _parse_bool,
    _read_bout_section_options,
    _reconstruct_detachment_proportional_term,
    _reconstruct_detachment_controller,
    _run_detachment_controller_example,
    _stage_detachment_controller_example,
    _strip_solver_option_lines,
    build_detachment_controller_campaign,
)


def test_reconstruct_detachment_controller_matches_reduced_position_form() -> None:
    control, proportional, integral, derivative = _reconstruct_detachment_controller(
        time_seconds=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        front_location=np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
        setpoint=1.0,
        velocity_form=False,
        min_time_for_change=0.0,
        min_error_for_change=0.0,
        minval_for_source_multiplier=-np.inf,
        maxval_for_source_multiplier=np.inf,
        control_offset=0.0,
        initial_control=1.0,
        controller_gain=2.0,
        integral_time=10.0,
        derivative_time=0.0,
        buffer_size=5,
        response_sign=1.0,
        reset_integral_on_first_crossing=True,
        settling_time=0.0,
    )

    np.testing.assert_allclose(control, np.asarray([1.0, 1.0, 1.1], dtype=np.float64))
    np.testing.assert_allclose(proportional, np.asarray([0.0, 1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(integral, np.asarray([0.0, 0.0, 0.1], dtype=np.float64))
    np.testing.assert_allclose(derivative, np.asarray([0.0, 0.0, 0.0], dtype=np.float64))


def test_strip_solver_option_lines_removes_beuler_only_options() -> None:
    text = "type = cvode\nsnes_type = newtonls\nksp_type = gmres\nlag_jacobian = 500\n"

    updated = _strip_solver_option_lines(text, ("snes_type", "ksp_type", "lag_jacobian"))

    assert "snes_type" not in updated
    assert "ksp_type" not in updated
    assert "lag_jacobian" not in updated
    assert "type = cvode" in updated


def test_create_detachment_controller_campaign_package_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.detachment_controller_campaign.build_detachment_controller_campaign",
        lambda **kwargs: {
            "summary": {
                "family": "impurity_radiation_and_detachment_control",
                "metric_count": 1,
                "passed_metric_count": 1,
                "metrics": [
                    {
                        "name": "detachment_control_src_mult_balance_exact",
                        "kind": "max_abs_error",
                        "value": 1.0e-12,
                        "target": 1.0e-12,
                        "passed": True,
                        "notes": "demo",
                    }
                ],
            },
            "series": type(
                "Series",
                (),
                {
                    "time_points": np.asarray([0.0, 1.0], dtype=np.float64),
                    "front_location": np.asarray([0.2, 0.2], dtype=np.float64),
                    "setpoint": 1.0,
                    "source_multiplier": np.asarray([1.0, 1.2], dtype=np.float64),
                    "reconstructed_multiplier": np.asarray([1.0, 1.2], dtype=np.float64),
                    "proportional_term": np.asarray([0.0, 0.2], dtype=np.float64),
                    "reconstructed_proportional_term": np.asarray([0.0, 0.2], dtype=np.float64),
                    "integral_term": np.asarray([0.0, 0.01], dtype=np.float64),
                    "derivative_term": np.asarray([0.0, 0.0], dtype=np.float64),
                    "source_feedback": np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
                    "reconstructed_source_feedback": np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
                },
            )(),
        },
    )

    artifacts = create_detachment_controller_campaign_package(
        output_root=tmp_path / "output",
        reference_root=tmp_path,
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "impurity_radiation_and_detachment_control"
    assert payload["passed_metric_count"] == 1


def test_build_detachment_controller_campaign_maps_series_to_summary(monkeypatch) -> None:
    series = type(
        "Series",
        (),
        {
            "time_points": np.asarray([0.0, 1.0], dtype=np.float64),
            "front_location": np.asarray([0.2, 0.2], dtype=np.float64),
            "setpoint": 1.0,
            "source_multiplier": np.asarray([1.0, 1.2], dtype=np.float64),
            "reconstructed_multiplier": np.asarray([1.0, 1.2], dtype=np.float64),
            "proportional_term": np.asarray([0.0, 0.2], dtype=np.float64),
            "reconstructed_proportional_term": np.asarray([0.0, 0.2], dtype=np.float64),
            "integral_term": np.asarray([0.0, 0.01], dtype=np.float64),
            "derivative_term": np.asarray([0.0, 0.0], dtype=np.float64),
            "source_feedback": np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
            "reconstructed_source_feedback": np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
        },
    )()
    monkeypatch.setattr(
        "jax_drb.validation.detachment_controller_campaign._build_detachment_controller_series",
        lambda **kwargs: (series, {"total": 1.0}),
    )

    report = build_detachment_controller_campaign(reference_root="/tmp")

    assert report["summary"]["family"] == "impurity_radiation_and_detachment_control"
    assert report["summary"]["passed_metric_count"] == report["summary"]["metric_count"] == 6


def test_stage_detachment_controller_example_rewrites_and_strips_solver_lines(tmp_path: Path) -> None:
    example_dir = tmp_path / "example"
    example_dir.mkdir()
    (example_dir / "BOUT.inp").write_text(
        "nout = 40\n"
        "timestep = 5\n"
        "ny = 80\n"
        "type = beuler\n"
        "settling_time = 100\n"
        "snes_type = newtonls\n"
        "ksp_type = gmres\n",
        encoding="utf-8",
    )

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    _stage_detachment_controller_example(
        example_dir,
        workdir=workdir,
        nout=4,
        timestep=100.0,
        ny=8,
        solver_type="cvode",
        settling_time=0.0,
    )

    updated = (workdir / "BOUT.inp").read_text(encoding="utf-8")
    assert "nout = 4" in updated
    assert "timestep = 100" in updated
    assert "ny = 8" in updated
    assert "type = cvode" in updated
    assert "settling_time = 0" in updated
    assert "snes_type" not in updated
    assert "ksp_type" not in updated


def test_parse_detachment_section_options_and_booleans(tmp_path: Path) -> None:
    bout_input = tmp_path / "BOUT.inp"
    bout_input.write_text(
        "[detachment_controller]\n"
        "velocity_form = true\n"
        "actuator = \"particles\"\n",
        encoding="utf-8",
    )

    options = _read_bout_section_options(bout_input, section="detachment_controller")

    assert _parse_bool(options["velocity_form"]) is True
    assert _parse_bool("false") is False

    with pytest.raises(ValueError, match="Could not parse boolean value"):
        _parse_bool("maybe")
    with pytest.raises(KeyError, match="Missing section"):
        _read_bout_section_options(bout_input, section="missing")


def test_reconstruct_detachment_proportional_term_uses_signed_front_error() -> None:
    term = _reconstruct_detachment_proportional_term(
        front_location=np.asarray([0.6, 0.4], dtype=np.float64),
        setpoint=0.5,
        controller_gain=2.0,
        response_sign=-1.0,
    )

    np.testing.assert_allclose(term, np.asarray([0.2, -0.2], dtype=np.float64))


def test_run_detachment_controller_example_raises_on_timeout_nonzero_and_missing_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="hermes", timeout=30)

    monkeypatch.setattr("jax_drb.validation.detachment_controller_campaign.subprocess.run", _raise_timeout)
    with pytest.raises(RuntimeError, match="did not finish within 30s"):
        _run_detachment_controller_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)

    monkeypatch.setattr(
        "jax_drb.validation.detachment_controller_campaign.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=2),
    )
    with pytest.raises(RuntimeError, match="failed with exit code 2"):
        _run_detachment_controller_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)

    monkeypatch.setattr(
        "jax_drb.validation.detachment_controller_campaign.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0),
    )
    with pytest.raises(FileNotFoundError, match="did not produce BOUT.dmp.0.nc"):
        _run_detachment_controller_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)


def test_detachment_extract_helpers_cover_fallback_shapes() -> None:
    dataset = SimpleNamespace(
        variables={
            "t": np.asarray([0.0, 1.0], dtype=np.float64),
            "scalar2d": np.asarray([[1.0], [2.0]], dtype=np.float64),
        }
    )
    np.testing.assert_allclose(_extract_time_points(dataset), np.asarray([0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(_extract_scalar_series(dataset, "scalar2d"), np.asarray([1.0, 2.0], dtype=np.float64))

    class _Variable:
        def __init__(self, values, dimensions):
            self._values = np.asarray(values, dtype=np.float64)
            self.dimensions = dimensions

        def __getitem__(self, key):
            return self._values

    extracted = _extract_spatial_series(
        SimpleNamespace(variables={"sample": _Variable([[1.0, 2.0], [3.0, 4.0]], ("t", "y"))}),
        "sample",
        time_count=2,
    )
    assert extracted.shape == (2, 1, 2, 1)

    with pytest.raises(KeyError, match="missing time coordinate"):
        _extract_time_points(SimpleNamespace(variables={}))


def test_detachment_gradient_and_velocity_form_controller_cover_remaining_branches() -> None:
    assert _calculate_gradient([], []) == 0.0
    assert _calculate_gradient([1.0], [2.0]) == 0.0

    control, proportional, integral, derivative = _reconstruct_detachment_controller(
        time_seconds=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        front_location=np.asarray([1.5, 0.5, 1.5], dtype=np.float64),
        setpoint=1.0,
        velocity_form=True,
        min_time_for_change=0.0,
        min_error_for_change=0.0,
        minval_for_source_multiplier=0.0,
        maxval_for_source_multiplier=10.0,
        control_offset=0.0,
        initial_control=1.0,
        controller_gain=2.0,
        integral_time=np.inf,
        derivative_time=0.0,
        buffer_size=1,
        response_sign=1.0,
        reset_integral_on_first_crossing=True,
        settling_time=0.0,
    )

    assert np.all(np.isfinite(control))
    assert np.all(np.isfinite(proportional))
    assert np.all(np.isfinite(integral))
    assert np.all(np.isfinite(derivative))
    assert control[0] == 1.0


def test_build_detachment_controller_series_loads_staged_reference_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    example_dir = tmp_path / "examples" / "tokamak-1D" / "extra" / "1D-recycling-with-detachment-control"
    example_dir.mkdir(parents=True)
    (example_dir / "BOUT.inp").write_text(
        "[detachment_controller]\n"
        "detachment_front_setpoint = 1.0\n"
        "velocity_form = false\n"
        "min_time_for_change = 0.0\n"
        "min_error_for_change = 0.0\n"
        "minval_for_source_multiplier = 0.0\n"
        "maxval_for_source_multiplier = 10.0\n"
        "actuator = \"particles\"\n"
        "initial_control = 1.0\n"
        "control_offset = 0.2\n"
        "reset_integral_on_first_crossing = true\n"
        "controller_gain = 2.0\n"
        "integral_time = inf\n"
        "derivative_time = 0.0\n"
        "buffer_size = 5\n"
        "settling_time = 0.0\n"
        "\n"
        "nout = 40\n"
        "timestep = 5\n"
        "ny = 80\n"
        "type = beuler\n",
        encoding="utf-8",
    )

    time_points = np.asarray([0.0, 1.0], dtype=np.float64)
    front_location = np.asarray([0.5, 0.5], dtype=np.float64)
    control, proportional, integral, derivative = _reconstruct_detachment_controller(
        time_seconds=time_points,
        front_location=front_location,
        setpoint=1.0,
        velocity_form=False,
        min_time_for_change=0.0,
        min_error_for_change=0.0,
        minval_for_source_multiplier=0.0,
        maxval_for_source_multiplier=10.0,
        control_offset=0.2,
        initial_control=1.0,
        controller_gain=2.0,
        integral_time=np.inf,
        derivative_time=0.0,
        buffer_size=5,
        response_sign=1.0,
        reset_integral_on_first_crossing=True,
        settling_time=0.0,
    )
    monkeypatch.setattr(
        "jax_drb.validation.detachment_controller_campaign.discover_reference_binary",
        lambda **kwargs: tmp_path / "hermes",
    )

    def _fake_run(*, binary, workdir, timeout_seconds):
        with Dataset(workdir / "BOUT.dmp.0.nc", "w") as dataset:
            dataset.createDimension("t", 2)
            dataset.createDimension("x", 1)
            dataset.createDimension("y", 1)
            dataset.createDimension("z", 1)
            dataset.createDimension("scalar", 1)
            dataset.createVariable("t_array", "f8", ("t",))[:] = time_points
            dataset.createVariable("Omega_ci", "f8", ("scalar",))[:] = np.asarray([1.0], dtype=np.float64)
            dataset.createVariable("detachment_front_location", "f8", ("t",))[:] = front_location
            dataset.createVariable("detachment_control_src_mult", "f8", ("t",))[:] = control
            dataset.createVariable("detachment_control_proportional_term", "f8", ("t",))[:] = proportional
            dataset.createVariable("detachment_control_integral_term", "f8", ("t",))[:] = integral
            dataset.createVariable("detachment_control_derivative_term", "f8", ("t",))[:] = derivative
            dataset.createVariable("detachment_control_src_shape", "f8", ("t", "x", "y", "z"))[:] = 1.0
            dataset.createVariable("detachment_source_feedback", "f8", ("t", "x", "y", "z"))[:] = control[:, None, None, None]

    monkeypatch.setattr("jax_drb.validation.detachment_controller_campaign._run_detachment_controller_example", _fake_run)

    series, timing = _build_detachment_controller_series(
        reference_root=tmp_path,
        reference_binary=None,
        nout=4,
        timestep=100.0,
        ny=8,
        solver_type="cvode",
        settling_time=0.0,
        timeout_seconds=30,
    )

    np.testing.assert_allclose(series.source_multiplier, control)
    np.testing.assert_allclose(series.reconstructed_multiplier, control)
    np.testing.assert_allclose(series.proportional_term, proportional)
    np.testing.assert_allclose(series.reconstructed_proportional_term, proportional)
    assert timing["total"] >= 0.0
