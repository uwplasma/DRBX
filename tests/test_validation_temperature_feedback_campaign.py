from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

from jax_drb.validation import (
    create_temperature_feedback_campaign_package,
)
from jax_drb.validation.temperature_feedback_campaign import (
    _TemperatureFeedbackSeries,
    _build_temperature_feedback_series,
    _extract_scalar_series,
    _extract_target_temperature,
    _extract_time_points,
    _extract_spatial_series,
    _reconstruct_temperature_controller,
    _replace_bout_setting,
    _run_temperature_feedback_example,
    _stage_temperature_feedback_example,
    _strip_solver_option_lines,
    build_temperature_feedback_campaign,
)


def test_reconstruct_temperature_controller_matches_trapezoid_pi_update() -> None:
    time_points = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
    error = np.asarray([2.0, 1.0, -1.0], dtype=np.float64)

    integral_state, proportional_term, integral_term, multiplier = _reconstruct_temperature_controller(
        time_points=time_points,
        error=error,
        proportional_gain=10.0,
        integral_gain=0.5,
        integral_positive=False,
        source_positive=True,
    )

    np.testing.assert_allclose(integral_state, np.asarray([0.0, 1.5, 1.5], dtype=np.float64))
    np.testing.assert_allclose(proportional_term, np.asarray([20.0, 10.0, -10.0], dtype=np.float64))
    np.testing.assert_allclose(integral_term, np.asarray([0.0, 0.75, 0.75], dtype=np.float64))
    np.testing.assert_allclose(multiplier, np.asarray([20.0, 10.75, 0.0], dtype=np.float64))


def test_create_temperature_feedback_campaign_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.build_temperature_feedback_campaign",
        lambda **kwargs: {
            "summary": {
                "family": "temperature_feedback",
                "metric_count": 1,
                "passed_metric_count": 1,
                "metrics": [
                    {
                        "name": "temperature_feedback_src_mult_e_exact",
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
                    "target_temperature": np.asarray([0.1, 0.2], dtype=np.float64),
                    "setpoint": 0.15,
                    "reference_multiplier": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reconstructed_multiplier": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reference_proportional": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reconstructed_proportional": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reference_integral": np.asarray([0.0, 0.5], dtype=np.float64),
                    "reconstructed_integral": np.asarray([0.0, 0.5], dtype=np.float64),
                    "reference_integral_state": np.asarray([0.0, 50.0], dtype=np.float64),
                    "reconstructed_integral_state": np.asarray([0.0, 50.0], dtype=np.float64),
                    "reference_energy_source": np.asarray([[[[1.0]]], [[[2.0]]]], dtype=np.float64),
                    "reconstructed_energy_source": np.asarray([[[[1.0]]], [[[2.0]]]], dtype=np.float64),
                },
            )(),
        },
    )

    artifacts = create_temperature_feedback_campaign_package(
        output_root=tmp_path / "output",
        reference_root=tmp_path,
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "temperature_feedback"
    assert payload["passed_metric_count"] == 1


def test_replace_bout_setting_handles_numeric_values_without_backreference_bug() -> None:
    text = "nout = 400\nny = 80\n"

    updated = _replace_bout_setting(text, "nout", "4")
    updated = _replace_bout_setting(updated, "ny", "20")

    assert "nout = 4\n" in updated
    assert "ny = 20\n" in updated


def test_run_temperature_feedback_example_streams_to_file_without_capture_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invoked: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        invoked.update(kwargs)
        stdout = kwargs["stdout"]
        stdout.write("controller run\n")
        (Path(kwargs["cwd"]) / "BOUT.dmp.0.nc").write_bytes(b"stub")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign.subprocess.run", fake_run)

    _run_temperature_feedback_example(
        binary=tmp_path / "hermes",
        workdir=tmp_path,
        timeout_seconds=30,
    )

    assert invoked["stderr"] == subprocess.STDOUT
    assert invoked["timeout"] == 30
    assert "capture_output" not in invoked
    assert (tmp_path / "run.stdout").read_text(encoding="utf-8") == "controller run\n"


def test_build_temperature_feedback_campaign_maps_series_to_summary(monkeypatch) -> None:
    series = _TemperatureFeedbackSeries(
        time_points=np.asarray([0.0, 1.0], dtype=np.float64),
        target_temperature=np.asarray([0.5, 0.75], dtype=np.float64),
        setpoint=1.0,
        reference_multiplier=np.asarray([1.0, 1.2], dtype=np.float64),
        reconstructed_multiplier=np.asarray([1.0, 1.2], dtype=np.float64),
        reference_proportional=np.asarray([0.2, 0.1], dtype=np.float64),
        reconstructed_proportional=np.asarray([0.2, 0.1], dtype=np.float64),
        reference_integral=np.asarray([0.0, 0.05], dtype=np.float64),
        reconstructed_integral=np.asarray([0.0, 0.05], dtype=np.float64),
        reference_integral_state=np.asarray([0.0, 1.0], dtype=np.float64),
        reconstructed_integral_state=np.asarray([0.0, 1.0], dtype=np.float64),
        reference_energy_source=np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
        reconstructed_energy_source=np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
    )
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign._build_temperature_feedback_series",
        lambda **kwargs: (series, {"total": 1.0}),
    )

    report = build_temperature_feedback_campaign(reference_root="/tmp")

    assert report["summary"]["family"] == "temperature_feedback"
    assert report["summary"]["passed_metric_count"] == report["summary"]["metric_count"] == 6


def test_stage_temperature_feedback_example_rewrites_input(tmp_path: Path) -> None:
    example_dir = tmp_path / "example"
    example_dir.mkdir()
    (example_dir / "BOUT.inp").write_text(
        "nout = 40\ntimestep = 5\nny = 80\ntype = beuler\nsnes_type = newtonls\nksp_type = gmres\n",
        encoding="utf-8",
    )
    (example_dir / "extra.dat").write_text("payload", encoding="utf-8")
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    _stage_temperature_feedback_example(
        example_dir,
        workdir=workdir,
        nout=4,
        timestep=100.0,
        ny=16,
        solver_type="cvode",
    )

    updated = (workdir / "BOUT.inp").read_text(encoding="utf-8")
    assert "nout = 4" in updated
    assert "timestep = 100" in updated
    assert "ny = 16" in updated
    assert "type = cvode" in updated
    assert "snes_type" not in updated
    assert "ksp_type" not in updated
    assert (workdir / "extra.dat").read_text(encoding="utf-8") == "payload"


def test_strip_solver_option_lines_removes_beuler_only_options() -> None:
    text = "type = cvode\nsnes_type = newtonls\nksp_type = gmres\nlag_jacobian = 500\n"

    updated = _strip_solver_option_lines(text, ("snes_type", "ksp_type", "lag_jacobian"))

    assert "snes_type" not in updated
    assert "ksp_type" not in updated
    assert "lag_jacobian" not in updated
    assert "type = cvode" in updated


def test_extract_spatial_series_broadcasts_static_scalar() -> None:
    class _Variable:
        dimensions = ()

        def __getitem__(self, key):
            return np.asarray(3.5, dtype=np.float64)

    class _Dataset:
        variables = {"sample": _Variable()}

    extracted = _extract_spatial_series(_Dataset(), "sample", time_count=3)

    assert extracted.shape == (3, 1, 1, 1)
    np.testing.assert_allclose(extracted[:, 0, 0, 0], np.asarray([3.5, 3.5, 3.5], dtype=np.float64))


def test_replace_bout_setting_raises_when_key_is_missing() -> None:
    with pytest.raises(ValueError, match="Could not replace"):
        _replace_bout_setting("nout = 10\n", "ny", "8")


def test_run_temperature_feedback_example_raises_on_timeout_nonzero_and_missing_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="hermes", timeout=30)

    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign.subprocess.run", _raise_timeout)
    with pytest.raises(RuntimeError, match="did not finish within 30s"):
        _run_temperature_feedback_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)

    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=1),
    )
    with pytest.raises(RuntimeError, match="failed with exit code 1"):
        _run_temperature_feedback_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)

    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0),
    )
    with pytest.raises(FileNotFoundError, match="did not produce BOUT.dmp.0.nc"):
        _run_temperature_feedback_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)


def test_extract_temperature_helpers_cover_fallback_shapes() -> None:
    class _Variable:
        def __init__(self, values, dimensions):
            self._values = np.asarray(values, dtype=np.float64)
            self.dimensions = dimensions

        def __getitem__(self, key):
            return self._values

    dataset = SimpleNamespace(
        variables={
            "t": _Variable([0.0, 1.0], ("t",)),
            "Te": _Variable([[[[1.0], [2.0]]], [[[3.0], [4.0]]]], ("t", "x", "y", "z")),
            "scalar2d": _Variable([[1.0], [2.0]], ("t", "x")),
        }
    )

    np.testing.assert_allclose(_extract_time_points(dataset), np.asarray([0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(_extract_target_temperature(dataset, control_target=True), np.asarray([2.0, 4.0], dtype=np.float64))
    np.testing.assert_allclose(_extract_scalar_series(dataset, "scalar2d"), np.asarray([1.0, 2.0], dtype=np.float64))

    spatial_dataset = SimpleNamespace(variables={"sample": _Variable([[1.0, 2.0], [3.0, 4.0]], ("t", "y"))})
    extracted = _extract_spatial_series(spatial_dataset, "sample", time_count=2)
    assert extracted.shape == (2, 1, 2, 1)

    missing_time = SimpleNamespace(variables={})
    with pytest.raises(KeyError, match="missing time coordinate"):
        _extract_time_points(missing_time)


def test_reconstruct_temperature_controller_validates_shape_and_positive_clamps() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        _reconstruct_temperature_controller(
            time_points=np.asarray([0.0, 1.0], dtype=np.float64),
            error=np.asarray([1.0], dtype=np.float64),
            proportional_gain=1.0,
            integral_gain=1.0,
            integral_positive=False,
            source_positive=False,
        )

    integral_state, proportional_term, integral_term, multiplier = _reconstruct_temperature_controller(
        time_points=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        error=np.asarray([-1.0, -1.0, 1.0], dtype=np.float64),
        proportional_gain=1.0,
        integral_gain=1.0,
        integral_positive=True,
        source_positive=True,
    )

    assert integral_state[1] == 0.0
    assert multiplier[0] == 0.0
    assert proportional_term[-1] == 1.0
    assert integral_term[-1] >= 0.0


def test_build_temperature_feedback_series_loads_staged_reference_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    example_dir = tmp_path / "examples" / "tokamak-1D" / "extra" / "1D-recycling-with-Tt-control"
    example_dir.mkdir(parents=True)
    (example_dir / "BOUT.inp").write_text(
        "[hermes]\n"
        "Tnorm = 10\n"
        "\n"
        "[e]\n"
        "temperature_setpoint = 5\n"
        "temperature_controller_p = 2.0\n"
        "temperature_controller_i = 0.5\n"
        "control_target_temperature = true\n"
        "temperature_integral_positive = false\n"
        "temperature_source_positive = true\n"
        "\n"
        "nout = 40\n"
        "timestep = 5\n"
        "ny = 80\n"
        "type = beuler\n",
        encoding="utf-8",
    )

    time_points = np.asarray([0.0, 1.0], dtype=np.float64)
    target_temperature = np.asarray([0.0, 0.25], dtype=np.float64)
    error = 0.5 - target_temperature
    integral_state, proportional, integral, multiplier = _reconstruct_temperature_controller(
        time_points=time_points,
        error=error,
        proportional_gain=2.0,
        integral_gain=0.5,
        integral_positive=False,
        source_positive=True,
    )
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.discover_reference_binary",
        lambda **kwargs: tmp_path / "hermes",
    )

    def _fake_run(*, binary, workdir, timeout_seconds):
        with Dataset(workdir / "BOUT.dmp.0.nc", "w") as dataset:
            dataset.createDimension("t", 2)
            dataset.createDimension("x", 1)
            dataset.createDimension("y", 1)
            dataset.createDimension("z", 1)
            dataset.createVariable("t_array", "f8", ("t",))[:] = time_points
            dataset.createVariable("Te", "f8", ("t", "x", "y", "z"))[:] = target_temperature[:, None, None, None]
            dataset.createVariable("temperature_feedback_src_mult_e", "f8", ("t",))[:] = multiplier
            dataset.createVariable("temperature_feedback_src_p_e", "f8", ("t",))[:] = proportional
            dataset.createVariable("temperature_feedback_src_i_e", "f8", ("t",))[:] = integral
            dataset.createVariable("e_temperature_error_integral", "f8", ("t",))[:] = integral_state
            dataset.createVariable("temperature_feedback_src_shape_e", "f8", ("t", "x", "y", "z"))[:] = 1.0
            dataset.createVariable("SPe_feedback", "f8", ("t", "x", "y", "z"))[:] = multiplier[:, None, None, None]

    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign._run_temperature_feedback_example", _fake_run)

    series, timing = _build_temperature_feedback_series(
        reference_root=tmp_path,
        reference_binary=None,
        nout=4,
        timestep=100.0,
        ny=16,
        solver_type="cvode",
        timeout_seconds=30,
    )

    np.testing.assert_allclose(series.reference_multiplier, multiplier)
    np.testing.assert_allclose(series.reconstructed_multiplier, multiplier)
    np.testing.assert_allclose(series.reference_energy_source.reshape(2), multiplier)
    assert timing["total"] >= 0.0
