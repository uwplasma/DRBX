from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

from jax_drb.validation import ControllerFeedbackMetric, create_controller_feedback_campaign_package
from jax_drb.validation.controller_feedback_campaign import (
    _case_input_path,
    _controller_integral_series_from_term,
    _build_controller_series_report,
    _extract_field_series_at_target_cell,
    _native_history_and_diagnostics,
    _save_controller_feedback_plot,
    build_controller_feedback_campaign,
)


def test_create_controller_feedback_campaign_package_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.controller_feedback_campaign.build_controller_feedback_campaign",
        lambda **kwargs: (
            ControllerFeedbackMetric(
                name="density_feedback_src_mult_d+",
                max_abs_diff=1.0e-4,
                target=5.0e-4,
                passed=True,
                notes="demo",
            ),
        ),
    )
    artifacts = create_controller_feedback_campaign_package(
        output_root=tmp_path / "output",
        reference_root=tmp_path,
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "controller_feedback"
    assert payload["passed_metric_count"] == 1


def test_build_controller_feedback_campaign_maps_series_report_to_metric_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.controller_feedback_campaign._build_controller_series_report",
        lambda **kwargs: {
            "d+_density_error_integral": 2.5e-3,
            "density_feedback_src_mult_d+": 4.0e-4,
            "density_feedback_src_p_d+": 4.0e-4,
            "density_feedback_src_i_d+": 1.0e-6,
            "Sd_target_recycle": 1.0e-4,
            "Sd+_feedback": 0.0,
        },
    )

    metrics = build_controller_feedback_campaign(reference_root="/tmp")

    assert len(metrics) == 6
    assert all(metric.passed for metric in metrics)
    assert metrics[0].name == "d+_density_error_integral"
    assert metrics[-1].name == "Sd+_feedback"


def test_controller_integral_series_from_term_handles_zero_gain() -> None:
    values = _controller_integral_series_from_term([1.0, 2.0, 3.0], controller_gain=0.0)

    assert list(values) == [0.0, 0.0, 0.0]


def test_build_controller_series_report_matches_reference_and_native_histories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "BOUT.inp"
    input_path.write_text(
        "[d+]\n"
        "type = density, upstream_density_feedback\n"
        "density_controller_i = 2.0\n"
        "\n"
        "[d]\n"
        "target_recycle = true\n"
        "recycle_as = d\n",
        encoding="utf-8",
    )

    workdir = tmp_path / "reference"
    workdir.mkdir()
    with Dataset(workdir / "BOUT.dmp.0.nc", "w") as dataset:
        dataset.createDimension("t", 2)
        dataset.createDimension("x", 1)
        dataset.createDimension("y", 1)
        dataset.createDimension("z", 1)
        dataset.createVariable("density_feedback_src_mult_d+", "f8", ("t",))[:] = np.asarray([1.0, 1.1])
        dataset.createVariable("density_feedback_src_p_d+", "f8", ("t",))[:] = np.asarray([0.5, 0.4])
        dataset.createVariable("density_feedback_src_i_d+", "f8", ("t",))[:] = np.asarray([0.0, 0.2])
        dataset.createVariable("Sd+_feedback", "f8", ("t", "x", "y", "z"))[:] = np.asarray([[[[0.0]]], [[[0.0]]]])
        dataset.createVariable("Sd_target_recycle", "f8", ("t", "x", "y", "z"))[:] = np.asarray([[[[0.1]]], [[[0.2]]]])

    monkeypatch.setattr(
        "jax_drb.validation.controller_feedback_campaign._case_input_path",
        lambda case_name, reference_root: input_path,
    )
    monkeypatch.setattr(
        "jax_drb.validation.controller_feedback_campaign.run_reference_case",
        lambda *args, **kwargs: SimpleNamespace(summary=SimpleNamespace(workdir=workdir)),
    )
    monkeypatch.setattr(
        "jax_drb.validation.controller_feedback_campaign._native_history_and_diagnostics",
        lambda **kwargs: (
            {},
            {
                "density_feedback_src_mult_d+": np.asarray([1.0, 1.1], dtype=np.float64),
                "density_feedback_src_p_d+": np.asarray([0.5, 0.4], dtype=np.float64),
                "density_feedback_src_i_d+": np.asarray([0.0, 0.2], dtype=np.float64),
                "Sd+_feedback": np.asarray([[[[0.0]]], [[[0.0]]]], dtype=np.float64),
                "Sd_target_recycle": np.asarray([[[[0.1]]], [[[0.2]]]], dtype=np.float64),
            },
            np.asarray([0.0, 25.0], dtype=np.float64),
            SimpleNamespace(xstart=0, ystart=0, yend=0),
        ),
    )

    summary = _build_controller_series_report(
        case_name="recycling_1d_one_step",
        reference_root=tmp_path,
        reference_binary=None,
        timestep=25.0,
        steps=1,
    )

    assert summary["density_feedback_src_mult_d+"] == 0.0
    assert summary["density_feedback_src_p_d+"] == 0.0
    assert summary["density_feedback_src_i_d+"] == 0.0
    assert summary["d+_density_error_integral"] == 0.0
    assert summary["Sd+_feedback"] == 0.0
    assert summary["Sd_target_recycle"] == 0.0


def test_case_input_path_rejects_unknown_case(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported controller campaign case"):
        _case_input_path("unknown_case", tmp_path)


def test_extract_field_series_at_target_cell_supports_lower_edge_and_validates_rank() -> None:
    values = np.asarray(
        [
            [[[1.0], [2.0]]],
            [[[3.0], [4.0]]],
        ],
        dtype=np.float64,
    )
    mesh = SimpleNamespace(xstart=0, ystart=0, yend=1)

    extracted = _extract_field_series_at_target_cell(values, mesh, target_edge="lower")

    np.testing.assert_allclose(extracted, np.asarray([1.0, 3.0], dtype=np.float64))
    with pytest.raises(ValueError, match="expected a 4D time series"):
        _extract_field_series_at_target_cell(np.asarray([1.0, 2.0], dtype=np.float64), mesh)


def test_native_history_and_diagnostics_stacks_rhs_time_series(monkeypatch, tmp_path: Path) -> None:
    history = SimpleNamespace(
        variable_history={"Nd+": np.asarray([[[[1.0]]], [[[2.0]]]], dtype=np.float64)},
        feedback_integral_history={"d+": np.asarray([0.0, 1.0], dtype=np.float64)},
    )
    mesh = SimpleNamespace()
    recorded_integrals: list[dict[str, float]] = []

    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign._case_input_path", lambda *args, **kwargs: tmp_path / "BOUT.inp")
    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign.load_bout_input", lambda path: object())
    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign.RunConfiguration.from_config", lambda config: object())
    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign.build_structured_mesh", lambda config, run_config: mesh)
    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign.build_structured_metrics", lambda config, run_config, mesh: object())
    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign.resolved_dataset_scalars", lambda run_config: {})
    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign.advance_recycling_1d_implicit_history", lambda *args, **kwargs: history)

    def _fake_rhs(*args, **kwargs):
        recorded_integrals.append(kwargs["feedback_integrals"])
        marker = kwargs["field_overrides"]["Nd+"].reshape(-1)[0]
        return SimpleNamespace(variables={"Sd+": np.asarray([[[[marker]]]], dtype=np.float64)})

    monkeypatch.setattr("jax_drb.validation.controller_feedback_campaign.compute_recycling_1d_rhs", _fake_rhs)

    variable_history, diagnostics, time_points, returned_mesh = _native_history_and_diagnostics(
        case_name="recycling_1d_one_step",
        reference_root=tmp_path,
        timestep=25.0,
        steps=1,
    )

    assert variable_history is history.variable_history
    np.testing.assert_allclose(diagnostics["Sd+"].reshape(-1), np.asarray([1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(time_points, np.asarray([0.0, 25.0], dtype=np.float64))
    assert returned_mesh is mesh
    assert recorded_integrals == [{"d+": 0.0}, {"d+": 1.0}]


def test_save_controller_feedback_plot_writes_png(tmp_path: Path) -> None:
    metrics = (
        ControllerFeedbackMetric("a_metric", 1.0e-4, 2.0e-4, True, "ok"),
        ControllerFeedbackMetric("b_metric", 3.0e-4, 2.0e-4, False, "bad"),
    )

    output_path = tmp_path / "controller_feedback.png"
    _save_controller_feedback_plot(metrics, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
