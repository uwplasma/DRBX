from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.validation import (
    analyze_blob2d_array_payload,
    compare_blob2d_analysis_results,
    compare_blob2d_array_payloads,
    load_blob2d_analysis_json,
    save_blob2d_parity_plot,
    write_blob2d_analysis_json,
    write_blob2d_parity_json,
)


def _reference_analysis():
    return load_blob2d_analysis_json(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_metrics"
        / "blob2d_short_window_metrics.json"
    )


def _small_reference_payload():
    return load_portable_array_payload(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "blob2d_one_step.npz"
    )


def test_analyze_blob2d_extracts_peak_and_center_of_mass_histories() -> None:
    result = _reference_analysis()

    assert result.density_variable == "Ne"
    assert result.background_density == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert result.time_points.shape == (51,)
    assert result.peak_excess_history.shape == (51,)
    assert result.center_of_mass_x_history.shape == (51,)
    assert result.center_of_mass_z_history.shape == (51,)
    assert result.peak_excess_history[-1] == pytest.approx(0.21942969845857396, rel=1e-12, abs=1e-12)
    assert result.center_of_mass_x_history[-1] == pytest.approx(153.841366748258, rel=1e-12, abs=1e-12)
    assert result.center_of_mass_z_history[-1] == pytest.approx(131.181047392118, rel=1e-12, abs=1e-12)


def test_compare_blob2d_identical_payload_has_zero_errors() -> None:
    analysis = _reference_analysis()
    result = compare_blob2d_analysis_results(analysis, analysis)

    assert result.peak_max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.peak_rms_error == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.center_of_mass_x_max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.center_of_mass_z_max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_compare_blob2d_tracks_density_offset_error() -> None:
    expected = _small_reference_payload()
    actual = {
        **expected,
        "variables": {name: np.asarray(value, dtype=np.float64).copy() for name, value in expected["variables"].items()},
    }
    excess_shape = np.maximum(actual["variables"]["Ne"][-1] - 1.0, 0.0)
    actual["variables"]["Ne"][-1] += 1.0e-3 * excess_shape / np.max(excess_shape)

    result = compare_blob2d_array_payloads(expected, actual)

    assert result.peak_max_abs_error == pytest.approx(1.0e-3, rel=1e-3, abs=1e-6)
    assert result.peak_rms_error == pytest.approx(1.0e-3 / np.sqrt(2.0), rel=1e-3, abs=1e-6)
    assert result.center_of_mass_x_max_abs_error < 1.0e-10
    assert result.center_of_mass_z_max_abs_error < 1.0e-10


def test_blob2d_parity_json_and_plot_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    analysis = _reference_analysis()
    result = compare_blob2d_analysis_results(analysis, analysis)

    json_path = write_blob2d_parity_json(result, tmp_path / "blob2d_parity.json")
    plot_path = save_blob2d_parity_plot(result, tmp_path / "blob2d_parity.png")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["peak_max_abs_error"] == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert data["center_of_mass_x_max_abs_error"] == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_blob2d_analysis_json_round_trip(tmp_path: Path) -> None:
    analysis = _reference_analysis()
    path = write_blob2d_analysis_json(analysis, tmp_path / "blob2d_analysis.json")
    loaded = load_blob2d_analysis_json(path)

    assert loaded.density_variable == analysis.density_variable
    np.testing.assert_allclose(loaded.time_points, analysis.time_points)
    np.testing.assert_allclose(loaded.peak_excess_history, analysis.peak_excess_history)
