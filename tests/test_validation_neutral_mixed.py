from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jax_drb.native import run_curated_case, run_input_case
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.validation import (
    analyze_neutral_mixed_array_payload,
    compare_neutral_mixed_analysis_results,
    compare_neutral_mixed_artifacts,
    compare_neutral_mixed_array_payloads,
    load_neutral_mixed_analysis_json,
    save_neutral_mixed_diagnostic_plot,
    save_neutral_mixed_parity_plot,
    write_neutral_mixed_analysis_json,
    write_neutral_mixed_parity_json,
)

_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_REFERENCE_INPUT = _REFERENCE_ROOT / "tests" / "integrated" / "neutral_mixed" / "data" / "BOUT.inp"


def _reference_analysis():
    return load_neutral_mixed_analysis_json(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_metrics"
        / "neutral_mixed_short_window_metrics.json"
    )


def _reference_payload():
    return load_portable_array_payload(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "neutral_mixed_short_window.npz"
    )


def _small_reference_payload():
    return load_portable_array_payload(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "neutral_mixed_one_step.npz"
    )


def _cropped_payload(payload: dict[str, object], *, output_points: int) -> dict[str, object]:
    cropped = {**payload}
    cropped["time_points"] = list(payload["time_points"][:output_points])
    cropped["variables"] = {
        name: np.asarray(value, dtype=np.float64)[:output_points]
        for name, value in payload["variables"].items()
    }
    return cropped


def test_analyze_neutral_mixed_matches_committed_short_window_metrics() -> None:
    result = _reference_analysis()

    assert result.density_variable == "Nh"
    assert result.pressure_variable == "Ph"
    assert result.momentum_variable == "NVh"
    assert (result.center_index_x, result.center_index_y, result.center_index_z) == (5, 3, 5)
    assert result.time_points.shape == (16,)
    assert result.center_density_history[-1] == pytest.approx(0.786199787445947, rel=1e-12, abs=1e-12)
    assert result.center_pressure_history[-1] == pytest.approx(0.07861859645931955, rel=1e-12, abs=1e-12)
    assert result.center_momentum_history[1] == pytest.approx(-0.0010341777519552598, rel=1e-12, abs=1e-12)
    assert result.center_temperature_history[-1] == pytest.approx(0.09999824181423447, rel=1e-12, abs=1e-12)
    assert result.total_density_history[-1] == pytest.approx(786.1978749225259, rel=1e-12, abs=1e-12)
    assert result.total_pressure_history[-1] == pytest.approx(78.61840629869042, rel=1e-12, abs=1e-12)
    assert result.momentum_rms_history[-1] == pytest.approx(5.561217670555262e-08, rel=1e-12, abs=1e-12)


def test_compare_neutral_mixed_identical_payload_has_zero_errors() -> None:
    analysis = _reference_analysis()
    result = compare_neutral_mixed_analysis_results(analysis, analysis)

    assert set(result.series_errors) == {
        "center_density",
        "center_momentum",
        "center_pressure",
        "center_temperature",
        "momentum_rms",
        "total_density",
        "total_pressure",
    }
    for error in result.series_errors.values():
        assert error.max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert error.rms_error == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert np.all(error.error_history == 0.0)


def test_compare_neutral_mixed_tracks_center_density_offset() -> None:
    expected = _small_reference_payload()
    actual = {
        **expected,
        "variables": {name: np.asarray(value, dtype=np.float64).copy() for name, value in expected["variables"].items()},
    }
    actual["variables"]["Nh"][:, 5, 3, 5] += 1.0e-3

    result = compare_neutral_mixed_array_payloads(
        expected,
        actual,
        x_index=5,
        y_index=3,
        z_index=5,
    )

    assert result.series_errors["center_density"].max_abs_error == pytest.approx(1.0e-3, rel=1e-12, abs=1e-12)
    assert result.series_errors["center_density"].rms_error == pytest.approx(1.0e-3, rel=1e-12, abs=1e-12)
    assert result.series_errors["total_density"].max_abs_error == pytest.approx(1.0e-3, rel=1e-12, abs=1e-12)
    assert result.series_errors["total_density"].rms_error == pytest.approx(1.0e-3, rel=1e-12, abs=1e-12)
    assert result.series_errors["center_pressure"].max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.series_errors["center_momentum"].max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_compare_neutral_mixed_artifacts_accepts_json_and_npz() -> None:
    analysis = _reference_analysis()
    payload = _reference_payload()

    result = compare_neutral_mixed_artifacts(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_metrics"
        / "neutral_mixed_short_window_metrics.json",
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "neutral_mixed_short_window.npz",
        x_index=analysis.center_index_x,
        y_index=analysis.center_index_y,
        z_index=analysis.center_index_z,
    )

    for error in result.series_errors.values():
        assert error.max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)

    recomputed = analyze_neutral_mixed_array_payload(
        payload,
        x_index=analysis.center_index_x,
        y_index=analysis.center_index_y,
        z_index=analysis.center_index_z,
    )
    np.testing.assert_allclose(recomputed.center_density_history, analysis.center_density_history)


def test_neutral_mixed_analysis_json_and_plot_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    analysis = _reference_analysis()
    parity = compare_neutral_mixed_analysis_results(analysis, analysis)

    analysis_path = write_neutral_mixed_analysis_json(analysis, tmp_path / "neutral_analysis.json")
    parity_path = write_neutral_mixed_parity_json(parity, tmp_path / "neutral_parity.json")
    diagnostic_plot = save_neutral_mixed_diagnostic_plot(analysis, tmp_path / "neutral_diagnostic.png")
    parity_plot = save_neutral_mixed_parity_plot(parity, tmp_path / "neutral_parity.png")

    loaded = load_neutral_mixed_analysis_json(analysis_path)
    assert loaded.center_index_y == analysis.center_index_y
    np.testing.assert_allclose(loaded.center_momentum_history, analysis.center_momentum_history)

    data = json.loads(parity_path.read_text(encoding="utf-8"))
    assert data["series_errors"]["center_density"]["max_abs_error"] == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert diagnostic_plot.exists()
    assert diagnostic_plot.stat().st_size > 0
    assert parity_plot.exists()
    assert parity_plot.stat().st_size > 0


def test_neutral_mixed_one_step_native_parity_stays_within_operational_center_band() -> None:
    if os.environ.get("JAX_DRB_RUN_NEUTRAL_MIXED_ONE_STEP_PARITY") != "1":
        pytest.skip("set JAX_DRB_RUN_NEUTRAL_MIXED_ONE_STEP_PARITY=1 to run the bounded neutral one-step parity gate")

    expected = _small_reference_payload()
    result = run_curated_case("neutral_mixed_one_step", reference_root=_REFERENCE_ROOT)
    actual = {
        **expected,
        "time_points": list(result.payload["time_points"]),
        "variables": {name: np.asarray(value, dtype=np.float64) for name, value in result.variables.items()},
    }

    parity = compare_neutral_mixed_array_payloads(
        expected,
        actual,
        x_index=5,
        y_index=3,
        z_index=5,
    )

    assert parity.series_errors["center_density"].max_abs_error <= 8.5e-3
    assert parity.series_errors["center_pressure"].max_abs_error <= 6.5e-4
    assert parity.series_errors["center_momentum"].max_abs_error <= 1.0e-3
    assert parity.series_errors["center_temperature"].max_abs_error <= 3.5e-4
    assert parity.series_errors["total_density"].max_abs_error <= 3.5e-1
    assert parity.series_errors["total_pressure"].max_abs_error <= 3.5e-2
    assert parity.series_errors["momentum_rms"].max_abs_error <= 2.0e-3


def test_neutral_mixed_short_window_prefix_native_parity_stays_within_operational_center_band() -> None:
    if os.environ.get("JAX_DRB_RUN_NEUTRAL_MIXED_SHORT_WINDOW_PREFIX") != "1":
        pytest.skip("set JAX_DRB_RUN_NEUTRAL_MIXED_SHORT_WINDOW_PREFIX=1 to run the bounded neutral short-window prefix gate")
    if not _REFERENCE_INPUT.exists():
        pytest.skip("local neutral_mixed reference input is unavailable")

    expected = _cropped_payload(_reference_payload(), output_points=4)
    result = run_input_case(
        _REFERENCE_INPUT,
        case_name="neutral_mixed_short_window_prefix",
        parity_mode="short_window",
        compare_variables=("Nh", "Ph", "NVh"),
        output_steps=3,
    )
    actual = {
        **expected,
        "time_points": list(result.payload["time_points"]),
        "variables": {name: np.asarray(value, dtype=np.float64) for name, value in result.variables.items()},
    }

    parity = compare_neutral_mixed_array_payloads(
        expected,
        actual,
        x_index=5,
        y_index=3,
        z_index=5,
    )

    assert parity.series_errors["center_density"].max_abs_error <= 9.5e-2
    assert parity.series_errors["center_pressure"].max_abs_error <= 1.0e-2
    assert parity.series_errors["center_momentum"].max_abs_error <= 3.0e-3
    assert parity.series_errors["center_temperature"].max_abs_error <= 2.0e-4
    assert parity.series_errors["momentum_rms"].max_abs_error <= 3.0e-3


def test_neutral_mixed_short_window_native_parity_stays_within_operational_center_band() -> None:
    if os.environ.get("JAX_DRB_RUN_NEUTRAL_MIXED_SHORT_WINDOW") != "1":
        pytest.skip("set JAX_DRB_RUN_NEUTRAL_MIXED_SHORT_WINDOW=1 to run the bounded neutral short-window centerline gate")
    if not _REFERENCE_INPUT.exists():
        pytest.skip("local neutral_mixed reference input is unavailable")

    expected = _reference_payload()
    result = run_input_case(
        _REFERENCE_INPUT,
        case_name="neutral_mixed_short_window",
        parity_mode="short_window",
        compare_variables=("Nh", "Ph", "NVh"),
    )
    actual = {
        **expected,
        "time_points": list(result.payload["time_points"]),
        "variables": {name: np.asarray(value, dtype=np.float64) for name, value in result.variables.items()},
    }

    parity = compare_neutral_mixed_array_payloads(
        expected,
        actual,
        x_index=5,
        y_index=3,
        z_index=5,
    )

    assert parity.series_errors["center_density"].max_abs_error <= 9.5e-3
    assert parity.series_errors["center_pressure"].max_abs_error <= 7.5e-4
    assert parity.series_errors["center_momentum"].max_abs_error <= 3.0e-3
    assert parity.series_errors["center_temperature"].max_abs_error <= 3.5e-4
    assert parity.series_errors["total_density"].max_abs_error <= 3.5e-1
    assert parity.series_errors["total_pressure"].max_abs_error <= 3.0e-2
    assert parity.series_errors["momentum_rms"].max_abs_error <= 3.0e-3


def test_neutral_mixed_short_window_native_parity_stays_within_operational_full_array_band() -> None:
    if os.environ.get("JAX_DRB_RUN_NEUTRAL_MIXED_SHORT_WINDOW") != "1":
        pytest.skip("set JAX_DRB_RUN_NEUTRAL_MIXED_SHORT_WINDOW=1 to run the bounded neutral short-window full-array gate")
    if not _REFERENCE_INPUT.exists():
        pytest.skip("local neutral_mixed reference input is unavailable")

    expected = _reference_payload()
    result = run_input_case(
        _REFERENCE_INPUT,
        case_name="neutral_mixed_short_window",
        parity_mode="short_window",
        compare_variables=("Nh", "Ph", "NVh"),
    )
    actual_variables = {name: np.asarray(value, dtype=np.float64) for name, value in result.variables.items()}

    field_thresholds = {
        "Nh": {"max_abs": 1.5e-2, "rms": 3.0e-3},
        "Ph": {"max_abs": 1.5e-3, "rms": 3.0e-4},
        "NVh": {"max_abs": 4.0e-3, "rms": 6.0e-4},
    }
    for name, thresholds in field_thresholds.items():
        expected_values = np.asarray(expected["variables"][name], dtype=np.float64)
        diff = actual_variables[name] - expected_values
        max_abs = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(np.square(diff))))
        assert max_abs <= thresholds["max_abs"], (name, max_abs, thresholds)
        assert rms <= thresholds["rms"], (name, rms, thresholds)
