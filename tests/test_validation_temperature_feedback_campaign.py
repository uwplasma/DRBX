from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation import (
    create_temperature_feedback_campaign_package,
)
from jax_drb.validation.temperature_feedback_campaign import (
    _reconstruct_temperature_controller,
    _replace_bout_setting,
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
