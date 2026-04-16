from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import ControllerFeedbackMetric, create_controller_feedback_campaign_package


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

