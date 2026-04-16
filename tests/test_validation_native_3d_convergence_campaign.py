from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_native_3d_convergence_campaign_package


def test_create_native_3d_convergence_campaign_package_writes_summary_and_plot(tmp_path: Path) -> None:
    artifacts = create_native_3d_convergence_campaign_package(output_root=tmp_path / "output")
    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["case"] == "native_3d_convergence_campaign"
    assert len(payload["entries"]) == 4
    assert payload["min_observed_order"] > 0.9
