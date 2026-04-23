from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation.open_field_operator_campaign import (
    build_open_field_operator_campaign_report,
    create_open_field_operator_campaign_package,
)


def test_open_field_operator_campaign_verifies_convergence_and_identities() -> None:
    report = build_open_field_operator_campaign_report(resolutions=(32, 64, 128), length=2.0 * np.pi)

    runs = report["convergence_runs"]
    assert runs[1]["gradient_l2"] < runs[0]["gradient_l2"]
    assert runs[2]["gradient_l2"] < runs[1]["gradient_l2"]
    assert runs[1]["force_balance_l2"] < runs[0]["force_balance_l2"]
    assert runs[2]["force_balance_l2"] < runs[1]["force_balance_l2"]

    assert report["min_observed_order"]["gradient"] > 1.8
    assert report["min_observed_order"]["force_balance"] > 1.8

    recycling = report["target_recycling_identity"]
    assert recycling["max_density_source_abs_error"] < 1.0e-12
    assert recycling["max_energy_source_abs_error"] < 1.0e-12
    assert recycling["max_lower_density_source_abs_error"] < 1.0e-12
    assert recycling["max_lower_energy_source_abs_error"] < 1.0e-12
    assert recycling["mean_active_energy_to_density_ratio"] == 3.5

    autodiff = report["autodiff_check"]
    assert autodiff["absolute_error"] < 1.0e-8
    assert autodiff["relative_error"] < 1.0e-6


def test_open_field_operator_campaign_package_writes_publication_artifacts(tmp_path: Path) -> None:
    artifacts = create_open_field_operator_campaign_package(
        output_root=tmp_path / "output",
        resolutions=(32, 64),
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()

    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "open_field_operator_campaign"
    assert payload["case"] == "open_field_operator_campaign"
    assert payload["equations"]["parallel_gradient"] == "D_y f_j = (f_{j+1} - f_{j-1}) / (2 dy)"
    assert payload["literature_anchors"][0]["url"] == "https://arxiv.org/abs/1602.06747"

    arrays = np.load(artifacts.arrays_npz_path)
    assert np.array_equal(arrays["resolutions"], np.asarray([32, 64]))
    assert arrays["gradient_l2"].shape == (2,)
    assert arrays["force_balance_l2"].shape == (2,)
    assert arrays["gradient_order"].shape == (1,)
    assert arrays["force_balance_order"].shape == (1,)
    assert arrays["recycling_velocity"].shape == (13,)
    assert arrays["recycling_lower_density_source"].shape == (13,)
