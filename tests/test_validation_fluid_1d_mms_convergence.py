from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from drbx.validation.fluid_1d_mms_convergence import (
    build_fluid_1d_mms_convergence_report,
    create_fluid_1d_mms_convergence_package,
)


def test_build_fluid_1d_mms_convergence_report_shows_refinement_improvement() -> None:
    report = build_fluid_1d_mms_convergence_report(
        resolutions=(32, 64, 128),
        timestep=0.05,
        steps=2,
        substeps=20,
    )

    runs = report["runs"]
    assert runs[1]["errors"]["density_l2"] < runs[0]["errors"]["density_l2"]
    assert runs[2]["errors"]["density_l2"] < runs[1]["errors"]["density_l2"]
    assert runs[1]["errors"]["pressure_l2"] < runs[0]["errors"]["pressure_l2"]
    assert runs[2]["errors"]["pressure_l2"] < runs[1]["errors"]["pressure_l2"]
    assert runs[1]["errors"]["momentum_l2"] < runs[0]["errors"]["momentum_l2"]
    assert runs[2]["errors"]["momentum_l2"] < runs[1]["errors"]["momentum_l2"]

    first_order = report["observed_orders"][0]
    assert first_order["density_order"] > 1.5
    assert first_order["pressure_order"] > 1.5
    assert first_order["momentum_order"] > 1.5
    assert report["min_observed_order"]["density"] > 1.5
    assert report["min_observed_order"]["pressure"] > 1.5
    assert report["min_observed_order"]["momentum"] > 1.5


def test_create_fluid_1d_mms_convergence_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_fluid_1d_mms_convergence_package(
        output_root=tmp_path / "output",
        resolutions=(32, 64),
        timestep=0.05,
        steps=2,
        substeps=20,
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.summary_plot_png_path.exists()

    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "manufactured_solution_convergence"
    assert payload["case"] == "fluid_1d_mms_convergence"
    assert payload["resolutions"] == [32, 64]

    arrays = np.load(artifacts.arrays_npz_path)
    assert np.array_equal(arrays["resolutions"], np.asarray([32, 64]))
    assert arrays["density_l2"].shape == (2,)
    assert arrays["pressure_l2"].shape == (2,)
    assert arrays["momentum_l2"].shape == (2,)
    assert arrays["density_order"].shape == (1,)
    assert arrays["pressure_order"].shape == (1,)
    assert arrays["momentum_order"].shape == (1,)
