from __future__ import annotations

import json
from pathlib import Path

import pytest

from jax_drb.validation import create_essos_biot_savart_campaign_package, resolve_essos_landreman_qa_json


def _has_landreman_qa_coil_json() -> bool:
    try:
        resolve_essos_landreman_qa_json()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _has_landreman_qa_coil_json(), reason="ESSOS Landreman-Paul QA coil JSON is not available")
def test_essos_biot_savart_campaign_generates_closed_open_artifacts(tmp_path: Path) -> None:
    artifacts = create_essos_biot_savart_campaign_package(output_root=tmp_path / "essos_biot_savart", nx=10, ny=12, nz=18)

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    closed = report["regions"]["closed_like_inner_annulus"]
    open_region = report["regions"]["open_sol_like_outer_annulus"]

    assert report["passed"] is True
    assert report["coil_json_file"] == "ESSOS_biot_savart_LandremanPaulQA.json"
    assert closed["boundary_fraction"] < open_region["boundary_fraction"]
    assert closed["final_rms_fluctuation"] > 1.0e-3
    assert open_region["final_rms_fluctuation"] > 1.0e-3
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    assert artifacts.movie_gif_path.exists()
