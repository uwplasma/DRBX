from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jax_drb.validation import analyze_blob2d_array_payload, create_blob2d_meeting_package, write_blob2d_analysis_json


def _synthetic_blob_payload() -> dict[str, object]:
    time_points = np.linspace(0.0, 0.5, 4)
    x = np.arange(8, dtype=np.float64)
    z = np.arange(8, dtype=np.float64)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    history = []
    for shift in (2.0, 2.5, 3.0, 3.5):
        radius = (xx - shift) ** 2 + (zz - 4.0) ** 2
        history.append(1.0 + 0.2 * np.exp(-0.25 * radius))
    density = np.asarray(history, dtype=np.float64)[:, :, None, :]
    return {
        "time_points": time_points.tolist(),
        "variables": {"Ne": density},
    }


def test_create_blob2d_meeting_package_writes_expected_artifacts(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is unavailable")

    payload = _synthetic_blob_payload()
    analysis = analyze_blob2d_array_payload(payload, density_variable="Ne", background_density=1.0)
    reference_metrics = tmp_path / "reference_metrics.json"
    write_blob2d_analysis_json(analysis, reference_metrics)

    native_arrays = tmp_path / "native_blob.npz"
    from jax_drb.parity.arrays import write_portable_array_payload

    write_portable_array_payload(
        {
            "case_name": "synthetic_blob",
            "parity_mode": "short_window",
            "producer": "test",
            "overrides": [],
            "compare_variables": ["Ne"],
            "component_labels": [],
            "dimensions": {"t": 4, "x": 8, "y": 1, "z": 8},
            "time_points": payload["time_points"],
            "dataset_scalars": {},
            "variable_dimensions": {"Ne": ["t", "x", "y", "z"]},
            "variables": {"Ne": payload["variables"]["Ne"]},
            "effective_output_points": 4,
        },
        native_arrays,
    )

    artifacts = create_blob2d_meeting_package(
        payload,
        output_root=tmp_path,
        native_arrays_path=native_arrays,
        reference_metrics_path=reference_metrics,
        density_variable="Ne",
        background_density=1.0,
        case_label="synthetic_blob_meeting",
        fps=4,
    )

    for path in (
        artifacts.analysis_json_path,
        artifacts.parity_json_path,
        artifacts.snapshots_png_path,
        artifacts.parity_png_path,
        artifacts.poster_png_path,
        artifacts.movie_2d_path,
        artifacts.movie_3d_path,
    ):
        assert path.exists()
        assert path.stat().st_size > 0
