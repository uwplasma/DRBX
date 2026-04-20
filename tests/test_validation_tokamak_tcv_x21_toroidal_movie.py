from __future__ import annotations

from pathlib import Path

import numpy as np

from jax_drb.validation import create_tcv_x21_toroidal_movie_package


def test_create_tcv_x21_toroidal_movie_package_writes_artifacts(tmp_path: Path) -> None:
    arrays_path = tmp_path / "input.npz"
    time_points = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    x = np.linspace(0.0, 1.0, 12, dtype=np.float64)
    y = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    history = np.stack(
        [
            np.sin(yy) * np.exp(-((xx - 0.45) ** 2) / 0.04),
            np.sin(yy + 0.4) * np.exp(-((xx - 0.48) ** 2) / 0.05),
            np.sin(yy + 0.8) * np.exp(-((xx - 0.52) ** 2) / 0.06),
        ],
        axis=0,
    )
    np.savez_compressed(arrays_path, field_history=history, time_points=time_points, field_name=np.asarray("phi"))

    artifacts = create_tcv_x21_toroidal_movie_package(
        arrays_npz_path=arrays_path,
        output_root=tmp_path / "out",
        toroidal_samples=18,
        radial_stride=2,
        poloidal_stride=2,
        interpolation_substeps=2,
        fps=2,
    )

    assert artifacts.arrays_npz_path.exists()
    assert artifacts.summary_json_path.exists()
    assert artifacts.poster_png_path.exists()
    assert artifacts.movie_gif_path.exists()
    summary = artifacts.summary_json_path.read_text(encoding="utf-8")
    assert "toroidal_opening_degrees" in summary
