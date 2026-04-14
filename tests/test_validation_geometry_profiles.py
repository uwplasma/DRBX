from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation import (
    build_diagnostic_profile_report,
    save_diagnostic_profile_summary_plot,
    write_diagnostic_profile_arrays_npz,
)


def _build_report() -> dict[str, object]:
    density = np.arange(2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2)
    potential = 0.5 * density
    return build_diagnostic_profile_report(
        diagnostic_positions={
            "OMP": (0, np.array([-1.0, 0.0, 1.0], dtype=np.float64)),
            "TARGET": (1, np.array([-2.0, -1.0, 0.0], dtype=np.float64)),
        },
        derived_histories={
            "density": ("1/m^3", density),
            "potential": ("V", potential),
        },
        time_points=np.array([0.0, 1.0], dtype=np.float64),
        normalization={"status": "code_units"},
    )


def test_build_diagnostic_profile_report_summarizes_observables() -> None:
    report = _build_report()
    assert report["available"] is True
    assert report["parse_status"] == "ok"
    assert report["time_window"] == {"tmin": 0.0, "tmax": 1.0, "stored_states": 2}
    omp_density = report["diagnostics"]["OMP"]["density"]
    assert omp_density["units"] == "1/m^3"
    assert omp_density["positions"] == [-1.0, 0.0, 1.0]
    assert len(omp_density["mean"]) == 3
    assert omp_density["minimum"] == 0.0
    assert omp_density["maximum"] > omp_density["minimum"]


def test_write_diagnostic_profile_arrays_npz_saves_payload(tmp_path: Path) -> None:
    report = _build_report()
    path = write_diagnostic_profile_arrays_npz(report, tmp_path / "profiles.npz")
    data = np.load(path, allow_pickle=False)
    assert "OMP:density:mean" in data.files
    assert "TARGET:potential:std" in data.files
    metadata = json.loads(str(data["__metadata__"]))
    assert metadata["time_window"]["stored_states"] == 2


def test_save_diagnostic_profile_summary_plot_writes_png(tmp_path: Path) -> None:
    report = _build_report()
    path = save_diagnostic_profile_summary_plot(
        report,
        tmp_path / "profiles.png",
        diagnostic_order=("OMP", "TARGET"),
        observable_order=(("density", "Density"), ("potential", "Potential")),
        title="Generic 3D Diagnostics",
    )
    assert path.exists()
    assert path.stat().st_size > 0
