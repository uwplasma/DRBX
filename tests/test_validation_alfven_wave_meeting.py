from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jax_drb.parity.arrays import write_portable_array_payload
from jax_drb.validation import create_alfven_wave_meeting_package


_ALFVEN_WAVE_INPUT = """
nout = 20
timestep = 10

[mesh]
nx = 5
ny = 32
nz = 27
Lx = 0.1
Ly = 10
Lz = 1
B = 0.2
dx = Lx / (nx - 4)
dy = Ly / ny
dz = Lz / nz
g11 = 1
g22 = 1
g33 = 1
J = 1

[mesh:paralleltransform]
type = identity

[model]
components = (e, i, electromagnetic, vorticity)
Nnorm = 1e19
Tnorm = 100
Bnorm = 0.2

[e]
AA = 1/1836

[i]
AA = 2
density = 1e19
"""


def test_create_alfven_wave_meeting_package_writes_expected_artifacts(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is unavailable")

    input_file = tmp_path / "BOUT.inp"
    input_file.write_text(_ALFVEN_WAVE_INPUT, encoding="utf-8")

    nt = 6
    nx = 5
    ny = 32
    nz = 27
    time_points = (np.arange(nt, dtype=np.float64) * 10.0).tolist()
    phase = np.linspace(0.0, 2.0 * np.pi, nt)[:, None, None, None]
    yy = np.linspace(0.0, 2.0 * np.pi, ny)[None, None, :, None]
    zz = np.linspace(0.0, 2.0 * np.pi, nz)[None, None, None, :]
    phi = 1.0e-4 * np.sin(phase + yy) * np.cos(zz) * np.ones((1, nx, 1, 1), dtype=np.float64)

    payload = {
        "case_name": "synthetic_alfven_meeting",
        "parity_mode": "short_window",
        "producer": "test",
        "overrides": [],
        "compare_variables": ["phi"],
        "component_labels": [],
        "dimensions": {"t": nt, "x": nx, "y": ny, "z": nz},
        "time_points": time_points,
        "dataset_scalars": {
            "Nnorm": 1.0e19,
            "Tnorm": 100.0,
            "Bnorm": 0.2,
            "Cs0": 1.0,
            "Omega_ci": 1.0e7,
            "rho_s0": 1.0e-3,
        },
        "variable_dimensions": {"phi": ["t", "x", "y", "z"]},
        "variables": {"phi": phi},
        "effective_output_points": nt,
    }

    expected_arrays = tmp_path / "expected.npz"
    actual_arrays = tmp_path / "actual.npz"
    write_portable_array_payload(payload, expected_arrays)
    write_portable_array_payload(payload, actual_arrays)

    artifacts = create_alfven_wave_meeting_package(
        payload,
        input_file=input_file,
        expected_arrays_path=expected_arrays,
        native_arrays_path=actual_arrays,
        output_root=tmp_path,
        field_variable="phi",
        x_index=2,
        case_label="synthetic_alfven_meeting",
        fps=4,
    )

    for path in (
        artifacts.analysis_json_path,
        artifacts.parity_json_path,
        artifacts.snapshots_png_path,
        artifacts.diagnostics_png_path,
        artifacts.parity_png_path,
        artifacts.poster_png_path,
        artifacts.movie_2d_path,
        artifacts.movie_3d_path,
    ):
        assert path.exists()
        assert path.stat().st_size > 0
