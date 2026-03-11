from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_benchmark_bundle_respects_config_normalization_and_coeff_lengths(tmp_path: Path):
    coeff_path = tmp_path / "coeffs.npz"
    np.savez(coeff_path, Lx=np.asarray(3.0), Ly=np.asarray(2.5))

    cfg_path = tmp_path / "input.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[normalization]",
                "enabled = true",
                'mode = "physics"',
                "n0 = 1.0e19",
                "Te0_eV = 50.0",
                "B0 = 1.0",
                "m_i_amu = 1.0",
                "Z_i = 1.0",
                "[geometry]",
                'coeff_path = "coeffs.npz"',
            ]
        ),
        encoding="utf-8",
    )

    nt, nz, nx, ny = 4, 3, 4, 5
    times = np.linspace(0.0, 0.1, nt)
    z = np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False)[:, None, None]
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)[None, :, None]
    y = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)[None, None, :]
    base = np.sin(x + y) + 0.2 * np.cos(z)
    snapshots_n = np.stack([1.0 + (0.1 + 0.01 * i) * base for i in range(nt)], axis=0)
    snapshots_Te = np.stack([0.5 + (0.05 + 0.01 * i) * base for i in range(nt)], axis=0)
    snapshots_omega = np.stack([(0.02 + 0.005 * i) * base for i in range(nt)], axis=0)
    snapshots_phi = np.stack([(0.03 + 0.007 * i) * base for i in range(nt)], axis=0)

    run_npz = tmp_path / "run.npz"
    np.savez(
        run_npz,
        times=times,
        t=times,
        snapshots_n=snapshots_n,
        snapshots_Te=snapshots_Te,
        snapshots_omega=snapshots_omega,
        snapshots_phi=snapshots_phi,
    )

    out_path = tmp_path / "bundle.npz"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "tools/build_benchmark_bundle.py",
            "--code",
            "jax",
            "--input",
            str(run_npz),
            "--output",
            str(out_path),
            "--config",
            str(cfg_path),
            "--geometry",
            "tokamak_open_field",
        ],
        cwd=repo_root,
        check=True,
    )

    data = np.load(out_path, allow_pickle=True)
    meta = json.loads(str(data["meta_normalization_json"]))
    assert meta["m_i_amu"] == 1.0

    omega_ci = meta["omega_ci_s"]
    np.testing.assert_allclose(np.asarray(data["times_si"]), times / omega_ci)

    dy = 2.5 / ny
    expected_ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)
    np.testing.assert_allclose(np.asarray(data["diag__ky_m-1"]), expected_ky)
