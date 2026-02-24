from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from jaxdrb.benchmarking import BenchmarkBundle, BenchmarkNormalization, save_bundle_npz


def _bundle(code: str) -> BenchmarkBundle:
    t = np.linspace(0.0, 1.0, 16)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    base = np.sin(xx) * np.cos(yy)
    scale = 1.0 if code == "hermes" else 1.05

    return BenchmarkBundle(
        code=code,
        geometry="tokamak_open_field",
        normalization=BenchmarkNormalization(1e19, 50.0, 1.0),
        times_norm=t,
        times_si=t * 1e-7,
        diagnostics={
            "rms_n_fluct": 0.1 + 0.01 * t,
            "rms_Te_fluct": 0.08 + 0.02 * t,
            "rms_omega_fluct": 0.05 + 0.015 * t,
            "rms_phi_fluct": 0.04 + 0.01 * t,
            "ky_m-1": np.linspace(0.0, 20.0, 13),
            "psd_n_ky": np.linspace(1.0, 0.1, 13),
            "freq_hz": np.linspace(0.0, 200.0, 33),
            "psd_n_f": np.linspace(1.0, 0.05, 33),
            "pdf_n_x": np.linspace(-1.0, 1.0, 41),
            "pdf_n_y": np.exp(-np.linspace(-1.0, 1.0, 41) ** 2),
            "coh_freq_hz": np.linspace(0.0, 200.0, 33),
            "coh_n_phi": np.linspace(0.2, 0.9, 33),
            "phase_n_phi": np.linspace(-0.5, 0.5, 33),
            "gamma_r_profile": np.linspace(-0.1, 0.1, 32),
        },
        snapshots={
            "n_fluct_last": scale * base,
            "n_last": 1.0 + scale * base,
        },
        metadata={},
    )


def test_plot_benchmark_panel_runs(tmp_path):
    hermes_path = tmp_path / "bundle_hermes.npz"
    jax_path = tmp_path / "bundle_jax.npz"
    out_path = tmp_path / "panel.png"

    save_bundle_npz(_bundle("hermes"), hermes_path)
    save_bundle_npz(_bundle("jax_drb"), jax_path)

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "tools/plot_benchmark_panel.py",
        "--hermes",
        str(hermes_path),
        "--jax",
        str(jax_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
