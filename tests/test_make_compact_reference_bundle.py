from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from jaxdrb.benchmarking import (
    BenchmarkBundle,
    BenchmarkNormalization,
    load_bundle_npz,
    save_bundle_npz,
)


def _full_bundle() -> BenchmarkBundle:
    t = np.linspace(0.0, 0.1, 11)
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 8)
    z = np.linspace(0.0, 1.0, 4)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    n = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy) * (1.0 + zz)
    phi = np.cos(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy) * (1.0 + zz)
    return BenchmarkBundle(
        code="hermes",
        geometry="tokamak_open_field",
        normalization=BenchmarkNormalization(1e19, 50.0, 1.0),
        times_norm=t,
        times_si=t * 1.0e-7,
        diagnostics={
            "rms_n_fluct": 0.01 + 0.1 * t,
            "rms_phi_fluct": 0.02 + 0.2 * t,
            "freq_hz": np.linspace(1.0, 10.0, 6),
            "psd_n_f": np.linspace(1.0, 0.1, 6),
        },
        snapshots={
            "n_fluct_last": n,
            "phi_fluct_last": phi,
        },
        metadata={},
    )


def test_make_compact_reference_bundle_runs(tmp_path):
    full_path = tmp_path / "full.npz"
    out_path = tmp_path / "compact.npz"
    save_bundle_npz(_full_bundle(), full_path)

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "tools/make_compact_reference_bundle.py",
            "--input",
            str(full_path),
            "--output",
            str(out_path),
        ],
        cwd=repo_root,
        check=True,
    )

    compact = load_bundle_npz(out_path)
    assert compact.metadata["reference_kind"] == "compact"
    assert set(compact.snapshots) == {
        "n_fluct_last_xz",
        "n_fluct_last_xy",
        "phi_fluct_last_xz",
        "phi_fluct_last_xy",
    }
    assert compact.snapshots["n_fluct_last_xz"].ndim == 2
    assert compact.snapshots["n_fluct_last_xy"].ndim == 2
