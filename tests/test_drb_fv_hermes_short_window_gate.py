from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.benchmarking import compare_bundle_diagnostics, finite_run_gate, load_bundle_npz
from jaxdrb.driver import run_simulation

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hermes_short_window_compact.npz"
_CFG = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "open_field_line"
    / "input_tokamak_bxcv_benchmark_hermes_strict.toml"
)

_EXPECTED = {
    "rms_n_fluct": 0.9869660621864147,
    "rms_Te_fluct": 0.9468870958512322,
    "rms_omega_fluct": 0.9848929111974901,
    "rms_phi_fluct": 0.9752254649645201,
    "psd_n_f": 12.39744920874179,
    "psd_n_ky": 0.9999761717643812,
}
_EXPECTED_MEAN = 2.8818994857843045
_EXPECTED_MAX = 12.39744920874179
_RTOL = 1e-6
_ATOL = 1e-9


def _run_candidate_bundle(tmp_path: Path) -> Path:
    with _CFG.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["time"]["nsteps"] = 10
    cfg["time"]["save_every"] = 1
    cfg["time"]["return_numpy"] = True
    cfg["time"]["diag_mode"] = "full"
    cfg["time"]["save_fields"] = True
    cfg["time"]["snapshot_fields"] = ["n", "Te", "omega", "phi"]

    result = run_simulation(cfg, as_numpy=True)
    payload = dict(result.diagnostics)
    payload.setdefault("times", np.asarray(result.times, dtype=np.float64))
    payload.setdefault("t", np.asarray(result.times, dtype=np.float64))

    run_npz = tmp_path / "jax_short.npz"
    bundle_npz = tmp_path / "bundle_jax_short.npz"
    np.savez(run_npz, **payload)

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
            str(bundle_npz),
            "--config",
            str(_CFG),
            "--geometry",
            "tokamak_open_field",
        ],
        cwd=repo_root,
        check=True,
    )
    return bundle_npz


def test_drb_fv_hermes_short_window_regression_gate(tmp_path: Path) -> None:
    reference = load_bundle_npz(_FIXTURE)
    candidate = load_bundle_npz(_run_candidate_bundle(tmp_path))

    passed, reason, growth, peak = finite_run_gate(
        candidate.diagnostics,
        max_growth_factor=20.0,
        max_rms_abs=5.0,
    )
    assert passed, f"finite-run gate failed: {reason} growth={growth:.3e} peak={peak:.3e}"

    comparison = compare_bundle_diagnostics(reference, candidate)
    np.testing.assert_allclose(comparison.mean_rel_l2, _EXPECTED_MEAN, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(comparison.max_rel_l2, _EXPECTED_MAX, rtol=_RTOL, atol=_ATOL)
    assert set(comparison.per_key_rel_l2) == set(_EXPECTED)
    for key, expected in _EXPECTED.items():
        np.testing.assert_allclose(
            comparison.per_key_rel_l2[key],
            expected,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"Hermes-coupled short-window mismatch drifted for {key}",
        )
