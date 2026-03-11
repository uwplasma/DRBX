from __future__ import annotations

from pathlib import Path

import numpy as np

from jaxdrb.benchmarking import compare_bundle_diagnostics, finite_run_gate, load_bundle_npz

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hermes_short_window_compact.npz"
_CANDIDATE_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "jax_short_window_literal_candidate.npz"
)

_EXPECTED = {
    "rms_n_fluct": 0.9946316166254552,
    "rms_Te_fluct": 0.9573479084759355,
    "rms_omega_fluct": 0.9916586635839145,
    "rms_phi_fluct": 0.9840357034421593,
    "psd_n_f": 2.066778997179102,
    "psd_n_ky": 0.9999943290166118,
}
_EXPECTED_MEAN = 1.1657412030538632
_EXPECTED_MAX = 2.066778997179102
_RTOL = 1e-6
_ATOL = 1e-9


def test_drb_fv_hermes_short_window_regression_gate() -> None:
    reference = load_bundle_npz(_FIXTURE)
    candidate = load_bundle_npz(_CANDIDATE_FIXTURE)

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
