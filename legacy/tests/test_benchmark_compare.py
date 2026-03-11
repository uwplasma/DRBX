from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import (
    BenchmarkBundle,
    BenchmarkNormalization,
    compare_bundle_diagnostics,
)


def _bundle(scale: float = 1.0, *, nt: int = 11) -> BenchmarkBundle:
    t = np.linspace(0.0, 0.1, nt)
    f = np.linspace(1.0e4, 6.0e4, 8)
    ky = np.linspace(0.0, 20.0, 13)
    return BenchmarkBundle(
        code="jax_drb",
        geometry="tokamak_open_field",
        normalization=BenchmarkNormalization(1e19, 50.0, 1.0),
        times_norm=t,
        times_si=t * 1.0e-7,
        diagnostics={
            "rms_n_fluct": scale * (0.01 + 0.2 * t),
            "rms_Te_fluct": scale * (0.02 + 0.15 * t),
            "rms_omega_fluct": scale * (0.03 + 0.10 * t),
            "rms_phi_fluct": scale * (0.04 + 0.05 * t),
            "freq_hz": f,
            "psd_n_f": scale * np.exp(-f / f.max()),
            "ky_m-1": ky,
            "psd_n_ky": scale * np.exp(-ky / max(ky.max(), 1.0)),
        },
        metadata={},
    )


def test_compare_bundle_diagnostics_exact_match():
    ref = _bundle()
    cmp = _bundle()
    out = compare_bundle_diagnostics(ref, cmp)
    assert out.mean_rel_l2 == 0.0
    assert out.max_rel_l2 == 0.0
    assert all(v == 0.0 for v in out.per_key_rel_l2.values())


def test_compare_bundle_diagnostics_interpolates_rms_time_axis():
    ref = _bundle(scale=1.0, nt=11)
    cmp = _bundle(scale=1.1, nt=21)
    out = compare_bundle_diagnostics(ref, cmp, keys=("rms_n_fluct", "rms_Te_fluct"))
    assert set(out.per_key_rel_l2) == {"rms_n_fluct", "rms_Te_fluct"}
    assert out.mean_rel_l2 > 0.0
    assert out.max_rel_l2 > 0.0
