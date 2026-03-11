from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import (
    compute_cross_coherence_phase,
    compute_fluctuation_rms,
    compute_frequency_psd,
    compute_ky_psd,
    compute_pdf,
    compute_radial_particle_flux_profile,
    compute_target_fluxes,
    finite_run_gate,
)


def test_fluctuation_rms_and_gates():
    t = np.linspace(0.0, 1.0, 32)
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    a = np.stack([1.0 + 0.05 * np.sin(xx + yy + 0.2 * ti) for ti in t], axis=0)
    rms, rms_fluct, eq = compute_fluctuation_rms(a, equilibrium_mode="t0")

    assert rms.shape == (t.size,)
    assert rms_fluct.shape == (t.size,)
    assert np.allclose(eq, a[0])
    assert np.all(rms > 0.0)
    assert np.all(rms_fluct >= 0.0)

    passed, reason, growth, peak = finite_run_gate(
        {
            "rms_n_fluct": rms_fluct,
            "rms_Te_fluct": 0.6 * rms_fluct,
            "rms_omega_fluct": 0.5 * rms_fluct,
            "rms_phi_fluct": 0.3 * rms_fluct,
        },
        max_growth_factor=50.0,
        max_rms_abs=10.0,
    )
    assert passed, reason
    assert growth > 0.0
    assert peak > 0.0

    fail, reason, _, _ = finite_run_gate(
        {
            "rms_n_fluct": np.array([1.0, np.inf]),
            "rms_Te_fluct": np.array([1.0, 1.0]),
            "rms_omega_fluct": np.array([1.0, 1.0]),
            "rms_phi_fluct": np.array([1.0, 1.0]),
        },
        max_growth_factor=10.0,
        max_rms_abs=10.0,
    )
    assert not fail
    assert reason.startswith("nonfinite")


def test_psd_pdf_coherence_and_fluxes():
    dt = 1.0e-3
    t = np.arange(0.0, 0.5, dt)
    f0 = 25.0
    x = np.sin(2.0 * np.pi * f0 * t)
    y = np.sin(2.0 * np.pi * f0 * t + 0.2)

    f, p = compute_frequency_psd(x, dt=dt, nperseg=128)
    assert f.ndim == 1 and p.ndim == 1
    assert f.size == p.size
    assert float(np.max(p)) > 0.0

    ff, coh, phase = compute_cross_coherence_phase(x, y, dt=dt, nperseg=128)
    assert ff.size == coh.size == phase.size
    assert np.all(coh >= 0.0)
    assert np.all(coh <= 1.0 + 1e-8)

    nx, ny = 20, 18
    xx, yy = np.meshgrid(
        np.linspace(0.0, 1.0, nx, endpoint=False),
        np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False),
        indexing="ij",
    )
    n2d = 1.0 + 0.2 * np.sin(2.0 * np.pi * xx)
    phi2d = 0.1 * np.cos(yy)

    ky, pky = compute_ky_psd(n2d, dy=(2.0 * np.pi / ny), axis_y=-1)
    assert ky.size == pky.size
    assert np.all(pky >= 0.0)

    centers, hist = compute_pdf(n2d, bins=40)
    assert centers.size == hist.size == 40
    assert np.all(hist >= 0.0)

    gamma_r = compute_radial_particle_flux_profile(n2d, phi2d, dy=(2.0 * np.pi / ny), B0=1.0)
    assert gamma_r.ndim == 1
    assert gamma_r.size == nx

    nt, nz = 32, 12
    n3 = np.ones((nt, nz, nx, ny)) * 0.8
    vi3 = np.ones((nt, nz, nx, ny)) * 0.5
    te3 = np.ones((nt, nz, nx, ny)) * 0.4
    gamma_t, qe_t, qi_t = compute_target_fluxes(n3, vi3, te3, axis_par=1)
    assert gamma_t.shape == (nt,)
    assert qe_t.shape == (nt,)
    assert qi_t.shape == (nt,)
    assert np.all(gamma_t > 0.0)
    assert np.all(qe_t > 0.0)
    assert np.all(qi_t >= 0.0)
