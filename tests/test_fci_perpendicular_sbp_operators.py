"""Algebra-only tests of the host 1-D SBP operators (radial closure, Fourier rings, transfer pair)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from drbx.geometry.sbp_operators import (
    CLOSURE_ROWS,
    INTERIOR,
    T_LEFT,
    TWO_PI,
    W_CLOSURE,
    deta_matrix,
    fourier_pair,
    radial_block,
    ring_basis,
    ring_dtheta,
)

F = Fraction
DELTA = np.pi / 32


def test_closure_constants_exact():
    assert W_CLOSURE == (F(433, 384), F(95, 128), F(451, 384), F(367, 384))
    assert T_LEFT == (F(15, 8), F(-5, 4), F(3, 8), F(0))
    assert INTERIOR == (F(1, 12), F(-2, 3), F(0), F(2, 3), F(-1, 12))
    assert CLOSURE_ROWS == (
        (F(-675, 433), F(1885, 866), F(-293, 433), F(51, 866), F(0), F(0)),
        (F(-17, 114), F(-20, 19), F(59, 38), F(-20, 57), F(0), F(0)),
        (F(23, 451), F(-525, 902), F(-27, 451), F(597, 902), F(-32, 451), F(0)),
        (F(-51, 734), F(100, 367), F(-597, 734), F(0), F(256, 367), F(-32, 367)),
    )
    for row in CLOSURE_ROWS:
        assert sum(row) == 0
    assert sum(INTERIOR) == 0
    assert sum(T_LEFT) == 1


@pytest.mark.parametrize("m", [8, 9, 16, 27, 64])
def test_radial_block_sbp_identity(m):
    D, w, tL, tR = radial_block(m)
    W = np.diag(w)
    resid = W @ D + (W @ D).T - (np.outer(tR, tR) - np.outer(tL, tL))
    assert np.abs(resid).max() <= 1e-14
    assert np.array_equal(tR, tL[::-1])
    assert np.array_equal(w, w[::-1])


def test_radial_block_rejects_small_m():
    with pytest.raises(ValueError):
        radial_block(7)


@pytest.mark.parametrize("m", [8, 16, 27])
def test_radial_block_exactness(m):
    D, w, tL, tR = radial_block(m)
    x = (np.arange(m) + 0.5) / m
    interior = np.arange(4, m - 4)
    for q in range(5):
        dx = D @ x**q * m
        exact = q * x ** (q - 1) if q else np.zeros(m)
        if q <= 2:
            assert np.abs(dx - exact).max() <= 1e-12
        if interior.size:
            assert np.abs((dx - exact)[interior]).max() <= 1e-12
    for q in range(4):
        assert abs(w @ x**q / m - 1.0 / (q + 1)) <= 1e-13
    xu = np.arange(m) + 0.5
    for q in range(3):
        target = 1.0 if q == 0 else 0.0
        assert abs(tL @ xu**q - target) <= 1e-13
        assert abs(tR @ (xu - m) ** q - target) <= 1e-13


@pytest.mark.parametrize("N", [4, 8, 16, 32, 48, 64])
def test_ring_fourier_algebra(N):
    B, Binv, keys, om = ring_basis(N, DELTA)
    th = DELTA + TWO_PI * np.arange(N) / N
    assert np.abs(Binv @ B - np.eye(N)).max() <= 1e-13
    rng = np.random.default_rng(N)
    a, b = rng.standard_normal(N), rng.standard_normal(N)
    assert abs(TWO_PI / N * (B @ a) @ (B @ b) - np.sum(om * a * b)) <= 1e-13 * N
    Dth = ring_dtheta(N, DELTA)
    assert np.abs(Dth + Dth.T).max() <= 1e-13
    for m in range(1, N // 2):
        assert np.abs(Dth @ np.cos(m * th) + m * np.sin(m * th)).max() <= 1e-12
        assert np.abs(Dth @ np.sin(m * th) - m * np.cos(m * th)).max() <= 1e-12
    assert np.abs(Dth @ (-1.0) ** np.arange(N)).max() <= 1e-14 * N
    assert np.abs(Dth @ np.ones(N)).max() <= 1e-14 * N


@pytest.mark.parametrize("Na,Nb", [(8, 16), (16, 32), (16, 48), (32, 16), (64, 16), (16, 16)])
def test_fourier_transfer_pair(Na, Nb):
    Iab, Iba = fourier_pair(Na, Nb, DELTA)
    assert Iab.shape == (Nb, Na) and Iba.shape == (Na, Nb)
    assert np.abs((TWO_PI / Na) * Iba - Iab.T * (TWO_PI / Nb)).max() <= 1e-15
    tha = DELTA + TWO_PI * np.arange(Na) / Na
    thb = DELTA + TWO_PI * np.arange(Nb) / Nb
    for m in range(0, min(Na, Nb) // 2):
        assert np.abs(Iab @ np.cos(m * tha) - np.cos(m * thb)).max() <= 1e-13
        assert np.abs(Iab @ np.sin(m * tha) - np.sin(m * thb)).max() <= 1e-13
    # round trip on the coarser side, with its Nyquist mode removed
    Nc = min(Na, Nb)
    Bc, Binv_c, keys, _ = ring_basis(Nc, DELTA)
    keep = np.array([k[1] != "n" for k in keys])
    proj = Bc[:, keep] @ Binv_c[keep]
    rt = Iba @ Iab if Na <= Nb else Iab @ Iba
    assert np.abs(rt @ proj - proj).max() <= 1e-13
    if Na == Nb:
        assert np.array_equal(Iab, np.eye(Na))


def test_deta_matrix_periodic_fourth_order():
    n_eta = 16
    deta = TWO_PI / n_eta
    D = deta_matrix(n_eta, deta)
    assert np.abs(D + D.T).max() <= 1e-14
    eta = (np.arange(n_eta) + 0.5) * deta
    assert np.abs(D @ np.ones(n_eta)).max() <= 1e-13
    assert np.abs(D @ np.sin(eta) - np.cos(eta)).max() <= 2e-3
