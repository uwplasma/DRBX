"""Host-side 1-D summation-by-parts operators for the nodal perpendicular scheme.

The radial block is a cell-centred, diagonal-norm SBP operator with a fourth-order
interior stencil and a ``b = 4`` closure (extrapolation support ``s = 3``); its
constants are stored as exact :class:`fractions.Fraction` values and converted to
floats on use. Angular rings carry a Fourier basis: the derivative is a dense,
exactly antisymmetric ``N x N`` matrix in the ring quadrature, and two rings of
different resolution are coupled by a zero-padding/truncation transfer pair that is
the exact transpose in the ring weights.

Everything here is host NumPy, built once per layout (see the JAX boundary in
``docs/code_structure.md``). Applying the operators per right-hand-side evaluation
happens in the JAX modules that consume :mod:`drbx.geometry.nodal_layout`.
"""
from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np

TWO_PI = 2.0 * np.pi

_F = Fraction

# Diagonal norm of the closure rows (interior weight 1).
W_CLOSURE: tuple[Fraction, ...] = (_F(433, 384), _F(95, 128), _F(451, 384), _F(367, 384))
# Boundary extrapolation t_L (quadratic, 3-point); t_R is the mirror.
T_LEFT: tuple[Fraction, ...] = (_F(15, 8), _F(-5, 4), _F(3, 8), _F(0))
# Interior stencil at offsets -2..2 (times 1/du).
INTERIOR: tuple[Fraction, ...] = (_F(1, 12), _F(-8, 12), _F(0), _F(8, 12), _F(-1, 12))
# Closure rows (times 1/du), columns 0..5.
CLOSURE_ROWS: tuple[tuple[Fraction, ...], ...] = (
    (_F(-675, 433), _F(1885, 866), _F(-293, 433), _F(51, 866), _F(0), _F(0)),
    (_F(-17, 114), _F(-20, 19), _F(59, 38), _F(-20, 57), _F(0), _F(0)),
    (_F(23, 451), _F(-525, 902), _F(-27, 451), _F(597, 902), _F(-32, 451), _F(0)),
    (_F(-51, 734), _F(100, 367), _F(-597, 734), _F(0), _F(256, 367), _F(-32, 367)),
)
CLOSURE_B = 4
MIN_RADIAL_POINTS = 2 * CLOSURE_B


class RadialBlockOps(NamedTuple):
    """Unit-spacing radial SBP block: ``D_unit``, norm ``w`` and boundary extrapolations ``tL``, ``tR``."""

    D_unit: np.ndarray
    w: np.ndarray
    tL: np.ndarray
    tR: np.ndarray


def _float_array(values) -> np.ndarray:
    return np.array([float(v) for v in values])


def radial_block(m: int) -> RadialBlockOps:
    """SBP radial block on ``m >= 8`` cell-centred points at unit spacing.

    The physical derivative is ``D_unit / du`` and the level quadrature weights are ``w * du``.
    Satisfies ``diag(w) D + D^T diag(w) = tR tR^T - tL tL^T``.
    """
    m = int(m)
    if m < MIN_RADIAL_POINTS:
        raise ValueError(f"radial block needs m >= {MIN_RADIAL_POINTS} points, got {m}")
    D = np.zeros((m, m))
    interior = _float_array(INTERIOR)
    for i in range(CLOSURE_B, m - CLOSURE_B):
        D[i, i - 2:i + 3] = interior
    cols = np.arange(CLOSURE_B + 2)
    for i, row in enumerate(CLOSURE_ROWS):
        r = _float_array(row)
        D[i, :CLOSURE_B + 2] = r
        D[m - 1 - i, m - 1 - cols] = -r
    w = np.ones(m)
    w[:CLOSURE_B] = _float_array(W_CLOSURE)
    w[m - CLOSURE_B:] = w[:CLOSURE_B][::-1]
    tL = np.zeros(m)
    tL[:CLOSURE_B] = _float_array(T_LEFT)
    return RadialBlockOps(D, w, tL, tL[::-1].copy())


def ring_basis(N: int, delta: float) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], np.ndarray]:
    """Fourier basis on ``N`` nodes ``theta_j = delta + 2 pi j / N``.

    Returns ``(B, Binv, keys, om)``: nodal values are ``B @ amplitudes``, ``keys`` label the columns
    ``(m, 'c' | 's' | 'n')`` (cosine, sine, Nyquist) and ``om`` are the Parseval weights
    (``2 pi`` for ``m = 0`` and the Nyquist mode, ``pi`` otherwise), so ``(2 pi / N) sum f g = sum om a b``.
    """
    N = int(N)
    j = np.arange(N)
    th = delta + TWO_PI * j / N
    M = (N + 1) // 2 - 1
    cols, keys, om = [np.ones(N)], [(0, "c")], [TWO_PI]
    for m in range(1, M + 1):
        cols += [np.cos(m * th), np.sin(m * th)]
        keys += [(m, "c"), (m, "s")]
        om += [np.pi, np.pi]
    if N % 2 == 0 and N >= 2:
        cols.append((-1.0) ** j)
        keys.append((N // 2, "n"))
        om.append(TWO_PI)
    B = np.column_stack(cols)
    om = np.array(om)
    Binv = (B.T * (TWO_PI / N)) / om[:, None]
    return B, Binv, keys, om


def ring_dtheta(N: int, delta: float) -> np.ndarray:
    """Dense Fourier derivative on a ring: exactly antisymmetric, zero on the Nyquist mode."""
    B, Binv, keys, _ = ring_basis(N, delta)
    lam = np.zeros((N, N))
    pos = {k: i for i, k in enumerate(keys)}
    for (m, typ), i in pos.items():
        if typ == "c" and m >= 1:
            js = pos[(m, "s")]
            lam[i, js] = m
            lam[js, i] = -m
    return B @ lam @ Binv


def fourier_pair(Na: int, Nb: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Transfer pair between rings of ``Na`` and ``Nb`` nodes.

    ``Iab`` (``Nb x Na``) zero-pads or truncates the physical amplitudes, dropping the Nyquist mode of the
    smaller grid; ``Iba = (Na / 2 pi) Iab^T (2 pi / Nb)`` so that ``Omega_a Iba = Iab^T Omega_b`` with
    ``Omega = 2 pi / N``. Both are identities when ``Na == Nb``.
    """
    Na, Nb = int(Na), int(Nb)
    if Na == Nb:
        return np.eye(Na), np.eye(Na)
    Ba, Binv_a, ka, _ = ring_basis(Na, delta)
    Bb, _, kb, _ = ring_basis(Nb, delta)
    pb = {k: i for i, k in enumerate(kb)}
    E = np.zeros((Nb, Na))
    for a, k in enumerate(ka):
        if k[1] != "n" and k in pb:
            E[pb[k], a] = 1.0
    Iab = Bb @ E @ Binv_a
    Iba = (Na / TWO_PI) * Iab.T * (TWO_PI / Nb)
    return Iab, Iba


def deta_matrix(n_eta: int, deta: float) -> np.ndarray:
    """Periodic fourth-order central derivative along eta (reference and audit only)."""
    D = np.zeros((n_eta, n_eta))
    interior = _float_array(INTERIOR)
    for k in range(n_eta):
        for o, c in zip(range(-2, 3), interior):
            D[k, (k + o) % n_eta] += c / deta
    return D
