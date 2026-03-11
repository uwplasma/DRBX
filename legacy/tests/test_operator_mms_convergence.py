from __future__ import annotations

import numpy as np

from jaxdrb.bc import BC2D
from jaxdrb.operators.fd2d import laplacian


def _lap_error(nx: int, ny: int) -> float:
    lx = 2.0 * np.pi
    ly = 2.0 * np.pi
    dx = lx / nx
    dy = ly / ny

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")

    kx = 2.0
    ky = 3.0
    f = np.sin(kx * xx) * np.cos(ky * yy)
    lap_exact = -(kx**2 + ky**2) * f

    lap_num = np.asarray(laplacian(f, dx, dy, BC2D.periodic()))
    err = np.sqrt(np.mean((lap_num - lap_exact) ** 2))
    return float(err)


def test_fd_laplacian_mms_second_order_convergence():
    e16 = _lap_error(16, 16)
    e32 = _lap_error(32, 32)
    e64 = _lap_error(64, 64)

    # 2nd-order expected: error should drop by ~4x each refinement.
    assert e32 < e16
    assert e64 < e32
    assert (e16 / e32) > 3.0
    assert (e32 / e64) > 3.0
