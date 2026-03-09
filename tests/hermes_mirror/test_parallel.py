from __future__ import annotations

import numpy as np

from jaxdrb.hermes_mirror.parallel import div_par_centered, div_par_mod


def test_div_par_mod_uses_ghost_midpoint_boundary_flux() -> None:
    f = np.zeros((3, 1, 1), dtype=np.float64)
    v = np.zeros_like(f)
    wave = np.ones_like(f)

    div = np.asarray(
        div_par_mod(
            f,
            v,
            wave,
            dz=2.0,
            limiter="mc",
            ghost_low_f=np.array([[2.0]], dtype=np.float64),
            ghost_high_f=np.array([[4.0]], dtype=np.float64),
            ghost_low_v=np.array([[6.0]], dtype=np.float64),
            ghost_high_v=np.array([[8.0]], dtype=np.float64),
        )
    )

    expected = np.zeros_like(div)
    expected[0, 0, 0] = -(0.5 * 2.0) * (0.5 * 6.0) / 2.0
    expected[-1, 0, 0] = (0.5 * 4.0) * (0.5 * 8.0) / 2.0
    np.testing.assert_allclose(div, expected, atol=1e-12, rtol=1e-12)


def test_div_par_mod_uses_boundary_cell_metric_factor() -> None:
    div = np.asarray(
        div_par_mod(
            np.zeros((3, 1, 1), dtype=np.float64),
            np.zeros((3, 1, 1), dtype=np.float64),
            np.ones((3, 1, 1), dtype=np.float64),
            dz=2.0,
            limiter="mc",
            J=np.ones((3, 1, 1), dtype=np.float64),
            gpar=np.array([[[4.0]], [[9.0]], [[16.0]]], dtype=np.float64),
            ghost_low_f=np.array([[2.0]], dtype=np.float64),
            ghost_low_v=np.array([[2.0]], dtype=np.float64),
        )
    )

    expected = np.zeros((3, 1, 1), dtype=np.float64)
    flux = (0.5 * 2.0) * (0.5 * 2.0)
    expected[0, 0, 0] = -flux / (2.0 * np.sqrt(4.0))
    np.testing.assert_allclose(div, expected, atol=1e-12, rtol=1e-12)


def test_div_par_centered_uses_ghost_face_current() -> None:
    f = np.zeros((4, 1, 1), dtype=np.float64)
    div = np.asarray(
        div_par_centered(
            f,
            dz=2.0,
            ghost_low=np.array([[2.0]], dtype=np.float64),
            ghost_high=np.array([[6.0]], dtype=np.float64),
        )
    )

    expected = np.zeros_like(div)
    expected[0, 0, 0] = -(0.5 * 2.0) / 2.0
    expected[-1, 0, 0] = (0.5 * 6.0) / 2.0
    np.testing.assert_allclose(div, expected, atol=1e-12, rtol=1e-12)
