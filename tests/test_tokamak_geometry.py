from __future__ import annotations

import numpy as np

from jaxdrb.geometry.tokamak import CircularTokamakGeometry, SAlphaGeometry


def test_circular_tokamak_shapes_and_positive_kperp2() -> None:
    geom = CircularTokamakGeometry.make(nl=32, shat=0.8, q=3.0, R0=1.0, epsilon=0.3)
    k2 = geom.kperp2(kx=0.0, ky=0.3)
    assert k2.shape == (32,)
    assert np.all(np.asarray(k2) > 0.0)


def test_salpha_alpha_modifies_kperp2_variation() -> None:
    geom0 = SAlphaGeometry.make(nl=64, shat=0.8, alpha=0.0, q=1.4, R0=1.0, epsilon=0.18)
    geom1 = SAlphaGeometry.make(nl=64, shat=0.8, alpha=1.0, q=1.4, R0=1.0, epsilon=0.18)

    k2_0 = np.asarray(geom0.kperp2(kx=0.0, ky=0.3))
    k2_1 = np.asarray(geom1.kperp2(kx=0.0, ky=0.3))

    # Alpha changes the ballooning structure through g_xy(theta).
    assert np.max(np.abs(k2_1 - k2_0)) > 1e-10
