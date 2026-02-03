from __future__ import annotations

import numpy as np

from jaxdrb.linear.ideal_ballooning import ideal_ballooning_gamma_hat


def test_ideal_ballooning_alpha_zero_is_stable() -> None:
    g = float(ideal_ballooning_gamma_hat(shat=0.0, alpha=0.0, nh=129))
    assert abs(g) < 1e-10


def test_ideal_ballooning_has_threshold_near_shat0() -> None:
    # Halpern et al. report a finite threshold around alpha_crit ~ O(0.5) for Dirichlet BCs and Lh=2π.
    g_lo = float(ideal_ballooning_gamma_hat(shat=0.0, alpha=0.2, nh=257))
    g_hi = float(ideal_ballooning_gamma_hat(shat=0.0, alpha=1.0, nh=257))
    assert g_hi > g_lo
    assert g_hi > 1e-3


def test_ideal_ballooning_shear_stabilizes_trend() -> None:
    # Pick an alpha in the clearly-unstable region so the trend is robust to discretization.
    alpha = 2.0
    g0 = float(ideal_ballooning_gamma_hat(shat=0.0, alpha=alpha, nh=257))
    g1 = float(ideal_ballooning_gamma_hat(shat=1.5, alpha=alpha, nh=257))
    assert np.isfinite(g0) and np.isfinite(g1)
    assert g1 < g0
