from __future__ import annotations

import numpy as np

from jaxdrb.analysis.lp import solve_lp_fixed_point
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model


def test_lp_fixed_point_converges_and_matches_ratio_definition() -> None:
    nl = 8
    geom = SlabGeometry.make(nl=nl, shat=0.5, curvature0=0.2)
    ky = np.linspace(0.06, 0.6, 8)
    q = 2.5

    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    res = solve_lp_fixed_point(
        params,
        geom,
        q=q,
        ky=ky,
        Lp0=10.0,
        omega_n_scale=1.0,
        max_iter=10,
        tol=2e-2,
        relax=0.8,
        arnoldi_m=14,
        arnoldi_tol=4e-3,
        nev=3,
        verbose=False,
    )

    assert np.isfinite(res.Lp) and res.Lp > 0
    assert np.isfinite(res.ky_star) and res.ky_star > 0
    assert np.isfinite(res.gamma_over_ky_star) and res.gamma_over_ky_star > 0

    # Last history row is [Lp, ky*, gamma*, (gamma/ky)*, Lp_target].
    Lp_hist, ky_star, gamma_star, ratio_star, Lp_target = res.history[-1]
    assert abs(ky_star - res.ky_star) < 1e-12
    assert abs(ratio_star - res.gamma_over_ky_star) < 1e-12
    assert abs(gamma_star - res.gamma_star) < 1e-12

    # Fixed point condition Lp ≈ q * (gamma/ky)_max should hold after convergence.
    assert abs(Lp_hist - Lp_target) / Lp_hist < 2e-1


def test_lp_fixed_point_runs_for_em_model() -> None:
    nl = 8
    geom = SlabGeometry.make(nl=nl, shat=0.3, curvature0=0.2)
    ky = np.linspace(0.08, 0.5, 6)

    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        beta=0.1,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        Dpsi=0.0,
    )

    res = solve_lp_fixed_point(
        params,
        geom,
        q=2.0,
        ky=ky,
        Lp0=10.0,
        omega_n_scale=1.0,
        model=get_model("em"),
        max_iter=5,
        arnoldi_m=12,
        arnoldi_tol=5e-3,
        nev=3,
        verbose=False,
    )

    assert np.isfinite(res.Lp) and res.Lp > 0
