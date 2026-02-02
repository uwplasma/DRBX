from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.matvec import linear_matvec_from_rhs
from jaxdrb.models.em_drb import equilibrium as em_equilibrium
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs
from jaxdrb.models.params import DRBParams
from tests.helpers import leading_eig_dense


def test_em_no_drive_limit_is_neutrally_stable() -> None:
    nl = 12
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.05,
        beta=0.1,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        Dpsi=0.0,
    )

    y0 = em_equilibrium(nl)
    matvec = linear_matvec_from_rhs(em_rhs, y0, params, geom, kx=0.0, ky=0.3)
    lam = leading_eig_dense(matvec, y0)
    assert abs(lam.real) < 1e-10


def test_em_beta_changes_growth_rate_trend() -> None:
    """Finite beta should change the spectrum compared to a smaller-beta case."""

    nl = 12
    geom = SlabGeometry.make(nl=nl, shat=0.3, curvature0=0.2)
    base = dict(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        Dpsi=0.0,
        boussinesq=True,
    )

    y0 = em_equilibrium(nl)
    g_lo = leading_eig_dense(
        linear_matvec_from_rhs(em_rhs, y0, DRBParams(**base, beta=0.02), geom, kx=0.0, ky=0.3),
        y0,
    ).real
    g_hi = leading_eig_dense(
        linear_matvec_from_rhs(em_rhs, y0, DRBParams(**base, beta=0.2), geom, kx=0.0, ky=0.3),
        y0,
    ).real

    assert np.isfinite(g_lo) and np.isfinite(g_hi)
    assert abs(g_hi - g_lo) > 1e-6
