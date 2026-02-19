from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.matvec import linear_matvec, linear_matvec_from_rhs
from jaxdrb.models.cold_ion_drb import equilibrium as cold_equilibrium
from jaxdrb.models.hot_ion_drb import equilibrium as hot_equilibrium
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_rhs
from jaxdrb.models.params import DRBParams
from tests.helpers import leading_eig_dense


def test_hot_ion_tau_zero_matches_cold_ion_leading_growth() -> None:
    nl = 12
    geom = SlabGeometry.make(nl=nl, shat=0.3, curvature0=0.2)
    kx = 0.0
    ky = 0.3

    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        omega_Ti=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        DTi=0.01,
        tau_i=0.0,
    )

    y0_cold = cold_equilibrium(nl)
    y0_hot = hot_equilibrium(nl)

    g_cold = leading_eig_dense(linear_matvec(y0_cold, params, geom, kx=kx, ky=ky), y0_cold).real
    g_hot = leading_eig_dense(
        linear_matvec_from_rhs(hot_rhs, y0_hot, params, geom, kx=kx, ky=ky),
        y0_hot,
    ).real

    assert np.isfinite(g_cold) and np.isfinite(g_hot)
    assert abs(g_hot - g_cold) < 1e-10


def test_hot_ion_tau_changes_growth_rate() -> None:
    nl = 12
    geom = SlabGeometry.make(nl=nl, shat=0.3, curvature0=0.2)
    kx = 0.0
    ky = 0.3

    base = dict(
        omega_n=0.8,
        omega_Te=0.0,
        omega_Ti=0.8,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        DTi=0.01,
        boussinesq=True,
    )

    y0_hot = hot_equilibrium(nl)
    g_tau0 = leading_eig_dense(
        linear_matvec_from_rhs(hot_rhs, y0_hot, DRBParams(**base, tau_i=0.0), geom, kx=kx, ky=ky),
        y0_hot,
    ).real
    g_tau1 = leading_eig_dense(
        linear_matvec_from_rhs(hot_rhs, y0_hot, DRBParams(**base, tau_i=1.0), geom, kx=kx, ky=ky),
        y0_hot,
    ).real

    assert np.isfinite(g_tau0) and np.isfinite(g_tau1)
    assert abs(g_tau1 - g_tau0) > 1e-6
