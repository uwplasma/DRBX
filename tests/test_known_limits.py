from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.geometry.tokamak import CircularTokamakGeometry
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import equilibrium
from jaxdrb.models.params import DRBParams
from tests.helpers import leading_eig_dense


def test_no_drive_limit_is_neutrally_stable() -> None:
    """With no gradient drives and no dissipation, the system should be neutral (Re ~ 0)."""

    nl = 16
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
    )

    lam = leading_eig_dense(linear_matvec(equilibrium(nl), params, geom, kx=0.0, ky=0.3), nl)
    assert abs(lam.real) < 1e-10


def test_connection_length_effect_via_q() -> None:
    """Increasing q (longer connection length) reduces parallel stabilization and can increase growth."""

    nl = 16
    kx = 0.0
    ky = 0.3

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

    geom_q2 = CircularTokamakGeometry.make(
        nl=nl, shat=0.0, q=2.0, R0=1.0, epsilon=0.18, curvature0=0.18
    )
    geom_q8 = CircularTokamakGeometry.make(
        nl=nl, shat=0.0, q=8.0, R0=1.0, epsilon=0.18, curvature0=0.18
    )

    g2 = leading_eig_dense(linear_matvec(equilibrium(nl), params, geom_q2, kx=kx, ky=ky), nl).real
    g8 = leading_eig_dense(linear_matvec(equilibrium(nl), params, geom_q8, kx=kx, ky=ky), nl).real

    assert np.isfinite(g2) and np.isfinite(g8)
    assert g8 > g2
