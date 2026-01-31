from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import equilibrium
from jaxdrb.models.params import DRBParams
from tests.helpers import leading_eig_dense


def test_driftwave_frequency_scales_with_ky() -> None:
    nl = 16
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    lam1 = leading_eig_dense(linear_matvec(equilibrium(nl), params, geom, kx=0.0, ky=0.2), nl)
    lam2 = leading_eig_dense(linear_matvec(equilibrium(nl), params, geom, kx=0.0, ky=0.4), nl)

    assert abs(lam2.imag) > abs(lam1.imag)


def test_resistive_branch_growth_increases_with_eta() -> None:
    nl = 16
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    ky = 0.3

    base = dict(
        omega_n=0.8,
        omega_Te=0.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )
    lam_lo = leading_eig_dense(
        linear_matvec(equilibrium(nl), DRBParams(**base, eta=0.1), geom, kx=0.0, ky=ky), nl
    )
    lam_hi = leading_eig_dense(
        linear_matvec(equilibrium(nl), DRBParams(**base, eta=5.0), geom, kx=0.0, ky=ky), nl
    )

    assert lam_hi.real > lam_lo.real


def test_ballooning_like_curvature_and_shear_trends() -> None:
    nl = 16
    ky = 0.05
    base_params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    # Curvature drive increases growth.
    geom0 = SlabGeometry.make(nl=nl, shat=1.0, curvature0=0.0)
    geom1 = SlabGeometry.make(nl=nl, shat=1.0, curvature0=0.3)
    g0 = leading_eig_dense(linear_matvec(equilibrium(nl), base_params, geom0, kx=0.0, ky=ky), nl).real
    g1 = leading_eig_dense(linear_matvec(equilibrium(nl), base_params, geom1, kx=0.0, ky=ky), nl).real
    assert g1 > g0

    # Magnetic shear stabilizes (for this ballooning-like case).
    geom_lo = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.2)
    geom_hi = SlabGeometry.make(nl=nl, shat=1.0, curvature0=0.2)
    g_lo = leading_eig_dense(
        linear_matvec(equilibrium(nl), base_params, geom_lo, kx=0.0, ky=ky), nl
    ).real
    g_hi = leading_eig_dense(
        linear_matvec(equilibrium(nl), base_params, geom_hi, kx=0.0, ky=ky), nl
    ).real
    assert g_hi < g_lo

