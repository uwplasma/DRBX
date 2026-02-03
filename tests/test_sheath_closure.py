from __future__ import annotations

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import equilibrium
from jaxdrb.models.params import DRBParams
from tests.helpers import leading_eig_dense


def test_open_slab_sheath_loss_stabilizes_no_drive_limit() -> None:
    nl = 33
    geom = OpenSlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0, length=6.0)
    y0 = equilibrium(nl)

    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        sheath_on=True,
        sheath_nu_factor=1.0,
    )

    lam = leading_eig_dense(linear_matvec(y0, params, geom, kx=0.0, ky=0.3), y0)
    assert lam.real < 0.0


def test_sheath_nu_factor_increases_damping() -> None:
    nl = 33
    geom = OpenSlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0, length=6.0)
    y0 = equilibrium(nl)

    base = dict(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        sheath_on=True,
    )

    lam_lo = leading_eig_dense(
        linear_matvec(y0, DRBParams(**base, sheath_nu_factor=0.5), geom, kx=0.0, ky=0.3),
        y0,
    )
    lam_hi = leading_eig_dense(
        linear_matvec(y0, DRBParams(**base, sheath_nu_factor=2.0), geom, kx=0.0, ky=0.3),
        y0,
    )

    assert lam_hi.real < lam_lo.real

