from __future__ import annotations

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.linear.matvec import linear_matvec_from_rhs
from jaxdrb.models.hot_ion_drb import equilibrium as hot_equilibrium
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_rhs
from jaxdrb.models.params import DRBParams
from tests.helpers import leading_eig_dense


def test_open_slab_loizu2012_hot_ion_bc_does_not_create_growth_in_no_drive_limit() -> None:
    """No-drive check for hot-ion + Loizu2012 full-set MPSE BCs.

    This mirrors `tests/test_sheath_closure.py` for the 6-field hot-ion system.
    """

    nl = 17  # keep the dense-Jacobian test fast
    geom = OpenSlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0, length=6.0)
    y0 = hot_equilibrium(nl)

    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        eta=0.0,
        me_hat=0.05,
        tau_i=1.0,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        sheath_bc_on=True,
        sheath_bc_model=1,
        sheath_bc_nu_factor=1.0,
        sheath_cos2=1.0,
    )

    matvec = linear_matvec_from_rhs(hot_rhs, y0, params, geom, kx=0.0, ky=0.3)
    lam = leading_eig_dense(matvec, y0)
    assert lam.real < 1e-8
