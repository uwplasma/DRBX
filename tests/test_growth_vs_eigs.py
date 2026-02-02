from __future__ import annotations

import jax

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams
from tests.helpers import leading_eig_dense


def test_growth_rate_matches_leading_eigenvalue() -> None:
    nl = 16
    kx = 0.0
    ky = 0.3

    geom = SlabGeometry.make(nl=nl, shat=0.5, curvature0=0.2)
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

    y0 = equilibrium(nl)
    matvec = linear_matvec(y0, params, geom, kx=kx, ky=ky)

    lam = leading_eig_dense(matvec, y0)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    # This case has multiple nearby modes; use a longer time to ensure the dominant mode wins.
    gr = estimate_growth_rate(matvec, v0, tmax=90.0, dt0=0.03, nsave=300, fit_window=0.5)

    assert abs(gr.gamma - lam.real) / (abs(lam.real) + 1e-12) < 5e-2
