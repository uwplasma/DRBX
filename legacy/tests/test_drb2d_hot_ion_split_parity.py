from __future__ import annotations

import numpy as np

import jax

from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D


def test_drb2d_hot_ion_split_parity_and_energy_budget() -> None:
    grid = Grid2D.make(nx=24, ny=24, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DHotIonParams(
        omega_n=0.6,
        omega_Te=0.2,
        omega_Ti=0.1,
        kpar=0.3,
        eta=0.2,
        me_hat=0.2,
        tau_i=1.0,
        alpha_Te_ohm=1.71,
        alpha_Ti=1.0,
        curvature_on=True,
        curvature_coeff=0.5,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        DTi=1e-3,
        Dn4=2e-4,
        DOmega4=2e-4,
        DTe4=2e-4,
        DTi4=2e-4,
        mu_zonal_omega=0.1,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DHotIonModel(params=params, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    amp = 1e-3
    y = DRB2DHotIonState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(3), shape),
        Te=amp * jax.random.normal(jax.random.key(4), shape),
        Ti=amp * jax.random.normal(jax.random.key(5), shape),
    )

    split = model.rhs_decomposed(0.0, y)
    rhs = model.rhs(0.0, y)
    total = split.total()
    err = float(
        jax.numpy.max(
            jax.numpy.asarray(
                [
                    jax.numpy.max(jax.numpy.abs(rhs.n - total.n)),
                    jax.numpy.max(jax.numpy.abs(rhs.omega - total.omega)),
                    jax.numpy.max(jax.numpy.abs(rhs.vpar_e - total.vpar_e)),
                    jax.numpy.max(jax.numpy.abs(rhs.vpar_i - total.vpar_i)),
                    jax.numpy.max(jax.numpy.abs(rhs.Te - total.Te)),
                    jax.numpy.max(jax.numpy.abs(rhs.Ti - total.Ti)),
                ]
            )
        )
    )
    assert err < 1e-12

    edot_full = float(model.energy_rate(y, rhs))
    edot_budget = float(model.energy_budget(y)["E_dot_total"])
    rel = abs(edot_full - edot_budget) / max(abs(edot_full), 1e-12)
    assert rel < 1e-10
