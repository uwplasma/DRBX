from __future__ import annotations

import numpy as np

import jax

from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.grid import Grid2D


def test_drb2d_em_split_parity_and_energy_budget() -> None:
    grid = Grid2D.make(nx=24, ny=24, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DEMParams(
        omega_n=0.6,
        omega_Te=0.2,
        kpar=0.3,
        eta=0.2,
        me_hat=0.2,
        beta=0.2,
        Dpsi=1e-3,
        curvature_on=True,
        curvature_coeff=0.4,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        Dn4=2e-4,
        DOmega4=2e-4,
        DTe4=2e-4,
        Dpsi4=2e-4,
        mu_zonal_omega=0.1,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DEMModel(params=params, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    amp = 1e-3
    y = DRB2DEMState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(1), shape),
        psi=amp * jax.random.normal(jax.random.key(2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(3), shape),
        Te=amp * jax.random.normal(jax.random.key(4), shape),
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
                    jax.numpy.max(jax.numpy.abs(rhs.psi - total.psi)),
                    jax.numpy.max(jax.numpy.abs(rhs.vpar_i - total.vpar_i)),
                    jax.numpy.max(jax.numpy.abs(rhs.Te - total.Te)),
                ]
            )
        )
    )
    assert err < 1e-12

    edot_full = float(model.energy_rate(y, rhs))
    edot_budget = float(model.energy_budget(y)["E_dot_total"])
    rel = abs(edot_full - edot_budget) / max(abs(edot_full), 1e-12)
    assert rel < 1e-10
