from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.drb2d_hot_ion import DRB2DHotIonModel, DRB2DHotIonParams, DRB2DHotIonState


def main() -> None:
    grid = Grid2D.make(nx=64, ny=64, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = DRB2DHotIonParams(
        omega_n=0.2,
        omega_Te=0.6,
        omega_Ti=0.3,
        eta=1.0,
        me_hat=0.2,
        tau_i=1.0,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        DTi=1e-3,
        bracket="arakawa",
        poisson="spectral",
    )
    model = DRB2DHotIonModel(params=params, grid=grid)

    key = jax.random.key(11)
    n0 = 1e-3 * jax.random.normal(key, (grid.nx, grid.ny))
    omega0 = 1e-3 * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny))
    y0 = DRB2DHotIonState(
        n=n0,
        omega=omega0,
        vpar_e=jnp.zeros_like(n0),
        vpar_i=jnp.zeros_like(n0),
        Te=jnp.zeros_like(n0),
        Ti=jnp.zeros_like(n0),
    )

    dy = model.rhs(0.0, y0)
    print("DRB2D hot-ion energy:", float(model.energy(y0)))
    print("RHS Ti norm:", float(jnp.sqrt(jnp.mean(dy.Ti**2))))


if __name__ == "__main__":
    main()
