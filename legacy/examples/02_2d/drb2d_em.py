from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState


def main() -> None:
    grid = Grid2D.make(nx=64, ny=64, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = DRB2DEMParams(
        omega_n=0.2,
        omega_Te=0.6,
        eta=1.0,
        me_hat=0.2,
        beta=0.05,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        Dpsi=1e-3,
        bracket="arakawa",
        poisson="spectral",
    )
    model = DRB2DEMModel(params=params, grid=grid)

    key = jax.random.key(12)
    n0 = 1e-3 * jax.random.normal(key, (grid.nx, grid.ny))
    omega0 = 1e-3 * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny))
    psi0 = 1e-3 * jax.random.normal(jax.random.split(key, 3)[2], (grid.nx, grid.ny))
    y0 = DRB2DEMState(
        n=n0,
        omega=omega0,
        vpar_e=jnp.zeros_like(n0),
        vpar_i=jnp.zeros_like(n0),
        Te=jnp.zeros_like(n0),
        psi=psi0,
    )

    dy = model.rhs(0.0, y0)
    print("DRB2D EM energy:", float(model.energy(y0)))
    print("RHS psi norm:", float(jnp.sqrt(jnp.mean(dy.psi**2))))


if __name__ == "__main__":
    main()
