from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState


def main() -> None:
    grid = Grid2D.make(nx=64, ny=64, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(
        kappa=0.2,
        alpha=1.0,
        Dn=1e-3,
        DOmega=1e-3,
        bracket="arakawa",
        poisson="spectral",
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(13)
    n0 = 1e-3 * jax.random.normal(key, (grid.nx, grid.ny))
    omega0 = 1e-3 * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny))
    y0 = HW2DState(n=n0, omega=omega0)

    dy = model.rhs(0.0, y0)
    diag = model.diagnostics(y0)
    print("HW2D diagnostics:", {"E": float(diag["E"]), "Z": float(diag["Z"])})
    print("RHS omega norm:", float(jnp.sqrt(jnp.mean(dy.omega**2))))


if __name__ == "__main__":
    main()
