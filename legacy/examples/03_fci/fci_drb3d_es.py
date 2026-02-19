from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.fci3d import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid


def main() -> None:
    grid = FCISlabGrid.make(
        nx=32,
        ny=32,
        nz=4,
        Lx=2 * jnp.pi,
        Ly=2 * jnp.pi,
        Lz=2.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=False,
    )
    params = FCIDRB3DFullParams(
        omega_n=0.2,
        omega_Te=0.6,
        eta_par=1.0,
        me_hat=0.2,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        em_on=False,
        hot_ion_on=False,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    key = jax.random.key(20)
    y0 = FCIDRB3DFullState.zeros((grid.nz, grid.nx, grid.ny))
    noise = 1e-3 * jax.random.normal(key, (grid.nz, grid.nx, grid.ny))
    y0 = FCIDRB3DFullState(
        n=noise,
        omega=noise,
        vpar_e=jnp.zeros_like(noise),
        vpar_i=jnp.zeros_like(noise),
        Te=jnp.zeros_like(noise),
    )

    dy = model.rhs(0.0, y0)
    print("FCI 3D ES RHS omega norm:", float(jnp.sqrt(jnp.mean(dy.omega**2))))


if __name__ == "__main__":
    main()
