from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.fci3d import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid


def main() -> None:
    grid = FCISlabGrid.make(
        nx=24,
        ny=24,
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
        omega_Ti=0.3,
        eta_par=1.0,
        me_hat=0.2,
        beta=0.05,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        DTi=1e-3,
        Dpsi=1e-3,
        em_on=True,
        hot_ion_on=True,
        tau_i=1.0,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    key = jax.random.key(21)
    noise = 1e-3 * jax.random.normal(key, (grid.nz, grid.nx, grid.ny))
    y0 = FCIDRB3DFullState(
        n=noise,
        omega=noise,
        vpar_e=jnp.zeros_like(noise),
        vpar_i=jnp.zeros_like(noise),
        Te=jnp.zeros_like(noise),
        Ti=jnp.zeros_like(noise),
        psi=jnp.zeros_like(noise),
    )

    dy = model.rhs(0.0, y0)
    print("FCI 3D EM+hot RHS psi norm:", float(jnp.sqrt(jnp.mean(dy.psi**2))))


if __name__ == "__main__":
    main()
