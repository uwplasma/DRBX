from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.params import DRBParams


def main() -> None:
    nl = 64
    geom = OpenSlabGeometry.make(nl=nl, length=6.0, shat=0.0, curvature0=0.0)
    params = DRBParams(
        omega_n=0.2,
        omega_Te=0.6,
        eta=1.0,
        me_hat=0.2,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        sheath_bc_on=False,
    )
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)

    key = jax.random.key(0)
    y = State.random(key, nl, amplitude=1e-3)
    dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.4, eq=eq)

    norm = lambda a: float(jnp.sqrt(jnp.mean(jnp.abs(a) ** 2)))
    print("Cold-ion ES line RHS norms:")
    print({"n": norm(dy.n), "omega": norm(dy.omega), "vpar_e": norm(dy.vpar_e)})


if __name__ == "__main__":
    main()
