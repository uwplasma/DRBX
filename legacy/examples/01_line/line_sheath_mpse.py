from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.params import DRBParams


def main() -> None:
    nl = 65
    geom = OpenSlabGeometry.make(nl=nl, length=6.0, shat=0.0, curvature0=0.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        sheath_bc_on=True,
        sheath_bc_model=1,  # Loizu-style linearized full set
        sheath_bc_nu_factor=1.0,
        sheath_cos2=1.0,
        sheath_end_damp_on=False,
        sheath_loss_on=False,
        sheath_heat_on=False,
    )
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)

    key = jax.random.key(3)
    y = State.random(key, nl, amplitude=1e-3)
    dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.35, eq=eq)

    mask = jnp.asarray(geom.sheath_mask, dtype=bool)
    print("MPSE BC boundary RHS norms:")
    print({
        "n": float(jnp.max(jnp.abs(dy.n[mask]))),
        "omega": float(jnp.max(jnp.abs(dy.omega[mask]))),
        "vpar_i": float(jnp.max(jnp.abs(dy.vpar_i[mask]))),
    })


if __name__ == "__main__":
    main()
