from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import State, rhs_nonlinear
from jaxdrb.models.params import DRBParams


def test_alpha_Te_ohm_enters_parallel_electron_momentum() -> None:
    # Sanity check that the Braginskii thermal-force coefficient alpha_Te_ohm is wired into
    # the electron parallel momentum equation as ∇_||(phi - n - alpha Te).
    nl = 33
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)

    # A simple Te perturbation varying along l so ∇_|| Te != 0.
    Te = jnp.sin(geom.l).astype(jnp.complex128)
    y = State(
        n=jnp.zeros((nl,), dtype=jnp.complex128),
        omega=jnp.zeros((nl,), dtype=jnp.complex128),
        vpar_e=jnp.zeros((nl,), dtype=jnp.complex128),
        vpar_i=jnp.zeros((nl,), dtype=jnp.complex128),
        Te=Te,
    )

    base = dict(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=1.0,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
    )

    r0 = rhs_nonlinear(0.0, y, DRBParams(**base, alpha_Te_ohm=0.0), geom, kx=0.0, ky=0.3)
    r1 = rhs_nonlinear(0.0, y, DRBParams(**base, alpha_Te_ohm=1.71), geom, kx=0.0, ky=0.3)

    # dv_e(alpha=1.71) - dv_e(alpha=0) = -(1.71) ∇_|| Te (since phi=n=0 here).
    dpar_Te = geom.dpar(Te)
    diff = r1.vpar_e - r0.vpar_e
    err = jnp.max(jnp.abs(diff + 1.71 * dpar_Te))
    assert float(err) < 5e-12
