from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.parity_fv import ParityFVParams, ParityFVRHS, ParityFVState
from jaxdrb.parity_fv.geometry import ParityFVGeometry


def test_parity_fv_rhs_scaffold_shapes() -> None:
    p = ParityFVParams(nx=4, ny=5, nz=3, dx=1.0, dy=1.0, dz=1.0)
    shape = p.shape()
    y = ParityFVState(
        n=jnp.ones(shape),
        pe=2.0 * jnp.ones(shape),
        vort=jnp.ones(shape),
        phi=jnp.ones(shape),
        vpar_e=jnp.ones(shape),
        vpar_i=jnp.ones(shape),
    )
    g = ParityFVGeometry(jacobian=jnp.ones(shape))
    rhs = ParityFVRHS(params=p, geom=g)

    dy = rhs(0.0, y)
    assert dy.n.shape == shape
    assert float(jnp.max(jnp.abs(dy.n))) == 0.0

    te = rhs.te(y)
    assert te.shape == shape
    assert float(jnp.mean(te)) == 2.0
