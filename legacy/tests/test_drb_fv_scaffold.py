from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.drb_fv import DRBFVParams, DRBFVRHS, DRBFVState
from jaxdrb.drb_fv.geometry import DRBFVGeometry


def test_drb_fv_rhs_scaffold_shapes() -> None:
    p = DRBFVParams(nx=4, ny=5, nz=3, dx=1.0, dy=1.0, dz=1.0)
    shape = p.shape()
    y = DRBFVState(
        n=jnp.ones(shape),
        pe=2.0 * jnp.ones(shape),
        vort=jnp.ones(shape),
        phi=jnp.ones(shape),
        vpar_e=jnp.ones(shape),
        vpar_i=jnp.ones(shape),
    )
    g = DRBFVGeometry(jacobian=jnp.ones(shape))
    rhs = DRBFVRHS(params=p, geom=g)

    dy = rhs(0.0, y)
    assert dy.n.shape == shape
    assert float(jnp.max(jnp.abs(dy.n))) == 0.0

    te = rhs.te(y)
    assert te.shape == shape
    assert float(jnp.mean(te)) == 2.0
