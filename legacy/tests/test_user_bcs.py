from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.bc import BC1D
from jaxdrb.models.bcs import bc_relaxation_1d


def test_dirichlet_bc_relaxation_is_zero_when_satisfied():
    f = jnp.array([0.0, 1.0, 2.0, 3.0, 1.5])
    f = f.at[0].set(2.0).at[-1].set(-1.0)
    bc = BC1D.dirichlet(left=2.0, right=-1.0, nu=5.0)
    df = bc_relaxation_1d(f, bc=bc, dl=0.1)
    assert df[0] == 0.0
    assert df[-1] == 0.0


def test_neumann_zero_grad_relaxation_is_zero_for_constant_field():
    f = jnp.ones((8,))
    bc = BC1D.neumann(left=0.0, right=0.0, nu=3.0)
    df = bc_relaxation_1d(f, bc=bc, dl=0.2)
    assert df[0] == 0.0
    assert df[-1] == 0.0
