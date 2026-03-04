from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.terms.fields import _poisson_bc_eval


def test_poisson_invert_set_keeps_neumann_when_no_reference() -> None:
    params = DRBSystemParams().update(poisson_invert_set=True)
    bc = BC2D(kind_x=2, kind_y=0, x_grad=0.0, y_grad=0.0)
    out = _poisson_bc_eval(params, bc, ref=None)
    assert out.kind_x == 2
    assert out.kind_y == 0


def test_poisson_invert_set_uses_dirichlet_from_reference_profile() -> None:
    params = DRBSystemParams().update(poisson_invert_set=True)
    bc = BC2D(kind_x=2, kind_y=0, x_grad=0.0, y_grad=0.0)
    ref = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = _poisson_bc_eval(params, bc, ref=ref)
    assert out.kind_x == 1
    np.testing.assert_allclose(np.asarray(out.x_value[0]), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(out.x_value[1]), np.asarray([5.0, 6.0]))
