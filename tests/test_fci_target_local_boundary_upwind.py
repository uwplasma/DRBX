"""Target-local characteristic upwinding for wall-terminating FCI legs."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    parallel_characteristic_split_matrices,
    target_local_characteristic_upwind_correction,
)


def _diagonal_matrix():
    return jnp.diag(jnp.asarray([1.0, -2.0, 0.0, 3.0, -4.0]))


def test_split_matrices_select_directional_characteristics() -> None:
    matrix = _diagonal_matrix()
    a_plus, a_minus, p_plus, p_minus = parallel_characteristic_split_matrices(
        matrix
    )
    np.testing.assert_allclose(np.diag(a_plus), [1.0, 0.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(np.diag(a_minus), [0.0, -2.0, 0.0, 0.0, -4.0])
    np.testing.assert_allclose(np.diag(p_plus), [1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(np.diag(p_minus), [0.0, 1.0, 0.0, 0.0, 1.0])


def test_backward_wall_uses_one_sided_characteristic_legs() -> None:
    center = jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2])[None, None, None, :]
    minus = jnp.asarray([9.0, 8.0, 7.0, 6.0, 5.0])[None, None, None, :]
    plus = jnp.asarray([2.5, 2.0, 5.5, 1.7, -1.2])[None, None, None, :]
    gradient = jnp.asarray([0.3, -0.4, 0.5, -0.6, 0.7])[None, None, None, :]
    matrix = _diagonal_matrix()[None, None, None, :, :]
    correction = target_local_characteristic_upwind_correction(
        center,
        minus,
        plus,
        jnp.ones((1, 1, 1)),
        2.0 * jnp.ones((1, 1, 1)),
        gradient,
        matrix,
        jnp.ones((1, 1, 1), dtype=bool),
        jnp.zeros((1, 1, 1), dtype=bool),
    )

    equilibrium = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
    backward_state = jnp.asarray([1.0, center[0, 0, 0, 1], center[0, 0, 0, 2], 0.0,
                                  center[0, 0, 0, 4]])
    delta_minus = center[0, 0, 0] - backward_state
    delta_plus = (plus[0, 0, 0] - center[0, 0, 0]) / 2.0
    expected_principal = jnp.asarray([1.0, 0.0, 0.0, 3.0, 0.0]) * delta_minus
    expected_principal += jnp.asarray([0.0, -2.0, 0.0, 0.0, -4.0]) * delta_plus
    corrected_rhs = -(_diagonal_matrix() @ gradient[0, 0, 0]) + correction[0, 0, 0]
    np.testing.assert_allclose(corrected_rhs, -expected_principal)
    assert equilibrium.shape == (5,)


def test_interior_row_is_unchanged_and_constant_equilibrium_is_exact() -> None:
    equilibrium = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])[None, None, None, :]
    zero_gradient = jnp.zeros_like(equilibrium)
    matrix = _diagonal_matrix()[None, None, None, :, :]
    common = dict(
        center=equilibrium,
        minus=equilibrium,
        plus=equilibrium,
        dx_minus=jnp.ones((1, 1, 1)),
        dx_plus=jnp.ones((1, 1, 1)),
        centered_gradient=zero_gradient,
        matrix=matrix,
        forward_wall=jnp.zeros((1, 1, 1), dtype=bool),
    )
    interior = target_local_characteristic_upwind_correction(
        **common, backward_wall=jnp.zeros((1, 1, 1), dtype=bool)
    )
    wall = target_local_characteristic_upwind_correction(
        **common, backward_wall=jnp.ones((1, 1, 1), dtype=bool)
    )
    np.testing.assert_allclose(interior, 0.0)
    np.testing.assert_allclose(wall, 0.0)


def test_upwind_correction_has_finite_jvp() -> None:
    matrix = _diagonal_matrix()[None, None, None, :, :]
    base = jnp.asarray([1.1, 0.9, 1.2, 0.1, -0.1])[None, None, None, :]

    def evaluate(center):
        return target_local_characteristic_upwind_correction(
            center,
            0.9 * center,
            1.1 * center,
            jnp.ones((1, 1, 1)),
            jnp.ones((1, 1, 1)),
            0.2 * center,
            matrix,
            jnp.ones((1, 1, 1), dtype=bool),
            jnp.zeros((1, 1, 1), dtype=bool),
        )

    _, tangent = jax.jvp(evaluate, (base,), (jnp.ones_like(base),))
    assert bool(jnp.all(jnp.isfinite(tangent)))
