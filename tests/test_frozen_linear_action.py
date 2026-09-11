"""Coefficient reuse must retain AD and refresh at a changed Newton state."""
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parents[1]/'work/boundary_load_audit'))
from frozen_linear_action import frozen_linear_kernels


def test_frozen_action_matches_ad_and_refreshes_state_and_parameter():
    def residual(x, coefficient):
        return jnp.stack((jnp.sin(x[0])*x[1], coefficient*x[0]**2+jnp.exp(x[1])))
    prepare, apply = frozen_linear_kernels(residual)
    direction = jnp.array([.3, -.7])
    previous = None
    for state, parameter in ((jnp.array([.2, .4]), 2.), (jnp.array([.5, -.3]), 4.)):
        primal, linear = prepare(state, parameter)
        expected = jax.jvp(lambda x: residual(x, parameter), (state,), (direction,))[1]
        result = apply(linear, direction)
        np.testing.assert_allclose(primal, residual(state, parameter), rtol=1.e-6)
        np.testing.assert_allclose(result, expected, rtol=1.e-6)
        np.testing.assert_allclose(apply(linear, 2*direction), 2*result, rtol=1.e-6)
        if previous is not None:
            assert not np.allclose(result, previous)
        previous = result


def test_selected_input_block_matches_full_derivative():
    def residual(x):
        return jnp.array([jnp.exp(x[0])*jnp.sin(x[2]), x[1]**2*x[2]+x[0]*x[3]])
    selected = np.array([2, 3])
    prepare, apply = frozen_linear_kernels(
        lambda z, frozen: residual(frozen.at[selected].set(z)))
    state = jnp.array([.4, -.3, .2, .7])
    primal, linear = prepare(state[selected], state)
    for direction in (jnp.array([.8, -.2]), jnp.array([0., 1.])):
        tangent = jnp.zeros_like(state).at[selected].set(direction)
        reference = jax.jvp(residual, (state,), (tangent,))[1]
        np.testing.assert_allclose(apply(linear, direction), reference, rtol=1.e-6)
    np.testing.assert_allclose(primal, residual(state), rtol=1.e-6)
