from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.native.array_backend import is_jax_array, use_jax_backend


def test_backend_selector_short_circuits_numpy_and_scalars() -> None:
    assert not is_jax_array(None)
    assert not is_jax_array(1.0)
    assert not is_jax_array(np.float64(1.0))
    assert not is_jax_array(np.array([1.0, 2.0]))
    assert not use_jax_backend(None, 1.0, np.array([1.0]))


def test_backend_selector_detects_jax_arrays_and_tracers() -> None:
    assert is_jax_array(jnp.array([1.0, 2.0]))
    assert use_jax_backend(np.array([1.0]), jnp.array([2.0]))

    def traced_selector(value):
        return jnp.where(use_jax_backend(value), value, value + 1.0)

    _, tangent = jax.jvp(traced_selector, (jnp.array(2.0),), (jnp.array(3.0),))
    np.testing.assert_allclose(np.asarray(tangent), np.array(3.0))
