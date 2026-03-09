from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jaxdrb.hermes_literal import (
    Field3DLayout,
    apply_neumann_boundary_average_z,
    apply_neumann_field3d,
    set_boundary_to,
)


def _layout() -> Field3DLayout:
    return Field3DLayout(pstart=2, pend=5, xstart=2, xend=5, guard_width=2)


def test_apply_neumann_boundary_average_z_matches_hermes_average_mirror() -> None:
    arr = jnp.arange(8 * 8 * 4, dtype=jnp.float64).reshape(8, 8, 4)
    out = apply_neumann_boundary_average_z(arr, layout=_layout())
    avg_low = np.mean(np.asarray(arr)[:, 2, :], axis=-1, keepdims=True)
    expect_low = 2.0 * avg_low - np.asarray(arr)[:, 2, :]
    avg_high = np.mean(np.asarray(arr)[:, 5, :], axis=-1, keepdims=True)
    expect_high = 2.0 * avg_high - np.asarray(arr)[:, 5, :]
    np.testing.assert_allclose(np.asarray(out)[:, 1, :], expect_low)
    np.testing.assert_allclose(np.asarray(out)[:, 0, :], expect_low)
    np.testing.assert_allclose(np.asarray(out)[:, 6, :], expect_high)
    np.testing.assert_allclose(np.asarray(out)[:, 7, :], expect_high)


def test_apply_neumann_field3d_is_differentiable() -> None:
    arr = jnp.arange(8 * 8 * 4, dtype=jnp.float64).reshape(8, 8, 4)

    def f(x):
        out = apply_neumann_field3d(
            x,
            axis=1,
            interior_start=2,
            interior_end=5,
            spacing=1.0,
            guard_width=2,
        )
        return jnp.sum(out * out)

    grad = jax.grad(f)(arr)
    assert np.isfinite(np.asarray(grad)).all()


def test_set_boundary_to_preserves_midpoint_recursively() -> None:
    arr = jnp.zeros((8, 8, 4), dtype=jnp.float64)
    ref = jnp.ones((8, 8, 4), dtype=jnp.float64)
    out = set_boundary_to(arr, ref, layout=_layout())
    np.testing.assert_allclose(np.asarray(out)[:, 1, :], 2.0)
    np.testing.assert_allclose(np.asarray(out)[:, 0, :], 0.0)
