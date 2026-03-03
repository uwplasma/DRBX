from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.parity_fv.poisson_vorticity import (
    apply_invert_set_x_guard,
    apply_parallel_free_y_guard,
    copy_outer_x_guard_cells,
    finalize_phi_after_poisson,
    prepare_phi_plus_pi_for_poisson,
)


def test_apply_invert_set_x_guard_midpoint_rule() -> None:
    nz, nx, ny = 2, 10, 12
    a = jnp.arange(nz * nx * ny, dtype=jnp.float64).reshape(nz, nx, ny)
    xstart, xend = 2, 7
    ystart, yend = 2, 9

    out = apply_invert_set_x_guard(a, xstart=xstart, xend=xend, ystart=ystart, yend=yend)

    lhs = np.asarray(out[:, xstart - 1, ystart : yend + 1])
    rhs = 0.5 * (
        np.asarray(a[:, xstart - 1, ystart : yend + 1])
        + np.asarray(a[:, xstart, ystart : yend + 1])
    )
    np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=0.0)

    lhs = np.asarray(out[:, xend + 1, ystart : yend + 1])
    rhs = 0.5 * (
        np.asarray(a[:, xend + 1, ystart : yend + 1]) + np.asarray(a[:, xend, ystart : yend + 1])
    )
    np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=0.0)


def test_copy_outer_x_guard_cells() -> None:
    nz, nx, ny = 1, 9, 8
    xstart, xend = 2, 6
    ystart, yend = 1, 6

    a = jnp.zeros((nz, nx, ny), dtype=jnp.float64)
    a = a.at[:, xstart - 1, ystart : yend + 1].set(3.0)
    a = a.at[:, xend + 1, ystart : yend + 1].set(-2.0)

    out = copy_outer_x_guard_cells(a, xstart=xstart, xend=xend, ystart=ystart, yend=yend)

    np.testing.assert_allclose(
        np.asarray(out[:, 0, ystart : yend + 1]),
        np.asarray(out[:, 1, ystart : yend + 1]),
    )
    np.testing.assert_allclose(
        np.asarray(out[:, xend + 2, ystart : yend + 1]),
        np.asarray(out[:, xend + 1, ystart : yend + 1]),
    )


def test_apply_parallel_free_y_guard() -> None:
    nz, nx, ny = 2, 8, 9
    xstart, xend = 2, 6
    ystart, yend = 2, 6

    rng = np.random.default_rng(123)
    a = jnp.asarray(rng.standard_normal((nz, nx, ny)))

    out = apply_parallel_free_y_guard(
        a,
        ystart=ystart,
        yend=yend,
        xstart=xstart,
        xend=xend,
    )

    np.testing.assert_allclose(
        np.asarray(out[:, xstart : xend + 1, ystart - 1]),
        2.0 * np.asarray(a[:, xstart : xend + 1, ystart])
        - np.asarray(a[:, xstart : xend + 1, ystart + 1]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(out[:, xstart : xend + 1, yend + 1]),
        2.0 * np.asarray(a[:, xstart : xend + 1, yend])
        - np.asarray(a[:, xstart : xend + 1, yend - 1]),
        rtol=0.0,
        atol=0.0,
    )


def test_one_step_poisson_guard_pipeline_semantics() -> None:
    nz, nx, ny = 2, 10, 10
    xstart, xend = 2, 7
    ystart, yend = 2, 7

    rng = np.random.default_rng(7)
    phi = jnp.asarray(rng.standard_normal((nz, nx, ny)))
    pi_hat = jnp.asarray(rng.standard_normal((nz, nx, ny)))

    u_pre = prepare_phi_plus_pi_for_poisson(
        phi,
        pi_hat,
        xstart=xstart,
        xend=xend,
        ystart=ystart,
        yend=yend,
    )

    # In this unit test we bypass the solver and verify guard semantics only.
    phi_post = u_pre - pi_hat
    out = finalize_phi_after_poisson(
        phi_post,
        xstart=xstart,
        xend=xend,
        ystart=ystart,
        yend=yend,
        parallel_free_y=True,
    )

    # Outer x guards are copies of adjacent guards.
    np.testing.assert_allclose(
        np.asarray(out[:, 0, ystart : yend + 1]),
        np.asarray(out[:, 1, ystart : yend + 1]),
    )
    np.testing.assert_allclose(
        np.asarray(out[:, xend + 2, ystart : yend + 1]),
        np.asarray(out[:, xend + 1, ystart : yend + 1]),
    )
    # Parallel y free-guard relations hold over x interior.
    np.testing.assert_allclose(
        np.asarray(out[:, xstart : xend + 1, ystart - 1]),
        2.0 * np.asarray(out[:, xstart : xend + 1, ystart])
        - np.asarray(out[:, xstart : xend + 1, ystart + 1]),
        rtol=0.0,
        atol=0.0,
    )
