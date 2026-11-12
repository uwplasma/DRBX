"""Padded sparse rows of a fixed linear pair: same action, exact adjoint, reverse-mode gradients.

The current/phi pair is probed once per geometry (``assemble_sparse_rows``); every Krylov
iteration then applies it as one gather and a row sum instead of its mapped gathers and
nested transposes.  Checked on a sheared 2-D field-line map with a non-uniform measure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.fci_braginskii.native.fci_support_pair import (
    adjoint_sparse_rows,
    assemble_sparse_rows,
    build_weighted_negative_adjoint,
    sparse_rows_apply,
)

jax.config.update("jax_enable_x64", True)
SHAPE = (12, 14)


def _pair():
    nx, nz = SHAPE
    x = (np.arange(nx) + 0.5) / nx
    position = np.arange(nx) + (0.5 + 0.4 * np.sin(2 * np.pi * x)) * nx / nz
    i0 = np.floor(position).astype(int)
    w = position - i0

    def divergence(q):
        plus = jnp.roll(q, -1, axis=1)
        shifted = (1.0 - w)[:, None] * plus[i0 % nx] + w[:, None] * plus[(i0 + 1) % nx]
        return (shifted - jnp.roll(q, 1, axis=1)) * 3.0

    mass = jnp.asarray((1.0 + 0.3 * np.cos(2 * np.pi * x))[:, None] * np.ones((1, nz)) / (nx * nz))
    return divergence, build_weighted_negative_adjoint(divergence, mass, mass), mass


def _rows():
    divergence, gradient, mass = _pair()
    d = assemble_sparse_rows(divergence, SHAPE)
    assert d is not None
    m = np.asarray(mass)
    everything = np.ones(SHAPE, bool)
    g = adjoint_sparse_rows(*d, m, m, everything, everything)
    to = lambda r: tuple(jnp.asarray(a) for a in r)
    return (divergence, gradient, mass), (to(d), to(g))


def test_rows_reproduce_the_matrix_free_pair():
    (divergence, gradient, _), (d, g) = _rows()
    x = jnp.asarray(np.random.default_rng(0).normal(size=SHAPE))
    for op, rows in ((divergence, d), (gradient, g)):
        assert jnp.max(jnp.abs(sparse_rows_apply(*rows, x) - op(x))) <= 1e-13 * jnp.max(jnp.abs(op(x)))
    batch = jnp.stack((x, 2.0 * x))
    assert jnp.allclose(sparse_rows_apply(*d, batch), jnp.stack((divergence(x), 2.0 * divergence(x))), atol=1e-13)


def test_rows_keep_the_weighted_adjoint_to_round_off():
    (_, _, mass), (d, g) = _rows()
    rng = np.random.default_rng(1)
    x, y = (jnp.asarray(rng.normal(size=SHAPE)) for _ in range(2))
    lhs, rhs = jnp.sum(mass * sparse_rows_apply(*g, x) * y), -jnp.sum(mass * x * sparse_rows_apply(*d, y))
    assert abs(lhs - rhs) <= 1e-13 * abs(lhs)


def test_rows_give_the_reverse_mode_gradient():
    (divergence, gradient, _), (d, g) = _rows()
    x = jnp.asarray(np.random.default_rng(2).normal(size=SHAPE))
    loss = lambda dop, gop: lambda f: jnp.sum(dop(jnp.sin(gop(f))) ** 2)
    want = jax.grad(loss(divergence, gradient))(x)
    got = jax.grad(loss(lambda f: sparse_rows_apply(*d, f), lambda f: sparse_rows_apply(*g, f)))(x)
    assert jnp.max(jnp.abs(got - want)) <= 1e-12 * jnp.max(jnp.abs(want))


def test_a_global_coupling_is_not_sparse():
    assert assemble_sparse_rows(lambda f: f - jnp.mean(f), SHAPE) is None
