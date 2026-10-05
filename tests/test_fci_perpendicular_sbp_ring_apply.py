"""Ring derivatives of the nodal SBP ops (banded radial stencil, FFT angular) against the dense plan matrices."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry.sbp_operators import radial_block, ring_dtheta
from drbx.native.fci_perpendicular_sbp_ops import d1, d2
from drbx.stencils.nodal_plan import RingBlockArrays

SIZES_N = [8, 16, 32, 64]
SIZES_M = [5, 8, 28, 56]    # 5 < 8: dense fallback with an arbitrary (non-SBP) matrix
TOL = 1e-13
E, TRAIL = 3, (2, 3)


def ring_arrays(m, N, delta=np.pi / 32, du=1.0 / 32, seed=0):
    Du = radial_block(m).D_unit / du if m >= 8 else np.random.default_rng(seed).standard_normal((m, m))
    z = np.zeros(m)
    return RingBlockArrays(Du, ring_dtheta(N, delta), z, z, np.ones(m))


def fake_plan(*rings):
    """Minimal stand-in exposing what ``d1``/``d2`` read: ``structure.blocks`` and ``blocks`` (rings are ``(m, N, arrays)``)."""
    blocks, descs, off = [], [], 0
    for m, N, arr in rings:
        descs.append(("ring", off, m * N, m, N, 0))
        blocks.append(arr)
        off += m * N
    return SimpleNamespace(structure=SimpleNamespace(blocks=tuple(descs)), blocks=tuple(blocks)), off


def field(P, seed=1, trailing=TRAIL):
    return np.random.default_rng(seed).standard_normal((E, P) + trailing)


def rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def dense_d1(arr, x, m, N):
    gl = x.reshape((E, m, N) + x.shape[2:])
    return np.einsum("ab,ebj...->eaj...", arr.Du, gl).reshape(x.shape)


def dense_d2(arr, x, m, N):
    gl = x.reshape((E, m, N) + x.shape[2:])
    return np.einsum("jl,eal...->eaj...", arr.Dth, gl).reshape(x.shape)


@pytest.mark.parametrize("trailing", [(), TRAIL])
@pytest.mark.parametrize("m", SIZES_M)
@pytest.mark.parametrize("N", SIZES_N)
def test_ring_derivatives_match_dense(N, m, trailing):
    arr = ring_arrays(m, N)
    plan, P = fake_plan((m, N, arr))
    x = field(P, trailing=trailing)
    assert rel(np.asarray(d1(plan, jnp.asarray(x))), dense_d1(arr, x, m, N)) <= TOL
    assert rel(np.asarray(d2(plan, jnp.asarray(x))), dense_d2(arr, x, m, N)) <= TOL


def test_multiple_ring_blocks_and_jit_plan_argument():
    rings = [(8, 16, ring_arrays(8, 16)), (28, 32, ring_arrays(28, 32, du=1.0 / 64)), (6, 8, ring_arrays(6, 8))]
    plan, P = fake_plan(*rings)
    x = field(P)
    off, ref1, ref2 = 0, [], []
    for m, N, arr in rings:
        xb = x[:, off:off + m * N]
        ref1.append(dense_d1(arr, xb, m, N))
        ref2.append(dense_d2(arr, xb, m, N))
        off += m * N
    # plan arrays enter jit as traced arguments, so coefficients come from the (traced) Du
    arrays = tuple(jax.tree_util.tree_map(jnp.asarray, r[2]) for r in rings)

    @jax.jit
    def apply(blocks, xx):
        p = SimpleNamespace(structure=plan.structure, blocks=blocks)
        return d1(p, xx), d2(p, xx)

    g1, g2 = apply(arrays, jnp.asarray(x))
    assert rel(np.asarray(g1), np.concatenate(ref1, axis=1)) <= TOL
    assert rel(np.asarray(g2), np.concatenate(ref2, axis=1)) <= TOL


@pytest.mark.parametrize("N", SIZES_N)
def test_theta_derivative_independent_of_node_offset(N):
    m = 8
    x = field(m * N)
    ref = None
    for delta in (0.0, np.pi / N, 0.37):
        arr = ring_arrays(m, N, delta=delta)
        plan, _ = fake_plan((m, N, arr))
        out = np.asarray(d2(plan, jnp.asarray(x)))
        assert rel(out, dense_d2(arr, x, m, N)) <= TOL
        ref = out if ref is None else ref
        assert rel(out, ref) <= TOL


@pytest.mark.parametrize("m,N", [(5, 16), (8, 8), (28, 32), (56, 64)])
def test_transpose_matches_dense(m, N):
    arr = ring_arrays(m, N)
    plan, P = fake_plan((m, N, arr))
    x = jnp.asarray(field(P, trailing=(2,)))
    y = jnp.asarray(np.random.default_rng(7).standard_normal(x.shape))
    dense = {
        d1: lambda v: jnp.einsum("ab,ebj...->eaj...", arr.Du, v.reshape((E, m, N, 2))).reshape(v.shape),
        d2: lambda v: jnp.einsum("jl,eal...->eaj...", arr.Dth, v.reshape((E, m, N, 2))).reshape(v.shape),
    }
    for op, ref in dense.items():
        (xt,) = jax.linear_transpose(lambda v, op=op: op(plan, v), x)(y)
        (ref_t,) = jax.linear_transpose(ref, x)(y)
        assert rel(np.asarray(xt), np.asarray(ref_t)) <= TOL
        _, pull = jax.vjp(lambda v, op=op: op(plan, v), x)
        assert rel(np.asarray(pull(y)[0]), np.asarray(ref_t)) <= TOL
        lhs = float(jnp.vdot(op(plan, x), y))
        assert abs(lhs - float(jnp.vdot(x, xt))) <= 1e-12 * max(abs(lhs), 1.0)
