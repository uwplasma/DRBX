"""Fast synthetic tests of ``drbx.native.fci_perpendicular_plane_preconditioner``.

The production preconditioner is a port of ``scripts/p08_step5_local/preconditioners.plane_ring_block``; the tests
compare it with that implementation on the synthetic ring operator of ``test_p08_step5_preconditioners`` (rings of
variable owner count, periodic angular coupling, weak plane coupling, random owner permutation).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from drbx.native.fci_perpendicular_plane_preconditioner import (                                      # noqa: E402
    PlanePreconditioner, apply_plane_preconditioner, build_plane_preconditioner, owner_layout)
from p08_step5_local import preconditioners as pc                                                     # noqa: E402
from tests.test_p08_step5_preconditioners import build_ring_operator                                  # noqa: E402


@pytest.fixture(scope="module")
def rworld():
    op, ring, plane, theta = build_ring_operator()
    return op, ring, plane, theta


def _rand(n, seed=3):
    return np.random.default_rng(seed).normal(size=n)


# ---------------------------------------------------------------------------
# owner_layout
# ---------------------------------------------------------------------------
def _cube_map(n):
    return np.arange(n ** 3, dtype=np.int64)


def test_owner_layout_identity_cube():
    n = 4
    ring, plane, theta = owner_layout(_cube_map(n), n)
    cell = np.arange(n ** 3)
    np.testing.assert_array_equal(ring, cell // n ** 2)
    np.testing.assert_array_equal(theta, (cell // n) % n)
    np.testing.assert_array_equal(plane, cell % n)


def test_owner_layout_merged_owners_and_missing_cells():
    n = 4
    cell = np.arange(n ** 3)
    jj = (cell // n) % n
    # merge pairs of theta cells (same ring, plane): owner of cell = (ring, j//2, plane); -1 for some cells
    ring_, plane_ = cell // n ** 2, cell % n
    owner = (ring_ * (n // 2) + jj // 2) * n + plane_
    raw = owner.copy()
    ring, plane, theta = owner_layout(raw, n)
    assert len(ring) == n * (n // 2) * n
    np.testing.assert_array_equal(ring[owner], ring_)
    np.testing.assert_array_equal(plane[owner], plane_)
    first = np.minimum.reduceat(jj[np.argsort(owner, kind="stable")], np.r_[0, np.flatnonzero(np.diff(np.sort(owner))) + 1])
    np.testing.assert_array_equal(theta, first)
    gap = raw.copy()
    gap[gap == 3] = -1                                      # owner 3 loses its only raw cell
    with pytest.raises(ValueError, match="no raw cell"):
        owner_layout(gap, n)


def test_owner_layout_validation():
    n = 3
    raw = _cube_map(n)
    with pytest.raises(ValueError, match="shape"):
        owner_layout(raw[:-1], n)
    bad_ring = raw.copy()
    bad_ring[26] = 0                                        # owner 0 gets a cell of ring 2 (owners 0..25 remain)
    with pytest.raises(ValueError, match="ring"):
        owner_layout(bad_ring, n)
    bad_plane = raw.copy()
    bad_plane[26] = 18                                      # same ring (2), plane 2 instead of 0
    with pytest.raises(ValueError, match="plane"):
        owner_layout(bad_plane, n)
    with pytest.raises(ValueError, match="no valid"):
        owner_layout(-np.ones(n ** 3, dtype=np.int64), n)


# ---------------------------------------------------------------------------
# equality with the scripts implementation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_matches_scripts_plane_ring_block(rworld, dtype):
    op, ring, plane, theta = rworld
    ref = pc.plane_ring_block(op, owner_ring=ring, owner_plane=plane, owner_theta=theta, factor_dtype=dtype)
    prec = build_plane_preconditioner(op.matrix, ring, plane, theta, factor_dtype=dtype)
    assert isinstance(prec, PlanePreconditioner)
    for key, leaf in (("lo", prec.lo), ("up", prec.up), ("dinv", prec.dinv), ("idx", prec.idx), ("pos", prec.pos)):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(ref.data[key]))
    r = _rand(op.matrix.shape[0])
    z = np.asarray(jax.jit(apply_plane_preconditioner)(prec, jnp.asarray(r)))
    zref = np.asarray(jax.jit(ref.apply)(jnp.asarray(r)))
    np.testing.assert_allclose(z, zref, rtol=1e-13 if dtype == "float64" else 1e-6,
                               atol=1e-14 * np.abs(zref).max() if dtype == "float64" else 1e-6 * np.abs(zref).max())
    assert prec.dtype == dtype and prec.S == ref.info["S"] and prec.B == ref.info["B"] and prec.w == ref.info["w"]
    info = prec.info
    assert info["storage_bytes"] == prec.nbytes == ref.info["storage_bytes"]
    assert info["max_block_cond1"] == pytest.approx(ref.info["max_block_cond1"], rel=1e-12)
    assert info["setup_seconds"] > 0 and info["n_owners"] == op.matrix.shape[0]


def test_exact_plane_block_solve(rworld):
    op, ring, plane, theta = rworld
    prec = build_plane_preconditioner(op.matrix, ring, plane, theta, factor_dtype="float64")
    a = op.matrix.toarray()
    r = _rand(a.shape[0], 9)
    ref = np.zeros_like(r)
    for k in np.unique(plane):
        idx = np.flatnonzero(plane == k)
        ref[idx] = np.linalg.solve(a[np.ix_(idx, idx)], r[idx])
    z = np.asarray(apply_plane_preconditioner(prec, jnp.asarray(r)))
    np.testing.assert_allclose(z, ref, rtol=1e-10, atol=1e-12 * np.abs(ref).max())


# ---------------------------------------------------------------------------
# jit: data as an argument, dtypes
# ---------------------------------------------------------------------------
def test_prec_is_a_pytree_and_factors_are_not_jaxpr_constants(rworld):
    op, ring, plane, theta = rworld
    prec = build_plane_preconditioner(op.matrix, ring, plane, theta)
    leaves, treedef = jax.tree_util.tree_flatten(prec)
    assert len(leaves) == 5 and all(hasattr(x, "shape") for x in leaves)
    again = jax.tree_util.tree_unflatten(treedef, leaves)
    assert again.meta == prec.meta and again.info == {}
    r = jnp.asarray(_rand(op.matrix.shape[0]))
    closed = jax.make_jaxpr(lambda d, v: apply_plane_preconditioner(d, v))(prec, r)
    const_bytes = sum(int(np.asarray(c).nbytes) for c in closed.consts)
    factor_bytes = sum(int(x.nbytes) for x in (prec.lo, prec.up, prec.dinv))
    assert const_bytes < 0.01 * factor_bytes
    # a closure over the same pytree does embed them (what the argument form avoids)
    closed_closure = jax.make_jaxpr(lambda v: apply_plane_preconditioner(prec, v))(r)
    assert sum(int(np.asarray(c).nbytes) for c in closed_closure.consts) >= factor_bytes
    # a second preconditioner of the same shape reuses the compiled function
    fn = jax.jit(apply_plane_preconditioner)
    z = fn(prec, r)
    n_cached = fn._cache_size()                             # the cache is shared by all jits of the function
    prec2 = build_plane_preconditioner(op.matrix * 2.0, ring, plane, theta)
    z2 = fn(prec2, r)
    np.testing.assert_allclose(np.asarray(z2), 0.5 * np.asarray(z), rtol=1e-5, atol=1e-6 * float(jnp.abs(z).max()))
    assert fn._cache_size() == n_cached


def test_float32_vs_float64(rworld):
    op, ring, plane, theta = rworld
    p64 = build_plane_preconditioner(op.matrix, ring, plane, theta, factor_dtype="float64")
    p32 = build_plane_preconditioner(op.matrix, ring, plane, theta, factor_dtype="float32")
    r = jnp.asarray(_rand(op.matrix.shape[0], 12))
    z64, z32 = apply_plane_preconditioner(p64, r), apply_plane_preconditioner(p32, r)
    assert z64.dtype == jnp.float64 and z32.dtype == jnp.float64
    assert p32.lo.dtype == jnp.float32 and p32.dinv.dtype == jnp.float32 and p32.idx.dtype == jnp.int32
    assert float(jnp.linalg.norm(z32 - z64) / jnp.linalg.norm(z64)) < 1e-5
    assert p32.nbytes < 0.55 * p64.nbytes


def test_validation(rworld):
    op, ring, plane, theta = rworld
    with pytest.raises(ValueError, match="factor_dtype"):
        build_plane_preconditioner(op.matrix, ring, plane, theta, factor_dtype="float16")
    with pytest.raises(ValueError, match="owner_ring"):
        build_plane_preconditioner(op.matrix, ring[:-1], plane, theta)
    with pytest.raises(ValueError, match="max_block"):
        build_plane_preconditioner(op.matrix, ring, plane, theta, max_block=2)
    with pytest.raises(ValueError, match="ill-conditioned|singular"):
        build_plane_preconditioner(op.matrix, ring, plane, theta, cond_max=1.0)
