"""Host-side tests of the nodal layout: validation, node bookkeeping, wall points, expansion to raw."""
from __future__ import annotations

import numpy as np
import pytest

from drbx.geometry.nodal_layout import (
    DenseBlock,
    RingLevelBlock,
    Side,
    build_nodal_layout,
    expand_to_raw,
    node_raw_ids,
    nodal_owner_layout_arrays,
    nodal_plane_layout,
    wall_points,
)
from drbx.geometry.sbp_operators import TWO_PI

N_GRID = 32
LEVELS = [(8, 16, 16), (16, 32, 32)]


def annulus(**kw):
    return build_nodal_layout(N_GRID, LEVELS, n_eta=8, inner="wall", **kw)


def fake_core(u_face, n_nodes=5, N=16):
    z = np.zeros((N, n_nodes))
    side = Side(N=N, T=z, TF1=z, TF2=z, u_face=u_face)
    inner = Side(N=N, T=z, TF1=z, TF2=z, u_face=0.0)
    return DenseBlock(D1=np.zeros((n_nodes, n_nodes)), D2=np.zeros((n_nodes, n_nodes)), wxy=np.full(n_nodes, 0.1),
                      u=np.full(n_nodes, 0.1), theta=np.linspace(0.0, 1.0, n_nodes),
                      sides={"inner": inner, "outer": side})


def test_annulus_layout_bookkeeping():
    lay = annulus()
    assert lay.P == 8 * 16 + 16 * 32 and lay.offsets == (0, 128)
    assert len(lay.blocks) == 2 and len(lay.faces) == 1
    face = lay.faces[0]
    assert (face.A, face.B, face.NA, face.NB, face.X) == ((0, "outer"), (1, "inner"), 16, 32, "B")
    assert [(w.block_idx, w.side, w.sign) for w in lay.walls] == [(1, "outer", 1), (0, "inner", -1)]
    assert lay.deta == pytest.approx(TWO_PI / 8)
    assert abs(lay.wxy.sum() - TWO_PI * (1.0 - 8 / N_GRID)) <= 1e-13
    blk = lay.blocks[0]
    assert isinstance(blk, RingLevelBlock)
    assert blk.sides["inner"].u_face == 8 / N_GRID and blk.sides["outer"].u_face == 16 / N_GRID
    assert blk.sides["outer"].rows == (blk.m - 3, blk.m - 2, blk.m - 1)
    assert blk.sides["inner"].rows == (0, 1, 2)
    # trace of a ring field: sum_r t_r g[r]
    rng = np.random.default_rng(0)
    g = rng.standard_normal((blk.m, blk.N))
    assert np.abs(blk.sides["outer"].T @ g.ravel() - blk.radial.tR @ g).max() <= 1e-14
    # node coordinates
    assert np.allclose(lay.node_u[:16], (8.5 / N_GRID))
    assert np.allclose(lay.node_theta[:16], np.pi / N_GRID + TWO_PI * np.arange(16) / 16)
    assert lay.node_ring[0] == 8 and lay.node_ring[-1] == N_GRID - 1


def test_core_layout_and_validation():
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=8, core=fake_core(8 / N_GRID))
    assert lay.P == 5 + 8 * 16 + 16 * 32 and lay.offsets == (0, 5, 5 + 128)
    assert [(f.A, f.B) for f in lay.faces] == [((0, "outer"), (1, "inner")), ((1, "outer"), (2, "inner"))]
    assert [(w.block_idx, w.sign) for w in lay.walls] == [(2, 1)]
    assert np.all(lay.node_ring[:5] == -1) and np.all(lay.node_raw_theta[:5] == -1)
    with pytest.raises(ValueError, match="u_face"):
        build_nodal_layout(N_GRID, LEVELS, core=fake_core(7 / N_GRID))
    with pytest.raises(ValueError, match="requires a core"):
        build_nodal_layout(N_GRID, LEVELS)
    with pytest.raises(ValueError, match="no core"):
        build_nodal_layout(N_GRID, LEVELS, core=fake_core(8 / N_GRID), inner="wall")


@pytest.mark.parametrize(
    "levels,match",
    [
        ([(8, 15, 16), (15, 32, 32)], "at least"),
        ([(8, 16, 12), (16, 32, 32)], "divide"),
        ([(8, 16, 16), (17, 32, 32)], "contiguous"),
        ([(8, 16, 16), (16, 30, 32)], "must end"),
        ([(0, 16, 16), (16, 32, 32)], "i0 > 0"),
        ([], "at least one"),
    ],
)
def test_layout_rejects_bad_levels(levels, match):
    with pytest.raises(ValueError, match=match):
        build_nodal_layout(N_GRID, levels, inner="wall")


def test_node_raw_ids_formula():
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=N_GRID, inner="wall")
    ids = node_raw_ids(lay)
    assert ids.shape == (N_GRID, lay.P) and ids.dtype == np.int64
    n = N_GRID
    for p in (0, 1, 15, 16, 127, 128, 129, 128 + 31, 128 + 32, lay.P - 1):
        i, N = lay.node_ring[p], 16 if p < 128 else 32
        j = (p - (0 if p < 128 else 128)) % N
        for k in (0, 5, n - 1):
            assert ids[k, p] == (i * n + j * (n // N)) * n + k
    assert np.unique(ids).size == ids.size
    # raw cell centre of the node
    i, j, k = ids // n**2, (ids // n) % n, ids % n
    assert np.allclose((i + 0.5) / n, lay.node_u[None, :])
    assert np.allclose(lay.delta + TWO_PI * j / n, lay.node_theta[None, :])
    core = build_nodal_layout(n, LEVELS, n_eta=n, core=fake_core(8 / n))
    cid = node_raw_ids(core)
    assert np.all(cid[:, :5] == -1) and np.all(cid[:, 5:] >= 0)
    with pytest.raises(ValueError):
        node_raw_ids(annulus())  # n_eta = 8 != n


def test_nodal_owner_layout_arrays_and_plane_layout():
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=8, core=fake_core(8 / N_GRID))
    ring, plane, theta = nodal_owner_layout_arrays(lay)
    E, P = lay.n_eta, lay.P
    assert ring.shape == plane.shape == theta.shape == (E * P,)
    assert np.array_equal(plane, np.repeat(np.arange(E), P))
    assert np.array_equal(ring[:P], lay.node_ring) and np.array_equal(ring[P:2 * P], lay.node_ring)
    assert np.all(ring[:5] == -1) and np.array_equal(theta[:5], np.arange(5))
    assert np.array_equal(theta[5:5 + 16], np.arange(16) * 2)
    assert np.array_equal(theta[5 + 128:5 + 128 + 32], np.arange(32))
    pl = nodal_plane_layout(lay)
    assert np.array_equal(pl.perm, np.arange(P)) and np.array_equal(pl.inverse, np.arange(P)) and pl.m == P


def test_wall_points():
    lay = annulus()
    outer, inner = lay.walls
    pts = wall_points(lay, outer)
    assert pts.shape == (8, 32, 3)
    assert np.allclose(pts[..., 0], 1.0)
    assert np.allclose(pts[0, :, 1], np.pi / N_GRID + TWO_PI * np.arange(32) / 32)
    assert np.allclose(pts[:, 0, 2], (np.arange(8) + 0.5) * TWO_PI / 8)
    pin = wall_points(lay, inner)
    assert pin.shape == (8, 16, 3) and np.allclose(pin[..., 0], 8 / N_GRID)


def test_expand_to_raw_band_limited_and_identity():
    lay = annulus()
    n, E = lay.n, lay.n_eta
    eta = (np.arange(E) + 0.5) * lay.deta

    def field(u, th, et):
        return u * np.cos(3 * th) + np.sin(th - 2 * et) * (1 + u)

    vals = field(lay.node_u[None, :], lay.node_theta[None, :], eta[:, None])
    raw = expand_to_raw(lay, vals)
    assert raw.shape == (n, n, E)
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(E), indexing="ij")
    exact = field((ii + 0.5) / n, lay.delta + TWO_PI * jj / n, eta[kk])
    assert np.all(np.isnan(raw[:8]))
    assert np.abs(raw[8:] - exact[8:]).max() <= 1e-13
    # trailing component axis, and identity where N == n
    vec = np.stack([vals, 2 * vals], axis=-1)
    rawv = expand_to_raw(lay, vec)
    assert rawv.shape == (n, n, E, 2)
    assert np.array_equal(rawv[16:, :, :, 0], raw[16:]) and np.abs(rawv[8:, :, :, 1] - 2 * raw[8:]).max() <= 1e-13
    outer = lay.blocks[1]
    assert np.array_equal(raw[16:, :, 3], vals[3, lay.offsets[1]:].reshape(outer.m, outer.N))
    with pytest.raises(ValueError):
        expand_to_raw(lay, vals[:, :-1])


def test_expand_to_raw_core_evaluate_at():
    base = fake_core(8 / N_GRID)
    calls = []

    def evaluate_at(values, u, theta):
        calls.append((values.shape, u.shape))
        return np.full((values.shape[0], u.size), 7.0)

    core = DenseBlock(base.D1, base.D2, base.wxy, base.u, base.theta, base.sides, evaluate_at=evaluate_at)
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=8, core=core)
    raw = expand_to_raw(lay, np.ones((8, lay.P)))
    assert calls == [((8, 5), (8 * N_GRID,))]
    assert np.all(raw[:8] == 7.0) and np.all(raw[8:] == pytest.approx(1.0, abs=1e-13))
