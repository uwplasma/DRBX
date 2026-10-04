"""Host tests of the nodal plan, metric gathering, wall data, norms and region tables (algebra only)."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry.nodal_layout import DenseBlock, Side, build_nodal_layout, node_raw_ids, wall_points
from drbx.geometry.sbp_operators import TWO_PI, ring_basis
from drbx.native.fci_perpendicular_sbp_boundary import (
    SatBoundaryData,
    sat_boundary_data_from_callables,
    zero_sat_boundary_data,
)
from drbx.native.fci_perpendicular_sbp_norms import h_inner, h_norm, h_weights, region_errors, ring_region_masks
from drbx.stencils.nodal_plan import (
    NodalMetric,
    NodalMetricProvider,
    NodalPlan,
    build_nodal_plan,
    gather_ring_metric,
    level_d5c_maps,
    load_nodal_plan,
    metric_from_geometry_provider,
    nodal_metric_from_callable,
    plan_identity,
    raw_id_lookup,
    save_nodal_plan,
)

N_GRID = 32
LEVELS = [(8, 16, 16), (16, 32, 32)]
DATA = Path(__file__).parent / "data" / "p09_nodal_metric_n32_excerpt.npz"
FULL_EXTRACT = Path(os.environ.get(
    "DRBX_P09_N32_EXTRACT",
    Path(__file__).resolve().parents[2] / "work" / "p09_optionB_20261003" / "hsx_n32_extract.npz"))


def smooth_metric(points):
    u, th, et = points.T
    h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], axis=-1)
    jac = u * (1.0 + 0.1 * np.cos(th) * np.cos(et))
    return h, jac, 1.0 + 0.3 * u, np.stack([0.1 * u, np.cos(th), 0.2 * np.sin(et)], axis=-1)


def annulus(levels=LEVELS, n_eta=8, **kw):
    return build_nodal_layout(N_GRID, levels, n_eta=n_eta, inner="wall", **kw)


def fake_core(u_face, n_nodes=5, N=16):
    z = np.zeros((N, n_nodes))
    outer = Side(N=N, T=z + 0.5, TF1=z + 0.25, TF2=z, u_face=u_face)
    inner = Side(N=N, T=z, TF1=z, TF2=z, u_face=0.0)
    return DenseBlock(D1=np.eye(n_nodes), D2=2 * np.eye(n_nodes), wxy=np.full(n_nodes, 0.1), u=np.full(n_nodes, 0.1),
                      theta=np.linspace(0.0, 1.0, n_nodes), sides={"inner": inner, "outer": outer})


# ------------------------------------------------------------------------------------------------ plan
def test_plan_construction():
    lay = annulus()
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, smooth_metric))
    E, P = 8, lay.P
    s = plan.structure
    assert (s.n, s.n_eta, s.P) == (N_GRID, E, P)
    assert s.blocks == (("ring", 0, 128, 8, 16, 8), ("ring", 128, 512, 16, 32, 16))
    assert s.faces == ((0, 1, 16, 32, "B"),)
    assert s.walls == ((1, "outer", 1, 32), (0, "inner", -1, 16))
    assert s.side_rows[0] == (0, "inner", (0, 1, 2)) and s.side_rows[1] == (0, "outer", (5, 6, 7))
    hash(s)
    assert plan.wxy.shape == (P,) and plan.h.shape == (E, P, 3) and plan.K.shape == (E, P, 3)
    assert plan.jac.shape == plan.B.shape == plan.Hp.shape == (E, P)
    assert np.array_equal(plan.Hp, plan.wxy[None, :] * plan.jac)
    assert np.allclose(np.asarray(h_weights(plan)), plan.Hp * lay.deta)
    blk = plan.blocks[0]
    assert blk.Du.shape == (8, 8) and blk.Dth.shape == (16, 16) and blk.w.shape == (8,)
    assert np.array_equal(blk.Du, lay.blocks[0].radial.D_unit * N_GRID)
    face = plan.faces[0]
    assert face.Iab.shape == (32, 16) and face.Iba.shape == (16, 32)
    assert (face.CAA.shape, face.CAB.shape, face.CBA.shape, face.CBB.shape) == ((16, 16), (16, 32), (32, 16), (32, 32))
    assert np.array_equal(face.Iab, lay.faces[0].Iab)


def test_plan_rejects_bad_metric():
    lay = annulus()
    good = nodal_metric_from_callable(lay, smooth_metric)
    with pytest.raises(ValueError, match="positive"):
        build_nodal_plan(lay, good._replace(jac=good.jac.copy() * np.where(np.arange(lay.P) == 3, -1.0, 1.0)))
    bad = good.jac.copy()
    bad[2, 5] = 0.0
    with pytest.raises(ValueError, match="positive"):
        build_nodal_plan(lay, good._replace(jac=bad))
    bad = good.h.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_nodal_plan(lay, good._replace(h=bad))
    with pytest.raises(ValueError, match="shape"):
        build_nodal_plan(lay, good._replace(B=good.B[:, :-1]))


def test_plan_with_dense_core_block():
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=8, core=fake_core(8 / N_GRID))
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, smooth_metric))
    assert plan.structure.blocks[0] == ("dense", 0, 5, 0, 0, -1)
    assert plan.faces[0].CAA is None and plan.faces[1].CAA is not None
    assert plan.blocks[0].outer.T.shape == (16, 5)
    assert np.array_equal(plan.blocks[0].D2, 2 * np.eye(5))


def test_plan_pytree_round_trip_and_jit():
    lay = annulus()
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, smooth_metric))
    leaves, treedef = jax.tree_util.tree_flatten(plan)
    assert all(isinstance(x, np.ndarray) for x in leaves)
    again = jax.tree_util.tree_unflatten(treedef, leaves)
    assert again.structure == plan.structure and plan_identity(again) == plan_identity(plan)
    names = {f.name for f in dataclasses.fields(NodalPlan)}
    assert names == {"wxy", "jac", "h", "B", "K", "Hp", "blocks", "faces", "structure"}
    traces = []

    @jax.jit
    def total(p):
        traces.append(1)
        return jnp.sum(p.Hp) * p.structure.deta

    expected = np.sum(plan.Hp) * lay.deta
    assert float(total(plan)) == pytest.approx(expected, rel=1e-14)
    scaled = jax.tree_util.tree_map(lambda x: 2.0 * x, plan)
    assert float(total(scaled)) == pytest.approx(2 * expected, rel=1e-14)
    assert len(traces) == 1


def test_plan_save_load_identity(tmp_path):
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=8, core=fake_core(8 / N_GRID))
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, smooth_metric))
    path = tmp_path / "plan.npz"
    identity = save_nodal_plan(plan, path)
    assert identity == plan_identity(plan) and len(identity) == 64
    loaded = load_nodal_plan(path, expected_identity=identity)
    assert loaded.structure == plan.structure and plan_identity(loaded) == identity
    for a, b in zip(jax.tree_util.tree_leaves(plan), jax.tree_util.tree_leaves(loaded)):
        assert np.array_equal(a, b)
    assert loaded.faces[0].CAA is None
    with pytest.raises(ValueError, match="expected"):
        load_nodal_plan(path, expected_identity="0" * 64)
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files}
    payload["jac"] = payload["jac"] * (1.0 + 1e-9)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **payload)
    with pytest.raises(ValueError, match="identity"):
        load_nodal_plan(bad)
    other = build_nodal_plan(lay, nodal_metric_from_callable(lay, lambda p: (lambda r: (r[0], 2 * r[1], r[2], r[3]))(smooth_metric(p))))
    assert plan_identity(other) != identity


# ------------------------------------------------------------------------------------------------ D5c maps
def prototype_level_trace_matched(layout, phi):
    """Port of the prototype's level D5c formula (``Grid2.level_trace_matched``, level-level faces) on one plane."""
    out = phi.copy()

    def rows(blk, side):
        t = blk.radial.tL if side == "inner" else blk.radial.tR
        r = np.flatnonzero(t)
        return [blk_off[blk] + q * blk.N + np.arange(blk.N) for q in r], t[r]

    blk_off = {b: o for b, o in zip(layout.blocks, layout.offsets)}
    for f in layout.faces:
        sides = []
        for blk_idx, name in (f.A, f.B):
            blk = layout.blocks[blk_idx]
            N = blk.N
            B, Binv, keys, om = ring_basis(N, layout.delta)
            idx, w = rows(blk, name)
            tr = sum(wr * phi[ix] for ix, wr in zip(idx, w))
            sides.append((N, B, Binv, keys, idx, w, Binv @ tr))
        (NA, BA, BiA, kA, iA, wA, aA), (NB, BB, BiB, kB, iB, wB, aB) = sides
        posB = {k: q for q, k in enumerate(kB)}
        dA, dB = np.zeros_like(aA), np.zeros_like(aB)
        for q, k in enumerate(kA):
            if k[1] == "n" or k not in posB:
                continue
            tgt = 0.5 * (aA[q] + aB[posB[k]])
            dA[q] = tgt - aA[q]
            dB[posB[k]] = tgt - aB[posB[k]]
        for (B, d, idx, w) in ((BA, dA, iA, wA), (BB, dB, iB, wB)):
            corr = B @ d / np.sum(w * w)
            for ix, wr in zip(idx, w):
                out[ix] += wr * corr
    return out


def plan_level_trace_matched(layout, plan, phi):
    """The same correction applied through the plan's D5c maps."""
    out = phi.copy()
    for face, arr in zip(layout.faces, plan.faces):
        blkA, blkB = layout.blocks[face.A[0]], layout.blocks[face.B[0]]
        a = blkA.sides["outer"].T @ phi[layout.offsets[face.A[0]]:][:blkA.n_nodes]
        b = blkB.sides["inner"].T @ phi[layout.offsets[face.B[0]]:][:blkB.n_nodes]
        dA = arr.CAA @ a + arr.CAB @ b
        dB = arr.CBA @ a + arr.CBB @ b
        for blk_idx, name, d in ((face.A[0], "outer", dA), (face.B[0], "inner", dB)):
            blk = layout.blocks[blk_idx]
            t = blk.radial.tR if name == "outer" else blk.radial.tL
            for q in np.flatnonzero(t):
                sl = layout.offsets[blk_idx] + q * blk.N
                out[sl:sl + blk.N] += t[q] * d / np.sum(t * t)
    return out


@pytest.mark.parametrize("levels", [[(2, 10, 8), (10, 20, 16), (20, 32, 32)], [(4, 12, 32), (12, 24, 16), (24, 32, 8)],
                                    [(4, 16, 16), (16, 32, 16)]])
def test_level_d5c_maps_reproduce_prototype_formula(levels):
    lay = annulus(levels, n_eta=4)
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, smooth_metric))
    rng = np.random.default_rng(11)
    for _ in range(3):
        phi = rng.standard_normal(lay.P)
        ref = prototype_level_trace_matched(lay, phi)
        got = plan_level_trace_matched(lay, plan, phi)
        assert np.abs(got - ref).max() <= 1e-13 * np.abs(phi).max()
    # after matching, the common non-Nyquist trace modes of the two sides agree
    phi_t = plan_level_trace_matched(lay, plan, phi)
    for f in lay.faces:
        blkA, blkB = lay.blocks[f.A[0]], lay.blocks[f.B[0]]
        aA = ring_basis(f.NA, lay.delta)[1] @ (blkA.sides["outer"].T @ phi_t[lay.offsets[f.A[0]]:][:blkA.n_nodes])
        aB = ring_basis(f.NB, lay.delta)[1] @ (blkB.sides["inner"].T @ phi_t[lay.offsets[f.B[0]]:][:blkB.n_nodes])
        kA, kB = ring_basis(f.NA, lay.delta)[2], ring_basis(f.NB, lay.delta)[2]
        posB = {k: i for i, k in enumerate(kB)}
        for q, k in enumerate(kA):
            if k[1] != "n" and k in posB:
                assert abs(aA[q] - aB[posB[k]]) <= 1e-12


def test_level_d5c_maps_are_projections_on_common_modes():
    CAA, CAB, CBA, CBB = level_d5c_maps(16, 32, np.pi / 32)
    # equal traces need no correction
    th_a, th_b = np.pi / 32 + TWO_PI * np.arange(16) / 16, np.pi / 32 + TWO_PI * np.arange(32) / 32
    a, b = np.cos(3 * th_a) + 0.5, np.cos(3 * th_b) + 0.5
    assert np.abs(CAA @ a + CAB @ b).max() <= 1e-13 and np.abs(CBA @ a + CBB @ b).max() <= 1e-13
    # a mode only the finer side carries is not touched
    b_hi = np.cos(12 * th_b)
    assert np.abs(CBB @ b_hi).max() <= 1e-13 and np.abs(CAB @ b_hi).max() <= 1e-13


# ------------------------------------------------------------------------------------------------ metric gathering
def synthetic_raw(n=N_GRID):
    ids = np.arange(n**3)
    i, j, k = ids // n**2, (ids // n) % n, ids % n
    pts = np.stack([(i + 0.5) / n, np.pi / n + TWO_PI * j / n, (k + 0.5) * TWO_PI / n], axis=-1)
    h, jac, B, K = smooth_metric(pts)
    return ids, h, jac, B, K, pts


def test_gather_ring_metric_synthetic_raw_arrays():
    lay = build_nodal_layout(N_GRID, LEVELS, inner="wall")
    ids, h, jac, B, K, pts = synthetic_raw()
    perm = np.random.default_rng(0).permutation(ids.size)
    m = gather_ring_metric(lay, ids[perm], h[perm], jac[perm], B[perm], K[perm], pts[perm])
    direct = nodal_metric_from_callable(lay, smooth_metric)
    for a, b in zip(m, direct):
        assert np.abs(a - b).max() <= 1e-14
    raw = node_raw_ids(lay)
    assert np.array_equal(m.h, h[raw]) and np.array_equal(m.jac, jac[raw])
    # core nodes are NaN
    core_lay = build_nodal_layout(N_GRID, LEVELS, n_eta=N_GRID, core=fake_core(8 / N_GRID))
    mc = gather_ring_metric(core_lay, ids, h, jac, B, K)
    assert np.all(np.isnan(mc.h[:, :5])) and np.all(np.isfinite(mc.h[:, 5:]))
    with pytest.raises(ValueError, match="finite"):
        build_nodal_plan(core_lay, mc)
    # mismatched points and missing ids
    with pytest.raises(ValueError, match="raw_points"):
        gather_ring_metric(lay, ids, h, jac, B, K, pts + 1e-6)
    with pytest.raises(KeyError, match="raw ids"):
        gather_ring_metric(lay, ids[:-1000], h[:-1000], jac[:-1000], B[:-1000], K[:-1000])


def test_hsx_metric_excerpt_gather():
    z = np.load(DATA)
    assert int(z["n"]) == 32 and z["raw_ids"].size == 6 * 32 * 4
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=N_GRID, inner="wall")
    planes = np.array([0, 1, 2, 31])
    ring = np.flatnonzero(np.isin(lay.node_ring, [8, 9, 15, 16, 30, 31]))
    ids = node_raw_ids(lay)[np.ix_(planes, ring)]
    idx = raw_id_lookup(z["raw_ids"], ids.ravel())
    assert np.array_equal(z["raw_ids"][idx], ids.ravel())
    for key, tail in (("h", (3,)), ("jac", ()), ("B", ()), ("K", (3,))):
        got = z[key][idx].reshape((planes.size, ring.size) + tail)
        assert np.all(np.isfinite(got))
    assert np.all(z["jac"] > 0)
    pts = z["pts"][idx].reshape(planes.size, ring.size, 3)
    eta = (planes + 0.5) * TWO_PI / N_GRID
    assert np.abs(pts[..., 0] - lay.node_u[ring][None, :]).max() <= 1e-13
    assert np.abs(pts[..., 1] - lay.node_theta[ring][None, :]).max() <= 1e-13
    assert np.abs(pts[..., 2] - eta[:, None]).max() <= 1e-13
    with pytest.raises(KeyError):
        raw_id_lookup(z["raw_ids"], np.array([(10 * N_GRID + 0) * N_GRID + 0]))
    with pytest.raises(KeyError):
        gather_ring_metric(lay, z["raw_ids"], z["h"], z["jac"], z["B"], z["K"], z["pts"])


@pytest.mark.skipif(not FULL_EXTRACT.exists(), reason="full n=32 HSX extract not present")
def test_hsx_full_extract_gather_bitwise():
    z = np.load(FULL_EXTRACT)
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=N_GRID, inner="wall")
    m = gather_ring_metric(lay, z["raw_ids"], z["h"], np.abs(z["jac"]), z["B"], z["K"], z["pts"])
    raw = node_raw_ids(lay)
    assert np.array_equal(m.h, z["h"][raw]) and np.array_equal(m.jac, np.abs(z["jac"])[raw])
    assert np.array_equal(m.B, z["B"][raw]) and np.array_equal(m.K, z["K"][raw])
    plan = build_nodal_plan(lay, m)
    assert plan.Hp.shape == (N_GRID, lay.P)


# ------------------------------------------------------------------------------------------------ providers
class FakeProvider:
    def p05_metric(self, points):
        q = len(points)
        return np.tile([0.0, 0.0, 1.0], (q, 1)), -np.arange(1.0, q + 1)

    def p06_curvature(self, points):
        q = len(points)
        return -np.ones(q), 2.0 * np.arange(q), np.tile([1.0, 2.0, 3.0], (q, 1))


def test_metric_from_geometry_provider_takes_abs_jacobian():
    adapter = metric_from_geometry_provider(FakeProvider())
    assert isinstance(adapter, NodalMetricProvider)
    pts = np.random.default_rng(0).random((5, 3))
    h, jac, B, K = adapter.nodal_metric(pts)
    assert np.array_equal(jac, np.arange(1.0, 6.0)) and np.array_equal(B, 2.0 * np.arange(5))
    assert np.array_equal(K, np.tile([1.0, 2.0, 3.0], (5, 1))) and h.shape == (5, 3)
    lay = annulus(n_eta=4)
    metric = nodal_metric_from_callable(lay, adapter.nodal_metric)
    assert metric.jac.shape == (4, lay.P) and np.all(metric.jac > 0)
    assert np.array_equal(metric.jac.ravel(), np.arange(1.0, 4 * lay.P + 1))


# ------------------------------------------------------------------------------------------------ boundary data
def test_sat_boundary_data_from_callables_once_per_wall():
    lay = annulus()
    calls, ncalls = [], []

    def dirichlet(points):
        calls.append(points.copy())
        q = len(points)
        return np.stack([points[:, 1], points[:, 2]], axis=-1), np.zeros((q, 3, 2))

    def normal(points):
        ncalls.append(len(points))
        return points[:, :1] * np.ones((1, 2))

    bc = sat_boundary_data_from_callables(lay, dirichlet=dirichlet, normal=normal)
    assert isinstance(bc, SatBoundaryData) and len(calls) == len(lay.walls) == 2 and len(ncalls) == 2
    for wall, pts, val, nrm in zip(lay.walls, calls, bc.value, bc.normal_derivative):
        wp = wall_points(lay, wall)
        assert np.array_equal(pts, wp.reshape(-1, 3))
        assert val.shape == (8, wp.shape[1], 2) == nrm.shape
        assert np.array_equal(np.asarray(val[..., 0]), wp[..., 1]) and np.array_equal(np.asarray(val[..., 1]), wp[..., 2])
        assert np.all(np.asarray(nrm) == wp[0, 0, 0])
    none = sat_boundary_data_from_callables(lay)
    assert none.value is None and none.normal_derivative is None
    only = sat_boundary_data_from_callables(lay, dirichlet=dirichlet)
    assert only.value is not None and only.normal_derivative is None


def test_zero_sat_boundary_data():
    lay = annulus()
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, smooth_metric))
    bc = zero_sat_boundary_data(plan, 3)
    assert [v.shape for v in bc.value] == [(8, 32, 3), (8, 16, 3)] and bc.normal_derivative is None
    assert all(float(jnp.abs(v).max()) == 0.0 for v in bc.value)


# ------------------------------------------------------------------------------------------------ norms and regions
def test_norms_and_region_errors():
    lay = annulus()
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, smooth_metric))
    H = h_weights(plan)
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((8, lay.P)), rng.standard_normal((8, lay.P))
    Hn = np.asarray(H)
    assert float(h_inner(a, b, H)) == pytest.approx(np.sum(Hn * a * b), rel=1e-13)
    assert float(h_norm(a, H)) == pytest.approx(np.sqrt(np.sum(Hn * a * a)), rel=1e-13)
    f = rng.standard_normal((8, lay.P, 2))
    assert float(h_inner(f, f, H)) == pytest.approx(np.sum(Hn[..., None] * f * f), rel=1e-13)
    masks = ring_region_masks(lay, 8)
    assert {"interior", "wall", "level_bands", "inner_wall", "all"} <= set(masks) and "core" not in masks
    r = np.broadcast_to(lay.node_ring[None, :], (8, lay.P))
    assert np.array_equal(masks["wall"], r >= 28)
    assert np.array_equal(masks["level_bands"], (r >= 12) & (r <= 19))
    assert not np.any(masks["interior"] & (masks["wall"] | masks["level_bands"] | masks["inner_wall"]))
    err = np.where(masks["wall"], 1e-3, 1e-6) * a
    ref = np.ones_like(a)
    out = region_errors(err, ref, Hn, masks)
    assert out["all"]["volume"] == pytest.approx(Hn.sum(), rel=1e-13)
    assert out["all"]["share"] == pytest.approx(1.0) and 0.99 < out["wall"]["share"] <= 1.0
    assert out["wall"]["volume"] == pytest.approx(Hn[masks["wall"]].sum(), rel=1e-13)
    assert out["wall"]["max"] == pytest.approx(np.abs(err[masks["wall"]]).max())
    assert out["all"]["rel_region"] == pytest.approx(np.sqrt(np.sum(Hn * err**2) / Hn.sum()), rel=1e-12)
    core_lay = build_nodal_layout(N_GRID, LEVELS, n_eta=4, core=fake_core(8 / N_GRID))
    cm = ring_region_masks(core_lay, 4)
    assert cm["core"][:, :5].all() and cm["core_band"].any() and "inner_wall" not in cm
