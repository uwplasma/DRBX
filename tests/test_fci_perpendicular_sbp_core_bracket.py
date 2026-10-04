"""Nodal SBP bracket with the Zernike core (family A): energy identity, constants, polynomial bracket, dissipation with
core shell damping, plan and metric round trips, probing assembly and JAX hygiene."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry.nodal_families import build_family_a_layout
from drbx.geometry.sbp_core import shell_projector
from drbx.native import fci_perpendicular_sbp_bracket as br
from drbx.native import fci_perpendicular_sbp_dissipation as dis
from drbx.native import fci_perpendicular_sbp_ops as ops
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData, sat_boundary_data_from_callables
from drbx.stencils.nodal_plan import (
    CoreBlockArrays,
    build_nodal_plan,
    load_nodal_metric,
    load_nodal_plan,
    nodal_metric_from_callable,
    plan_identity,
    save_nodal_metric,
    save_nodal_plan,
)
from drbx.validation.sbp_audit import assemble_by_probing, jax_linear_operator, rk4_dt

TWO_PI = 2.0 * np.pi
RHO = 0.05


def logical_metric(kind="straight"):
    def fn(p):
        u, th, et = p.T
        if kind == "straight":
            return (np.zeros((len(u), 3)) + [0.0, 0.0, 1.0], u, 1.0 + 0 * u, np.zeros((len(u), 3)))
        h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
        K = np.stack([0.3 * np.cos(th), 0.1 * u, 0.2 * np.sin(et)], -1)
        return h, u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), 1.0 + 0.2 * u, K

    return fn


def make(n, n_eta=4, kind="straight"):
    lay = build_family_a_layout(n, n_eta=n_eta)
    return lay, build_nodal_plan(lay, nodal_metric_from_callable(lay, logical_metric(kind)))


def hsum(plan, a, b):
    Hp = np.asarray(plan.Hp)
    return float(np.sum(Hp * np.asarray(a) * np.asarray(b)))


# ------------------------------------------------------------------------------------------------ C10
@pytest.mark.parametrize("n,tol", [(16, 1e-13), (32, 1e-13), (64, 1e-12)])
def test_energy_identity_with_core(n, tol):
    lay, plan = make(n, kind="smooth")
    rng = np.random.default_rng(n)
    F = rng.standard_normal((4, lay.P, 3))
    g = rng.standard_normal((4, lay.P))
    Tg = np.asarray(jax.jit(br.transport)(plan, F, g))
    A = np.asarray(jax.jit(lambda p, F_, g_: br.advect(p, *(ops.extend_periodic(x, 3) for x in (F_, g_))))(plan, F, g))
    scale = np.sqrt(hsum(plan, g, g)) * np.sqrt(hsum(plan, A, A))
    wall = 0.0
    for blk, side, sign, N in plan.structure.walls:
        a = np.asarray(ops.trace(plan, blk, side, g))
        v = np.asarray(ops.flux_trace(plan, blk, side, F[..., 0], F[..., 1]))
        wall += -0.5 * (TWO_PI / N) * sign * np.sum(v * a * a)
    assert len(plan.structure.walls) == 1
    assert abs(hsum(plan, g, Tg) - wall) / scale <= tol


def test_constants_preserved_and_polynomial_bracket_with_core():
    flux, lin = jax.jit(br.velocity_flux), jax.jit(br.linear_operator)
    for n in (16, 32):
        lay, plan = make(n)
        E = lay.n_eta
        rng = np.random.default_rng(n)
        F = rng.standard_normal((E, lay.P, 3))
        ones = np.ones((E, lay.P, 1))
        unit = SatBoundaryData((jnp.ones((E, plan.structure.walls[0][3], 1)),), None)
        L1 = np.asarray(lin(plan, F, ones, unit))
        assert np.abs(L1).max() <= 1e-11 * np.abs(F).max() / np.asarray(plan.Hp).min()
        # phi = x, g = y with h = (0, 0, 1), |J| = u: L g = -1 / rho* on the core and the rings
        u, th = lay.node_u, lay.node_theta
        phi = np.broadcast_to(u * np.cos(th), (E, lay.P)).copy()
        g = np.broadcast_to((u * np.sin(th))[None, :, None], (E, lay.P, 1)).copy()

        def dirichlet(points):
            return (points[:, 0] * np.sin(points[:, 1]))[:, None], np.zeros((len(points), 3, 1))

        bcd = sat_boundary_data_from_callables(lay, dirichlet=dirichlet)
        F = flux(plan, phi, RHO)
        L = np.asarray(lin(plan, F, g, bcd))
        assert np.abs(L + 1.0 / RHO).max() <= 1e-11 / RHO
        core = slice(0, lay.blocks[0].n_nodes)
        assert np.abs(L[:, core] + 1.0 / RHO).max() <= 1e-11 / RHO


def test_core_d5c_flux_uses_core_gradient_not_difference_operator():
    """With the polynomial phi = x the core flux is exactly the Cartesian one: F = (0, 1, 0) / rho*."""
    lay, plan = make(16)
    phi = np.broadcast_to(lay.node_u * np.cos(lay.node_theta), (lay.n_eta, lay.P)).copy()
    F = np.asarray(br.velocity_flux(plan, phi, RHO))
    n_c = lay.blocks[0].n_nodes
    assert np.abs(F[:, :n_c, 0]).max() <= 1e-10 / RHO and np.abs(F[:, :n_c, 1] - 1.0 / RHO).max() <= 1e-10 / RHO


# ------------------------------------------------------------------------------------------------ C11
def test_dissipation_with_core_damping_is_negative_semidefinite_and_c_kappa_is_traced():
    lay, plan = make(32, kind="smooth")
    E, P = lay.n_eta, lay.P
    rng = np.random.default_rng(1)
    F = rng.standard_normal((E, P, 3))
    H = np.asarray(plan.Hp) * plan.structure.deta
    traces = []

    def diss(p, F_, g_, ck):
        traces.append(1)
        return dis.dissipation(p, F_, g_, ck)

    jd = jax.jit(diss)
    for _ in range(50):
        x = rng.standard_normal((E, P))
        Dx = np.asarray(jd(plan, F, x, 1.0))
        assert np.sum(H * x * Dx) <= 1e-14 * np.sqrt(np.sum(H * x * x) * np.sum(H * Dx * Dx))
    one = np.asarray(jd(plan, F, np.ones((E, P)), 0.3))
    assert len(traces) == 1                                           # second c_kappa value: no retrace
    kappa_scale = 0.3 * plan.structure.core[1] / plan.structure.core[2] * np.abs(F).max() / np.asarray(plan.jac).min()
    assert np.abs(one).max() <= 1e-11 * max(1.0, kappa_scale)
    # the core damping alone: only on the core nodes, -kappa P_h g, kappa from max |V_xy|
    n_c = plan.structure.core[3]
    Fc = np.zeros((E, P, 3))
    Fc[:, :n_c, 0] = 3.0 * np.asarray(plan.jac)[:, :n_c]
    Fc[:, :n_c, 1] = 4.0 * np.asarray(plan.jac)[:, :n_c]
    g = rng.standard_normal((E, P))
    out = np.asarray(dis.core_damping(plan, Fc, g, 0.5))
    assert np.abs(out[:, n_c:]).max() == 0.0
    kappa = 0.5 * 5.0 * plan.structure.core[1] / plan.structure.core[2]
    Vm = plan.blocks[0].Vm
    for k in range(E):
        Hk = np.asarray(plan.wxy)[:n_c] * np.asarray(plan.jac)[k, :n_c]
        ref = -kappa * (g[k, :n_c] - shell_projector(Vm, Hk) @ g[k, :n_c])
        assert np.abs(out[k, :n_c] - ref).max() <= 1e-10 * np.abs(ref).max()
    assert np.abs(np.asarray(dis.core_damping(plan, Fc, g, 0.0))).max() == 0.0
    # no core: zero
    from drbx.geometry.nodal_layout import build_nodal_layout
    lay2 = build_nodal_layout(32, [(8, 16, 16), (16, 32, 32)], n_eta=4, inner="wall")
    plan2 = build_nodal_plan(lay2, nodal_metric_from_callable(lay2, logical_metric("smooth")))
    assert np.abs(np.asarray(dis.core_damping(plan2, np.ones((4, lay2.P, 3)), np.ones((4, lay2.P))))).max() == 0.0


# ------------------------------------------------------------------------------------------------ C12
def test_plan_v2_round_trip_with_core_and_v1_refused(tmp_path):
    lay, plan = make(16, kind="smooth")
    s = plan.structure
    assert s.core == (0, 4, 2 / 16, 30, 10, 1) and s.blocks[0][0] == "core"
    assert isinstance(plan.blocks[0], CoreBlockArrays) and plan.blocks[0].inner is None
    assert plan.core_Ginv.shape == (4, 10, 10)
    assert s.side_rows[0] == (0, "outer", ()) and s.side_rows[1][:2] == (1, "inner")
    path = tmp_path / "plan.npz"
    ident = save_nodal_plan(plan, path)
    assert ident == plan_identity(plan)
    loaded = load_nodal_plan(path, expected_identity=ident)
    assert loaded.structure == s and plan_identity(loaded) == ident
    for a, b in zip(jax.tree_util.tree_leaves(plan), jax.tree_util.tree_leaves(loaded)):
        assert np.array_equal(a, b)
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema"] = np.array("drbx.nodal-plan.v1")
    old = tmp_path / "old.npz"
    np.savez(old, **payload)
    with pytest.raises(ValueError, match="schema"):
        load_nodal_plan(old)
    # the core's block-frame metric: jac / u, h and K transformed
    core = lay.blocks[0]
    h_log, jac_log, _B, K_log = nodal_metric_from_callable(lay, logical_metric("smooth"))
    hb, jb = core.to_block_frame(h_log[:, :30], jac_log[:, :30])
    assert np.array_equal(plan.h[:, :30], hb) and np.array_equal(plan.jac[:, :30], jb)
    assert np.array_equal(plan.K[:, :30], core.to_block_frame_K(K_log[:, :30])) and np.array_equal(plan.K[:, 30:], K_log[:, 30:])


def test_nodal_metric_save_load_and_point_check(tmp_path):
    lay, _plan = make(16)
    metric = nodal_metric_from_callable(lay, logical_metric("smooth"))
    E, P = lay.n_eta, lay.P
    eta = (np.arange(E) + 0.5) * lay.deta
    points = np.stack([np.broadcast_to(lay.node_u, (E, P)), np.broadcast_to(lay.node_theta, (E, P)),
                       np.broadcast_to(eta[:, None], (E, P))], -1)
    meta = {"N": 16, "K": 2, "p": 4, "provider": "synthetic"}
    path = tmp_path / "N16" / "nodal_metric.npz"
    ident = save_nodal_metric(path, metric, points, meta)
    assert ident == save_nodal_metric(path, metric, points, meta)
    loaded, meta2 = load_nodal_metric(path, lay, expected_identity=ident)
    assert meta2 == meta
    for a, b in zip(loaded, metric):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="expected"):
        load_nodal_metric(path, lay, expected_identity="0" * 64)
    save_nodal_metric(tmp_path / "bad.npz", metric, points + 1e-6, meta)
    with pytest.raises(ValueError, match="nodes"):
        load_nodal_metric(tmp_path / "bad.npz", lay)
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files}
    payload["jac"] = payload["jac"] * (1 + 1e-9)
    np.savez(tmp_path / "tamper.npz", **payload)
    with pytest.raises(ValueError, match="identity"):
        load_nodal_metric(tmp_path / "tamper.npz", lay)


# ------------------------------------------------------------------------------------------------ C15
def test_probing_and_matrix_free_operator_on_family_a():
    lay, plan = make(16, n_eta=4, kind="smooth")      # n_eta = 4 keeps the dense reference cheap (spec: 8)
    E, P = lay.n_eta, lay.P
    phi = np.broadcast_to(lay.node_u * np.cos(lay.node_theta) + 0.3 * np.sin(2 * lay.node_theta), (E, P)).copy()
    F = br.velocity_flux(plan, phi, RHO)
    Fe = ops.extend_periodic(F, 3)

    def apply(x):
        return br.sbp_bracket_ext(plan, Fe, ops.extend_periodic(x, 3), None, 0.7)

    A = assemble_by_probing(apply, P, E)
    dense = np.asarray(jax.jacfwd(lambda x: apply(x.reshape(E, P)).ravel())(jnp.zeros(E * P)))
    assert np.abs(A.toarray() - dense).max() <= 1e-13 * np.abs(dense).max()
    op = jax_linear_operator(apply, E, P)
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(E * P), rng.standard_normal(E * P)
    assert np.abs(op.matvec(x) - A @ x).max() <= 1e-12 * np.abs(A @ x).max()
    assert np.abs(op.rmatvec(y) - A.T @ y).max() <= 1e-12 * np.abs(A.T @ y).max()


def test_rk4_dt_bisection():
    assert rk4_dt([1j * 10.0]) == pytest.approx(2.0 * np.sqrt(2.0) / 10.0, rel=1e-9)
    assert rk4_dt([-1.0 + 0j]) == pytest.approx(2.7852, rel=1e-4)
    assert rk4_dt([-3.0, 1j * 2.0]) <= rk4_dt([1j * 2.0])


# ------------------------------------------------------------------------------------------------ C16
def test_jax_hygiene_with_core():
    lay, plan = make(16, kind="smooth")
    E, P = lay.n_eta, lay.P
    rng = np.random.default_rng(2)
    phi = jnp.asarray(np.broadcast_to(lay.node_u * np.cos(lay.node_theta), (E, P)).copy())
    g = jnp.asarray(rng.standard_normal((E, P, 1)))
    traces = []

    def fn(p, ph, gg, rho, ck):
        traces.append(1)
        return br.sbp_bracket(p, ph, gg, None, rho, ck)

    jf = jax.jit(fn)
    out1 = jf(plan, phi, g, RHO, 1.0)
    metric2 = nodal_metric_from_callable(lay, lambda pts: tuple(a * (1.2 if i == 1 else 1.0)
                                                                for i, a in enumerate(logical_metric("smooth")(pts))))
    plan2 = build_nodal_plan(lay, metric2)
    out2 = jf(plan2, phi, g, RHO, 0.3)
    assert len(traces) == 1 and np.all(np.isfinite(out1)) and np.abs(np.asarray(out1) - np.asarray(out2)).max() > 0
    # jvp in phi and in c_kappa (the core damping has a jnp.max in kappa)
    jvp = jax.jit(lambda p, ph, dph, gg: jax.jvp(lambda x, ck: br.sbp_bracket(p, x, gg, None, RHO, ck), (ph, 1.0),
                                                 (dph, 1.0))[1])
    tangent = np.asarray(jvp(plan, phi, phi * 0.1 + 1.0, g))
    assert np.all(np.isfinite(tangent)) and np.abs(tangent).max() > 0
