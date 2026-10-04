"""Algebra-only tests of the nodal SBP bracket: energy identities, constants, straight-field c, polynomial bracket,
dense-block parity and JAX hygiene."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry.nodal_layout import DenseBlock, Side, build_nodal_layout
from drbx.native import fci_perpendicular_sbp_bracket as br
from drbx.native import fci_perpendicular_sbp_dissipation as dis
from drbx.native import fci_perpendicular_sbp_ops as ops
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData, sat_boundary_data_from_callables
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable

TWO_PI = 2.0 * np.pi
N_GRID = 32
LEVELS = [(8, 16, 16), (16, 32, 32)]
E = 8
RHO = 0.05


def metric_fn(kind="random", jac_kind="eta"):
    def fn(p):
        u, th, et = p.T
        if kind == "straight":
            h = np.stack([0 * u, 0 * u, 1.0 + 0 * u], -1)
        else:
            h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
        jac = u * (1.0 + 0.1 * np.cos(th) * np.cos(et)) if jac_kind == "eta" else u
        return h, jac, 1.0 + 0 * u, np.zeros((len(u), 3))

    return fn


def make(levels=LEVELS, kind="random", jac_kind="eta", n_eta=E):
    lay = build_nodal_layout(N_GRID, levels, n_eta=n_eta, inner="wall")
    return lay, build_nodal_plan(lay, nodal_metric_from_callable(lay, metric_fn(kind, jac_kind)))


def hsum(plan, a, b):
    Hp = np.asarray(plan.Hp)
    Hp = Hp.reshape(Hp.shape + (1,) * (np.ndim(a) - 2))
    return float(np.sum(Hp * np.asarray(a) * np.asarray(b)))


def traces_and_speeds(plan, F, g):
    out = []
    for blk, side, sign, N in plan.structure.walls:
        a = np.asarray(ops.trace(plan, blk, side, g))
        v = np.asarray(ops.flux_trace(plan, blk, side, F[..., 0], F[..., 1]))
        out.append((sign, N, a, v))
    return out


@pytest.fixture(scope="module")
def energy_case():
    lay, plan = make()
    rng = np.random.default_rng(7)
    F = rng.standard_normal((E, lay.P, 3))
    g = rng.standard_normal((E, lay.P))
    return lay, plan, F, g


def test_energy_identity_transport(energy_case):
    """(a) and (b): the SBP boundary terms of every wall and level face are cancelled by the SATs."""
    lay, plan, F, g = energy_case
    Tg = np.asarray(br.transport(plan, F, g))
    A = np.asarray(br.advect(plan, *br._wrap(F, g)))
    scale = np.sqrt(hsum(plan, g, g)) * np.sqrt(hsum(plan, A, A))
    wall = sum(-0.5 * (TWO_PI / N) * np.sum(sign * v * a * a) for sign, N, a, v in traces_and_speeds(plan, F, g))
    assert abs(hsum(plan, g, Tg) - wall) / scale <= 1e-13
    # interface: the centred SAT cancels the raw SBP boundary terms at the level face
    (a_blk, b_blk, NA, NB, _x), = plan.structure.faces
    a, b = np.asarray(ops.trace(plan, a_blk, "outer", g)), np.asarray(ops.trace(plan, b_blk, "inner", g))
    fA = np.asarray(ops.flux_trace(plan, a_blk, "outer", F[..., 0] * g, F[..., 1] * g))
    fB = np.asarray(ops.flux_trace(plan, b_blk, "inner", F[..., 0] * g, F[..., 1] * g))
    raw_face = -0.5 * (TWO_PI / NA * np.sum(a * fA) - TWO_PI / NB * np.sum(b * fB))
    iface = np.asarray(br.interface_centred(plan, F, g))
    assert abs(hsum(plan, g, iface) + raw_face) / scale <= 1e-13


def test_energy_identity_upwind_and_full_operator(energy_case):
    lay, plan, F, g = energy_case
    A = np.asarray(br.advect(plan, *br._wrap(F, g)))
    scale = np.sqrt(hsum(plan, g, g)) * np.sqrt(hsum(plan, A, A))
    (a_blk, b_blk, NA, NB, X), = plan.structure.faces
    face = lay.faces[0]
    a, b = np.asarray(ops.trace(plan, a_blk, "outer", g)), np.asarray(ops.trace(plan, b_blk, "inner", g))
    vA = np.asarray(ops.flux_trace(plan, a_blk, "outer", F[..., 0], F[..., 1]))
    vB = np.asarray(ops.flux_trace(plan, b_blk, "inner", F[..., 0], F[..., 1]))
    assert X == "B"
    j = b - a @ face.Iab.T
    gamma = 0.5 * np.abs(0.5 * (vA @ face.Iab.T + vB))
    up_pred = -np.sum((TWO_PI / NB) * gamma * j * j)
    up = np.asarray(br.interface_upwind(plan, F, g))
    assert abs(hsum(plan, g, up) - up_pred) / scale <= 1e-13
    # (d): L = T + c/2 + inflow + upwind against the predicted energy
    L = np.asarray(br.linear_operator(plan, F, g, None))
    c = np.asarray(br.compressibility(plan, F))
    wall = sum(-0.5 * (TWO_PI / N) * np.sum(sign * v * a_ * a_) for sign, N, a_, v in traces_and_speeds(plan, F, g))
    inflow = -sum((TWO_PI / N) * np.sum(np.maximum(-sign * v, 0.0) * a_ * a_) for sign, N, a_, v in traces_and_speeds(plan, F, g))
    pred = wall + 0.5 * hsum(plan, c, g * g) + inflow + up_pred
    assert abs(hsum(plan, g, L) - pred) / scale <= 1e-13
    # c excludes inflow and upwind: c = -2 T(1)
    T1 = np.asarray(br.transport(plan, F, np.ones((E, lay.P))))
    assert np.abs(c + 2 * T1).max() == 0.0


def test_inner_wall_inflow_sign_and_energy():
    """The sigma = -1 wall: inflow where F.n_out < 0, i.e. flux in the +u direction at the inner wall."""
    lay, plan = make()
    rng = np.random.default_rng(1)
    F = np.zeros((E, lay.P, 3))
    F[..., 0] = 1.0  # +u flux: enters through the inner wall, leaves through the outer wall
    g = rng.standard_normal((E, lay.P))
    inflow = np.asarray(br.wall_inflow(plan, F, g, None))
    inner_nodes = slice(lay.offsets[0], lay.offsets[0] + 3 * 16)   # first three rings carry the inner trace
    outer_nodes = slice(lay.P - 3 * 32, lay.P)
    assert np.abs(inflow[:, inner_nodes]).max() > 0.0 and np.abs(inflow[:, outer_nodes]).max() == 0.0
    F[..., 0] = -1.0
    inflow = np.asarray(br.wall_inflow(plan, F, g, None))
    assert np.abs(inflow[:, inner_nodes]).max() == 0.0 and np.abs(inflow[:, outer_nodes]).max() > 0.0
    # energy of the inflow term on the inner wall is -Omega sum tau a^2 with tau = max(v, 0)
    F[..., 0] = 1.0
    a_in = np.asarray(ops.trace(plan, 0, "inner", g))
    v_in = np.asarray(ops.flux_trace(plan, 0, "inner", F[..., 0], F[..., 1]))
    inflow = np.asarray(br.wall_inflow(plan, F, g, None))
    assert hsum(plan, g, inflow) == pytest.approx(-(TWO_PI / 16) * np.sum(np.maximum(v_in, 0.0) * a_in**2), rel=1e-12)


def wall_data_ones(lay):
    return SatBoundaryData(tuple(jnp.ones((E, w_N, 1)) for w_N in (32, 16)), None)


def test_constants_preserved_with_unit_wall_data(energy_case):
    lay, plan, F, _ = energy_case
    g1 = np.ones((E, lay.P, 1))
    L1 = np.asarray(br.linear_operator(plan, F, g1, wall_data_ones(lay)))
    assert np.abs(L1).max() <= 1e-11 * np.abs(F).max() / np.asarray(plan.Hp).min()


def smooth_phi(lay, eta_dep=False):
    u, th = lay.node_u[None, :], lay.node_theta[None, :]
    phi = (1.0 + u + 3.0 * u**4) * (np.cos(2 * th + 0.3) + 0.5 * np.sin(3 * th)) + np.sin(5 * u) * np.cos(th)
    return np.broadcast_to(phi, (E, lay.P)).copy()


def test_straight_field_compressibility_and_negative_control():
    lay, plan = make(kind="straight", jac_kind="u")
    phi = smooth_phi(lay)
    F = np.asarray(br.velocity_flux(plan, phi, RHO))
    assert np.abs(F[..., 2]).max() == 0.0
    c = np.asarray(br.compressibility(plan, F))
    assert np.abs(c).max() <= 1e-10 * N_GRID * np.abs(F).max() / np.asarray(plan.jac).min()
    F_raw = np.asarray(br.flux_from_potential(plan, ops.extend_periodic(jnp.asarray(phi), 2), RHO))
    c_raw = np.asarray(br.compressibility(plan, F_raw))
    assert np.abs(c_raw).max() > 1e-6


def test_polynomial_bracket_exact():
    lay, plan = make(kind="straight", jac_kind="u")
    u, th = lay.node_u, lay.node_theta
    phi = np.broadcast_to(u * np.cos(th), (E, lay.P)).copy()
    g = np.broadcast_to((u * np.sin(th))[None, :, None], (E, lay.P, 1)).copy()

    def dirichlet(points):
        value = (points[:, 0] * np.sin(points[:, 1]))[:, None]
        return value, np.zeros((len(points), 3, 1))

    bcd = sat_boundary_data_from_callables(lay, dirichlet=dirichlet)
    F = br.velocity_flux(plan, phi, RHO)
    L = np.asarray(br.linear_operator(plan, F, g, bcd))
    assert np.abs(L + 1.0 / RHO).max() <= 1e-11 / RHO


def test_dense_block_parity_with_structured_path():
    lay, plan = make(levels=[(8, 32, 32)])
    rb = lay.blocks[0]
    eye_n, eye_m = np.eye(rb.N), np.eye(rb.m)

    def side(t, u_face):
        T = np.kron(t[None, :], eye_n)
        return Side(N=rb.N, T=T, TF1=T, TF2=np.zeros_like(T), u_face=u_face)

    dense = DenseBlock(D1=np.kron(rb.Du, eye_n), D2=np.kron(eye_m, rb.Dth), wxy=rb.wxy, u=rb.u, theta=rb.theta,
                       sides={"inner": side(rb.radial.tL, rb.i0 / N_GRID), "outer": side(rb.radial.tR, rb.i1 / N_GRID)})
    lay_d = dataclasses.replace(lay, blocks=(dense,))
    plan_d = build_nodal_plan(lay_d, nodal_metric_from_callable(lay_d, metric_fn()))
    assert plan_d.structure.blocks[0][0] == "dense"
    rng = np.random.default_rng(5)
    F = rng.standard_normal((E, lay.P, 3))
    g = rng.standard_normal((E, lay.P, 2))
    bcd = SatBoundaryData(tuple(jnp.asarray(rng.standard_normal((E, 32, 2))) for _ in range(2))[:1]
                          + (jnp.asarray(rng.standard_normal((E, 32, 2))),), None)

    def rel(a, b):
        return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max()

    assert rel(br.transport(plan_d, F, g), br.transport(plan, F, g)) <= 1e-13
    assert rel(br.wall_inflow(plan_d, F, g, bcd), br.wall_inflow(plan, F, g, bcd)) <= 1e-13
    assert rel(br.compressibility(plan_d, F), br.compressibility(plan, F)) <= 1e-13
    assert rel(dis.dissipation(plan_d, F, g), dis.dissipation_eta(plan, F, g)) <= 1e-13


def test_jax_hygiene_jit_plan_argument_and_jvp():
    lay, plan = make()
    rng = np.random.default_rng(2)
    phi = jnp.asarray(smooth_phi(lay))
    g = jnp.asarray(rng.standard_normal((E, lay.P, 1)))
    bcd = wall_data_ones(lay)
    traces = []

    def fn(p, ph, gg, bc_, rho):
        traces.append(1)
        return br.sbp_bracket(p, ph, gg, bc_, rho)

    jit_fn = jax.jit(fn)
    out1 = jit_fn(plan, phi, g, bcd, RHO)
    plan2 = build_nodal_plan(lay, nodal_metric_from_callable(lay, lambda p: tuple(
        a * (1.3 if i == 1 else 1.0) for i, a in enumerate(metric_fn()(p)))))
    out2 = jit_fn(plan2, phi, g, bcd, RHO)
    assert len(traces) == 1
    assert np.all(np.isfinite(out1)) and np.abs(np.asarray(out1) - np.asarray(out2)).max() > 0
    assert len(traces) == 1 and np.all(np.isfinite(np.asarray(jit_fn(plan, phi, g, bcd, RHO))))
    jvp_fn = jax.jit(lambda p, ph, dph, gg, bc_: jax.jvp(lambda x: br.sbp_bracket(p, x, gg, bc_, RHO), (ph,), (dph,))[1])
    tangent = jvp_fn(plan, phi, phi * 0.1 + 1.0, g, bcd)
    assert np.all(np.isfinite(np.asarray(tangent)))


def test_extended_halo_matches_periodic_wrap(energy_case):
    lay, plan, F, g = energy_case
    ref = np.asarray(br.sbp_bracket_ext(plan, *br._wrap(F, g), None))
    big = np.asarray(br.sbp_bracket_ext(plan, ops.extend_periodic(F, 5), ops.extend_periodic(g, 5), None))
    assert np.abs(big - ref).max() <= 1e-13 * np.abs(ref).max()


def test_non_ring_face_raises_not_implemented():
    z = np.zeros((16, 5))
    side = Side(N=16, T=z, TF1=z, TF2=z, u_face=8 / N_GRID)
    core = DenseBlock(D1=np.eye(5), D2=np.eye(5), wxy=np.full(5, 0.1), u=np.full(5, 0.1), theta=np.zeros(5),
                      sides={"inner": side, "outer": side})
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=E, core=core)
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, metric_fn()))
    with pytest.raises(NotImplementedError):
        br.velocity_flux(plan, np.zeros((E, lay.P)), RHO)
