"""Tests of the host audit tools on a tiny nodal SBP operator and on synthetic eta-banded maps."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry.nodal_layout import build_nodal_layout
from drbx.native import fci_perpendicular_sbp_bracket as br
from drbx.native import fci_perpendicular_sbp_ops as ops
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable
from drbx.validation.sbp_audit import (
    assemble_by_probing,
    cayley_band,
    edge_band,
    energy_identity,
    numerical_abscissa,
    probe_colors,
)

RHO = 0.05
E = 8


@pytest.fixture(scope="module")
def tiny():
    lay = build_nodal_layout(16, [(8, 16, 8)], n_eta=E, inner="wall")

    def fn(p):
        u, th, et = p.T
        return (np.zeros((len(u), 3)) + [0.0, 0.0, 1.0], u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), 1.0 + 0 * u,
                np.zeros((len(u), 3)))

    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, fn))
    u, th = lay.node_u[None, :], lay.node_theta[None, :]
    eta = (np.arange(E)[:, None] + 0.5) * lay.deta
    phi = u * np.cos(th) + 0.3 * np.sin(2 * th) + 0.2 * np.cos(eta)
    F = br.velocity_flux(plan, phi, RHO)
    Fe = ops.extend_periodic(F, 3)

    def apply(x):
        return br.sbp_bracket_ext(plan, Fe, ops.extend_periodic(x, 3), None)

    A = assemble_by_probing(apply, lay.P, E)
    dense = np.asarray(jax.jacfwd(lambda x: apply(x.reshape(E, lay.P)).ravel())(jnp.zeros(E * lay.P)))
    return lay, plan, F, apply, A, dense


def test_probe_colors():
    assert [probe_colors(n) for n in (8, 14, 16, 32, 7, 6)] == [8, 7, 8, 8, 7, 6]
    assert probe_colors(32, eta_reach=2) == 5 + 3 and probe_colors(30, eta_reach=2) == 5


def test_assemble_by_probing_matches_dense_jacobian(tiny):
    lay, plan, F, apply, A, dense = tiny
    assert A.shape == (E * lay.P, E * lay.P)
    assert np.abs(A.toarray() - dense).max() <= 1e-13 * np.abs(dense).max()


@pytest.mark.parametrize("n_eta", [14, 16, 9])
def test_probing_colouring_on_eta_banded_map(n_eta):
    """Colours with c < n_eta (and the n_eta fallback) on a map with eta reach 3 and arbitrary in-plane coupling."""
    P = 5
    rng = np.random.default_rng(n_eta)
    M = rng.standard_normal((P, P))
    w = rng.standard_normal(7)
    S = rng.standard_normal((P, P))

    def apply(x):
        out = x @ M.T
        for o, wk in zip(range(-3, 4), w):
            out = out + wk * jnp.roll(x, o, axis=0) @ S.T
        return out

    A = assemble_by_probing(apply, P, n_eta, eta_reach=3, batch=7)
    dense = np.asarray(jax.jacfwd(lambda v: apply(v.reshape(n_eta, P)).ravel())(jnp.zeros(n_eta * P)))
    assert probe_colors(n_eta) == (7 if n_eta in (14,) else 8 if n_eta == 16 else 9)
    assert np.abs(A.toarray() - dense).max() <= 1e-13 * np.abs(dense).max()


def test_assemble_refuses_large_problems():
    with pytest.raises(ValueError, match="allow_large"):
        assemble_by_probing(lambda x: x, 2000, 32)


def test_energy_identity_residual(tiny):
    lay, plan, F, apply, A, dense = tiny
    Fn = np.asarray(F)

    def T_apply(g):
        return br.transport(plan, F, g)

    def wall_term(g):
        total = 0.0
        for blk, side, sign, N in plan.structure.walls:
            a = np.asarray(ops.trace(plan, blk, side, g))
            v = np.asarray(ops.flux_trace(plan, blk, side, Fn[..., 0], Fn[..., 1]))
            total += -0.5 * (2 * np.pi / N) * sign * np.sum(v * a * a)
        return total

    res = energy_identity(T_apply, plan.Hp, wall_term, np.random.default_rng(0),
                          scale_apply=lambda g: br.advect(plan, *(ops.extend_periodic(x, 3) for x in (F, g))))
    assert res <= 1e-13
    bad = energy_identity(T_apply, plan.Hp, lambda g: 1.1 * wall_term(g), np.random.default_rng(0))
    assert bad > 1e-6


def test_numerical_abscissa_and_cayley_band(tiny):
    lay, plan, F, apply, A, dense = tiny
    H = np.asarray(plan.Hp) * plan.structure.deta
    ev = np.linalg.eigvals(dense)
    rightmost = ev[np.argmax(ev.real)]
    scale = np.abs(ev).max()
    omega = numerical_abscissa(A, H)
    assert omega >= rightmost.real
    assert omega <= 1e-9 * scale                                   # zero wall data, no inflow growth: energy stable
    assert abs(numerical_abscissa(A, H, dense_below=0) - omega) <= 1e-6 * scale
    band, resid = cayley_band(A, sigma=2.0 * scale, k=6)
    assert band.size >= 2 and np.all(resid <= 1e-8)
    assert band[0].real == pytest.approx(rightmost.real, rel=1e-8)
    assert np.min(np.abs(band - rightmost)) <= 1e-8 * abs(rightmost)


def test_edge_band_is_residual_gated(tiny):
    lay, plan, F, apply, A, dense = tiny
    ev = np.linalg.eigvals(dense)
    band, resid = edge_band(A, k=6)
    assert np.all(resid <= 1e-8)
    for lam in band:
        assert np.min(np.abs(ev - lam)) <= 1e-6 * abs(lam)
