"""Algebra-only tests of the face-jump dissipation of the nodal SBP scheme."""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry.nodal_layout import build_nodal_layout
from drbx.native import fci_perpendicular_sbp_dissipation as dis
from drbx.native.fci_perpendicular_sbp_norms import h_weights
from drbx.native.fci_perpendicular_sbp_ops import extend_periodic
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable

N_GRID = 32
E = 8
LEVELS = [(8, 16, 16), (16, 32, 32)]


@pytest.fixture(scope="module")
def case():
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=E, inner="wall")

    def fn(p):
        u, th, et = p.T
        return np.zeros((len(u), 3)) + [0, 0, 1.0], u * (1 + 0.1 * np.cos(th) * np.cos(et)), 1 + 0 * u, np.zeros((len(u), 3))

    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, fn))
    F = np.random.default_rng(3).standard_normal((E, lay.P, 3))
    return lay, plan, F, np.asarray(h_weights(plan))


def hdot(H, a, b):
    return float(np.sum(H.reshape(H.shape + (1,) * (a.ndim - 2)) * a * b))


def test_dissipation_is_symmetric_and_negative_semidefinite(case):
    lay, plan, F, H = case
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, E, lay.P))
    Dx, Dy = np.asarray(dis.dissipation(plan, F, x)), np.asarray(dis.dissipation(plan, F, y))
    scale = np.sqrt(hdot(H, x, x) * hdot(H, Dy, Dy))
    assert abs(hdot(H, x, Dy) - hdot(H, y, Dx)) <= 1e-13 * scale
    for _ in range(50):
        x = rng.standard_normal((E, lay.P))
        Dx = np.asarray(dis.dissipation(plan, F, x))
        assert hdot(H, x, Dx) <= 1e-14 * np.sqrt(hdot(H, x, x) * hdot(H, Dx, Dx))


def test_dissipation_kills_constants_and_quadratics(case):
    lay, plan, F, H = case
    assert np.abs(np.asarray(dis.dissipation(plan, F, np.ones((E, lay.P))))).max() <= 1e-14
    quad = np.broadcast_to(lay.node_u**2, (E, lay.P)).copy()
    assert np.abs(np.asarray(dis.dissipation(plan, F, quad))).max() <= 1e-13
    assert np.abs(np.asarray(dis.dissipation_ring(plan, F, quad))).max() <= 1e-13


def test_dissipation_has_no_u_faces_across_level_end(case):
    """A jump confined to the ring just inside a level end is damped only in theta/eta, never across the level face."""
    lay, plan, F, H = case
    g = np.zeros((E, lay.P))
    last_ring_of_level0 = slice(lay.offsets[0] + 7 * 16, lay.offsets[0] + 8 * 16)
    g[:, last_ring_of_level0] = 1.0
    D = np.asarray(dis.dissipation_ring(plan, F, g))
    level1 = slice(lay.offsets[1], lay.P)
    assert np.abs(D[:, level1]).max() == 0.0
    assert np.abs(D[:, lay.offsets[0]:lay.offsets[1]]).max() > 0.0


def test_dissipation_terms_and_halo_independence(case):
    lay, plan, F, H = case
    g = np.random.default_rng(4).standard_normal((E, lay.P, 2))
    total = np.asarray(dis.dissipation(plan, F, g))
    assert np.abs(total - np.asarray(dis.dissipation_ring(plan, F, g)) - np.asarray(dis.dissipation_eta(plan, F, g))).max() <= 1e-14
    wide = np.asarray(dis.dissipation_ext(plan, extend_periodic(F, 5), extend_periodic(g, 5)))
    assert np.abs(wide - total).max() <= 1e-13 * np.abs(total).max()
    with pytest.raises(ValueError, match="halo"):
        dis.dissipation_ext(plan, extend_periodic(F, 2), extend_periodic(g, 2))


def test_eta_dissipation_matches_dense_reference(case):
    lay, plan, F, H = case
    g = np.random.default_rng(6).standard_normal((E, lay.P))
    C = np.array([1.0, -3.0, 3.0, -1.0]) / 8.0
    jump = (np.roll(g, 1, 0) * C[0] + g * C[1] + np.roll(g, -1, 0) * C[2] + np.roll(g, -2, 0) * C[3])
    F3 = F[..., 2]
    s = 0.5 * plan.wxy[None, :] * np.abs(0.5 * (F3 + np.roll(F3, -1, 0)))
    q = s * jump
    ref = -(np.roll(q, -1, 0) * C[0] + q * C[1] + np.roll(q, 1, 0) * C[2] + np.roll(q, 2, 0) * C[3]) / H
    assert np.abs(np.asarray(dis.dissipation_eta(plan, F, g)) - ref).max() <= 1e-13 * np.abs(ref).max()
