"""Algebra-only tests of the Zernike disk core, its SBP operator, frame transforms, D5c data and family A."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import eval_jacobi

from drbx.geometry.nodal_families import build_family_a_layout, family_a
from drbx.geometry.nodal_layout import Block, expand_to_raw
from drbx.geometry.sbp_core import (
    CoreBlock,
    build_core_block,
    jacobi_alpha0,
    polar_gauss_nodes,
    shell_gram_inverse,
    shell_kappa,
    shell_projector,
    zernike_indices,
    zernike_vandermonde,
)
from drbx.geometry.sbp_operators import TWO_PI, ring_basis


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


@pytest.fixture(scope="module")
def cores():
    return {(n, K, p): build_core_block(n, K, p) for n, K, p in ((16, 2, 4), (32, 4, 6), (64, 8, 10))}


# ------------------------------------------------------------------------------------------------ C1
def test_jacobi_recurrence_against_scipy():
    t = np.concatenate([np.linspace(-1.0, 1.0, 197), [-1.0, 1.0, 0.0]])
    p = 30
    for alpha in range(p + 1):
        kmax = (p - alpha) // 2
        P, dP = jacobi_alpha0(alpha, kmax, t)
        for k in range(kmax + 1):
            ref = eval_jacobi(k, alpha, 0, t)
            assert rel(P[k], ref) <= 1e-12
            if k:
                dref = 0.5 * (k + alpha + 1) * eval_jacobi(k - 1, alpha + 1, 1, t)
                assert rel(dP[k], dref) <= 1e-12
        if kmax >= 2:   # centred finite difference of the recurrence itself
            h = 1e-6
            tt = np.linspace(-0.9, 0.9, 11)
            fd = (jacobi_alpha0(alpha, kmax, tt + h)[0] - jacobi_alpha0(alpha, kmax, tt - h)[0]) / (2 * h)
            assert rel(fd, jacobi_alpha0(alpha, kmax, tt)[1]) <= 1e-6


@pytest.mark.parametrize("p", [4, 6, 10])
def test_zernike_cartesian_derivatives_against_finite_differences(p):
    R = 0.3
    rng = np.random.default_rng(p)
    r, th = R * np.sqrt(rng.random(40)) * 0.95 + 1e-3, TWO_PI * rng.random(40)
    x, y = r * np.cos(th), r * np.sin(th)
    V, Vx, Vy, deg = zernike_vandermonde(r, th, p, R)
    h = 1e-6

    def at(dx, dy):
        xx, yy = x + dx, y + dy
        return zernike_vandermonde(np.hypot(xx, yy), np.arctan2(yy, xx), p, R)[0]

    assert rel((at(h, 0) - at(-h, 0)) / (2 * h), Vx) <= 1e-6
    assert rel((at(0, h) - at(0, -h)) / (2 * h), Vy) <= 1e-6
    assert V.shape[1] == (p + 1) * (p + 2) // 2 and deg.max() == p and len(zernike_indices(p)) == V.shape[1]


# ------------------------------------------------------------------------------------------------ C2
@pytest.mark.parametrize("p", [4, 6, 8, 10, 16, 24])
def test_gram_is_identity(p):
    R = 0.125
    u, th, w, _ = polar_gauss_nodes(p, R)
    V = zernike_vandermonde(u, th, p, R)[0]
    assert np.abs(V.T @ (w[:, None] * V) - np.eye(V.shape[1])).max() <= 1e-12
    assert u.size == (p // 2 + 1) * (2 * p + 2) and np.all(w > 0)


@pytest.mark.parametrize("p", [4, 6, 10])
def test_quadrature_exact_to_degree_2p_plus_1(p):
    R = 0.4
    u, th, w, _ = polar_gauss_nodes(p, R)
    x, y = u * np.cos(th), u * np.sin(th)
    for a in range(2 * p + 2):
        for b in range(2 * p + 2 - a):
            exact = 0.0
            if a % 2 == 0 and b % 2 == 0:
                exact = R ** (a + b + 2) * math.exp(math.lgamma((a + 1) / 2) + math.lgamma((b + 1) / 2)
                                                    - math.lgamma((a + b + 4) / 2))
            assert abs(np.sum(w * x**a * y**b) - exact) <= 1e-13 * R ** (a + b + 2)


# ------------------------------------------------------------------------------------------------ C3
def monomial_core(n, K, p):
    """Reference of the same operator in the monomial basis (formulas of the projection-type SBP construction)."""
    R = K / n
    u, th, wxy, _ = polar_gauss_nodes(p, R)
    ex = [(a, d - a) for d in range(p + 1) for a in range(d, -1, -1)]

    def vander(x, y):
        X, Y = x / R, y / R
        V = np.stack([X**a * Y**b for a, b in ex], 1)
        Vx = np.stack([(a * X ** max(a - 1, 0) * Y**b / R) if a else 0 * X for a, b in ex], 1)
        Vy = np.stack([(b * X**a * Y ** max(b - 1, 0) / R) if b else 0 * X for a, b in ex], 1)
        return V, Vx, Vy

    V, Vx, Vy = vander(u * np.cos(th), u * np.sin(th))
    H = np.diag(wxy)
    VH = np.linalg.solve(V.T @ H @ V, V.T @ H)
    Pi = V @ VH
    thg = np.pi / n + TWO_PI * np.arange(n) / n
    Vg = vander(R * np.cos(thg), R * np.sin(thg))[0]
    Rx = Vg @ VH
    Bg = R * TWO_PI / n
    Ex = Rx.T @ np.diag(Bg * np.cos(thg)) @ Rx
    Ey = Rx.T @ np.diag(Bg * np.sin(thg)) @ Rx
    I = np.eye(len(u))
    Qx = H @ (Vx @ VH) + Pi.T @ Ex @ (I - Pi) + 0.5 * (I - Pi.T) @ Ex @ (I - Pi)
    Qy = H @ (Vy @ VH) + Pi.T @ Ey @ (I - Pi) + 0.5 * (I - Pi.T) @ Ey @ (I - Pi)
    low = [i for i, (a, b) in enumerate(ex) if a + b <= p - 1]
    Vl = V[:, low]
    Pi_low = Vl @ np.linalg.solve(Vl.T @ H @ Vl, Vl.T @ H)
    return dict(D1=Qx / wxy[:, None], D2=Qy / wxy[:, None], Rx=Rx, Pi=Pi, Pi_low=Pi_low)


@pytest.mark.parametrize("p", [2, 4, 6, 8, 10])
def test_zernike_matches_monomial_reference(p):
    core = build_core_block(48, 6, p)
    ref = monomial_core(48, 6, p)
    Vl = core.Vm
    Pi_low = Vl @ np.linalg.solve(Vl.T @ (core.wxy[:, None] * Vl), Vl.T * core.wxy[None, :])
    errs = {"D1": rel(core.D1, ref["D1"]), "D2": rel(core.D2, ref["D2"]), "Rx": rel(core.Rx, ref["Rx"]),
            "Pi": rel(core.Pi, ref["Pi"]), "Pi_low": rel(Pi_low, ref["Pi_low"])}
    print(f"C3 p={p}: " + ", ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    assert max(errs.values()) <= 1e-10, errs


# ------------------------------------------------------------------------------------------------ C4
def test_core_sbp_property_and_exactness(cores):
    for (n, K, p), c in cores.items():
        assert np.abs(c.Qx + c.Qx.T - c.Ex).max() <= 1e-13 * np.abs(c.Ex).max()
        assert np.abs(c.Qy + c.Qy.T - c.Ey).max() <= 1e-13 * np.abs(c.Ey).max()
        assert rel(c.D1 @ c.V, c.Vx) <= 1e-12 and rel(c.D2 @ c.V, c.Vy) <= 1e-12
        comm = c.D1 @ (c.D2 @ c.V) - c.D2 @ (c.D1 @ c.V)
        assert np.abs(comm).max() <= 1e-11 * np.abs(c.Vx).max()
        assert rel(c.Rx @ c.V, c.Vg) <= 1e-12
        assert np.abs(c.Pi @ c.V - c.V).max() <= 1e-12


def test_core_node_counts_and_weights(cores):
    assert {k: c.n_nodes for k, c in cores.items()} == {(16, 2, 4): 30, (32, 4, 6): 56, (64, 8, 10): 132}
    c = cores[(32, 4, 6)]
    assert abs(c.wxy.sum() - np.pi * c.R_c**2) <= 1e-14 and np.all(c.u < c.R_c) and np.all(c.u > 0)
    assert c.theta[0] == 0.0 and c.sides["outer"].u_face == c.R_c and c.sides["outer"].N == 32
    assert "inner" not in c.sides and isinstance(c, Block) and c.kind == "core" and c.frame == "cartesian"


# ------------------------------------------------------------------------------------------------ C5
def test_flux_trace_of_radial_field(cores):
    for c in cores.values():
        s = c.sides["outer"]
        ft = s.TF1 @ (c.u * np.cos(c.theta)) + s.TF2 @ (c.u * np.sin(c.theta))
        assert np.abs(ft - c.R_c**2).max() <= 1e-13


# ------------------------------------------------------------------------------------------------ C6
def poly_fields(rng, deg=3):
    ex = [(a, b) for a in range(deg + 1) for b in range(deg + 1 - a)]
    coef = rng.standard_normal(len(ex))

    def fn(x, y):
        v = sum(c * x**a * y**b for c, (a, b) in zip(coef, ex))
        vx = sum(c * a * x ** max(a - 1, 0) * y**b for c, (a, b) in zip(coef, ex) if a)
        vy = sum(c * b * x**a * y ** max(b - 1, 0) for c, (a, b) in zip(coef, ex) if b)
        return v, vx, vy

    return fn


def test_frame_transforms_preserve_bracket_and_curvature_contraction(cores):
    c = cores[(32, 4, 6)]
    rng = np.random.default_rng(3)
    E = 3
    h = rng.standard_normal((E, c.n_nodes, 3))
    jac = 0.5 + rng.random((E, c.n_nodes))
    K = rng.standard_normal((E, c.n_nodes, 3))
    fa, fb = poly_fields(rng), poly_fields(rng)
    x, y = c.x[None, :], c.y[None, :]
    eta_a, eta_b = rng.standard_normal((2, E, c.n_nodes))

    def grads(f, eta):
        _v, fx, fy = f(x, y)
        fx, fy = np.broadcast_to(fx, (E, c.n_nodes)), np.broadcast_to(fy, (E, c.n_nodes))
        blk = np.stack([fx, fy, eta], -1)
        log = np.stack([np.cos(c.theta) * fx + np.sin(c.theta) * fy,
                        c.u * (-np.sin(c.theta) * fx + np.cos(c.theta) * fy), eta], -1)
        return blk, log

    ga_b, ga_l = grads(fa, eta_a)
    gb_b, gb_l = grads(fb, eta_b)
    hb, jb = c.to_block_frame(h, jac)
    Kb = c.to_block_frame_K(K)

    def pb(hh, jj, a, b):
        return -np.sum(np.cross(hh, a) * b, -1) / np.abs(jj)

    assert rel(pb(hb, jb, ga_b, gb_b), pb(h, jac, ga_l, gb_l)) <= 1e-13
    assert rel(np.sum(Kb * ga_b, -1), np.sum(K * ga_l, -1)) <= 1e-13
    assert np.array_equal(hb[..., 2], h[..., 2]) and np.array_equal(Kb[..., 2], K[..., 2])
    assert np.allclose(jb, jac / c.u[None, :])


# ------------------------------------------------------------------------------------------------ C7
def test_core_d5c_matches_boundary_modes_and_reproduces_polynomial_gradients(cores):
    c = cores[(32, 4, 6)]
    rng = np.random.default_rng(5)
    _B, Binv, keys, _om = ring_basis(32, np.pi / 32)
    sel = list(c.D5c_sel)
    assert [keys[i][0] for i in sel] == sorted(keys[i][0] for i in sel) and len(sel) == 2 * c.p
    phi_c, tr = rng.standard_normal(c.n_nodes), rng.standard_normal(32)
    q = c.Q_phi @ phi_c + c.Q_tr @ tr
    assert np.abs(Binv[sel] @ (c.Vg @ q) - Binv[sel] @ tr).max() <= 1e-12 * np.abs(tr).max()
    f = poly_fields(rng, deg=c.p)
    phi, fx, fy = f(c.x, c.y)
    tr_exact = f(c.R_c * np.cos(c.theta_g), c.R_c * np.sin(c.theta_g))[0]
    assert rel(c.G1_phi @ phi + c.G1_tr @ tr_exact, fx) <= 1e-11
    assert rel(c.G2_phi @ phi + c.G2_tr @ tr_exact, fy) <= 1e-11


# ------------------------------------------------------------------------------------------------ C8
def test_shell_damping_projector_and_kappa(cores):
    c = cores[(32, 4, 6)]
    rng = np.random.default_rng(8)
    jac_b = 0.5 + rng.random(c.n_nodes)
    Hk = c.wxy * jac_b
    assert c.Vm.shape == (c.n_nodes, c.p * (c.p + 1) // 2)
    Ph = np.eye(c.n_nodes) - shell_projector(c.Vm, Hk)
    HPh = Hk[:, None] * Ph
    scale = np.abs(HPh).max()
    assert np.abs(HPh - HPh.T).max() <= 1e-13 * scale
    assert np.linalg.eigvalsh(0.5 * (HPh + HPh.T)).min() >= -1e-13 * scale
    assert np.abs(Ph @ c.Vm).max() <= 1e-12 * np.abs(c.Vm).max()
    assert np.allclose(shell_gram_inverse(c.Vm, Hk) @ (c.Vm.T @ (Hk[:, None] * c.Vm)), np.eye(c.Vm.shape[1]), atol=1e-9)
    kappa = shell_kappa(0.5, c.p, c.R_c, 5.0)
    assert kappa == pytest.approx(0.5 * 5.0 * 6 / c.R_c, rel=1e-15)
    for _ in range(50):
        x = rng.standard_normal(c.n_nodes)
        assert x @ (Hk * (-kappa * (Ph @ x))) <= 1e-14 * kappa * np.sqrt((Hk * x * x).sum() * (Hk * (Ph @ x) ** 2).sum())
    # exactness degree: the removed space is exactly P_{p-1}; the top-degree Zernike functions are damped
    top = c.V[:, c.degree == c.p]
    assert np.abs(Ph @ top).max() > 1e-3


# ------------------------------------------------------------------------------------------------ C9 and layout
def test_family_a_rule_and_layout_sizes():
    expected = {16: (2, 4), 32: (4, 6), 48: (6, 8), 64: (8, 10), 128: (16, 18)}
    for n, (K, p) in expected.items():
        assert family_a(n) == (K, p, ((K, n, n),))
        lay = build_family_a_layout(n, n_eta=4)
        n_core = (p // 2 + 1) * (2 * p + 2)
        assert lay.P == n_core + (n - K) * n and isinstance(lay.blocks[0], CoreBlock)
        assert lay.offsets == (0, n_core) and lay.inner == "core" and len(lay.faces) == 1 and len(lay.walls) == 1
        assert lay.faces[0].NA == lay.faces[0].NB == n and lay.faces[0].X == "B"
        assert np.array_equal(lay.faces[0].Iab, np.eye(n))
    assert {n: build_family_a_layout(n, n_eta=4).P for n in (32, 48, 64)} == {32: 952, 48: 2106, 64: 3716}
    for bad in (8, 20, 36, 15):
        with pytest.raises(ValueError):
            family_a(bad)


def test_core_block_rejects_bad_parameters():
    with pytest.raises(ValueError):
        build_core_block(16, 2, 9)
    with pytest.raises(ValueError):
        build_core_block(16, 0, 4)


def test_expand_to_raw_with_core_evaluate_at_reproduces_polynomials():
    lay = build_family_a_layout(32, n_eta=32)
    c = lay.blocks[0]
    f = poly_fields(np.random.default_rng(1), deg=c.p)
    n, E = lay.n, lay.n_eta
    eta = (np.arange(E) + 0.5) * lay.deta
    x, y = lay.node_u * np.cos(lay.node_theta), lay.node_u * np.sin(lay.node_theta)
    vals = np.broadcast_to(f(x, y)[0], (E, lay.P)) * (1.0 + 0.0 * eta[:, None])
    raw = expand_to_raw(lay, np.ascontiguousarray(vals))
    ii, jj = np.meshgrid(np.arange(c.K), np.arange(n), indexing="ij")
    u, th = (ii + 0.5) / n, lay.delta + TWO_PI * jj / n
    exact = f(u * np.cos(th), u * np.sin(th))[0]
    assert np.abs(raw[: c.K, :, 0] - exact).max() <= 1e-9 * np.abs(exact).max()
    assert np.all(np.isfinite(raw))
