"""Nodal SBP perpendicular Laplacian (narrow face-flux energy form): 1-D operator identities, host assembly against the JAX
apply, prototype parity, symmetry / definiteness / null space, polynomial exactness, Neumann data conversion (conormal and
physical-normal, oblique term against finite differences), plan audits and JAX hygiene."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.geometry import sbp_laplacian as sl
from drbx.geometry.nodal_families import build_family_a_layout
from drbx.geometry.nodal_layout import wall_points
from drbx.geometry.sbp_operators import (
    T3_CLOSURE_ROWS, W_CLOSURE, TWO_PI, deta_matrix, radial_block, ring_dtheta)
from drbx.native import fci_perpendicular_sbp_laplacian as lap
from drbx.validation.sbp_audit import assemble_by_probing
from drbx.validation.sbp_laplacian_audit import (
    LaplacianAssembly, audit_laplacian_plan, definiteness_audit, h_symmetry)
from tests import sbp_laplacian_testbed as tb

DATA = Path(__file__).parent / "data" / "p09_laplacian_prototype_parity.npz"
FORM = jax.jit(lap.laplacian_form, static_argnames=("kinds", "neumann_mode"))
KINDS3 = ("dirichlet", "neumann", "dirichlet")


@pytest.fixture(scope="module")
def c16():
    return tb.case(16, 8)


def rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


# ------------------------------------------------------------------------------------------------ 1-D operators
def test_t3_closure_is_sbp_with_the_shared_norm_and_a_cubic_trace():
    r3, r2 = radial_block(14, "t3"), radial_block(14, "s3")
    W = np.diag(r3.w)
    assert np.abs(W @ r3.D_unit + r3.D_unit.T @ W - np.outer(r3.tR, r3.tR) + np.outer(r3.tL, r3.tL)).max() <= 1e-15
    assert np.array_equal(r3.w, r2.w) and r3.w[0] == float(W_CLOSURE[0])
    c = np.arange(14) + 0.5
    for d in range(4):                                       # trace exact to degree 3: sum t c^d = 0^d
        assert abs(r3.tL @ c ** d - (1.0 if d == 0 else 0.0)) <= 1e-13
        assert abs(r3.tR @ (14 - c) ** d - (1.0 if d == 0 else 0.0)) <= 1e-13
    assert np.abs(r3.tL[:4] - np.array([35, -35, 21, -5]) / 16).max() <= 1e-15
    assert abs(r2.tL @ c ** 3) > 1.0                           # the bracket's s3 trace is only quadratic
    for d in range(3):                                       # rows exact to degree 2, interior to degree 4
        assert np.abs(r3.D_unit @ c ** d - (d * c ** (d - 1) if d else 0.0)).max() <= 1e-12
    assert np.abs(r3.D_unit[5:-5] @ c ** 4 - 4 * c[5:-5] ** 3).max() <= 1e-10
    assert np.array_equal(r3.D_unit[5:-5], r2.D_unit[5:-5]) and len(T3_CLOSURE_ROWS) == 4


def test_staggered_pair_identities():
    m = 16
    sp_ = sl.staggered_pair(m)
    Dp, Dm, wc, wf = sp_.Dp, sp_.Dm, sp_.wc, sp_.wf
    c, x = np.arange(m) + 0.5, np.arange(m + 1.0)
    eL, eR = np.eye(m + 1)[0], np.eye(m + 1)[-1]
    # SBP: Hc Dm = -Dp^T Hf + tR eR^T - tL eL^T
    assert np.abs(np.diag(wc) @ Dm + Dp.T @ np.diag(wf) - np.outer(sp_.tR, eR) + np.outer(sp_.tL, eL)).max() <= 1e-14
    assert np.array_equal(wc, radial_block(m, "t3").w)                       # the shared centre norm
    assert np.abs(sp_.tL[:4] - np.array([35, -35, 21, -5]) / 16).max() <= 1e-15      # the collocated t3 trace
    assert np.abs(Dp[8, 6:10] - np.array([1, -27, 27, -1]) / 24).max() <= 1e-15
    assert np.abs(Dm[8, 7:11] - np.array([1, -27, 27, -1]) / 24).max() <= 1e-14
    for d in range(3):                                       # D+ rows and D- closure rows exact to degree 2, traces cubic
        assert np.abs(Dp @ c ** d - (d * x ** (d - 1) if d else 0 * x)).max() <= 1e-10   # closure entries carry 1e-14 round-off
        assert np.abs(Dm @ x ** d - (d * c ** (d - 1) if d else 0 * c)).max() <= 1e-11
    for d in range(4):
        assert abs(sp_.tL @ c ** d - (1.0 if d == 0 else 0.0)) <= 1e-13
    assert np.abs((Dm @ Dp) @ c ** 2 - 2.0).max() <= 1e-11   # the composite second derivative is exact on quadratics
    assert np.abs(Dp.sum(axis=1)).max() <= 1e-13
    assert np.array_equal(Dp[::-1, ::-1], -Dp) or np.abs(Dp[::-1, ::-1] + Dp).max() <= 1e-14
    assert np.abs(wf[::-1] - wf).max() == 0 and np.abs(wc[::-1] - wc).max() == 0


def test_angular_eta_and_face_interpolation_operators():
    N, E, delta = 16, 8, np.pi / 16
    th = delta + TWO_PI * np.arange(N) / N
    thh = th + np.pi / N
    Gt, Ih = sl.stag_fourier(N, delta), sl.fourier_half_interp(N, delta)
    for m in range(1, N // 2):
        assert np.abs(Gt @ np.cos(m * th) + m * np.sin(m * thh)).max() <= 1e-12
        assert np.abs(Ih @ np.sin(m * th) - np.sin(m * thh)).max() <= 1e-12
    assert np.abs(Ih @ np.ones(N) - 1).max() <= 1e-13
    s = np.linalg.svd(Gt, compute_uv=False)
    assert np.sum(s > 1e-9 * s[0]) == N - 1                  # kernel = constants (the Nyquist mode is a cosine)
    assert abs(Gt[:, N // 2:N // 2 + 1].sum()) >= 0 and np.abs(Gt @ (-1.0) ** np.arange(N)).max() > 1.0
    deta = TWO_PI / E
    Ge = sl.stag_eta(E, deta)
    assert np.abs(Ge @ np.ones(E)).max() <= 1e-13
    k = np.arange(0.0, E)
    for m in (1, 2):
        err = np.abs(Ge @ np.cos(m * (k + 0.5) * deta) + m * np.sin(m * (k + 1.0) * deta)).max()
        assert err <= (0.03 if m == 1 else 0.5)               # fourth-order accurate, not exact
    Iu = sl.radial_face_interp(14)
    c, x = np.arange(14) + 0.5, np.arange(15.0)
    for d in range(4):
        assert np.abs(Iu @ c ** d - x ** d).max() <= 1e-10
    assert np.abs(Iu.sum(axis=1) - 1).max() <= 1e-13 and np.abs(Iu[5, 3:7] - np.array([-1, 9, 9, -1]) / 16).max() <= 1e-13
    # JAX FFT implementations equal the dense matrices
    rng = np.random.default_rng(0)
    g = rng.standard_normal((3, 5, N, 2))
    assert rel(lap._half_derivative(g, 2, N), np.einsum("hj,emjf->emhf", Gt, g)) <= 1e-13
    assert rel(lap._half_interp(g, 2, N), np.einsum("hj,emjf->emhf", Ih, g)) <= 1e-13
    assert rel(lap._node_derivative(g, 2, N), np.einsum("hj,emjf->emhf", ring_dtheta(N, delta), g)) <= 1e-13
    ge = rng.standard_normal((E, 4))
    assert rel(lap._half_interp(ge, 0, E), sl.fourier_half_interp(E, 0.0) @ ge) <= 1e-13
    assert rel(lap._stag_eta(ge, deta), Ge @ ge) <= 1e-13
    assert rel(lap._eta_derivative(ge, deta), deta_matrix(E, deta) @ ge) <= 1e-13
    plan = tb.case(16, 8, with_exact=False).plan
    m = plan.structure.m
    h = rng.standard_normal((2, m, 16, 3))
    assert rel(lap._stag_d1(plan.Dp, h, m), np.einsum("km,emjf->ekjf", np.asarray(plan.Dp), h)) <= 1e-13


# ------------------------------------------------------------------------------------------------ host vs JAX
def test_jax_apply_matches_host_assembly_for_mixed_kinds_coefficient_and_c_kappa(c16):
    rng = np.random.default_rng(3)
    E, P, N = c16.E, c16.P, c16.N
    f = rng.standard_normal((E, P, 3))
    gD, gN = rng.standard_normal((E, N, 3)), rng.standard_normal((E, N, 3))
    bcd = lap.LaplacianBoundaryData(value=(gD,), conormal=(gN,))
    for coeff, ck in ((None, 1.0), (c16.coeff, 2.5)):
        out = np.asarray(FORM(c16.plan, f, bcd, KINDS3, coeff, ck))
        asm = LaplacianAssembly(c16.plan, coeff, ck)
        for i, kind in enumerate(KINDS3):
            g = gD[..., i] if kind == "dirichlet" else gN[..., i]
            ref = (asm.matrix(kind) @ f[..., i].ravel() - asm.data_vector(g, kind)).reshape(E, P)
            assert rel(out[..., i], ref) <= 1e-11
    # the action is -H^-1 of the form
    act = np.asarray(lap.laplacian_action(c16.plan, f, bcd, KINDS3))
    H = np.asarray(c16.plan.Hp) * c16.plan.structure.deta
    assert rel(act, -np.asarray(FORM(c16.plan, f, bcd, KINDS3, None, 1.0)) / H[..., None]) <= 1e-13
    assert rel(lap.positive_action(c16.plan, f, bcd, KINDS3), -act) <= 1e-13


def test_physical_normal_neumann_matches_host_assembly(c16):
    rng = np.random.default_rng(4)
    E, P, N = c16.E, c16.P, c16.N
    f = rng.standard_normal((E, P, 2))
    gn = rng.standard_normal((E, N, 2))
    bcd = lap.LaplacianBoundaryData(normal_derivative=(gn,))
    out = np.asarray(FORM(c16.plan, f, bcd, ("neumann",) * 2, c16.coeff, 1.0))
    asm = LaplacianAssembly(c16.plan, c16.coeff, 1.0)
    M = asm.matrix("neumann", "physical")
    for i in range(2):
        ref = (M @ f[..., i].ravel() - asm.data_vector(gn[..., i], "neumann", "physical")).reshape(E, P)
        assert rel(out[..., i], ref) <= 1e-11
    with pytest.raises(ValueError, match="not both"):
        lap.laplacian_form(c16.plan, f, lap.LaplacianBoundaryData(normal_derivative=(gn,), conormal=(gn,)), "neumann")
    with pytest.raises(ValueError, match="kinds"):
        lap.laplacian_form(c16.plan, f, None, ("dirichlet",) * 3)


def test_in_plane_blocks_are_the_diagonal_plane_blocks_of_the_full_matrix(c16):
    import scipy.sparse as sp

    asm = LaplacianAssembly(c16.plan, c16.coeff)
    P, E = c16.P, c16.E
    mask = sp.kron(sp.identity(E), sp.csr_matrix(np.ones((P, P))))
    for kind in ("dirichlet", "neumann"):
        full = asm.matrix(kind)
        assert abs(asm.matrix(kind, inplane=True) - full.multiply(mask)).max() <= 1e-12 * abs(full).max()


# ------------------------------------------------------------------------------------------------ prototype parity
@pytest.mark.parametrize("n", [16, 24])
def test_matrix_parity_with_the_prototype_lap(n):
    """``M f`` and the data vectors of the P09 prototype ``Lap`` (same synthetic metric, fixed random ``f`` and wall data)."""
    ref = np.load(DATA)
    lay = build_family_a_layout(n, n_eta=8)
    met = sl.nodal_laplacian_metric_from_callable(lay, tb.synthetic_geometry)
    plan = sl.build_laplacian_plan(lay, met)
    E, P, N = 8, lay.P, lay.blocks[1].N
    rng = np.random.default_rng(100 + n)
    f, g = rng.standard_normal((E, P)), rng.standard_normal((E, N))
    assert abs(float(plan.tau) - ref[f"n{n}_tau"][0]) <= 1e-12 * ref[f"n{n}_tau"][0]
    assert abs(float(plan.tau_w) - ref[f"n{n}_tau"][1]) <= 1e-12 * ref[f"n{n}_tau"][1]
    asm = LaplacianAssembly(plan)
    for kind in ("dirichlet", "neumann"):
        Mf, b = ref[f"n{n}_{kind}_Mf"], ref[f"n{n}_{kind}_b"]
        assert rel(asm.matrix(kind) @ f.ravel(), Mf) <= 1e-13
        assert rel(asm.data_vector(g, kind), b) <= 1e-13
    if n == 16:                                              # and the JAX apply, both kinds in one call
        out = np.asarray(FORM(plan, np.stack([f, f], -1), lap.LaplacianBoundaryData(value=(np.stack([g, g], -1),),
                                                                                   conormal=(np.stack([g, g], -1),)),
                              ("dirichlet", "neumann"), None, 1.0))
        assert rel(out[..., 0], (ref["n16_dirichlet_Mf"] - ref["n16_dirichlet_b"]).reshape(E, P)) <= 1e-11
        assert rel(out[..., 1], (ref["n16_neumann_Mf"] - ref["n16_neumann_b"]).reshape(E, P)) <= 1e-11


# ------------------------------------------------------------------------------------------------ symmetry, SBP, definiteness
def test_probed_jax_operator_is_h_symmetric_and_equals_the_assembly(c16):
    plan = c16.plan
    Mp = assemble_by_probing(lambda f: lap.laplacian_form(plan, f, None, "dirichlet"), c16.P, c16.E, eta_reach=3, batch=256)
    asm = LaplacianAssembly(plan)
    Mh = asm.matrix("dirichlet")
    assert abs(Mp - Mh).max() <= 1e-13 * abs(Mh).max()
    assert h_symmetry(Mp, asm.H) <= 1e-14
    for kind in ("dirichlet", "neumann"):
        assert h_symmetry(asm.matrix(kind), asm.H) <= 1e-14
    # coefficient and shell rate keep the symmetry
    asm2 = LaplacianAssembly(plan, c16.coeff, 3.0)
    assert h_symmetry(asm2.matrix("dirichlet"), asm2.H) <= 1e-14


def test_energy_identity_green_identity_and_constants(c16):
    """Fields of one ``K3`` call (dirichlet, neumann, dirichlet): H-symmetry of two Dirichlet fields, the discrete divergence
    theorem for the Neumann field, and the constants (``L 1 = 0`` for zero Neumann data and for matching Dirichlet data)."""
    plan = c16.plan
    rng = np.random.default_rng(5)
    E, P, N = c16.E, c16.P, c16.N
    H = (np.asarray(plan.Hp) * plan.structure.deta)[..., None]
    zero = np.zeros((E, N, 3))
    for coeff in (None, c16.coeff):
        f = rng.standard_normal((E, P, 3))
        out = -np.asarray(FORM(plan, f, lap.LaplacianBoundaryData(value=(zero,), conormal=(zero,)), KINDS3, coeff, 1.0)) / H
        Lf, Lv = out[..., 0], out[..., 2]
        f0, v = f[..., 0], f[..., 2]
        h = H[..., 0]
        assert abs(np.sum(h * v * Lf) - np.sum(h * f0 * Lv)) <= 1e-12 * np.sqrt(np.sum(h * Lf**2) * np.sum(h * v**2))
        assert np.sum(h * f0 * Lf) < 0.0 and np.sum(h * v * Lv) < 0.0         # negative definite with zero data
        # discrete divergence theorem: sum H L f = sum Om g (conormal data g), any f
        g = rng.standard_normal((E, N, 3))
        Lg = -np.asarray(FORM(plan, f, lap.LaplacianBoundaryData(value=(zero,), conormal=(g,)), KINDS3, coeff, 1.0)) / H
        assert abs(np.sum(h * Lg[..., 1]) - np.sum(g[..., 1]) * TWO_PI / N * plan.structure.deta) \
            <= 1e-12 * np.sum(np.abs(h * Lg[..., 1]))
        # constants: Neumann field 1 with zero data, Dirichlet field 0 with the matching constant datum
        ones = np.ones((E, P, 3))
        Lc = -np.asarray(FORM(plan, ones, lap.LaplacianBoundaryData(value=(np.ones((E, N, 3)),), conormal=(zero,)), KINDS3,
                              coeff, 1.0)) / H
        assert np.abs(Lc).max() <= 1e-9 * np.abs(Lf).max()


def test_dirichlet_definiteness_needs_the_shell_penalty_and_neumann_null_space_is_the_constants(c16):
    asm = LaplacianAssembly(c16.plan)
    dirichlet = definiteness_audit(asm.matrix("dirichlet"), asm.H, k=3, method="shift-invert")
    assert not dirichlet["negative"] and dirichlet["lowest"][0] > 1.0
    neumann = definiteness_audit(asm.matrix("neumann"), asm.H, k=3, kind="neumann", method="shift-invert")
    assert not neumann["negative"] and neumann["null_defect"] <= 1e-14
    assert abs(neumann["lowest"][0]) <= 1e-10 * neumann["rho"] and neumann["lowest"][1] > 0.3     # one null direction only
    # lobpcg (the large-case path, plane-block preconditioned) agrees; the constants are constrained out
    lob = definiteness_audit(asm.matrix("neumann"), asm.H, k=3, kind="neumann", method="lobpcg", planes=c16.E)
    assert lob["method"] == "lobpcg" and abs(lob["lowest"][0] - neumann["lowest"][1]) <= 1e-5 * neumann["lowest"][1]
    lobd = definiteness_audit(asm.matrix("dirichlet"), asm.H, k=3, method="lobpcg", planes=c16.E)
    assert abs(lobd["lowest"][0] - dirichlet["lowest"][0]) <= 1e-5 * dirichlet["lowest"][0]
    # without the shell penalty the core's non-polynomial modes are null directions of the energy
    asm0 = LaplacianAssembly(c16.plan, None, 0.0)
    ev0 = definiteness_audit(asm0.matrix("dirichlet"), asm0.H, k=2, method="shift-invert")
    assert abs(ev0["lowest"][0]) <= 1e-9 * ev0["rho"]
    # a smooth polarization coefficient keeps both kinds semidefinite with the same constants-only null space
    asm1 = LaplacianAssembly(c16.plan, c16.coeff)
    assert not definiteness_audit(asm1.matrix("dirichlet"), asm1.H, k=2, method="lobpcg", planes=c16.E)["negative"]
    nc = definiteness_audit(asm1.matrix("neumann"), asm1.H, k=2, kind="neumann", method="lobpcg", planes=c16.E)
    assert not nc["negative"] and nc["lowest"][0] > 0.01 and nc["null_defect"] <= 1e-14


def test_plan_audit_reports_symmetry_definiteness_and_flags(c16):
    res = audit_laplacian_plan(c16.plan, k=2)
    assert res["flags"] == [] and res["seconds"] > 0
    for kind in ("dirichlet", "neumann"):
        assert res[kind]["h_symmetry"] <= 1e-14 and not res[kind]["negative"]
    M = LaplacianAssembly(c16.plan).matrix("dirichlet")
    h1 = np.ones(c16.E * c16.P)
    bad = definiteness_audit(-M, h1, k=2, method="lobpcg", planes=c16.E)
    assert bad["negative"] and bad["n_negative"] >= 1                         # a negative direction is flagged
    # a barely negative direction (the Rayleigh quotient of the lowest mode pushed through zero) is flagged by lobpcg, and a
    # strongly negative one by the dense solve of a small case
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    lam, vec = spla.eigsh(M, k=1, sigma=0.0, which="LM")
    tiny = M - sp.csr_matrix(2.0 * lam[0] * np.outer(vec[:, 0], vec[:, 0]))
    res = definiteness_audit(tiny, h1, k=2, method="lobpcg", planes=c16.E)
    assert res["negative"] and res["lowest"][0] < -0.5 * lam[0]
    small = tb.case(16, 4, with_exact=False).plan
    Ms = LaplacianAssembly(small).matrix("dirichlet")
    v = np.random.default_rng(0).standard_normal(Ms.shape[0])
    rho = abs(spla.eigsh(Ms, k=1, which="LM", return_eigenvectors=False)[0])
    far = definiteness_audit(Ms - sp.csr_matrix(np.outer(v, v)) * 3.0 * rho / (v @ v), np.ones(Ms.shape[0]), k=2,
                             method="dense")
    assert far["negative"] and far["lowest"][0] < -0.1 * rho
    with pytest.raises(ValueError, match="bound"):
        definiteness_audit(M, h1, method="shift-invert", max_unknowns=100)
    with pytest.raises(ValueError, match="planes"):
        definiteness_audit(M, h1, method="lobpcg")


# ------------------------------------------------------------------------------------------------ polynomial exactness
def test_polynomial_exactness_in_the_ring_interior_with_cross_terms_and_a_coefficient():
    n, E = 24, 8
    lay, plan = tb.flat_plan(n, E)
    a = tb.A_CART
    quartic = {(0, 0): 1.0, (1, 0): 0.3, (0, 1): -0.2, (2, 0): 1.0, (1, 1): 0.5, (0, 2): -1.0, (3, 0): 0.4, (1, 2): -0.3,
               (4, 0): 0.2, (2, 2): -0.7, (0, 4): 0.1}
    cubic = {k: v for k, v in quartic.items() if sum(k) <= 3}
    cof = {(0, 0): 1.0, (1, 0): 0.2, (0, 1): -0.1}
    x = lay.node_u * np.cos(lay.node_theta)
    y = lay.node_u * np.sin(lay.node_theta)
    wp = wall_points(lay, lay.walls[0])
    xw, yw = wp[..., 0] * np.cos(wp[..., 1]), wp[..., 0] * np.sin(wp[..., 1])
    Nc, N, m = lay.blocks[0].n_nodes, lay.blocks[1].N, lay.blocks[1].m
    lo, hi = 6 * N, (m - 7) * N                                           # rings 6 .. m - 7: clear of every closure and SAT
    H = np.asarray(plan.Hp) * plan.structure.deta

    def run(cf, coeff):
        f = np.broadcast_to(tb.poly(cf, x, y), (E, lay.P)).copy()
        bcd = lap.LaplacianBoundaryData(value=(tb.poly(cf, xw, yw),))
        return -np.asarray(FORM(plan, f, bcd, "dirichlet", coeff, 1.0)) / H

    one = np.ones((E, lay.P))
    for cf in (quartic, cubic):                                           # a_ij f_ij, exact to degree 4 in the interior
        fx = lambda dx, dy: tb.poly(cf, x, y, dx, dy)  # noqa: E731
        L0 = a[0, 0] * fx(2, 0) + 2 * a[0, 1] * fx(1, 1) + a[1, 1] * fx(0, 2)
        out = run(cf, one)
        err = np.abs(out - L0[None])[:, Nc:][:, lo:hi]
        assert err.max() <= 1e-9 * np.abs(L0).max() and np.abs(out - out[:1]).max() <= 1e-12 * np.abs(L0).max()
        assert np.abs(out - L0[None])[:, Nc:].max() > 1e-3 * np.abs(L0).max()   # closure rows are inexact: the test is sharp
    # with the coefficient c = 1 + 0.2 x - 0.1 y (face values interpolated from c A): div(c a grad f) = c a_ij f_ij + a_ij c_i f_j
    fx = lambda dx, dy: tb.poly(cubic, x, y, dx, dy)  # noqa: E731
    L0 = a[0, 0] * fx(2, 0) + 2 * a[0, 1] * fx(1, 1) + a[1, 1] * fx(0, 2)
    cx = tb.poly(cof, x, y)
    ref = cx * L0 + 0.2 * (a[0, 0] * fx(1, 0) + a[0, 1] * fx(0, 1)) - 0.1 * (a[1, 0] * fx(1, 0) + a[1, 1] * fx(0, 1))
    outc = run(cubic, np.broadcast_to(cx, (E, lay.P)).copy())
    assert np.abs(outc - ref[None])[:, Nc:][:, lo:hi].max() <= 1e-9 * np.abs(ref).max()


# ------------------------------------------------------------------------------------------------ Neumann data
def test_conormal_conversion_against_exact_data_and_the_oblique_term_against_finite_differences():
    c = tb.case(16, 16)
    plan, E, N = c.plan, c.E, c.N
    exact = c.wall_conormal
    out = np.asarray(lap.conormal_from_normal(plan, c.vals, c.wall_normal))
    scale = np.abs(exact).max(axis=(0, 1))
    assert (np.abs(out - exact).max(axis=(0, 1)) / scale <= 2e-3).all()           # 5e-4 at n = 16, 3e-5 at n = 32
    # the oblique part is not negligible: dropping it leaves a 15 - 20 % error
    only = np.asarray(plan.wall_alpha)[..., None] * c.wall_normal
    assert (np.abs(only - exact).max(axis=(0, 1)) / scale >= 0.1).all()
    # exact identity (A grad f)^u = alpha g_n + beta_th d_th f + beta_eta d_eta f with central finite differences of f
    wp = c.wall_pts.reshape(-1, 3)
    Aw, Jw, gw = tb._batched(tb._geometry_fn, wp)
    eps = 1e-5
    fd = []
    for axis in (1, 2):
        d = np.zeros(3)
        d[axis] = eps
        fd.append(((tb.values(wp + d) - tb.values(wp - d)) / (2 * eps)).reshape(E, N, -1))
    gw = gw.reshape(E, N, 3)
    Aw = Aw.reshape(E, N, 3, 3)[..., 0, :]
    alpha = Aw[..., 0] / np.sqrt(gw[..., 0])
    beta_th = Aw[..., 1] - Aw[..., 0] * gw[..., 1] / gw[..., 0]
    beta_eta = Aw[..., 2] - Aw[..., 0] * gw[..., 2] / gw[..., 0]
    rebuilt = alpha[..., None] * c.wall_normal + beta_th[..., None] * fd[0] + beta_eta[..., None] * fd[1]
    assert (np.abs(rebuilt - exact).max(axis=(0, 1)) / scale <= 1e-8).all()
    # the plan's wall data are the traces of the nodal metric: they converge to the exact ones
    assert np.abs(np.asarray(plan.wall_alpha) - alpha).max() <= 2e-3 * np.abs(alpha).max()
    # operator level (host assembly, equal to the JAX apply by the tests above): the oblique term of the physical-normal
    # form against the same finite differences
    asm = LaplacianAssembly(plan)
    om = asm.Om.reshape(E, N)
    Mp, Mc = asm.matrix("neumann", "physical"), asm.matrix("neumann", "conormal")
    for i in range(3):
        flux = beta_th * fd[0][..., i] + beta_eta * fd[1][..., i]
        want = -(asm.Tw.T @ (om * flux).ravel())
        got = (Mp - Mc) @ c.vals[..., i].ravel()
        assert np.abs(got - want).max() <= 1e-2 * np.abs(want).max()
    # equivalent conormal and physical-normal data give nearly the same action (they differ by the tangential-gradient error)
    for i in (0, 1):
        a_ = Mc @ c.vals[..., i].ravel() - asm.data_vector(c.wall_conormal[..., i], "neumann")
        b_ = Mp @ c.vals[..., i].ravel() - asm.data_vector(c.wall_normal[..., i], "neumann", "physical")
        Hf = c.H.ravel()
        assert np.sqrt(np.sum((a_ - b_) ** 2 / Hf) / np.sum(a_ ** 2 / Hf)) <= 2e-2


def test_static_accuracy_on_the_smooth_testbed(c16):
    """One resolution: the Dirichlet, conormal and physical-normal Neumann actions track the exact Laplacian (the orders are
    in the slow convergence test)."""
    for name, bcd, kind, tol in (("D", lap.LaplacianBoundaryData(value=(c16.wall_val,)), "dirichlet", 0.1),
                                 ("N", lap.LaplacianBoundaryData(conormal=(c16.wall_conormal,)), "neumann", 0.05),
                                 ("Np", lap.LaplacianBoundaryData(normal_derivative=(c16.wall_normal,)), "neumann", 0.05)):
        out = np.asarray(lap.laplacian_action(c16.plan, c16.vals, bcd, (kind,) * 3))
        for i in (0, 1):                                                  # n, Ti (omega is under-resolved at n = 16)
            assert c16.h_rel_error(out[..., i] - c16.lap[..., i], c16.lap[..., i]) <= tol, (name, i)
    # the polarization coefficient: div(c A grad f) / J
    out = np.asarray(lap.laplacian_action(c16.plan, c16.vals, lap.LaplacianBoundaryData(value=(c16.wall_val,)),
                                          ("dirichlet",) * 3, c16.coeff))
    for i in (0, 1):
        assert c16.h_rel_error(out[..., i] - c16.lap_c[..., i], c16.lap_c[..., i]) <= 0.2


# ------------------------------------------------------------------------------------------------ plan and JAX hygiene
def test_jit_takes_the_plan_as_an_argument_and_does_not_retrace_on_new_values(c16):
    traces = []

    def fn(plan, f, g, ck, cf):
        traces.append(1)
        return lap.laplacian_action(plan, f, lap.LaplacianBoundaryData(value=(g,)), "dirichlet", cf, ck)

    jf = jax.jit(fn)
    rng = np.random.default_rng(6)
    f = rng.standard_normal((c16.E, c16.P))
    g = rng.standard_normal((c16.E, c16.N))
    o1 = np.asarray(jf(c16.plan, f, g, 1.0, c16.coeff))
    o2 = np.asarray(jf(c16.plan, f, g, 0.3, c16.coeff))               # new c_kappa
    o3 = np.asarray(jf(c16.plan, f, g, 0.3, 1.7 * c16.coeff))         # new coefficient
    plan2 = sl.build_laplacian_plan(c16.layout, sl.nodal_laplacian_metric_from_callable(c16.layout, tb.synthetic_geometry))
    o4 = np.asarray(jf(plan2, f, g, 0.3, c16.coeff))                  # another plan of the same structure
    assert len(traces) == 1
    assert np.abs(o1 - o2).max() > 1e-6 * np.abs(o1).max() and np.abs(o2 - o3).max() > 1e-6 * np.abs(o2).max()
    assert np.abs(o3 - o4).max() > 1e-6 * np.abs(o3).max()
    closed = jax.make_jaxpr(fn)(c16.plan, f, g, 1.0, c16.coeff)
    assert max(np.size(cst) for cst in closed.consts) < c16.plan.A.size                # the plan arrays are arguments
    # the linear part is linear and differentiable: JVP = apply of the tangent with zero data
    lin = jax.jit(lap.laplacian_linear, static_argnames=("kinds", "neumann_mode"))
    t = rng.standard_normal(f.shape)
    _, jv = jax.jvp(lambda x: lap.laplacian_linear(c16.plan, x, "dirichlet", c16.coeff), (f,), (t,))
    assert rel(jv, lin(c16.plan, t, "dirichlet", c16.coeff)) <= 1e-12
    assert rel(lin(c16.plan, f + 2 * t, "dirichlet", c16.coeff), lin(c16.plan, f, "dirichlet", c16.coeff)
               + 2 * lin(c16.plan, t, "dirichlet", c16.coeff)) <= 1e-12


def test_plan_builder_validates_its_inputs():
    lay = build_family_a_layout(16, n_eta=4)
    P, E = lay.P, lay.n_eta
    good = sl.nodal_laplacian_metric_from_callable(lay, tb.synthetic_geometry)
    with pytest.raises(ValueError, match="needs A_log"):
        sl.build_laplacian_plan(lay, sl.LaplacianMetric(good.A_log[:, :-1], good.J_log))
    bad = good.J_log.copy()
    bad[0, 0] = 0.0
    with pytest.raises(ValueError, match="jacobian"):
        sl.build_laplacian_plan(lay, sl.LaplacianMetric(good.A_log, bad))
    with pytest.raises(ValueError, match="ginv_u"):
        sl.build_laplacian_plan(lay, sl.LaplacianMetric(good.A_log, good.J_log, good.A_log[:, :-1, :, 0]))
    from drbx.geometry.nodal_layout import build_nodal_layout

    wall_layout = build_nodal_layout(16, ((2, 16, 16),), n_eta=4, inner="wall")
    with pytest.raises(NotImplementedError, match="family-A"):
        sl.build_laplacian_plan(wall_layout, sl.LaplacianMetric(np.zeros((E, wall_layout.P, 3, 3)), np.ones((E, wall_layout.P))))
    plan_nometric = sl.build_laplacian_plan(lay, sl.LaplacianMetric(good.A_log, good.J_log))
    assert plan_nometric.wall_alpha is None
    with pytest.raises(ValueError, match="wall metric"):
        lap.laplacian_form(plan_nometric, np.zeros((E, P)), lap.LaplacianBoundaryData(normal_derivative=(np.zeros((E, 16)),)),
                           "neumann")
    st = sl.plane_keys(plan_nometric.structure)
    assert st[0].shape == (E * P,) and st[0][0] == -1 and st[1][P] == 1 and st[2][lay.blocks[0].n_nodes] == 0


def test_penalty_rule_bounds_the_exact_trace_constants():
    """The rule ``C_X = gamma_X * (local ratio)`` stays above the exact constants ``sup |F_X f|^2 / E_block(f)`` (and below
    1.6 times them) on the testbed; the exact values equal the prototype's calibration data at n = 16."""
    from drbx.validation.sbp_laplacian_audit import trace_constants

    c = tb.case(16, 4, with_exact=False)
    lp, st = c.plan, c.plan.structure
    exact = trace_constants(LaplacianAssembly(lp))
    assert abs(exact["C_c"] - 12.0837052) <= 1e-5 and abs(exact["C_r"] - 4.5929593) <= 1e-5
    assert abs(exact["C_w"] - 80.0945009) <= 1e-4
    tau, tau_w, rule = sl.penalty_rule(np.asarray(lp.auu_f), np.asarray(lp.A)[:, :st.Nc], float(sl.staggered_pair(st.m).wf[0]),
                                       st.du, st.p, st.R_c)
    for key in exact:
        assert exact[key] <= rule[key] <= 1.6 * exact[key], key
    assert abs(tau - float(lp.tau)) <= 1e-12 * tau and abs(tau_w - float(lp.tau_w)) <= 1e-12 * tau_w
    assert abs(tau - (rule["C_c"] / 4 + rule["C_r"] / 2)) <= 1e-12 * tau and abs(tau_w - 2 * rule["C_w"]) <= 1e-12 * tau_w
