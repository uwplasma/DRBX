"""Fast synthetic tests of ``scripts/p08_step5_local/preconditioners`` and ``bench_preconditioners``.

The synthetic operator is a 3-D anisotropic flux-form operator ``A = M^-1 (K + eps S)`` on a small ``(u, theta, eta)``
grid: ``K`` is the symmetric positive face-flux stiffness (weak eta coupling), ``S`` a skew-symmetric perturbation (so the
M-weighted symmetric part, hence positive definiteness, is unchanged while ``MA`` is no longer symmetric), plus wall
conductances (Dirichlet) on the grid boundary, ``M = diag(V)`` with random cell volumes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[1]              # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from drbx.native.fci_perpendicular_p07_solve import (                                                # noqa: E402
    P07SolveConfig, p07_linear_system, solve_p07_dirichlet_jit)
from drbx.native.fci_perpendicular_p07_sparse import P07SparseOperator, save_p07_sparse              # noqa: E402
from p08_step5_local import bench_preconditioners as bp                                              # noqa: E402
from p08_step5_local import preconditioners as pc                                                    # noqa: E402
from p08_step5_local import solver_study as ss                                                       # noqa: E402

NU, NT, NE = 10, 12, 8
N = NU * NT * NE
CFG = P07SolveConfig(rtol=1e-10, restart=50, max_restarts=40)


def cell(i, j, k, dims=(NU, NT, NE)):
    return (i * dims[1] + j) * dims[2] + k


def build_operator(eps_eta=1e-2, skew=0.15, seed=0, dims=(NU, NT, NE)):
    """``(op, owner_plane)``; ``eps_eta = 0`` decouples the eta planes (block-diagonal matrix)."""
    nu, nt, ne = dims
    n = nu * nt * ne
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.6, 1.4, n)
    coef = (1.0, 0.7, eps_eta)
    kr, kc, kv = [], [], []
    for i in range(nu):
        for j in range(nt):
            for k in range(ne):
                for axis, (di, dj, dk) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                    i2, j2, k2 = i + di, j + dj, k + dk
                    if i2 >= nu or j2 >= nt or k2 >= ne or coef[axis] == 0.0:
                        continue
                    a, b = cell(i, j, k, dims), cell(i2, j2, k2, dims)
                    c = coef[axis] * rng.uniform(0.8, 1.2)
                    kr += [a, a, b, b]
                    kc += [a, b, a, b]
                    kv += [c, -c, -c, c]
                    s = skew * c                                  # skew-symmetric part, in-plane and across planes
                    kr += [a, b]
                    kc += [b, a]
                    kv += [s, -s]
    wall = [cell(i, j, k, dims) for i in range(nu) for j in range(nt) for k in range(ne)
            if i in (0, nu - 1) or j in (0, nt - 1) or k in (0, ne - 1)]
    kr += wall
    kc += wall
    kv += list(rng.uniform(0.8, 1.2, len(wall)))
    kmat = sp.csr_matrix((kv, (kr, kc)), shape=(n, n))
    a = (sp.diags(1.0 / vol) @ kmat).tocsr()
    a.sort_indices()
    empty = sp.csr_matrix((n, 0))
    op = P07SparseOperator("dirichlet", a, sp.csr_matrix((n, 3)), sp.csr_matrix((n, 6)), empty, vol,
                           np.zeros((3, 3)), np.zeros((0, 3)))
    plane = np.tile(np.arange(ne), nu * nt)
    return op, plane


@pytest.fixture(scope="module")
def world():
    op, plane = build_operator()
    rng = np.random.default_rng(5)
    phi = rng.normal(size=N)
    rhs = op.matrix @ phi
    return op, plane, phi, rhs


@pytest.fixture(scope="module")
def built(world):
    op, plane, _, _ = world
    return {
        "jacobi": pc.jacobi(op),
        "cheb4": pc.chebyshev(op, degree=4),
        "cheb8": pc.chebyshev(op, degree=8),
        "plane": pc.plane_block(op, owner_plane=plane),
        "sa_jac": pc.smoothed_aggregation(op, max_coarse=60, smoother="jacobi", keep_host=True),
        "sa_cheb": pc.smoothed_aggregation(op, max_coarse=60, smoother="chebyshev", keep_host=True),
        "sa_jac_vs": pc.smoothed_aggregation(op, max_coarse=60, smoother="jacobi", volume_scaling=True),
        "sa_cheb_f": pc.smoothed_aggregation(op, max_coarse=60, smoother="chebyshev", filter_smoothing=True),
        "ilu": pc.ilu(op),
    }


@pytest.fixture(scope="module")
def solves(world, built):
    op, _, phi, rhs = world
    system = p07_linear_system(op)
    out = {}
    for name, prec in built.items():
        x, info = solve_p07_dirichlet_jit(system, jnp.asarray(rhs), config=CFG, preconditioner=prec.apply)
        out[name] = (np.asarray(x), {k: np.asarray(v) for k, v in info.items()})
    return out


# ---------------------------------------------------------------------------
# synthetic operator sanity
# ---------------------------------------------------------------------------
def test_synthetic_operator_properties(world):
    op, plane, _, _ = world
    a, vol = op.matrix, op.owner_volume
    ma = (sp.diags(vol) @ a).toarray()
    asym = np.linalg.norm(ma - ma.T) / np.linalg.norm(ma + ma.T)
    assert 0.01 < asym < 0.5
    assert np.linalg.eigvalsh(0.5 * (ma + ma.T)).min() > 1e-6               # positive definite in the M sense
    assert np.all(a.diagonal() > 0)
    assert np.array_equal(plane[:NE], np.arange(NE))


def test_every_builder_returns_a_preconditioner(built):
    for name, prec in built.items():
        assert isinstance(prec, pc.Preconditioner)
        assert prec.setup_seconds > 0.0 and prec.nbytes > 0
        assert isinstance(prec.jax_native, bool) and isinstance(prec.info, dict)
        json.dumps(ss._json_safe(prec.info))                                 # info is JSON-able
    assert built["plane"].jax_native is False and built["ilu"].jax_native is False
    assert built["sa_jac"].jax_native and built["cheb4"].jax_native and built["jacobi"].jax_native


# ---------------------------------------------------------------------------
# apply: jittable and equal to a numpy reference
# ---------------------------------------------------------------------------
def _rand(seed=11):
    return np.random.default_rng(seed).normal(size=N)


def test_jacobi_apply(world, built):
    op, *_ = world
    r = _rand()
    z = np.asarray(jax.jit(built["jacobi"].apply)(jnp.asarray(r)))
    np.testing.assert_allclose(z, r / op.matrix.diagonal(), rtol=1e-14)


def _dense_cheb_reference(a: np.ndarray, r: np.ndarray, degree: int, lo: float, hi: float) -> np.ndarray:
    """z = B^-1 (I - E_k(B)) D^-1 r with E_k(x) = T_k((theta - x)/delta) / T_k(theta/delta), B = D^-1 A (dense)."""
    d = np.diag(a)
    b = a / d[:, None]
    theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
    ident = np.eye(len(r))
    xm = (theta * ident - b) / delta
    t_prev, t_cur = ident, xm
    for _ in range(1, degree):
        t_prev, t_cur = t_cur, 2.0 * xm @ t_cur - t_prev
    tk = t_cur if degree >= 1 else ident
    tk_scalar = np.polynomial.chebyshev.chebval(theta / delta, [0] * degree + [1])
    err = tk / tk_scalar
    return np.linalg.solve(b, (ident - err) @ (r / d))


@pytest.mark.parametrize("name,degree", [("cheb4", 4), ("cheb8", 8)])
def test_chebyshev_matches_dense_reference_and_jits(world, built, name, degree):
    op, *_ = world
    prec = built[name]
    lo, hi = prec.info["interval"]
    assert hi == pytest.approx(1.1 * prec.info["lambda_max"]) and lo == pytest.approx(prec.info["lambda_max"] / 30.0)
    r = _rand()
    z = np.asarray(jax.jit(prec.apply)(jnp.asarray(r)))
    ref = _dense_cheb_reference(op.matrix.toarray(), r, degree, lo, hi)
    np.testing.assert_allclose(z, ref, rtol=1e-8, atol=1e-8 * np.abs(ref).max())
    # fixed linear operator
    r2 = _rand(12)
    za, zb = (np.asarray(prec.apply(jnp.asarray(v))) for v in (r, r2))
    zab = np.asarray(prec.apply(jnp.asarray(r + 2.5 * r2)))
    np.testing.assert_allclose(zab, za + 2.5 * zb, rtol=1e-10, atol=1e-10 * np.abs(zab).max())


def test_chebyshev_lambda_max_is_close_to_true(world, built):
    op, *_ = world
    dinv_a = sp.diags(1.0 / op.matrix.diagonal()) @ op.matrix
    true = np.abs(np.linalg.eigvals(dinv_a.toarray())).max()
    est = built["cheb4"].info["lambda_max"]
    assert 0.8 * true <= est <= 1.05 * true


def test_chebyshev_degree_one_is_scaled_jacobi(world):
    op, *_ = world
    prec = pc.chebyshev(op, degree=1)
    lo, hi = prec.info["interval"]
    r = _rand()
    z = np.asarray(jax.jit(prec.apply)(jnp.asarray(r)))
    np.testing.assert_allclose(z, r / op.matrix.diagonal() / (0.5 * (lo + hi)), rtol=1e-13)


def test_plane_block_apply_matches_dense_blocks(world, built):
    op, plane, *_ = world
    r = _rand()
    z = np.asarray(jax.jit(built["plane"].apply)(jnp.asarray(r)))
    a = op.matrix.toarray()
    ref = np.zeros(N)
    for k in range(NE):
        idx = np.flatnonzero(plane == k)
        ref[idx] = np.linalg.solve(a[np.ix_(idx, idx)], r[idx])
    np.testing.assert_allclose(z, ref, rtol=1e-9, atol=1e-10 * np.abs(ref).max())
    info = built["plane"].info
    assert info["n_planes"] == NE and sum(info["block_sizes"]) == N and info["total_lu_nnz"] > 0
    with pytest.raises(ValueError):
        pc.plane_block(op)


def test_plane_block_exact_for_block_diagonal_matrix():
    op, plane = build_operator(eps_eta=0.0, skew=0.15, seed=2)
    # remove the (zero-weighted) cross-plane skew entries: the matrix must be exactly block diagonal
    a = op.matrix.tocoo()
    keep = plane[a.row] == plane[a.col]
    op2 = P07SparseOperator("dirichlet", sp.csr_matrix((a.data[keep], (a.row[keep], a.col[keep])), shape=(N, N)),
                            op.dirichlet_value, op.dirichlet_tangential, op.neumann_normal, op.owner_volume,
                            op.dirichlet_points, op.neumann_points)
    rhs = np.random.default_rng(3).normal(size=N)
    system = p07_linear_system(op2)
    prec = pc.plane_block(op2, owner_plane=plane)
    x, info = solve_p07_dirichlet_jit(system, jnp.asarray(rhs), config=CFG, preconditioner=prec.apply)
    assert bool(info["converged"]) and int(info["iterations"]) == 1
    np.testing.assert_allclose(op2.matrix @ np.asarray(x), rhs, rtol=1e-8, atol=1e-9)


def test_ilu_apply_matches_spilu(world, built):
    op, *_ = world
    r = _rand()
    ref = spla.spilu(sp.csc_matrix(op.matrix), drop_tol=1e-4, fill_factor=10.0).solve(r)
    z = np.asarray(jax.jit(built["ilu"].apply)(jnp.asarray(r)))
    np.testing.assert_allclose(z, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
    assert built["ilu"].info["lu_nnz"] > op.matrix.nnz * 0.5


def _np_vcycle(levels, level, b, smoother, omega, npre, npost, cdeg, ratio):
    """Independent numpy reference of the V-cycle of ``smoothed_aggregation`` (host hierarchy)."""
    lv = levels[level]
    if level == len(levels) - 1:
        return np.linalg.solve(lv.a.toarray(), b)
    a, dinv = lv.a, lv.dinv

    def sweep(x):
        res = b if x is None else b - a @ x
        if smoother == "jacobi":
            corr = (omega if omega is not None else 4.0 / (3.0 * lv.rho)) * dinv * res
        else:
            lo, hi = lv.rho / ratio, 1.1 * lv.rho
            theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
            sigma = theta / delta
            rho_c = 1.0 / sigma
            rr = dinv * res
            d = rr / theta
            corr = d.copy()
            for _ in range(1, cdeg):
                rr = rr - dinv * (a @ d)
                rho_n = 1.0 / (2.0 * sigma - rho_c)
                d = rho_n * rho_c * d + 2.0 * rho_n / delta * rr
                corr = corr + d
                rho_c = rho_n
        return corr if x is None else x + corr

    x = None
    for _ in range(npre):
        x = sweep(x)
    x = np.zeros_like(b) if x is None else x
    ec = _np_vcycle(levels, level + 1, lv.r @ (b - a @ x), smoother, omega, npre, npost, cdeg, ratio)
    x = x + lv.p @ ec
    for _ in range(npost):
        x = sweep(x)
    return x


@pytest.mark.parametrize("name,smoother", [("sa_jac", "jacobi"), ("sa_cheb", "chebyshev")])
def test_sa_vcycle_matches_numpy_reference_and_jits(built, name, smoother):
    prec = built[name]
    levels = prec.info["_host"]["levels"]
    r = _rand()
    z = np.asarray(jax.jit(prec.apply)(jnp.asarray(r)))
    ref = _np_vcycle(levels, 0, r, smoother, None, 1, 1, 2, 10.0)
    np.testing.assert_allclose(z, ref, rtol=1e-8, atol=1e-9 * np.abs(ref).max())


def test_sa_hierarchy_structure(built):
    for name in ("sa_jac", "sa_cheb"):
        info = built[name].info
        host = info["_host"]["levels"]
        sizes = [lv["n"] for lv in info["levels"]]
        assert info["n_levels"] == len(host) > 1
        assert sizes[0] == N and all(a > b for a, b in zip(sizes, sizes[1:]))
        assert sizes[-1] <= 60 and not info["coarsening_stalled"]
        assert info["operator_complexity"] > 1.0 and 1.0 < info["grid_complexity"] < 2.0
        assert set(info["setup_breakdown_s"]) >= {"aggregation", "galerkin", "prolongator"}
        for lvl in host[:-1]:
            p = lvl.p
            assert p.shape == (lvl.a.shape[0], host[host.index(lvl) + 1].a.shape[0])
            assert np.linalg.matrix_rank(p.toarray()) == p.shape[1]            # full column rank
            np.testing.assert_allclose((lvl.r @ (lvl.a @ p)).toarray(), host[host.index(lvl) + 1].a.toarray(),
                                       atol=1e-10)                              # Galerkin
        assert built[name].nbytes > 0


def test_sa_aggregation_covers_all_nodes_and_is_deterministic(world):
    op, *_ = world
    g = pc._strong_graph(sp.csr_matrix(op.matrix), 0.08)
    agg, nc = pc._aggregate(g)
    assert agg.min() == 0 and agg.max() == nc - 1 and len(np.unique(agg)) == nc
    agg2, nc2 = pc._aggregate(g)
    assert nc == nc2 and np.array_equal(agg, agg2)
    assert 1 < nc < N / 3
    # weak eta coupling: strong connections stay inside a plane
    plane = np.tile(np.arange(NE), NU * NT)
    coo = g.tocoo()
    assert np.all(plane[coo.row] == plane[coo.col])


def test_sa_aggregation_attaches_to_strongest_neighbour_and_makes_singletons():
    # 0-1 and 3-4 seed aggregates A and B; node 2 touches 1 (weight 1) and 4 (weight 5): pass 2 attaches it to B;
    # node 5 has no strong neighbour: singleton
    rows, cols, w = [0, 1, 3, 4, 2, 1, 2, 4], [1, 0, 4, 3, 1, 2, 4, 2], [1, 1, 1, 1, 1, 1, 5, 5]
    g = sp.csr_matrix((np.array(w, float), (rows, cols)), shape=(6, 6))
    agg, nc = pc._aggregate(g)
    assert nc == 3 and agg[0] == agg[1] and agg[3] == agg[4] == agg[2] and agg[0] != agg[3]
    assert np.sum(agg == agg[5]) == 1 and agg[5] not in (agg[0], agg[3])


def test_sa_no_prolongator_smoothing_and_validation(world):
    op, *_ = world
    prec = pc.smoothed_aggregation(op, max_coarse=60, prolongator_smoothing=False, keep_host=True)
    p = prec.info["_host"]["levels"][0].p
    assert np.allclose(np.asarray((p.T @ p).diagonal()), 1.0)                  # normalised piecewise-constant columns
    assert p.nnz == N
    with pytest.raises(ValueError):
        pc.smoothed_aggregation(op, smoother="gauss")


# ---------------------------------------------------------------------------
# inside the FGMRES solve
# ---------------------------------------------------------------------------
def test_all_candidates_converge_and_beat_jacobi(world, solves):
    op, _, phi, rhs = world
    iters = {k: int(v[1]["iterations"]) for k, v in solves.items()}
    for name, (x, info) in solves.items():
        assert bool(info["converged"]), (name, iters)
        assert float(info["relative_residual"]) <= 1e-10, name
        res = np.sqrt(np.sum(op.owner_volume * (rhs - op.matrix @ x) ** 2) / np.sum(op.owner_volume * rhs ** 2))
        assert res <= 2e-10, (name, res)
        err = np.sqrt(np.sum(op.owner_volume * (x - phi) ** 2) / np.sum(op.owner_volume * phi ** 2))
        assert err < 1e-6, (name, err)
    for name in ("cheb4", "cheb8", "plane", "sa_jac", "sa_cheb", "sa_jac_vs", "sa_cheb_f", "ilu"):
        assert iters[name] < iters["jacobi"], iters
    assert iters["cheb8"] <= iters["cheb4"] or iters["cheb4"] < iters["jacobi"]


# ---------------------------------------------------------------------------
# ring/plane block solvers: plane_ring_block (JAX plane solve) and block Gauss-Seidel across planes
# ---------------------------------------------------------------------------
RING_COUNTS = [1, 2, 2, 4, 4, 8, 8, 8]            # owners per ring per plane (agglomeration), n = 8
NPL = 8
EXPECTED_SUPER = [[0, 1, 2], [3, 4], [5], [6], [7]]   # greedy merge with B = 8


def build_ring_operator(counts=RING_COUNTS, n_planes=NPL, eps1=2e-2, eps2=8e-3, skew=0.12, seed=0, permute=True):
    """``(op, ring, plane, theta_key)``: rings of variable owner count, periodic theta coupling, radial reach 2 (3 for
    ring 0), periodic weak plane coupling at k +- 1 and k +- 2 (also across neighbouring owners), mild non-symmetry
    (skew part, so sym(MA) = K stays positive definite), owners randomly permuted."""
    rng = np.random.default_rng(seed)
    nodes = [(i, t) for i, c in enumerate(counts) for t in range(c)]
    node_id = {nd: m for m, nd in enumerate(nodes)}
    edges: dict = {}

    def edge(u, v, wgt):
        edges[tuple(sorted((u, v)))] = wgt * rng.uniform(0.8, 1.2)

    for i, c in enumerate(counts):
        if c > 2:
            for t in range(c):
                edge((i, t), (i, (t + 1) % c), 1.0)
        elif c == 2:
            edge((i, 0), (i, 1), 1.0)
    for i, c in enumerate(counts):
        for j in range(i + 1, min(len(counts), i + (3 if i == 0 else 2) + 1)):
            for t in range(c):
                tp = int((t + 0.5) / c * counts[j])
                edge((i, t), (j, tp), 0.5 ** (j - i - 1))
                if counts[j] > 1:
                    edge((i, t), (j, (tp + 1) % counts[j]), 0.3 * 0.5 ** (j - i - 1))
    nn = len(nodes)
    n = nn * n_planes
    gid = lambda nd, k: node_id[nd] * n_planes + k % n_planes                      # canonical: (ring, theta, plane)
    kr, kc, kv = [], [], []
    spr, spc, spv = [], [], []

    def link(a, b, c):
        kr.extend([a, b, a, b]); kc.extend([a, b, b, a]); kv.extend([c, c, -c, -c])
        spr.extend([a, b]); spc.extend([b, a]); spv.extend([skew * c, -skew * c])

    for (u, v), wgt in edges.items():
        for k in range(n_planes):
            link(gid(u, k), gid(v, k), wgt)
    pairs = [(u, v, wgt) for (u, v), wgt in edges.items()] + [(v, u, wgt) for (u, v), wgt in edges.items()]
    pairs += [(nd, nd, 1.0) for nd in nodes]
    for d, eps in ((1, eps1), (2, eps2)):
        for u, v, wgt in pairs:
            for k in range(n_planes):
                link(gid(u, k), gid(v, k + d), eps * wgt)
    ring = np.array([nd[0] for nd in nodes for _ in range(n_planes)])
    theta = np.array([nd[1] for nd in nodes for _ in range(n_planes)])
    plane = np.array([k for _ in nodes for k in range(n_planes)])
    for nd in nodes:
        if nd[0] == len(counts) - 1:
            for k in range(n_planes):
                kr.append(gid(nd, k)); kc.append(gid(nd, k)); kv.append(rng.uniform(0.5, 1.0))
    kmat = sp.csr_matrix((kv, (kr, kc)), shape=(n, n)) + sp.csr_matrix((spv, (spr, spc)), shape=(n, n))
    vol = rng.uniform(0.6, 1.4, n)
    a = (sp.diags(1.0 / vol) @ kmat).tocsr()
    perm = rng.permutation(n) if permute else np.arange(n)
    a = a[perm][:, perm].tocsr()
    a.sort_indices()
    vol, ring, plane, theta = vol[perm], ring[perm], plane[perm], theta[perm] * 7 + 5      # key, not a position
    op = P07SparseOperator("dirichlet", a, sp.csr_matrix((n, 3)), sp.csr_matrix((n, 6)), sp.csr_matrix((n, 0)), vol,
                           np.zeros((3, 3)), np.zeros((0, 3)))
    return op, ring, plane, theta


@pytest.fixture(scope="module")
def rworld():
    op, ring, plane, theta = build_ring_operator()
    n = op.matrix.shape[0]
    rng = np.random.default_rng(8)
    phi = rng.normal(size=n)
    return {"op": op, "ring": ring, "plane": plane, "theta": theta, "n": n, "phi": phi, "rhs": op.matrix @ phi,
            "dense": op.matrix.toarray()}


@pytest.fixture(scope="module")
def rbuilt(rworld):
    kw = dict(owner_ring=rworld["ring"], owner_plane=rworld["plane"], owner_theta=rworld["theta"])
    op = rworld["op"]
    return {
        "plane": pc.plane_block(op, owner_plane=rworld["plane"]),
        "plane_jax": pc.plane_ring_block(op, keep_host=True, **kw),
        "bgs_fwd": pc.plane_bgs_forward(op, keep_host=True, **kw),
        "bgs_mc4": pc.plane_bgs_multicolor(op, colors=4, keep_host=True, **kw),
        "bgs_mc4_sym": pc.plane_bgs_multicolor(op, colors=4, symmetric=True, **kw),
        "bgs_mc4_host": pc.plane_bgs_multicolor(op, colors=4, plane_solver="host", **kw),
    }


def _rrand(rw, seed=21):
    return np.random.default_rng(seed).normal(size=rw["n"])


def _dense_plane_solve(a, plane, r):
    ref = np.zeros_like(r)
    for k in np.unique(plane):
        idx = np.flatnonzero(plane == k)
        ref[idx] = np.linalg.solve(a[np.ix_(idx, idx)], r[idx])
    return ref


def test_ring_operator_properties(rworld):
    op, a, plane, ring = rworld["op"], rworld["dense"], rworld["plane"], rworld["ring"]
    vol = op.owner_volume
    ma = vol[:, None] * a
    assert 0.01 < np.linalg.norm(ma - ma.T) / np.linalg.norm(ma + ma.T) < 0.5          # mild non-symmetry
    assert np.linalg.eigvalsh(0.5 * (ma + ma.T)).min() > 1e-6                          # sym(MA) positive definite
    assert [int(np.sum((ring == i) & (plane == 0))) for i in range(len(RING_COUNTS))] == RING_COUNTS
    rows, cols = np.nonzero(a)
    dk = (plane[cols] - plane[rows]) % NPL
    assert set(np.unique(dk)) == {0, 1, 2, NPL - 2, NPL - 1}                           # plane reach +-2 (periodic)
    inpl = dk == 0
    assert abs(ring[cols] - ring[rows])[inpl].max() == 3
    in_norm = np.abs(a[rows[inpl], cols[inpl]]).sum() / np.abs(a[rows, cols]).sum()
    assert in_norm > 0.9                                                               # weak plane coupling


def test_ring_layout_partition_and_bandwidth(rworld, rbuilt):
    lay = pc.ring_layout(rworld["op"].matrix, rworld["ring"], rworld["plane"], rworld["theta"])
    assert lay.B == 8 and lay.S == len(EXPECTED_SUPER) and lay.uniform_planes
    assert [[int(lay.rings[r]) for r in np.flatnonzero(lay.super_of_ring == s)] for s in range(lay.S)] == EXPECTED_SUPER
    assert np.array_equal(lay.counts[:, 0], RING_COUNTS)
    # bandwidth from brute force on the dense matrix
    sup_of_ring = {r: s for s, grp in enumerate(EXPECTED_SUPER) for r in grp}
    rows, cols = np.nonzero(rworld["dense"])
    inpl = rworld["plane"][rows] == rworld["plane"][cols]
    sr = np.array([sup_of_ring[int(r)] for r in rworld["ring"]])
    assert lay.w == int(np.abs(sr[cols] - sr[rows])[inpl].max()) == 2
    assert lay.ring_reach.max() == 3 and lay.ring_reach[-1] == 2
    # slots: unique (plane, super, slot), theta_pos follows the key order, slots inside the block
    keys = {(int(p), int(s), int(sl)) for p, s, sl in zip(lay.plane_rank, lay.super_of, lay.slot)}
    assert len(keys) == rworld["n"] and lay.slot.max() < lay.B
    for i, c in enumerate(RING_COUNTS):
        sel = (rworld["ring"] == i) & (rworld["plane"] == 3)
        order = np.argsort(rworld["theta"][sel])
        assert np.array_equal(lay.theta_pos[sel][order], np.arange(c))
    info = rbuilt["plane_jax"].info
    assert (info["S"], info["B"], info["w"]) == (lay.S, lay.B, lay.w)
    assert info["padded_fraction"] == pytest.approx(1.0 - rworld["n"] / (lay.S * NPL * lay.B))
    json.dumps(ss._json_safe(info["super_rings"]))
    # bad input
    with pytest.raises(ValueError):
        pc.ring_layout(rworld["op"].matrix, rworld["ring"], rworld["plane"], rworld["theta"], max_block=4)
    with pytest.raises(ValueError):
        pc.ring_layout(rworld["op"].matrix, rworld["ring"][:-1], rworld["plane"], rworld["theta"])
    with pytest.raises(ValueError):
        pc.plane_ring_block(rworld["op"], owner_plane=rworld["plane"])


def test_padded_layout_round_trips(rworld, rbuilt):
    sol = rbuilt["plane_jax"].info["_solver"]
    lay = rbuilt["plane_jax"].info["_layout"]
    v = _rrand(rworld) + 3.0                                     # no accidental zeros
    g = np.asarray(sol.gather(jnp.asarray(v)))
    assert g.shape == (lay.S, NPL, lay.B)
    assert int(np.sum(g != 0)) == rworld["n"] and int(np.sum(g == 0)) == lay.S * NPL * lay.B - rworld["n"]
    np.testing.assert_array_equal(np.asarray(sol.scatter(jnp.asarray(g))), v)
    # every owner sits at (super-ring, plane, slot) of the layout
    for m in (0, 17, rworld["n"] - 1):
        assert g[lay.super_of[m], lay.plane_rank[m], lay.slot[m]] == v[m]


def test_plane_ring_block_matches_dense_and_host_plane_block(rworld, rbuilt):
    r = _rrand(rworld)
    prec = rbuilt["plane_jax"]
    z = np.asarray(jax.jit(prec.apply)(jnp.asarray(r)))
    ref = _dense_plane_solve(rworld["dense"], rworld["plane"], r)
    host = np.asarray(rbuilt["plane"].apply(jnp.asarray(r)))
    np.testing.assert_allclose(z, ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
    np.testing.assert_allclose(z, host, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
    assert prec.jax_native and prec.nbytes > 0 and prec.name == "plane_jax"
    info = prec.info
    w, big_b, n_s = info["w"], info["B"], info["S"]
    lay_bytes = 8 * n_s * NPL * (2 * w + 1) * big_b ** 2 + 4 * (n_s * NPL * big_b + rworld["n"])
    assert prec.nbytes == info["storage_bytes"] == lay_bytes
    assert info["max_block_cond1"] < 1e6 and set(info["setup_breakdown_s"]) >= {"layout", "assemble", "factor", "to_jax"}
    json.dumps(ss._json_safe({k: v for k, v in info.items() if not k.startswith("_")}))
    # no callbacks in the program
    jaxpr = str(jax.make_jaxpr(prec.apply)(jnp.asarray(r)))
    assert "callback" not in jaxpr and "scan" in jaxpr


def test_plane_ring_block_larger_block_and_non_uniform_input(rworld):
    # a larger block size only adds padding; the solve is unchanged
    kw = dict(owner_ring=rworld["ring"], owner_plane=rworld["plane"], owner_theta=rworld["theta"])
    r = _rrand(rworld, 4)
    ref = _dense_plane_solve(rworld["dense"], rworld["plane"], r)
    prec = pc.plane_ring_block(rworld["op"], max_block=12, **kw)
    assert prec.info["B"] == 12 and prec.info["S"] < len(EXPECTED_SUPER) + 1
    np.testing.assert_allclose(np.asarray(jax.jit(prec.apply)(jnp.asarray(r))), ref, rtol=1e-10,
                               atol=1e-10 * np.abs(ref).max())
    # the angular key may be any monotone key (here: reversed order): same exact solve
    prec2 = pc.plane_ring_block(rworld["op"], owner_ring=rworld["ring"], owner_plane=rworld["plane"],
                                owner_theta=-rworld["theta"])
    np.testing.assert_allclose(np.asarray(prec2.apply(jnp.asarray(r))), ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
    with pytest.raises(ValueError, match="ill-conditioned"):
        pc.plane_ring_block(rworld["op"], cond_max=1.0, **kw)


def test_plane_ring_block_on_tensor_grid_matches_plane_block(world):
    # second structure: one ring per u index (12 owners per ring and plane), reach 1 -> w = 1
    op, plane, *_ = world
    ring = np.repeat(np.arange(NU), NT * NE)
    theta = np.tile(np.repeat(np.arange(NT), NE), NU)
    prec = pc.plane_ring_block(op, owner_ring=ring, owner_plane=plane, owner_theta=theta)
    assert (prec.info["S"], prec.info["B"], prec.info["w"]) == (NU, NT, 1)
    r = _rand()
    np.testing.assert_allclose(np.asarray(jax.jit(prec.apply)(jnp.asarray(r))),
                               np.asarray(pc.plane_block(op, owner_plane=plane).apply(jnp.asarray(r))), rtol=1e-10,
                               atol=1e-10 * np.abs(r).max())


def test_multicolor_colors_are_independent(rworld, rbuilt):
    a, plane = rworld["dense"], rworld["plane"]
    cop = rbuilt["bgs_mc4"].info["_color_of_plane"]                   # color of every plane rank
    lay = rbuilt["bgs_mc4"].info["_layout"]
    assert list(lay.planes) == list(range(NPL)) and np.array_equal(cop, np.arange(NPL) % 4)
    for c in range(4):
        idx = np.flatnonzero(cop[plane] == c)
        sub = a[np.ix_(idx, idx)]
        same = plane[idx][:, None] == plane[idx][None, :]
        assert np.all(sub[~same] == 0.0)                              # no coupling between planes of one color
    assert rbuilt["bgs_mc4"].info["interplane_offsets"] == sorted({-2, -1, 1, 2, NPL - 2, NPL - 1, 2 - NPL, 1 - NPL})
    for bad in (2, 3):                                                # planes k, k + 2 (or the periodic wrap) coupled
        with pytest.raises(ValueError, match="coupled"):
            pc.plane_bgs_multicolor(rworld["op"], colors=bad, owner_ring=rworld["ring"], owner_plane=plane,
                                    owner_theta=rworld["theta"])
    with pytest.raises(ValueError):
        pc.plane_bgs_multicolor(rworld["op"], colors=4, plane_solver="gpu", owner_ring=rworld["ring"],
                                owner_plane=plane, owner_theta=rworld["theta"])


def _np_bgs(a, r, plane, color_of_plane, order):
    """Numpy reference: for c in order, z|_c += A_cc^-1 (r - A z)|_c with dense plane-block solves."""
    z = np.zeros_like(r)
    for c in order:
        res = r - a @ z
        for k in np.flatnonzero(color_of_plane == c):
            idx = np.flatnonzero(plane == k)
            z[idx] += np.linalg.solve(a[np.ix_(idx, idx)], res[idx])
    return z


def _bgs_reference(rworld, name, r):
    a, plane = rworld["dense"], rworld["plane"]
    if name == "bgs_fwd":
        return _np_bgs(a, r, plane, np.arange(NPL), range(NPL))
    cop = np.arange(NPL) % 4
    order = [0, 1, 2, 3] + ([3, 2, 1, 0] if "sym" in name else [])      # literal 0..3 then 3..0
    return _np_bgs(a, r, plane, cop, order)


@pytest.mark.parametrize("name", ["bgs_fwd", "bgs_mc4", "bgs_mc4_sym", "bgs_mc4_host"])
def test_bgs_variants_are_linear_and_match_numpy_reference(rworld, rbuilt, name):
    prec = rbuilt[name]
    r1, r2 = _rrand(rworld, 31), _rrand(rworld, 32)
    z1 = np.asarray(jax.jit(prec.apply)(jnp.asarray(r1)))
    ref = _bgs_reference(rworld, name.replace("_host", ""), r1)
    np.testing.assert_allclose(z1, ref, rtol=1e-9, atol=1e-10 * np.abs(ref).max())
    z2 = np.asarray(prec.apply(jnp.asarray(r2)))
    z12 = np.asarray(prec.apply(jnp.asarray(r1 - 1.7 * r2)))
    np.testing.assert_allclose(z12, z1 - 1.7 * z2, rtol=1e-9, atol=1e-10 * np.abs(z12).max())
    assert prec.name == name and prec.jax_native == (name != "bgs_mc4_host") and prec.nbytes > 0
    assert prec.info["n_colors"] == (NPL if name == "bgs_fwd" else 4)
    json.dumps(ss._json_safe({k: v for k, v in prec.info.items() if not k.startswith("_")}))


def test_bgs_one_sweep_structure(rworld, rbuilt):
    # forward sweep over colors: the last color's residual vanishes (A_cc z_c = r_c - A_c,other z_other)
    r = _rrand(rworld, 41)
    z = np.asarray(rbuilt["bgs_mc4"].apply(jnp.asarray(r)))
    res = r - rworld["dense"] @ z
    idx = np.flatnonzero(rworld["plane"] % 4 == 3)
    assert np.abs(res[idx]).max() < 1e-10 * np.abs(r).max()
    # symmetric sweep: color 0 is updated last, so its residual block vanishes instead
    zs = np.asarray(rbuilt["bgs_mc4_sym"].apply(jnp.asarray(r)))
    ress = r - rworld["dense"] @ zs
    idx0 = np.flatnonzero(rworld["plane"] % 4 == 0)
    assert np.abs(ress[idx0]).max() < 1e-10 * np.abs(r).max()
    # the sequential forward sweep is the full plane-by-plane Gauss-Seidel: last plane's residual vanishes
    zf = np.asarray(rbuilt["bgs_fwd"].apply(jnp.asarray(r)))
    resf = r - rworld["dense"] @ zf
    assert np.abs(resf[rworld["plane"] == NPL - 1]).max() < 1e-10 * np.abs(r).max()
    assert rbuilt["bgs_mc4"].info["plane_solves_per_apply"] == 4 and rbuilt["bgs_mc4_sym"].info["plane_solves_per_apply"] == 7
    assert rbuilt["bgs_mc4"].info["matvec_equivalents_per_apply"] < 1.0 + 1e-12


@pytest.fixture(scope="module")
def rsolves(rworld, rbuilt):
    op = rworld["op"]
    system = p07_linear_system(op)
    out = {}
    for name, prec in rbuilt.items():
        x, info = solve_p07_dirichlet_jit(system, jnp.asarray(rworld["rhs"]), config=CFG, preconditioner=prec.apply)
        out[name] = (np.asarray(x), {k: np.asarray(v) for k, v in info.items()})
    return out


def test_ring_plane_candidates_converge_in_fgmres(rworld, rsolves):
    op = rworld["op"]
    iters = {k: int(v[1]["iterations"]) for k, v in rsolves.items()}
    for name, (x, info) in rsolves.items():
        assert bool(info["converged"]) and float(info["relative_residual"]) <= 1e-10, (name, iters)
        res = np.linalg.norm(rworld["rhs"] - op.matrix @ x) / np.linalg.norm(rworld["rhs"])
        assert res < 1e-9, (name, res)
        assert np.linalg.norm(x - rworld["phi"]) / np.linalg.norm(rworld["phi"]) < 1e-6, name
    assert iters["plane_jax"] == iters["plane"]                        # same exact preconditioner
    assert iters["bgs_mc4_host"] == iters["bgs_mc4"]
    assert iters["bgs_mc4_sym"] <= iters["bgs_mc4"] <= iters["plane_jax"], iters
    assert iters["bgs_fwd"] <= iters["plane_jax"], iters


# ---------------------------------------------------------------------------
# benchmark: geometry loader and registry on a fake export
# ---------------------------------------------------------------------------
def test_owner_geometry_index():
    n = 4
    g = [1, 2, 2, 4]                                                   # theta cells per owner (agglomeration), per ring i
    raw = np.empty(n ** 3, dtype=np.int64)
    ids = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ids.setdefault((i, (j // g[i]), k), len(ids))
                raw[(i * n + j) * n + k] = ids[(i, j // g[i], k)]
    ring, plane, theta = bp.owner_geometry_index(raw, n, len(ids))
    for (i, jj, k), m in ids.items():
        assert (ring[m], plane[m], theta[m]) == (i, k, jj * g[i])
    assert np.array_equal(bp.owner_plane_index(raw, n, len(ids)), plane)
    bad = raw.copy()
    bad[(1 * n + 0) * n + 0] = raw[0]                                  # a ring-1 cell attached to a ring-0 owner
    with pytest.raises(ValueError):
        bp.owner_geometry_index(bad, n, len(ids))
    # ring layout of the agglomerated owner map: counts per (ring, plane) = n / g
    a = sp.identity(len(ids), format="csr")
    lay = pc.ring_layout(a, ring, plane, theta)
    assert list(lay.counts[:, 0]) == [n // x for x in g] and lay.B == 4 and lay.uniform_planes
    assert [[int(lay.rings[r]) for r in np.flatnonzero(lay.super_of_ring == s)] for s in range(lay.S)] == [[0], [1, 2], [3]]


def test_registry_contains_ring_plane_candidates():
    for name, (builder, params) in {"plane_jax": ("plane_ring_block", {}), "bgs_fwd": ("plane_bgs_forward", {}),
                                    "bgs_mc4": ("plane_bgs_multicolor", {"colors": 4, "symmetric": False}),
                                    "bgs_mc4_sym": ("plane_bgs_multicolor", {"colors": 4, "symmetric": True}),
                                    "bgs_mc4_host": ("plane_bgs_multicolor", {"colors": 4, "symmetric": False,
                                                                              "plane_solver": "host"})}.items():
        assert bp.REGISTRY[name] == (builder, params)
        assert builder in bp.BUILDERS and builder in bp.RING_BUILDERS


# ---------------------------------------------------------------------------
# benchmark CLI on a tiny fake export
# ---------------------------------------------------------------------------
def _fake_export(tmp_path, n=4):
    dims = (n, n, n)
    op, plane = build_operator(dims=dims, seed=7)
    nn = n ** 3
    # a raw -> owner map with the layout of the real export: raw index with eta fastest, some raw cells shared
    raw_to_owner = np.arange(nn)
    rng = np.random.default_rng(1)
    phi_bar = rng.normal(size=(nn, 2))
    export = tmp_path / "EXPORT"
    dpts = np.column_stack([np.arange(3.0), np.zeros(3), np.zeros(3)])
    op = P07SparseOperator("dirichlet", op.matrix, sp.csr_matrix(rng.normal(size=(nn, 3)) * (rng.random((nn, 3)) < 0.05)),
                           sp.csr_matrix((nn, 6)), sp.csr_matrix((nn, 0)), op.owner_volume, dpts, np.zeros((0, 3)))
    save_p07_sparse(export / f"N{n}" / "p07_dirichlet.npz", op, {})
    np.savez(export / f"N{n}" / "owner_map.npz", owner_volume=op.owner_volume, raw_to_owner=raw_to_owner,
             centers_u=np.zeros(nn), centers_theta=np.zeros(nn), centers_eta=np.zeros(nn))
    bc_val = rng.normal(size=(3, 2))
    study = tmp_path / "study"
    (study / f"N{n}").mkdir(parents=True)
    np.savez(study / f"N{n}" / "boundary_data.npz", dirichlet_value=bc_val, dirichlet_tangential=rng.normal(size=(3, 2, 2)),
             neumann_normal=np.zeros((0, 2)), dirichlet_points_sha256=ss._points_sha(dpts),
             neumann_points_sha256=ss._points_sha(np.zeros((0, 3))))
    from drbx.native.fci_perpendicular_p07_sparse import boundary_source
    from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData
    bsrc = boundary_source(op, BoundaryData(bc_val, np.load(study / f"N{n}" / "boundary_data.npz")["dirichlet_tangential"],
                                            np.zeros((0, 2))))
    d_act = op.matrix @ phi_bar + bsrc
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    np.savez(frozen / f"N{n}.owner_values.npz", values=phi_bar)
    np.savez(frozen / f"N{n}.global.npz", volume=op.owner_volume, D=d_act, O_q3=d_act + 1e-3 * rng.normal(size=d_act.shape))
    return export, study, frozen, phi_bar


def test_owner_plane_index():
    n = 3
    raw = np.array([(i % 3) + 3 * (i // 9) for i in range(27)])        # owner = k + 3 (raw // 9): cells share k
    assert np.array_equal(bp.owner_plane_index(raw, n, 9), np.tile(np.arange(3), 3))
    with pytest.raises(ValueError):
        bp.owner_plane_index(np.repeat(np.arange(9), 3), n, 9)          # raw cells 3m..3m+2 have planes 0, 1, 2
    with pytest.raises(ValueError):
        bp.owner_plane_index(np.zeros(27, dtype=int), n, 2)             # owner 1 has no raw cell


def test_benchmark_cli_end_to_end(tmp_path):
    export, study, frozen, phi_bar = _fake_export(tmp_path)
    out = tmp_path / "bench"
    cands = ["jacobi", "cheb4", "plane", "sa_jac", "ilu"]
    rc = bp.main(["--export", str(export), "--study-out", str(study), "--frozen", str(frozen), "--grids", "4",
                  "--candidates", *cands, "--rtol", "1e-10", "--restart", "50", "--max-restarts", "20", "--repeats", "1",
                  "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "bench_N4.json").read_text())
    assert res["n_owners"] == 64 and set(res["candidates"]) == set(cands)
    for name in cands:
        r = res["candidates"][name]
        assert r["error"] is None, (name, r["error"])
        assert r["converged"] and r["rel_residual"] < 1e-9 and r["rel_error"] < 1e-6
        assert r["time_1"] == pytest.approx(r["setup_s"] + r["warmup_s"] + r["solve_s"])
        assert r["time_100"] == pytest.approx(r["setup_s"] + r["warmup_s"] + 100 * r["solve_s"])
        assert r["speedup_1"] is not None
    jac = res["candidates"]["jacobi"]
    assert jac["speedup_1"] == pytest.approx(1.0) and jac["speedup_100"] == pytest.approx(1.0)
    md = (out / "bench_N4.md").read_text()
    for name in cands:
        assert f"| {name} |" in md
    assert "speedup_100" in md


def test_benchmark_records_candidate_failure(tmp_path, monkeypatch):
    export, study, frozen, _ = _fake_export(tmp_path)

    def boom(op, *, owner_plane=None, **kw):
        raise RuntimeError("broken builder")

    monkeypatch.setitem(bp.BUILDERS, "ilu", boom)
    out = tmp_path / "bench"
    rc = bp.main(["--export", str(export), "--study-out", str(study), "--frozen", str(frozen), "--grids", "4",
                  "--candidates", "ilu", "jacobi", "--repeats", "1", "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "bench_N4.json").read_text())
    assert "broken builder" in res["candidates"]["ilu"]["error"] and res["candidates"]["jacobi"]["error"] is None
    assert "FAILED" in (out / "bench_N4.md").read_text()


def test_benchmark_ring_plane_candidates_on_fake_export(tmp_path):
    export, study, frozen, phi_bar = _fake_export(tmp_path)
    out = tmp_path / "bench"
    cands = ["jacobi", "plane_jax", "bgs_fwd", "bgs_mc4", "bgs_mc4_sym", "bgs_mc4_host"]
    rc = bp.main(["--export", str(export), "--study-out", str(study), "--frozen", str(frozen), "--grids", "4",
                  "--candidates", *cands, "--rtol", "1e-10", "--restart", "50", "--max-restarts", "20", "--repeats", "1",
                  "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "bench_N4.json").read_text())
    for name in cands:
        r = res["candidates"][name]
        assert r["error"] is None, (name, r["error"])
        assert r["converged"] and r["rel_residual"] < 1e-9 and r["rel_error"] < 1e-6
    info = res["candidates"]["plane_jax"]["info"]
    assert (info["S"], info["B"], info["w"], info["n_planes"]) == (4, 4, 1, 4)
    assert res["candidates"]["bgs_mc4_host"]["jax_native"] is False and res["candidates"]["bgs_mc4"]["jax_native"] is True
    md = (out / "bench_N4.md").read_text()
    assert "Ring/plane block solvers" in md and "- plane_jax: S = 4 super-rings, B = 4, w = 1" in md


# ---------------------------------------------------------------------------
# float32 factors, preconditioner data as a jit argument, warm starts, bench_warm_start
# ---------------------------------------------------------------------------
from p08_step5_local import bench_warm_start as bw                                                   # noqa: E402


@pytest.fixture(scope="module")
def r32(rworld):
    kw = dict(owner_ring=rworld["ring"], owner_plane=rworld["plane"], owner_theta=rworld["theta"])
    return pc.plane_ring_block(rworld["op"], factor_dtype="float32", **kw)


def test_plane_jax32_apply_matches_float64_and_halves_storage(rworld, rbuilt, r32):
    r = _rrand(rworld, 51)
    z64 = np.asarray(jax.jit(rbuilt["plane_jax"].apply)(jnp.asarray(r)))
    z32 = jax.jit(r32.apply)(jnp.asarray(r))
    assert z32.dtype == jnp.float64                                           # float32 inside, float64 out
    z32 = np.asarray(z32)
    assert np.linalg.norm(z32 - z64) / np.linalg.norm(z64) < 1e-5
    assert r32.name == "plane_jax32" and r32.info["factor_dtype"] == "float32" and r32.jax_native
    assert r32.data["lo"].dtype == jnp.float32 and r32.data["dinv"].dtype == jnp.float32
    assert r32.data["idx"].dtype == jnp.int32
    i = r32.info
    assert r32.nbytes == i["storage_bytes"] == 4 * i["S"] * NPL * (2 * i["w"] + 1) * i["B"] ** 2 + 4 * (
        i["S"] * NPL * i["B"] + rworld["n"])
    assert r32.nbytes < 0.55 * rbuilt["plane_jax"].nbytes
    with pytest.raises(ValueError, match="factor_dtype"):
        pc.plane_ring_block(rworld["op"], factor_dtype="float16", owner_ring=rworld["ring"],
                            owner_plane=rworld["plane"], owner_theta=rworld["theta"])
    assert bp.REGISTRY["plane_jax32"] == ("plane_ring_block", {"factor_dtype": "float32"})


def test_plane_jax32_converges_in_fgmres(rworld, rsolves, r32):
    system = p07_linear_system(rworld["op"])
    x, info = solve_p07_dirichlet_jit(system, jnp.asarray(rworld["rhs"]), config=CFG, preconditioner=r32.apply)
    assert bool(info["converged"]) and float(info["relative_residual"]) <= 1e-10
    assert int(info["iterations"]) <= int(rsolves["plane_jax"][1]["iterations"]) + 2
    assert np.linalg.norm(np.asarray(x) - rworld["phi"]) / np.linalg.norm(rworld["phi"]) < 1e-6


def test_apply_with_data_and_data_argument_solve_match_closure_solve(rworld, rbuilt, r32):
    system = p07_linear_system(rworld["op"])
    rhs, r = jnp.asarray(rworld["rhs"]), jnp.asarray(_rrand(rworld, 52))
    for prec in (rbuilt["plane_jax"], r32):
        assert prec.data is not None and prec.apply_with is not None
        np.testing.assert_array_equal(np.asarray(prec.apply(r)), np.asarray(jax.jit(prec.apply_with)(prec.data, r)))
        run = pc.make_solver(system, prec, CFG)
        assert run.uses_data_argument
        x, info = run(rhs)
        xc, infoc = solve_p07_dirichlet_jit(system, rhs, config=CFG, preconditioner=prec.apply)
        np.testing.assert_allclose(np.asarray(x), np.asarray(xc), rtol=1e-12, atol=1e-13 * np.abs(np.asarray(xc)).max())
        assert int(info["iterations"]) == int(infoc["iterations"]) and bool(info["converged"])
    # boundary term and x0 are honoured
    bt = jnp.asarray(0.1 * _rrand(rworld, 53))
    run = pc.make_solver(system, rbuilt["plane_jax"], CFG)
    x, _ = run(rhs + bt, bt)
    xc, _ = solve_p07_dirichlet_jit(system, rhs + bt, boundary_term=bt, config=CFG, preconditioner=rbuilt["plane_jax"].apply)
    np.testing.assert_allclose(np.asarray(x), np.asarray(xc), rtol=1e-12, atol=1e-13 * np.abs(np.asarray(xc)).max())
    # preconditioners without data fall back to the closure apply
    jac = pc.jacobi(rworld["op"])
    run_j = pc.make_solver(system, jac, CFG)
    assert not run_j.uses_data_argument
    xj, _ = run_j(rhs)
    xjc, _ = solve_p07_dirichlet_jit(system, rhs, config=CFG, preconditioner=jac.apply)
    np.testing.assert_allclose(np.asarray(xj), np.asarray(xjc), rtol=1e-12, atol=1e-13 * np.abs(np.asarray(xjc)).max())


def test_data_argument_keeps_factors_out_of_the_program_constants(rworld, rbuilt):
    # qualitative stand-in for the warm-up difference: the closure apply embeds the factor arrays as program constants,
    # apply_with takes them as arguments (only small things may remain)
    prec = rbuilt["plane_jax"]
    r = jnp.asarray(_rrand(rworld, 54))

    def const_bytes(closed):
        return sum(int(np.asarray(c).nbytes) for c in closed.consts)

    closed_closure = jax.make_jaxpr(prec.apply)(r)
    closed_data = jax.make_jaxpr(prec.apply_with)(prec.data, r)
    factor_bytes = sum(int(prec.data[k].nbytes) for k in ("lo", "up", "dinv"))
    assert const_bytes(closed_closure) >= factor_bytes
    assert const_bytes(closed_data) < 0.01 * factor_bytes


def test_warm_start_converges_faster(rworld, rbuilt):
    system = p07_linear_system(rworld["op"])
    run = pc.make_solver(system, rbuilt["plane_jax"], CFG)
    phi, rhs = rworld["phi"], jnp.asarray(rworld["rhs"])
    _, cold = run(rhs)
    x, exact = run(rhs, x0=phi)                                               # x0 = exact solution
    assert int(exact["iterations"]) == 0 and bool(exact["converged"])
    np.testing.assert_allclose(np.asarray(x), phi, rtol=1e-12, atol=1e-12)
    near = phi + 1e-4 * np.linalg.norm(phi) / np.sqrt(rworld["n"]) * _rrand(rworld, 55)
    xw, warm = run(rhs, x0=near)
    assert bool(warm["converged"]) and 0 < int(warm["iterations"]) < int(cold["iterations"])
    assert np.linalg.norm(np.asarray(xw) - phi) / np.linalg.norm(phi) < 1e-6


def test_warm_start_trajectory_and_omega(rworld):
    op = rworld["op"]
    vol = op.owner_volume
    n = rworld["n"]
    rng = np.random.default_rng(7)
    grid = {"op": op, "phi_bar": rng.normal(size=(n, 3)), "bsrc": rng.normal(size=(n, 3))}
    p0, p1 = grid["phi_bar"][:, 0], grid["phi_bar"][:, 1]
    for delta in (1e-2, 1e-3):
        om = bw.omega_for_delta(p0, p1, vol, delta, 6)
        assert bw._mean_step_change(p0, p1, vol, om, 6) == pytest.approx(delta, rel=1e-9)
        for s, phi, rhs, bs in bw.trajectory(grid, om, 6):
            np.testing.assert_allclose(rhs - bs, op.matrix @ phi, rtol=1e-12, atol=1e-12)
            c, sn = np.cos(om * s), np.sin(om * s)
            np.testing.assert_allclose(bs, c * grid["bsrc"][:, 0] + sn * grid["bsrc"][:, 1])
    with pytest.raises(ValueError):
        bw.omega_for_delta(p0, p1, vol, 1e-2, 1)


def test_bench_warm_start_end_to_end_on_fake_export(tmp_path):
    export, study, frozen, _ = _fake_export(tmp_path)
    out = tmp_path / "warm"
    rc = bw.main(["--export", str(export), "--study-out", str(study), "--frozen", str(frozen), "--grids", "4",
                  "--preconds", "jacobi", "plane_jax", "plane_jax32", "--steps", "5", "--deltas", "1e-2", "1e-3",
                  "--rtols", "1e-10", "1e-8", "--max-restarts", "20", "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "warm_N4.json").read_text())
    assert set(res["preconditioners"]) == {"jacobi", "plane_jax", "plane_jax32"}
    for name, e in res["preconditioners"].items():
        assert e["error"] is None, (name, e["error"])
        assert len(e["runs"]) == 4                                            # 2 rtols x 2 deltas
        for r in e["runs"]:
            assert r["all_converged"] and len(r["cold"]["iterations"]) == 5 and len(r["warm"]["iterations"]) == 5
            assert r["max_error"] < 1e-5 and r["warm_its"] <= r["cold_its"]
            assert r["cold"]["iterations"][0] == r["warm"]["iterations"][0]   # step 0 is cold in both
            assert r["warm_speedup"] > 0
    md = (out / "warm_N4.md").read_text()
    assert "warm speedup" in md and "| plane_jax32 | 1e-08 | 0.001 |" in md and "relative to ||rhs||" in md
