"""Fast synthetic tests of ``scripts/p08_step5_local/solver_study`` (no real geometry, no environment build).

The synthetic operator is a conservative flux-form 2-D grid operator ``A = M^-1 S F`` (``F`` the face-flux rows with
zero-sum weights, ``S`` the +-1 face scatter), laid out like a ``P07SparseOperator``.  The Dirichlet kind adds wall
diagonal couplings, the Neumann kind adds one-sided wall fluxes (zero-sum weights, so ``A 1 = 0`` holds exactly) that
break the ``1^T M A = 0`` identity and hence ``l = V / sum V``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]              # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from drbx.native.fci_perpendicular_p07_solve import P07SolveConfig, direct_solve_p07                # noqa: E402
from drbx.native.fci_perpendicular_p07_sparse import (                                                # noqa: E402
    P07SparseOperator, boundary_source, load_p07_sparse, save_p07_sparse)
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData                           # noqa: E402
from p08_step5_local import solver_study as ss                                                        # noqa: E402

NX, NY, NF = 12, 10, 5
N = NX * NY


def _cell(i, j):
    return i * NY + j


def build_world(eps=0.15, wall_terms=True, seed=0):
    """``(op_d, op_n, bc, phi_bar, regions)``; ``eps = 0`` and ``wall_terms = False`` give the M-symmetric case."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.6, 1.4, N)
    rows, cols, vals = [], [], []                   # flux rows F (faces x cells), then A_int = M^-1 S F

    def add_face(a, b, extra, c):
        f = len(faces)
        faces.append((a, b))
        for cell, wgt in ((b, c), (a, -c)):
            rows.append(f); cols.append(cell); vals.append(wgt)
        if extra is not None and eps:
            a2, b2 = extra
            for cell, wgt in ((b2, eps * c), (a2, -eps * c)):
                rows.append(f); cols.append(cell); vals.append(wgt)

    faces = []
    for i in range(NX):
        for j in range(NY):
            if i + 1 < NX:
                add_face(_cell(i, j), _cell(i + 1, j), (_cell(i, j + 1), _cell(i + 1, j + 1)) if j + 1 < NY else None,
                         rng.uniform(0.8, 1.2))
            if j + 1 < NY:
                add_face(_cell(i, j), _cell(i, j + 1), (_cell(i + 1, j), _cell(i + 1, j + 1)) if i + 1 < NX else None,
                         rng.uniform(0.8, 1.2))
    nf = len(faces)
    flux = sp.csr_matrix((vals, (rows, cols)), shape=(nf, N))
    scat = sp.csr_matrix((np.concatenate([-np.ones(nf), np.ones(nf)]),
                          (np.array([a for a, _ in faces] + [b for _, b in faces]), np.tile(np.arange(nf), 2))),
                         shape=(N, nf))
    a_int = sp.diags(1.0 / vol) @ scat @ flux

    wall = [(i, j) for i in range(NX) for j in range(NY) if i in (0, NX - 1) or j in (0, NY - 1)]
    q_d = len(wall)
    # Dirichlet: wall flux c_w (u_i - g_q) -> diagonal + lift column
    cw = rng.uniform(0.8, 1.2, q_d)
    wall_cells = np.array([_cell(i, j) for i, j in wall])
    a_d = (a_int + sp.csr_matrix((cw / vol[wall_cells], (wall_cells, wall_cells)), shape=(N, N))).tocsr()
    q_n = q_d

    def rand_block(shape, density=0.04, scale=0.1):
        return sp.random(*shape, density=density, random_state=int(rng.integers(1 << 30)), format="csr") * scale

    b_val = (sp.csr_matrix((-cw / vol[wall_cells], (wall_cells, np.arange(q_d))), shape=(N, q_d))
             + rand_block((N, q_d))).tocsr()
    b_tan = rand_block((N, 2 * q_d))
    b_nn = rand_block((N, q_n))
    # Neumann: one-sided zero-sum wall fluxes (depend on u through the donors) when wall_terms
    a_n = a_int.tolil()
    if wall_terms:
        for (i, j), cell in zip(wall, wall_cells):
            di, dj = (1, 0) if i == 0 else (-1, 0) if i == NX - 1 else (0, 1) if j == 0 else (0, -1)
            n1, n2 = _cell(i + di, j + dj), _cell(i + 2 * di, j + 2 * dj)
            wa, wb = rng.uniform(0.1, 0.4, 2)
            a_n[cell, cell] += wa / vol[cell]
            a_n[cell, n1] += (wb - wa) / vol[cell]
            a_n[cell, n2] += -wb / vol[cell]
    a_n = a_n.tocsr()
    dpts = np.column_stack([np.arange(q_d, dtype=float), np.zeros(q_d), np.zeros(q_d)])
    npts = np.column_stack([np.arange(q_n, dtype=float), np.ones(q_n), np.zeros(q_n)])
    common = dict(owner_volume=vol, dirichlet_points=dpts, neumann_points=npts)
    op_d = P07SparseOperator("dirichlet", a_d, b_val, b_tan, sp.csr_matrix((N, q_n)), **common)
    op_n = P07SparseOperator("neumann", a_n, b_val * 0.5, b_tan * 0.5, b_nn, **common)
    bc = BoundaryData(rng.normal(size=(q_d, NF)), rng.normal(size=(q_d, 2, NF)), rng.normal(size=(q_n, NF)))
    phi_bar = rng.normal(size=(N, NF))
    phi_bar[:, -1] = 1.0                                              # the constant field
    wall_mask = np.zeros(N, bool)
    wall_mask[wall_cells] = True
    regions = {"boundary": wall_mask, "interior": ~wall_mask, "empty": np.zeros(N, bool)}
    return op_d, op_n, bc, phi_bar, regions


def perturbations(seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, N)[:, None] * np.arange(1, NF + 1)[None, :]
    return 1e-3 * np.sin(7 * x + rng.normal(size=(1, NF))), 2e-3 * np.cos(5 * x + rng.normal(size=(1, NF)))


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture(scope="module")
def sym_world():
    return build_world(eps=0.0, wall_terms=False, seed=3)


def wl2(x, v):
    return float(np.sqrt(np.sum(v * x * x) / np.sum(v)))


# ---------------------------------------------------------------------------
# synthetic operator sanity
# ---------------------------------------------------------------------------
def test_synthetic_world_properties(world, sym_world):
    op_d, op_n, *_ = world
    assert np.max(np.abs(op_n.matrix @ np.ones(N))) < 1e-13
    assert np.max(np.abs(op_d.matrix @ np.ones(N))) > 1e-3                         # wall coupling
    assert np.max(np.abs(sym_world[1].matrix @ np.ones(N))) < 1e-13


# ---------------------------------------------------------------------------
# matrix diagnostics
# ---------------------------------------------------------------------------
def test_diagnostics_symmetric_spd(sym_world):
    op_d, op_n, *_ = sym_world
    d = ss.matrix_diagnostics(op_d)
    assert d["n"] == N and d["nnz"] == op_d.matrix.nnz and d["eig_method"] == "dense"
    assert d["asymmetry"] < 1e-14
    assert d["positive_definite"] is True and d["lambda_min"] > 0 and d["ratio_min_max"] > 0
    dense = sla.eigh(0.5 * (np.diag(op_d.owner_volume) @ op_d.matrix.toarray()
                            + (np.diag(op_d.owner_volume) @ op_d.matrix.toarray()).T), np.diag(op_d.owner_volume),
                     eigvals_only=True)
    assert d["lambda_min"] == pytest.approx(dense[0], rel=1e-10) and d["lambda_max"] == pytest.approx(dense[-1], rel=1e-10)
    dn = ss.matrix_diagnostics(op_n)
    assert dn["A1_rel"] < 1e-13 and abs(dn["lambda_min"]) < 1e-12 * dn["lambda_max"]
    assert dn["positive_definite"] is False and dn["indefinite"] is False
    assert dn["lambda_second"] > 0


def test_diagnostics_asymmetric_and_eigsh_path(world, monkeypatch):
    op_d, op_n, *_ = world
    dense_d, dense_n = ss.matrix_diagnostics(op_d), ss.matrix_diagnostics(op_n)
    assert dense_d["asymmetry"] > 1e-3 and dense_d["positive_definite"] is True
    monkeypatch.setattr(ss, "DENSE_THRESHOLD", 10)                                  # exercise eigsh / shift-invert
    for dense, op in ((dense_d, op_d), (dense_n, op_n)):
        sp_diag = ss.matrix_diagnostics(op, k=3)
        assert sp_diag["eig_method"] == "eigsh" and not sp_diag["eig_errors"]
        assert sp_diag["lambda_max"] == pytest.approx(dense["lambda_max"], rel=1e-6)
        assert sp_diag["lambda_smallest"][-1] == pytest.approx(dense["lambda_smallest"][2], rel=1e-6, abs=1e-9)
        assert sp_diag["lambda_smallest"][0] == pytest.approx(dense["lambda_smallest"][0], rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------
def test_consistency(world):
    op_d, op_n, bc, phi_bar, _ = world
    frozen = op_d.matrix @ phi_bar + boundary_source(op_d, bc)
    c = ss.consistency(op_d, phi_bar, bc, frozen)
    assert c["max_abs"] < 1e-12 and set(c["fields"]) == set(ss.FIELD_NAMES)
    bad = ss.consistency(op_d, phi_bar, bc, frozen + 1e-3)
    assert bad["fields"]["field_b1"]["max_abs"] == pytest.approx(1e-3, rel=1e-6)
    assert bad["fields"]["field_b1"]["rel"] == pytest.approx(1e-3 / np.max(np.abs(frozen[:, 0] + 1e-3)), rel=1e-2)


# ---------------------------------------------------------------------------
# Dirichlet solves
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dirichlet_result(world):
    op_d, _, bc, phi_bar, regions = world
    d_rhs = op_d.matrix @ phi_bar + boundary_source(op_d, bc)
    do, dr = perturbations()
    rhs = {"D": d_rhs, "O_q3": d_rhs + do, "R": d_rhs + do + dr}
    cfg = P07SolveConfig(rtol=1e-12, restart=60, max_restarts=10)
    return ss.dirichlet_solves(op_d, phi_bar, bc, rhs, regions, config=cfg), (do, dr)


def test_dirichlet_discrete_consistency_returns_phi_bar(dirichlet_result):
    res, _ = dirichlet_result
    assert res["factor_seconds"] >= 0 and set(res["rhs"]) == {"D", "O_q3", "R"}
    for fname, ent in res["rhs"]["D"].items():
        assert ent["direct"]["rel_l2"] < 1e-12, fname
        for s in ("none", "jacobi"):
            assert ent[s]["converged"] and ent[s]["rel_l2"] < 1e-9, (fname, s)
            assert ent[s]["iterations"] >= 0 and ent[s]["wall_s"] >= 0 and ent[s]["relative_residual"] < 1e-10
            assert ent[s]["diff_vs_direct_rel"] < 1e-9


def test_dirichlet_errors_are_inverse_of_perturbations(world, dirichlet_result):
    op_d, _, _, phi_bar, regions = world
    res, (do, dr) = dirichlet_result
    ad = op_d.matrix.toarray()
    vol = op_d.owner_volume
    for j, fname in enumerate(ss.FIELD_NAMES):
        e_o = np.linalg.solve(ad, do[:, j])
        e_r = np.linalg.solve(ad, do[:, j] + dr[:, j])
        eo, er = res["rhs"]["O_q3"][fname]["direct"], res["rhs"]["R"][fname]["direct"]
        assert eo["l2"] == pytest.approx(wl2(e_o, vol), rel=1e-6) and er["l2"] == pytest.approx(wl2(e_r, vol), rel=1e-6)
        assert eo["max"] == pytest.approx(np.max(np.abs(e_o)), rel=1e-6)
        assert eo["rel_l2"] == pytest.approx(wl2(e_o, vol) / wl2(phi_bar[:, j], vol), rel=1e-6)
        assert eo["regions"]["boundary"] == pytest.approx(wl2(e_o[regions["boundary"]], vol[regions["boundary"]]),
                                                          rel=1e-6)
        assert eo["regions"]["empty"] is None
        for s in ("none", "jacobi"):
            assert res["rhs"]["R"][fname][s]["l2"] == pytest.approx(er["l2"], rel=1e-5)


def test_dirichlet_decomposition_identity(world, dirichlet_result):
    op_d, *_ = world
    res, (_, dr) = dirichlet_result
    ad = op_d.matrix.toarray()
    for j, fname in enumerate(ss.FIELD_NAMES):
        d = res["decomposition"][fname]
        assert d["identity_defect_max"] < 1e-12 and d["identity_defect_l2"] < 1e-12
        assert d["solve_error_from_O_minus_R"]["l2"] == pytest.approx(wl2(np.linalg.solve(ad, dr[:, j]),
                                                                           op_d.owner_volume), rel=1e-6)


def test_direct_solver_matches_package_oracle(world):
    op_d, _, bc, phi_bar, regions = world
    rhs = op_d.matrix @ phi_bar + boundary_source(op_d, bc) + 1e-3
    res = ss.dirichlet_solves(op_d, phi_bar, bc, {"X": rhs}, regions, solvers=("direct",))
    assert "none" not in res["rhs"]["X"]["field_b1"]
    one = BoundaryData(bc.dirichlet_value[:, :1], bc.dirichlet_tangential[:, :, :1], bc.neumann_normal[:, :1])
    phi = direct_solve_p07(op_d, rhs[:, 0], one)
    assert res["rhs"]["X"]["field_b1"]["direct"]["l2"] == pytest.approx(wl2(phi - phi_bar[:, 0], op_d.owner_volume),
                                                                         rel=1e-9)


# ---------------------------------------------------------------------------
# Neumann report
# ---------------------------------------------------------------------------
def _neumann_rhs(op_n, bc, phi_bar):
    n_rhs = op_n.matrix @ phi_bar + boundary_source(op_n, bc)
    do, dr = perturbations()
    return {"N": n_rhs, "O_q3": n_rhs + do, "R": n_rhs + do + dr}, n_rhs


def test_neumann_symmetric_left_null_is_volume_weight(sym_world):
    _, op_n, bc, phi_bar, regions = sym_world
    rhs, _ = _neumann_rhs(op_n, bc, phi_bar)
    rep = ss.neumann_report(op_n, phi_bar, bc, rhs, regions)
    assert rep["right_null"]["rel"] < 1e-13 and rep["bordered"]["ok"] and rep["bordered"]["cond1_est"] > 1
    assert not rep["bordered"]["likely_singular"]
    ln = rep["left_null"]
    assert ln["sum"] == pytest.approx(1.0, abs=1e-13) and abs(ln["mu"]) < 1e-12 and ln["residual_rel"] < 1e-12
    assert ln["vs_w"]["global"]["rel_l2"] < 1e-10 and ln["vs_w"]["global"]["max_rel"] < 1e-9
    w = op_n.owner_volume / op_n.owner_volume.sum()
    assert np.allclose(rep["_arrays"]["ell"], w, rtol=1e-9, atol=0)


def test_neumann_perturbed_left_null_and_lambda(world):
    _, op_n, bc, phi_bar, regions = world
    rhs, n_rhs = _neumann_rhs(op_n, bc, phi_bar)
    rep = ss.neumann_report(op_n, phi_bar, bc, rhs, regions)
    ln = rep["left_null"]
    ell = rep["_arrays"]["ell"]
    assert ln["sum"] == pytest.approx(1.0, abs=1e-12) and ln["residual_rel"] < 1e-10
    assert np.max(np.abs(op_n.matrix.T @ ell)) < 1e-10
    assert ln["vs_w"]["global"]["rel_l2"] > 1e-3
    assert ln["vs_w"]["regions"]["boundary"]["count"] == regions["boundary"].sum()
    assert ln["vs_w"]["regions"]["empty"]["count"] == 0 and ln["vs_w"]["regions"]["empty"]["rel_l2"] is None
    # independent dense left null vector
    z = sla.null_space(op_n.matrix.toarray().T)
    assert z.shape[1] == 1
    ell_dense = z[:, 0] / z[:, 0].sum()
    assert np.allclose(ell, ell_dense, atol=1e-9)
    bsrc = boundary_source(op_n, bc)
    for rname, fields in rep["solves"].items():
        for j, fname in enumerate(ss.FIELD_NAMES):
            v = fields[fname]
            expect = float(ell_dense @ (rhs[rname][:, j] - bsrc[:, j]))
            assert v["defect"] == pytest.approx(expect, abs=1e-9)
            assert abs(v["lambda_minus_defect"]) < 1e-9 and v["residual_max"] < 1e-9
            if rname == "N":
                assert abs(v["lambda"]) < 1e-9 and v["l2"] < 1e-9           # phi = phi_bar (same gauge)
            else:
                assert abs(v["lambda"]) > 1e-8
    # the O/R errors solve the bordered system with the perturbed rhs; compare to dense bordered solve
    k = np.block([[op_n.matrix.toarray(), np.ones((N, 1))], [(op_n.owner_volume / op_n.owner_volume.sum())[None, :],
                                                             np.zeros((1, 1))]])
    j = 1
    sol = np.linalg.solve(k, np.concatenate([rhs["R"][:, j] - bsrc[:, j],
                                             [op_n.owner_volume @ phi_bar[:, j] / op_n.owner_volume.sum()]]))
    assert rep["solves"]["R"][ss.FIELD_NAMES[j]]["lambda"] == pytest.approx(sol[-1], rel=1e-8)
    assert rep["solves"]["R"][ss.FIELD_NAMES[j]]["l2"] == pytest.approx(wl2(sol[:-1] - phi_bar[:, j],
                                                                             op_n.owner_volume), rel=1e-6)


def test_neumann_extra_kernel_is_flagged(sym_world):
    _, op_n, bc, phi_bar, regions = sym_world
    a = op_n.matrix.tolil()
    a[5, :] = 0.0                                          # cell 5 decoupled: e_5 joins the kernel of A
    for i in range(N):                                     # fold column 5 into the diagonals: keeps A 1 = 0
        if i != 5 and a[i, 5] != 0.0:
            a[i, i] += a[i, 5]
            a[i, 5] = 0.0
    a = sp.csr_matrix(a)
    assert np.max(np.abs(a @ np.ones(N))) < 1e-13 and np.linalg.matrix_rank(a.toarray()) == N - 2
    bad = P07SparseOperator("neumann", a, op_n.dirichlet_value, op_n.dirichlet_tangential,
                            op_n.neumann_normal, op_n.owner_volume, op_n.dirichlet_points, op_n.neumann_points)
    rep = ss.neumann_report(bad, phi_bar, bc, {"N": np.zeros((N, NF))}, regions)
    assert (not rep["bordered"]["ok"]) or rep["bordered"]["likely_singular"]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summarize_and_json_safe(world, dirichlet_result):
    op_d, op_n, bc, phi_bar, regions = world
    rhs_n, n_rhs = _neumann_rhs(op_n, bc, phi_bar)
    frozen_d = op_d.matrix @ phi_bar + boundary_source(op_d, bc)
    res = {"grid": 32, "n_owners": N,
           "diagnostics": {"dirichlet": ss.matrix_diagnostics(op_d), "neumann": ss.matrix_diagnostics(op_n)},
           "consistency": {"dirichlet": ss.consistency(op_d, phi_bar, bc, frozen_d),
                           "neumann": ss.consistency(op_n, phi_bar, bc, n_rhs)},
           "dirichlet": dirichlet_result[0], "neumann": ss.neumann_report(op_n, phi_bar, bc, rhs_n, regions),
           "seconds": {"total": 1.5}}
    safe = ss._json_safe(res)
    assert "_arrays" not in safe["neumann"]
    text = json.dumps(safe, allow_nan=False)
    assert json.loads(text)["n_owners"] == N
    md = ss.summarize({32: safe})
    for token in ("## N32", "### Matrix diagnostics", "| kind | n | nnz |", "### Consistency", "### Dirichlet solves",
                  "| rhs | field | direct L2 |", "jacobi: iters", "Reference mismatch", "### Neumann",
                  "| region (l vs w", "| rhs | field | lambda |", "bordered LU ok"):
        assert token in md, token
    assert "field_b1" in md and "physical" not in md.split("### Neumann")[0]
    assert "not available" in ss.summarize({"N48": {"n_owners": 3}})


# ---------------------------------------------------------------------------
# IO / CLI layer on a fake export
# ---------------------------------------------------------------------------
class FakeAdapter:
    def __init__(self, bc):
        self.bc = bc
        self.calls = 0

    def dirichlet(self, points):
        self.calls += 1
        idx = np.asarray(points)[:, 0].astype(int)
        grads = np.concatenate([np.full((len(idx), 1, NF), 7.0), self.bc.dirichlet_tangential[idx]], axis=1)
        return self.bc.dirichlet_value[idx], grads

    def normal(self, points):
        idx = np.asarray(points)[:, 0].astype(int)
        return self.bc.neumann_normal[idx]


@pytest.fixture()
def fake_export(tmp_path, world):
    op_d, op_n, bc, phi_bar, regions = world
    n = 4
    export = tmp_path / "EXPORT"
    save_p07_sparse(export / f"N{n}" / "p07_dirichlet.npz", op_d, {})
    save_p07_sparse(export / f"N{n}" / "p07_neumann.npz", op_n, {})
    np.savez(export / f"N{n}" / "owner_map.npz", owner_volume=op_d.owner_volume, raw_to_owner=np.arange(N),
             centers_u=np.zeros(N), centers_theta=np.zeros(N), centers_eta=np.zeros(N))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    do, dr = perturbations()
    d_act = op_d.matrix @ phi_bar + boundary_source(op_d, bc)
    n_act = op_n.matrix @ phi_bar + boundary_source(op_n, bc)
    np.savez(frozen / f"N{n}.owner_values.npz", values=phi_bar)
    np.savez(frozen / f"N{n}.global.npz", owner_ids=np.arange(N), volume=op_d.owner_volume, N=n_act, D=d_act,
             O_q3=d_act + do, R=d_act + do + dr, region_boundary=regions["boundary"],
             region_physical_wall=regions["boundary"] & (np.arange(N) % 2 == 0),
             region_ordinary=regions["interior"])
    return SimpleNamespaceLike(n=n, export=export, frozen=frozen, bc=bc, tmp=tmp_path, phi_bar=phi_bar)


class SimpleNamespaceLike:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_cli_end_to_end_and_boundary_cache(fake_export, monkeypatch):
    fx = fake_export
    adapters = []

    def make_adapter(n, phi_bar, input_root, sidecar):
        assert n == fx.n and phi_bar.shape == (N, NF)
        adapters.append(FakeAdapter(fx.bc))
        return adapters[-1]

    monkeypatch.setattr(ss, "_make_adapter", make_adapter)
    out = fx.tmp / "OUT"
    argv = ["--export", str(fx.export), "--out", str(out), "--grids", str(fx.n), "--frozen", str(fx.frozen),
            "--solvers", "direct", "jacobi", "--rtol", "1e-12", "--batch", "16", "--k-eigs", "3"]
    assert ss.main(argv) == 0
    res = json.loads((out / f"N{fx.n}" / "results.json").read_text())
    assert res["n_owners"] == N and len(adapters) == 1 and adapters[0].calls == 3     # 36 wall points / batch 16
    assert res["consistency"]["dirichlet"]["max_abs"] < 1e-12 and res["consistency"]["neumann"]["max_abs"] < 1e-12
    for fname, ent in res["dirichlet"]["rhs"]["D"].items():
        assert ent["direct"]["rel_l2"] < 1e-12 and ent["jacobi"]["converged"]
    assert "none" not in res["dirichlet"]["rhs"]["D"]["field_b1"]
    assert set(res["neumann"]["solves"]) == {"N", "O_q3", "R"}
    assert "interior" in res["neumann"]["left_null"]["vs_w"]["regions"]
    assert abs(res["neumann"]["solves"]["N"]["field_e3"]["lambda"]) < 1e-9
    with np.load(out / f"N{fx.n}" / "neumann_left_null.npz") as z:
        assert z["ell"].shape == (N,) and z["ell"].sum() == pytest.approx(1.0)
    with np.load(out / f"N{fx.n}" / "boundary_data.npz") as z:
        assert np.allclose(z["dirichlet_tangential"], fx.bc.dirichlet_tangential)
        assert np.allclose(z["neumann_normal"], fx.bc.neumann_normal)
    report = (out / f"report_N{fx.n}.md").read_text()
    assert "## N4" in report and "### Neumann" in report

    # second run: the cached boundary data is reused (the adapter must not be built)
    def no_adapter(*a, **k):
        raise AssertionError("boundary data should have been reused")

    monkeypatch.setattr(ss, "_make_adapter", no_adapter)
    assert ss.main(argv + ["--skip-neumann", "--solvers", "direct"]) == 0
    res2 = json.loads((out / f"N{fx.n}" / "results.json").read_text())
    assert "neumann" not in res2 and set(res2["diagnostics"]) == {"dirichlet"}
    assert "not run" in (out / f"report_N{fx.n}.md").read_text()


def test_cli_rejects_volume_mismatch(fake_export, monkeypatch):
    fx = fake_export
    monkeypatch.setattr(ss, "_make_adapter", lambda *a, **k: FakeAdapter(fx.bc))
    path = fx.frozen / f"N{fx.n}.global.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["volume"] = data["volume"] * (1 + 1e-9)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="owner_volume differs"):
        ss.main(["--export", str(fx.export), "--out", str(fx.tmp / "OUT"), "--grids", str(fx.n),
                 "--frozen", str(fx.frozen), "--solvers", "direct"])
    data["volume"] = data["volume"][:-1]
    np.savez(path, **data)
    with pytest.raises(ValueError, match="owners"):
        ss.main(["--export", str(fx.export), "--out", str(fx.tmp / "OUT2"), "--grids", str(fx.n),
                 "--frozen", str(fx.frozen), "--solvers", "direct"])


def test_boundary_cache_invalidated_by_point_table_change(tmp_path):
    bc_path = tmp_path / "bd.npz"
    dp, npt = np.arange(12.0).reshape(4, 3), np.arange(6.0).reshape(2, 3)

    class A:
        def dirichlet(self, p):
            return np.ones((len(p), 2)), np.zeros((len(p), 3, 2))

        def normal(self, p):
            return np.full((len(p), 2), 2.0)

    ss.load_or_build_boundary_data(bc_path, dp, npt, A, 2, batch=3)
    calls = []
    ss.load_or_build_boundary_data(bc_path, dp, npt, lambda: calls.append(1), 2)
    assert not calls
    ss.load_or_build_boundary_data(bc_path, dp + 1.0, npt, A, 2)
    with np.load(bc_path) as z:
        assert z["dirichlet_value"].shape == (4, 2) and z["dirichlet_tangential"].shape == (4, 2, 2)
