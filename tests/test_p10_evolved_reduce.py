"""P10 evolved MMS, chunk C6 (reduce): synthetic campaign trees (E = 4, P = 50) following the run-directory contract; known orders,
relative measures, amplification factor, missing / stopped runs, coupled extras and the output files. NumPy only."""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "scripts"),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from p10_evolved_mms import reduce as R   # noqa: E402

E, P = 4, 50
T = 2.0
TIMES = T * np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
_rng = np.random.default_rng(11)
H = _rng.uniform(0.5, 2.0, (E, P))
WALL = np.zeros((E, P), dtype=bool)
WALL[:, 40:] = True
MASKS = {"all": np.ones((E, P), dtype=bool), "interior": ~WALL, "wall": WALL}
BASE = _rng.uniform(1.0, 2.0, (E, P, 4))
DELTA = _rng.uniform(-0.5, 0.5, (E, P, 4))
G = _rng.uniform(0.5, 1.5, (4, E, P))     # error shapes per field
G2 = _rng.uniform(0.5, 1.5, (4, E, P))
PSI0, PHI0 = _rng.uniform(1.0, 2.0, (2, E, P))
GE = _rng.uniform(0.5, 1.5, (4, E, P))    # error shapes of psi, phi, psi_ctrl, phi_ctrl


def q_exact(k):
    return BASE + (TIMES[k] / T) * DELTA


def write_run(path, *, err_of, arm="filtered", n=32, mode="diffusion", pattern="p0", source="continuum", status="complete",
              stop_reason=None, stop_step=None, tau_of=None, extras_of=None, times=TIMES, chunks=None):
    """Write one contract-conforming run directory. ``err_of(k)`` is the ``(E, P, 4)`` error ``q - q_exact`` at snapshot k."""
    path = Path(path)
    (path / "snapshots").mkdir(parents=True)
    names = np.array(list(MASKS))
    np.savez(path / "geometry.npz", H=H, region_names=names, region_masks=np.stack([MASKS[k] for k in MASKS]), n=n, n_eta=E)
    for k, t in enumerate(times):
        qx = BASE + (t / T) * DELTA
        data = dict(t=t, step=10 * k, q=qx + err_of(k), q_exact=qx,
                    tau=np.zeros((E, P, 4)) if tau_of is None else tau_of(k, t))
        if extras_of is not None:
            e = extras_of(k)
            px, hx = PSI0 + (t / T), PHI0 - (t / T)
            data.update(psi=px + e["psi"], phi=hx + e["phi"], psi_exact=px, phi_exact=hx,
                        psi_ctrl=px + e["psi_ctrl"], phi_ctrl=hx + e["phi_ctrl"])
        np.savez(path / "snapshots" / f"snap_{k:02d}.npz", **data)
    if chunks is None:
        chunks = [{"step_end": 25, "t_end": 1.0, "max_cg_iterations": 7, "max_cg_relative_residual": 1e-9, "all_converged": True,
                   "min_n": 0.9, "min_Te": 0.8, "min_Ti": 0.7, "all_finite": True, "wall_s": 1.0},
                  {"step_end": 50, "t_end": T, "max_cg_iterations": 9, "max_cg_relative_residual": 2e-9, "all_converged": True,
                   "min_n": 0.5, "min_Te": 0.6, "min_Ti": 0.4, "all_finite": True, "wall_s": 1.0}]
    run = {"schema": "drbx.p10-run-v1", "identity": {"bundle_sha256": f"sha-{arm}-{n}", "arm": arm, "n": n}, "mode": mode,
           "source": source, "pattern": pattern, "params": {"rho_star": 0.1, "tau": 1.0, "D": 0.01}, "dt": 0.04, "T": T,
           "nsteps": 50, "chunk": 25, "git_commit": "abc123", "status": status, "stop_reason": stop_reason,
           "stop_step": stop_step, "chunks": chunks, "timings": {}}
    (path / "run.json").write_text(json.dumps(run))


def run_dir(root, arm, n, mode, pattern, source):
    return Path(root) / arm / f"n{n}" / mode / pattern / source


def rows_of(analysis, arm="filtered", mode="diffusion", pattern="p0"):
    out = {}
    for g in analysis["groups"]:
        if (g["arm"], g["mode"], g["pattern"]) == (arm, mode, pattern):
            for r in g["orders"]:
                out[(r["field"], r["measure"], r["region"], r["interval"])] = r
    return out


def group_of(analysis, arm="filtered", mode="diffusion", pattern="p0"):
    return next(g for g in analysis["groups"] if (g["arm"], g["mode"], g["pattern"]) == (arm, mode, pattern))


def _field_err(shape_fields, k):
    out = np.zeros((E, P, 4))
    for i, f in enumerate(shape_fields):
        out[..., i] = f
    return out


def _hn(a, mask=None):
    w = H if mask is None else H * mask
    return math.sqrt(float(np.sum(w * a ** 2)))


# ---------------------------------------------------------------------------------------------------------------------
# 1. known orders
# ---------------------------------------------------------------------------------------------------------------------
def _err_N(n, k):
    s, w = n / 32.0, k / 5.0
    e = np.zeros((E, P, 4))
    e[..., 0] = w * (s ** -2 * G[0] * ~WALL + s ** -1 * G[0] * WALL)   # p = 2 interior, p = 1 in the wall region
    e[..., 1] = w * s ** -3 * G[1]                                      # p = 3
    e[..., 2] = w * s ** -2 * G[2]
    e[..., 3] = w * s ** -2 * G[3]
    return e


def _err_O(n, k):
    s, w = n / 32.0, k / 5.0
    e = _err_N(n, k)
    o = np.empty_like(e)
    o[..., 0] = 0.5 * e[..., 0]
    o[..., 1] = e[..., 1] + w * s ** -2 * G2[1]       # N - O = -w s^-2 G2: order 2
    o[..., 2] = 0.5 * e[..., 2]
    o[..., 3] = 2.0 * e[..., 3]
    return o


def test_known_orders(tmp_path):
    for n in (32, 48, 64):
        write_run(run_dir(tmp_path, "filtered", n, "diffusion", "p0", "continuum"), err_of=lambda k, n=n: _err_N(n, k), n=n)
        write_run(run_dir(tmp_path, "filtered", n, "diffusion", "p0", "discrete"), err_of=lambda k, n=n: _err_O(n, k), n=n,
                  source="discrete")
    an = R.reduce_campaign(tmp_path)
    rows = rows_of(an)
    ivs = ("32-48", "48-64", "32-64")
    exp = {  # (field, measure, region) -> order
        ("n", "N-R", "wall"): 1.0, ("n", "N-R", "interior"): 2.0, ("Te", "N-R", "global"): 3.0, ("Ti", "N-R", "global"): 2.0,
        ("Omega", "N-R", "global"): 2.0,
        ("n", "N-O", "wall"): 1.0, ("n", "N-O", "interior"): 2.0, ("Te", "N-O", "global"): 2.0, ("Ti", "N-O", "global"): 2.0,
        ("Omega", "N-O", "global"): 2.0,
        ("n", "O-R", "wall"): 1.0, ("n", "O-R", "interior"): 2.0, ("Ti", "O-R", "global"): 2.0, ("Omega", "O-R", "global"): 2.0}
    for (f, m, reg), p in exp.items():
        for iv in ivs:
            assert rows[(f, m, reg, iv)]["order"] == pytest.approx(p, abs=1e-12), (f, m, reg, iv)
    # Te O-R is a mix of orders 3 and 2: compare with the orders of the analytic norms
    for iv in ivs:
        a, b = map(int, iv.split("-"))
        ea, eb = _hn(_err_O(a, 5)[..., 1]), _hn(_err_O(b, 5)[..., 1])
        assert rows[("Te", "O-R", "global", iv)]["order"] == pytest.approx(math.log(ea / eb) / math.log(b / a), abs=1e-12)
    # an order-1 region and an order-2 region combine to something in between globally
    assert 1.0 < rows[("n", "N-R", "global", "32-64")]["order"] < 2.0
    # e_a, e_b are the absolute region H-norms
    r = rows[("Ti", "N-R", "wall", "32-48")]
    assert r["e_a"] == pytest.approx(_hn(_err_N(32, 5)[..., 2], WALL), rel=1e-12)
    assert r["e_b"] == pytest.approx(_hn(_err_N(48, 5)[..., 2], WALL), rel=1e-12)
    # O - R / N - R
    ratio = group_of(an)["ratio_OR_NR"]
    for n in ("32", "48", "64"):
        assert ratio[n]["Ti"]["global"] == pytest.approx(0.5, abs=1e-12)
        assert ratio[n]["Omega"]["global"] == pytest.approx(2.0, abs=1e-12)
        assert ratio[n]["n"]["wall"] == pytest.approx(0.5, abs=1e-12)
    # orders by snapshot: zero error at t = 0 (no order), order of the final time afterwards
    h = group_of(an)["histories"]["Ti"]
    assert h["orders_by_snapshot"]["32-64"][0] is None
    assert h["orders_by_snapshot"]["32-64"][1:] == pytest.approx([2.0] * 5, abs=1e-12)
    # scaled overlay collapses onto the N = 32 history when the order is exact
    assert h["p"] == pytest.approx(2.0, abs=1e-12)
    for n in ("48", "64"):
        assert h["scaled"][n] == pytest.approx(h["e"]["32"], rel=1e-10, abs=1e-14)


# ---------------------------------------------------------------------------------------------------------------------
# 2. relative measures and max norm
# ---------------------------------------------------------------------------------------------------------------------
def test_relative_measures_and_max_location(tmp_path):
    err = np.zeros((E, P, 4))
    err[2, 17, 2] = 0.3     # Ti: global max in the interior
    err[3, 45, 2] = 0.1     # Ti: max inside the wall region
    err[1, 3, 2] = -0.05
    write_run(run_dir(tmp_path, "raw", 32, "diffusion", "p0", "continuum"), err_of=lambda k: err * (k == 5), arm="raw")
    an = R.reduce_campaign(tmp_path, expected_n=(32,))
    snap = group_of(an, "raw")["per_n"]["32"]["continuum"]["snapshots"][-1]
    assert snap["t"] == pytest.approx(T) and snap["k"] == 5
    tab = snap["err"]["Ti"]
    e = math.sqrt(H[2, 17] * 0.09 + H[3, 45] * 0.01 + H[1, 3] * 0.0025)
    exT = q_exact(5)[..., 2]
    change = (q_exact(5) - q_exact(0))[..., 2]
    assert tab["global"]["abs"] == pytest.approx(e, rel=1e-12)
    assert tab["global"]["rel_exact"] == pytest.approx(e / _hn(exT), rel=1e-12)
    assert tab["global"]["rel_change"] == pytest.approx(e / _hn(change), rel=1e-12)
    assert tab["global"]["max"] == pytest.approx(0.3) and tab["global"]["max_loc"] == [2, 17]
    assert tab["wall"]["max"] == pytest.approx(0.1) and tab["wall"]["max_loc"] == [3, 45]
    assert tab["interior"]["max_loc"] == [2, 17]
    ew = math.sqrt(H[3, 45] * 0.01)
    assert tab["wall"]["abs"] == pytest.approx(ew, rel=1e-12)
    assert tab["wall"]["rel_change"] == pytest.approx(ew / _hn(change, WALL), rel=1e-12)
    assert tab["wall"]["rel_exact"] == pytest.approx(ew / _hn(exT, WALL), rel=1e-12)
    # untouched fields have zero error and no order
    assert snap["err"]["n"]["global"]["abs"] == 0.0
    assert not group_of(an, "raw")["orders"]


# ---------------------------------------------------------------------------------------------------------------------
# 3. amplification factor
# ---------------------------------------------------------------------------------------------------------------------
def test_amplification_factor_with_known_tau(tmp_path):
    times = T * np.array([0.0, 0.1, 0.3, 0.6, 0.8, 1.0])      # non-uniform; ||tau(t)|| linear in t -> trapezoid exact
    a, b = 3.0, 5.0
    shape = G[0]

    def tau_of(k, t):
        out = np.zeros((E, P, 4))
        out[..., 0] = (a + b * t) * shape
        out[..., 3] = 2.0 * (a + b * t) * shape
        return out

    err = np.zeros((E, P, 4))
    err[..., 0] = 0.07 * G2[0]
    err[..., 3] = 0.11 * G2[1]
    write_run(run_dir(tmp_path, "filtered", 32, "hyperbolic", "p0", "continuum"), err_of=lambda k: err * (k == 5),
              mode="hyperbolic", tau_of=tau_of, times=times)
    an = R.reduce_campaign(tmp_path, expected_n=(32,))
    g = group_of(an, mode="hyperbolic")
    integral = (a * T + 0.5 * b * T ** 2) * _hn(shape)
    assert g["tau_integral"]["32"]["n"] == pytest.approx(integral, rel=1e-12)
    assert g["tau_integral"]["32"]["Omega"] == pytest.approx(2 * integral, rel=1e-12)
    assert g["amplification"]["32"]["n"] == pytest.approx(0.07 * _hn(G2[0]) / integral, rel=1e-12)
    assert g["amplification"]["32"]["Omega"] == pytest.approx(0.11 * _hn(G2[1]) / (2 * integral), rel=1e-12)
    assert g["amplification"]["32"]["Te"] is None       # zero tau integral and zero error
    # tau history stored per snapshot
    taus = [s["tau"]["n"]["global"] for s in g["per_n"]["32"]["continuum"]["snapshots"]]
    assert taus == pytest.approx([(a + b * t) * _hn(shape) for t in times], rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# 4. missing and stopped runs
# ---------------------------------------------------------------------------------------------------------------------
def test_missing_and_stopped_runs(tmp_path):
    for n in (32, 48, 64):
        if n != 48:
            write_run(run_dir(tmp_path, "filtered", n, "diffusion", "p0", "continuum"), err_of=lambda k, n=n: _err_N(n, k), n=n)
        status = dict(status="stopped", stop_reason="nonfinite", stop_step=17) if n == 64 else {}
        write_run(run_dir(tmp_path, "filtered", n, "diffusion", "p0", "discrete"), err_of=lambda k, n=n: _err_O(n, k), n=n,
                  source="discrete", **status)
    an = R.reduce_campaign(tmp_path)
    rows = rows_of(an)
    nr = sorted({r[3] for r in rows if r[1] == "N-R"})
    assert nr == ["32-64"]
    assert sorted({r[3] for r in rows if r[1] == "O-R"}) == ["32-48"]     # the stopped N = 64 run is excluded
    assert not [r for r in rows if r[1] == "N-O"]                          # no N with both runs, apart from N = 32
    # tau is static: taken from the continuum run, else the discrete one, so N = 48 (discrete only) still enters
    assert sorted({r[3] for r in rows if r[1] == "tau0"}) == ["32-48", "32-64", "48-64"]
    miss = an["missing_runs"]
    assert [(m["n"], m["source"]) for m in miss] == [(48, "continuum")]
    exc = an["excluded_runs"]
    assert len(exc) == 1 and (exc[0]["n"], exc[0]["source"]) == (64, "discrete")
    assert exc[0]["status"] == "stopped" and exc[0]["stop_reason"] == "nonfinite" and exc[0]["stop_step"] == 17
    st = {(r["n"], r["source"]): r for r in an["runs"]}
    assert st[(64, "discrete")]["usable"] is False and st[(48, "continuum")]["present"] is False
    assert group_of(an)["per_n"]["64"]["discrete"] is None
    # per-run solver statistics from the chunks and identities
    s = st[(32, "continuum")]
    assert s["stats"]["max_cg_iterations"] == 9 and s["stats"]["max_cg_relative_residual"] == 2e-9
    assert s["stats"]["all_converged"] is True and s["stats"]["min_n"] == 0.5 and s["stats"]["min_Ti"] == 0.4
    assert s["bundle_sha256"] == "sha-filtered-32" and s["git_commit"] == "abc123" and s["dt"] == 0.04 and s["nsteps"] == 50
    assert s["params"]["rho_star"] == 0.1
    report = R.render_report(an)
    assert "nonfinite" in report and "n48" in report and "missing" in report


# ---------------------------------------------------------------------------------------------------------------------
# 5. coupled extras
# ---------------------------------------------------------------------------------------------------------------------
def _extras(n, k):
    s, w = n / 32.0, k / 5.0
    return {"psi": w * s ** -2 * GE[0], "phi": w * s ** -3 * GE[1],
            "psi_ctrl": w * s ** -4 * GE[2], "phi_ctrl": w * s ** -1 * GE[3]}


def test_coupled_extras_and_elliptic_controls(tmp_path):
    for n in (32, 48, 64):
        for src, fn in (("continuum", _err_N), ("discrete", _err_O)):
            write_run(run_dir(tmp_path, "filtered", n, "coupled", "p0", src), err_of=lambda k, n=n, fn=fn: fn(n, k), n=n,
                      mode="coupled", source=src, extras_of=lambda k, n=n: _extras(n, k))
    an = R.reduce_campaign(tmp_path)
    rows = rows_of(an, mode="coupled")
    for iv in ("32-48", "48-64", "32-64"):
        assert rows[("psi", "N-R", "global", iv)]["order"] == pytest.approx(2.0, abs=1e-12)
        assert rows[("phi", "N-R", "global", iv)]["order"] == pytest.approx(3.0, abs=1e-12)
        assert rows[("psi", "O-R", "global", iv)]["order"] == pytest.approx(2.0, abs=1e-12)
        assert rows[("psi_ctrl", "ctrl-R", "global", iv)]["order"] == pytest.approx(4.0, abs=1e-12)
        assert rows[("phi_ctrl", "ctrl-R", "wall", iv)]["order"] == pytest.approx(1.0, abs=1e-12)
        # psi and phi of the continuum run equal those of the discrete run (same extras): N - O vanishes
        assert rows[("psi", "N-O", "global", iv)]["order"] is None
        assert rows[("psi", "N-O", "global", iv)]["e_a"] == 0.0
        assert ("psi_ctrl", "N-R", "global", iv) not in rows         # controls are their own measure
    h = group_of(an, mode="coupled")["histories"]
    assert h["phi_ctrl"]["orders_by_snapshot"]["32-64"][1:] == pytest.approx([1.0] * 5, abs=1e-12)
    assert h["psi_ctrl"]["p"] == pytest.approx(4.0, abs=1e-12)
    assert set(h) == {"n", "Te", "Ti", "Omega", "psi", "phi", "psi_ctrl", "phi_ctrl"}
    # exact normalisers use psi_exact / phi_exact
    tab = group_of(an, mode="coupled")["per_n"]["32"]["continuum"]["snapshots"][-1]["err"]["psi_ctrl"]["global"]
    assert tab["rel_exact"] == pytest.approx(tab["abs"] / _hn(PSI0 + 1.0), rel=1e-12)
    assert tab["rel_change"] == pytest.approx(tab["abs"] / _hn(np.ones((E, P))), rel=1e-12)
    report = R.render_report(an)
    assert "Elliptic controls" in report and "psi_ctrl" in report


# ---------------------------------------------------------------------------------------------------------------------
# 6. outputs
# ---------------------------------------------------------------------------------------------------------------------
def test_outputs(tmp_path):
    root = tmp_path / "campaign"
    for arm in ("filtered", "raw", "other"):
        for n in (32, 64):
            for src, fn in (("continuum", _err_N), ("discrete", _err_O)):
                write_run(run_dir(root, arm, n, "diffusion", "p0", src), err_of=lambda k, n=n, fn=fn: fn(n, k), n=n, arm=arm,
                          source=src, tau_of=lambda k, t: np.full((E, P, 4), 0.1 * (1 + t)))
    out = tmp_path / "out"
    assert R.main([str(root), "--out", str(out), "--expected-n", "32", "64"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["analysis.json", "orders.csv", "report.md"]

    an = json.loads((out / "analysis.json").read_text())
    assert an["schema"] == "drbx.p10-reduce-v1"
    assert json.loads(json.dumps(an)) == an
    run0 = next(r for r in an["runs"] if r["n"] == 32 and r["source"] == "continuum" and r["arm"] == "raw")
    assert run0["bundle_sha256"] == "sha-raw-32" and run0["git_commit"] == "abc123" and run0["nsteps"] == 50
    assert run0["dt"] == 0.04 and run0["params"]["tau"] == 1.0 and run0["identity"]["arm"] == "raw"
    # the in-memory result agrees with the file
    assert json.loads(json.dumps(R.reduce_campaign(root, (32, 64)))) == an

    with open(out / "orders.csv", newline="") as fh:
        rd = csv.DictReader(fh)
        assert rd.fieldnames == ["arm", "mode", "pattern", "field", "measure", "region", "interval", "order", "e_a", "e_b"]
        rows = list(rd)
    assert {r["arm"] for r in rows} == {"filtered", "raw", "other"}
    assert {r["measure"] for r in rows} == {"N-R", "N-O", "O-R", "tau0"}
    assert {r["interval"] for r in rows} == {"32-64"}
    r = next(r for r in rows if (r["arm"], r["field"], r["measure"], r["region"]) == ("raw", "Ti", "N-R", "global"))
    assert float(r["order"]) == pytest.approx(2.0, abs=1e-12) and float(r["e_a"]) > float(r["e_b"]) > 0
    tau0 = next(r for r in rows if (r["arm"], r["field"], r["measure"], r["region"]) == ("raw", "n", "tau0", "global"))
    assert tau0["order"] == "0.0" or float(tau0["order"]) == pytest.approx(0.0, abs=1e-12)    # n-independent tau

    report = (out / "report.md").read_text()
    assert re.search(r"^## Arm `filtered` \(gated\)$", report, re.M)
    assert re.search(r"^## Arm `raw` \(reported\)$", report, re.M)
    assert re.search(r"^## Arm `other`$", report, re.M)
    assert "### Mode `diffusion`" in report and "### pattern `p0`" in report
    for title in ("#### Run status", "#### N - R", "#### O - R", "#### N - O", "#### O - R / N - R",
                  "#### Amplification A_N", "#### Error histories", "Orders per region", "Static truncation"):
        assert title in report, title
    assert "| field | N=32 | N=64 | p 32-64 |" in report
    assert an["missing_runs"] == []
    # with the default expected N the absent N = 48 runs are listed as missing, not an error
    assert len(R.reduce_campaign(root)["missing_runs"]) == 6      # 3 arms x 2 sources
    assert not re.search(r"pass|fail", report, re.I)
