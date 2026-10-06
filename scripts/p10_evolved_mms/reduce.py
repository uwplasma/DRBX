"""P10 evolved MMS, chunk C6 (reduce): turn a campaign tree of run directories into error tables, orders and a report.

Pure NumPy (no JAX import). Input: ``ROOT/{arm}/n{N}/{mode}/{pattern}/{source}/`` run directories as written by ``evolve.py``
(``run.json``, ``geometry.npz``, ``snapshots/snap_KK.npz``; see ``work/p09_integration_design_20261004/p10_c3_c6_contract.md``).
``source = continuum`` is the main run "N", ``source = discrete`` the control run "O".

    python -m p10_evolved_mms.reduce ROOT --out DIR [--expected-n 32 48 64]

writes ``analysis.json`` (everything), ``orders.csv`` (one row per arm, mode, pattern, field, measure, region, interval) and
``report.md`` (machine-generated tables). No verdicts are produced: the tables are evidence for the user.

Definitions (all norms are H-weighted, ``||a||_R = sqrt(sum_{nodes in R} H a^2)``; region ``global`` is the whole ``(E, P)`` domain,
the other regions are the masks of ``geometry.npz``; the final snapshot is the one with the largest ``KK`` and ``T`` its time):

* measure ``N-R``: ``q_continuum_run - q_exact`` (fields n, Te, Ti, Omega and, with ``psi`` in the snapshot, psi, phi);
  ``O-R``: ``q_discrete_run - q_exact``; ``N-O``: ``q_continuum_run(T) - q_discrete_run(T)``;
  ``ctrl-R``: ``psi_ctrl - psi_exact``, ``phi_ctrl - phi_exact`` (elliptic control, continuum run); ``tau0``: ``||tau(t=0)||``.
* ``abs`` = ``||err||_R``; ``rel_exact`` = ``abs / ||q_exact(T)||_R``; ``rel_change`` = ``abs / ||q_exact(T) - q_exact(0)||_R``
  (normalisers are those of the final and the first snapshots, also used for every earlier snapshot); ``max`` = ``max |err|``
  over the region with its ``(plane, node)`` location.
* order between ``N_a < N_b``: ``p = ln(e_a / e_b) / ln(N_b / N_a)`` (``None`` unless both errors are positive).
  Intervals: consecutive pairs of the available N, plus first-last when there are three or more.
* ``A_N = ||e_N(T)||_global / integral_0^T ||tau(t)||_global dt`` (trapezoid over the snapshot times), ``e_N`` = ``N-R``.
* scaled history ``e_N(t) (N / 32)^p`` with ``p`` the final-time 32-64 ``N-R`` order of the same field.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

SCHEMA = "drbx.p10-reduce-v1"
FIELDS = ("n", "Te", "Ti", "Omega")
EXTRA_FIELDS = ("psi", "phi")
CTRL_FIELDS = {"psi_ctrl": "psi_exact", "phi_ctrl": "phi_exact"}
SOURCES = ("continuum", "discrete")
MODE_ORDER = ("diffusion", "hyperbolic", "coupled")
ARM_LABELS = {"filtered": "gated", "raw": "reported"}
MEASURES = ("N-R", "N-O", "O-R", "ctrl-R", "tau0")
CSV_COLUMNS = ("arm", "mode", "pattern", "field", "measure", "region", "interval", "order", "e_a", "e_b")
DEFAULT_EXPECTED_N = (32, 48, 64)
REF_N = 32

DEFINITIONS = {
    "norm": "||a||_R = sqrt(sum_{nodes in R} H a^2); region 'global' = whole (E, P) domain, others from geometry.npz masks",
    "N-R": "continuum-source run minus exact, per snapshot",
    "O-R": "discrete-source run minus exact, per snapshot",
    "N-O": "continuum-source run minus discrete-source run at the final snapshot",
    "ctrl-R": "elliptic control (psi_ctrl, phi_ctrl) minus exact, continuum-source run",
    "tau0": "||tau|| at the first snapshot (t = 0)",
    "rel_exact": "abs / ||q_exact(T)||_R",
    "rel_change": "abs / ||q_exact(T) - q_exact(0)||_R",
    "order": "ln(e_a / e_b) / ln(N_b / N_a), None unless both errors > 0",
    "A_N": "||e_N(T)||_global / trapezoid integral of ||tau||_global over the snapshot times",
    "scaled_history": "e_N(t) * (N / %d)^p, p = final-time %d-64 order of N-R (global) of the same field" % (REF_N, REF_N),
    "intervals": "consecutive pairs of the available N, plus first-last for three or more",
}


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def order(e_a, n_a, e_b, n_b):
    """Observed order ``ln(e_a / e_b) / ln(n_b / n_a)``; ``None`` unless both errors are positive and finite."""
    if e_a is None or e_b is None:
        return None
    if not (e_a > 0 and e_b > 0 and math.isfinite(e_a) and math.isfinite(e_b)):
        return None
    return float(math.log(e_a / e_b) / math.log(n_b / n_a))


def intervals(ns):
    """Consecutive pairs of the sorted ``ns`` plus (first, last) when there are three or more."""
    ns = sorted(ns)
    out = [(a, b) for a, b in zip(ns[:-1], ns[1:])]
    if len(ns) > 2:
        out.append((ns[0], ns[-1]))
    return out


def interval_label(a, b):
    return f"{a}-{b}"


def trapezoid(t, f):
    return float(sum(0.5 * (f[i] + f[i + 1]) * (t[i + 1] - t[i]) for i in range(len(t) - 1)))


def _regions(H, region_names, region_masks):
    regs = [("global", None)]
    for i, name in enumerate(region_names):
        m = np.asarray(region_masks[i], dtype=bool)
        if name != "global" and m.any():
            regs.append((name, m))
    return regs


def _h_norms(a, H, regs):
    a2 = H * np.asarray(a, dtype=np.float64) ** 2
    return {name: float(np.sqrt(a2.sum() if m is None else a2[m].sum())) for name, m in regs}


def _max_regions(d, regs):
    ad = np.abs(np.asarray(d, dtype=np.float64))
    out = {}
    for name, m in regs:
        masked = ad if m is None else np.where(m, ad, -1.0)
        idx = np.unravel_index(int(np.argmax(masked)), masked.shape)
        out[name] = {"max": float(masked[idx]), "max_loc": [int(idx[0]), int(idx[1])]}
    return out


def _err_table(d, H, regs, norm):
    """``{region: {abs, rel_exact, rel_change, max, max_loc}}`` for the difference ``d`` with normalisers ``norm``."""
    e = _h_norms(d, H, regs)
    mx = _max_regions(d, regs)
    out = {}
    for name, _ in regs:
        nx = norm["exact"][name]
        nc = None if norm["change"] is None else norm["change"][name]
        out[name] = {"abs": e[name], "rel_exact": e[name] / nx if nx > 0 else None,
                     "rel_change": e[name] / nc if (nc is not None and nc > 0) else None, **mx[name]}
    return out


def _exact_fields(z):
    """``{field: q_exact array (E, P)}`` of a snapshot (controls use the exact psi / phi)."""
    q = np.asarray(z["q_exact"], dtype=np.float64)
    out = {f: q[..., i] for i, f in enumerate(FIELDS)}
    if "psi" in z.files:
        for f in EXTRA_FIELDS:
            out[f] = np.asarray(z[f + "_exact"], dtype=np.float64)
        for c, ex in CTRL_FIELDS.items():
            if c in z.files:
                out[c] = np.asarray(z[ex], dtype=np.float64)
    return out


def _value_fields(z):
    q = np.asarray(z["q"], dtype=np.float64)
    out = {f: q[..., i] for i, f in enumerate(FIELDS)}
    if "psi" in z.files:
        for f in EXTRA_FIELDS:
            out[f] = np.asarray(z[f], dtype=np.float64)
        for c in CTRL_FIELDS:
            if c in z.files:
                out[c] = np.asarray(z[c], dtype=np.float64)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------------
class RunData:
    """Metrics of one run plus the final-snapshot arrays kept for the N - O comparison."""

    def __init__(self):
        self.snapshots = []        # [{k, t, step, err: {field: table}, tau: {field: {region: norm}}}]
        self.H = None
        self.regs = None
        self.norms = None          # {field: {"exact": {region: v}, "change": {region: v} | None}}
        self.final_fields = None   # {field: (E, P)} (no controls)
        self.t_final = None


def _snapshot_files(run_dir: Path):
    out = {}
    for p in (run_dir / "snapshots").glob("snap_*.npz"):
        m = re.fullmatch(r"snap_(\d+)\.npz", p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def load_run(run_dir: Path) -> RunData:
    """Reduce all snapshots of one run directory (two passes: normalisers first, then per-snapshot tables)."""
    rd = RunData()
    with np.load(run_dir / "geometry.npz") as g:
        rd.H = np.asarray(g["H"], dtype=np.float64)
        names = [str(s) for s in g["region_names"]]
        masks = np.asarray(g["region_masks"], dtype=bool)
    rd.regs = _regions(rd.H, names, masks)
    files = _snapshot_files(run_dir)
    if not files:
        raise FileNotFoundError("no snapshots")
    k_final = max(files)
    with np.load(files[k_final]) as z:
        ex_T = _exact_fields(z)
    ex_0 = None
    if 0 in files and k_final != 0:
        with np.load(files[0]) as z:
            ex_0 = _exact_fields(z)
    rd.norms = {}
    for f, exT in ex_T.items():
        rd.norms[f] = {"exact": _h_norms(exT, rd.H, rd.regs),
                       "change": None if ex_0 is None else _h_norms(exT - ex_0[f], rd.H, rd.regs)}
    for k, path in files.items():
        with np.load(path) as z:
            vals, exs = _value_fields(z), _exact_fields(z)
            err = {f: _err_table(vals[f] - exs[f], rd.H, rd.regs, rd.norms[f]) for f in vals}
            tau = {}
            if "tau" in z.files:
                tq = np.asarray(z["tau"], dtype=np.float64)
                tau = {f: _h_norms(tq[..., i], rd.H, rd.regs) for i, f in enumerate(FIELDS)}
            rd.snapshots.append({"k": int(k), "t": float(z["t"]), "step": int(z["step"]), "err": err, "tau": tau})
            if k == k_final:
                rd.final_fields = {f: v for f, v in vals.items() if f not in CTRL_FIELDS}
                rd.t_final = float(z["t"])
    return rd


def _chunk_stats(chunks):
    def agg(key, fn):
        vals = [c[key] for c in chunks if c.get(key) is not None]
        return fn(vals) if vals else None
    return {"n_chunks": len(chunks), "max_cg_iterations": agg("max_cg_iterations", max),
            "max_cg_relative_residual": agg("max_cg_relative_residual", max), "all_converged": agg("all_converged", all),
            "min_n": agg("min_n", min), "min_Te": agg("min_Te", min), "min_Ti": agg("min_Ti", min),
            "all_finite": agg("all_finite", all)}


def _status_record(arm, n, mode, pattern, source, path: Path, root: Path):
    rec = {"arm": arm, "n": n, "mode": mode, "pattern": pattern, "source": source,
           "path": str(path.relative_to(root)), "present": path.is_dir(), "status": None, "stop_reason": None,
           "stop_step": None, "usable": False, "exclusion_reason": None}
    if not rec["present"]:
        rec["exclusion_reason"] = "missing"
        return rec
    rj = path / "run.json"
    if not rj.is_file():
        rec["exclusion_reason"] = "no run.json"
        return rec
    try:
        run = json.loads(rj.read_text())
    except (OSError, ValueError) as exc:
        rec["exclusion_reason"] = f"unreadable run.json: {exc}"
        return rec
    ident = run.get("identity") or {}
    rec.update({"status": run.get("status"), "stop_reason": run.get("stop_reason"), "stop_step": run.get("stop_step"),
                "identity": ident, "bundle_sha256": ident.get("bundle_sha256"), "git_commit": run.get("git_commit"),
                "params": run.get("params"), "dt": run.get("dt"), "T": run.get("T"), "nsteps": run.get("nsteps"),
                "chunk": run.get("chunk"), "stats": _chunk_stats(run.get("chunks") or [])})
    if run.get("status") != "complete":
        rec["exclusion_reason"] = f"status={run.get('status')}"
    else:
        rec["usable"] = True
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------------------------------------------------
def _subdirs(p: Path):
    return sorted(d for d in p.iterdir() if d.is_dir()) if p.is_dir() else []


def discover(root: Path):
    """``{arm: {(mode, pattern): set(N found)}}`` from the directory tree (directories only)."""
    out = {}
    for arm_d in _subdirs(root):
        groups = {}
        for nd in _subdirs(arm_d):
            m = re.fullmatch(r"n(\d+)", nd.name)
            if not m:
                continue
            for md in _subdirs(nd):
                for pd in _subdirs(md):
                    groups.setdefault((md.name, pd.name), set()).add(int(m.group(1)))
        out[arm_d.name] = groups
    return out


def _mode_key(mode):
    return (MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER), mode)


# ----------------------------------------------------------------------------------------------------------------------
# group reduction
# ----------------------------------------------------------------------------------------------------------------------
def _run_summary(rd: RunData):
    return {"t": [s["t"] for s in rd.snapshots], "snapshots": rd.snapshots}


def _final_tables(entry, measure):
    """``{field: {region: table}}`` for ``measure`` from a ``per_n`` entry (``None`` when unavailable)."""
    if measure in ("N-R", "O-R", "ctrl-R"):
        run = entry.get("discrete" if measure == "O-R" else "continuum")
        if run is None:
            return None
        errs = run["snapshots"][-1]["err"]
        keep = (lambda f: f in CTRL_FIELDS) if measure == "ctrl-R" else (lambda f: f not in CTRL_FIELDS)
        return {f: t for f, t in errs.items() if keep(f)} or None
    if measure == "N-O":
        return entry.get("N-O")
    if measure == "tau0":
        run = entry.get("continuum") or entry.get("discrete")
        if run is None or not run["snapshots"][0]["tau"]:
            return None
        return {f: {r: {"abs": v} for r, v in regs.items()} for f, regs in run["snapshots"][0]["tau"].items()}
    raise ValueError(measure)


def _order_rows(per_n, measure):
    vals = {}
    for key, entry in per_n.items():
        tabs = _final_tables(entry, measure)
        if tabs is not None:
            vals[int(key)] = tabs
    rows = []
    for a, b in intervals(vals):
        for f in vals[a]:
            for region in vals[a][f]:
                if f not in vals[b] or region not in vals[b][f]:
                    continue
                ea, eb = vals[a][f][region]["abs"], vals[b][f][region]["abs"]
                rows.append({"field": f, "measure": measure, "region": region, "interval": interval_label(a, b),
                             "order": order(ea, a, eb, b), "e_a": ea, "e_b": eb})
    return rows


def _histories(per_n, order_rows):
    out = {}
    fields = []
    for entry in per_n.values():
        run = entry.get("continuum")
        if run:
            for f in run["snapshots"][-1]["err"]:
                if f not in fields:
                    fields.append(f)
    for f in fields:
        series = {}
        for key, entry in per_n.items():
            run = entry.get("continuum")
            if run and f in run["snapshots"][-1]["err"]:
                series[int(key)] = ([s["t"] for s in run["snapshots"]],
                                    [s["err"][f]["global"]["abs"] for s in run["snapshots"]])
        if not series:
            continue
        ns = sorted(series)
        meas = "ctrl-R" if f in CTRL_FIELDS else "N-R"
        p = None
        for r in order_rows:
            if r["measure"] == meas and r["field"] == f and r["region"] == "global" and \
                    r["interval"] == interval_label(REF_N, 64):
                p = r["order"]
        scaled = None
        if p is not None:
            scaled = {str(n): [e * (n / REF_N) ** p for e in series[n][1]] for n in ns}
        by_snap = {}
        for a, b in intervals(ns):
            m = min(len(series[a][1]), len(series[b][1]))
            by_snap[interval_label(a, b)] = [order(series[a][1][k], a, series[b][1][k], b) for k in range(m)]
        out[f] = {"t": {str(n): series[n][0] for n in ns}, "e": {str(n): series[n][1] for n in ns}, "p": p,
                  "scaled": scaled, "orders_by_snapshot": by_snap}
    return out


def _derived(per_n):
    """Orders, O-R / N-R ratios, tau integrals, amplification factors and histories of one group."""
    rows = []
    for m in MEASURES:
        rows.extend(_order_rows(per_n, m))
    ratio, amp, tau_int = {}, {}, {}
    for key, entry in per_n.items():
        nr, orr = _final_tables(entry, "N-R"), _final_tables(entry, "O-R")
        if nr is not None and orr is not None:
            ratio[key] = {f: {r: (orr[f][r]["abs"] / nr[f][r]["abs"] if nr[f][r]["abs"] > 0 else None)
                              for r in nr[f] if f in orr and r in orr[f]} for f in nr if f in orr}
        run = entry.get("continuum")
        if run is not None and nr is not None and run["snapshots"][0]["tau"]:
            t = [s["t"] for s in run["snapshots"]]
            amp[key], tau_int[key] = {}, {}
            for f in FIELDS:
                integ = trapezoid(t, [s["tau"][f]["global"] for s in run["snapshots"]])
                tau_int[key][f] = integ
                amp[key][f] = nr[f]["global"]["abs"] / integ if integ > 0 else None
    return rows, ratio, amp, tau_int, _histories(per_n, rows)


def reduce_campaign(root, expected_n=DEFAULT_EXPECTED_N):
    """Reduce the campaign tree under ``root``; returns the ``analysis`` dict (JSON-serialisable)."""
    root = Path(root)
    analysis = {"schema": SCHEMA, "root": str(root), "expected_n": [int(n) for n in expected_n],
                "definitions": DEFINITIONS, "arms": {}, "runs": [], "missing_runs": [], "excluded_runs": [],
                "warnings": [], "groups": []}
    found = discover(root)
    for arm in sorted(found):
        groups = found[arm]
        analysis["arms"][arm] = {"label": ARM_LABELS.get(arm, arm)}
        for mode, pattern in sorted(groups, key=lambda mp: (_mode_key(mp[0]), mp[1])):
            ns = sorted(set(int(n) for n in expected_n) | groups[(mode, pattern)])
            per_n = {}
            for n in ns:
                entry = {}
                datas = {}
                for src in SOURCES:
                    path = root / arm / f"n{n}" / mode / pattern / src
                    rec = _status_record(arm, n, mode, pattern, src, path, root)
                    rd = None
                    if rec["usable"]:
                        try:
                            rd = load_run(path)
                        except (OSError, KeyError, ValueError) as exc:
                            rec["usable"], rec["exclusion_reason"] = False, f"load error: {exc}"
                    analysis["runs"].append(rec)
                    if not rec["usable"]:
                        (analysis["missing_runs"] if not rec["present"] else analysis["excluded_runs"]).append(
                            {k: rec[k] for k in ("arm", "n", "mode", "pattern", "source", "status", "stop_reason",
                                                 "stop_step", "exclusion_reason")})
                    datas[src] = rd
                    entry[src] = _run_summary(rd) if rd is not None else None
                entry["N-O"] = _n_minus_o(datas, analysis["warnings"], (arm, n, mode, pattern))
                per_n[str(n)] = entry
            rows, ratio, amp, tau_int, hist = _derived(per_n)
            analysis["groups"].append({
                "arm": arm, "label": ARM_LABELS.get(arm, arm), "mode": mode, "pattern": pattern, "ns": ns,
                "per_n": per_n, "orders": rows, "ratio_OR_NR": ratio, "tau_integral": tau_int,
                "amplification": amp, "histories": hist})
    return analysis


def _n_minus_o(datas, warnings, ident):
    c, d = datas.get("continuum"), datas.get("discrete")
    if c is None or d is None:
        return None
    if c.final_fields["n"].shape != d.final_fields["n"].shape:
        warnings.append(f"{ident}: N and O final arrays differ in shape; N-O skipped")
        return None
    if not np.isclose(c.t_final, d.t_final, rtol=1e-12, atol=0.0):
        warnings.append(f"{ident}: final times differ (N {c.t_final}, O {d.t_final})")
    out = {}
    for f, qc in c.final_fields.items():
        if f in d.final_fields:
            out[f] = _err_table(qc - d.final_fields[f], c.H, c.regs, c.norms[f])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------------------------------------------------
def orders_rows(analysis):
    rows = []
    for g in analysis["groups"]:
        for r in g["orders"]:
            rows.append({"arm": g["arm"], "mode": g["mode"], "pattern": g["pattern"], **{k: r[k] for k in CSV_COLUMNS[3:]}})
    return rows


def write_orders_csv(analysis, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in orders_rows(analysis):
            w.writerow(["" if r[c] is None else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in CSV_COLUMNS])


def _fe(x):
    return "-" if x is None else f"{x:.4e}"


def _fo(x):
    return "-" if x is None else f"{x:.4f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines + [""]


def _sorted_n(keys):
    return sorted(int(k) for k in keys)


def _order_lookup(g, measure):
    return {(r["field"], r["region"], r["interval"]): r["order"] for r in g["orders"] if r["measure"] == measure}


def _report_group(g, runs, lines):
    per_n = g["per_n"]
    ns = _sorted_n(per_n)
    lines += [f"### pattern `{g['pattern']}`", ""]
    # run status
    lines += ["#### Run status", ""]
    rows = []
    for rec in runs:
        st = rec.get("stats") or {}
        rows.append([rec["n"], rec["source"], "missing" if not rec["present"] else rec["status"], rec["stop_reason"] or "-",
                     "-" if rec["stop_step"] is None else rec["stop_step"], "yes" if rec["usable"] else "no",
                     st.get("max_cg_iterations", "-"), _fe(st.get("max_cg_relative_residual")), st.get("all_converged", "-"),
                     _fe(st.get("min_n")), _fe(st.get("min_Te")), _fe(st.get("min_Ti"))])
    lines += _table(["N", "source", "status", "stop_reason", "stop_step", "in orders", "max CG its", "max CG res",
                     "all converged", "min n", "min Te", "min Ti"], rows)
    excl = [r for r in runs if not r["usable"]]
    if excl:
        lines += ["Missing or excluded runs: " + "; ".join(
            f"n{r['n']}/{r['source']} ({r['exclusion_reason']}" + (f", {r['stop_reason']}" if r["stop_reason"] else "") + ")"
            for r in excl), ""]
    else:
        lines += ["Missing or excluded runs: none", ""]

    titles = {"N-R": "N - R (continuum-source run vs exact), final time T",
              "O-R": "O - R (discrete-source run vs exact), final time T",
              "N-O": "N - O (continuum-source vs discrete-source run), final time T",
              "ctrl-R": "Elliptic controls (psi_ctrl, phi_ctrl vs exact), final time T",
              "tau0": "Static truncation ||tau|| at t = 0"}
    ivs = [interval_label(a, b) for a, b in intervals(ns)]
    for measure in ("N-R", "N-O", "O-R", "ctrl-R", "tau0"):
        tabs = {n: _final_tables(per_n[str(n)], measure) for n in ns}
        avail = [n for n in ns if tabs[n] is not None]
        if not avail:
            continue
        if measure == "ctrl-R" and not any("psi_ctrl" in tabs[n] for n in avail):
            continue
        fields = [f for f in tabs[avail[0]]]
        lk = _order_lookup(g, measure)
        lines += [f"#### {titles[measure]}", "", "Absolute H-norm (global) and observed orders:", ""]
        lines += _table(["field"] + [f"N={n}" for n in ns] + [f"p {iv}" for iv in ivs],
                        [[f] + [_fe(tabs[n][f]["global"]["abs"]) if tabs[n] and f in tabs[n] else "-" for n in ns] +
                         [_fo(lk.get((f, "global", iv))) for iv in ivs] for f in fields])
        if measure != "tau0":
            for key, title in (("rel_exact", "relative to ||q_exact(T)||"), ("rel_change", "relative to ||q_exact(T) - q_exact(0)||")):
                lines += [f"Error {title} (global):", ""]
                lines += _table(["field"] + [f"N={n}" for n in ns],
                                [[f] + [_fe(tabs[n][f]["global"][key]) if tabs[n] and f in tabs[n] else "-" for n in ns]
                                 for f in fields])
            lines += ["Max norm (global) at (plane, node):", ""]
            lines += _table(["field"] + [f"N={n}" for n in ns],
                            [[f] + [(f"{tabs[n][f]['global']['max']:.4e} @ ({tabs[n][f]['global']['max_loc'][0]}, "
                                     f"{tabs[n][f]['global']['max_loc'][1]})") if tabs[n] and f in tabs[n] else "-" for n in ns]
                             for f in fields])
        # regions
        regions = [r for r in tabs[avail[0]][fields[0]] if r != "global"]
        if regions and ivs:
            lines += ["Orders per region (diagnostic):", ""]
            for iv in ivs:
                lines += [f"Interval {iv}:", ""]
                lines += _table(["region"] + fields, [[r] + [_fo(lk.get((f, r, iv))) for f in fields] for r in regions])
    # ratio
    if g["ratio_OR_NR"]:
        lines += ["#### O - R / N - R (global, final time)", ""]
        rn = [n for n in ns if str(n) in g["ratio_OR_NR"]]
        fields = list(g["ratio_OR_NR"][str(rn[0])])
        lines += _table(["field"] + [f"N={n}" for n in rn],
                        [[f] + [_fo(g["ratio_OR_NR"][str(n)].get(f, {}).get("global")) for n in rn] for f in fields])
    # amplification
    if g["amplification"]:
        lines += ["#### Amplification A_N = ||e_N(T)|| / integral of ||tau|| dt", ""]
        an = [n for n in ns if str(n) in g["amplification"]]
        lines += _table(["field"] + [f"N={n} integral" for n in an] + [f"N={n} A_N" for n in an],
                        [[f] + [_fe(g["tau_integral"][str(n)][f]) for n in an] + [_fe(g["amplification"][str(n)][f]) for n in an]
                         for f in FIELDS])
    # histories
    if g["histories"]:
        lines += ["#### Error histories e_N(t) (N - R, global H-norm)", ""]
        for f, h in g["histories"].items():
            ck = sorted(h["e"], key=int)
            nt = max(len(h["t"][k]) for k in ck)
            lines += [f"Field `{f}` (scaling exponent p = {_fo(h['p'])}):", ""]
            t_ref = h["t"][ck[0]]
            rows = [[f"N={k} e", *[_fe(v) for v in h["e"][k]]] for k in ck]
            if h["scaled"]:
                rows += [[f"N={k} e (N/{REF_N})^p", *[_fe(v) for v in h["scaled"][k]]] for k in ck]
            rows += [[f"p {iv}", *[_fo(v) for v in vals]] for iv, vals in h["orders_by_snapshot"].items()]
            lines += _table(["quantity"] + [f"t={t:.4g}" for t in t_ref] + [f"snap {k}" for k in range(len(t_ref), nt)], rows)


def render_report(analysis):
    lines = ["# P10 evolved MMS: reduction report", "",
             "Machine-generated tables; no verdicts. Norms are H-weighted; orders p = ln(e_a/e_b)/ln(N_b/N_a).", "",
             f"Root: `{analysis['root']}`; expected N: {analysis['expected_n']}", ""]
    if analysis["warnings"]:
        lines += ["Warnings:", ""] + [f"- {w}" for w in analysis["warnings"]] + [""]
    runs_by_group = {}
    for rec in analysis["runs"]:
        runs_by_group.setdefault((rec["arm"], rec["mode"], rec["pattern"]), []).append(rec)
    arms = {}
    for g in analysis["groups"]:
        arms.setdefault(g["arm"], {}).setdefault(g["mode"], []).append(g)
    for arm, modes in arms.items():
        label = analysis["arms"][arm]["label"]
        lines += [f"## Arm `{arm}`" + (f" ({label})" if label != arm else ""), ""]
        for mode, groups in modes.items():
            lines += [f"### Mode `{mode}`", ""]
            for g in groups:
                runs = sorted(runs_by_group[(g["arm"], g["mode"], g["pattern"])],
                              key=lambda r: (r["n"], SOURCES.index(r["source"])))
                _report_group(g, runs, lines)
    return "\n".join(lines) + "\n"


def write_outputs(analysis, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(analysis))
    (out / "analysis.json").write_text(json.dumps(analysis, indent=1))
    write_orders_csv(analysis, out / "orders.csv")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-n", type=int, nargs="+", default=list(DEFAULT_EXPECTED_N))
    args = ap.parse_args(argv)
    analysis = reduce_campaign(args.root, args.expected_n)
    write_outputs(analysis, args.out)
    print(f"wrote {args.out}/analysis.json, orders.csv, report.md "
          f"({len(analysis['groups'])} groups, {len(analysis['missing_runs'])} missing, "
          f"{len(analysis['excluded_runs'])} excluded runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
