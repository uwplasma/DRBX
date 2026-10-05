"""Reduce the static gates: global H-norm errors, orders, pass/fail, constant controls, regions, reference budget.

Reads ``<out>/N{n}/references/<set>.npz`` and ``<out>/N{n}/static/<set>.npz`` for every available ``n`` in (32, 48, 64)
and writes ``<out>/summary.json`` (and prints the gate table). With ``e = N - R`` (nodal production minus exact nodal bracket)
and ``H = Hp * deta`` of the plan (block-frame ``Hp`` is the physical volume weight):

- ``E_n = sqrt(sum H e^2 / sum H)`` and order ``log(E_a / E_b) / log(n_b / n_a)`` on 32 -> 48 and 48 -> 64;
- **pass** (gating catalogues, nonconstant pairs): order >= 1.8 on both intervals; constant controls: ``max|N| <= tol *``
  bracket scale of the catalogue (largest ``max|R|``); the no-dissipation arm ``N0`` is reported alongside;
- regions of ``NRM.ring_region_masks`` with rms, max, volume and share of the global squared error (reporting only);
- reference budget (P05): the change of the exact bracket when the finite-difference step of the actual vorticity is halved,
  on the budget planes, against the finest ``N - R`` error on the same nodes (criterion: below 10 %).

    python reduce.py --out ROOT
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NPROC", "2")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                                      # noqa: E402

import fields as F                                                                      # noqa: E402
from references import available_sets, load_layout_metric, log                                          # noqa: E402

CONFIG = F.CONFIG
GRIDS = tuple(CONFIG["resolutions"])
GATE_ORDER = float(CONFIG["gate_minimum_order"])
CONST_TOL = float(CONFIG["constant_tolerance"])
BUDGET = float(CONFIG["reference_budget_fraction"])


def order(ea: float, eb: float, na: int, nb: int):
    if not (ea > 0 and eb > 0):
        return None
    return math.log(ea / eb) / math.log(nb / na)


def h_rms(H, e):
    return float(np.sqrt(np.sum(H * e * e) / np.sum(H)))


def reduce_grid(n: int, out_root: Path) -> dict:
    from drbx.native.fci_perpendicular_sbp_norms import h_weights, region_errors, ring_region_masks
    from drbx.stencils.nodal_plan import build_nodal_plan

    layout, metric, _meta = load_layout_metric(n, out_root)
    plan = build_nodal_plan(layout, metric)
    H_all = np.asarray(h_weights(plan))
    masks_all = ring_region_masks(layout, n)
    masks = masks_all
    res = {"n": n, "P": layout.P, "regions_available": sorted(masks_all)}
    for name in available_sets(out_root, n, static=True):
        zr, zs = np.load(out_root / f"N{n}" / "references" / f"{name}.npz"), np.load(out_root / f"N{n}" / "static" / f"{name}.npz")
        pairs = json.loads(str(zr["pairs"]))
        planes = zr["planes"] if "planes" in zr.files else np.arange(layout.n_eta)
        subset = len(planes) < layout.n_eta
        R, N, N0 = zr["R"], zs["N"][:, planes], zs["N0"][:, planes]
        H, masks = H_all[planes], {k: v[planes] for k, v in masks_all.items()}
        scale = max((float(np.abs(R[k]).max()) for k, p in enumerate(pairs) if not p["constant"]), default=1.0)
        entry = {"bracket_scale": scale, "c_kappa": float(zs["c_kappa"]), "pairs": {}, "planes": planes.tolist(),
                 "subset": bool(subset)}
        for k, p in enumerate(pairs):
            row = {"group": p["group"], "gate": p["gate"], "constant": p["constant"], "subset": bool(subset)}
            for arm, X in (("N", N[k]), ("N0", N0[k])):
                e = X - R[k]
                row[arm] = {"E": h_rms(H, e), "max": float(np.abs(e).max())}
                if arm == "N":
                    row["N"]["max_abs_N"] = float(np.abs(X).max())
                    row["N"]["regions"] = region_errors(e, R[k], H, masks)
            row["ref_rms"] = h_rms(H, R[k])
            entry["pairs"][p["label"]] = row
        if "budget_planes" in zr.files:
            bplanes = zr["budget_planes"]
            Rh = zr["budget_R_half"]
            pos = np.searchsorted(planes, bplanes)
            Hs = H_all[bplanes]
            budget = {"planes": bplanes.tolist(), "step": zr["budget_step"].tolist(), "pairs": {}}
            for k, p in enumerate(pairs):
                dR = h_rms(Hs, Rh[k] - R[k][pos])
                eN = h_rms(Hs, N[k][pos] - R[k][pos])
                budget["pairs"][p["label"]] = {"ref_change_rms": dR, "N_minus_R_rms": eN,
                                               "ratio": dR / eN if eN > 0 else None}
            entry["budget"] = budget
        res.setdefault("sets", {})[name] = entry
    return res


def summarize(per_grid: dict) -> dict:
    ns = sorted(per_grid)
    out = {"grids": ns, "gate_minimum_order": GATE_ORDER, "constant_tolerance": CONST_TOL, "sets": {}}
    for name in F.SET_NAMES:
        if not all(name in per_grid[n].get("sets", {}) for n in ns):
            continue
        rows = {}
        for label, row0 in per_grid[ns[0]]["sets"][name]["pairs"].items():
            row = {"group": row0["group"], "gate": row0["gate"], "constant": row0["constant"],
                   "subset": row0.get("subset", False)}
            for arm in ("N", "N0"):
                E = [per_grid[n]["sets"][name]["pairs"][label][arm]["E"] for n in ns]
                row[arm] = {"E": E, "orders": [order(E[i], E[i + 1], ns[i], ns[i + 1]) for i in range(len(ns) - 1)]}
            if row["constant"]:
                scale = per_grid[ns[-1]]["sets"][name]["bracket_scale"]
                worst = max(per_grid[n]["sets"][name]["pairs"][label]["N"]["max_abs_N"] / per_grid[n]["sets"][name]["bracket_scale"]
                            for n in ns)
                row["constant_control"] = {"max_abs_N_over_scale": worst, "pass": bool(worst <= CONST_TOL)}
            elif row["gate"]:
                ords = row["N"]["orders"]
                row["pass"] = bool(len(ords) >= 2 and all(o is not None and o >= GATE_ORDER for o in ords))
            rows[label] = row
        out["sets"][name] = {"pairs": rows,
                             "gate_pass": all(r["pass"] for r in rows.values() if "pass" in r),
                             "constant_pass": all(r["constant_control"]["pass"] for r in rows.values() if "constant_control" in r)}
        last = per_grid[ns[-1]]["sets"][name]
        if "budget" in last:
            ratios = [v["ratio"] for v in last["budget"]["pairs"].values() if v["ratio"] is not None]
            out["sets"][name]["budget"] = {"max_ratio": max(ratios) if ratios else None, "criterion": BUDGET,
                                           "pass": bool(max(ratios) < BUDGET) if ratios else None,
                                           "pairs": last["budget"]["pairs"]}
        # region highlights at the finest grid: largest share of the global N - R error per pair
        hl = {}
        for label, row in last["pairs"].items():
            if row["constant"]:
                continue
            regs = {k: v for k, v in row["N"]["regions"].items() if k != "all" and not k.startswith("uband")}
            top = max(regs, key=lambda k: regs[k]["share"])
            hl[label] = {"top_region": top, "share": regs[top]["share"], "rms": regs[top]["rms"],
                         "volume": regs[top]["volume"],
                         "core_share": regs.get("core", {}).get("share"), "wall_ring_share": regs.get("wall_ring", {}).get("share")}
        out["sets"][name]["region_highlights"] = hl
    if all("p05full" in per_grid[n].get("sets", {}) for n in ns[:1]) and "p05w" in out["sets"]:
        full = per_grid[ns[0]]["sets"]["p05full"]["pairs"]["mms_omega"]["N"]["E"]
        sub = per_grid[ns[0]]["sets"]["p05w"]["pairs"]["mms_omega"]["N"]["E"]
        out["sampling_check_N%d" % ns[0]] = {"E_all_planes": full, "E_8_plane_subset": sub, "relative_difference": abs(sub - full) / full}
    return out


def print_table(summary: dict) -> None:
    ns = summary["grids"]
    for name, s in summary["sets"].items():
        gated = any(r["gate"] and not r["constant"] for r in s["pairs"].values())
        log(f"== {name}: gate {('PASS' if s['gate_pass'] else 'FAIL') if gated else 'n/a (report only)'}, constant controls {'PASS' if s['constant_pass'] else 'FAIL'}")
        for label, r in s["pairs"].items():
            es = " ".join(f"{e:.3e}" for e in r["N"]["E"])
            os_ = " ".join("-" if o is None else f"{o:.2f}" for o in r["N"]["orders"])
            o0 = " ".join("-" if o is None else f"{o:.2f}" for o in r["N0"]["orders"])
            tag = ("const " + ("ok" if r["constant_control"]["pass"] else "FAIL") + f" {r['constant_control']['max_abs_N_over_scale']:.1e}"
                   if r["constant"] else ("PASS" if r.get("pass") else "FAIL" if r["gate"] else "report"))
            sub = " [8-plane subset]" if r.get("subset") else ""
            log(f"   {label[:62]:62s} E[{es}] ord[{os_}] ord0[{o0}] {tag}{sub}")
        if "budget" in s:
            log(f"   reference budget max ratio {s['budget']['max_ratio']} (criterion < {BUDGET}): {s['budget']['pass']}")


def run(out_root: Path) -> dict:
    per_grid = {}
    for n in GRIDS:
        if (out_root / f"N{n}" / "static").exists():
            per_grid[n] = reduce_grid(n, out_root)
            (out_root / f"N{n}" / "results.json").write_text(json.dumps(per_grid[n], indent=1))
            log(f"N{n} reduced")
    summary = summarize(per_grid)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))
    print_table(summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    run(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
