"""Reduce the static curvature gates: N - R per field, H-weighted L2 (absolute and relative), regions, orders, pass/fail.

Reads ``<out>/<arm>/N{n}/references/<case>.npz`` and ``static/<case>.npz`` for every available ``n`` and writes
``<out>/<arm>/summary.json`` plus the combined ``<out>/summary.json``. With ``e = N - R`` (nodal operator minus the exact nodal
RHS, ``R`` from ``curv_references``) and ``H = Hp * deta`` of the plan:

- ``E_n = sqrt(sum H e^2 / sum H)``, ``rel = E_n / rms_H(R)`` and the order ``log(E_a / E_b) / log(n_b / n_a)``;
- regions of ``fci_perpendicular_sbp_norms.ring_region_masks`` (``region_errors``: rms, relative, max, volume, share);
- **gate** (gated arm and gated cases, gated equations of the case): order >= 1.8 on both intervals; constant controls:
  ``max|N| <= tol * scale`` with ``scale`` the largest ``max|R|`` over the nonconstant cases of the same resolution;
- no ``O - R`` exists on the nodes (there is no owner projection): ``N - R`` is the whole derivative error.

    python curv_reduce.py [--arms raw,filtered]
"""
from __future__ import annotations

import curv_common as C

import argparse
import json
import math

import numpy as np


def order(ea, eb, na, nb):
    return math.log(ea / eb) / math.log(nb / na) if ea > 0 and eb > 0 else None


def hrms(H, e):
    return float(np.sqrt(np.sum(H * e * e) / np.sum(H)))


def reduce_grid(arm: str, n: int) -> dict:
    from drbx.native.fci_perpendicular_sbp_norms import h_weights, region_errors, ring_region_masks

    layout, _metric, _meta, plan = C.load_plan(arm, n)
    H = np.asarray(h_weights(plan))
    masks = ring_region_masks(layout, n)
    base = C.arm_dir(arm, n)
    res = {"n": n, "P": layout.P, "regions_available": sorted(masks), "cases": {}}
    for f in sorted((base / "static").glob("*.npz")):
        zr, zs = np.load(base / "references" / f.name), np.load(f)
        label = str(zr["label"])
        R, N, Nleg = zr["R"], zs["N"], zs["Nleg"]
        variants = [str(v) for v in zs["variants"]]
        entry = {"constant": bool(zr["constant"]), "gate_eqs": [str(x) for x in zr["gate_eqs"]],
                 "max_abs_R": [float(np.abs(R[k]).max()) for k in range(4)], "R_rms": [hrms(H, R[k]) for k in range(4)],
                 "Rdiff_legacy": float(zr["Rdiff_legacy"]), "legacy_minus_pi_centered_max": float(np.abs(Nleg - N[0]).max()),
                 "variants": {}}
        for vi, v in enumerate(variants):
            rows = {}
            for k, eq in enumerate(C.EQUATIONS):
                e = N[vi, k] - R[k]
                row = {"E": hrms(H, e), "rel": hrms(H, e) / entry["R_rms"][k] if entry["R_rms"][k] > 0 else None,
                       "max": float(np.abs(e).max()), "max_abs_N": float(np.abs(N[vi, k]).max())}
                if not entry["constant"]:
                    row["regions"] = region_errors(e, R[k], H, masks)
                rows[eq] = row
            entry["variants"][v] = rows
        res["cases"][label] = entry
    return res


def summarize(arm: str, per_grid: dict) -> dict:
    ns = sorted(per_grid)
    gated_arm = arm in C.CONFIG["gated_arms"]
    gate_min, tol = float(C.CONFIG["gate_minimum_order"]), float(C.CONFIG["constant_tolerance"])
    out = {"arm": arm, "gated_arm": gated_arm, "grids": ns, "gate_minimum_order": gate_min, "cases": {}}
    labels = sorted({l for n in ns for l in per_grid[n]["cases"]})
    scale = {n: max((max(c["max_abs_R"]) for c in per_grid[n]["cases"].values() if not c["constant"]), default=1.0) for n in ns}
    for label in labels:
        gl = [n for n in ns if label in per_grid[n]["cases"]]          # grids on which this case was run
        c0 = per_grid[gl[0]]["cases"][label]
        row = {"constant": c0["constant"], "gate_eqs": c0["gate_eqs"], "grids": gl, "variants": {},
               "legacy_minus_pi_centered_max": [per_grid[n]["cases"][label]["legacy_minus_pi_centered_max"] for n in gl],
               "Rdiff_legacy": [per_grid[n]["cases"][label]["Rdiff_legacy"] for n in gl]}
        for v in c0["variants"]:
            vrow = {}
            for eq in C.EQUATIONS:
                E = [per_grid[n]["cases"][label]["variants"][v][eq]["E"] for n in gl]
                rel = [per_grid[n]["cases"][label]["variants"][v][eq]["rel"] for n in gl]
                ords = [order(E[i], E[i + 1], gl[i], gl[i + 1]) for i in range(len(gl) - 1)]
                d = {"E": E, "rel": rel, "orders": ords}
                if c0["constant"]:
                    worst = max(per_grid[n]["cases"][label]["variants"][v][eq]["max_abs_N"] / scale[n] for n in gl)
                    d["constant_control"] = {"max_abs_N_over_scale": worst, "pass": bool(worst <= tol)}
                elif eq in c0["gate_eqs"]:
                    d["pass"] = bool(len(gl) == len(C.GRIDS) and all(o is not None and o >= gate_min for o in ords))
                    d["complete"] = len(gl) == len(C.GRIDS)
                vrow[eq] = d
            row["variants"][v] = vrow
        out["cases"][label] = row
    out["gate_pass"] = {v: all(d["pass"] for r in out["cases"].values() for d in r["variants"][v].values() if "pass" in d)
                        for v in C.VARIANTS}
    out["gate_complete"] = all(d["complete"] for r in out["cases"].values() for d in r["variants"]["centered"].values()
                               if "complete" in d)
    out["constant_pass"] = {v: all(d["constant_control"]["pass"] for r in out["cases"].values()
                                   for d in r["variants"][v].values() if "constant_control" in d) for v in C.VARIANTS}
    hl = {}
    for label in labels:
        c = per_grid[max(n for n in ns if label in per_grid[n]["cases"])]["cases"][label]
        if c["constant"]:
            continue
        for eq in C.EQUATIONS:
            regs = {k: r for k, r in c["variants"]["centered"][eq]["regions"].items() if k != "all" and not k.startswith("uband")}
            top = max(regs, key=lambda k: regs[k]["share"])
            hl[f"{label}/{eq}"] = {"top_region": top, "share": regs[top]["share"], "rms": regs[top]["rms"],
                                   "core_share": regs.get("core", {}).get("share"),
                                   "wall_ring_share": regs.get("wall_ring", {}).get("share")}
    out["region_highlights_finest_centered"] = hl
    return out


def print_table(summary: dict) -> None:
    ns = summary["grids"]
    gated = summary["gated_arm"]
    C.log(f"== arm {summary['arm']} ({'gated' if gated else 'reported'}), grids {ns}; gate pass per variant {summary['gate_pass']} "
          f"(complete: {summary['gate_complete']}); constant controls {summary['constant_pass']}")
    for label, r in summary["cases"].items():
        for v, vr in r["variants"].items():
            for eq, d in vr.items():
                es = " ".join(f"{e:.3e}" for e in d["E"])
                os_ = " ".join("-" if o is None else f"{o:.2f}" for o in d["orders"])
                tag = ("const " + ("ok" if d["constant_control"]["pass"] else "FAIL") if "constant_control" in d
                       else ("PASS" if d.get("pass") else "FAIL") if "pass" in d else "report")
                C.log(f"   {label[:40]:40s} {v:14s} {eq:5s} E[{es}] ord[{os_}] {tag}")


def run(arms) -> dict:
    allsum = {}
    for arm in arms:
        per = {}
        for n in C.GRIDS:
            if (C.arm_dir(arm, n) / "static").exists() and any((C.arm_dir(arm, n) / "static").glob("*.npz")):
                per[n] = reduce_grid(arm, n)
                (C.arm_dir(arm, n) / "results.json").write_text(json.dumps(per[n], indent=1))
                C.log(f"{arm} N{n} reduced")
        if per:
            allsum[arm] = summarize(arm, per)
            (C.OUT / arm / "summary.json").write_text(json.dumps(allsum[arm], indent=1))
            print_table(allsum[arm])
    (C.OUT / "summary.json").write_text(json.dumps(allsum, indent=1))
    return allsum


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(C.ARM_ROOT))
    run(ap.parse_args(argv).arms.split(","))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
