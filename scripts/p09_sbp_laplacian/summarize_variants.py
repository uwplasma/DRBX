#!/usr/bin/env python3
"""Tables of the face-coefficient variants (interpolated, eta, eta+theta, all evaluated) of one arm against the M5b numbers.

Reads, per resolution ``N``, the M5b files of ``<out>/<arm>/N<N>/`` (the interpolated baseline: ``audit.json``, ``static.json``,
``solve.json``) and the variant files of ``<faces-root>/<arm>/N<N>/<variant>/`` (``audit.json``, ``static.json``, ``solve.json``)
and prints the face-coefficient minima, the matrix-free Dirichlet / Neumann lowest eigenvalues, the static errors and orders and the
CG iterations; the result is written to ``<faces-root>/<arm>/summary.json``.

    python summarize_variants.py --arm raw [--out ROOT] [--faces-root ROOT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import CONFIG                                                               # noqa: E402

LABELS = {"interp": "interpolated (M5b)", "ee": "eta evaluated", "ee_tt": "eta+theta evaluated", "all": "all evaluated"}
STATIC_PICK = {"transverse": ("dirichlet", ("n", "Te", "Ti", "omega", "psi")),
               "p07n_dirichlet": ("dirichlet", ("field_b1", "field_e3", "field_e12", "heldout_field_b2")),
               "p07n_neumann_conormal": ("neumann_conormal", ("field_b1", "field_e3", "field_e12", "heldout_field_b2")),
               "p07n_neumann_normal": ("neumann_normal", ("field_b1", "field_e3", "field_e12", "heldout_field_b2"))}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _dir(out_root, faces_root, arm, n, variant):
    return C.arm_dir(out_root, arm, n) if variant == "interp" else C.variant_dir(faces_root, arm, n, variant)


def _order(ea, eb, na, nb):
    return math.log(ea / eb) / math.log(nb / na) if ea and eb and ea > 0 and eb > 0 else None


def summarize(out_root: Path, faces_root: Path, arm: str) -> dict:
    res = {"arm": arm, "grids": list(CONFIG["resolutions"]), "variants": {}}
    for v in LABELS:
        entry = {"label": LABELS[v], "coefficients": {}, "dirichlet": {}, "neumann": {}, "solve": {}, "static": {}}
        for n in CONFIG["resolutions"]:
            d = _dir(out_root, faces_root, arm, n, v)
            a, s, so = _load(d / "audit.json"), _load(d / "static.json"), _load(d / "solve.json")
            if a:
                c = a["coefficients"]
                entry["coefficients"][n] = {k: dict(min=c[k]["min"], max=c[k]["max"], negative=c[k]["negative"]["count"],
                                                    fraction_negative=c[k]["negative"]["fraction"])
                                            for k in ("auu_f", "att_h", "aee_h")}
                mf = a["audit"].get("matrix_free")
                if mf:
                    for kind in ("dirichlet", "neumann"):
                        r = mf[kind]
                        entry[kind][n] = dict(lowest=r["lowest"], rho=r["rho"], min_eig_rel=r["min_eig_rel"], n_negative=r["n_negative"],
                                              iterations=r["iterations"], max_residual_rel=max(r["residuals"]),
                                              null_defect=r.get("null_defect"), h_symmetry=r.get("h_symmetry_probe"))
            if s:
                rows = {}
                for key, (mode, names) in STATIC_PICK.items():
                    sname = "p07n" if key.startswith("p07n") else "transverse"
                    rows[key] = {f: s["sets"][sname][mode]["fields"][f]["rel_global"] for f in names}
                rows["constant_max_abs_N"] = {m: s["sets"]["p07n"][m]["fields"]["constant"]["max_abs_N"]
                                              for m in ("dirichlet", "neumann_conormal", "neumann_normal")}
                entry["static"][n] = rows
            if so:
                entry["solve"][n] = dict(iterations=so["cg"]["iterations"], factor_dtype=so["factor_dtype"],
                                         relative_residual=so["cg"].get("relative_residual"), warm_median_seconds=so["cg"]["warm_median_seconds"],
                                         controls={k: dict(iterations=c["iterations"], solution_error_rel=c["solution_error_rel"],
                                                           true_residual=c["true_residual_L2_rel"])
                                                   for k, c in so["controls"].items()})
        res["variants"][v] = entry
    return res


def _fmt(x, f="%.3e"):
    return "-" if x is None else f % x


def print_tables(res: dict) -> None:
    ns = res["grids"]
    vs = res["variants"]
    print(f"== arm {res['arm']}")
    print("\n-- face coefficients (min over the plan; negative entries)")
    for n in ns:
        for v, e in vs.items():
            c = e["coefficients"].get(n) or e["coefficients"].get(str(n))
            if c:
                print(f"N{n} {v:6s} " + "  ".join(f"{k}: min {_fmt(c[k]['min'])} neg {c[k]['negative']}" for k in c))
    for kind in ("dirichlet", "neumann"):
        print(f"\n-- {kind}: lowest eigenvalues of the H-normalised matrix (matrix-free LOBPCG)")
        for n in ns:
            for v, e in vs.items():
                r = e[kind].get(n)
                if r:
                    print(f"N{n} {v:6s} lowest {[float('%.5g' % x) for x in r['lowest'][:4]]} /rho {_fmt(r['min_eig_rel'], '%.2e')} "
                          f"neg {r['n_negative']} it {r['iterations']} resid/rho {_fmt(r['max_residual_rel'], '%.1e')}")
    print("\n-- static E_rel_global (orders 32-48, 48-64)")
    for key, (mode, names) in STATIC_PICK.items():
        print(f"[{key}]")
        for f in names:
            for v, e in vs.items():
                st = [e["static"].get(n) for n in ns]
                if all(st):
                    E = [x[key][f] for x in st]
                    o = [_order(E[i], E[i + 1], ns[i], ns[i + 1]) for i in range(len(ns) - 1)]
                    print(f"  {f:18s} {v:6s} " + " ".join(_fmt(x) for x in E) + "  orders " + " ".join(_fmt(x, "%.2f") for x in o))
    print("\n-- CG (rtol 1e-10)")
    for n in ns:
        for v, e in vs.items():
            r = e["solve"].get(n)
            if r:
                ctl = " ".join(f"{k.split('/')[1][:4]}:{c['iterations']}it/{c['solution_error_rel']:.2e}" for k, c in r["controls"].items()
                               if k.startswith("transverse/"))
                print(f"N{n} {v:6s} it {r['iterations']} ({r['factor_dtype']}) warm {r['warm_median_seconds']:.3f}s  controls {ctl}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    ap.add_argument("--faces-root", type=Path, default=C.FACES_ROOT)
    args = ap.parse_args(argv)
    res = summarize(args.out, args.faces_root, args.arm)
    C.write_json(Path(args.faces_root) / args.arm / "summary.json", res)
    print_tables(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
