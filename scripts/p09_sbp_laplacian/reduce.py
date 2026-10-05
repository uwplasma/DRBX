#!/usr/bin/env python3
"""Reduce the static tables of one arm: orders 32 -> 48 and 48 -> 64, the gate and the constant control.

Reads ``<out>/<arm>/N<n>/static.json`` for every finished ``n`` and writes ``<out>/<arm>/static_summary.json``.

* order of a field: ``log(E_a / E_b) / log(n_b / n_a)`` with ``E`` the H-weighted global relative L2 error ``|N - R| / |R|``
  (and, per region, the region rms of ``N - R``);
* gate (gated arms only, ``configuration.json``): every nonconstant field of the gating sets (``transverse``: ``n, Te, Ti,
  omega, psi``; ``p07n``: the four nonconstant fields, Dirichlet and both Neumann data types) has order >= 1.8 on both
  intervals; the constant control has ``max |N| <= 1e-8``. Ungated arms (raw) and the ``transverse_wave`` control are reported.

    python reduce.py --arm raw|filtered [--out ROOT] [--variant V] [--faces-root ROOT]

``--variant`` reduces ``<faces-root>/<arm>/N<n>/<variant>/static.json`` into ``<faces-root>/<arm>/static_summary_<variant>.json``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import CONFIG, log                                                          # noqa: E402

import json                                                                             # noqa: E402

GATE_ORDER = float(CONFIG["gate_minimum_order"])
CONST_TOL = float(CONFIG["constant_tolerance"])
GATED_FIELDS = {"transverse": ("n", "Te", "Ti", "omega", "psi"), "p07n": ("field_b1", "field_e3", "field_e12", "heldout_field_b2")}
GATED_MODES = {"transverse": ("dirichlet",), "p07n": ("dirichlet", "neumann_conormal", "neumann_normal")}
REGIONS = ("core", "core_band", "interior", "adjacent_band", "wall_ring", "wall")


def order(ea, eb, na, nb):
    if ea is None or eb is None or not (ea > 0 and eb > 0):
        return None
    return math.log(ea / eb) / math.log(nb / na)


def reduce_arm(out_root: Path, arm: str, variant: str | None = None) -> dict:
    per = {}
    for n in CONFIG["resolutions"]:
        p = (C.arm_dir(out_root, arm, n) if variant is None else C.variant_dir(out_root, arm, n, variant)) / "static.json"
        if p.exists():
            per[n] = json.loads(p.read_text())
    ns = sorted(per)
    gated_arm = arm in CONFIG["gated_arms"]
    out = dict(arm=arm, variant=variant, grids=ns, gated_arm=gated_arm, gate_minimum_order=GATE_ORDER, constant_tolerance=CONST_TOL, sets={})
    for sname, s0 in per[ns[0]]["sets"].items():
        sres = {}
        for mode, m0 in s0.items():
            fres = {}
            for fname in m0["fields"]:
                rows = [per[n]["sets"][sname][mode]["fields"][fname] for n in ns]
                const = fname == "constant"
                entry = dict(constant=const)
                if const:
                    entry["max_abs_N"] = [r["max_abs_N"] for r in rows]
                    entry["pass"] = bool(max(entry["max_abs_N"]) <= CONST_TOL)
                else:
                    E = [r["rel_global"] for r in rows]
                    entry["E_rel_global"] = E
                    entry["orders"] = [order(E[i], E[i + 1], ns[i], ns[i + 1]) for i in range(len(ns) - 1)]
                    entry["rms"] = [r["rms"] for r in rows]
                    entry["orders_rms"] = [order(entry["rms"][i], entry["rms"][i + 1], ns[i], ns[i + 1]) for i in range(len(ns) - 1)]
                    reg = {}
                    for rn in REGIONS:
                        if all(rn in r["regions"] for r in rows):
                            rr = [r["regions"][rn]["rms"] for r in rows]
                            reg[rn] = dict(rms=rr, orders=[order(rr[i], rr[i + 1], ns[i], ns[i + 1]) for i in range(len(ns) - 1)],
                                           share_at_finest=rows[-1]["regions"][rn]["share"])
                    entry["regions"] = reg
                    gated = fname in GATED_FIELDS.get(sname, ()) and mode in GATED_MODES.get(sname, ())
                    entry["gated"] = bool(gated and gated_arm)
                    if gated:
                        entry["pass"] = bool(len(entry["orders"]) >= 2 and all(o is not None and o >= GATE_ORDER for o in entry["orders"]))
                fres[fname] = entry
            sres[mode] = fres
        out["sets"][sname] = sres
    gate_entries = [e for s in out["sets"].values() for m in s.values() for e in m.values() if e.get("gated")]
    const_entries = [e for sn, s in out["sets"].items() if sn != "transverse_wave" for m in s.values() for e in m.values() if e["constant"]]
    out["gate_pass"] = bool(all(e["pass"] for e in gate_entries)) if gate_entries else None
    out["constant_pass"] = bool(all(e["pass"] for e in const_entries)) if const_entries else None
    return out


def print_table(out: dict) -> None:
    ns = out["grids"]
    log(f"== arm {out['arm']} (gated: {out['gated_arm']}) grids {ns}: gate {out['gate_pass']} constant {out['constant_pass']}")
    for sname, s in out["sets"].items():
        for mode, f in s.items():
            log(f"-- {sname}/{mode}")
            for fname, e in f.items():
                if e["constant"]:
                    log(f"   {fname:18s} max|N| {['%.2e' % x for x in e['max_abs_N']]} {'ok' if e['pass'] else 'FAIL'}")
                else:
                    o = " ".join("-" if x is None else f"{x:.2f}" for x in e["orders"])
                    tag = ("PASS" if e.get("pass") else "FAIL") if "pass" in e else ""
                    log(f"   {fname:18s} E {['%.3e' % x for x in e['E_rel_global']]} orders [{o}] {tag}{' (gated)' if e['gated'] else ''}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    ap.add_argument("--variant", choices=tuple(C.VARIANTS), default=None)
    ap.add_argument("--faces-root", type=Path, default=C.FACES_ROOT)
    args = ap.parse_args(argv)
    root = args.out if args.variant is None else args.faces_root
    out = reduce_arm(root, args.arm, args.variant)
    C.write_json(Path(root) / args.arm / ("static_summary.json" if args.variant is None else f"static_summary_{args.variant}.json"), out)
    print_table(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
