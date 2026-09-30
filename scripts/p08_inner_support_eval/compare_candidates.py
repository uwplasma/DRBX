"""Per-ring (absolute radius) comparison of candidate runs against C0: P07 N-O and transverse reconstruction.

    python -m p08_inner_support_eval.compare_candidates C1=DIR1 C2=DIR2 ...
        (each DIR holds the ``summary.json`` written by ``analyze.py``, with modes C0 and C1 = the candidate)

``compare`` is also called by ``campaign.py analyze``, which keeps the JSON and the text.  The u-bands come from
``configuration.json``; ``NS`` are the grids every summary must hold.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NS = ("32", "48", "64")
TRIV = {"common", "constant", "fA"}
METRICS = ("b_p07.N-O.rms", "a_cell.transverse.rms")


def default_bands():
    return [tuple(b) for b in json.loads((HERE / "configuration.json").read_text())["bands"]]


def load(d):
    """``d``: a summary dict, a summary.json path, or a folder holding summary.json -> (summary, kept field indices)."""
    if isinstance(d, dict):
        s = d
    else:
        p = Path(d)
        s = json.loads((p / "summary.json" if p.is_dir() else p).read_text())
    missing = [n for n in NS if int(n) not in s["grids"]]
    if missing:
        raise ValueError(f"summary lacks grids {missing} (has {s['grids']})")
    keep = [i for i, f in enumerate(s["fields"]) if f not in TRIV]
    return s, keep


def arr(s, keep, mode, group, met):
    try:
        return np.array([s["values"][mode][group][met][n] for n in NS], float)[:, keep]
    except KeyError:
        return None


def orders(a):
    return np.log(a[:-1] / a[1:]) / np.log(np.array([48 / 32, 64 / 48]))[:, None]


def band(s, kp, mode, met, lo, hi):
    """per grid: RMS over the sampled rings whose centre lies in [lo, hi) (each ring value is an owner RMS)."""
    out = []
    for n in NS:
        N = int(n)
        vals = []
        for ring in range(N):
            if not lo <= (ring + .5) / N < hi:
                continue
            try:
                vals.append(np.array(s["values"][mode][f"ring:{ring}"][met][n], float)[kp])
            except KeyError:
                pass
        out.append(np.sqrt(np.mean(np.square(vals), axis=0)) if vals else np.full(len(kp), np.nan))
    return np.array(out)


def _num(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _nums(v):
    return [_num(x) for x in np.ravel(v)]


def rebound_flags(s, mode):
    out = set()
    for e in s["flags"]["rebound"]:
        if e["mode"] != mode:
            continue
        for x in e["fields"]:
            f = x if isinstance(x, str) else x["field"]
            if f not in TRIV:
                out.add((e["group"], e["metric"], e["interval"], f))
    return out


def compare(summaries, bands=None):
    """``summaries``: ``{name: summary dict | path}`` -> ``(text, data)``; ``data`` is JSON-serialisable."""
    bands = [tuple(b) for b in bands] if bands is not None else default_bands()
    S = {k: load(v) for k, v in summaries.items()}
    s0, keep = next(iter(S.values()))
    lines, data = [], dict(grids=[int(n) for n in NS], candidates=list(S), last=s0["last"], metrics={}, rebound={})
    for met in METRICS:
        lines.append(f"\n{met}: fixed u-bands. C0: min order 32-48/48-64 | cand: N64 geo-mean ratio to C0 (max), "
                     f"min order 32-48/48-64, #fields with N48->N64 rebound")
        lines.append("band           C0 min ord   " + "   ".join(f"{k:>34s}" for k in S))
        md = dict(bands=[], pooled_inner={})
        for lo, hi in bands:
            a0 = band(s0, keep, "C0", met, lo, hi)
            if not np.all(np.isfinite(a0)):
                lines.append(f"[{lo:.2f},{hi:.2f})  (not sampled on every grid)")
                md["bands"].append(dict(lo=lo, hi=hi, sampled=False))
                continue
            o0 = orders(a0).min(axis=1)
            cells, bd = [], dict(lo=lo, hi=hi, sampled=True, c0_min_order=_nums(o0), candidates={})
            for k, (s, kp) in S.items():
                a = band(s, kp, "C1", met, lo, hi)
                r = a[2] / a0[2]
                o = orders(a).min(axis=1)
                reb = int(np.sum(a[2] > a[1]))
                gm = float(np.exp(np.mean(np.log(r))))
                cells.append(f"{gm:5.2f} ({r.max():5.2f}) {o[0]:5.2f}/{o[1]:5.2f} reb {reb:2d}")
                bd["candidates"][k] = dict(geo_mean_ratio_n64=_num(gm), max_ratio_n64=_num(r.max()),
                                           min_order=_nums(o), n_rebound_fields=reb)
            lines.append(f"[{lo:.2f},{hi:.2f})  {o0[0]:5.2f}/{o0[1]:5.2f}   " + "   ".join(f"{c:>34s}" for c in cells))
            md["bands"].append(bd)
        lines.append("pooled inner: C1/C0 geo-mean per grid and min order (both intervals)")
        for k, (s, kp) in S.items():
            a0 = arr(s, kp, "C0", "pool:inner", met)
            a = arr(s, kp, "C1", "pool:inner", met)
            if a0 is None or a is None:
                lines.append(f"  {k:5s} (no pool:inner values)")
                continue
            ratio = np.exp(np.mean(np.log(a / a0), axis=1))
            lines.append(f"  {k:5s} ratio {ratio.round(3)}  min order C0 {orders(a0).min(axis=1).round(2)} "
                         f"cand {orders(a).min(axis=1).round(2)}")
            md["pooled_inner"][k] = dict(ratio_per_grid=_nums(ratio), min_order_c0=_nums(orders(a0).min(axis=1)),
                                         min_order_candidate=_nums(orders(a).min(axis=1)))
        data["metrics"][met] = md
    for k, (s, kp) in S.items():
        c0, c1 = rebound_flags(s, "C0"), rebound_flags(s, "C1")
        lines.append(f"\n{k}: role/track rebound flags C0 {len(c0)} cand {len(c1)} (new {len(c1 - c0)}, removed {len(c0 - c1)})")
        data["rebound"][k] = dict(c0=len(c0), candidate=len(c1), new=len(c1 - c0), removed=len(c0 - c1))
    return "\n".join(lines).lstrip("\n") + "\n", data


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    text, _ = compare(dict(x.split("=", 1) for x in argv))
    print(text, end="")


if __name__ == "__main__":
    main()
