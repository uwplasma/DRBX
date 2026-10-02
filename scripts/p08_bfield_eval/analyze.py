"""Compare the ``spline`` and ``compact_c3`` runs of :mod:`p08_bfield_eval.run`.

    python -m p08_bfield_eval.analyze DIR           # reads DIR/N{n}_{mode}/ for every grid with both modes present
                                                    # writes DIR/report.md and DIR/summary.json

Per variant / field / term / comparison / region (and knot class): the spline and C3 errors (relative L2, absolute L2, max), the
C3/spline ratio per grid and the observed orders 32->48 and 48->64 of each mode; the coefficient table (relative change of
each closure geometry array between the evaluators, RMS and max, per region and knot class, on identical points); and the
flags (C3 worse than spline by more than ``--ratio-flag`` (default 1.5) in relative L2, or an order lower by more than
``--order-flag`` (0.5)).  When the two modes of a grid were built on different owner sets (a dropped owner) the metrics are
recomputed from ``arrays.npz`` on the common owners.  No jax import.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p08_bfield_eval import metrics as M                                      # noqa: E402
from p08_bfield_eval.sampling import KNOTS, REGIONS                           # noqa: E402

MODES = ("spline", "compact_c3")
SCHEMA = "drbx.p08-bfield-eval-summary.v1"
ERROR_METRICS = ("rel_l2", "l2", "max_abs")
RATIO_FLAG, ORDER_FLAG = 1.5, 0.5
#: trailing component dimensions of each coefficient array (the remaining axes index points)
COEFFICIENT_COMPONENTS = {
    "p05_raw_h": 1, "p05_raw_jacobian": 0, "p06_raw_J": 0, "p06_raw_B": 0, "p06_raw_K": 1, "p07_raw_tensor": 2,
    "p07_raw_divergence": 1, "p05_face_h": 1, "p05_face_jacobian": 0, "p06_face_J": 0, "p06_face_B": 0, "p06_face_K": 1,
    "p07_face_tensor": 2}
RAW_NAMES = tuple(k for k in COEFFICIENT_COMPONENTS if "_raw_" in k)
FACE_NAMES = tuple(k for k in COEFFICIENT_COMPONENTS if "_face_" in k)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def discover(directory) -> dict:
    """``{n: {mode: folder}}`` of the complete ``N{n}_{mode}`` folders of ``directory``."""
    found: dict = {}
    for folder in sorted(Path(directory).glob("N*_*")):
        match = re.fullmatch(r"N(\d+)_(spline|compact_c3)", folder.name)
        if match and (folder / "results.json").is_file() and (folder / "arrays.npz").is_file():
            found.setdefault(int(match.group(1)), {})[match.group(2)] = folder
    return dict(sorted(found.items()))


def load_run(folder) -> tuple:
    folder = Path(folder)
    results = json.loads((folder / "results.json").read_text())
    with np.load(folder / "arrays.npz", allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    return results, arrays


# ---------------------------------------------------------------------------
# error series
# ---------------------------------------------------------------------------
def grid_metrics(results: dict, arrays: dict, common: np.ndarray) -> dict:
    """The metrics of one run on the owners ``common`` (the stored ones when that is every owner of the run)."""
    keep = np.isin(arrays["owners"], common)
    if keep.all():
        return results["metrics"]
    return M.compute_metrics(M.restrict(arrays, keep), results["variants"], tuple(results["fields"]))


def _get(metrics: dict, subset: str, variant: str, field: str, term: str, comparison: str):
    try:
        return metrics[subset]["variants"][variant][field][term][comparison]
    except KeyError:
        return None


def _series(per_grid: dict, subset, variant, field, term, comparison, mode) -> dict:
    """``{metric: {n: value}}`` of one error over the grids."""
    out = {m: {} for m in ERROR_METRICS}
    for n, by_mode in per_grid.items():
        entry = _get(by_mode[mode], subset, variant, field, term, comparison)
        for m in ERROR_METRICS:
            out[m][n] = None if entry is None else entry[m]
    return out


def orders_of(series: dict) -> dict:
    """``{metric: [order between consecutive grids]}`` of ``{n: value}`` series."""
    out = {}
    for metric in ERROR_METRICS:
        ns = sorted(series[metric])
        out[metric] = [M.observed_order(series[metric][a], series[metric][b], a, b) for a, b in zip(ns, ns[1:])]
    return out


def _ratio(a, b):
    return None if a is None or b is None or b == 0.0 else a / b


def build_error_summary(per_grid: dict, results0: dict, subsets: list, ratio_flag: float, order_flag: float) -> tuple:
    """``(series, flags)``: ``series[variant][field]["term/comparison"][subset]`` and the flag list."""
    grids = sorted(per_grid)
    series: dict = {}
    flags: list = []
    for variant in results0["variants"]:
        for field in results0["fields"]:
            for term, comparison in M.COMPARISONS:
                for subset in subsets:
                    modes = {mode: _series(per_grid, subset, variant, field, term, comparison, mode) for mode in MODES}
                    if all(v is None for mode in MODES for v in modes[mode]["rel_l2"].values()):
                        continue
                    entry = {
                        "n_owners": {n: per_grid[n]["spline"].get(subset, {}).get("n_owners") for n in grids},
                        "spline": modes["spline"], "compact_c3": modes["compact_c3"],
                        "ratio": {m: {n: _ratio(modes["compact_c3"][m][n], modes["spline"][m][n]) for n in grids}
                                  for m in ERROR_METRICS},
                        "order": {mode: orders_of(modes[mode]) for mode in MODES},
                    }
                    series.setdefault(variant, {}).setdefault(field, {}).setdefault(f"{term}/{comparison}", {})[subset] = entry
                    key = dict(variant=variant, field=field, term=term, comparison=comparison, subset=subset)
                    for n in grids:
                        r = entry["ratio"]["rel_l2"][n]
                        if r is not None and r > ratio_flag:
                            flags.append({**key, "kind": "ratio", "n": n, "ratio": r, "spline": modes["spline"]["rel_l2"][n],
                                          "compact_c3": modes["compact_c3"]["rel_l2"][n]})
                    for i, (a, b) in enumerate(zip(grids, grids[1:])):
                        o_s, o_c = entry["order"]["spline"]["rel_l2"][i], entry["order"]["compact_c3"]["rel_l2"][i]
                        if o_s is not None and o_c is not None and o_c < o_s - order_flag:
                            flags.append({**key, "kind": "order_drop", "grids": [a, b], "order_spline": o_s,
                                          "order_compact_c3": o_c})
    return series, flags


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------
def relative_change(a: np.ndarray, b: np.ndarray, components: int) -> dict:
    """Relative change of ``b`` (C3) against ``a`` (spline) over the points of an array whose last ``components`` axes are
    the vector / tensor components: the pointwise relative norm ``|b - a| / |a|`` (RMS and max over the points) and the
    global ``rms(b - a) / rms(a)``."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    axes = tuple(range(a.ndim - components, a.ndim))
    na = np.sqrt(np.sum(a ** 2, axis=axes))
    nd = np.sqrt(np.sum((b - a) ** 2, axis=axes))
    if na.size == 0:
        return {"n_points": 0, "rms_rel": None, "max_rel": None, "global_rel": None}
    rel = nd / np.maximum(na, 1.0e-300)
    denom = float(np.sqrt(np.mean(na ** 2)))
    return {"n_points": int(na.size), "rms_rel": float(np.sqrt(np.mean(rel ** 2))), "max_rel": float(np.max(rel)),
            "global_rel": float(np.sqrt(np.mean(nd ** 2)) / denom) if denom > 0.0 else None}


def _align(ids_a: np.ndarray, ids_b: np.ndarray):
    """``(common ids, indices into a, indices into b)`` of two sorted unique id arrays."""
    common = np.intersect1d(ids_a, ids_b)
    return common, np.searchsorted(ids_a, common), np.searchsorted(ids_b, common)


def coefficient_changes(arrays_a: dict, arrays_b: dict, common_owners: np.ndarray) -> dict:
    """``{array: {subset: relative_change}}`` of C3 (``b``) against spline (``a``) on the cells / faces they share.  Raises when
    a shared point differs (the points do not depend on the field).  A raw cell belongs to its owner, a face to its lower owner
    when that is a common owner and else to its upper one."""
    owners = np.sort(np.asarray(common_owners))
    region_of = dict(zip(arrays_a["owners"].tolist(), arrays_a["region"].tolist()))
    knot_of = dict(zip(arrays_a["owners"].tolist(), arrays_a["knot"].tolist()))
    out: dict = {}
    for kind in ("raw", "face"):
        id_key = "raw_ids" if kind == "raw" else "face_rows"
        ids, ia, ib = _align(arrays_a[id_key], arrays_b[id_key])
        if kind == "raw":
            owner_of = arrays_a["raw_owner"][ia]
            keep = np.isin(owner_of, owners)
            names, point_keys = RAW_NAMES, ("raw_points",)
        else:
            lo, hi = arrays_a["face_owner_lo"][ia], arrays_a["face_owner_hi"][ia]
            lo_in, hi_in = np.isin(lo, owners), np.isin(hi, owners)
            keep = lo_in | hi_in
            owner_of = np.where(lo_in, lo, hi)
            names, point_keys = FACE_NAMES, ("face_points", "p07_face_points")
        for key in point_keys:
            gk = f"geom__{key}"
            if gk in arrays_a and gk in arrays_b and not np.array_equal(arrays_a[gk][ia], arrays_b[gk][ib]):
                raise ValueError(f"the {key} of the two modes differ: the coefficient comparison needs identical points")
        picked = owner_of[keep]
        region = np.array([region_of[int(o)] for o in picked], dtype=np.int64)
        knot = np.array([knot_of[int(o)] for o in picked], dtype=np.int64)
        masks = M.subset_masks(region, knot)
        for name in names:
            ga, gb = arrays_a.get(f"geom__{name}"), arrays_b.get(f"geom__{name}")
            if ga is None or gb is None:
                continue
            a, b = ga[ia][keep], gb[ib][keep]
            out[name] = {subset: relative_change(a[mask], b[mask], COEFFICIENT_COMPONENTS[name])
                         for subset, mask in masks.items() if mask.any()}
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def build_summary(directory, *, ratio_flag: float = RATIO_FLAG, order_flag: float = ORDER_FLAG) -> dict:
    found = discover(directory)
    grids = [n for n, modes in found.items() if set(modes) == set(MODES)]
    if not grids:
        raise ValueError(f"no grid with both modes under {directory} (found {({n: sorted(m) for n, m in found.items()})})")
    per_grid, runs, owner_sets, coefficients, receipts = {}, {}, {}, {}, {}
    for n in grids:
        loaded = {mode: load_run(found[n][mode]) for mode in MODES}
        res_s, res_c = loaded["spline"][0], loaded["compact_c3"][0]
        if res_s["sample"]["drawn_owners"] != res_c["sample"]["drawn_owners"]:
            raise ValueError(f"N{n}: the two modes drew different owner samples")
        common = np.intersect1d(loaded["spline"][1]["owners"], loaded["compact_c3"][1]["owners"])
        owner_sets[n] = {"drawn": len(res_s["sample"]["drawn_owners"]), "common": int(len(common)),
                         "dropped": {mode: loaded[mode][0]["dropped_owners"]["owners"] for mode in MODES},
                         "equal": bool(all(len(loaded[mode][1]["owners"]) == len(common) for mode in MODES))}
        per_grid[n] = {mode: grid_metrics(loaded[mode][0], loaded[mode][1], common) for mode in MODES}
        runs[n] = {mode: loaded[mode][0] for mode in MODES}
        receipts[n] = {mode: {**loaded[mode][0]["receipt"], "sanity": loaded[mode][0]["sanity"],
                              "diffusion_convention_check": loaded[mode][0]["diffusion_convention_check"]} for mode in MODES}
        coefficients[n] = coefficient_changes(loaded["spline"][1], loaded["compact_c3"][1], common)
        del loaded
    results0 = runs[grids[0]]["spline"]
    all_subsets = list(M.subset_masks(np.zeros(0, dtype=int), np.zeros(0, dtype=int)))
    subsets = [s for s in all_subsets if any(s in per_grid[n][mode] for n in grids for mode in MODES)]
    series, flags = build_error_summary(per_grid, results0, subsets, ratio_flag, order_flag)
    return {"schema": SCHEMA, "grids": grids, "modes": list(MODES), "variants": results0["variants"],
            "fields": results0["fields"], "comparisons": [list(c) for c in M.COMPARISONS], "subsets": subsets,
            "ratio_flag": ratio_flag, "order_flag": order_flag, "owner_sets": owner_sets,
            "sample_cells": {n: runs[n]["spline"]["sample"]["cells"] for n in grids},
            "series": series, "coefficients": coefficients, "flags": flags, "receipts": receipts}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(x, spec="{:.2e}") -> str:
    return "-" if x is None else spec.format(x)


def _per_grid(values: dict, grids, spec="{:.2e}") -> str:
    return " / ".join(_fmt(values.get(n), spec) for n in grids)


def _orders(orders: list) -> str:
    return " / ".join(_fmt(o, "{:.2f}") for o in orders) if orders else "-"


def _table(summary: dict, variant: str, key: str, subsets: list, metric: str = "rel_l2") -> list:
    grids = summary["grids"]
    ns = " / ".join(f"N{n}" for n in grids)
    lines = [f"| field | subset | owners | spline {metric} ({ns}) | C3 {metric} ({ns}) | C3/spline ({ns}) | order spline | order C3 |",
             "|---|---|---|---|---|---|---|---|"]
    for field in summary["fields"]:
        by_subset = summary["series"].get(variant, {}).get(field, {}).get(key, {})
        for subset in subsets:
            e = by_subset.get(subset)
            if e is None:
                continue
            lines.append(f"| {field} | {subset} | {_per_grid(e['n_owners'], grids, '{}')} | {_per_grid(e['spline'][metric], grids)} | "
                         f"{_per_grid(e['compact_c3'][metric], grids)} | {_per_grid(e['ratio'][metric], grids, '{:.2f}')} | "
                         f"{_orders(e['order']['spline'][metric])} | {_orders(e['order']['compact_c3'][metric])} |")
    return lines


def render_report(summary: dict) -> str:
    grids = summary["grids"]
    lines = ["# P08 B-evaluator comparison: spline vs compact C3", "",
             f"Grids present: {', '.join('N%d' % n for n in grids)}.  Errors are owner-volume-weighted; `rel_l2` is normalized like "
             "`step3_gates.term_metrics`.  Each evaluator is internally consistent (operator and references share it): compare "
             "error magnitudes and orders, not the operators with each other.  Orders are `log(e_coarse/e_fine)/log(n_fine/n_coarse)` "
             "of consecutive grids (indicative: bounded sample).  For the diffusion term `N-O` is against the exact face flux O, "
             "`O-R` compares O with the midpoint reference R and `N-R` is against R; the other terms are against "
             "`reference_rhs`.", "", "## Runs", "",
             "| grid | mode | owners | dropped | wall s | peak RSS GiB | finite | zero refs |", "|---|---|---|---|---|---|---|---|"]
    for n in grids:
        for mode in MODES:
            r = summary["receipts"][n][mode]
            lines.append(f"| N{n} | {mode} | {summary['owner_sets'][n]['common']} of {summary['owner_sets'][n]['drawn']} | "
                         f"{len(summary['owner_sets'][n]['dropped'][mode])} | {r['timings_seconds']['total']:.0f} | "
                         f"{r['peak_rss_gib']:.2f} | {r['sanity']['finite']} | {len(r['sanity']['zero_references'])} |")
    lines += ["", "### Sample cells (region|knot: kept/available)", ""]
    for n in grids:
        cells = summary["sample_cells"][n]
        lines.append(f"* N{n}: " + ", ".join(f"{k} {v['kept']}/{v['available']}" for k, v in cells.items()))
    lines += ["", "## Diffusion references: O (exact face flux) vs R (midpoint), pooled", "",
              "`|O - R| / |O|` and the least-squares scale `<O,R>/<R,R>` (about 1 when the sign and coefficient conventions agree; "
              "the two are different discretizations of the same continuum term, so they differ at the discretization level).", "",
              "| grid | mode | variant/field | rel L2 | fit scale | ok |", "|---|---|---|---|---|---|"]
    for n in grids:
        for mode in MODES:
            for name, e in summary["receipts"][n][mode]["diffusion_convention_check"]["entries"].items():
                lines.append(f"| N{n} | {mode} | {name} | {_fmt(e['rel_l2_O_minus_R'], '{:.3f}')} | {_fmt(e['fit_scale'], '{:.3f}')} | {e['ok']} |")
    flags = summary["flags"]
    kinds = {k: sum(1 for f in flags if f["kind"] == k) for k in ("ratio", "order_drop")}
    lines += ["", "## Flags", "",
              f"C3 worse than spline by more than {summary['ratio_flag']}x in relative L2: {kinds['ratio']} cases; an observed order "
              f"lower than spline's by more than {summary['order_flag']}: {kinds['order_drop']} cases (all in `summary.json`; at most "
              "40 of each below).  Small cells (about 6 owners) are noisy.", ""]
    ratio_flags = sorted((f for f in flags if f["kind"] == "ratio"), key=lambda f: -f["ratio"])[:40]
    if ratio_flags:
        lines += ["| ratio | N | variant | field | term/comparison | subset |", "|---|---|---|---|---|---|"]
        lines += [f"| {f['ratio']:.2f} | {f['n']} | {f['variant']} | {f['field']} | {f['term']}/{f['comparison']} | {f['subset']} |"
                  for f in ratio_flags]
    drops = [f for f in flags if f["kind"] == "order_drop"][:40]
    if drops:
        lines += ["", "| grids | variant | field | term/comparison | subset | order spline | order C3 |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {f['grids'][0]}->{f['grids'][1]} | {f['variant']} | {f['field']} | {f['term']}/{f['comparison']} | {f['subset']} | "
                  f"{f['order_spline']:.2f} | {f['order_compact_c3']:.2f} |" for f in drops]
    lines += ["", "## Geometry coefficients: relative change of C3 against spline (identical points)", "",
              "Per array and subset: RMS over points of the pointwise relative change `|dA|/|A|`, its max, and `rms(dA)/rms(A)`. "
              "`*_jacobian` and `*_J` are metric-only controls (expected 0).  The full region x knot table is in `summary.json`.", ""]
    coeff_subsets = [f"{r}|all" for r in ("all", *REGIONS)] + [f"all|{k}" for k in KNOTS]
    for n in grids:
        lines += [f"### N{n}", "", "| array | subset | points | rms rel | max rel | global rel |", "|---|---|---|---|---|---|"]
        for name, per_subset in summary["coefficients"][n].items():
            for subset in coeff_subsets:
                c = per_subset.get(subset)
                if c is not None:
                    lines.append(f"| {name} | {subset} | {c['n_points']} | {_fmt(c['rms_rel'])} | {_fmt(c['max_rel'])} | {_fmt(c['global_rel'])} |")
        lines.append("")
    region_subsets = [f"{r}|all" for r in ("all", *REGIONS)]
    knot_subsets = [s for s in summary["subsets"] if s not in region_subsets]
    lines += ["## Errors by region (pooled over the knot classes)", ""]
    for variant in summary["variants"]:
        for term, comparison in summary["comparisons"]:
            key = f"{term}/{comparison}"
            lines += [f"### {variant}: {key}", ""] + _table(summary, variant, key, region_subsets) + [""]
    lines += ["## Errors by knot class (and region x knot class)", ""]
    for variant in summary["variants"]:
        for term, comparison in summary["comparisons"]:
            key = f"{term}/{comparison}"
            lines += [f"### {variant}: {key}", ""] + _table(summary, variant, key, knot_subsets) + [""]
    return "\n".join(lines) + "\n"


def write_outputs(directory, summary: dict) -> tuple:
    out = Path(directory)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False))
    (out / "report.md").write_text(render_report(summary))
    return out / "report.md", out / "summary.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--ratio-flag", type=float, default=RATIO_FLAG)
    parser.add_argument("--order-flag", type=float, default=ORDER_FLAG)
    args = parser.parse_args(argv)
    summary = build_summary(args.directory, ratio_flag=args.ratio_flag, order_flag=args.order_flag)
    report, summary_path = write_outputs(args.directory, summary)
    print(f"wrote {report} and {summary_path} ({len(summary['flags'])} flags, grids {summary['grids']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
