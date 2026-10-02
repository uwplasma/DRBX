"""Across-grid analysis of the P08 step-6 campaign: the 5.3 report (orders, solver table, potential error, psi
diffusion), the u-band tables, and the comparison against the re-freeze campaign's catalogue ``main_phi_dirichlet``.

Everything here reads reduced grids (``reduction.load_grid``) and JSON; pure NumPy. Nothing accepts or rejects the
step: the user decides.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p08_step5_combined import analysis as ana5                                   # noqa: E402
from p08_step5_combined import reduction                                          # noqa: E402

SCHEMA = "drbx.p08-step6-summary.v1"
TITLE_5_3 = "# P08 step 5.3: static Dirichlet-phi combined full-grid MMS"
TITLE = "# P08 step 6: transverse-wave check on the compact_c3 re-freeze (full grid, prescribed and solved phi)"
ARMS = ("presc_vs_ref", "solved_vs_ref")
FIELDS = reduction.FIELDS


def band_names(cfg: dict) -> list:
    from p08_step6_global import references as refs6
    e = [float(x) for x in cfg["u_band_edges"]]
    return [refs6.band_name(lo, hi) for lo, hi in zip(e[:-1], e[1:])]


def band_rows(ord_: dict, cfg: dict, ns: list) -> list:
    """One row per variant, arm, field and band (term ``total``): the relative L2 / L2 error per grid and the orders."""
    rows = []
    for variant, spec in cfg["variants"].items():
        if spec["constant"]:
            continue
        for cmp in ARMS:
            for field in FIELDS:
                for band in band_names(cfg):
                    entry = ord_.get(reduction.row_key(variant, cmp, field, "total", band))
                    if entry is None:
                        continue
                    rows.append({"variant": variant, "cmp": cmp, "field": field, "band": band, "n": entry["n"],
                                 "rel_l2": entry["rel_l2"], "l2": entry["l2"], "order_rel_l2": entry["order_rel_l2"],
                                 "order_l2": entry["order_l2"], "degenerate": entry["degenerate"]})
    return rows


def psi_band_rows(summaries: dict, cfg: dict) -> list:
    """psi diffusion triple per variant, comparison and band: relative L2 per grid and its orders."""
    rows = []
    ns = sorted(summaries)
    for variant in cfg["variants"]:
        for cmp in reduction.PSI_CMPS:
            for band in band_names(cfg):
                cells = []
                for n in ns:
                    m = summaries[n]["variants"][variant]["psi_diffusion"][cmp].get(band)
                    cells.append(m["rel_l2"] if m else None)
                rows.append({"variant": variant, "cmp": cmp, "band": band, "n": ns, "rel_l2": cells,
                             "order_rel_l2": [ana5.observed_order(cells[i], cells[i + 1], ns[i], ns[i + 1])
                                              for i in range(len(ns) - 1)]})
    return rows


def read_catalogue_summary(refreeze, identity: str | None = None) -> dict:
    """The re-freeze campaign's ``summary/step5_combined_summary.json`` (its identity checked when given)."""
    path = Path(refreeze) / "summary" / "step5_combined_summary.json"
    if not path.is_file():
        raise ValueError(f"missing re-freeze catalogue summary: {path}")
    summary = json.loads(path.read_text())
    if identity is not None and summary.get("identity") != identity:
        raise ValueError("the re-freeze catalogue summary belongs to another campaign identity")
    return summary


def catalogue_comparison(catalogue: dict, ord_: dict, summaries: dict, cfg: dict) -> dict:
    """Global ``total`` errors of every transverse variant against the catalogue's ``main_phi_dirichlet`` (and
    ``dirichlet_rich``) of the re-freeze campaign, per field and arm: the catalogue's and the transverse relative L2 /
    L2 error per grid, their ratio, and both order sets; plus the phi error and the psi N - O of both."""
    cname = cfg["comparison"]["catalogue_variants"]
    ns = sorted(summaries)
    cat_orders = catalogue["orders"]
    out = {"catalogue_variants": cname, "grids": ns, "rows": [], "phi_psi": []}

    def series(entry, metric):
        have = dict(zip(entry["n"], entry[metric])) if entry else {}
        return [have.get(n) for n in ns]

    for variant, spec in cfg["variants"].items():
        if spec["constant"]:
            continue
        for ref_variant in cname:
            for cmp in ARMS:
                for field in FIELDS:
                    key_c = reduction.row_key(ref_variant, cmp, field, "total", "global")
                    key_t = reduction.row_key(variant, cmp, field, "total", "global")
                    ec, et = cat_orders.get(key_c), ord_.get(key_t)
                    row = {"variant": variant, "catalogue_variant": ref_variant, "cmp": cmp, "field": field,
                           "rel_l2_catalogue": series(ec, "rel_l2"), "rel_l2_transverse": series(et, "rel_l2"),
                           "l2_catalogue": series(ec, "l2"), "l2_transverse": series(et, "l2"),
                           "order_rel_l2_catalogue": (ec or {}).get("order_rel_l2"),
                           "order_rel_l2_transverse": (et or {}).get("order_rel_l2")}
                    row["rel_l2_ratio"] = [None if (a is None or b is None or not b) else a / b
                                           for a, b in zip(row["rel_l2_transverse"], row["rel_l2_catalogue"])]
                    out["rows"].append(row)
        for ref_variant in cname:
            rec = {"variant": variant, "catalogue_variant": ref_variant, "n": ns}
            for label, getter in (
                    ("phi_error_rel_l2", lambda s, v: (s["variants"][v]["phi_error"].get("global") or {}).get("rel_l2")),
                    ("n_minus_o_rel_l2", lambda s, v: (s["variants"][v]["psi_diffusion"]["N_minus_O"].get("global")
                                                       or {}).get("rel_l2"))):
                rec[f"{label}_transverse"] = [getter(summaries[n], variant) for n in ns]
                rec[f"{label}_catalogue"] = [_catalogue_value(catalogue, n, ref_variant, label) for n in ns]
            out["phi_psi"].append(rec)
    return out


def _catalogue_value(catalogue: dict, n: int, variant: str, label: str):
    try:
        if label == "phi_error_rel_l2":
            return catalogue["phi_error"][str(n)][variant]["global"]["rel_l2"]
        return catalogue["psi_diffusion"][str(n)][variant]["N_minus_O"]["global"]["rel_l2"]
    except (KeyError, TypeError):
        return None


def _orders_cell(orders_):
    return ana5._orders_cell(orders_ or [])


def build_extra_report(*, band: list, psi_band: list, comparison: dict, ns: list) -> str:
    fmt = ana5._fmt
    lines = ["", "## u-band errors (term `total`, relative L2 within the band; orders between consecutive grids)", "",
             "Bands are owner ring-centre `u` ranges; each relative error is relative to the band's own reference L2 "
             "(degenerate-reference rule of 5.3 applied per band).", ""]
    head = ["variant", "arm", "field", "band"] + [f"rel_l2 N{n}" for n in ns] + ["orders"]
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in band:
        vals = dict(zip(r["n"], r["rel_l2"]))
        lines.append("| " + " | ".join([r["variant"], r["cmp"], r["field"], r["band"]] + [fmt(vals.get(n)) for n in ns] +
                                      [_orders_cell(r["order_rel_l2"])]) + " |")
    lines += ["", "## psi diffusion by u-band (relative L2; orders)", ""]
    head = ["variant", "comparison", "band"] + [f"rel_l2 N{n}" for n in ns] + ["orders"]
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in psi_band:
        vals = dict(zip(r["n"], r["rel_l2"]))
        lines.append("| " + " | ".join([r["variant"], r["cmp"], r["band"]] + [fmt(vals.get(n)) for n in ns] +
                                      [_orders_cell(r["order_rel_l2"])]) + " |")
    lines += ["", "## Comparison with the re-freeze catalogue (global `total`, relative L2)", "",
              "Ratio = transverse / catalogue per grid. The catalogue numbers are read from the re-freeze campaign's "
              "`summary/step5_combined_summary.json`.", ""]
    head = ["variant", "vs", "arm", "field"] + [f"cat N{n}" for n in ns] + [f"trans N{n}" for n in ns] + \
           ["cat orders", "trans orders"]
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in comparison["rows"]:
        lines.append("| " + " | ".join([r["variant"], r["catalogue_variant"], r["cmp"], r["field"]] +
                                      [fmt(v) for v in r["rel_l2_catalogue"]] + [fmt(v) for v in r["rel_l2_transverse"]] +
                                      [_orders_cell(r["order_rel_l2_catalogue"]),
                                       _orders_cell(r["order_rel_l2_transverse"])]) + " |")
    lines += ["", "### Potential error and psi N - O (global relative L2)", ""]
    for r in comparison["phi_psi"]:
        for label in ("phi_error_rel_l2", "n_minus_o_rel_l2"):
            lines.append(f"- {r['variant']} vs {r['catalogue_variant']} {label}: transverse " +
                         ", ".join(f"N{n} {fmt(v)}" for n, v in zip(ns, r[f"{label}_transverse"])) + "; catalogue " +
                         ", ".join(f"N{n} {fmt(v)}" for n, v in zip(ns, r[f"{label}_catalogue"])))
    lines.append("")
    return "\n".join(lines)


def analyze(*, output, identity: str, grids, cfg: dict, catalogue: dict) -> dict:
    """Write ``summary/step6_report.md`` and ``summary/step6_summary.json`` from the reduced grids."""
    from p_shared import runner

    output = Path(output)
    summaries, results_by_n = {}, {}
    for n in grids:
        summaries[int(n)], results_by_n[int(n)] = reduction.load_grid(output, n, identity)
    ns = sorted(summaries)
    headline = ana5.headline_criterion(results_by_n, cfg)
    ord_ = ana5.orders(results_by_n)
    base = ana5.build_report(identity=identity, cfg=cfg, summaries=summaries, results_by_n=results_by_n, headline=headline)
    if TITLE_5_3 not in base:
        raise ValueError("the 5.3 report layout changed; update the step-6 title replacement")
    band = band_rows(ord_, cfg, ns)
    psi_band = psi_band_rows(summaries, cfg)
    comparison = catalogue_comparison(catalogue, ord_, summaries, cfg)
    report = base.replace(TITLE_5_3, TITLE) + build_extra_report(band=band, psi_band=psi_band, comparison=comparison, ns=ns)
    (output / "summary").mkdir(parents=True, exist_ok=True)
    (output / "summary" / "step6_report.md").write_text(report)

    def clean(entry):
        return {k: ([None if (isinstance(x, float) and not math.isfinite(x)) else x for x in v]
                    if isinstance(v, list) else v) for k, v in entry.items()}

    payload = {"schema": SCHEMA, "identity": identity, "grids": ns, "operator_options": cfg["operator_options"],
               "params": cfg["params"], "fields": cfg["fields"], "field_sets": cfg["field_sets"],
               "phi_solve": {"rtol": cfg["phi_solve_rtol"], "rtol_production_default": cfg["phi_solve_rtol_production_default"],
                             "restart": cfg["phi_solve_restart"], "max_restarts": cfg["phi_solve_max_restarts"],
                             "factor_dtype": cfg["phi_solve_factor_dtype"], "preconditioner": cfg["phi_solve_preconditioner"]},
               "acceptance_gate": None, "grid_gates": {str(n): summaries[n]["gates"] for n in summaries},
               "solver": ana5.solver_table(summaries), "headline_order_criterion": headline,
               "orders": {k: clean(v) for k, v in ord_.items()},
               "phi_error": {str(n): {v: rec["phi_error"] for v, rec in summaries[n]["variants"].items()} for n in summaries},
               "psi_diffusion": {str(n): {v: rec["psi_diffusion"] for v, rec in summaries[n]["variants"].items()}
                                 for n in summaries},
               "regions": summaries[min(summaries)]["regions"], "u_band_edges": cfg["u_band_edges"],
               "u_band_table": band, "psi_band_table": psi_band, "catalogue_comparison": comparison}
    runner.write_json(output / "summary" / "step6_summary.json", payload)
    return payload
