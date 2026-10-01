"""Across-grid analysis of the P08 step-5.3 campaign: observed orders, the (informational) headline order criterion,
the solver-gate table and the markdown / JSON reports.

Everything here reads the reduced grids (``reduction.load_grid``) and is pure NumPy / JSON. Nothing in this module
accepts or rejects the step: the solver gates and finiteness are recorded as pass/fail by the reduction and
``campaign validate``; the order criterion below is recorded and marked informational ("the user decides").
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p08_step5_combined import reduction                                           # noqa: E402

SCHEMA = "drbx.p08-step5-combined-summary.v1"
ORDER_METRICS = ("rel_l2", "l2", "max_abs")


def observed_order(coarse, fine, n_coarse: int, n_fine: int):
    """``log(e_coarse / e_fine) / log(n_fine / n_coarse)`` (``None`` unless both errors are positive and finite)."""
    if coarse is None or fine is None:
        return None
    if not (math.isfinite(coarse) and math.isfinite(fine) and coarse > 0.0 and fine > 0.0):
        return None
    return float(math.log(coarse / fine) / math.log(n_fine / n_coarse))


def _value(m, name):
    v = m.get(name) if m else None
    return v if (v is not None and math.isfinite(v)) else None


def orders(results_by_n: dict) -> dict:
    """``{row key: {"n": [...], <metric>: [...], "order_<metric>": [... one per consecutive pair]}}`` over the grids
    that contain the row (``results_by_n = {n: reduction.load_results(...)}``)."""
    ns = sorted(results_by_n)
    keys = sorted(set().union(*(set(results_by_n[n]) for n in ns)))
    out = {}
    for key in keys:
        have = [n for n in ns if key in results_by_n[n]]
        entry = {"n": have}
        for metric in ORDER_METRICS:
            series = [_value(results_by_n[n][key], metric) for n in have]
            entry[metric] = series
            entry[f"order_{metric}"] = [observed_order(series[i], series[i + 1], have[i], have[i + 1])
                                        for i in range(len(have) - 1)]
        entry["degenerate"] = [bool(results_by_n[n][key]["degenerate"]) for n in have]
        entry["finite"] = [bool(results_by_n[n][key]["finite"]) for n in have]
        out[key] = entry
    return out


def headline_criterion(results_by_n: dict, cfg: dict) -> dict:
    """The roadmap's order criterion (informational): the global relative L2 order of the ``total`` term of every
    non-constant variant and field, for the prescribed and the solved arm against the reference, is >= ``min_order``
    on every consecutive interval. ``pass`` is ``False`` as soon as one row fails, ``None`` when no row fails but some
    order is undefined (or fewer than two grids), else ``True``."""
    spec = cfg["order_criterion"]
    ns = sorted(results_by_n)
    ord_ = orders(results_by_n)
    rows, failed, undefined = [], 0, 0
    for variant, vspec in cfg["variants"].items():
        if vspec["constant"]:
            continue
        for cmp in spec["arms"]:
            for field in ("density", "Te", "Ti", "vorticity"):
                key = reduction.row_key(variant, cmp, field, spec["term"], "global")
                entry = ord_.get(key)
                orders_ = entry["order_rel_l2"] if entry else []
                if len(ns) < 2 or not orders_ or any(o is None for o in orders_):
                    status = None
                    undefined += 1
                else:
                    status = bool(all(o >= spec["min_order"] for o in orders_))
                    failed += 0 if status else 1
                rows.append({"variant": variant, "cmp": cmp, "field": field, "term": spec["term"],
                             "rel_l2": entry["rel_l2"] if entry else None, "orders": orders_, "pass": status})
    overall = False if failed else (None if undefined else True)
    return {"informational": True, "user_decides": True, "min_order": spec["min_order"], "term": spec["term"],
            "grids": ns, "pass": overall, "rows_failed": failed, "rows_undefined": undefined, "rows": rows}


def solver_table(summaries: dict) -> list:
    """One row per grid and variant: the solver gates and numbers."""
    rows = []
    for n in sorted(summaries):
        for variant, rec in summaries[n]["variants"].items():
            info = rec["info"]
            rows.append({"n": n, "variant": variant,
                         "consistency_error": info["consistency"]["relative_error_M"],
                         "consistency_pass": info["consistency"]["pass"],
                         "consistency_iterations": info["consistency"]["solve"]["iterations"],
                         "converged": info["gates"]["converged"], "iterations": info["solved_solve"]["iterations"],
                         "seconds": info["solved_solve"]["seconds"],
                         "relative_residual": info["solved_solve"]["relative_residual"],
                         "n_minus_o_psi": info["n_minus_o_psi"]["relative"],
                         "all_finite": rec["all_finite"], "gates_pass": info["gates_pass"]})
    return rows


def _fmt(x, spec=".2e"):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, spec)


def _orders_cell(orders_):
    return " / ".join("-" if o is None else f"{o:.2f}" for o in orders_) or "-"


def _series_table(title: str, rows: list, ns: list) -> list:
    head = ["row"] + [f"rel_l2 N{n}" for n in ns] + ["orders"]
    lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for label, entry in rows:
        vals = dict(zip(entry["n"], entry["rel_l2"]))
        lines.append("| " + " | ".join([label] + [_fmt(vals.get(n)) for n in ns] +
                                      [_orders_cell(entry["order_rel_l2"])]) + " |")
    return lines + [""]


def build_report(*, identity: str, cfg: dict, summaries: dict, results_by_n: dict, headline: dict) -> str:
    ns = sorted(summaries)
    ord_ = orders(results_by_n)
    lines = [f"# P08 step 5.3: static Dirichlet-phi combined full-grid MMS", "",
             f"Campaign identity `{identity}`; grids {ns}. Decision data only: nothing here accepts or rejects the "
             "step (the user decides). Orders are `log(e_coarse/e_fine)/log(n_fine/n_coarse)` of the owner-volume "
             "weighted L2 error relative to the reference L2 (global region); `-` = undefined.", "",
             f"Operator options {cfg['operator_options']}; parameters {cfg['params']}; phi solve: Dirichlet, "
             f"{cfg['phi_solve_preconditioner']}, {cfg['phi_solve_factor_dtype']} factors, rtol "
             f"{cfg['phi_solve_rtol']:g} (measurement setting; production default "
             f"{cfg['phi_solve_rtol_production_default']:g}), restart {cfg['phi_solve_restart']}, max_restarts "
             f"{cfg['phi_solve_max_restarts']}.", "", "## Gates", "",
             "| N | variant | consistency error | pass | cons. it | converged | solve it | solve s | rel. residual | "
             "N-O(psi) | finite | gates |", "|" + "|".join(["---"] * 12) + "|"]
    for r in solver_table(summaries):
        lines.append(f"| {r['n']} | {r['variant']} | {_fmt(r['consistency_error'])} | {r['consistency_pass']} | "
                     f"{r['consistency_iterations']} | {r['converged']} | {r['iterations']} | "
                     f"{_fmt(r['seconds'], '.1f')} | {_fmt(r['relative_residual'])} | {_fmt(r['n_minus_o_psi'])} | "
                     f"{r['all_finite']} | {'PASS' if r['gates_pass'] else 'FAIL'} |")
    lines += ["", f"Grid gates (solver gates and finiteness): " +
              ", ".join(f"N{n}: {'PASS' if summaries[n]['gates']['grid_pass'] else 'FAIL'}" for n in ns), "",
              "## Headline order criterion (informational)", "",
              f"Global `{headline['term']}` relative L2 order >= {headline['min_order']} on every interval, "
              f"prescribed and solved arm, every non-constant variant and field: "
              f"**{'PASS' if headline['pass'] else ('FAIL' if headline['pass'] is False else 'not evaluable')}** "
              f"({headline['rows_failed']} failing, {headline['rows_undefined']} undefined rows). "
              "Recorded only; the user decides.", "",
              "| variant | arm | field | " + " | ".join(f"rel_l2 N{n}" for n in ns) + " | orders | pass |",
              "|" + "|".join(["---"] * (5 + len(ns))) + "|"]
    for r in headline["rows"]:
        vals = dict(zip(headline["grids"], r["rel_l2"] or []))
        lines.append(f"| {r['variant']} | {r['cmp']} | {r['field']} | " + " | ".join(_fmt(vals.get(n)) for n in ns) +
                     f" | {_orders_cell(r['orders'])} | {r['pass']} |")
    lines += ["", "## Per-term errors (global, relative L2; orders between consecutive grids)", ""]
    for variant in cfg["variants"]:
        rows = []
        for cmp in reduction.CMPS:
            for field in reduction.FIELDS:
                for term in reduction.TERMS:
                    key = reduction.row_key(variant, cmp, field, term, "global")
                    if key in ord_:
                        flag = " (degenerate ref.)" if any(ord_[key]["degenerate"]) else ""
                        rows.append((f"{cmp} / {field} / {term}{flag}", ord_[key]))
        lines += _series_table(variant, rows, ns)
    lines += ["## Potential error phi_h - phi_bar (global, relative to phi_bar)", ""]
    for variant in cfg["variants"]:
        cells = []
        for n in ns:
            m = summaries[n]["variants"][variant]["phi_error"].get("global")
            cells.append(m["rel_l2"] if m else None)
        order_list = [observed_order(cells[i], cells[i + 1], ns[i], ns[i + 1]) for i in range(len(ns) - 1)]
        lines.append(f"- {variant}: " + ", ".join(f"N{n} {_fmt(c)}" for n, c in zip(ns, cells)) +
                     f"; orders {_orders_cell(order_list)}")
    lines += ["", "## psi diffusion (global, relative): O - R_mid, N - O, N - R_mid", ""]
    for variant in cfg["variants"]:
        for cmp in reduction.PSI_CMPS:
            cells = []
            for n in ns:
                m = summaries[n]["variants"][variant]["psi_diffusion"][cmp].get("global")
                cells.append(m["rel_l2"] if m else None)
            order_list = [observed_order(cells[i], cells[i + 1], ns[i], ns[i + 1]) for i in range(len(ns) - 1)]
            lines.append(f"- {variant} {cmp}: " + ", ".join(f"N{n} {_fmt(c)}" for n, c in zip(ns, cells)) +
                         f"; orders {_orders_cell(order_list)}")
    lines += ["", "Region-resolved rows (all P06N region masks) and absolute / max errors are in "
              "`summary/step5_combined_summary.json` and `N{n}/results.npz`.", ""]
    return "\n".join(lines)


def analyze(*, output, identity: str, grids, cfg: dict) -> dict:
    """Write ``summary/step5_combined_report.md`` and ``summary/step5_combined_summary.json`` from the reduced grids."""
    from p_shared import runner

    output = Path(output)
    summaries, results_by_n = {}, {}
    for n in grids:
        summaries[int(n)], results_by_n[int(n)] = reduction.load_grid(output, n, identity)
    headline = headline_criterion(results_by_n, cfg)
    ord_ = orders(results_by_n)
    report = build_report(identity=identity, cfg=cfg, summaries=summaries, results_by_n=results_by_n,
                          headline=headline)
    (output / "summary").mkdir(parents=True, exist_ok=True)
    (output / "summary" / "step5_combined_report.md").write_text(report)

    def clean(entry):
        return {k: ([None if (isinstance(x, float) and not math.isfinite(x)) else x for x in v]
                    if isinstance(v, list) else v) for k, v in entry.items()}

    payload = {"schema": SCHEMA, "identity": identity, "grids": sorted(summaries),
               "operator_options": cfg["operator_options"], "params": cfg["params"],
               "phi_solve": {"rtol": cfg["phi_solve_rtol"], "rtol_production_default": cfg["phi_solve_rtol_production_default"],
                             "restart": cfg["phi_solve_restart"], "max_restarts": cfg["phi_solve_max_restarts"],
                             "factor_dtype": cfg["phi_solve_factor_dtype"],
                             "preconditioner": cfg["phi_solve_preconditioner"]},
               "acceptance_gate": None, "grid_gates": {str(n): summaries[n]["gates"] for n in summaries},
               "solver": solver_table(summaries), "headline_order_criterion": headline,
               "orders": {k: clean(v) for k, v in ord_.items()},
               "phi_error": {str(n): {v: rec["phi_error"] for v, rec in summaries[n]["variants"].items()}
                             for n in summaries},
               "psi_diffusion": {str(n): {v: rec["psi_diffusion"] for v, rec in summaries[n]["variants"].items()}
                                 for n in summaries},
               "regions": summaries[min(summaries)]["regions"]}
    runner.write_json(output / "summary" / "step5_combined_summary.json", payload)
    return payload
