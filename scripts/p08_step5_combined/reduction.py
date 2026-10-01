"""The reduction of the P08 step-5.3 campaign: one grid's checkpoints -> ``results.npz`` + ``summary.json``.

Pure NumPy: it reads the owner context (volumes and region masks), the merged references and the per-variant JAX
checkpoints, and never builds an environment. For every variant, field, term and region it reduces three comparisons

* ``presc_vs_ref``   ``N_presc  - R``   (prescribed arm against the re-frozen reference),
* ``solved_vs_ref``  ``N_solved - R``   (solved arm against the reference),
* ``solved_vs_presc`` ``N_solved - N_presc`` (what the phi solve changes),

each as the owner-volume-weighted L2 and the max of the difference, absolute and relative to the reference's own L2 /
max. **Degenerate references** (the step-3 design rule, section 8.2): a term whose reference L2 in the region is below
``degenerate_fraction`` of the largest production-term reference L2 of the same variant, field and region has no scale
of its own (e.g. a bracket ``[phi, g]`` with ``phi`` a function of ``g``); its relative errors use that largest term
(``degenerate`` is set). When even that scale is zero (the constant control, whose references vanish identically) the
relative errors are undefined (NaN in ``results.npz``, ``None`` in JSON) and only the absolute errors are meaningful.

Also reduced: the potential error ``phi_h - phi_bar`` (relative to ``phi_bar``), and the psi diffusion triple
``O - R_mid`` (exact face flux against the autodiff midpoint reference), ``N - O`` (the discrete P07 action on ``psi_bar``
against ``O``; the solver gate (b)) and ``N - R_mid``, with the solver's iterations / seconds / residuals and the gates.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import runner                                                       # noqa: E402
from p_shared.replay_support import owner_weighted_l2                             # noqa: E402
from p08_step5_combined import references                                         # noqa: E402
from p08_step5_combined.checkpoints import arm_key, artifact_sha, jax_identity, load_variant   # noqa: E402

SCHEMA = "drbx.p08-step5-combined-results.v1"
FIELDS = references.FIELDS
TERMS = references.REF_TERMS
#: the production terms that set the scale of a degenerate reference
PRODUCTION_TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion")
CMPS = ("presc_vs_ref", "solved_vs_ref", "solved_vs_presc")
PSI_CMPS = ("O_minus_R", "N_minus_O", "N_minus_R")
METRICS = ("l2", "max_abs", "ref_l2", "ref_max", "rel_l2", "rel_max")
RESULTS = "results.npz"
SUMMARY = "summary.json"


def row_key(variant: str, cmp: str, field: str, term: str, region: str) -> str:
    return "|".join((variant, cmp, field, term, region))


def grid_dir(output, n: int) -> Path:
    return Path(output) / f"N{int(n)}"


def _nan(x):
    return float("nan") if x is None else float(x)


def _masked(arr, mask):
    return arr if mask is None else arr[mask]


def term_metrics(diff, ref, volume, mask=None, *, fallback_l2=None, fallback_max=None, degenerate_fraction=1e-8):
    """L2 / max of ``diff`` (absolute, and relative to ``ref``'s L2 / max), the degenerate-reference rule above.
    ``None`` for an empty region."""
    diff = np.asarray(diff, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    l2 = owner_weighted_l2(diff, volume, mask)
    ref_l2 = owner_weighted_l2(ref, volume, mask)
    if l2 is None or ref_l2 is None:
        return None
    d, r = _masked(diff, mask), _masked(ref, mask)
    max_abs, ref_max = float(np.max(np.abs(d))), float(np.max(np.abs(r)))
    degenerate = bool(fallback_l2 is not None and ref_l2 < degenerate_fraction * fallback_l2)
    scale_l2, scale_max = (fallback_l2, fallback_max) if degenerate else (ref_l2, ref_max)

    def ratio(a, b):
        return float(a / b) if (b is not None and b > 0.0 and math.isfinite(a) and math.isfinite(b)) else float("nan")

    return {"l2": float(l2), "max_abs": max_abs, "ref_l2": float(ref_l2), "ref_max": ref_max,
            "rel_l2": ratio(l2, scale_l2), "rel_max": ratio(max_abs, scale_max), "degenerate": degenerate,
            "finite": bool(np.all(np.isfinite(d)) and np.all(np.isfinite(r))), "count": int(d.size)}


def region_masks(regions: dict) -> dict:
    """``{"global": None, <region>: mask, ...}`` in a fixed order."""
    return {"global": None, **{name: regions[name] for name in sorted(regions)}}


def reduce_variant(variant: str, spec: dict, arrays: dict, refs: dict, volume, regions: dict, *,
                   degenerate_fraction: float) -> dict:
    """``{"rows": {key: metrics}, "phi": {region: metrics}, "psi": {cmp: {region: metrics}}}`` of one variant."""
    fs = spec["field_set"]
    masks = region_masks(regions)
    rows: dict = {}
    for field in FIELDS:
        ref = {term: np.asarray(refs[references.ref_key(fs, field, term)], dtype=np.float64) for term in TERMS}
        presc = {term: np.asarray(arrays[arm_key("presc", field, term)], dtype=np.float64) for term in TERMS}
        solved = {term: np.asarray(arrays[arm_key("solved", field, term)], dtype=np.float64) for term in TERMS}
        for region, mask in masks.items():
            scales = [(owner_weighted_l2(ref[t], volume, mask), float(np.max(np.abs(_masked(ref[t], mask)))))
                      for t in PRODUCTION_TERMS if (mask is None or mask.any())]
            scales = [s for s in scales if s[0] is not None and math.isfinite(s[0])]
            fallback_l2 = max((s[0] for s in scales), default=None)
            fallback_max = max((s[1] for s in scales), default=None)
            for term in TERMS:
                for cmp, diff in (("presc_vs_ref", presc[term] - ref[term]), ("solved_vs_ref", solved[term] - ref[term]),
                                  ("solved_vs_presc", solved[term] - presc[term])):
                    m = term_metrics(diff, ref[term], volume, mask, fallback_l2=fallback_l2,
                                     fallback_max=fallback_max, degenerate_fraction=degenerate_fraction)
                    if m is not None:
                        rows[row_key(variant, cmp, field, term, region)] = m
    phi = {}
    for region, mask in masks.items():
        m = term_metrics(arrays["phi_h"] - arrays["phi_bar"], arrays["phi_bar"], volume, mask,
                         degenerate_fraction=degenerate_fraction)
        if m is not None:
            phi[region] = m
    o_ref = np.asarray(refs[references.psi_key(fs, "O")], dtype=np.float64)
    r_mid = np.asarray(refs[references.psi_key(fs, "R_mid")], dtype=np.float64)
    n_psi = np.asarray(arrays["psi_N"], dtype=np.float64)
    psi = {cmp: {} for cmp in PSI_CMPS}
    for region, mask in masks.items():
        for cmp, diff, ref in (("O_minus_R", o_ref - r_mid, r_mid), ("N_minus_O", n_psi - o_ref, o_ref),
                               ("N_minus_R", n_psi - r_mid, r_mid)):
            m = term_metrics(diff, ref, volume, mask, degenerate_fraction=degenerate_fraction)
            if m is not None:
                psi[cmp][region] = m
    return {"rows": rows, "phi": phi, "psi": psi}


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------
def save_results(path, rows: dict) -> None:
    keys = sorted(rows)
    arrays = {"keys": np.asarray(keys)}
    for name in METRICS:
        arrays[name] = np.asarray([_nan(rows[k][name]) for k in keys], dtype=np.float64)
    arrays["degenerate"] = np.asarray([rows[k]["degenerate"] for k in keys], dtype=bool)
    arrays["finite"] = np.asarray([rows[k]["finite"] for k in keys], dtype=bool)
    arrays["count"] = np.asarray([rows[k]["count"] for k in keys], dtype=np.int64)
    runner.save_npz(path, **arrays)


def load_results(path) -> dict:
    """``{key: {metric: value}}`` of a ``results.npz`` (NaN stays NaN)."""
    with np.load(path, allow_pickle=False) as z:
        keys = [str(k) for k in z["keys"]]
        data = {name: z[name] for name in (*METRICS, "degenerate", "finite", "count")}
    out = {}
    for i, key in enumerate(keys):
        out[key] = {name: (bool(data[name][i]) if name in ("degenerate", "finite") else
                           (int(data[name][i]) if name == "count" else float(data[name][i])))
                    for name in data}
    return out


def _clean(m):
    """A metrics dict with NaN / inf replaced by ``None`` (JSON)."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in m.items()}


# ---------------------------------------------------------------------------
# One grid
# ---------------------------------------------------------------------------
def reduce_grid(*, n: int, cfg: dict, output, identity: str, inputs: dict) -> dict:
    """Reduce grid ``n`` (all variants must have a valid JAX checkpoint); writes ``N{n}/results.npz`` and
    ``N{n}/summary.json`` and returns the summary."""
    output = Path(output)
    started = time.time()
    refs = references.load_references(output, n, identity)
    volume, regions = references.load_context_file(output, n)
    jid = jax_identity(identity, n, artifact_sha(inputs, n))
    degenerate_fraction = float(cfg["degenerate_fraction"])
    all_rows: dict = {}
    variants: dict = {}
    for variant, spec in cfg["variants"].items():
        arrays, info = load_variant(output, n, variant, jid)
        reduced = reduce_variant(variant, spec, arrays, refs, volume, regions, degenerate_fraction=degenerate_fraction)
        all_rows.update(reduced["rows"])
        global_rows = {}
        for key, m in reduced["rows"].items():
            _v, cmp, field, term, region = key.split("|")
            if region == "global":
                global_rows.setdefault(cmp, {}).setdefault(field, {})[term] = _clean(m)
        variants[variant] = {
            "field_set": spec["field_set"], "constant": bool(spec["constant"]), "info": info,
            "global_rows": global_rows,
            "phi_error": {r: _clean(m) for r, m in reduced["phi"].items()},
            "psi_diffusion": {c: {r: _clean(m) for r, m in by.items()} for c, by in reduced["psi"].items()},
            "all_finite": bool(info["gates"]["finite"] and all(m["finite"] for m in reduced["rows"].values())
                               and all(m["finite"] for m in reduced["phi"].values())),
        }
        del arrays
    results_path = grid_dir(output, n) / RESULTS
    save_results(results_path, all_rows)
    solver_gates = {v: bool(rec["info"]["gates"]["consistency"] and rec["info"]["gates"]["converged"])
                    for v, rec in variants.items()}
    grid = {"solver_gates_pass": bool(all(solver_gates.values())), "solver_gates": solver_gates,
            "all_finite": bool(all(rec["all_finite"] for rec in variants.values()))}
    grid["grid_pass"] = bool(grid["solver_gates_pass"] and grid["all_finite"])
    summary = {"schema": SCHEMA, "identity": identity, "jax_identity": jid, "n": int(n),
               "n_owners": int(len(volume)), "regions": ["global", *sorted(regions)],
               "region_owner_counts": {r: int(np.count_nonzero(m)) for r, m in regions.items()},
               "params": cfg["params"], "gates": grid, "variants": variants, "rows": len(all_rows),
               "results_file": RESULTS, "results_sha256": runner.sha256_file(results_path),
               "wall_seconds": time.time() - started}
    runner.write_json(grid_dir(output, n) / SUMMARY, summary)
    return summary


def load_grid(output, n: int, identity: str) -> tuple[dict, dict]:
    """``(summary, results)`` of a reduced grid; the identity and the results checksum are verified."""
    folder = grid_dir(output, n)
    summary_path, results_path = folder / SUMMARY, folder / RESULTS
    if not (summary_path.is_file() and results_path.is_file()):
        raise ValueError(f"no reduction for N{n} under {folder}; run the reduce stage first")
    summary = json.loads(summary_path.read_text())
    if summary.get("identity") != identity:
        raise ValueError(f"the reduction of N{n} belongs to another campaign identity")
    if summary.get("results_sha256") != runner.sha256_file(results_path):
        raise ValueError(f"results.npz of N{n} does not match its summary")
    return summary, load_results(results_path)
