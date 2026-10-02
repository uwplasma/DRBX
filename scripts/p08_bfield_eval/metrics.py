"""Owner-volume-weighted error metrics and their aggregation by region / knot class (numpy only, no jax).

Per-owner arrays (the content of ``arrays.npz``; ``V`` variants, ``F`` fields, ``T`` terms, ``m`` owners):

* ``N[v, f, t, m]``   the operator (combined RHS), term order :data:`TERMS` (the last one is ``total``);
* ``R[v, f, t, m]``   the ``perpendicular_reference_rhs.reference_rhs`` reference of that term (for the diffusion term this is O,
  the exact-face-flux O_q3 reference);
* ``Rmid[v, f, m]``   the midpoint diffusion reference (:mod:`p08_bfield_eval.references`);
* ``volume[m]``, ``region[m]`` (codes into ``REGIONS``), ``knot[m]`` (codes into ``KNOTS``).

Comparisons per (variant, field): ``N-R`` of the bracket / curvature / total terms against ``R``; for the diffusion term
``N-O`` (against ``R`` = the exact face flux O), ``O-R`` (O against the midpoint reference ``Rmid``) and ``N-R`` (against
``Rmid``).  Subsets are named ``"region|knot"`` with ``all`` for the pooled axis (:func:`subset_masks`).  The relative errors
are normalized as ``p_shared.step3_gates.term_metrics`` does (``fallback`` = the largest reference L2 / max over the terms of
the field, used when a reference vanishes identically); the formulas are re-implemented here so that the analysis does not
import jax (a test checks them against ``term_metrics``).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from p08_bfield_eval.sampling import KNOTS, REGIONS

TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion", "total")
FIELDS = ("density", "Te", "Ti", "vorticity")
DIFFUSION = "perpendicular_diffusion"
DEGENERATE_FRACTION = 1.0e-8
#: (term, comparison) pairs reported
COMPARISONS = tuple((t, "N-R") for t in ("poisson_bracket", "curvature")) + (
    (DIFFUSION, "N-O"), (DIFFUSION, "O-R"), (DIFFUSION, "N-R"), ("total", "N-R"))


def owner_weighted_l2(diff, volume) -> Optional[float]:
    diff = np.asarray(diff, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    total = float(np.sum(volume))
    if diff.shape[0] == 0 or total <= 0.0:
        return None
    return float(np.sqrt(np.sum(volume * np.sum(diff.reshape(diff.shape[0], -1) ** 2, axis=1)) / total))


def _finite(x) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else float(x)


def term_metrics(combined, reference, volume, *, fallback_l2=None, fallback_max=None,
                 degenerate: float = DEGENERATE_FRACTION) -> dict:
    """Volume-weighted L2 / max of ``combined - reference`` (absolute and relative) and the least-squares scale; the same
    formulas as ``p_shared.step3_gates.term_metrics``."""
    combined = np.asarray(combined, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    diff = combined - reference
    l2 = owner_weighted_l2(diff, volume)
    ref_l2 = owner_weighted_l2(reference, volume)
    max_abs, ref_max = float(np.max(np.abs(diff))), float(np.max(np.abs(reference)))
    is_degenerate = bool(fallback_l2 is not None and ref_l2 < degenerate * fallback_l2)
    scale_l2, scale_max = (fallback_l2, fallback_max) if is_degenerate else (ref_l2, ref_max)
    rr = float(np.sum(volume * reference ** 2))
    ratio = lambda a, b: None if a is None or b is None or b == 0.0 or not math.isfinite(a) else float(a / b)
    return {"l2": _finite(l2), "max_abs": _finite(max_abs), "ref_l2": _finite(ref_l2), "ref_max": _finite(ref_max),
            "rel_l2": ratio(l2, scale_l2), "rel_max": ratio(max_abs, scale_max),
            "fit_scale": None if is_degenerate else ratio(float(np.sum(volume * combined * reference)), rr),
            "degenerate_reference": is_degenerate,
            "finite": bool(np.all(np.isfinite(combined)) and np.all(np.isfinite(reference)))}


def observed_order(coarse, fine, n_coarse: int, n_fine: int) -> Optional[float]:
    """``log(e_coarse / e_fine) / log(n_fine / n_coarse)`` (``None`` unless both errors are positive and finite)."""
    if coarse is None or fine is None or not (coarse > 0.0 and fine > 0.0):
        return None
    return float(math.log(coarse / fine) / math.log(n_fine / n_coarse))


def subset_name(region: str, knot: str) -> str:
    return f"{region}|{knot}"


def subset_masks(region: np.ndarray, knot: np.ndarray) -> dict:
    """``{"region|knot": bool mask}`` for every region (and ``all``) times every knot class (and ``all``), including empty ones
    (the callers skip those)."""
    region, knot = np.asarray(region), np.asarray(knot)
    regions = {"all": np.ones(len(region), dtype=bool), **{n: region == i for i, n in enumerate(REGIONS)}}
    knots = {"all": np.ones(len(knot), dtype=bool), **{n: knot == i for i, n in enumerate(KNOTS)}}
    return {subset_name(rn, kn): rm & km for rn, rm in regions.items() for kn, km in knots.items()}


def comparison_arrays(arrays: dict, v: int, f: int, term: str, comparison: str):
    """``(combined, reference)`` of one comparison."""
    t = TERMS.index(term)
    n, r = arrays["N"][v, f, t], arrays["R"][v, f, t]
    if term != DIFFUSION or comparison == "N-O":
        return n, r
    rmid = arrays["Rmid"][v, f]
    return (r, rmid) if comparison == "O-R" else (n, rmid)


def compute_metrics(arrays: dict, variants, fields=FIELDS, keep: Optional[np.ndarray] = None) -> dict:
    """``{subset: {"n_owners": k, "variants": {variant: {field: {term: {comparison: metrics}}}}}}`` over the non-empty subsets
    of ``arrays`` (optionally restricted to the owners where ``keep`` is true)."""
    keep = np.ones(len(arrays["volume"]), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    out: dict = {}
    for name, mask in subset_masks(arrays["region"], arrays["knot"]).items():
        sel = mask & keep
        if not sel.any():
            continue
        vol = np.asarray(arrays["volume"])[sel]
        entry = {"n_owners": int(sel.sum()), "variants": {}}
        for v, variant in enumerate(variants):
            per_field = {}
            for f, field in enumerate(fields):
                refs = arrays["R"][v, f][:, sel]
                fallback_l2 = max(owner_weighted_l2(refs[t], vol) for t in range(len(TERMS)))
                fallback_max = max(float(np.max(np.abs(refs[t]))) for t in range(len(TERMS)))
                terms: dict = {}
                for term, comparison in COMPARISONS:
                    combined, reference = comparison_arrays(arrays, v, f, term, comparison)
                    terms.setdefault(term, {})[comparison] = term_metrics(
                        combined[sel], reference[sel], vol, fallback_l2=fallback_l2, fallback_max=fallback_max)
                per_field[field] = terms
            entry["variants"][variant] = per_field
        out[name] = entry
    return out


def restrict(arrays: dict, keep: np.ndarray) -> dict:
    """The per-owner arrays of ``arrays`` at the owners where ``keep`` is true."""
    keep = np.asarray(keep, dtype=bool)
    out = dict(arrays)
    for key in ("owners", "volume", "region", "knot", "phi"):
        if key in arrays:
            out[key] = np.asarray(arrays[key])[keep]
    out["N"], out["R"], out["Rmid"] = arrays["N"][..., keep], arrays["R"][..., keep], arrays["Rmid"][..., keep]
    return out


def sanity(arrays: dict, variants, fields=FIELDS) -> dict:
    """Finite / nonzero checks of the pooled arrays: ``finite`` (no NaN/Inf anywhere) and per (variant, field, term) the
    reference L2 (``zero_references`` lists the entries whose reference L2 is exactly zero)."""
    vol = np.asarray(arrays["volume"])
    finite = bool(all(np.all(np.isfinite(arrays[k])) for k in ("N", "R", "Rmid")))
    zero = []
    for v, variant in enumerate(variants):
        for f, field in enumerate(fields):
            for t, term in enumerate(TERMS):
                if owner_weighted_l2(arrays["R"][v, f, t], vol) == 0.0:
                    zero.append(f"{variant}/{field}/{term}")
    return {"finite": finite, "zero_references": zero}


def diffusion_convention_check(arrays: dict, variants, fields=FIELDS, tolerance: float = 0.25) -> dict:
    """Sign/coefficient check of the two continuum references of the diffusion term, pooled over the owners: per (variant, field)
    the relative L2 ``|O - R| / |O|`` and the least-squares scale ``<O, R> / <R, R>`` (1 for identical conventions, -1 for a
    sign error).  ``ok`` when every relative difference is below ``tolerance`` and every fit scale is within ``tolerance``
    of 1.  (O and R are two consistent discretizations of the same continuum term, so they differ at the discretization-error
    level, not at roundoff.)"""
    vol = np.asarray(arrays["volume"])
    table, ok = {}, True
    for v, variant in enumerate(variants):
        for f, field in enumerate(fields):
            o = arrays["R"][v, f, TERMS.index(DIFFUSION)]
            r = arrays["Rmid"][v, f]
            m = term_metrics(o, r, vol)
            rel = None if not m["ref_l2"] else owner_weighted_l2(o - r, vol) / owner_weighted_l2(o, vol)
            fit = m["fit_scale"]
            good = bool(rel is not None and fit is not None and rel < tolerance and abs(fit - 1.0) < tolerance)
            ok = ok and good
            table[f"{variant}/{field}"] = {"rel_l2_O_minus_R": rel, "fit_scale": fit, "ok": good}
    return {"tolerance": tolerance, "ok": ok, "entries": table}
