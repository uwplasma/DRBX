"""Build / storage scaling of ``core_schur`` and its z-free variant: measured N32/48/64 -> N96/128/256 (no JAX arrays; the layout
structure ``E, P, Nc, m, N`` of every ``n`` comes from the family-A layout, which is cheap up to n = 256).

Stored arrays (``E = n`` planes; ``rpb`` rings per super-ring, ``S = ceil(m / rpb)``, ``B = rpb N``, ``w`` block bandwidth)::

    lo (S, E, B, w, B)   dinv (S, E, B, B)   z (E, S B, Nc)   sinv (E, Nc, Nc)        z-free: z -> mrc (E, nr, Nc)

Two extrapolations: ``shape_model`` (exact layout structure at the target ``n``, ``rpb`` and ``w`` taken from the largest measured
``n``, ``nr / N`` likewise) and ``power_fit`` (log-log least squares of the measured totals, per component). The shape model is
first checked against the measured bytes at the measured ``n`` (with their measured ``rpb``, ``w``).
"""
from __future__ import annotations

import math

import numpy as np

import pgb_cases as K

ITEMSIZE = {"f32": 4, "f64": 8}
COMPONENTS = ("lo", "dinv", "z", "sinv")


def core_schur_components(st: dict, rpb: int, w: int, dt: str) -> dict:
    """Stored bytes per array of ``core_schur`` for the structure ``st = {E, Nc, m, N}``."""
    isz = ITEMSIZE[dt]
    S = -(-st["m"] // rpb)
    B = rpb * st["N"]
    E, Nc = st["E"], st["Nc"]
    return {"lo": S * E * B * w * B * isz, "dinv": S * E * B * B * isz, "z": E * S * B * Nc * isz, "sinv": E * Nc * Nc * isz}


def zfree_bytes(st: dict, rpb: int, w: int, dt: str, nr: int) -> int:
    c = core_schur_components(st, rpb, w, dt)
    return int(c["lo"] + c["dinv"] + c["sinv"] + st["E"] * nr * st["Nc"] * ITEMSIZE[dt])


def power_fit(ns, ys) -> dict | None:
    """``y ~ a n^p`` by least squares in log-log (``None`` with fewer than 2 points)."""
    ns, ys = np.asarray(ns, float), np.asarray(ys, float)
    if len(ns) < 2 or np.any(ys <= 0):
        return None
    p, la = np.polyfit(np.log(ns), np.log(ys), 1)
    res = np.log(ys) - (p * np.log(ns) + la)
    return {"exponent": float(p), "prefactor": float(math.exp(la)), "max_log_residual": float(np.abs(res).max()), "points": len(ns)}


def evaluate_fit(fit: dict | None, n: float) -> float | None:
    return None if fit is None else float(fit["prefactor"] * n ** fit["exponent"])


def _gb(b):
    return None if b is None else b / 1e9


def compute_scaling(builds: dict, zfree: dict, layout_structure, extrapolate_n=K.FIT_EXTRAPOLATE_N,
                    user_estimate=K.USER_ESTIMATE_F32_GB) -> dict:
    """``builds[dt][n]`` = results of the ``build`` case (``structure``, ``components_bytes``, ``host_build_s``,
    ``peak_rss_after_gib``); ``zfree[dt][n]`` = results of the ``core_schur_zfree`` precond case (``storage_bytes``, ``nr``).
    ``layout_structure(n) -> {E, Nc, m, N}``."""
    out = {"extrapolate_n": list(extrapolate_n), "measured_n": {dt: sorted(builds.get(dt, {})) for dt in K.DTYPES}}
    shape, fits, check = {}, {}, {}
    for dt in K.DTYPES:
        meas = {n: r for n, r in sorted(builds.get(dt, {}).items())}
        if not meas:
            continue
        n_last = max(meas)
        rpb, w = meas[n_last]["structure"]["rpb"], meas[n_last]["structure"]["w"]
        zl = zfree.get(dt, {})
        nr_ratio = (zl[max(zl)]["nr"] / zl[max(zl)]["structure"]["N"]) if zl else None
        # model check at the measured n (measured rpb, w)
        check[dt] = {}
        for n, r in meas.items():
            s = r["structure"]
            comp = core_schur_components(s, s["rpb"], s["w"], dt)
            tot = sum(comp.values())
            check[dt][str(n)] = {"model_total_bytes": int(tot), "measured_total_bytes": int(r["storage_bytes"]),
                                 "rel_diff": float(abs(tot - r["storage_bytes"]) / r["storage_bytes"]),
                                 "model_components": {k: int(v) for k, v in comp.items()},
                                 "measured_components": r["components_bytes"]}
        table = {}
        for n in sorted(set(meas) | set(extrapolate_n) | {32, 48, 64}):
            st = layout_structure(n)
            comp = core_schur_components(st, rpb, w, dt)
            tot = sum(comp.values())
            row = {"structure": st, "components_bytes": {k: int(v) for k, v in comp.items()}, "total_bytes": int(tot),
                   "total_gb": _gb(tot), "per_plane_mb": tot / st["E"] / 1e6, "z_fraction": comp["z"] / tot,
                   "measured": n in meas}
            if nr_ratio is not None:
                nr = int(round(nr_ratio * st["N"]))
                zb = zfree_bytes(st, rpb, w, dt, nr)
                row["zfree_total_bytes"], row["zfree_total_gb"], row["zfree_nr"] = int(zb), _gb(zb), nr
                row["zfree_over_core_schur"] = zb / tot
            table[str(n)] = row
        shape[dt] = {"assumption": {"rpb": rpb, "w": w, "from_n": n_last, "nr_over_N": nr_ratio}, "table": table}
        # power-law fits of the measured totals and components
        ns = sorted(meas)
        f = {"total": power_fit(ns, [meas[n]["storage_bytes"] for n in ns])}
        for c in COMPONENTS:
            f[c] = power_fit(ns, [meas[n]["components_bytes"][c] for n in ns])
        f["host_build_s"] = power_fit(ns, [meas[n]["host_build_s"] for n in ns])
        f["peak_rss_gib"] = power_fit(ns, [meas[n]["peak_rss_after_gib"] for n in ns])
        zz = zfree.get(dt, {})
        f["zfree_total"] = power_fit(sorted(zz), [zz[n]["storage_bytes"] for n in sorted(zz)]) if zz else None
        ext = {}
        for n in extrapolate_n:
            ext[str(n)] = {"total_gb": _gb(evaluate_fit(f["total"], n)), "host_build_s": evaluate_fit(f["host_build_s"], n),
                           "peak_rss_gib": evaluate_fit(f["peak_rss_gib"], n),
                           "zfree_total_gb": _gb(evaluate_fit(f["zfree_total"], n)),
                           "components_gb": {c: _gb(evaluate_fit(f[c], n)) for c in COMPONENTS}}
        fits[dt] = {"fits": f, "extrapolated": ext}
    out["shape_model"], out["power_fit"], out["shape_model_check"] = shape, fits, check
    if "f32" in shape:
        t = shape["f32"]["table"]
        comp = {}
        for n_s, est in user_estimate.items():
            row = t.get(n_s)
            fit_gb = fits["f32"]["extrapolated"].get(n_s, {}).get("total_gb") if "f32" in fits else None
            comp[n_s] = {"user_estimate_gb": est, "shape_model_gb": row["total_gb"] if row else None,
                         "shape_model_over_estimate": (row["total_gb"] / est) if row else None, "power_fit_gb": fit_gb}
        n32 = t.get("32")
        out["user_estimate_check"] = {"per_plane_mb_N32_f32": n32["per_plane_mb"] if n32 else None,
                                      "user_per_plane_mb_N32": 0.9, "totals": comp}
    return out
