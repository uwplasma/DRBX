"""Local solver-study harness for the exported P07 sparse owner operators (P08 step 5).

Pure functions (numpy / scipy; JAX only through ``drbx.native.fci_perpendicular_p07_solve``) plus a thin IO / CLI layer:

* :func:`matrix_diagnostics`   -- size, asymmetry of ``M A``, positivity of the generalised problem ``sym(MA) x = lam M x``.
* :func:`consistency`          -- ``A phi_bar + B g`` against the frozen campaign action (D for Dirichlet, N for Neumann).
* :func:`dirichlet_solves`     -- direct LU and FGMRES (no / Jacobi preconditioner) solves of the Dirichlet problem.
* :func:`neumann_report`       -- right / left null vectors, bordered system, compatibility defect and the solves.
* :func:`summarize`            -- compact markdown tables.

CLI: ``python -m p08_step5_local.solver_study --export EXPORT --out OUT --grids 32 48`` (from ``DRBX/scripts``).
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                              # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from drbx.native.fci_perpendicular_p07_solve import (                                # noqa: E402
    P07SolveConfig, p07_linear_system, solve_p07_dirichlet_jit)
from drbx.native.fci_perpendicular_p07_sparse import boundary_source, load_p07_sparse  # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData          # noqa: E402

__all__ = ["FIELD_NAMES", "matrix_diagnostics", "consistency", "dirichlet_solves", "neumann_report", "summarize",
           "load_or_build_boundary_data", "run_grid", "main"]

FIELD_NAMES = ("field_b1", "field_e3", "field_e12", "heldout_field_b2", "constant")
WORKSPACE = _SCRIPTS.parents[1]                                   # .../HSX drbx
DEFAULT_FROZEN = WORKSPACE / "work/p07n_field_derived_274e93e9_20260927T054625Z_72cfa1"
DEFAULT_SIDECAR_REL = "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
DENSE_THRESHOLD = 400


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _names(n_fields: int, names: Sequence[str] | None = None) -> tuple[str, ...]:
    if names is not None:
        if len(names) != n_fields:
            raise ValueError(f"{len(names)} field names for {n_fields} fields")
        return tuple(names)
    return FIELD_NAMES if n_fields == len(FIELD_NAMES) else tuple(f"f{j}" for j in range(n_fields))


def _wl2(x: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """Volume-weighted L2, ``sqrt(sum V x^2 / sum V)``, optionally restricted to ``mask``."""
    if mask is not None:
        x, v = x[mask], v[mask]
    total = float(np.sum(v))
    return None if x.size == 0 or total <= 0.0 else float(math.sqrt(float(np.sum(v * x * x)) / total))


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b == 0.0 else float(a / b)


def _error_metrics(e: np.ndarray, ref: np.ndarray, v: np.ndarray, regions: Mapping[str, np.ndarray]) -> dict:
    """Volume-weighted L2 / max / relative-to-``ref`` L2 of ``e`` and the L2 per region."""
    l2 = _wl2(e, v)
    return {"l2": l2, "max": float(np.max(np.abs(e))), "rel_l2": _ratio(l2, _wl2(ref, v)),
            "regions": {name: _wl2(e, v, mask) for name, mask in regions.items()}}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _row_scale(matrix: sp.spmatrix) -> float:
    """Largest absolute row sum of ``matrix`` (the natural scale of ``A @ 1`` and of ``A.T @ l``)."""
    return float(np.max(np.asarray(abs(sp.csr_matrix(matrix)).sum(axis=1)).ravel()))


# ---------------------------------------------------------------------------
# 1. matrix diagnostics
# ---------------------------------------------------------------------------
def matrix_diagnostics(op, k: int = 6) -> dict:
    """Size, asymmetry of ``M A`` and positivity of the generalised problem ``sym(MA) x = lam M x`` (``M = diag(V)``).

    The ``k`` eigenvalues closest to 0 (Dirichlet: shift-invert at ``sigma = 0``; Neumann, where the kernel makes
    ``sym(MA)`` singular: at a small negative shift) and the largest one are computed with ``eigsh``; matrices of order
    ``<= DENSE_THRESHOLD`` use a dense ``eigh``. Failures are recorded in ``eig_errors``, never raised. Only the
    eigenvalues nearest zero are inspected, so a far-negative eigenvalue is not detected.
    """
    a = sp.csr_matrix(op.matrix)
    n = a.shape[0]
    vol = np.asarray(op.owner_volume, dtype=np.float64)
    ma = sp.diags(vol) @ a
    sym = ((ma + ma.T) * 0.5).tocsc()
    asym_num = float(spla.norm(ma - ma.T, "fro"))
    asym_den = float(spla.norm(ma + ma.T, "fro"))
    ones_res = a @ np.ones(n)
    scale = _row_scale(a)
    out = {"kind": op.kind, "n": int(n), "nnz": int(a.nnz), "nnz_per_row": float(a.nnz / n),
           "diag_min": float(a.diagonal().min()), "asymmetry": asym_num / asym_den if asym_den > 0 else 0.0,
           "A1_max_abs": float(np.max(np.abs(ones_res))), "row_abs_sum_max": scale,
           "A1_rel": float(np.max(np.abs(ones_res)) / scale) if scale > 0 else 0.0, "eig_errors": []}
    smallest, lam_max = None, None
    if k <= 0:
        out["eig_method"] = "skipped"
    elif n <= DENSE_THRESHOLD:
        try:
            w = sla.eigh(sym.toarray(), np.diag(vol), eigvals_only=True)
            smallest, lam_max, out["eig_method"] = np.sort(w)[:k], float(w[-1]), "dense"
        except Exception as exc:                                        # noqa: BLE001
            out["eig_errors"].append(f"dense eigh: {type(exc).__name__}: {exc}")
    else:
        out["eig_method"] = "eigsh"
        mmat = sp.diags(vol).tocsc()
        try:
            lam_max = float(spla.eigsh(sym, k=1, M=mmat, which="LA", tol=1e-8, return_eigenvectors=False)[0])
        except Exception as exc:                                        # noqa: BLE001
            out["eig_errors"].append(f"largest eigsh: {type(exc).__name__}: {exc}")
        sigma = 0.0 if op.kind == "dirichlet" else -1e-6 * (lam_max if lam_max else float(np.max(np.abs(sym.diagonal()))
                                                              / np.min(vol)))
        out["sigma"] = float(sigma)
        try:
            smallest = np.sort(spla.eigsh(sym, k=k, M=mmat, sigma=sigma, which="LM", tol=1e-10,
                                          return_eigenvectors=False))
        except Exception as exc:                                        # noqa: BLE001
            out["eig_errors"].append(f"smallest eigsh: {type(exc).__name__}: {exc}")
    out["lambda_max"] = lam_max
    out["lambda_smallest"] = None if smallest is None else [float(x) for x in smallest]
    lam_min = None if smallest is None else float(smallest[0])
    out["lambda_min"] = lam_min
    out["lambda_second"] = None if smallest is None or len(smallest) < 2 else float(smallest[1])
    out["ratio_min_max"] = _ratio(lam_min, lam_max)
    out["ratio_second_max"] = _ratio(out["lambda_second"], lam_max)
    out["positive_definite"] = None if lam_min is None or not lam_max else bool(lam_min > 1e-12 * lam_max)
    out["indefinite"] = None if lam_min is None or not lam_max else bool(lam_min < -1e-12 * lam_max)
    return out


# ---------------------------------------------------------------------------
# 2. consistency with the frozen campaign action
# ---------------------------------------------------------------------------
def consistency(op, phi_bar, bc: BoundaryData, frozen_action, field_names: Sequence[str] | None = None) -> dict:
    """``max |A phi_bar + B g - frozen|`` per field, absolute and relative to ``max |frozen|`` of that field."""
    phi_bar = np.asarray(phi_bar, dtype=np.float64)
    frozen = np.asarray(frozen_action, dtype=np.float64)
    diff = np.asarray(op.matrix @ phi_bar + boundary_source(op, bc)) - frozen
    names = _names(phi_bar.shape[1], field_names)
    gmax = float(np.max(np.abs(frozen)))
    fields = {}
    for j, name in enumerate(names):
        fmax = float(np.max(np.abs(frozen[:, j])))
        d = float(np.max(np.abs(diff[:, j])))
        fields[name] = {"max_abs": d, "frozen_max_abs": fmax, "rel": _ratio(d, fmax), "rel_global": _ratio(d, gmax)}
    return {"kind": op.kind, "fields": fields, "max_abs": float(np.max(np.abs(diff))),
            "rel_global": _ratio(float(np.max(np.abs(diff))), gmax)}


# ---------------------------------------------------------------------------
# 3. Dirichlet solves
# ---------------------------------------------------------------------------
def dirichlet_solves(op, phi_bar, bc: BoundaryData, rhs_sets: Mapping[str, np.ndarray],
                     regions: Mapping[str, np.ndarray], solvers: Sequence[str] = ("direct", "none", "jacobi"),
                     config: P07SolveConfig | None = None, field_names: Sequence[str] | None = None,
                     log: Callable[[str], None] | None = None) -> dict:
    """Solve ``A phi = rhs - B g`` for every rhs set and field; errors ``phi - phi_bar`` against the exact averages.

    ``solvers``: ``"direct"`` (SuperLU, factored once), ``"none"`` / ``"jacobi"`` (restarted FGMRES, preconditioner
    of that name; ``config`` carries rtol / restart / max_restarts). When ``"direct"`` is given and the rhs sets contain
    ``"O_q3"`` and ``"R"`` the decomposition ``e_R = e_O + A^-1 (R - O_q3)`` is reported (``A^-1`` with zero boundary term).
    """
    config = config or P07SolveConfig()
    phi_bar = np.asarray(phi_bar, dtype=np.float64)
    vol = np.asarray(op.owner_volume, dtype=np.float64)
    names = _names(phi_bar.shape[1], field_names)
    bsrc = np.asarray(boundary_source(op, bc), dtype=np.float64)
    say = log or (lambda m: None)
    krylov = [s for s in solvers if s != "direct"]
    use_direct = "direct" in solvers
    out: dict = {"solvers": list(solvers), "rtol": config.rtol, "restart": config.restart,
                 "max_restarts": config.max_restarts, "rhs": {}}

    lu = None
    if use_direct:
        t0 = time.perf_counter()
        lu = spla.splu(sp.csc_matrix(op.matrix))
        out["factor_seconds"] = time.perf_counter() - t0
        say(f"  direct LU factored in {out['factor_seconds']:.2f} s")
    system = None
    if krylov:
        system = p07_linear_system(op)
        first = next(iter(rhs_sets.values()))
        out["warmup_seconds"] = {}
        for s in krylov:
            cfg = dataclasses.replace(config, preconditioner=s)
            t0 = time.perf_counter()
            x, _ = solve_p07_dirichlet_jit(system, jnp.asarray(first[:, 0]), boundary_term=jnp.asarray(bsrc[:, 0]),
                                           config=cfg)
            np.asarray(x)
            out["warmup_seconds"][s] = time.perf_counter() - t0
            say(f"  FGMRES[{s}] warm-up (compile) {out['warmup_seconds'][s]:.2f} s")

    for rname, rhs in rhs_sets.items():
        rhs = np.asarray(rhs, dtype=np.float64)
        out["rhs"][rname] = {}
        for j, fname in enumerate(names):
            entry: dict = {}
            ref = phi_bar[:, j]
            phi_direct = None
            if use_direct:
                t0 = time.perf_counter()
                phi_direct = lu.solve(rhs[:, j] - bsrc[:, j])
                entry["direct"] = {**_error_metrics(phi_direct - ref, ref, vol, regions),
                                   "wall_s": time.perf_counter() - t0}
            for s in krylov:
                cfg = dataclasses.replace(config, preconditioner=s)
                t0 = time.perf_counter()
                x, info = solve_p07_dirichlet_jit(system, jnp.asarray(rhs[:, j]), boundary_term=jnp.asarray(bsrc[:, j]),
                                                  config=cfg)
                phi = np.asarray(x)
                dt = time.perf_counter() - t0
                rec = {**_error_metrics(phi - ref, ref, vol, regions), "iterations": int(info["iterations"]),
                       "converged": bool(info["converged"]), "wall_s": dt,
                       "residual_norm": float(info["residual_norm"]),
                       "relative_residual": float(info["relative_residual"])}
                if phi_direct is not None:
                    d = _wl2(phi - phi_direct, vol)
                    rec["diff_vs_direct_l2"] = d
                    rec["diff_vs_direct_rel"] = _ratio(d, _wl2(phi_direct, vol))
                entry[s] = rec
            out["rhs"][rname][fname] = entry
        say(f"  Dirichlet solves for rhs {rname!r} done")

    if use_direct and "O_q3" in rhs_sets and "R" in rhs_sets:
        o, r = np.asarray(rhs_sets["O_q3"], dtype=np.float64), np.asarray(rhs_sets["R"], dtype=np.float64)
        deco = {}
        for j, fname in enumerate(names):
            ref = phi_bar[:, j]
            shift = lu.solve(r[:, j] - o[:, j])                                  # A^-1 (R - O), zero boundary term
            e_o = lu.solve(o[:, j] - bsrc[:, j]) - ref
            e_r = lu.solve(r[:, j] - bsrc[:, j]) - ref
            deco[fname] = {"solve_error_from_O_minus_R": _error_metrics(shift, ref, vol, regions),
                           "identity_defect_l2": _wl2(e_r - e_o - shift, vol),
                           "identity_defect_max": float(np.max(np.abs(e_r - e_o - shift)))}
        out["decomposition"] = deco
    return out


# ---------------------------------------------------------------------------
# 4. Neumann report
# ---------------------------------------------------------------------------
def _inverse_operator(lu, size: int) -> spla.LinearOperator:
    return spla.LinearOperator(
        (size, size), dtype=np.float64, matvec=lambda x: lu.solve(np.asarray(x, dtype=np.float64).reshape(-1)),
        rmatvec=lambda x: lu.solve(np.asarray(x, dtype=np.float64).reshape(-1), trans="T"),
        matmat=lambda x: lu.solve(np.asarray(x, dtype=np.float64)),
        rmatmat=lambda x: lu.solve(np.asarray(x, dtype=np.float64), trans="T"))


def _ell_vs_w(ell: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        ell, w = ell[mask], w[mask]
    if ell.size == 0:
        return {"count": 0, "rel_l2": None, "max_rel": None, "sum_ell": 0.0, "sum_w": 0.0}
    return {"count": int(ell.size), "rel_l2": _ratio(float(np.linalg.norm(ell - w)), float(np.linalg.norm(w))),
            "max_rel": float(np.max(np.abs(ell / w - 1.0))), "sum_ell": float(ell.sum()), "sum_w": float(w.sum())}


def neumann_report(op, phi_bar, bc: BoundaryData, rhs_sets: Mapping[str, np.ndarray],
                   regions: Mapping[str, np.ndarray], field_names: Sequence[str] | None = None) -> dict:
    """Null-space structure and solves of the Neumann kind.

    Bordered system ``K = [[A, 1], [m^T, 0]]`` with ``m = V / sum V``; ``K [phi; lam] = [rhs - B g; m^T phi_bar]``
    fixes the gauge to the volume mean of ``phi_bar``. ``K^T [l; mu] = [0; 1]`` gives ``A^T l + mu m = 0`` and
    ``1^T l = 1``; multiplying the first by ``1`` and using ``A 1 = 0`` gives ``mu = 0``, hence ``l^T A = 0``,
    ``sum l = 1``. ``lam = l^T (rhs - B g) / (l^T 1)`` is the solvability defect (reported both ways).
    The array ``l`` is returned under the private key ``_arrays`` (dropped by ``_json_safe``).
    """
    a = sp.csr_matrix(op.matrix)
    n = a.shape[0]
    phi_bar = np.asarray(phi_bar, dtype=np.float64)
    vol = np.asarray(op.owner_volume, dtype=np.float64)
    names = _names(phi_bar.shape[1], field_names)
    w = vol / vol.sum()
    scale = _row_scale(a)
    a1 = np.asarray(a @ np.ones(n)).ravel()
    out: dict = {"kind": op.kind, "right_null": {"max_abs": float(np.max(np.abs(a1))), "row_abs_sum_max": scale,
                                                 "rel": float(np.max(np.abs(a1)) / scale) if scale > 0 else 0.0}}
    k = sp.bmat([[a, sp.csr_matrix(np.ones((n, 1)))], [sp.csr_matrix(w.reshape(1, -1)), None]], format="csc")
    t0 = time.perf_counter()
    try:
        lu = spla.splu(k)
    except Exception as exc:                                            # noqa: BLE001
        out["bordered"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "note": "bordered matrix singular: ker A != span{1} (or m^T 1 = 0)"}
        return out
    bordered = {"ok": True, "factor_seconds": time.perf_counter() - t0}
    try:
        n_k = float(spla.onenormest(k))
        n_ki = float(spla.onenormest(_inverse_operator(lu, n + 1)))
        bordered.update({"norm1_K": n_k, "norm1_Kinv": n_ki, "cond1_est": n_k * n_ki,
                         "likely_singular": bool(n_k * n_ki > 1e13)})
    except Exception as exc:                                            # noqa: BLE001
        bordered["cond_error"] = f"{type(exc).__name__}: {exc}"
    out["bordered"] = bordered

    rhs_vec = np.zeros(n + 1)
    rhs_vec[n] = 1.0
    sol = lu.solve(rhs_vec, trans="T")
    ell, mu = sol[:n], float(sol[n])
    res = np.asarray(a.T @ ell).ravel()
    out["left_null"] = {"sum": float(ell.sum()), "mu": mu,
                        "residual_max": float(np.max(np.abs(res))),
                        "residual_rel": float(np.max(np.abs(res)) / (np.max(np.abs(ell)) * scale)) if scale > 0 else 0.0,
                        "min": float(ell.min()), "n_negative": int(np.sum(ell < 0.0)),
                        "vs_w": {"global": _ell_vs_w(ell, w), "regions": {nm: _ell_vs_w(ell, w, m)
                                                                          for nm, m in regions.items()}}}
    out["_arrays"] = {"ell": ell, "w": w}

    bsrc = np.asarray(boundary_source(op, bc), dtype=np.float64)
    gauge = w @ phi_bar
    out["solves"] = {}
    for rname, rhs in rhs_sets.items():
        eff = np.asarray(rhs, dtype=np.float64) - bsrc
        full = lu.solve(np.vstack([eff, gauge[None, :]]))
        phi, lam = full[:n], full[n]
        defect = (ell @ eff) / ell.sum()
        out["solves"][rname] = {}
        for j, fname in enumerate(names):
            ref = phi_bar[:, j]
            rms_rhs = _wl2(np.asarray(rhs)[:, j], vol)
            rms_eff = _wl2(eff[:, j], vol)
            resid = a @ phi[:, j] + lam[j] - eff[:, j]
            out["solves"][rname][fname] = {
                "lambda": float(lam[j]), "lambda_rel_rhs": _ratio(abs(float(lam[j])), rms_rhs),
                "lambda_rel_effective": _ratio(abs(float(lam[j])), rms_eff), "rhs_rms": rms_rhs,
                "defect": float(defect[j]), "lambda_minus_defect": float(lam[j] - defect[j]),
                "residual_max": float(np.max(np.abs(resid))), **_error_metrics(phi[:, j] - ref, ref, vol, regions)}
    return out


# ---------------------------------------------------------------------------
# 5. markdown summary
# ---------------------------------------------------------------------------
def _f(x, spec: str = ".3e") -> str:
    if isinstance(x, np.generic):
        x = x.item()
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return format(x, spec)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def summarize(results: Mapping) -> str:
    """Markdown report of ``{grid: grid_results}`` (grid: ``32`` or ``"N32"``); missing / errored parts are noted."""
    lines = ["# P07 solver study", ""]
    for key, res in results.items():
        label = f"N{key}" if isinstance(key, (int, np.integer)) else str(key)
        lines += [f"## {label} (n_owners = {res.get('n_owners', '?')})", ""]
        diag = res.get("diagnostics") or {}
        lines += ["### Matrix diagnostics", ""]
        rows = []
        for kind, d in diag.items():
            rows.append([kind, _f(d.get("n")), _f(d.get("nnz")), _f(d.get("nnz_per_row"), ".2f"),
                         _f(d.get("asymmetry")), _f(d.get("lambda_min")), _f(d.get("lambda_second")),
                         _f(d.get("lambda_max")), _f(d.get("ratio_min_max")), _f(d.get("positive_definite")),
                         _f(d.get("A1_rel")), "; ".join(d.get("eig_errors") or []) or "-"])
        lines += _table(["kind", "n", "nnz", "nnz/row", "asymmetry(MA)", "lam_min", "lam_2", "lam_max",
                         "lam_min/lam_max", "pos. def.", "max abs(A1)/rowscale", "eig errors"], rows)
        cons = res.get("consistency") or {}
        lines += ["### Consistency with the frozen action (max abs, relative to max abs frozen)", ""]
        rows = [[kind, fname, _f(v["max_abs"]), _f(v["rel"])] for kind, c in cons.items()
                for fname, v in (c.get("fields") or {}).items()]
        lines += _table(["kind", "field", "max abs diff", "relative"], rows)
        dsol = res.get("dirichlet")
        lines += ["### Dirichlet solves (error phi - phi_bar, volume-weighted L2)", ""]
        if not dsol or "error" in dsol:
            lines += [f"not available: {(dsol or {}).get('error', 'not run')}", ""]
        else:
            krylov = [s for s in dsol["solvers"] if s != "direct"]
            header = ["rhs", "field"] + (["direct L2", "direct rel L2", "direct max"] if "direct" in dsol["solvers"]
                                         else [])
            for s in krylov:
                header += [f"{s}: L2", f"{s}: iters", f"{s}: conv", f"{s}: s", f"{s}: rel diff vs direct"]
            rows = []
            for rname, fields in dsol["rhs"].items():
                for fname, ent in fields.items():
                    row = [rname, fname]
                    if "direct" in ent:
                        row += [_f(ent["direct"]["l2"]), _f(ent["direct"]["rel_l2"]), _f(ent["direct"]["max"])]
                    for s in krylov:
                        e = ent[s]
                        row += [_f(e["l2"]), _f(e["iterations"]), _f(e["converged"]), _f(e["wall_s"], ".2f"),
                                _f(e.get("diff_vs_direct_rel"))]
                    rows.append(row)
            lines += _table(header, rows)
            if "decomposition" in dsol:
                lines += ["Reference mismatch: L2 of direct(R) - direct(O_q3) = A^-1 (R - O_q3)", ""]
                rows = [[fname, _f(d["solve_error_from_O_minus_R"]["l2"]), _f(d["solve_error_from_O_minus_R"]["rel_l2"]),
                         _f(d["solve_error_from_O_minus_R"]["max"]), _f(d["identity_defect_l2"])]
                        for fname, d in dsol["decomposition"].items()]
                lines += _table(["field", "L2", "rel L2", "max", "identity defect (L2)"], rows)
        nr = res.get("neumann")
        lines += ["### Neumann: null vectors, bordered system, compatibility", ""]
        if not nr or "error" in nr:
            lines += [f"not available: {(nr or {}).get('error', 'not run')}", ""]
        else:
            rn, bd, ln = nr["right_null"], nr.get("bordered", {}), nr.get("left_null")
            lines += [f"- right null: max|A 1| = {_f(rn['max_abs'])} (relative to max abs row sum: {_f(rn['rel'])})"]
            if not bd.get("ok"):
                lines += [f"- bordered matrix LU FAILED: {bd.get('error')} ({bd.get('note')})", ""]
            else:
                lines += [f"- bordered LU ok ({_f(bd.get('factor_seconds'), '.2f')} s); 1-norm condition estimate "
                          f"{_f(bd.get('cond1_est'))}; likely singular: {_f(bd.get('likely_singular'))}"]
                lines += [f"- left null l: sum = {_f(ln['sum'], '.12f')}, mu = {_f(ln['mu'])}, "
                          f"max|A^T l| (relative) = {_f(ln['residual_rel'])}, negative entries = {ln['n_negative']}", ""]
                rows = [["global", *_vs_w_row(ln["vs_w"]["global"])]]
                rows += [[nm, *_vs_w_row(v)] for nm, v in ln["vs_w"]["regions"].items() if v["count"]]
                lines += _table(["region (l vs w = V/sum V)", "count", "rel L2(l-w)", "max abs(l/w-1)", "sum l", "sum w"],
                                rows)
                rows = []
                for rname, fields in (nr.get("solves") or {}).items():
                    for fname, v in fields.items():
                        rows.append([rname, fname, _f(v["lambda"]), _f(v["lambda_rel_rhs"]), _f(v["defect"]),
                                     _f(v["lambda_minus_defect"]), _f(v["l2"]), _f(v["rel_l2"]), _f(v["max"])])
                lines += _table(["rhs", "field", "lambda", "lambda/rms(rhs)", "l^T(rhs-Bg)/sum l", "lambda - defect",
                                 "phi err L2", "phi err rel L2", "phi err max"], rows)
        secs = res.get("seconds")
        if secs:
            lines += ["Timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in secs.items()), ""]
    return "\n".join(lines) + "\n"


def _vs_w_row(v: Mapping) -> list[str]:
    return [_f(v["count"]), _f(v["rel_l2"]), _f(v["max_rel"]), _f(v["sum_ell"], ".6f"), _f(v["sum_w"], ".6f")]


# ---------------------------------------------------------------------------
# IO layer
# ---------------------------------------------------------------------------
def _points_sha(points: np.ndarray) -> str:
    p = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    return hashlib.sha256(str(p.shape).encode() + p.tobytes()).hexdigest()


def _make_adapter(n: int, phi_bar: np.ndarray, input_root: Path, sidecar: Path):
    """The campaign field adapter (builds the real environment, ~20 s); tests monkeypatch this function."""
    from p_shared import campaign_fields as cf                                  # noqa: PLC0415
    from p_shared.replay_support import build_environment                       # noqa: PLC0415
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar), curvature="autodiff",
                            face_quadrature="q2", inner_support="fixed_radius")
    return cf.P07NAdapter(env.ref, env.t.g.eta_period, phi_bar)


def load_or_build_boundary_data(path: Path, dpts: np.ndarray, npts: np.ndarray, make_adapter: Callable[[], object],
                                n_fields: int, batch: int = 8192,
                                log: Callable[[str], None] | None = None) -> BoundaryData:
    """Wall data at the export's point tables; cached in ``path`` (reused when the point tables' sha256 match)."""
    say = log or (lambda m: None)
    sd, sn = _points_sha(dpts), _points_sha(npts)
    path = Path(path)
    if path.exists():
        with np.load(path) as z:
            if (str(z["dirichlet_points_sha256"]) == sd and str(z["neumann_points_sha256"]) == sn
                    and z["dirichlet_value"].shape == (len(dpts), n_fields)
                    and z["neumann_normal"].shape == (len(npts), n_fields)):
                say(f"  boundary data reused from {path}")
                return BoundaryData(np.asarray(z["dirichlet_value"]), np.asarray(z["dirichlet_tangential"]),
                                    np.asarray(z["neumann_normal"]))
        say(f"  cached boundary data {path} does not match the point tables; recomputing")
    adapter = make_adapter()
    val = np.empty((len(dpts), n_fields))
    tan = np.empty((len(dpts), 2, n_fields))
    nrm = np.empty((len(npts), n_fields))
    for lo in range(0, len(dpts), batch):
        v, g = adapter.dirichlet(dpts[lo:lo + batch])
        val[lo:lo + batch], tan[lo:lo + batch] = np.asarray(v), np.asarray(g)[:, 1:, :]
    for lo in range(0, len(npts), batch):
        nrm[lo:lo + batch] = np.asarray(adapter.normal(npts[lo:lo + batch]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, dirichlet_value=val, dirichlet_tangential=tan, neumann_normal=nrm,
                        dirichlet_points_sha256=sd, neumann_points_sha256=sn)
    os.replace(tmp, path)
    return BoundaryData(val, tan, nrm)


def run_grid(n: int, export: Path, out: Path, frozen: Path, *, solvers: Sequence[str], skip_neumann: bool,
             config: P07SolveConfig, input_root: Path, sidecar: Path, batch: int = 8192, k_eigs: int = 6,
             log: Callable[[str], None] | None = None) -> dict:
    """Run every study on one grid; returns the (JSON-safe) results and writes ``OUT/N{n}/``."""
    say = log or (lambda m: None)
    t_all = time.perf_counter()
    secs: dict = {}

    def lap(key: str, t0: float) -> None:
        secs[key] = time.perf_counter() - t0

    gdir = Path(export) / f"N{n}"
    odir = Path(out) / f"N{n}"
    odir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    op_d = load_p07_sparse(gdir / "p07_dirichlet.npz")
    op_n = None if skip_neumann else load_p07_sparse(gdir / "p07_neumann.npz")
    with np.load(gdir / "owner_map.npz") as z:
        owner_volume = np.asarray(z["owner_volume"], dtype=np.float64)
    with np.load(Path(frozen) / f"N{n}.global.npz") as z:
        fg = {k: np.asarray(z[k]) for k in z.files}
    with np.load(Path(frozen) / f"N{n}.owner_values.npz") as z:
        phi_bar = np.asarray(z["values"], dtype=np.float64)
    n_o = len(fg["volume"])
    for what, vol in (("owner_map", owner_volume), ("dirichlet operator", op_d.owner_volume),
                      *((("neumann operator", op_n.owner_volume),) if op_n is not None else ())):
        if len(vol) != n_o:
            raise ValueError(f"N{n}: {what} has {len(vol)} owners, the frozen campaign {n_o}")
        if not np.allclose(vol, fg["volume"], rtol=1e-12, atol=0.0):
            raise ValueError(f"N{n}: {what} owner_volume differs from the frozen volume "
                             f"(max rel {np.max(np.abs(vol / fg['volume'] - 1)):.3e})")
    if phi_bar.shape[0] != n_o:
        raise ValueError(f"N{n}: owner_values has {phi_bar.shape[0]} rows, expected {n_o}")
    if op_n is not None and not (np.array_equal(op_n.dirichlet_points, op_d.dirichlet_points)
                                 and np.array_equal(op_n.neumann_points, op_d.neumann_points)):
        raise ValueError(f"N{n}: the Dirichlet and Neumann exports have different point tables")
    regions = {k[len("region_"):]: fg[k].astype(bool) for k in fg if k.startswith("region_")}
    if "boundary" in regions and "physical_wall" in regions:
        regions["interior"] = ~(regions["boundary"] | regions["physical_wall"])
    n_fields = phi_bar.shape[1]
    lap("load", t0)
    say(f"N{n}: loaded n_owners={n_o}, {len(regions)} regions, {n_fields} fields in {secs['load']:.1f} s")

    t0 = time.perf_counter()
    bc = load_or_build_boundary_data(
        odir / "boundary_data.npz", op_d.dirichlet_points, op_d.neumann_points,
        lambda: _make_adapter(n, phi_bar, input_root, sidecar), n_fields, batch=batch, log=say)
    lap("boundary_data", t0)
    say(f"N{n}: boundary data ready in {secs['boundary_data']:.1f} s")

    res: dict = {"grid": n, "n_owners": n_o, "settings": {
        "solvers": list(solvers), "rtol": config.rtol, "restart": config.restart,
        "max_restarts": config.max_restarts, "skip_neumann": skip_neumann, "frozen": str(frozen)}}
    ops = {"dirichlet": op_d, **({"neumann": op_n} if op_n is not None else {})}
    frozen_key = {"dirichlet": "D", "neumann": "N"}

    def guarded(key: str, fn: Callable[[], dict]) -> dict:
        t = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:                                        # noqa: BLE001
            traceback.print_exc()
            r = {"error": f"{type(exc).__name__}: {exc}"}
        lap(key, t)
        say(f"N{n}: {key} done in {secs[key]:.1f} s")
        return r

    res["diagnostics"] = {kind: guarded(f"diagnostics_{kind}", lambda op=op: matrix_diagnostics(op, k=k_eigs))
                          for kind, op in ops.items()}
    res["consistency"] = {kind: guarded(f"consistency_{kind}",
                                        lambda op=op, kind=kind: consistency(op, phi_bar, bc, fg[frozen_key[kind]]))
                          for kind, op in ops.items()}
    res["dirichlet"] = guarded("dirichlet_solves", lambda: dirichlet_solves(
        op_d, phi_bar, bc, {"D": fg["D"], "O_q3": fg["O_q3"], "R": fg["R"]}, regions, solvers=solvers,
        config=config, log=say))
    if op_n is not None:
        res["neumann"] = guarded("neumann", lambda: neumann_report(
            op_n, phi_bar, bc, {"N": fg["N"], "O_q3": fg["O_q3"], "R": fg["R"]}, regions))
        arrays = res["neumann"].get("_arrays")
        if arrays:
            np.savez_compressed(odir / "neumann_left_null.npz", **arrays)
    lap("total", t_all)
    res["seconds"] = dict(secs)
    safe = _json_safe(res)
    (odir / "results.json").write_text(json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return safe


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True, type=Path, help="EXPORT directory holding N{n}/ subfolders")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--grids", nargs="+", type=int, default=[32, 48])
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN, help="frozen P07N campaign directory")
    ap.add_argument("--skip-neumann", action="store_true")
    ap.add_argument("--solvers", nargs="+", default=["direct", "none", "jacobi"], choices=["direct", "none", "jacobi"])
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--restart", type=int, default=50)
    ap.add_argument("--max-restarts", type=int, default=20)
    ap.add_argument("--k-eigs", type=int, default=6, help="eigenvalues nearest 0 per kind; 0 skips the eigen stage")
    ap.add_argument("--batch", type=int, default=8192, help="point batch of the wall-data evaluation")
    ap.add_argument("--input-root", type=Path, default=WORKSPACE)
    ap.add_argument("--sidecar", type=Path, default=WORKSPACE / DEFAULT_SIDECAR_REL)
    args = ap.parse_args(argv)

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:8.1f}s] {msg}", flush=True)

    config = P07SolveConfig(rtol=args.rtol, restart=args.restart, max_restarts=args.max_restarts)
    results: dict = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for n in args.grids:
        log(f"=== grid N{n} ===")
        results[n] = run_grid(n, args.export, args.out, args.frozen, solvers=args.solvers,
                              skip_neumann=args.skip_neumann, config=config, input_root=args.input_root,
                              sidecar=args.sidecar, batch=args.batch, k_eigs=args.k_eigs, log=log)
        (args.out / f"report_{'_'.join(f'N{g}' for g in args.grids)}.md").write_text(summarize(results))
        log(f"N{n}: results.json and report.md written")
    log("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
