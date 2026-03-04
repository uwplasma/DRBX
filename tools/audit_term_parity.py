from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict
from pathlib import Path
import tomllib

import numpy as np
import jax.numpy as jnp

E_CHARGE = 1.602176634e-19
M_PROTON = 1.67262192369e-27


def _read_scalar(ds, names: tuple[str, ...], default: float | None = None) -> float:
    for name in names:
        if name in ds.variables:
            arr = np.asarray(ds.variables[name][:], dtype=np.float64)
            return float(arr.reshape(-1)[0])
    if default is None:
        raise KeyError(f"Missing any of {names}")
    return float(default)


def _read_var_interior(ds, name: str, mxg: int, myg: int, mxsub: int, mysub: int) -> np.ndarray:
    arr = np.asarray(ds.variables[name][:], dtype=np.float64)
    if arr.ndim == 4:
        return arr[:, mxg : mxg + mxsub, myg : myg + mysub, :]
    if arr.ndim == 3:
        return arr[:, mxg : mxg + mxsub, myg : myg + mysub]
    raise ValueError(f"Unsupported rank for {name}: {arr.ndim}")


def _load_hermes_fields(
    data_dir: Path, names: tuple[str, ...], pattern: str
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required to read Hermes BOUT dumps.") from exc

    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Hermes dump files found in {data_dir} (pattern={pattern}).")

    with Dataset(str(files[0])) as ds0:
        times = np.asarray(ds0.variables["t"][:], dtype=np.float64)
        mxg = int(_read_scalar(ds0, ("MXG",), 2))
        myg = int(_read_scalar(ds0, ("MYG",), 2))
        mxsub = int(_read_scalar(ds0, ("MXSUB",), ds0.variables["Ne"].shape[1] - 2 * mxg))
        mysub = int(_read_scalar(ds0, ("MYSUB",), ds0.variables["Ne"].shape[2] - 2 * myg))
        nxpe = int(_read_scalar(ds0, ("NXPE",), 1))
        nype = int(_read_scalar(ds0, ("NYPE",), max(1, len(files) // max(nxpe, 1))))

    nt = int(times.size)
    nx = int(nxpe * mxsub)
    ny = int(nype * mysub)
    fields: dict[str, np.ndarray] = {}

    with Dataset(str(files[0])) as ds:
        has_te = "Te" in ds.variables
        has_pe = "Pe" in ds.variables
        for name in names:
            if name == "Te" and (not has_te) and has_pe:
                continue
            if name.startswith("ddt(") and name not in ds.variables:
                continue
            if name == "Ve" and name not in ds.variables and "NVe" in ds.variables:
                continue
            if (
                name == "Vd+"
                and name not in ds.variables
                and "NVd+" in ds.variables
                and "Nd+" in ds.variables
            ):
                continue
            if name not in ds.variables:
                if name.startswith("term_"):
                    continue
                raise KeyError(f"Hermes variable '{name}' not found in dump.")
            shape = _read_var_interior(ds, name, mxg, myg, mxsub, mysub).shape
            fields[name] = np.zeros((nt, nx, ny, *shape[3:]), dtype=np.float64)
        if not has_te and has_pe:
            shape = _read_var_interior(ds, "Pe", mxg, myg, mxsub, mysub).shape
            fields["Pe"] = np.zeros((nt, nx, ny, *shape[3:]), dtype=np.float64)

    for local_rank, fp in enumerate(files):
        with Dataset(str(fp)) as ds:
            pe_x = int(_read_scalar(ds, ("PE_XIND",), local_rank % max(nxpe, 1)))
            pe_y = int(_read_scalar(ds, ("PE_YIND",), local_rank // max(nxpe, 1)))
            x0 = pe_x * mxsub
            y0 = pe_y * mysub
            x1 = x0 + mxsub
            y1 = y0 + mysub
            for name in names:
                if name == "Te" and name not in ds.variables and "Pe" in ds.variables:
                    continue
                if name.startswith("ddt(") and name not in ds.variables:
                    continue
                if name == "Ve" and name not in ds.variables and "NVe" in ds.variables:
                    continue
                if (
                    name == "Vd+"
                    and name not in ds.variables
                    and "NVd+" in ds.variables
                    and "Nd+" in ds.variables
                ):
                    continue
                if name not in ds.variables:
                    if name.startswith("term_"):
                        continue
                    raise KeyError(f"Hermes variable '{name}' not found in dump.")
                sub = _read_var_interior(ds, name, mxg, myg, mxsub, mysub)
                fields[name][:, x0:x1, y0:y1, ...] = sub
            if "Te" not in ds.variables and "Pe" in ds.variables:
                fields["Pe"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, "Pe", mxg, myg, mxsub, mysub
                )

    if "Te" not in fields and "Pe" in fields and "Ne" in fields:
        fields["Te"] = fields["Pe"] / np.maximum(fields["Ne"], 1e-12)
        fields.pop("Pe", None)
    meta = {
        "nx": float(nx),
        "ny": float(ny),
        "nt": float(nt),
        "mxg": float(mxg),
        "myg": float(myg),
        "mxsub": float(mxsub),
        "mysub": float(mysub),
        "nxpe": float(nxpe),
        "nype": float(nype),
    }
    return times, fields, meta


def _parse_bout_input(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.split("!")[0].strip()
        if not raw or raw.startswith("&") or raw == "/":
            continue
        if "=" not in raw:
            continue
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip().strip(",")
        try:
            out[key] = float(val.replace("D", "E"))
        except ValueError:
            continue
    return out


def _eval_bout_expr(raw: str) -> float | None:
    import ast
    import operator as op

    raw = raw.replace("D", "E").strip()
    if not raw:
        return None

    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(raw, mode="eval")
        return float(_eval(tree.body))
    except Exception:
        return None


def _parse_bout_sections(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    sections: dict[str, dict[str, float]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.split("!")[0].split("#")[0].strip()
        if not raw:
            continue
        if raw.startswith("[") and raw.endswith("]"):
            current = raw[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if "=" not in raw:
            continue
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip().rstrip(",")
        parsed = _eval_bout_expr(val)
        if parsed is None:
            continue
        sections.setdefault(current, {})[key] = parsed
    return sections


def _find_repo_root(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return None


def _resolve_coeff_path(cfg: dict[str, object], cfg_path: Path) -> dict[str, object]:
    geom = dict(cfg.get("geometry", {}))
    coeff_path = geom.get("coeff_path", None)
    if isinstance(coeff_path, str) and not Path(coeff_path).is_absolute():
        candidate = (cfg_path.parent / coeff_path).resolve()
        if not candidate.exists():
            repo_root = _find_repo_root(cfg_path.parent)
            if repo_root is not None:
                candidate = (repo_root / coeff_path).resolve()
        geom["coeff_path"] = str(candidate)
    cfg["geometry"] = geom
    return cfg


def _hermes_time_unit(sections: dict[str, dict[str, float]]) -> float | None:
    hermes = sections.get("hermes", {})
    Bnorm = hermes.get("Bnorm", None)
    if Bnorm is None:
        return None
    ion = sections.get("d+", {})
    AA = ion.get("AA", 1.0)
    charge = ion.get("charge", 1.0)
    m_i = float(AA) * M_PROTON
    omega_ci = abs(float(charge)) * E_CHARGE * float(Bnorm) / max(m_i, 1e-30)
    if omega_ci <= 0.0:
        return None
    return 1.0 / omega_ci


def _hermes_time_unit_from_dumps(data_dir: Path, pattern: str) -> float | None:
    try:
        from netCDF4 import Dataset
    except Exception:
        return None

    files = sorted(data_dir.glob(pattern))
    if not files:
        return None
    try:
        with Dataset(str(files[0])) as ds0:
            if "Omega_ci" in ds0.variables:
                omega_ci = float(
                    np.asarray(ds0.variables["Omega_ci"][:], dtype=np.float64).reshape(-1)[0]
                )
                if omega_ci > 0.0:
                    return 1.0 / omega_ci
    except Exception:
        return None
    return None


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
    }


def _jax_geom_stats(system) -> dict[str, dict[str, float]]:
    geom = system.geom
    stats: dict[str, dict[str, float]] = {}
    for name in ("curv_x", "curv_y", "dpar_factor", "B", "bxcv", "gxx", "gxy", "gyy"):
        if hasattr(geom, name):
            arr = getattr(geom, name)
            if arr is not None:
                stats[name] = _stats(np.asarray(arr))
    grid = getattr(geom, "grid", None)
    if grid is not None:
        for name in ("dx", "dy", "dz"):
            if hasattr(grid, name):
                stats[f"grid_{name}"] = {"mean": float(getattr(grid, name))}
        perp = getattr(grid, "perp", None)
        if perp is not None:
            stats["grid_perp_dx"] = {"mean": float(perp.dx)}
            stats["grid_perp_dy"] = {"mean": float(perp.dy)}
    return stats


def _hermes_geom_stats(ds) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    candidates = (
        "dx",
        "dy",
        "g11",
        "g22",
        "g33",
        "g_11",
        "g_22",
        "g_33",
        "g12",
        "g_12",
        "g23",
        "g_23",
        "g13",
        "g_13",
        "Bxy",
        "bxcv",
        "logB",
    )
    for name in candidates:
        if name in ds.variables:
            stats[name] = _stats(np.asarray(ds.variables[name][:], dtype=np.float64))
    return stats


def _build_snapshot_fields(state) -> list[str]:
    fields = ["n", "omega", "vpar_e", "vpar_i", "Te"]
    if state.Ti is not None:
        fields.append("Ti")
    if state.psi is not None:
        fields.append("psi")
    if state.N is not None:
        fields.append("N")
    return fields


def _compute_term_metrics(
    system,
    snapshots: dict[str, np.ndarray],
    times: np.ndarray,
    *,
    start_index: int,
    nsteps: int,
    phi_override: np.ndarray | None = None,
):
    from jaxdrb.core.state import DRBSystemState

    out_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    total_fields: dict[str, list[np.ndarray]] = {}
    phi_rows: list[dict[str, object]] = []
    nsteps = int(max(0, min(nsteps, snapshots["n"].shape[0] - start_index)))
    engine = str(getattr(system, "engine", "unified")).lower()
    use_parity_engine = engine == "parity_fv"

    if not use_parity_engine:
        from jaxdrb.core.terms import build_context
        import equinox as eqx

    for ti in range(nsteps):
        idx = int(start_index + ti)
        y = DRBSystemState(
            n=snapshots["n"][idx],
            omega=snapshots["omega"][idx],
            vpar_e=snapshots["vpar_e"][idx],
            vpar_i=snapshots["vpar_i"][idx],
            Te=snapshots["Te"][idx],
            Ti=None if "Ti" not in snapshots else snapshots["Ti"][idx],
            psi=None if "psi" not in snapshots else snapshots["psi"][idx],
            N=None if "N" not in snapshots else snapshots["N"][idx],
        )
        if use_parity_engine:
            phi_arg = (
                None if phi_override is None else jnp.asarray(phi_override[idx], dtype=y.n.dtype)
            )
            split, term_map, phi_val, phi_iters = system.rhs_terms(
                float(times[idx]),
                y,
                phi_override=phi_arg,
            )
            ctx_phi = phi_val
            ctx_phi_iters = phi_iters
            pe_adv_override = None
        else:
            ctx = build_context(system.params, system.geom, y, return_phi_iters=True)
            if phi_override is not None:
                ctx = eqx.tree_at(
                    lambda c: c.phi,
                    ctx,
                    jnp.asarray(phi_override[idx], dtype=ctx.phi.dtype),
                )
            split, term_map = system.scheduler.run_with_terms(ctx, y)
            ctx_phi = ctx.phi
            ctx_phi_iters = ctx.phi_iters
            pe_adv_override = None
            adv_form = str(getattr(system.params, "exb_advection_form", "flux")).lower()
            if adv_form == "flux" and hasattr(ctx.geom, "exb_flux_divergence"):
                try:
                    pe_adv_override = -ctx.geom.exb_flux_divergence(
                        ctx.phi, ctx.n_phys * ctx.Te_phys, bc_phi=ctx.bcs.phi, bc_adv=ctx.bcs.Te
                    )
                except Exception:
                    pe_adv_override = None
            elif hasattr(ctx.geom, "bracket"):
                try:
                    pe_adv_override = -ctx.geom.bracket(
                        ctx.phi, ctx.n_phys * ctx.Te_phys, bc_phi=ctx.bcs.phi, bc_f=ctx.bcs.Te
                    )
                except Exception:
                    pe_adv_override = None
        total = split.total()
        phi = np.asarray(ctx_phi)
        phi_iters_val = None
        if ctx_phi_iters is not None:
            phi_iters_arr = np.asarray(ctx_phi_iters, dtype=np.float64)
            phi_iters_val = float(np.max(phi_iters_arr))
        phi_rows.append(
            {
                "step": int(ti),
                "t": float(times[idx]),
                "phi_rms": float(np.sqrt(np.mean(phi * phi))),
                "phi_maxabs": float(np.max(np.abs(phi))),
                "phi_iters": phi_iters_val,
            }
        )
        for field_name, arr in (
            ("n", total.n),
            ("omega", total.omega),
            ("vpar_e", total.vpar_e),
            ("vpar_i", total.vpar_i),
            ("Te", total.Te),
            ("Ti", total.Ti),
            ("psi", total.psi),
            ("N", total.N),
        ):
            if arr is None:
                continue
            total_fields.setdefault(field_name, []).append(np.asarray(arr))
            a = np.asarray(arr)
            total_rows.append(
                {
                    "step": int(ti),
                    "t": float(times[idx]),
                    "field": field_name,
                    "rhs_rms": float(np.sqrt(np.mean(a * a))),
                    "rhs_maxabs": float(np.max(np.abs(a))),
                }
            )
        for name, term in term_map.items():
            # Derived pressure contribution (Pe = n * Te) for parity with Hermes pressure equation.
            pe_term = None
            if "n" in snapshots and "Te" in snapshots:
                n_snap = snapshots["n"][idx]
                Te_snap = snapshots["Te"][idx]
                pe_term = n_snap * term.Te + Te_snap * term.n
            if name == "advection" and pe_adv_override is not None:
                pe_term = pe_adv_override
            for field_name, arr in (
                ("n", term.n),
                ("omega", term.omega),
                ("vpar_e", term.vpar_e),
                ("vpar_i", term.vpar_i),
                ("Te", term.Te),
                ("Ti", term.Ti),
                ("psi", term.psi),
                ("N", term.N),
            ):
                if arr is None:
                    continue
                a = np.asarray(arr)
                out_rows.append(
                    {
                        "step": int(ti),
                        "t": float(times[idx]),
                        "term": name,
                        "field": field_name,
                        "rms": float(np.sqrt(np.mean(a * a))),
                        "maxabs": float(np.max(np.abs(a))),
                        "mean": float(np.mean(a)),
                    }
                )
            if pe_term is not None:
                a = np.asarray(pe_term)
                out_rows.append(
                    {
                        "step": int(ti),
                        "t": float(times[idx]),
                        "term": name,
                        "field": "Pe",
                        "rms": float(np.sqrt(np.mean(a * a))),
                        "maxabs": float(np.max(np.abs(a))),
                        "mean": float(np.mean(a)),
                    }
                )

    return out_rows, total_rows, total_fields, phi_rows


def _compute_hermes_term_metrics(
    fields: dict[str, np.ndarray],
    times: np.ndarray,
    *,
    start_index: int,
    nsteps: int,
    ddt_scale: float,
    step_offset: int = 0,
    axis_map: str = "none",
    hermes_parallel_axis: str = "z",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    nt = int(fields["Ne"].shape[0])
    nsteps = int(max(0, nsteps))
    hermes_parallel_axis = str(hermes_parallel_axis).lower()

    def _parallel_axis_from_map(arr: np.ndarray) -> int | None:
        if arr.ndim < 3:
            return None
        if "->jax_" not in axis_map:
            return None
        order = axis_map.split("->jax_", 1)[1]
        if "_" in order:
            order = order.split("_", 1)[0]
        if not order.startswith("t"):
            return None
        # order is like "tyxz" / "tzxy" / "txy"
        spatial_order = order[1:]
        axis_char = "y" if hermes_parallel_axis == "y" else "z"
        idx = spatial_order.find(axis_char)
        if idx < 0:
            return None
        return int(idx)

    def _split_boundary_interior(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boundary = np.zeros_like(arr)
        parallel_axis = _parallel_axis_from_map(arr)
        if parallel_axis is None:
            # No reliable parallel axis mapping: keep backward-compatible behavior.
            return arr, boundary
        axis_full = 1 + parallel_axis
        if axis_full < 1 or axis_full >= arr.ndim:
            return arr, boundary
        if arr.shape[axis_full] <= 1:
            return arr, boundary
        sl0 = [slice(None)] * arr.ndim
        sl1 = [slice(None)] * arr.ndim
        sl0[axis_full] = 0
        sl1[axis_full] = -1
        boundary[tuple(sl0)] = arr[tuple(sl0)]
        boundary[tuple(sl1)] = arr[tuple(sl1)]
        return boundary, arr - boundary

    def _add_rows(arr: np.ndarray, field: str, term: str, step: int, tval: float):
        a = np.asarray(arr, dtype=np.float64)
        rows.append(
            {
                "step": int(step),
                "t": float(tval),
                "term": term,
                "field": field,
                "rms": float(np.sqrt(np.mean(a * a))),
                "maxabs": float(np.max(np.abs(a))),
                "mean": float(np.mean(a)),
            }
        )

    for name, arr in fields.items():
        if not name.startswith("term_"):
            continue
        if name.startswith("term_Ne_"):
            field = "n"
            term = name[len("term_Ne_") :]
        elif name.startswith("term_Pe_"):
            field = "Pe"
            term = name[len("term_Pe_") :]
        elif name.startswith("term_Vort_"):
            field = "omega"
            term = name[len("term_Vort_") :]
        else:
            field = "unknown"
            term = name
        for ti in range(nsteps):
            idx = int(start_index + ti + step_offset)
            if idx < 0 or idx >= nt or idx >= arr.shape[0]:
                continue
            _add_rows(arr[idx] * ddt_scale, field, term, ti, times[idx])

    # Explicit external/total source channels when present.
    def _append_source_rows(key: str, field: str, term: str):
        if key not in fields:
            return
        arr = np.asarray(fields[key], dtype=np.float64)
        if arr.ndim < 3:
            return
        for ti in range(nsteps):
            idx = int(start_index + ti + step_offset)
            if idx < 0 or idx >= nt or idx >= arr.shape[0]:
                continue
            _add_rows(arr[idx] * ddt_scale, field, term, ti, times[idx])

    _append_source_rows("Se_src", "n", "source_ext")
    _append_source_rows("Pe_src", "Pe", "source_ext")
    _append_source_rows("SNe", "n", "source_total")
    _append_source_rows("SPe", "Pe", "source_total")

    # Residual source channel: everything in total source that is not the explicit
    # external source. This often captures sheath/core damping or component-added
    # source/sink channels in Hermes runs.
    if "SNe" in fields and "Se_src" in fields:
        sne = np.asarray(fields["SNe"], dtype=np.float64)
        se = np.asarray(fields["Se_src"], dtype=np.float64)
        if sne.ndim >= 3 and se.ndim >= 3:
            residual = sne - se
            residual_boundary, residual_interior = _split_boundary_interior(residual)
            for ti in range(nsteps):
                idx = int(start_index + ti + step_offset)
                if idx < 0 or idx >= nt or idx >= residual.shape[0]:
                    continue
                _add_rows(residual[idx] * ddt_scale, "n", "source_residual", ti, times[idx])
                _add_rows(
                    residual_boundary[idx] * ddt_scale,
                    "n",
                    "source_residual_boundary",
                    ti,
                    times[idx],
                )
                _add_rows(
                    residual_interior[idx] * ddt_scale,
                    "n",
                    "source_residual_interior",
                    ti,
                    times[idx],
                )
    if "SPe" in fields and "Pe_src" in fields:
        spe = np.asarray(fields["SPe"], dtype=np.float64)
        se = np.asarray(fields["Pe_src"], dtype=np.float64)
        if spe.ndim >= 3 and se.ndim >= 3:
            residual = spe - se
            residual_boundary, residual_interior = _split_boundary_interior(residual)
            for ti in range(nsteps):
                idx = int(start_index + ti + step_offset)
                if idx < 0 or idx >= nt or idx >= residual.shape[0]:
                    continue
                _add_rows(residual[idx] * ddt_scale, "Pe", "source_residual", ti, times[idx])
                _add_rows(
                    residual_boundary[idx] * ddt_scale,
                    "Pe",
                    "source_residual_boundary",
                    ti,
                    times[idx],
                )
                _add_rows(
                    residual_interior[idx] * ddt_scale,
                    "Pe",
                    "source_residual_interior",
                    ti,
                    times[idx],
                )
    # Fallback omega channels present in some Hermes outputs when
    # term_Vort_* channels are not enabled.
    _append_source_rows("DivJdia", "omega", "divJdia")
    _append_source_rows("DivJpol", "omega", "divJpol")

    # Derive Te-term contributions when possible: dTe = (dP - Te * dN) / N
    if "Ne" in fields and "Te" in fields:
        # Collect suffixes available in pressure terms
        pe_terms = {name[len("term_Pe_") :]: name for name in fields if name.startswith("term_Pe_")}
        for suffix, name in pe_terms.items():
            for ti in range(nsteps):
                idx = int(start_index + ti + step_offset)
                if idx < 0 or idx >= nt or idx >= fields[name].shape[0]:
                    continue
                Ne = fields["Ne"][idx]
                Te = fields["Te"][idx]
                pterm = fields[name][idx] * ddt_scale
                nterm_name = f"term_Ne_{suffix}"
                if nterm_name in fields and idx < fields[nterm_name].shape[0]:
                    nterm = fields[nterm_name][idx] * ddt_scale
                else:
                    nterm = 0.0
                dte = (pterm - Te * nterm) / np.maximum(Ne, 1e-12)
                _add_rows(dte, "Te", suffix, ti, times[idx])

        def _append_te_source_rows(pe_key: str, ne_key: str, term_name: str):
            if pe_key not in fields or ne_key not in fields:
                return
            psrc = np.asarray(fields[pe_key], dtype=np.float64)
            nsrc = np.asarray(fields[ne_key], dtype=np.float64)
            if psrc.ndim < 3 or nsrc.ndim < 3:
                return
            for ti in range(nsteps):
                idx = int(start_index + ti + step_offset)
                if idx < 0 or idx >= nt or idx >= psrc.shape[0] or idx >= nsrc.shape[0]:
                    continue
                Ne = fields["Ne"][idx]
                Te = fields["Te"][idx]
                dte = ((psrc[idx] - Te * nsrc[idx]) / np.maximum(Ne, 1e-12)) * ddt_scale
                _add_rows(dte, "Te", term_name, ti, times[idx])

        _append_te_source_rows("Pe_src", "Se_src", "source_ext")
        _append_te_source_rows("SPe", "SNe", "source_total")
        if all(k in fields for k in ("SPe", "SNe", "Pe_src", "Se_src")):
            spe = np.asarray(fields["SPe"], dtype=np.float64)
            sne = np.asarray(fields["SNe"], dtype=np.float64)
            pe_src = np.asarray(fields["Pe_src"], dtype=np.float64)
            se_src = np.asarray(fields["Se_src"], dtype=np.float64)
            if all(arr.ndim >= 3 for arr in (spe, sne, pe_src, se_src)):
                p_res = spe - pe_src
                n_res = sne - se_src
                p_res_boundary, p_res_interior = _split_boundary_interior(p_res)
                n_res_boundary, n_res_interior = _split_boundary_interior(n_res)
                for ti in range(nsteps):
                    idx = int(start_index + ti + step_offset)
                    if idx < 0 or idx >= nt or idx >= p_res.shape[0] or idx >= n_res.shape[0]:
                        continue
                    Ne = fields["Ne"][idx]
                    Te = fields["Te"][idx]
                    dte = ((p_res[idx] - Te * n_res[idx]) / np.maximum(Ne, 1e-12)) * ddt_scale
                    dte_boundary = (
                        (p_res_boundary[idx] - Te * n_res_boundary[idx]) / np.maximum(Ne, 1e-12)
                    ) * ddt_scale
                    dte_interior = (
                        (p_res_interior[idx] - Te * n_res_interior[idx]) / np.maximum(Ne, 1e-12)
                    ) * ddt_scale
                    _add_rows(dte, "Te", "source_residual", ti, times[idx])
                    _add_rows(
                        dte_boundary,
                        "Te",
                        "source_residual_boundary",
                        ti,
                        times[idx],
                    )
                    _add_rows(
                        dte_interior,
                        "Te",
                        "source_residual_interior",
                        ti,
                        times[idx],
                    )

        # Aggregate Hermes pressure transport channels for parity with JAX
        # conservative temperature update, which carries flux+work together.
        if "par" in pe_terms and "work" in pe_terms:
            par_name = pe_terms["par"]
            work_name = pe_terms["work"]
            has_npar = "term_Ne_par" in fields
            for ti in range(nsteps):
                idx = int(start_index + ti + step_offset)
                if idx < 0 or idx >= nt or idx >= fields[par_name].shape[0]:
                    continue
                p_par = np.asarray(fields[par_name][idx], dtype=np.float64) * ddt_scale
                p_work = np.asarray(fields[work_name][idx], dtype=np.float64) * ddt_scale
                p_tot = p_par + p_work
                _add_rows(p_tot, "Pe", "par_total", ti, times[idx])
                if has_npar and idx < fields["term_Ne_par"].shape[0]:
                    n_par = np.asarray(fields["term_Ne_par"][idx], dtype=np.float64) * ddt_scale
                    Ne = np.asarray(fields["Ne"][idx], dtype=np.float64)
                    Te = np.asarray(fields["Te"][idx], dtype=np.float64)
                    dte_tot = (p_tot - Te * n_par) / np.maximum(Ne, 1e-12)
                    _add_rows(dte_tot, "Te", "par_total", ti, times[idx])

    return rows


def _compute_term_mismatch(
    jax_term_rows: list[dict[str, object]],
    hermes_term_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    mapping: dict[str, dict[str, tuple[str, ...]]] = {
        "n": {
            "advection": ("exb",),
            "parallel": ("par",),
            "volume_source": ("source_ext", "source"),
            "sheath": ("source_residual_boundary", "source_residual"),
            "diffusion": ("low_n_diff_perp",),
        },
        "Pe": {
            "advection": ("exb",),
            "parallel": ("par_total", "par"),
            "volume_source": ("source_ext", "source"),
            "sheath": ("source_residual_boundary", "source_residual"),
            "diffusion": ("low_n_diff_perp",),
        },
        "Te": {
            "advection": ("exb",),
            "parallel": ("par_total", "par"),
            "volume_source": ("source_ext", "source"),
            "sheath": ("source_residual_boundary", "source_residual"),
            "diffusion": ("low_n_diff_perp",),
        },
        "omega": {
            "advection": ("exb",),
            "parallel": ("jpar", "divJpol"),
            "diamagnetic_current": ("divJdia",),
            "extra_dissipation": ("vort_diss",),
        },
    }
    hermes_index: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in hermes_term_rows:
        step = int(row.get("step", 0))
        field = str(row.get("field"))
        term = str(row.get("term"))
        hermes_index[(step, field, term)] = row

    mismatch: list[dict[str, object]] = []
    for row in jax_term_rows:
        field = str(row.get("field"))
        term = str(row.get("term"))
        step = int(row.get("step", 0))
        if field not in mapping:
            continue
        if term not in mapping[field]:
            continue
        hermes_terms = mapping[field][term]
        hrow = None
        hermes_term = ""
        for candidate in hermes_terms:
            hrow = hermes_index.get((step, field, candidate))
            if hrow is not None:
                hermes_term = candidate
                break
        if hrow is None:
            continue
        jax_rms = float(row.get("rms", 0.0))
        hermes_rms = float(hrow.get("rms", 0.0))
        denom = max(1e-12, 0.1 * hermes_rms)
        mismatch.append(
            {
                "step": step,
                "t": float(row.get("t", 0.0)),
                "field": field,
                "term": term,
                "hermes_term": hermes_term,
                "jax_rms": jax_rms,
                "hermes_rms": hermes_rms,
                "rel_diff": abs(jax_rms - hermes_rms) / denom,
            }
        )
    return mismatch


def _compute_hermes_ddt(
    times: np.ndarray,
    fields: dict[str, np.ndarray],
    nsteps: int,
    *,
    start_index: int,
    step_offset: int = 0,
) -> dict[str, np.ndarray]:
    ddt: dict[str, np.ndarray] = {}
    nt = int(times.size)
    nsteps = int(max(0, nsteps))

    # Prefer direct ddt variables if present (interior values are already extracted).
    if "ddt(Ne)" in fields:
        rows = []
        ref_shape = fields["ddt(Ne)"].shape[1:]
        for ti in range(nsteps):
            idx = int(start_index + ti + step_offset)
            if idx < 0 or idx >= fields["ddt(Ne)"].shape[0]:
                rows.append(np.full(ref_shape, np.nan, dtype=np.float64))
            else:
                rows.append(np.asarray(fields["ddt(Ne)"][idx], dtype=np.float64))
        ddt["Ne"] = np.stack(rows, axis=0)
    if "ddt(Pe)" in fields and "Ne" in fields:
        rows = []
        ref_shape = fields["ddt(Pe)"].shape[1:]
        for ti in range(nsteps):
            idx = int(start_index + ti + step_offset)
            if idx < 0 or idx >= fields["ddt(Pe)"].shape[0] or idx >= fields["Ne"].shape[0]:
                rows.append(np.full(ref_shape, np.nan, dtype=np.float64))
                continue
            dpe = np.asarray(fields["ddt(Pe)"][idx], dtype=np.float64)
            n = np.asarray(fields["Ne"][idx], dtype=np.float64)
            if "Te" in fields and idx < fields["Te"].shape[0]:
                Te = np.asarray(fields["Te"][idx], dtype=np.float64)
            elif "Pe" in fields and idx < fields["Pe"].shape[0]:
                Te = np.asarray(fields["Pe"][idx] / np.maximum(fields["Ne"][idx], 1e-12))
            else:
                rows.append(np.full(ref_shape, np.nan, dtype=np.float64))
                continue
            dne = (
                np.asarray(ddt["Ne"][ti], dtype=np.float64)
                if "Ne" in ddt
                else np.zeros_like(dpe, dtype=np.float64)
            )
            rows.append((dpe - Te * dne) / np.maximum(n, 1e-12))
        ddt["Te"] = np.stack(rows, axis=0)

    if "ddt(Vort)" in fields:
        rows = []
        ref_shape = fields["ddt(Vort)"].shape[1:]
        for ti in range(nsteps):
            idx = int(start_index + ti + step_offset)
            if idx < 0 or idx >= fields["ddt(Vort)"].shape[0]:
                rows.append(np.full(ref_shape, np.nan, dtype=np.float64))
            else:
                rows.append(np.asarray(fields["ddt(Vort)"][idx], dtype=np.float64))
        ddt["Vort"] = np.stack(rows, axis=0)

    # Fallback to finite differences for missing fields.
    for name in ("Ne", "Te", "Vort", "phi"):
        if name in ddt:
            continue
        if name not in fields:
            continue
        arr = fields[name]
        rows = []
        ref_shape = arr.shape[1:]
        for ti in range(nsteps):
            idx = int(start_index + ti + step_offset)
            idxp = idx + 1
            if idx < 0 or idxp >= arr.shape[0] or idxp >= nt:
                rows.append(np.full(ref_shape, np.nan, dtype=np.float64))
                continue
            dt = float(times[idxp] - times[idx])
            if abs(dt) <= 0.0:
                rows.append(np.full(ref_shape, np.nan, dtype=np.float64))
                continue
            rows.append((arr[idxp] - arr[idx]) / dt)
        ddt[name] = np.stack(rows, axis=0)
    return ddt


def _compute_poisson_parity_metrics(
    system,
    snapshots: dict[str, np.ndarray],
    times: np.ndarray,
    *,
    start_index: int,
    nsteps: int,
) -> list[dict[str, object]]:
    from jaxdrb.core.state import DRBSystemState
    from jaxdrb.core.terms.bcs import resolve_bcs
    from jaxdrb.core.terms.fields import omega_from_phi, phi_from_omega, phys_n, phys_Te

    rows: list[dict[str, object]] = []
    nsteps = int(max(0, min(nsteps, snapshots["n"].shape[0] - start_index)))
    bcs = resolve_bcs(system.params, system.geom)
    crop_x = 2 if bcs.phi.kind_x != 0 else 0

    def _crop_x(arr: np.ndarray) -> np.ndarray:
        if crop_x <= 0:
            return arr
        if arr.ndim == 2 and arr.shape[0] > 2 * crop_x:
            return arr[crop_x:-crop_x, :]
        if arr.ndim == 3 and arr.shape[1] > 2 * crop_x:
            return arr[:, crop_x:-crop_x, :]
        return arr

    for ti in range(nsteps):
        idx = int(start_index + ti)
        y = DRBSystemState(
            n=snapshots["n"][idx],
            omega=snapshots["omega"][idx],
            vpar_e=snapshots["vpar_e"][idx],
            vpar_i=snapshots["vpar_i"][idx],
            Te=snapshots["Te"][idx],
            Ti=None if "Ti" not in snapshots else snapshots["Ti"][idx],
            psi=None if "psi" not in snapshots else snapshots["psi"][idx],
            N=None if "N" not in snapshots else snapshots["N"][idx],
        )
        n_phys = phys_n(system.params, y.n)
        Te_phys = phys_Te(system.params, y.Te)
        omega_ref = np.asarray(y.omega, dtype=np.float64)
        phi_ref = np.asarray(snapshots["phi"][idx], dtype=np.float64)
        omega_from_ref = np.asarray(
            omega_from_phi(
                system.params,
                system.geom,
                snapshots["phi"][idx],
                n_phys,
                bcs.phi,
                Ti=y.Ti,
                Te=Te_phys,
            ),
            dtype=np.float64,
        )
        phi_from_ref = np.asarray(
            phi_from_omega(
                system.params,
                system.geom,
                y.omega,
                n_phys,
                bcs.phi,
                Ti=y.Ti,
                Te=Te_phys,
                phi_guess=snapshots["phi"][idx],
            ),
            dtype=np.float64,
        )

        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            aa = a - float(np.mean(a))
            bb = b - float(np.mean(b))
            denom = float(np.sqrt(np.mean(aa * aa)) * np.sqrt(np.mean(bb * bb)))
            if denom <= 0.0:
                return 0.0
            return float(np.mean(aa * bb) / denom)

        def _ls_scale(a: np.ndarray, b: np.ndarray) -> float:
            # alpha minimizing ||alpha a - b||_2
            num = float(np.mean(a * b))
            den = float(np.mean(a * a))
            if den <= 1e-30:
                return 0.0
            return num / den

        omega_ref_core = _crop_x(omega_ref)
        omega_from_ref_core = _crop_x(omega_from_ref)
        phi_ref_core = _crop_x(phi_ref)
        phi_from_ref_core = _crop_x(phi_from_ref)

        rows.append(
            {
                "step": int(ti),
                "t": float(times[idx]),
                "crop_x": int(crop_x),
                "omega_ref_rms": float(np.sqrt(np.mean(omega_ref * omega_ref))),
                "omega_from_phi_rms": float(np.sqrt(np.mean(omega_from_ref * omega_from_ref))),
                "omega_from_phi_corr": _corr(omega_from_ref, omega_ref),
                "omega_from_phi_scale": _ls_scale(omega_from_ref, omega_ref),
                "omega_ref_rms_core": float(np.sqrt(np.mean(omega_ref_core * omega_ref_core))),
                "omega_from_phi_rms_core": float(
                    np.sqrt(np.mean(omega_from_ref_core * omega_from_ref_core))
                ),
                "omega_from_phi_corr_core": _corr(omega_from_ref_core, omega_ref_core),
                "omega_from_phi_scale_core": _ls_scale(omega_from_ref_core, omega_ref_core),
                "phi_ref_rms": float(np.sqrt(np.mean(phi_ref * phi_ref))),
                "phi_from_omega_rms": float(np.sqrt(np.mean(phi_from_ref * phi_from_ref))),
                "phi_from_omega_corr": _corr(phi_from_ref, phi_ref),
                "phi_from_omega_scale": _ls_scale(phi_from_ref, phi_ref),
                "phi_ref_rms_core": float(np.sqrt(np.mean(phi_ref_core * phi_ref_core))),
                "phi_from_omega_rms_core": float(
                    np.sqrt(np.mean(phi_from_ref_core * phi_from_ref_core))
                ),
                "phi_from_omega_corr_core": _corr(phi_from_ref_core, phi_ref_core),
                "phi_from_omega_scale_core": _ls_scale(phi_from_ref_core, phi_ref_core),
            }
        )
    return rows


def _align_hermes_fields(
    fields: dict[str, np.ndarray],
    jax_spatial_shape: tuple[int, ...],
    *,
    hermes_parallel_axis: str = "z",
) -> tuple[dict[str, np.ndarray], str]:
    aligned: dict[str, np.ndarray] = {}
    mapping = "none"
    hermes_parallel_axis = str(hermes_parallel_axis).lower()
    for name, arr in fields.items():
        if arr.ndim == 4:
            # Hermes: (t, x, y, z). JAX 3D: (t, z, x, y).
            if hermes_parallel_axis == "y":
                if jax_spatial_shape == (arr.shape[2], arr.shape[1], arr.shape[3]):
                    aligned[name] = np.transpose(arr, (0, 2, 1, 3))
                    mapping = "hermes_txyz->jax_tyxz"
                elif jax_spatial_shape == arr.shape[1:]:
                    aligned[name] = arr
                    mapping = "hermes_txyz->jax_txyz"
                elif jax_spatial_shape == (arr.shape[3], arr.shape[1], arr.shape[2]):
                    aligned[name] = np.transpose(arr, (0, 3, 1, 2))
                    mapping = "hermes_txyz->jax_tzxy"
                elif len(jax_spatial_shape) == 2:
                    aligned[name] = np.mean(arr, axis=-1)
                    mapping = "hermes_txyz->jax_txy_meanz"
                else:
                    aligned[name] = arr
                    mapping = "hermes_txyz->jax_unmatched"
                continue

            if jax_spatial_shape == arr.shape[1:]:
                aligned[name] = arr
                mapping = "hermes_txyz->jax_txyz"
            elif jax_spatial_shape == (arr.shape[3], arr.shape[1], arr.shape[2]):
                aligned[name] = np.transpose(arr, (0, 3, 1, 2))
                mapping = "hermes_txyz->jax_tzxy"
            elif len(jax_spatial_shape) == 2:
                aligned[name] = np.mean(arr, axis=-1)
                mapping = "hermes_txyz->jax_txy_meanz"
            else:
                aligned[name] = arr
                mapping = "hermes_txyz->jax_unmatched"
        elif arr.ndim == 3:
            if jax_spatial_shape == arr.shape[1:]:
                aligned[name] = arr
                mapping = "hermes_txy->jax_txy"
            elif len(jax_spatial_shape) == 2:
                aligned[name] = arr
                mapping = "hermes_txy->jax_txy"
            else:
                aligned[name] = arr
                mapping = "hermes_txy->jax_unmatched"
        else:
            aligned[name] = arr
            mapping = "hermes_unknown"
    return aligned, mapping


def _load_hermes_zshift(
    grid_path: Path | None,
    hermes_meta: dict[str, float],
    *,
    data_dir: Path,
    pattern: str,
) -> np.ndarray:
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required to read Hermes grid files.") from exc

    files = sorted(data_dir.glob(pattern))
    if files:
        with Dataset(str(files[0])) as ds0:
            if "zShift" in ds0.variables or "zShift_ylow" in ds0.variables:
                mxg = int(_read_scalar(ds0, ("MXG",), 0))
                myg = int(_read_scalar(ds0, ("MYG",), 0))
                mxsub = int(_read_scalar(ds0, ("MXSUB",), ds0.variables["Ne"].shape[1] - 2 * mxg))
                mysub = int(_read_scalar(ds0, ("MYSUB",), ds0.variables["Ne"].shape[2] - 2 * myg))
                nxpe = int(_read_scalar(ds0, ("NXPE",), 1))
                nype = int(_read_scalar(ds0, ("NYPE",), max(1, len(files) // max(nxpe, 1))))
                nx = nxpe * mxsub
                ny = nype * mysub
                zshift = np.zeros((nx, ny), dtype=np.float64)
                for local_rank, fp in enumerate(files):
                    with Dataset(str(fp)) as ds:
                        if "zShift" in ds.variables:
                            arr = np.asarray(ds.variables["zShift"][:], dtype=np.float64)
                        elif "zShift_ylow" in ds.variables:
                            arr = np.asarray(ds.variables["zShift_ylow"][:], dtype=np.float64)
                        else:
                            continue
                        pe_x = int(_read_scalar(ds, ("PE_XIND",), local_rank % max(nxpe, 1)))
                        pe_y = int(_read_scalar(ds, ("PE_YIND",), local_rank // max(nxpe, 1)))
                        x0 = pe_x * mxsub
                        y0 = pe_y * mysub
                        x1 = x0 + mxsub
                        y1 = y0 + mysub
                        arr = arr[mxg : mxg + mxsub, myg : myg + mysub]
                        zshift[x0:x1, y0:y1] = arr
                return zshift

    if grid_path is None:
        raise ValueError("No Hermes dumps with zShift and no grid file provided.")

    with Dataset(str(grid_path)) as ds:
        if "zShift" in ds.variables:
            zshift = np.asarray(ds.variables["zShift"][:], dtype=np.float64)
        elif "zShift_ylow" in ds.variables:
            zshift = np.asarray(ds.variables["zShift_ylow"][:], dtype=np.float64)
        else:
            raise KeyError("Hermes grid missing zShift/zShift_ylow for shifted metric.")
        mxg = int(_read_scalar(ds, ("MXG",), 0))
        myg = int(_read_scalar(ds, ("MYG",), 0))

    nx = int(hermes_meta.get("mxsub", 0) * hermes_meta.get("nxpe", 1))
    ny = int(hermes_meta.get("mysub", 0) * hermes_meta.get("nype", 1))
    if nx <= 0 or ny <= 0:
        raise ValueError("Hermes meta missing nx/ny for zShift slicing.")

    if zshift.ndim != 2:
        raise ValueError(f"zShift should be 2D, got shape {zshift.shape}.")

    return np.asarray(zshift[mxg : mxg + nx, myg : myg + ny], dtype=np.float64)


def _apply_shifted_binormal(arr: np.ndarray, *, shift_idx: np.ndarray) -> np.ndarray:
    """Shift field along binormal (last axis) by shift_idx (index units)."""
    if arr.ndim != 4:
        return arr
    nz, nx, ny = arr.shape[1:]
    if shift_idx.shape != (nz, nx):
        if shift_idx.shape == (nx, nz):
            shift_idx = shift_idx.T
        else:
            raise ValueError(
                f"shift_idx shape {shift_idx.shape} does not match (nz,nx)=({nz},{nx})."
            )
    y = np.arange(ny, dtype=np.float64)
    y_src = (y[None, None, :] + shift_idx[..., None]) % float(ny)
    y0 = np.floor(y_src).astype(int)
    y1 = (y0 + 1) % ny
    frac = y_src - y0
    y0_idx = y0[None, ...]
    y1_idx = y1[None, ...]
    f0 = np.take_along_axis(arr, y0_idx, axis=-1)
    f1 = np.take_along_axis(arr, y1_idx, axis=-1)
    return (1.0 - frac)[None, ...] * f0 + frac[None, ...] * f1


def _override_geometry_from_hermes(
    cfg: dict[str, object],
    grid_path: Path,
    *,
    hermes_meta: dict[str, float] | None = None,
    hermes_dx_mean: float | None = None,
    hermes_binormal_mean: float | None = None,
    override_lengths: bool = False,
) -> dict[str, object]:
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required to read Hermes grid files.") from exc

    with Dataset(str(grid_path)) as ds:
        dx = np.asarray(ds.variables["dx"][:], dtype=np.float64)
        dy = np.asarray(ds.variables["dy"][:], dtype=np.float64)
        nx_full = int(ds.dimensions["x"].size)
        ny_full = int(ds.dimensions["y"].size)

    if hermes_meta is not None and "mxg" in hermes_meta and "myg" in hermes_meta:
        mxg = int(round(hermes_meta.get("mxg", 0)))
        myg = int(round(hermes_meta.get("myg", 0)))
        nx = int(round(hermes_meta.get("nx", nx_full)))
        ny = int(round(hermes_meta.get("ny", ny_full)))
        x_slice = slice(mxg, mxg + nx)
        y_slice = slice(myg, myg + ny)
        dx_int = dx[x_slice, y_slice]
        dy_int = dy[x_slice, y_slice]
        Lx_default = float(np.mean(dx_int, axis=1).sum())
        # dy in BOUT/Hermes grid is usually parallel spacing, not binormal spacing.
        # Use an explicit binormal mean from dump metadata when available.
        Ly_default = float(np.mean(dy_int, axis=0).sum())
    else:
        nx = nx_full
        ny = ny_full
        Lx_default = float(dx.mean() * nx_full)
        Ly_default = float(dy.mean() * ny_full)

    geom = dict(cfg.get("geometry", {}))
    Lx_cfg = geom.get("Lx", None)
    Ly_cfg = geom.get("Ly", None)
    if override_lengths or Lx_cfg is None:
        if hermes_dx_mean is not None:
            Lx = float(hermes_dx_mean * nx)
        else:
            Lx = Lx_default
    else:
        Lx = float(Lx_cfg)
    if override_lengths or Ly_cfg is None:
        if hermes_binormal_mean is not None:
            Ly = float(hermes_binormal_mean * ny)
        else:
            Ly = Ly_default
    else:
        Ly = float(Ly_cfg)

    geom["nx"] = nx
    geom["ny"] = ny
    geom["Lx"] = Lx
    geom["Ly"] = Ly
    if "r_minor" in geom:
        geom["r_minor"] = Lx
    cfg["geometry"] = geom
    return cfg


def _crop_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    if a.ndim != b.ndim:
        raise ValueError(f"Cannot crop arrays with different ranks {a.ndim} vs {b.ndim}.")
    slices = []
    for dim_a, dim_b in zip(a.shape, b.shape, strict=True):
        n = min(dim_a, dim_b)
        slices.append(slice(0, n))
    slicer = tuple(slices)
    return a[slicer], b[slicer]


def _align_to_shape_strict(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Best-effort alignment for strict comparisons.

    Allows singleton expansion (broadcast) and insertion of singleton axes.
    """
    a = np.asarray(arr)
    if a.shape == target_shape:
        return a

    # Direct broadcast if already same rank.
    if a.ndim == len(target_shape):
        try:
            return np.broadcast_to(a, target_shape)
        except ValueError:
            pass

    # Insert singleton axes and try broadcast.
    if a.ndim < len(target_shape):
        add = len(target_shape) - a.ndim
        for axes in itertools.combinations(range(len(target_shape)), add):
            shape = list(a.shape)
            for ax in sorted(axes):
                shape.insert(ax, 1)
            try:
                reshaped = a.reshape(shape)
            except ValueError:
                continue
            try:
                return np.broadcast_to(reshaped, target_shape)
            except ValueError:
                continue

    raise ValueError(f"Cannot strict-align shape {a.shape} -> {target_shape}")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_csv_set(text: str) -> set[str]:
    return {tok.strip() for tok in str(text).split(",") if tok.strip()}


def _first_failing_term(
    term_mismatch_rows: list[dict[str, object]],
    *,
    fields: set[str],
    terms: set[str],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    filtered: list[dict[str, object]] = []
    for row in term_mismatch_rows:
        field = str(row.get("field", ""))
        term = str(row.get("term", ""))
        if fields and field not in fields:
            continue
        if terms and term not in terms:
            continue
        filtered.append(row)
    if not filtered:
        return None, []
    first_step = min(int(r.get("step", 0)) for r in filtered)
    step_rows = [r for r in filtered if int(r.get("step", 0)) == first_step]

    def _key(row: dict[str, object]) -> tuple[float, float]:
        weighted = float(row.get("weighted_rel", row.get("rel_diff", 0.0)))
        rel = float(row.get("rel_diff", 0.0))
        return weighted, rel

    step_rows.sort(key=_key, reverse=True)
    return step_rows[0], step_rows


def _coord_from_spacing(spacing: np.ndarray, axis: int) -> np.ndarray:
    if spacing.ndim == 2:
        if axis == 0:
            spacing_mean = np.mean(spacing, axis=1)
        else:
            spacing_mean = np.mean(spacing, axis=0)
    else:
        spacing_mean = np.asarray(spacing).reshape(-1)
    coord = np.zeros_like(spacing_mean)
    if spacing_mean.size > 1:
        coord[1:] = np.cumsum(spacing_mean[:-1])
    return coord


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit early-step term parity by dumping every term contribution and "
            "comparing JAX RHS with Hermes ddt estimates."
        )
    )
    parser.add_argument("--jax-config", required=True, help="jax_drb TOML config.")
    parser.add_argument("--hermes-data-dir", required=True, help="Hermes BOUT dump directory.")
    parser.add_argument(
        "--hermes-pattern",
        default="BOUT.dmp.[0-9]*.nc",
        help="Glob pattern for Hermes dump files (default: BOUT.dmp.[0-9]*.nc).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for audit files.")
    parser.add_argument("--nsteps", type=int, default=5, help="Number of steps to audit.")
    parser.add_argument(
        "--match-hermes-dt",
        action="store_true",
        help="Override jax dt to match Hermes dt.",
    )
    parser.add_argument(
        "--hermes-input",
        default="",
        help="Optional Hermes BOUT.inp for normalization metadata.",
    )
    parser.add_argument(
        "--hermes-grid",
        default="",
        help="Optional Hermes grid file (.nc) for Lx/Ly/nx/ny override.",
    )
    parser.add_argument(
        "--skip-geometry-override",
        action="store_true",
        help="Do not override JAX geometry from Hermes grid metadata.",
    )
    parser.add_argument(
        "--strict-axis",
        action="store_true",
        help="Fail if Hermes/JAX spatial shapes do not match after axis mapping.",
    )
    parser.add_argument(
        "--dump-term-arrays",
        action="store_true",
        help="Dump per-term arrays (npz) for each audited step.",
    )
    parser.add_argument(
        "--dump-terms",
        default="all",
        help="Comma-separated term list for array dumps (default: all terms).",
    )
    parser.add_argument(
        "--dump-fields",
        default="n,omega,Te,phi",
        help="Comma-separated fields to dump (default: n,omega,Te,phi).",
    )
    parser.add_argument(
        "--hermes-parallel-axis",
        default="y",
        choices=("y", "z"),
        help="Which Hermes axis corresponds to parallel direction (y or z).",
    )
    parser.add_argument(
        "--override-lengths-from-hermes",
        action="store_true",
        help=(
            "Override geometry Lx/Ly using Hermes spacing metadata. By default only nx/ny "
            "are overridden to avoid mixing parallel and binormal length scales."
        ),
    )
    parser.add_argument(
        "--hermes-shifted",
        action="store_true",
        help="Apply Hermes shifted-metric transform to aligned coordinates before comparison.",
    )
    parser.add_argument(
        "--hermes-shifted-sign",
        type=float,
        default=1.0,
        help="Sign for zShift when applying shifted-metric transform (default: +1).",
    )
    parser.add_argument(
        "--hermes-te-from-pe",
        action="store_true",
        help="Override Hermes Te with Pe/Ne when Pe is available (pressure-consistent Te).",
    )
    parser.add_argument(
        "--use-hermes-state",
        action="store_true",
        help="Evaluate JAX RHS directly on Hermes snapshots (skip JAX time integration).",
    )
    parser.add_argument(
        "--use-hermes-phi-in-terms",
        action="store_true",
        help=(
            "For term-level audits, use Hermes phi snapshots directly instead of solving "
            "phi from omega in JAX. This isolates Poisson-path mismatches."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info about snapshot selection and scaling.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into the time series for the audit window.",
    )
    parser.add_argument(
        "--time-target",
        type=float,
        default=None,
        help="Target time (Hermes t units) to anchor the audit window.",
    )
    parser.add_argument(
        "--no-time-scale",
        action="store_true",
        help="Disable scaling Hermes ddt into JAX time units.",
    )
    parser.add_argument(
        "--fail-fast-rel-diff",
        type=float,
        default=-1.0,
        help=("If >=0, exit non-zero when first failing term rel_diff exceeds this threshold."),
    )
    parser.add_argument(
        "--fail-fast-fields",
        default="n,Te,omega",
        help="Comma-separated fields considered by fail-fast (default: n,Te,omega).",
    )
    parser.add_argument(
        "--fail-fast-terms",
        default="advection,parallel",
        help="Comma-separated terms considered by fail-fast (default: advection,parallel).",
    )
    parser.add_argument(
        "--hermes-step-offset",
        type=int,
        default=0,
        help=(
            "Offset applied to Hermes term/ddt indexing relative to JAX step index. "
            "Useful when Hermes diagnostics are staggered in time."
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.jax_config)
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = _resolve_coeff_path(cfg, cfg_path)

    hermes_dir = Path(args.hermes_data_dir)
    hermes_input_path = Path(args.hermes_input) if args.hermes_input else None
    if hermes_input_path is None:
        candidate = hermes_dir / "BOUT.inp"
        if candidate.exists():
            hermes_input_path = candidate
    hermes_sections = _parse_bout_sections(hermes_input_path)
    hermes_term_names = (
        "term_Ne_exb",
        "term_Ne_par",
        "term_Ne_flutter",
        "term_Ne_low_n_diff",
        "term_Ne_low_n_diff_perp",
        "term_Ne_low_p_diff_perp",
        "term_Ne_hyper_z",
        "term_Ne_source",
        "term_Pe_exb",
        "term_Pe_par",
        "term_Pe_work",
        "term_Pe_flutter",
        "term_Pe_nvh",
        "term_Pe_low_n_diff",
        "term_Pe_low_n_diff_perp",
        "term_Pe_low_T_diff_perp",
        "term_Pe_low_p_diff_perp",
        "term_Pe_hyper_z",
        "term_Pe_hyper_z_T",
        "term_Pe_source",
        "term_Pe_damp_nt",
        "term_Vort_divJdia",
        "term_Vort_divJcol",
        "term_Vort_visc",
        "term_Vort_exb",
        "term_Vort_divJextra",
        "term_Vort_jpar",
        "term_Vort_jpar_flutter",
        "term_Vort_vort_diss",
        "term_Vort_phi_diss",
        "term_Vort_hyper_z",
        "term_Vort_phi_sheath",
        "term_Vort_damp_core",
    )
    hermes_times, hermes_fields, hermes_meta = _load_hermes_fields(
        hermes_dir,
        (
            "Ne",
            "Te",
            "Pe",
            "Vort",
            "phi",
            "DivJdia",
            "DivJpol",
            "SNe",
            "SPe",
            "Se_src",
            "Pe_src",
            "ddt(Ne)",
            "ddt(Pe)",
            "Ve",
            "NVe",
            "Vd+",
            "Nd+",
            "Pd+",
            "NVd+",
            *hermes_term_names,
        ),
        args.hermes_pattern,
    )
    if args.hermes_te_from_pe and "Pe" in hermes_fields and "Ne" in hermes_fields:
        hermes_fields["Te"] = hermes_fields["Pe"] / np.maximum(hermes_fields["Ne"], 1e-12)
    hermes_dx_mean = None
    hermes_dy_mean = None
    hermes_dz_mean = None
    try:
        from netCDF4 import Dataset

        files = sorted(hermes_dir.glob(args.hermes_pattern))
        if files:
            with Dataset(str(files[0])) as ds:
                if "dx" in ds.variables and "dy" in ds.variables:
                    hermes_dx_mean = float(np.mean(np.asarray(ds.variables["dx"][:])))
                    hermes_dy_mean = float(np.mean(np.asarray(ds.variables["dy"][:])))
                if "dz" in ds.variables:
                    hermes_dz_mean = float(np.mean(np.asarray(ds.variables["dz"][:])))
    except Exception:
        hermes_dx_mean = None
        hermes_dy_mean = None
        hermes_dz_mean = None
    if args.hermes_grid and not args.skip_geometry_override:
        hermes_binormal_mean = hermes_dy_mean
        if str(args.hermes_parallel_axis).lower() == "y" and hermes_dz_mean is not None:
            hermes_binormal_mean = hermes_dz_mean
        cfg = _override_geometry_from_hermes(
            cfg,
            Path(args.hermes_grid),
            hermes_meta=hermes_meta,
            hermes_dx_mean=hermes_dx_mean,
            hermes_binormal_mean=hermes_binormal_mean,
            override_lengths=bool(args.override_lengths_from_hermes),
        )
    start_index = int(max(0, args.start_index))
    if args.time_target is not None:
        start_index = int(np.argmin(np.abs(hermes_times - float(args.time_target))))

    dt_slice = hermes_times[start_index : start_index + max(2, args.nsteps + 1)]
    if dt_slice.size < 2:
        raise ValueError("Not enough Hermes frames to determine dt at requested time.")
    dt_hermes = float(np.mean(np.diff(dt_slice)))

    time_cfg = cfg.get("time", {})
    time_cfg = dict(time_cfg)
    nsteps_total = int(start_index + args.nsteps)
    time_cfg["nsteps"] = int(nsteps_total)
    time_cfg["save_every"] = 1
    time_cfg["save_fields"] = True
    time_cfg["return_numpy"] = True
    time_cfg["diag_mode"] = "basic"
    time_cfg["progress"] = False
    if args.match_hermes_dt:
        time_cfg["dt"] = dt_hermes
    cfg["time"] = time_cfg

    from jaxdrb.driver import build_system_from_config, run_simulation

    built = build_system_from_config(cfg)
    snapshots: dict[str, np.ndarray] = {}
    if args.use_hermes_state:
        hermes_fields_aligned, axis_map = _align_hermes_fields(
            hermes_fields,
            built.system.geom.shape(),
            hermes_parallel_axis=args.hermes_parallel_axis,
        )
        if args.hermes_shifted:
            if not args.hermes_grid:
                raise ValueError("--hermes-shifted requires --hermes-grid to load zShift.")
            zshift = _load_hermes_zshift(
                Path(args.hermes_grid) if args.hermes_grid else None,
                hermes_meta,
                data_dir=hermes_dir,
                pattern=args.hermes_pattern,
            )
            # Map zShift (x, y_parallel) to (z_parallel, x)
            if args.hermes_parallel_axis != "y":
                raise ValueError("Hermes shifted-metric transform expects hermes-parallel-axis=y.")
            zshift_aligned = zshift.T
            ny_binormal = hermes_fields_aligned["Ne"].shape[-1]
            if hermes_dz_mean is not None and hermes_dz_mean > 0:
                shift_idx = zshift_aligned / float(hermes_dz_mean)
            else:
                zperiod = hermes_sections.get("", {}).get("zperiod", None)
                if zperiod is None:
                    zperiod = 1.0
                zlength = 2.0 * np.pi / max(float(zperiod), 1e-12)
                shift_idx = zshift_aligned * float(ny_binormal) / zlength
            shift_idx = float(args.hermes_shifted_sign) * shift_idx
            for key, arr in hermes_fields_aligned.items():
                if arr.ndim == 4:
                    hermes_fields_aligned[key] = _apply_shifted_binormal(arr, shift_idx=shift_idx)
        if args.strict_axis and axis_map.endswith("unmatched"):
            raise ValueError(f"Axis mapping failed with strict axis mode: {axis_map}")
        aa_e = hermes_sections.get("e", {}).get("AA", 1.0 / 1836.0)
        aa_i = hermes_sections.get("d+", {}).get("AA", 1.0)
        density_floor = hermes_sections.get("evolve_momentum", {}).get(
            "density_floor",
            hermes_sections.get("e", {}).get("density_floor", 1e-7),
        )

        if (
            "Ve" not in hermes_fields_aligned
            and "NVe" in hermes_fields_aligned
            and "Ne" in hermes_fields_aligned
        ):
            Ne = hermes_fields_aligned["Ne"]
            Nlim = np.maximum(Ne, density_floor)
            hermes_fields_aligned["Ve"] = hermes_fields_aligned["NVe"] / (aa_e * Nlim)
        if (
            bool(getattr(built.system.params, "hot_ion_on", False))
            and "Pd+" in hermes_fields_aligned
            and "Nd+" in hermes_fields_aligned
        ):
            hermes_fields_aligned["Ti"] = hermes_fields_aligned["Pd+"] / np.maximum(
                hermes_fields_aligned["Nd+"], 1e-12
            )
        if "Vd+" in hermes_fields_aligned:
            hermes_fields_aligned["Vi"] = hermes_fields_aligned["Vd+"]
        elif "NVd+" in hermes_fields_aligned and "Nd+" in hermes_fields_aligned:
            Nd = hermes_fields_aligned["Nd+"]
            Nlim = np.maximum(Nd, density_floor)
            hermes_fields_aligned["Vi"] = hermes_fields_aligned["NVd+"] / (aa_i * Nlim)
        snapshots["n"] = np.asarray(hermes_fields_aligned["Ne"], dtype=np.float64)
        snapshots["omega"] = np.asarray(hermes_fields_aligned["Vort"], dtype=np.float64)
        # In jax_drb, `source_Te_is_pressure` only changes source-unit conversion:
        # the evolved field remains Te. Do not substitute Pe into state.Te.
        te_src = "Te" if "Te" in hermes_fields_aligned else "Pe"
        snapshots["Te"] = np.asarray(hermes_fields_aligned[te_src], dtype=np.float64)
        snapshots["phi"] = np.asarray(hermes_fields_aligned["phi"], dtype=np.float64)
        snapshots["vpar_e"] = np.asarray(
            hermes_fields_aligned.get("Ve", np.zeros_like(snapshots["n"])), dtype=np.float64
        )
        snapshots["vpar_i"] = np.asarray(
            hermes_fields_aligned.get("Vi", np.zeros_like(snapshots["n"])), dtype=np.float64
        )
        if (
            bool(getattr(built.system.params, "hot_ion_on", False))
            and "Ti" in hermes_fields_aligned
        ):
            snapshots["Ti"] = np.asarray(hermes_fields_aligned["Ti"], dtype=np.float64)
        times = hermes_times[: snapshots["n"].shape[0]]
    else:
        snapshot_fields = _build_snapshot_fields(built.state)
        cfg["time"]["snapshot_fields"] = snapshot_fields

        run = run_simulation(cfg, as_numpy=True)
        times = np.asarray(run.times, dtype=np.float64)

        for name in snapshot_fields:
            key = f"snapshots_{name}"
            if key in run.diagnostics:
                snapshots[name] = np.asarray(run.diagnostics[key], dtype=np.float64)
            else:
                raise KeyError(f"Missing {key} in diagnostics.")

        hermes_fields_aligned, axis_map = _align_hermes_fields(
            hermes_fields,
            snapshots["n"].shape[1:],
            hermes_parallel_axis=args.hermes_parallel_axis,
        )
        if args.hermes_shifted:
            if not args.hermes_grid:
                raise ValueError("--hermes-shifted requires --hermes-grid to load zShift.")
            zshift = _load_hermes_zshift(
                Path(args.hermes_grid) if args.hermes_grid else None,
                hermes_meta,
                data_dir=hermes_dir,
                pattern=args.hermes_pattern,
            )
            if args.hermes_parallel_axis != "y":
                raise ValueError("Hermes shifted-metric transform expects hermes-parallel-axis=y.")
            zshift_aligned = zshift.T
            ny_binormal = hermes_fields_aligned["Ne"].shape[-1]
            if hermes_dz_mean is not None and hermes_dz_mean > 0:
                shift_idx = zshift_aligned / float(hermes_dz_mean)
            else:
                zperiod = hermes_sections.get("", {}).get("zperiod", None)
                if zperiod is None:
                    zperiod = 1.0
                zlength = 2.0 * np.pi / max(float(zperiod), 1e-12)
                shift_idx = zshift_aligned * float(ny_binormal) / zlength
            shift_idx = float(args.hermes_shifted_sign) * shift_idx
            for key, arr in hermes_fields_aligned.items():
                if arr.ndim == 4:
                    hermes_fields_aligned[key] = _apply_shifted_binormal(arr, shift_idx=shift_idx)
        if args.strict_axis and axis_map.endswith("unmatched"):
            raise ValueError(f"Axis mapping failed with strict axis mode: {axis_map}")

    phi_override = snapshots["phi"] if bool(args.use_hermes_phi_in_terms) else None
    term_rows, total_rows, total_fields, phi_rows = _compute_term_metrics(
        built.system,
        snapshots,
        times,
        start_index=start_index,
        nsteps=int(args.nsteps),
        phi_override=phi_override,
    )
    if args.debug and phi_rows:
        print(f"[audit] phi row[0]={phi_rows[0]}")
    hermes_ddt = _compute_hermes_ddt(
        hermes_times,
        hermes_fields_aligned,
        int(args.nsteps),
        start_index=start_index,
        step_offset=int(args.hermes_step_offset),
    )
    hermes_time_unit = _hermes_time_unit_from_dumps(hermes_dir, args.hermes_pattern)
    if hermes_time_unit is None:
        hermes_time_unit = _hermes_time_unit(hermes_sections)
    jax_time_unit = None if built.normalization is None else float(built.normalization.time)
    ddt_scale = 1.0
    if (not args.no_time_scale) and hermes_time_unit and jax_time_unit:
        ddt_scale = jax_time_unit / hermes_time_unit
        for key in hermes_ddt:
            hermes_ddt[key] = hermes_ddt[key] * ddt_scale
    if args.debug:
        print(f"[audit] use_hermes_state={args.use_hermes_state}")
        geom_cfg = cfg.get("geometry", {})
        if isinstance(geom_cfg, dict):
            print(f"[audit] coeff_path={geom_cfg.get('coeff_path')}")
        if args.use_hermes_state:
            n0 = snapshots["n"][start_index]
            Te0 = snapshots["Te"][start_index]
            print(
                "[audit] snapshot rms n={:.3e} Te={:.3e} omega={:.3e}".format(
                    float(np.sqrt(np.mean(n0 * n0))),
                    float(np.sqrt(np.mean(Te0 * Te0))),
                    float(np.sqrt(np.mean(snapshots["omega"][start_index] ** 2))),
                )
            )
        print(
            "[audit] params poisson_metric_on={} poisson_b_weighted={} "
            "poisson_b_weighted_mode={} poisson_scale={}".format(
                getattr(built.system.params, "poisson_metric_on", None),
                getattr(built.system.params, "poisson_b_weighted", None),
                getattr(built.system.params, "poisson_b_weighted_mode", None),
                getattr(built.system.params, "poisson_scale", None),
            )
        )
        print(
            "[audit] params log_n={} log_Te={} hot_ion_on={}".format(
                getattr(built.system.params, "log_n", None),
                getattr(built.system.params, "log_Te", None),
                getattr(built.system.params, "hot_ion_on", None),
            )
        )
        print(
            "[audit] params me_hat={} average_atomic_mass={}".format(
                getattr(built.system.params, "me_hat", None),
                getattr(built.system.params, "average_atomic_mass", None),
            )
        )
        print(
            "[audit] params dpar_factor_scale={}".format(
                getattr(built.system.params, "dpar_factor_scale", None)
            )
        )
    hermes_term_rows = _compute_hermes_term_metrics(
        hermes_fields_aligned,
        hermes_times,
        start_index=start_index,
        nsteps=int(args.nsteps),
        ddt_scale=ddt_scale,
        step_offset=int(args.hermes_step_offset),
        axis_map=axis_map,
        hermes_parallel_axis=args.hermes_parallel_axis,
    )
    term_mismatch_rows = _compute_term_mismatch(term_rows, hermes_term_rows)
    field_map = {"n": "Ne", "Te": "Te", "omega": "Vort", "phi": "phi"}

    mismatch_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    for field_name, hermes_name in field_map.items():
        if field_name not in total_fields:
            continue
        jax_list = total_fields[field_name]
        for ti in range(min(args.nsteps, len(jax_list))):
            hermes_arr = hermes_ddt[hermes_name][ti]
            jax_arr = jax_list[ti]
            hermes_rms = float(np.sqrt(np.mean(hermes_arr * hermes_arr)))
            if args.strict_axis:
                if jax_arr.shape != hermes_arr.shape:
                    raise ValueError(
                        f"Strict axis mismatch for field {field_name}: "
                        f"jax {jax_arr.shape} vs hermes {hermes_arr.shape}"
                    )
            else:
                try:
                    jax_arr, hermes_arr = _crop_pair(jax_arr, hermes_arr)
                except ValueError:
                    continue
            jax_rms = float(np.sqrt(np.mean(jax_arr * jax_arr)))
            diff_rms = float(np.sqrt(np.mean((jax_arr - hermes_arr) ** 2)))
            denom = max(1e-12, 0.1 * hermes_rms)
            mismatch_rows.append(
                {
                    "step": int(ti),
                    "t": float(times[start_index + ti]),
                    "field": field_name,
                    "jax_rhs_rms": jax_rms,
                    "hermes_ddt_rms": hermes_rms,
                    "rel_diff": diff_rms / denom,
                }
            )

    # Add field-level weighting to term mismatch rows so first-fail ranking
    # prioritizes terms that materially contribute to the field RHS.
    rhs_index = {
        (int(row["step"]), str(row["field"])): float(row["hermes_ddt_rms"]) for row in mismatch_rows
    }
    rhs_term_fallback: dict[tuple[int, str], float] = {}
    for row in hermes_term_rows:
        key = (int(row.get("step", 0)), str(row.get("field", "")))
        rhs_term_fallback[key] = max(rhs_term_fallback.get(key, 0.0), float(row.get("rms", 0.0)))
    for row in term_mismatch_rows:
        step = int(row.get("step", 0))
        field = str(row.get("field", ""))
        rhs = rhs_index.get((step, field), 0.0)
        rhs = max(rhs, rhs_term_fallback.get((step, field), 0.0))
        term_rms = float(row.get("hermes_rms", 0.0))
        frac = term_rms / max(rhs, 1e-30)
        row["field_rhs_rms"] = rhs
        row["frac_of_field_rhs"] = frac
        row["weighted_rel"] = float(row.get("rel_diff", 0.0)) * frac

    out_dir = Path(args.out_dir)
    _write_csv(
        out_dir / "jax_term_contributions.csv",
        term_rows,
        ["step", "t", "term", "field", "rms", "maxabs", "mean"],
    )
    _write_csv(
        out_dir / "jax_total_rhs.csv",
        total_rows,
        ["step", "t", "field", "rhs_rms", "rhs_maxabs"],
    )
    _write_csv(
        out_dir / "jax_phi_stats.csv",
        phi_rows,
        ["step", "t", "phi_rms", "phi_maxabs", "phi_iters"],
    )
    _write_csv(
        out_dir / "hermes_ddt_rms.csv",
        mismatch_rows,
        ["step", "t", "field", "jax_rhs_rms", "hermes_ddt_rms", "rel_diff"],
    )
    _write_csv(
        out_dir / "hermes_term_contributions.csv",
        hermes_term_rows,
        ["step", "t", "term", "field", "rms", "maxabs", "mean"],
    )
    _write_csv(
        out_dir / "term_mismatch.csv",
        term_mismatch_rows,
        [
            "step",
            "t",
            "field",
            "term",
            "hermes_term",
            "jax_rms",
            "hermes_rms",
            "rel_diff",
            "field_rhs_rms",
            "frac_of_field_rhs",
            "weighted_rel",
        ],
    )
    if "phi" in snapshots and "omega" in snapshots:
        poisson_parity_rows = _compute_poisson_parity_metrics(
            built.system, snapshots, times, start_index=start_index, nsteps=int(args.nsteps)
        )
    else:
        poisson_parity_rows = []
    _write_csv(
        out_dir / "poisson_parity.csv",
        poisson_parity_rows,
        [
            "step",
            "t",
            "crop_x",
            "omega_ref_rms",
            "omega_from_phi_rms",
            "omega_from_phi_corr",
            "omega_from_phi_scale",
            "omega_ref_rms_core",
            "omega_from_phi_rms_core",
            "omega_from_phi_corr_core",
            "omega_from_phi_scale_core",
            "phi_ref_rms",
            "phi_from_omega_rms",
            "phi_from_omega_corr",
            "phi_from_omega_scale",
            "phi_ref_rms_core",
            "phi_from_omega_rms_core",
            "phi_from_omega_corr_core",
            "phi_from_omega_scale_core",
        ],
    )
    fail_fields = _split_csv_set(args.fail_fast_fields)
    fail_terms = _split_csv_set(args.fail_fast_terms)
    first_fail, first_step_rows = _first_failing_term(
        term_mismatch_rows, fields=fail_fields, terms=fail_terms
    )
    fail_summary_rows: list[dict[str, object]] = []
    if first_step_rows:
        for rank, row in enumerate(first_step_rows, start=1):
            fail_summary_rows.append(
                {
                    "rank": rank,
                    "step": int(row.get("step", 0)),
                    "t": float(row.get("t", 0.0)),
                    "field": str(row.get("field", "")),
                    "term": str(row.get("term", "")),
                    "hermes_term": str(row.get("hermes_term", "")),
                    "jax_rms": float(row.get("jax_rms", 0.0)),
                    "hermes_rms": float(row.get("hermes_rms", 0.0)),
                    "rel_diff": float(row.get("rel_diff", 0.0)),
                    "field_rhs_rms": float(row.get("field_rhs_rms", 0.0)),
                    "frac_of_field_rhs": float(row.get("frac_of_field_rhs", 0.0)),
                    "weighted_rel": float(row.get("weighted_rel", 0.0)),
                }
            )
    _write_csv(
        out_dir / "first_failing_terms.csv",
        fail_summary_rows,
        [
            "rank",
            "step",
            "t",
            "field",
            "term",
            "hermes_term",
            "jax_rms",
            "hermes_rms",
            "rel_diff",
            "field_rhs_rms",
            "frac_of_field_rhs",
            "weighted_rel",
        ],
    )

    summary = {
        "jax_config": str(cfg_path),
        "hermes_data_dir": str(hermes_dir),
        "jax_normalization": None if built.normalization is None else asdict(built.normalization),
        "jax_geometry": _jax_geom_stats(built.system),
        "jax_shape": list(built.system.geom.shape()),
        "jax_snapshot_shape": list(snapshots["n"].shape),
        "audit_start_index": int(start_index),
        "audit_time_target": None if args.time_target is None else float(args.time_target),
        "nsteps": int(args.nsteps),
        "match_hermes_dt": bool(args.match_hermes_dt),
        "hermes_time_unit": None if hermes_time_unit is None else float(hermes_time_unit),
        "jax_time_unit": None if jax_time_unit is None else float(jax_time_unit),
        "hermes_ddt_scale": float(ddt_scale),
        "jax_time": {
            "dt": float(cfg["time"].get("dt", 0.0)),
            "nsteps": int(cfg["time"].get("nsteps", 0)),
            "save_every": int(cfg["time"].get("save_every", 1)),
            "method": str(
                cfg["time"].get("method", cfg.get("integrator", {}).get("method", "diffrax"))
            ),
            "solver": str(cfg["time"].get("solver", "")),
            "rtol": float(cfg["time"].get("rtol", 0.0)),
            "atol": float(cfg["time"].get("atol", 0.0)),
        },
        "hermes_meta": hermes_meta,
        "hermes_dt": dt_hermes,
        "axis_mapping": axis_map,
        "fail_fast": {
            "fields": sorted(fail_fields),
            "terms": sorted(fail_terms),
            "threshold": float(args.fail_fast_rel_diff),
            "first_failing_term": first_fail,
        },
        "hermes_step_offset": int(args.hermes_step_offset),
        "use_hermes_phi_in_terms": bool(args.use_hermes_phi_in_terms),
    }

    hermes_inp = Path(args.hermes_input) if args.hermes_input else None
    if hermes_inp:
        summary["hermes_input"] = _parse_bout_input(hermes_inp)

    try:
        from netCDF4 import Dataset

        files = sorted(hermes_dir.glob(args.hermes_pattern))
        with Dataset(str(files[0])) as ds0:
            summary["hermes_geometry"] = _hermes_geom_stats(ds0)
    except Exception:
        summary["hermes_geometry"] = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    if args.dump_term_arrays:
        dump_dir = out_dir / "term_arrays"
        dump_dir.mkdir(parents=True, exist_ok=True)
        term_filter = [t.strip() for t in str(args.dump_terms).split(",") if t.strip()]
        dump_all_terms = len(term_filter) == 1 and term_filter[0].lower() == "all"
        field_filter = [f.strip() for f in str(args.dump_fields).split(",") if f.strip()]
        max_dump_steps = min(args.nsteps, snapshots["n"].shape[0] - start_index)
        for ti in range(max(0, max_dump_steps)):
            idx = int(start_index + ti)
            y = snapshots
            data: dict[str, np.ndarray] = {}
            from jaxdrb.core.state import DRBSystemState

            state = DRBSystemState(
                n=y["n"][idx],
                omega=y["omega"][idx],
                vpar_e=y["vpar_e"][idx],
                vpar_i=y["vpar_i"][idx],
                Te=y["Te"][idx],
                Ti=None if "Ti" not in y else y["Ti"][idx],
                psi=None if "psi" not in y else y["psi"][idx],
                N=None if "N" not in y else y["N"][idx],
            )
            if str(getattr(built.system, "engine", "unified")).lower() == "parity_fv":
                phi_arg = None
                if bool(args.use_hermes_phi_in_terms) and "phi" in y:
                    phi_arg = jnp.asarray(y["phi"][idx], dtype=state.n.dtype)
                split, term_map, phi_local, _ = built.system.rhs_terms(
                    float(times[idx]), state, phi_override=phi_arg
                )
                total = split.total()
                data["phi"] = np.asarray(phi_local)
                pe_adv_override = None
            else:
                from jaxdrb.core.terms import build_context
                import equinox as eqx

                ctx = build_context(built.system.params, built.system.geom, state)
                if bool(args.use_hermes_phi_in_terms) and "phi" in y:
                    ctx = eqx.tree_at(
                        lambda c: c.phi,
                        ctx,
                        jnp.asarray(y["phi"][idx], dtype=ctx.phi.dtype),
                    )
                split, term_map = built.system.scheduler.run_with_terms(ctx, state)
                total = split.total()
                data["phi"] = np.asarray(ctx.phi)
                pe_adv_override = None
                adv_form = str(getattr(built.system.params, "exb_advection_form", "flux")).lower()
                if adv_form == "flux" and hasattr(ctx.geom, "exb_flux_divergence"):
                    try:
                        pe_adv_override = -ctx.geom.exb_flux_divergence(
                            ctx.phi, ctx.n_phys * ctx.Te_phys, bc_phi=ctx.bcs.phi, bc_adv=ctx.bcs.Te
                        )
                    except Exception:
                        pe_adv_override = None
                elif hasattr(ctx.geom, "bracket"):
                    try:
                        pe_adv_override = -ctx.geom.bracket(
                            ctx.phi, ctx.n_phys * ctx.Te_phys, bc_phi=ctx.bcs.phi, bc_f=ctx.bcs.Te
                        )
                    except Exception:
                        pe_adv_override = None

            for field in field_filter:
                arr = getattr(total, field, None)
                if arr is not None:
                    arr_np = np.asarray(arr)
                hermes_key = field_map.get(field)
                if hermes_key is not None and hermes_key in hermes_ddt:
                    hermes_arr = hermes_ddt[hermes_key][ti]
                    if args.strict_axis:
                        if arr is not None:
                            try:
                                arr_np = _align_to_shape_strict(arr_np, hermes_arr.shape)
                            except ValueError as exc:
                                raise ValueError(
                                    f"Strict axis mismatch for field {field}: "
                                    f"jax {arr_np.shape} vs hermes {hermes_arr.shape}"
                                ) from exc
                            data[f"total_{field}"] = arr_np
                        data[f"hermes_ddt_{field}"] = np.asarray(hermes_arr)
                    else:
                        if arr is not None:
                            jax_arr, hermes_arr = _crop_pair(arr_np, hermes_arr)
                            data[f"total_{field}"] = jax_arr
                        data[f"hermes_ddt_{field}"] = hermes_arr

            for name, term in term_map.items():
                if (not dump_all_terms) and (name not in term_filter):
                    continue
                pe_term = None
                if "n" in y and "Te" in y:
                    n_snap = y["n"][idx]
                    Te_snap = y["Te"][idx]
                    pe_term = n_snap * term.Te + Te_snap * term.n
                if name == "advection" and pe_adv_override is not None:
                    pe_term = pe_adv_override
                for field in field_filter:
                    arr = getattr(term, field, None)
                    if arr is not None:
                        data[f"{name}_{field}"] = np.asarray(arr)
                if pe_term is not None:
                    data[f"{name}_Pe"] = np.asarray(pe_term)

            for name, arr in hermes_fields_aligned.items():
                if not name.startswith("term_"):
                    continue
                hidx = int(start_index + ti + int(args.hermes_step_offset))
                if hidx < 0 or hidx >= arr.shape[0]:
                    continue
                data[f"hermes_{name}"] = np.asarray(arr[hidx] * ddt_scale)
            np.savez(dump_dir / f"step_{ti:04d}.npz", **data)

            for field in field_filter:
                hermes_key = field_map.get(field)
                if hermes_key is None or hermes_key not in hermes_ddt:
                    continue
                hermes_arr = hermes_ddt[hermes_key][ti]
                total_arr = getattr(total, field, None)
                if total_arr is None:
                    continue
                jax_total = np.asarray(total_arr)
                if args.strict_axis:
                    try:
                        jax_total_c = _align_to_shape_strict(jax_total, hermes_arr.shape)
                    except ValueError as exc:
                        raise ValueError(
                            f"Strict axis mismatch for field {field}: "
                            f"jax {jax_total.shape} vs hermes {hermes_arr.shape}"
                        ) from exc
                    hermes_arr_c = hermes_arr
                else:
                    try:
                        jax_total_c, hermes_arr_c = _crop_pair(jax_total, hermes_arr)
                    except ValueError:
                        continue
                mismatch = hermes_arr_c - jax_total_c
                mismatch_rms = float(np.sqrt(np.mean(mismatch * mismatch)))
                for name, term in term_map.items():
                    arr = getattr(term, field, None)
                    if arr is None:
                        continue
                    jax_term = np.asarray(arr)
                    if args.strict_axis:
                        try:
                            jax_term_c = _align_to_shape_strict(jax_term, hermes_arr_c.shape)
                        except ValueError as exc:
                            raise ValueError(
                                f"Strict axis mismatch for term {name} field {field}: "
                                f"jax {jax_term.shape} vs hermes {hermes_arr_c.shape}"
                            ) from exc
                    else:
                        try:
                            jax_term_c, _ = _crop_pair(jax_term, hermes_arr_c)
                        except ValueError:
                            continue
                    term_rms = float(np.sqrt(np.mean(jax_term_c * jax_term_c)))
                    if term_rms <= 0.0 or mismatch_rms <= 0.0:
                        corr = 0.0
                        scale = 0.0
                        mismatch_after = mismatch_rms
                    else:
                        dot = float(np.mean(jax_term_c * mismatch))
                        corr = dot / (term_rms * mismatch_rms)
                        scale = dot / max(term_rms * term_rms, 1e-30)
                        mismatch_after = float(
                            np.sqrt(np.mean((mismatch - scale * jax_term_c) ** 2))
                        )
                    projection_rows.append(
                        {
                            "t": float(times[ti]),
                            "field": field,
                            "term": name,
                            "term_rms": term_rms,
                            "mismatch_rms": mismatch_rms,
                            "corr": corr,
                            "scale": scale,
                            "mismatch_rms_after": mismatch_after,
                        }
                    )

    if projection_rows:
        _write_csv(
            out_dir / "term_mismatch_projection.csv",
            projection_rows,
            [
                "t",
                "field",
                "term",
                "term_rms",
                "mismatch_rms",
                "corr",
                "scale",
                "mismatch_rms_after",
            ],
        )

    print(f"Wrote audit outputs to {out_dir}")

    if args.fail_fast_rel_diff >= 0.0 and first_fail is not None:
        rel = float(first_fail.get("rel_diff", 0.0))
        if rel > float(args.fail_fast_rel_diff):
            raise SystemExit(
                "fail-fast: first failing term "
                f"(step={int(first_fail.get('step', 0))}, "
                f"field={first_fail.get('field')}, term={first_fail.get('term')}) "
                f"rel_diff={rel:.6g} > threshold={args.fail_fast_rel_diff:.6g}"
            )

    if args.hermes_grid:
        try:
            from netCDF4 import Dataset

            with Dataset(str(args.hermes_grid)) as ds:
                dx = np.asarray(ds.variables["dx"][:], dtype=np.float64)
                dy = np.asarray(ds.variables["dy"][:], dtype=np.float64)
            hermes_x = _coord_from_spacing(dx, axis=0)
            hermes_y = _coord_from_spacing(dy, axis=1)
            jax_grid = getattr(built.system.geom, "grid", None)
            jax_x = None
            jax_y = None
            if jax_grid is not None:
                perp = getattr(jax_grid, "perp", None)
                if perp is not None:
                    jax_x = np.asarray(perp.x, dtype=np.float64)
                    jax_y = np.asarray(perp.y, dtype=np.float64)
            np.savez(
                out_dir / "axis_mapping.npz",
                hermes_x=hermes_x,
                hermes_y=hermes_y,
                jax_x=jax_x,
                jax_y=jax_y,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
