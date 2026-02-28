from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import tomllib

import numpy as np

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


def _read_var_interior(
    ds, name: str, mxg: int, myg: int, mxsub: int, mysub: int
) -> np.ndarray:
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
            if name not in ds.variables:
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
                sub = _read_var_interior(ds, name, mxg, myg, mxsub, mysub)
                fields[name][:, x0:x1, y0:y1, ...] = sub
            if "Te" not in ds.variables and "Pe" in ds.variables:
                fields["Pe"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, "Pe", mxg, myg, mxsub, mysub
                )

    if "Te" not in fields and "Pe" in fields and "Ne" in fields:
        fields["Te"] = fields["Pe"] / np.maximum(fields["Ne"], 1e-12)
        fields.pop("Pe", None)
    if "Ve" not in fields and "NVe" in fields and "Ne" in fields:
        fields["Ve"] = fields["NVe"] / np.maximum(fields["Ne"], 1e-12)

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
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return float(node.n)
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


def _compute_term_metrics(system, snapshots: dict[str, np.ndarray], times: np.ndarray):
    from jaxdrb.core.state import DRBSystemState
    from jaxdrb.core.terms import build_context

    out_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    total_fields: dict[str, list[np.ndarray]] = {}
    phi_rows: list[dict[str, object]] = []
    nsteps = snapshots["n"].shape[0]

    for ti in range(nsteps):
        y = DRBSystemState(
            n=snapshots["n"][ti],
            omega=snapshots["omega"][ti],
            vpar_e=snapshots["vpar_e"][ti],
            vpar_i=snapshots["vpar_i"][ti],
            Te=snapshots["Te"][ti],
            Ti=None if "Ti" not in snapshots else snapshots["Ti"][ti],
            psi=None if "psi" not in snapshots else snapshots["psi"][ti],
            N=None if "N" not in snapshots else snapshots["N"][ti],
        )
        ctx = build_context(system.params, system.geom, y, return_phi_iters=True)
        split, term_map = system.scheduler.run_with_terms(ctx, y)
        total = split.total()
        phi = np.asarray(ctx.phi)
        phi_rows.append(
            {
                "t": float(times[ti]),
                "phi_rms": float(np.sqrt(np.mean(phi * phi))),
                "phi_maxabs": float(np.max(np.abs(phi))),
                "phi_iters": None if ctx.phi_iters is None else float(np.asarray(ctx.phi_iters)),
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
                    "t": float(times[ti]),
                    "field": field_name,
                    "rhs_rms": float(np.sqrt(np.mean(a * a))),
                    "rhs_maxabs": float(np.max(np.abs(a))),
                }
            )
        for name, term in term_map.items():
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
                        "t": float(times[ti]),
                        "term": name,
                        "field": field_name,
                        "rms": float(np.sqrt(np.mean(a * a))),
                        "maxabs": float(np.max(np.abs(a))),
                        "mean": float(np.mean(a)),
                    }
                )

    return out_rows, total_rows, total_fields, phi_rows


def _compute_hermes_ddt(
    times: np.ndarray, fields: dict[str, np.ndarray], nsteps: int
) -> dict[str, np.ndarray]:
    ddt: dict[str, np.ndarray] = {}

    def _slice_steps(arr: np.ndarray) -> np.ndarray:
        return arr[:nsteps]

    # Prefer direct ddt variables if present (interior values are already extracted).
    if "ddt(Ne)" in fields:
        ddt["Ne"] = _slice_steps(fields["ddt(Ne)"])
    if "ddt(Pe)" in fields and "Ne" in fields:
        dpe = _slice_steps(fields["ddt(Pe)"])
        n = _slice_steps(fields["Ne"])
        if "Te" in fields:
            Te = _slice_steps(fields["Te"])
        elif "Pe" in fields:
            Te = _slice_steps(fields["Pe"] / np.maximum(fields["Ne"], 1e-12))
        else:
            Te = None
        if Te is not None:
            ddt["Te"] = (dpe - Te * ddt.get("Ne", 0.0)) / np.maximum(n, 1e-12)

    if "ddt(Vort)" in fields:
        ddt["Vort"] = _slice_steps(fields["ddt(Vort)"])

    # Fallback to finite differences for missing fields.
    dt = np.diff(times[: nsteps + 1])
    dt = dt.reshape(-1, *([1] * (fields[next(iter(fields))].ndim - 1)))
    for name in ("Ne", "Te", "Vort", "phi"):
        if name in ddt:
            continue
        if name not in fields:
            continue
        arr = fields[name]
        if arr.shape[0] < nsteps + 1:
            raise ValueError(f"Need at least {nsteps + 1} Hermes frames for {name}.")
        ddt[name] = (arr[1 : nsteps + 1] - arr[:nsteps]) / dt
    return ddt


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


def _apply_shifted_binormal(
    arr: np.ndarray, *, shift_idx: np.ndarray
) -> np.ndarray:
    """Shift field along binormal (last axis) by shift_idx (index units)."""
    if arr.ndim != 4:
        return arr
    nz, nx, ny = arr.shape[1:]
    if shift_idx.shape != (nz, nx):
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
    hermes_dy_mean: float | None = None,
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
        if hermes_dx_mean is not None and hermes_dy_mean is not None:
            Lx = float(hermes_dx_mean * nx)
            Ly = float(hermes_dy_mean * ny)
        else:
            x_slice = slice(mxg, mxg + nx)
            y_slice = slice(myg, myg + ny)
            dx_int = dx[x_slice, y_slice]
            dy_int = dy[x_slice, y_slice]
            Lx = float(np.mean(dx_int, axis=1).sum())
            Ly = float(np.mean(dy_int, axis=0).sum())
    else:
        nx = nx_full
        ny = ny_full
        Lx = float(dx.mean() * nx_full)
        Ly = float(dy.mean() * ny_full)

    geom = dict(cfg.get("geometry", {}))
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


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        default="z",
        choices=("y", "z"),
        help="Which Hermes axis corresponds to parallel direction (y or z).",
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
        "--use-hermes-state",
        action="store_true",
        help="Evaluate JAX RHS directly on Hermes snapshots (skip JAX time integration).",
    )
    parser.add_argument(
        "--no-time-scale",
        action="store_true",
        help="Disable scaling Hermes ddt into JAX time units.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.jax_config)
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    hermes_dir = Path(args.hermes_data_dir)
    hermes_input_path = Path(args.hermes_input) if args.hermes_input else None
    if hermes_input_path is None:
        candidate = hermes_dir / "BOUT.inp"
        if candidate.exists():
            hermes_input_path = candidate
    hermes_sections = _parse_bout_sections(hermes_input_path)
    hermes_times, hermes_fields, hermes_meta = _load_hermes_fields(
        hermes_dir,
        (
            "Ne",
            "Te",
            "Vort",
            "phi",
            "ddt(Ne)",
            "ddt(Pe)",
            "Ve",
            "NVe",
            "Nd+",
            "Pd+",
            "NVd+",
        ),
        args.hermes_pattern,
    )
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
        cfg = _override_geometry_from_hermes(
            cfg,
            Path(args.hermes_grid),
            hermes_meta=hermes_meta,
            hermes_dx_mean=hermes_dx_mean,
            hermes_dy_mean=hermes_dy_mean,
        )
    dt_hermes = float(np.mean(np.diff(hermes_times[: max(2, args.nsteps + 1)])))

    time_cfg = cfg.get("time", {})
    time_cfg = dict(time_cfg)
    time_cfg["nsteps"] = int(args.nsteps)
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
        density_floor = (
            hermes_sections.get("evolve_momentum", {}).get(
                "density_floor",
                hermes_sections.get("e", {}).get("density_floor", 1e-7),
            )
        )

        if "NVe" in hermes_fields_aligned and "Ne" in hermes_fields_aligned:
            Ne = hermes_fields_aligned["Ne"]
            Nlim = np.maximum(Ne, density_floor)
            hermes_fields_aligned["Ve"] = hermes_fields_aligned["NVe"] / (
                aa_e * Nlim
            )
        if "Pd+" in hermes_fields_aligned and "Nd+" in hermes_fields_aligned:
            hermes_fields_aligned["Ti"] = hermes_fields_aligned["Pd+"] / np.maximum(
                hermes_fields_aligned["Nd+"], 1e-12
            )
        if "NVd+" in hermes_fields_aligned and "Nd+" in hermes_fields_aligned:
            Nd = hermes_fields_aligned["Nd+"]
            Nlim = np.maximum(Nd, density_floor)
            hermes_fields_aligned["Vi"] = hermes_fields_aligned["NVd+"] / (
                aa_i * Nlim
            )
        snapshots["n"] = np.asarray(hermes_fields_aligned["Ne"], dtype=np.float64)
        snapshots["omega"] = np.asarray(hermes_fields_aligned["Vort"], dtype=np.float64)
        snapshots["Te"] = np.asarray(hermes_fields_aligned["Te"], dtype=np.float64)
        snapshots["phi"] = np.asarray(hermes_fields_aligned["phi"], dtype=np.float64)
        snapshots["vpar_e"] = np.asarray(
            hermes_fields_aligned.get("Ve", np.zeros_like(snapshots["n"])), dtype=np.float64
        )
        snapshots["vpar_i"] = np.asarray(
            hermes_fields_aligned.get("Vi", np.zeros_like(snapshots["n"])), dtype=np.float64
        )
        if "Ti" in hermes_fields_aligned:
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

    term_rows, total_rows, total_fields, phi_rows = _compute_term_metrics(
        built.system, snapshots, times
    )
    hermes_ddt = _compute_hermes_ddt(hermes_times, hermes_fields_aligned, args.nsteps)
    hermes_time_unit = _hermes_time_unit(hermes_sections)
    jax_time_unit = None if built.normalization is None else float(built.normalization.time)
    ddt_scale = 1.0
    if (not args.no_time_scale) and hermes_time_unit and jax_time_unit:
        ddt_scale = jax_time_unit / hermes_time_unit
        for key in hermes_ddt:
            hermes_ddt[key] = hermes_ddt[key] * ddt_scale
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
                    "t": float(times[ti]),
                    "field": field_name,
                    "jax_rhs_rms": jax_rms,
                    "hermes_ddt_rms": hermes_rms,
                    "rel_diff": diff_rms / denom,
                }
            )

    out_dir = Path(args.out_dir)
    _write_csv(
        out_dir / "jax_term_contributions.csv",
        term_rows,
        ["t", "term", "field", "rms", "maxabs", "mean"],
    )
    _write_csv(
        out_dir / "jax_total_rhs.csv",
        total_rows,
        ["t", "field", "rhs_rms", "rhs_maxabs"],
    )
    _write_csv(
        out_dir / "jax_phi_stats.csv",
        phi_rows,
        ["t", "phi_rms", "phi_maxabs", "phi_iters"],
    )
    _write_csv(
        out_dir / "hermes_ddt_rms.csv",
        mismatch_rows,
        ["t", "field", "jax_rhs_rms", "hermes_ddt_rms", "rel_diff"],
    )

    summary = {
        "jax_normalization": None if built.normalization is None else asdict(built.normalization),
        "jax_geometry": _jax_geom_stats(built.system),
        "jax_shape": list(built.system.geom.shape()),
        "jax_snapshot_shape": list(snapshots["n"].shape),
        "hermes_time_unit": None if hermes_time_unit is None else float(hermes_time_unit),
        "jax_time_unit": None if jax_time_unit is None else float(jax_time_unit),
        "hermes_ddt_scale": float(ddt_scale),
        "jax_time": {
            "dt": float(cfg["time"].get("dt", 0.0)),
            "nsteps": int(cfg["time"].get("nsteps", 0)),
            "save_every": int(cfg["time"].get("save_every", 1)),
            "method": str(cfg["time"].get("method", cfg.get("integrator", {}).get("method", "diffrax"))),
            "solver": str(cfg["time"].get("solver", "")),
            "rtol": float(cfg["time"].get("rtol", 0.0)),
            "atol": float(cfg["time"].get("atol", 0.0)),
        },
        "hermes_meta": hermes_meta,
        "hermes_dt": dt_hermes,
        "axis_mapping": axis_map,
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
        for ti in range(min(args.nsteps, snapshots["n"].shape[0])):
            y = snapshots
            data: dict[str, np.ndarray] = {}
            from jaxdrb.core.state import DRBSystemState
            from jaxdrb.core.terms import build_context

            state = DRBSystemState(
                n=y["n"][ti],
                omega=y["omega"][ti],
                vpar_e=y["vpar_e"][ti],
                vpar_i=y["vpar_i"][ti],
                Te=y["Te"][ti],
                Ti=None if "Ti" not in y else y["Ti"][ti],
                psi=None if "psi" not in y else y["psi"][ti],
                N=None if "N" not in y else y["N"][ti],
            )
            ctx = build_context(built.system.params, built.system.geom, state)
            split, term_map = built.system.scheduler.run_with_terms(ctx, state)
            total = split.total()
            data["phi"] = np.asarray(ctx.phi)

            for field in field_filter:
                arr = getattr(total, field, None)
                if arr is not None:
                    data[f"total_{field}"] = np.asarray(arr)
                hermes_key = field_map.get(field)
                if hermes_key is not None and hermes_key in hermes_ddt:
                    hermes_arr = hermes_ddt[hermes_key][ti]
                    if args.strict_axis:
                        if arr is not None and hermes_arr.shape != arr.shape:
                            raise ValueError(
                                f"Strict axis mismatch for field {field}: "
                                f"jax {arr.shape} vs hermes {hermes_arr.shape}"
                            )
                        data[f"hermes_ddt_{field}"] = np.asarray(hermes_arr)
                    else:
                        if arr is not None:
                            jax_arr, hermes_arr = _crop_pair(np.asarray(arr), hermes_arr)
                            data[f"total_{field}"] = jax_arr
                        data[f"hermes_ddt_{field}"] = hermes_arr

            for name, term in term_map.items():
                if (not dump_all_terms) and (name not in term_filter):
                    continue
                for field in field_filter:
                    arr = getattr(term, field, None)
                    if arr is not None:
                        data[f"{name}_{field}"] = np.asarray(arr)
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
                    if jax_total.shape != hermes_arr.shape:
                        raise ValueError(
                            f"Strict axis mismatch for field {field}: "
                            f"jax {jax_total.shape} vs hermes {hermes_arr.shape}"
                        )
                    jax_total_c = jax_total
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
                        if jax_term.shape != hermes_arr_c.shape:
                            raise ValueError(
                                f"Strict axis mismatch for term {name} field {field}: "
                                f"jax {jax_term.shape} vs hermes {hermes_arr_c.shape}"
                            )
                        jax_term_c = jax_term
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
