from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tomllib

import numpy as np


def _read_scalar(ds, names: tuple[str, ...], default: float | None = None) -> float:
    for n in names:
        if n in ds.variables:
            v = ds.variables[n][:]
            return float(np.asarray(v).reshape(-1)[0])
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
    data_dir: Path, names: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        from netCDF4 import Dataset
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required for Hermes mismatch tracing.") from e

    files = sorted(data_dir.glob("BOUT.dmp.*.nc"))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {data_dir}")

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

    has_te = False
    with Dataset(str(files[0])) as ds:
        for name in names:
            if name == "Te" and name not in ds.variables and "Pe" in ds.variables:
                continue
            if name not in ds.variables:
                raise KeyError(f"Hermes variable '{name}' not found in dump.")
            shape = _read_var_interior(ds, name, mxg, myg, mxsub, mysub).shape
            fields[name] = np.zeros((nt, nx, ny, *shape[3:]), dtype=np.float64)
        has_te = "Te" in ds.variables
        if not has_te and "Pe" in ds.variables:
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
                sub = _read_var_interior(ds, name, mxg, myg, mxsub, mysub)
                fields[name][:, x0:x1, y0:y1, ...] = sub
            if "Te" not in ds.variables and "Pe" in ds.variables:
                fields["Pe"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, "Pe", mxg, myg, mxsub, mysub
                )

    if "Te" not in fields and "Pe" in fields and "Ne" in fields:
        fields["Te"] = fields["Pe"] / np.maximum(fields["Ne"], 1e-12)
        fields.pop("Pe", None)

    return times, fields


def _load_jax_fields(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.load(path, allow_pickle=True)
    times = np.asarray(raw["times"] if "times" in raw else raw["t"], dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    mapping = {"n": "n", "Te": "Te", "omega": "omega", "phi": "phi"}
    for key, name in mapping.items():
        skey = f"snapshots_{name}"
        lkey = f"snapshot_{name}"
        if skey in raw:
            out[key] = np.asarray(raw[skey], dtype=np.float64)
        elif lkey in raw:
            out[key] = np.asarray(raw[lkey], dtype=np.float64)[None, ...]
        else:
            raise KeyError(f"Missing '{skey}'/'{lkey}' in jax npz.")
    return times, out


def _load_jax_rms(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.load(path, allow_pickle=True)
    times = np.asarray(raw["times"] if "times" in raw else raw["t"], dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for field in ("n", "Te", "omega", "phi"):
        key = f"rms_{field}_fluct"
        if key not in raw:
            raise KeyError(f"Missing '{key}' in jax npz.")
        out[field] = np.asarray(raw[key], dtype=np.float64)
    return times, out


def _interp_time_series(t_src: np.ndarray, arr_t: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    if arr_t.shape[0] == t_dst.size and np.allclose(t_src[: t_dst.size], t_dst):
        return arr_t[: t_dst.size]
    flat = arr_t.reshape(arr_t.shape[0], -1)
    out = np.empty((t_dst.size, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, flat[:, j])
    return out.reshape((t_dst.size, *arr_t.shape[1:]))


def _rel_l2_scalar(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale_floor = max(1e-12, 0.05 * float(np.max(np.abs(b))))
    return np.abs(a - b) / np.maximum(np.abs(b), scale_floor)


def _jax_initial_term_rms(cfg_path: Path) -> dict[str, float]:
    from jaxdrb.core.terms import build_context
    from jaxdrb.driver import build_system_from_config

    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    built = build_system_from_config(cfg)
    ctx = build_context(built.system.params, built.system.geom, built.state)
    split, term_map = built.system.scheduler.run_with_terms(ctx, built.state)

    def rms(arr) -> float:
        a = np.asarray(arr, dtype=np.float64)
        return float(np.sqrt(np.mean(a * a)))

    out: dict[str, float] = {}
    out["jax_rhs_rms_n"] = rms(split.total().n)
    out["jax_rhs_rms_Te"] = rms(split.total().Te)
    out["jax_rhs_rms_omega"] = rms(split.total().omega)
    out["jax_phi_rms"] = rms(ctx.phi)
    grid = getattr(built.system.geom, "grid", None)
    if grid is not None and hasattr(grid, "perp"):
        out["jax_dx"] = float(grid.perp.dx)
        out["jax_dy"] = float(grid.perp.dy)
        out["jax_dz"] = float(getattr(grid, "dz", 0.0))
    if hasattr(built.system.geom, "curv_x"):
        curv_x = np.asarray(built.system.geom.curv_x, dtype=np.float64)
        out["jax_curv_x_rms"] = float(np.sqrt(np.mean(curv_x * curv_x)))
        out["jax_curv_x_maxabs"] = float(np.max(np.abs(curv_x)))
    if hasattr(built.system.geom, "curv_y"):
        curv_y = np.asarray(built.system.geom.curv_y, dtype=np.float64)
        out["jax_curv_y_rms"] = float(np.sqrt(np.mean(curv_y * curv_y)))
        out["jax_curv_y_maxabs"] = float(np.max(np.abs(curv_y)))
    for name in ("parallel", "curvature", "drive", "volume_source", "diffusion", "sheath"):
        if name not in term_map:
            continue
        term = term_map[name]
        out[f"jax_{name}_rms_n"] = rms(term.n)
        out[f"jax_{name}_rms_Te"] = rms(term.Te)
        out[f"jax_{name}_rms_omega"] = rms(term.omega)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Trace the first Hermes-vs-jax mismatch in fluctuation fields and dump "
            "high-cadence channel diagnostics."
        )
    )
    p.add_argument("--hermes-data-dir", required=True, help="Hermes data dir with BOUT.dmp.*.nc.")
    p.add_argument("--jax-npz", required=True, help="jax_drb run npz.")
    p.add_argument(
        "--jax-config",
        default="",
        help=(
            "Optional jax_drb TOML config. If provided, dumps initial-term RMS and "
            "geometry scales into the summary table."
        ),
    )
    p.add_argument("--out-csv", required=True, help="Per-time mismatch table CSV.")
    p.add_argument("--summary-csv", required=True, help="Per-field first-crossing summary CSV.")
    p.add_argument("--rel-threshold", type=float, default=0.1, help="Relative L2 threshold.")
    p.add_argument(
        "--hermes-extra-vars",
        default="SNe,SPe,SNVe,Se_src,Pe_src,ddt(Ne),ddt(Pe),ddt(NVe),kappa_par_e",
        help="Comma-separated Hermes variables to emit RMS traces in summary rows.",
    )
    args = p.parse_args()

    h_times, h_fields_raw = _load_hermes_fields(
        Path(args.hermes_data_dir).resolve(), ("Ne", "Te", "Vort", "phi")
    )
    j_times, j_rms = _load_jax_rms(Path(args.jax_npz).resolve())
    jax_term_dump: dict[str, float] = {}
    if args.jax_config:
        jax_term_dump = _jax_initial_term_rms(Path(args.jax_config).resolve())

    field_map = {"n": "Ne", "Te": "Te", "omega": "Vort", "phi": "phi"}
    t_end = min(float(h_times[-1]), float(j_times[-1]))
    t_common = h_times[h_times <= t_end + 1e-12]
    if t_common.size < 2:
        raise RuntimeError("No overlapping times between Hermes and jax runs.")

    rows: list[dict[str, float | str]] = []
    summary: list[dict[str, float | str]] = []

    for field, hname in field_map.items():
        h = np.asarray(h_fields_raw[hname][: t_common.size], dtype=np.float64)
        h_eq = h[0]
        h_fluct = h - h_eq[None, ...]
        rms_h = np.sqrt(np.mean(h_fluct * h_fluct, axis=tuple(range(1, h_fluct.ndim))))
        rms_j = np.interp(t_common, j_times, np.asarray(j_rms[field], dtype=np.float64))
        rel_series = _rel_l2_scalar(rms_j, rms_h)

        first_idx = None
        first_rel = None
        for i, t in enumerate(t_common):
            rel = float(rel_series[i])
            rows.append(
                {
                    "field": field,
                    "time": float(t),
                    "rel_l2_fluct": rel,
                    "rms_hermes_fluct": float(rms_h[i]),
                    "rms_jax_fluct": float(rms_j[i]),
                }
            )
            if first_idx is None and rel > float(args.rel_threshold):
                first_idx = i
                first_rel = rel

        if first_idx is None:
            summary.append(
                {
                    "field": field,
                    "first_mismatch_time": float("nan"),
                    "first_rel_l2_fluct": float("nan"),
                    "threshold": float(args.rel_threshold),
                }
            )
        else:
            summary.append(
                {
                    "field": field,
                    "first_mismatch_time": float(t_common[first_idx]),
                    "first_rel_l2_fluct": float(first_rel),
                    "threshold": float(args.rel_threshold),
                }
            )

    # Append Hermes channel RMS at earliest mismatch time to help root-cause parity failures.
    extra_vars = tuple(v.strip() for v in args.hermes_extra_vars.split(",") if v.strip())
    if extra_vars:
        try:
            h_times_extra, h_extra = _load_hermes_fields(
                Path(args.hermes_data_dir).resolve(), extra_vars
            )
        except KeyError:
            h_times_extra, h_extra = None, {}
        for row in summary:
            t0 = row["first_mismatch_time"]
            if not np.isfinite(t0) or h_times_extra is None:
                continue
            idx = int(np.argmin(np.abs(h_times_extra - float(t0))))
            row["first_mismatch_dump_time"] = float(h_times_extra[idx])
            for name in extra_vars:
                if name not in h_extra:
                    continue
                arr = np.asarray(h_extra[name][idx], dtype=np.float64)
                row[f"rms_{name}"] = float(np.sqrt(np.mean(arr * arr)))
                row[f"maxabs_{name}"] = float(np.max(np.abs(arr)))
            row.update(jax_term_dump)

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["field", "time", "rel_l2_fluct", "rms_hermes_fluct", "rms_jax_fluct"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_csv = Path(args.summary_csv).resolve()
    base_cols = ["field", "first_mismatch_time", "first_rel_l2_fluct", "threshold"]
    extra_cols = [k for k in summary[0].keys() if k not in base_cols] if summary else []
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + extra_cols)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"Saved {out_csv}")
    print(f"Saved {summary_csv}")
    for row in summary:
        print(
            f"{row['field']}: first_mismatch_time={row['first_mismatch_time']} "
            f"rel={row['first_rel_l2_fluct']}"
        )


if __name__ == "__main__":
    main()
