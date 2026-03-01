#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_scalar(ds, name: str, default: float) -> int:
    if name in ds.variables:
        arr = np.asarray(ds.variables[name][:], dtype=np.float64)
        return int(arr.reshape(-1)[0])
    return int(default)


def _stitch_2d(files: list[Path], name: str) -> np.ndarray:
    from netCDF4 import Dataset

    with Dataset(str(files[0])) as ds0:
        mxg = _read_scalar(ds0, "MXG", 2)
        myg = _read_scalar(ds0, "MYG", 2)
        mxsub = _read_scalar(ds0, "MXSUB", ds0.variables[name].shape[0] - 2 * mxg)
        mysub = _read_scalar(ds0, "MYSUB", ds0.variables[name].shape[1] - 2 * myg)
        nxpe = _read_scalar(ds0, "NXPE", 1)
        nype = _read_scalar(ds0, "NYPE", max(1, len(files) // max(nxpe, 1)))

    nx = nxpe * mxsub
    ny = nype * mysub
    out = np.zeros((nx, ny), dtype=np.float64)

    for local_rank, fp in enumerate(files):
        with Dataset(str(fp)) as ds:
            pe_x = _read_scalar(ds, "PE_XIND", local_rank % max(nxpe, 1))
            pe_y = _read_scalar(ds, "PE_YIND", local_rank // max(nxpe, 1))
            x0 = pe_x * mxsub
            y0 = pe_y * mysub
            x1 = x0 + mxsub
            y1 = y0 + mysub
            arr = np.asarray(ds.variables[name][:], dtype=np.float64)
            arr = arr[mxg : mxg + mxsub, myg : myg + mysub]
            out[x0:x1, y0:y1] = arr
    return out


def _slice_grid(arr: np.ndarray, mxg: int, myg: int, nx: int, ny: int) -> np.ndarray:
    if arr.ndim != 2:
        return np.asarray(arr)
    return np.asarray(arr[mxg : mxg + nx, myg : myg + ny])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build axisymmetric coefficients from Hermes BOUT dumps"
    )
    p.add_argument("--dump-dir", required=True, help="Hermes BOUT dump directory")
    p.add_argument("--out", required=True, help="Output .npz file")
    p.add_argument(
        "--grid",
        default=None,
        help="Optional BOUT grid (tokamak.nc) for Rxy/Zxy/bxcv",
    )
    p.add_argument(
        "--dpar-from",
        choices=("g22", "g_22"),
        default="g22",
        help="Metric component used to build dpar_factor (sqrt).",
    )
    p.add_argument(
        "--zperiod",
        type=float,
        default=None,
        help="BOUT zperiod (number of toroidal periods). If set, Ly uses 2*pi/zperiod.",
    )
    p.add_argument(
        "--mz",
        type=int,
        default=None,
        help="BOUT MZ (number of toroidal points). Used with --zperiod for Ly.",
    )
    p.add_argument(
        "--ny-binormal",
        type=int,
        default=None,
        help="Optional binormal (toroidal) grid size for output coefficients.",
    )
    p.add_argument(
        "--bxcv-normalization",
        choices=("none", "hermes"),
        default="hermes",
        help="Normalization for bxcv when loaded from grid file.",
    )
    p.add_argument(
        "--bxcv-use-2overB",
        action="store_true",
        help="Scale bxcv by 2/Bxy when loading from grid file.",
    )
    p.add_argument("--Te0-eV", type=float, default=50.0)
    p.add_argument("--B0-T", type=float, default=1.0)
    p.add_argument("--mi-amu", type=float, default=2.0)
    p.add_argument(
        "--average-parallel",
        action="store_true",
        help="Average metric/B fields over the parallel (y) coordinate before saving.",
    )
    p.add_argument(
        "--preserve-parallel-curvature",
        action="store_true",
        help=(
            "Keep curv_x/curv_y/dpar_factor varying along the parallel coordinate "
            "(store as (ny_parallel, nx)) even when ny_binormal != ny_parallel."
        ),
    )
    p.add_argument(
        "--preserve-parallel-metrics",
        action="store_true",
        help=(
            "Keep gxx/gxy/gyy/J/B varying along the parallel coordinate "
            "(store as (ny_parallel, nx)) even when ny_binormal != ny_parallel."
        ),
    )
    args = p.parse_args()

    from netCDF4 import Dataset

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("BOUT.dmp.*.nc"))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files in {dump_dir}")

    g11 = _stitch_2d(files, "g11")
    g22 = _stitch_2d(files, "g22")
    g33 = _stitch_2d(files, "g33")
    g12 = _stitch_2d(files, "g12")
    g13 = _stitch_2d(files, "g13")
    try:
        g23 = _stitch_2d(files, "g23")
    except Exception:
        g23 = None
    try:
        g_22 = _stitch_2d(files, "g_22")
    except Exception:
        g_22 = None
    try:
        g_23 = _stitch_2d(files, "g_23")
    except Exception:
        g_23 = None
    try:
        z_shift = _stitch_2d(files, "zShift")
    except Exception:
        z_shift = None
    gpar = g_22
    Bxy = _stitch_2d(files, "Bxy")
    Bxy_raw = np.asarray(Bxy, dtype=np.float64)
    dx = _stitch_2d(files, "dx")
    dy = _stitch_2d(files, "dy")
    J = _stitch_2d(files, "J")

    nx, ny_parallel = g11.shape
    Lx = float(dx.mean() * nx)
    if args.zperiod is not None and args.mz is not None and args.mz > 0:
        Ly = float(2.0 * np.pi / max(args.zperiod, 1e-12))
    else:
        Ly = float(dy.mean() * ny_parallel)
    Lpar = float(dy.mean() * ny_parallel)
    z = np.linspace(0.0, Lpar, ny_parallel, endpoint=False, dtype=np.float64)

    if args.dpar_from == "g_22":
        gpar = _stitch_2d(files, "g_22")
        dpar_factor = np.sqrt(1.0 / np.maximum(gpar, 1e-30))
    else:
        dpar_factor = np.sqrt(np.maximum(g22, 1e-30))

    ny_binormal = int(args.ny_binormal or args.mz or ny_parallel)

    def _avg_parallel(a: np.ndarray) -> np.ndarray:
        if a.ndim != 2:
            return a
        mean = np.mean(a, axis=1, keepdims=True)
        return np.repeat(mean, ny_binormal, axis=1)

    if (args.average_parallel or ny_binormal != ny_parallel) and (
        not args.preserve_parallel_metrics
    ):
        g11 = _avg_parallel(g11)
        g33 = _avg_parallel(g33)
        g13 = _avg_parallel(g13)
        Bxy = _avg_parallel(Bxy)
        J = _avg_parallel(J)
        if gpar is not None:
            gpar = _avg_parallel(gpar)
        if not args.preserve_parallel_curvature:
            dpar_factor = _avg_parallel(dpar_factor)

    # Optional grid for Rxy/Zxy and bxcv
    Rxy = None
    Zxy = None
    curv_x = None
    curv_y = None
    curv_par = None
    if args.grid:
        grid = Path(args.grid)
        with Dataset(str(grid)) as ds:
            mxg = (
                int(np.asarray(ds.variables.get("MXG", 0)).reshape(-1)[0])
                if "MXG" in ds.variables
                else 0
            )
            myg = (
                int(np.asarray(ds.variables.get("MYG", 0)).reshape(-1)[0])
                if "MYG" in ds.variables
                else 0
            )
            if "Rxy" in ds.variables:
                Rxy = _slice_grid(
                    np.asarray(ds.variables["Rxy"][:], dtype=np.float64), mxg, myg, nx, ny_parallel
                )
            if "Zxy" in ds.variables:
                Zxy = _slice_grid(
                    np.asarray(ds.variables["Zxy"][:], dtype=np.float64), mxg, myg, nx, ny_parallel
                )
            if all(v in ds.variables for v in ("bxcvx", "bxcvy", "bxcvz")):
                bxcvx = _slice_grid(
                    np.asarray(ds.variables["bxcvx"][:], dtype=np.float64),
                    mxg,
                    myg,
                    nx,
                    ny_parallel,
                )
                bxcvy = _slice_grid(
                    np.asarray(ds.variables["bxcvy"][:], dtype=np.float64),
                    mxg,
                    myg,
                    nx,
                    ny_parallel,
                )
                bxcvz = _slice_grid(
                    np.asarray(ds.variables["bxcvz"][:], dtype=np.float64),
                    mxg,
                    myg,
                    nx,
                    ny_parallel,
                )
                # Hermes normalization if requested
                if args.bxcv_normalization == "hermes":
                    # rho_s0 = cs/omega_ci
                    e = 1.602176634e-19
                    m_p = 1.67262192369e-27
                    m_i = args.mi_amu * m_p
                    cs = np.sqrt(max(args.Te0_eV, 1e-12) * e / m_i)
                    omega_ci = max(args.B0_T, 1e-12) * e / m_i
                    rho_s = cs / omega_ci
                    bxcvx = bxcvx / max(args.B0_T, 1e-12)
                    bxcvy = bxcvy * rho_s * rho_s
                    bxcvz = bxcvz * rho_s * rho_s
                if args.bxcv_use_2overB:
                    # Use pre-averaged Bxy to avoid shape mismatch, then average bxcv later.
                    bxcvx = 2.0 * bxcvx / np.maximum(Bxy_raw, 1e-12)
                    bxcvy = 2.0 * bxcvy / np.maximum(Bxy_raw, 1e-12)
                    bxcvz = 2.0 * bxcvz / np.maximum(Bxy_raw, 1e-12)
                # map to curv_x/curv_y (x,z components for perpendicular plane)
                curv_x = bxcvx
                curv_y = bxcvz
                curv_par = bxcvy
                if args.average_parallel or ny_binormal != ny_parallel:
                    if not args.preserve_parallel_curvature:
                        curv_x = _avg_parallel(curv_x)
                        curv_y = _avg_parallel(curv_y)
                        curv_par = _avg_parallel(curv_par)
            if Rxy is not None and (args.average_parallel or ny_binormal != ny_parallel):
                Rxy = _avg_parallel(Rxy)
            if Zxy is not None and (args.average_parallel or ny_binormal != ny_parallel):
                Zxy = _avg_parallel(Zxy)

    if args.preserve_parallel_curvature and dpar_factor.ndim == 2:
        dpar_out = dpar_factor.T
    else:
        dpar_out = dpar_factor
    if z_shift is not None:
        z_shift_out = (
            z_shift.T if (args.preserve_parallel_curvature and z_shift.ndim == 2) else z_shift
        )
    else:
        z_shift_out = None
    if gpar is not None:
        gpar_out = gpar.T if (args.preserve_parallel_curvature and gpar.ndim == 2) else gpar
    else:
        gpar_out = None

    if args.preserve_parallel_metrics:
        if g11.ndim == 2:
            g11 = g11.T
        if g33.ndim == 2:
            g33 = g33.T
        if g13 is not None and g13.ndim == 2:
            g13 = g13.T
        if g23 is not None and g23.ndim == 2:
            g23 = g23.T
        if g_22 is not None and g_22.ndim == 2:
            g_22 = g_22.T
        if g_23 is not None and g_23.ndim == 2:
            g_23 = g_23.T
        if Bxy.ndim == 2:
            Bxy = Bxy.T
        if J is not None and J.ndim == 2:
            J = J.T

    out = {
        "nx": nx,
        "ny": ny_binormal,
        "Lx": Lx,
        "Ly": Ly,
        "z": z,
        "B": Bxy,
        "gxx": g11,
        "gxy": g13 if g13 is not None else g12 * 0.0,
        "gyy": g33 if g33 is not None else g22,
        "dpar_factor": dpar_out,
    }
    if g23 is not None:
        out["g23"] = g23
    if g_22 is not None:
        out["g_22"] = g_22
    if g_23 is not None:
        out["g_23"] = g_23
    if z_shift_out is not None:
        out["z_shift"] = z_shift_out
    if gpar_out is not None:
        out["gpar"] = gpar_out
    if J is not None:
        out["J"] = J
    if Rxy is not None:
        out["Rxy"] = Rxy
    if Zxy is not None:
        out["Zxy"] = Zxy
    if curv_x is not None:
        out["curv_x"] = curv_x.T if args.preserve_parallel_curvature else curv_x
    if curv_y is not None:
        out["curv_y"] = curv_y.T if args.preserve_parallel_curvature else curv_y
    if curv_par is not None:
        out["curv_par"] = curv_par.T if args.preserve_parallel_curvature else curv_par

    np.savez(args.out, **out)
    print(f"Wrote {args.out} with keys: {sorted(out.keys())}")


if __name__ == "__main__":
    main()
