#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require(module: str):
    try:
        return __import__(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{module}'. Install it to read BOUT++ grids."
        ) from exc


def _slice_var(arr: np.ndarray, x_index: int | None) -> np.ndarray:
    if arr.ndim >= 2 and x_index is not None:
        return np.asarray(arr[x_index, ...])
    return np.asarray(arr)


def _reconstruct_logB(logB: np.ndarray, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    if logB.ndim == 1:
        return logB, np.zeros_like(logB)
    if logB.ndim == 2 and logB.shape[-1] >= 3:
        logB0 = logB[..., 0]
        logB1 = logB[..., 1]
        logB2 = logB[..., 2]
        logB_z = logB0 + logB1 * np.cos(zeta) + logB2 * np.sin(zeta)
        dlogB_dz = -logB1 * np.sin(zeta) + logB2 * np.cos(zeta)
        return logB_z, dlogB_dz
    if logB.ndim == 2:
        return logB, np.zeros_like(logB)
    raise ValueError(f"Unsupported logB shape {logB.shape}")


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


def _radial_from_dr(r0: float, dr: float, nx: int, mxg: int) -> np.ndarray:
    h = dr / max(nx - 2 * mxg, 1)
    return np.linspace(
        r0 - 0.5 * dr - (mxg - 0.5) * h,
        r0 + 0.5 * dr + (mxg - 0.5) * h,
        nx,
    )


def _scalar(ds, name: str, default: float) -> float:
    if name in ds.variables:
        arr = np.asarray(ds.variables[name][:])
        if arr.size == 1:
            return float(arr.ravel()[0])
        if arr.ndim == 0:
            return float(arr)
    return float(default)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert BOUT++ metric grid to axisymmetric coefficients (.npz)"
    )
    p.add_argument("--grid", required=True, type=str, help="Path to BOUT++ grid (.nc)")
    p.add_argument("--out", required=True, type=str, help="Output .npz path")
    p.add_argument("--x-index", type=int, default=0, help="Radial index to extract")
    p.add_argument(
        "--zeta", type=float, default=0.0, help="Toroidal angle for logB Fourier reconstruction"
    )
    p.add_argument(
        "--radial-coordinate",
        choices=("flux", "physical"),
        default="physical",
        help="Use flux coordinate or physical minor radius for x-derivative",
    )
    p.add_argument(
        "--radial-from",
        choices=("auto", "dr", "dx_btor", "dx"),
        default="auto",
        help="How to build physical radial coordinate when radial-coordinate=physical",
    )
    p.add_argument(
        "--curv-x-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Which axis derivative defines curv_x",
    )
    p.add_argument(
        "--curv-y-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Which axis derivative defines curv_y",
    )
    p.add_argument("--curv-sign-x", type=float, default=1.0, help="Optional sign flip for curv_x")
    p.add_argument("--curv-sign-y", type=float, default=1.0, help="Optional sign flip for curv_y")
    p.add_argument("--use-metric", action="store_true", help="Use gxx/gxy/gyy for scaling")
    p.add_argument("--gxx-var", default="gxx_ballooning")
    p.add_argument("--gxy-var", default="gxy_ballooning")
    p.add_argument("--gyy-var", default="gyy_ballooning")
    p.add_argument("--logb-var", default="logB")
    p.add_argument("--hthe-var", default="hthe")
    p.add_argument("--bxy-var", default="Bxy")
    p.add_argument("--bpxy-var", default="Bpxy")
    p.add_argument("--dx-var", default="dx")
    p.add_argument("--dy-var", default="dy")
    p.add_argument("--mxg-var", default="mxg")
    p.add_argument("--theta-var", default="theta_ballooning")
    args = p.parse_args()

    netcdf4 = _require("netCDF4")
    ds = netcdf4.Dataset(str(Path(args.grid)), "r")

    if args.logb_var not in ds.variables:
        raise ValueError("logB variable missing in BOUT grid")

    logB_raw = np.asarray(ds.variables[args.logb_var][:])
    logB_slice = _slice_var(logB_raw, args.x_index)
    logB_z, dlogB_dz = _reconstruct_logB(logB_slice, args.zeta)

    dx = np.asarray(ds.variables[args.dx_var][:])
    dy = np.asarray(ds.variables[args.dy_var][:]) if args.dy_var in ds.variables else None

    Bxy_full = np.asarray(ds.variables[args.bxy_var][:])
    Bpxy_full = np.asarray(ds.variables[args.bpxy_var][:])
    Rxy_full = np.asarray(ds.variables["Rxy"][:]) if "Rxy" in ds.variables else None

    # Build x-coordinate for derivative.
    if args.radial_coordinate == "flux":
        x_coord = _coord_from_spacing(dx, axis=0)
    else:
        mode = args.radial_from
        if mode == "auto":
            if "dr" in ds.variables and args.mxg_var in ds.variables and "r0" in ds.variables:
                mode = "dr"
            elif Rxy_full is not None:
                mode = "dx_btor"
            else:
                mode = "dx"
        if mode == "dr":
            r0 = float(np.asarray(ds.variables["r0"][:]).ravel()[0])
            dr = float(np.asarray(ds.variables["dr"][:]).ravel()[0])
            mxg = int(np.asarray(ds.variables[args.mxg_var][:]).ravel()[0])
            nx = int(np.asarray(ds.variables["nx"][:]).ravel()[0])
            x_coord = _radial_from_dr(r0, dr, nx, mxg)
        elif mode == "dx_btor" and Rxy_full is not None:
            dr_est = dx / np.maximum(Rxy_full * Bpxy_full, 1e-12)
            x_coord = _coord_from_spacing(dr_est, axis=0)
        else:
            x_coord = _coord_from_spacing(dx, axis=0)

    logB_xy = logB_raw
    if logB_xy.ndim >= 3 and logB_xy.shape[-1] >= 3:
        logB_xy = (
            logB_xy[..., 0]
            + logB_xy[..., 1] * np.cos(args.zeta)
            + logB_xy[..., 2] * np.sin(args.zeta)
        )

    dlogB_dx_full = np.gradient(logB_xy, x_coord, axis=0, edge_order=2)
    dlogB_dx = _slice_var(dlogB_dx_full, args.x_index)

    if dy is not None:
        y_coord = _coord_from_spacing(dy, axis=1)
        dlogB_dy_full = np.gradient(logB_xy, y_coord, axis=1, edge_order=2)
        dlogB_dy = _slice_var(dlogB_dy_full, args.x_index)
    else:
        y_coord = np.arange(logB_z.shape[-1])
        dlogB_dy = np.zeros_like(dlogB_dx)

    Bxy = _slice_var(Bxy_full, args.x_index)
    Bpxy = _slice_var(Bpxy_full, args.x_index)
    hthe = _slice_var(np.asarray(ds.variables[args.hthe_var][:]), args.x_index)

    scale_x = 1.0
    scale_y = 1.0
    gxx = None
    gxy = None
    gyy = None
    if (args.gxx_var in ds.variables) and (args.gyy_var in ds.variables):
        gxx = _slice_var(np.asarray(ds.variables[args.gxx_var][:]), args.x_index)
        gyy = _slice_var(np.asarray(ds.variables[args.gyy_var][:]), args.x_index)
        if args.gxy_var in ds.variables:
            gxy = _slice_var(np.asarray(ds.variables[args.gxy_var][:]), args.x_index)
        else:
            gxy = np.zeros_like(gxx)
        if args.use_metric:
            gxx_safe = np.maximum(gxx, 1e-12)
            gperp = np.maximum(gyy - (gxy**2) / gxx_safe, 1e-12)
            scale_x = np.sqrt(gxx_safe)
            scale_y = np.sqrt(gperp)

    derivs = {"x": dlogB_dx, "y": dlogB_dy, "z": dlogB_dz}
    curv_x = -Bxy * scale_y * derivs[args.curv_x_axis] * float(args.curv_sign_x)
    curv_y = Bxy * scale_x * derivs[args.curv_y_axis] * float(args.curv_sign_y)
    dpar_factor = Bpxy / np.maximum(Bxy * hthe, 1e-12)

    # z (field-aligned coordinate)
    if args.theta_var in ds.variables:
        z = _slice_var(np.asarray(ds.variables[args.theta_var][:]), args.x_index)
    else:
        z = _slice_var(y_coord, args.x_index)

    # Lengths
    Lx = float(x_coord[-1] - x_coord[0] + (x_coord[1] - x_coord[0]) if x_coord.size > 1 else 1.0)
    Ly = float(y_coord[-1] - y_coord[0] + (y_coord[1] - y_coord[0]) if y_coord.size > 1 else 1.0)
    nx = int(_scalar(ds, "nx", curv_x.shape[-1]))
    ny = int(_scalar(ds, "ny", curv_x.shape[-1]))

    out = {
        "z": np.asarray(z).reshape(-1),
        "curv_x": np.asarray(curv_x).reshape(-1),
        "curv_y": np.asarray(curv_y).reshape(-1),
        "dpar_factor": np.asarray(dpar_factor).reshape(-1),
        "B": np.asarray(Bxy).reshape(-1),
        "Lx": Lx,
        "Ly": Ly,
        "nx": nx,
        "ny": ny,
    }

    if gxx is not None:
        out["gxx"] = np.asarray(gxx)
    if gxy is not None:
        out["gxy"] = np.asarray(gxy)
    if gyy is not None:
        out["gyy"] = np.asarray(gyy)

    np.savez(Path(args.out), **out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
