#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _require(module: str):
    try:
        return __import__(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{module}'. Install it to read BOUT++ grids."
        ) from exc


def _pick_var(ds, name: str) -> np.ndarray:
    if name not in ds.variables:
        raise KeyError(f"Variable '{name}' not found in grid file.")
    return np.asarray(ds.variables[name][:])


def _slice_var(arr: np.ndarray, x_index: int | None, z_index: int | None) -> np.ndarray:
    out = arr
    if out.ndim >= 3 and z_index is not None:
        out = out[..., z_index]
    if out.ndim >= 2 and x_index is not None:
        out = out[x_index, ...]
    if out.ndim > 1:
        # reduce any remaining singleton dimensions
        out = np.squeeze(out)
    return out


def _scalar(ds, name: str, default: float) -> float:
    if name in ds.variables:
        arr = np.asarray(ds.variables[name][:])
        if arr.size == 1:
            return float(arr.ravel()[0])
        if arr.ndim == 0:
            return float(arr)
    return float(default)


def _maybe_salpha_coeffs(ds, x_index: int | None) -> dict[str, np.ndarray] | None:
    if "theta_ballooning" not in ds.variables:
        return None
    theta = _slice_var(np.asarray(ds.variables["theta_ballooning"][:]), x_index, None)
    R0 = _scalar(ds, "R0", 1.0)
    r0 = _scalar(ds, "r0", 0.0)
    q0 = _scalar(ds, "q0", 1.0)
    eps = r0 / max(R0, 1e-12)
    B = 1.0 / (1.0 + eps * np.cos(theta))
    curv_x = eps * np.sin(theta) * B
    curv_y = eps * np.cos(theta) * B
    dpar_factor = np.ones_like(theta) / max(q0 * R0, 1e-12)
    return {
        "z": np.asarray(theta).reshape(-1),
        "curv_x": np.asarray(curv_x).reshape(-1),
        "curv_y": np.asarray(curv_y).reshape(-1),
        "dpar_factor": np.asarray(dpar_factor).reshape(-1),
        "B": np.asarray(B).reshape(-1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BOUT++ grid to axisymmetric coefficients (.npz)")
    parser.add_argument("--grid", required=True, type=str, help="Path to BOUT++ grid (.nc)")
    parser.add_argument("--out", required=True, type=str, help="Output .npz path")
    parser.add_argument("--curv-x-var", default="curv_x", help="Variable name for curv_x")
    parser.add_argument("--curv-y-var", default="curv_y", help="Variable name for curv_y")
    parser.add_argument("--dpar-factor-var", default="dpar_factor", help="Variable name for dpar_factor")
    parser.add_argument("--b-var", default="Bxy", help="Variable name for B magnitude")
    parser.add_argument("--z-var", default=None, help="Variable name for parallel coordinate (e.g., y, theta)")
    parser.add_argument("--x-index", type=int, default=0, help="Radial index to extract")
    parser.add_argument("--z-index", type=int, default=0, help="Toroidal index to extract if 3D")
    args = parser.parse_args()

    netcdf4 = _require("netCDF4")
    ds = netcdf4.Dataset(str(Path(args.grid)), "r")

    curv_x = curv_y = dpar_factor = None
    try:
        curv_x = _slice_var(_pick_var(ds, args.curv_x_var), args.x_index, args.z_index)
    except KeyError:
        pass
    try:
        curv_y = _slice_var(_pick_var(ds, args.curv_y_var), args.x_index, args.z_index)
    except KeyError:
        pass
    try:
        dpar_factor = _slice_var(_pick_var(ds, args.dpar_factor_var), args.x_index, args.z_index)
    except KeyError:
        pass

    B = None
    if args.b_var in ds.variables:
        B = _slice_var(_pick_var(ds, args.b_var), args.x_index, args.z_index)

    salpha = None
    if curv_x is None or curv_y is None or dpar_factor is None or B is None:
        salpha = _maybe_salpha_coeffs(ds, args.x_index)
        if salpha is None and (curv_x is None or curv_y is None or dpar_factor is None):
            raise KeyError("Missing curvature/dpar coefficients and no s-alpha metadata found.")
    if curv_x is None:
        curv_x = salpha["curv_x"]
    if curv_y is None:
        curv_y = salpha["curv_y"]
    if dpar_factor is None:
        dpar_factor = salpha["dpar_factor"]
    if B is None:
        B = salpha["B"]

    if args.z_var is not None and args.z_var in ds.variables:
        z = _slice_var(np.asarray(ds.variables[args.z_var][:]), args.x_index, args.z_index)
    else:
        z = None
        for name in ("y", "theta", "theta_ballooning"):
            if name in ds.variables:
                z = _slice_var(np.asarray(ds.variables[name][:]), args.x_index, None)
                break
        if z is None and salpha is not None:
            z = salpha["z"]
        if z is None:
            z = np.linspace(-np.pi, np.pi, curv_x.shape[-1], endpoint=False)

    out = {
        "z": np.asarray(z).reshape(-1),
        "curv_x": np.asarray(curv_x).reshape(-1),
        "curv_y": np.asarray(curv_y).reshape(-1),
        "dpar_factor": np.asarray(dpar_factor).reshape(-1),
        "B": np.asarray(B).reshape(-1),
    }

    # Optional metadata if present
    for name in ("nx", "ny", "Lx", "Ly"):
        if name in ds.variables:
            try:
                val = np.asarray(ds.variables[name][:]).item()
                out[name] = val
            except Exception:
                pass

    np.savez(Path(args.out), **out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
