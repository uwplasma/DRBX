from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _interp_axis(arr: np.ndarray, src: np.ndarray, dst: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(arr, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = np.empty((dst.size, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    out = out.reshape((dst.size, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)


def _resample_2d(arr: np.ndarray, nx: int, ny: int) -> np.ndarray:
    sx = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    sy = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float64)
    tx = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    ty = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    out = _interp_axis(arr.astype(np.float64), sx, tx, axis=0)
    out = _interp_axis(out, sy, ty, axis=1)
    return out


def _downsample_file(inp: Path, out: Path, nx: int, ny: int) -> None:
    raw = np.load(inp)
    data: dict[str, np.ndarray | float | int] = {}
    nx0 = None
    ny0 = None
    if "Rxy" in raw and np.asarray(raw["Rxy"]).ndim == 2:
        nx0, ny0 = np.asarray(raw["Rxy"]).shape
    elif "Zxy" in raw and np.asarray(raw["Zxy"]).ndim == 2:
        nx0, ny0 = np.asarray(raw["Zxy"]).shape
    elif "mask_core" in raw and np.asarray(raw["mask_core"]).ndim == 2:
        nx0, ny0 = np.asarray(raw["mask_core"]).shape
    if nx0 is None or ny0 is None:
        raise ValueError("Input coefficients must contain 2D 'curv_x' to infer original grid.")

    for key in raw.files:
        val = np.asarray(raw[key])
        if key in ("nx", "ny"):
            continue
        if val.ndim == 0:
            data[key] = val.item()
            continue
        if val.ndim == 1:
            if val.size == nx0:
                s = np.linspace(0.0, 1.0, nx0, dtype=np.float64)
                t = np.linspace(0.0, 1.0, nx, dtype=np.float64)
                data[key] = np.interp(t, s, val.astype(np.float64))
            elif val.size == ny0:
                s = np.linspace(0.0, 1.0, ny0, dtype=np.float64)
                t = np.linspace(0.0, 1.0, ny, dtype=np.float64)
                data[key] = np.interp(t, s, val.astype(np.float64))
            else:
                data[key] = val
            continue
        if val.ndim == 2 and val.shape == (nx0, ny0):
            if key.startswith("mask_"):
                sampled = _resample_2d(val, nx, ny)
                data[key] = (sampled >= 0.5).astype(np.float64)
            else:
                data[key] = _resample_2d(val, nx, ny)
            continue
        if val.ndim >= 3 and val.shape[0] == nx0 and val.shape[1] == ny0:
            first = _resample_2d(val[..., 0], nx, ny)
            out_arr = np.empty((nx, ny, *val.shape[2:]), dtype=np.float64)
            out_arr[..., 0] = first
            for i in np.ndindex(val.shape[2:]):
                if i == tuple(0 for _ in val.shape[2:]):
                    continue
                out_arr[(slice(None), slice(None), *i)] = _resample_2d(
                    val[(slice(None), slice(None), *i)], nx, ny
                )
            data[key] = out_arr
            continue
        data[key] = val

    data["nx"] = np.array(nx, dtype=np.int32)
    data["ny"] = np.array(ny, dtype=np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)


def main() -> None:
    p = argparse.ArgumentParser(description="Downsample axisymmetric coefficient .npz to nx,ny.")
    p.add_argument("--input", required=True, help="Input axisymmetric .npz coefficients.")
    p.add_argument("--output", required=True, help="Output .npz path.")
    p.add_argument("--nx", type=int, required=True, help="Target radial grid points.")
    p.add_argument("--ny", type=int, required=True, help="Target poloidal grid points.")
    args = p.parse_args()

    if args.nx <= 2 or args.ny <= 2:
        raise ValueError("nx and ny must be > 2.")
    _downsample_file(Path(args.input).resolve(), Path(args.output).resolve(), args.nx, args.ny)
    print(f"Saved {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
