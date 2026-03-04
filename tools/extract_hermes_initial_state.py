#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def _canonical_hermes(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 4:
        # Hermes canonical: (t, x, y_binormal, z_parallel)
        return np.transpose(data, (0, 1, 3, 2))
    return data


def _crop_xy(arr: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
    if arr.ndim < 2:
        return arr
    xs = slice(crop_x, -crop_x if crop_x > 0 else None)
    ys = slice(crop_y, -crop_y if crop_y > 0 else None)
    if arr.ndim == 2:
        return arr[xs, ys]
    if arr.ndim == 3:
        return arr[:, xs, ys]
    if arr.ndim == 4:
        return arr[:, xs, ys, :]
    return arr


def _to_jax_zxy(arr_txyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_txyz, dtype=np.float64)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        # canonical Hermes is (x, y, z_parallel) -> jax (z, x, y)
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported field rank for initial-state export: {arr.ndim}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract Hermes t-index state to jax_drb init-state npz."
    )
    p.add_argument("--hermes-data", required=True, help="Directory with BOUT.dmp.*.nc")
    p.add_argument("--out", required=True, help="Output .npz with fields in jax (z,x,y)")
    p.add_argument("--time-index", type=int, default=0, help="Time index to export")
    p.add_argument("--crop-x", type=int, default=0, help="Crop x guard layers after stitching")
    p.add_argument("--crop-y", type=int, default=0, help="Crop y guard layers after stitching")
    args = p.parse_args()

    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required to read Hermes BOUT dumps.") from exc

    data_dir = Path(args.hermes_data).resolve()
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
        var_names = set(ds0.variables.keys())

    nt = int(times.size)
    tidx = args.time_index if args.time_index >= 0 else nt + args.time_index
    if tidx < 0 or tidx >= nt:
        raise IndexError(f"time-index {args.time_index} out of bounds for nt={nt}")

    nx = int(nxpe * mxsub)
    ny = int(nype * mysub)

    def alloc(name: str) -> np.ndarray:
        with Dataset(str(files[0])) as ds:
            arr = _read_var_interior(ds, name, mxg, myg, mxsub, mysub)
            if arr.ndim == 4:
                return np.zeros((nt, nx, ny, arr.shape[-1]), dtype=np.float64)
            return np.zeros((nt, nx, ny), dtype=np.float64)

    has_te = "Te" in var_names
    has_pe = "Pe" in var_names
    if not has_te and not has_pe:
        raise KeyError("Hermes dump missing Te (or Pe)")

    fields: dict[str, np.ndarray] = {
        "Ne": alloc("Ne"),
        "Vort": alloc("Vort"),
        "phi": alloc("phi"),
    }
    if has_te:
        fields["Te"] = alloc("Te")
    else:
        fields["Pe"] = alloc("Pe")

    ion_density = None
    ion_momentum = None
    if "Nd+" in var_names and "NVd+" in var_names:
        ion_density, ion_momentum = "Nd+", "NVd+"
    else:
        nv_plus = sorted([v for v in var_names if v.startswith("NV") and v.endswith("+")])
        for nv in nv_plus:
            n_candidate = "N" + nv[2:]
            if n_candidate in var_names:
                ion_density, ion_momentum = n_candidate, nv
                break
    if ion_density and ion_momentum:
        fields["ION_N"] = alloc(ion_density)
        fields["ION_NV"] = alloc(ion_momentum)

    has_ve = "Ve" in var_names
    has_nve = "NVe" in var_names
    if has_ve:
        fields["Ve"] = alloc("Ve")
    elif has_nve:
        fields["NVe"] = alloc("NVe")

    for local_rank, fp in enumerate(files):
        with Dataset(str(fp)) as ds:
            pe_x = int(_read_scalar(ds, ("PE_XIND",), local_rank % max(nxpe, 1)))
            pe_y = int(_read_scalar(ds, ("PE_YIND",), local_rank // max(nxpe, 1)))
            x0 = pe_x * mxsub
            y0 = pe_y * mysub
            x1 = x0 + mxsub
            y1 = y0 + mysub
            fields["Ne"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                ds, "Ne", mxg, myg, mxsub, mysub
            )
            fields["Vort"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                ds, "Vort", mxg, myg, mxsub, mysub
            )
            fields["phi"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                ds, "phi", mxg, myg, mxsub, mysub
            )
            if has_te:
                fields["Te"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, "Te", mxg, myg, mxsub, mysub
                )
            else:
                fields["Pe"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, "Pe", mxg, myg, mxsub, mysub
                )
            if ion_density and ion_momentum:
                fields["ION_N"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, ion_density, mxg, myg, mxsub, mysub
                )
                fields["ION_NV"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, ion_momentum, mxg, myg, mxsub, mysub
                )
            if has_ve:
                fields["Ve"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, "Ve", mxg, myg, mxsub, mysub
                )
            elif has_nve:
                fields["NVe"][:, x0:x1, y0:y1, ...] = _read_var_interior(
                    ds, "NVe", mxg, myg, mxsub, mysub
                )

    canon = {k: _crop_xy(_canonical_hermes(v), args.crop_x, args.crop_y) for k, v in fields.items()}
    if "Te" not in canon:
        canon["Te"] = canon["Pe"] / np.maximum(canon["Ne"], 1e-12)
    if "ION_N" in canon and "ION_NV" in canon:
        canon["vpar_i"] = canon["ION_NV"] / np.maximum(canon["ION_N"], 1e-12)
    if "Ve" in canon:
        canon["vpar_e"] = canon["Ve"]
    elif "NVe" in canon:
        canon["vpar_e"] = canon["NVe"] / np.maximum(canon["Ne"], 1e-12)

    out = {
        "n": _to_jax_zxy(canon["Ne"][tidx]),
        "Te": _to_jax_zxy(canon["Te"][tidx]),
        "omega": _to_jax_zxy(canon["Vort"][tidx]),
        "phi": _to_jax_zxy(canon["phi"][tidx]),
        "t_source": np.asarray(times[tidx], dtype=np.float64),
    }
    if "vpar_i" in canon:
        out["vpar_i"] = _to_jax_zxy(canon["vpar_i"][tidx])
    if "vpar_e" in canon:
        out["vpar_e"] = _to_jax_zxy(canon["vpar_e"][tidx])

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved Hermes initial state: {out_path}")


if __name__ == "__main__":
    main()
