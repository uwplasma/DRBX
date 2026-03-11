#!/usr/bin/env python3
"""Build a global interior shifted-transform fixture from Hermes BOUT dumps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_scalar(ds, name: str) -> int:
    return int(np.asarray(ds.variables[name][:]).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bout-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--time-index", type=int, default=0)
    args = parser.parse_args()

    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required to read Hermes BOUT dumps.") from exc

    files = sorted(args.bout_data_dir.glob("BOUT.dmp.*.nc"))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {args.bout_data_dir}")

    with Dataset(str(files[0])) as ds0:
        mxg = _read_scalar(ds0, "MXG")
        myg = _read_scalar(ds0, "MYG")
        mxsub = _read_scalar(ds0, "MXSUB")
        mysub = _read_scalar(ds0, "MYSUB")
        nxpe = _read_scalar(ds0, "NXPE")
        nype = _read_scalar(ds0, "NYPE")
        nz = int(np.asarray(ds0.variables[args.field][args.time_index]).shape[-1])
        nx = nxpe * mxsub
        ny = nype * mysub
        global_field = np.zeros((nx, ny, nz), dtype=np.float64)
        global_z_shift = np.zeros((nx, ny), dtype=np.float64)
        dz_sum = 0.0
        dz_count = 0

    for fp in files:
        with Dataset(str(fp)) as ds:
            pe_x = _read_scalar(ds, "PE_XIND")
            pe_y = _read_scalar(ds, "PE_YIND")
            x0 = pe_x * mxsub
            x1 = x0 + mxsub
            y0 = pe_y * mysub
            y1 = y0 + mysub

            raw_field = np.asarray(ds.variables[args.field][args.time_index], dtype=np.float64)
            raw_shift = np.asarray(ds.variables["zShift"][:], dtype=np.float64)
            raw_dz = np.asarray(ds.variables["dz"][:], dtype=np.float64)

            global_field[x0:x1, y0:y1, :] = raw_field[mxg : mxg + mxsub, myg : myg + mysub, :]
            global_z_shift[x0:x1, y0:y1] = raw_shift[mxg : mxg + mxsub, myg : myg + mysub]

            dz_interior = raw_dz[mxg : mxg + mxsub, myg : myg + mysub]
            dz_sum += float(dz_interior.sum())
            dz_count += int(dz_interior.size)

    zlength = (dz_sum / max(dz_count, 1)) * nz

    payload = {
        "field": np.transpose(global_field, (1, 0, 2)),  # (y, x, z)
        "z_shift": np.transpose(global_z_shift, (1, 0)),  # (y, x)
        "zlength": np.asarray(zlength, dtype=np.float64),
        "time_index": np.asarray(args.time_index, dtype=np.int32),
        "field_name": np.asarray(args.field),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)


if __name__ == "__main__":
    main()
