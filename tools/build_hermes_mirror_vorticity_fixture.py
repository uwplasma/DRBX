#!/usr/bin/env python3
"""Build a stitched global Hermes vorticity mirror fixture from BOUT dumps."""

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
    parser.add_argument("--time-index", type=int, default=0)
    args = parser.parse_args()

    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required to read Hermes BOUT dumps.") from exc

    files = sorted(args.bout_data_dir.glob("BOUT.dmp.*.nc"))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {args.bout_data_dir}")

    fields_3d = ("Ne", "Te", "Nd+", "Td+", "Vort", "phi", "term_Vort_exb")
    with Dataset(str(files[0])) as ds0:
        mxg = _read_scalar(ds0, "MXG")
        myg = _read_scalar(ds0, "MYG")
        mxsub = _read_scalar(ds0, "MXSUB")
        mysub = _read_scalar(ds0, "MYSUB")
        nxpe = _read_scalar(ds0, "NXPE")
        nype = _read_scalar(ds0, "NYPE")
        nbinorm = int(np.asarray(ds0.variables["phi"][args.time_index]).shape[-1])
        nx = nxpe * mxsub
        npar = nype * mysub
        payload: dict[str, np.ndarray] = {
            "mxsub": np.asarray(mxsub, dtype=np.int32),
            "mysub": np.asarray(mysub, dtype=np.int32),
            "nxpe": np.asarray(nxpe, dtype=np.int32),
            "nype": np.asarray(nype, dtype=np.int32),
            "time_index": np.asarray(args.time_index, dtype=np.int32),
        }
        for name in fields_3d:
            payload[name] = np.zeros((npar, nx, nbinorm), dtype=np.float64)

    for fp in files:
        with Dataset(str(fp)) as ds:
            pe_x = _read_scalar(ds, "PE_XIND")
            pe_y = _read_scalar(ds, "PE_YIND")
            x0 = pe_x * mxsub
            x1 = x0 + mxsub
            y0 = pe_y * mysub
            y1 = y0 + mysub

            for name in fields_3d:
                raw = np.asarray(ds.variables[name][args.time_index], dtype=np.float64)
                payload[name][y0:y1, x0:x1, :] = raw[
                    mxg : mxg + mxsub,
                    myg : myg + mysub,
                    :,
                ].transpose(1, 0, 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)


if __name__ == "__main__":
    main()
