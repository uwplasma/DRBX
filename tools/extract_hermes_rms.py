from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract total and fluctuation RMS channels from Hermes BOUT dump files."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing BOUT.dmp.<rank>.nc files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output npz path.",
    )
    return parser.parse_args()


def _collect_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("BOUT.dmp.*.nc"))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {data_dir}")
    return files


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    files = _collect_files(data_dir)

    with Dataset(str(files[0])) as ds0:
        times = np.asarray(ds0.variables["t"][:], dtype=np.float64)
        mxg = int(ds0.variables["MXG"][:])
        myg = int(ds0.variables["MYG"][:])
        mxs = int(ds0.variables["MXSUB"][:])
        mys = int(ds0.variables["MYSUB"][:])

    names = ("Ne", "Te", "Vort", "phi")
    rms_total = {name: np.zeros_like(times) for name in names}
    rms_fluct = {name: np.zeros_like(times) for name in names}

    baseline: dict[tuple[Path, str], np.ndarray] = {}
    for fp in files:
        with Dataset(str(fp)) as ds:
            for name in names:
                baseline[(fp, name)] = np.asarray(
                    ds.variables[name][0, mxg : mxg + mxs, myg : myg + mys, :], dtype=np.float64
                )

    for ti in range(times.size):
        sums_total = {name: 0.0 for name in names}
        sums_fluct = {name: 0.0 for name in names}
        count = 0
        for fp in files:
            with Dataset(str(fp)) as ds:
                for name in names:
                    arr = np.asarray(
                        ds.variables[name][ti, mxg : mxg + mxs, myg : myg + mys, :],
                        dtype=np.float64,
                    )
                    delta = arr - baseline[(fp, name)]
                    sums_total[name] += float(np.sum(arr * arr))
                    sums_fluct[name] += float(np.sum(delta * delta))
                if count == 0:
                    nz = int(ds.variables["Ne"].shape[-1])
                count += mxs * mys * nz
        for name in names:
            rms_total[name][ti] = np.sqrt(sums_total[name] / max(count, 1))
            rms_fluct[name][ti] = np.sqrt(sums_fluct[name] / max(count, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        times=times,
        rms_n=rms_total["Ne"],
        rms_Te=rms_total["Te"],
        rms_omega=rms_total["Vort"],
        rms_phi=rms_total["phi"],
        rms_n_fluct=rms_fluct["Ne"],
        rms_Te_fluct=rms_fluct["Te"],
        rms_omega_fluct=rms_fluct["Vort"],
        rms_phi_fluct=rms_fluct["phi"],
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
