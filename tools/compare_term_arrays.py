from __future__ import annotations

import argparse
import csv
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


def _load_hermes_fields(data_dir: Path, names: tuple[str, ...]):
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required for Hermes comparison.") from exc

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

    with Dataset(str(files[0])) as ds:
        has_te = "Te" in ds.variables
        has_pe = "Pe" in ds.variables
        for name in names:
            if name == "Te" and (not has_te) and has_pe:
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


def _align_hermes_to_jax(arr: np.ndarray, jax_spatial_shape: tuple[int, ...]) -> np.ndarray:
    if arr.ndim == 4:
        # Hermes: t,x,y,z. JAX: t,z,x,y.
        if jax_spatial_shape == (arr.shape[3], arr.shape[1], arr.shape[2]):
            return np.transpose(arr, (0, 3, 1, 2))
    if arr.ndim == 3 and jax_spatial_shape == arr.shape[1:]:
        return arr
    raise ValueError(f"Cannot align Hermes array shape {arr.shape} to {jax_spatial_shape}")


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "maxabs": float(np.max(np.abs(arr))),
        "mean": float(np.mean(arr)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare jax term arrays against Hermes ddt for one step."
    )
    p.add_argument("--hermes-data-dir", required=True, help="Hermes BOUT dump directory.")
    p.add_argument("--jax-term-npz", required=True, help="JAX term array npz (step_XXXX.npz).")
    p.add_argument(
        "--step",
        type=int,
        default=0,
        help="Hermes time index to compare (uses ddt between step and step+1).",
    )
    p.add_argument("--out-csv", required=True, help="Output CSV summary.")
    args = p.parse_args()

    term_npz = Path(args.jax_term_npz)
    term_data = np.load(term_npz)
    jax_fields = {}
    for key in term_data.files:
        jax_fields[key] = np.asarray(term_data[key], dtype=np.float64)

    # Determine spatial shape from one array.
    example = next(iter(jax_fields.values()))
    spatial_shape = example.shape

    hermes_times, hermes_fields = _load_hermes_fields(
        Path(args.hermes_data_dir), ("Ne", "Te", "Vort", "phi")
    )
    if args.step + 1 >= hermes_times.size:
        raise ValueError("Hermes step out of range for ddt.")

    hermes_ddt = {}
    dt = hermes_times[args.step + 1] - hermes_times[args.step]
    for name, arr in hermes_fields.items():
        aligned = _align_hermes_to_jax(arr, spatial_shape)
        hermes_ddt[name] = (aligned[args.step + 1] - aligned[args.step]) / dt

    field_map = {"n": "Ne", "Te": "Te", "omega": "Vort", "phi": "phi"}
    rows = []

    for field, hermes_name in field_map.items():
        if f"total_{field}" not in jax_fields:
            continue
        hermes_arr = hermes_ddt[hermes_name]
        jax_total = jax_fields[f"total_{field}"]
        stats_total = _stats(jax_total)
        stats_hermes = _stats(hermes_arr)
        diff = jax_total - hermes_arr
        stats_diff = _stats(diff)
        rows.append(
            {
                "term": "TOTAL",
                "field": field,
                "jax_rms": stats_total["rms"],
                "jax_maxabs": stats_total["maxabs"],
                "hermes_rms": stats_hermes["rms"],
                "hermes_maxabs": stats_hermes["maxabs"],
                "diff_rms": stats_diff["rms"],
                "diff_maxabs": stats_diff["maxabs"],
            }
        )
        # Per-term
        for key, arr in jax_fields.items():
            if not key.endswith(f"_{field}"):
                continue
            if key.startswith("total_"):
                continue
            stats_term = _stats(arr)
            diff_term = arr - hermes_arr
            stats_diff_term = _stats(diff_term)
            rows.append(
                {
                    "term": key.replace(f"_{field}", ""),
                    "field": field,
                    "jax_rms": stats_term["rms"],
                    "jax_maxabs": stats_term["maxabs"],
                    "hermes_rms": stats_hermes["rms"],
                    "hermes_maxabs": stats_hermes["maxabs"],
                    "diff_rms": stats_diff_term["rms"],
                    "diff_maxabs": stats_diff_term["maxabs"],
                }
            )

    _write_csv(
        Path(args.out_csv),
        rows,
        [
            "term",
            "field",
            "jax_rms",
            "jax_maxabs",
            "hermes_rms",
            "hermes_maxabs",
            "diff_rms",
            "diff_maxabs",
        ],
    )
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
