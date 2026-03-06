"""Build compact `(nz, nx, ny)` fixtures for Hermes mirror unit tests.

Phase 1 uses two sources:

- an existing `.npz` bundle already stored in JAX layout
- a raw local `BOUT.dmp.*.nc` file from Hermes/BOUT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Input `.npz` bundle.")
    src.add_argument("--bout-dump", type=Path, help="Input local `BOUT.dmp.*.nc` file.")
    parser.add_argument("--output", type=Path, required=True, help="Output `.npz` fixture.")
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        required=True,
        help="Array key to copy. Repeat for multiple fields.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index for `--bout-dump` extraction.",
    )
    parser.add_argument("--x0", type=int, default=None, help="Inclusive x slice start.")
    parser.add_argument("--x1", type=int, default=None, help="Exclusive x slice end.")
    parser.add_argument("--y0", type=int, default=None, help="Inclusive y slice start.")
    parser.add_argument("--y1", type=int, default=None, help="Exclusive y slice end.")
    parser.add_argument(
        "--copy-key",
        action="append",
        dest="copy_keys",
        default=[],
        help="Extra scalar or metadata key to copy without slicing.",
    )
    return parser.parse_args()


def build_fixture(
    *,
    input_path: Path,
    output_path: Path,
    fields: list[str],
    x0: int | None,
    x1: int | None,
    y0: int | None,
    y1: int | None,
    copy_keys: list[str],
) -> None:
    xs = slice(x0, x1)
    ys = slice(y0, y1)
    payload: dict[str, np.ndarray] = {}

    with np.load(input_path, allow_pickle=False) as data:
        for name in fields:
            if name not in data:
                raise KeyError(f"Field {name!r} not found in {input_path}.")
            arr = np.asarray(data[name])
            if arr.ndim != 3:
                raise ValueError(
                    f"Field {name!r} has shape {arr.shape}; expected `(nz, nx, ny)` for slicing."
                )
            payload[name] = arr[:, xs, ys]

        for name in copy_keys:
            if name not in data:
                raise KeyError(f"Metadata key {name!r} not found in {input_path}.")
            payload[name] = np.asarray(data[name])

    payload["slice_x"] = np.asarray([x0 if x0 is not None else -1, x1 if x1 is not None else -1])
    payload["slice_y"] = np.asarray([y0 if y0 is not None else -1, y1 if y1 is not None else -1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def build_bout_dump_fixture(
    *,
    bout_dump: Path,
    output_path: Path,
    fields: list[str],
    time_index: int,
) -> None:
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("netCDF4 is required to read raw BOUT dumps.") from exc

    with Dataset(str(bout_dump)) as ds:
        mxg = int(np.asarray(ds.variables["MXG"][:]).reshape(-1)[0])
        myg = int(np.asarray(ds.variables["MYG"][:]).reshape(-1)[0])
        mxsub = int(np.asarray(ds.variables["MXSUB"][:]).reshape(-1)[0])
        mysub = int(np.asarray(ds.variables["MYSUB"][:]).reshape(-1)[0])
        xstart = mxg
        xend = mxg + mxsub - 1
        ystart = myg
        yend = myg + mysub - 1

        payload: dict[str, np.ndarray] = {
            "xstart": np.asarray(xstart, dtype=np.int32),
            "xend": np.asarray(xend, dtype=np.int32),
            "ystart": np.asarray(ystart, dtype=np.int32),
            "yend": np.asarray(yend, dtype=np.int32),
            "time_index": np.asarray(time_index, dtype=np.int32),
        }

        for name in fields:
            if name not in ds.variables:
                raise KeyError(f"Field {name!r} not found in {bout_dump}.")
            var = ds.variables[name]
            raw = (
                np.asarray(var[time_index], dtype=np.float64)
                if var.ndim == 4
                else np.asarray(var[:], dtype=np.float64)
            )
            if raw.ndim == 3:
                arr = np.transpose(raw, (2, 0, 1))
                payload[name] = arr

                avg_lower = arr[:, xstart, :].mean(axis=0, keepdims=True)
                avg_upper = arr[:, xend, :].mean(axis=0, keepdims=True)
                payload[f"{name}__neumann_lower"] = 2.0 * avg_lower - arr[:, xstart, :]
                payload[f"{name}__neumann_upper"] = 2.0 * avg_upper - arr[:, xend, :]
            elif raw.ndim == 2:
                payload[name] = raw
            else:
                raise ValueError(
                    f"Field {name!r} has shape {raw.shape}; expected local `(x, y, z)` or `(x, y)` data."
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = _parse_args()
    if args.bout_dump is not None:
        build_bout_dump_fixture(
            bout_dump=args.bout_dump,
            output_path=args.output,
            fields=list(args.fields),
            time_index=int(args.time_index),
        )
    else:
        build_fixture(
            input_path=args.input,
            output_path=args.output,
            fields=list(args.fields),
            x0=args.x0,
            x1=args.x1,
            y0=args.y0,
            y1=args.y1,
            copy_keys=list(args.copy_keys),
        )


if __name__ == "__main__":
    main()
