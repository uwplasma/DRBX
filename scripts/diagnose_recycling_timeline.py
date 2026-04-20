#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.native import run_curated_case
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.parity.reference import resolve_reference_case, run_reference_case
from jax_drb.config.boutinp import load_bout_input
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_FIELDS = (
    "Nd+",
    "Pd+",
    "NVd+",
    "Nt+",
    "Pt+",
    "NVt+",
    "Nhe+",
    "Phe+",
    "NVhe+",
    "Nd",
    "Pd",
    "NVd",
    "Nt",
    "Pt",
    "NVt",
    "Nhe",
    "Phe",
    "NVhe",
    "Pe",
    "Ve",
    "Epar",
)

FIELD_PATTERNS = (
    re.compile(r"^SNV"),
    re.compile(r"^S[A-Za-z0-9+]+_feedback$"),
    re.compile(r"^density_feedback_src_(?:mult|p|i|shape)_[A-Za-z0-9+]+$"),
    re.compile(r"^DivPiPar_"),
    re.compile(r"^F.*_(?:coll|iz|rec|cx)$"),
    re.compile(r".*_target_recycle$"),
)


def _load_series(path: Path) -> tuple[dict[str, np.ndarray], tuple[float, ...]]:
    with Dataset(path) as dataset:
        time_points = tuple(float(value) for value in dataset.variables["t_array"][:]) if "t_array" in dataset.variables else ()
        variables = {
            name: np.asarray(variable[:], dtype=np.float64)
            for name, variable in dataset.variables.items()
            if getattr(variable, "ndim", 0) > 0 and name not in {"t_array"}
        }
    return variables, time_points


def _trim_series(series: dict[str, np.ndarray], *, mesh, trim_x: bool, trim_y: bool) -> dict[str, np.ndarray]:
    if not trim_x and not trim_y:
        return series
    trimmed: dict[str, np.ndarray] = {}
    for name, values in series.items():
        if values.ndim < 3:
            trimmed[name] = values
            continue
        slices: list[slice] = [slice(None)] * values.ndim
        if trim_x:
            slices[1] = slice(mesh.xstart, mesh.xend + 1)
        if trim_y:
            slices[2] = slice(mesh.ystart, mesh.yend + 1)
            trimmed[name] = values[tuple(slices)]
    return trimmed


def _shared_fields(reference: dict[str, np.ndarray], native: dict[str, np.ndarray], requested: tuple[str, ...]) -> tuple[str, ...]:
    if requested:
        return tuple(name for name in requested if name in reference and name in native)
    names = [name for name in DEFAULT_FIELDS if name in reference and name in native]
    extra = [
        name
        for name in sorted(set(reference) & set(native))
        if name not in names and any(pattern.search(name) for pattern in FIELD_PATTERNS)
    ]
    return tuple(names + extra)


def _diagnostic_fields(series: dict[str, np.ndarray]) -> tuple[str, ...]:
    return tuple(
        name
        for name in sorted(series)
        if any(pattern.search(name) for pattern in FIELD_PATTERNS)
    )


def _compare_frame(
    reference: np.ndarray,
    native: np.ndarray,
    *,
    target_edge: str,
    target_band_width: int,
) -> tuple[float, tuple[int, ...] | None, float, float, tuple[tuple[int, float], ...]]:
    delta = np.abs(np.asarray(native, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    if delta.ndim >= 3:
        width = max(int(target_band_width), 1)
        if target_edge == "upper":
            y_start = max(delta.shape[1] - width, 0)
        else:
            y_start = 0
        band = delta[:, y_start:, ...]
        band_rows = tuple(
            (y_start + local_y, float(np.nanmax(band[:, local_y, ...])))
            for local_y in range(band.shape[1])
        )
        delta = band
        ref_frame = np.asarray(reference, dtype=np.float64)[:, y_start:, ...]
        native_frame = np.asarray(native, dtype=np.float64)[:, y_start:, ...]
        if delta.size == 0:
            return 0.0, None, float("nan"), float("nan"), ()
        flat_index = int(np.nanargmax(delta))
        location = np.unravel_index(flat_index, delta.shape)
        x_local, y_local, *rest = location
        global_location = (x_local, y_start + y_local, *rest)
        max_abs = float(delta[location])
        return max_abs, global_location, float(native_frame[location]), float(ref_frame[location]), band_rows
    if delta.size == 0:
        return 0.0, None, float("nan"), float("nan"), ()
    flat_index = int(np.nanargmax(delta))
    location = np.unravel_index(flat_index, delta.shape)
    max_abs = float(delta[location])
    return max_abs, location, float(np.asarray(native, dtype=np.float64)[location]), float(np.asarray(reference, dtype=np.float64)[location]), ()


def _format_location(location: tuple[int, ...] | None) -> str:
    if location is None:
        return "n/a"
    return "(" + ", ".join(str(index) for index in location) + ")"


def _target_band_rows(mesh, *, target_edge: str, target_band_width: int) -> tuple[int, ...]:
    width = max(int(target_band_width), 1)
    if target_edge == "upper":
        start = max(mesh.yend - width + 1, mesh.ystart)
        return tuple(range(start, mesh.yend + 1))
    stop = min(mesh.ystart + width, mesh.yend + 1)
    return tuple(range(mesh.ystart, stop))


def _matching_diagnostics(reference: dict[str, np.ndarray], native: dict[str, np.ndarray]) -> tuple[str, ...]:
    names = [
        name
        for name in sorted(set(reference) & set(native))
        if name not in DEFAULT_FIELDS and any(pattern.search(name) for pattern in FIELD_PATTERNS)
    ]
    return tuple(names)


def _print_series_rows(
    *,
    header: str,
    fields: tuple[str, ...],
    reference_vars: dict[str, np.ndarray],
    native_vars: dict[str, np.ndarray],
    time_index: int,
    target_edge: str,
    target_band_width: int,
    max_fields: int,
) -> None:
    rows: list[tuple[str, float, tuple[int, ...] | None, float, float, tuple[tuple[int, float], ...]]] = []
    for name in fields:
        reference_series = reference_vars.get(name)
        native_series = native_vars.get(name)
        if reference_series is None or native_series is None:
            continue
        if reference_series.shape[0] <= time_index or native_series.shape[0] <= time_index:
            continue
        rows.append(
            (
                name,
                *_compare_frame(
                    reference_series[time_index],
                    native_series[time_index],
                    target_edge=target_edge,
                    target_band_width=target_band_width,
                ),
            )
        )
    rows.sort(key=lambda item: item[1], reverse=True)
    if not rows:
        print(f"{header}: none")
        return
    print(header)
    for name, max_abs, location, native_value, reference_value, band_rows in rows[: max(int(max_fields), 1)]:
        row_text = ""
        if band_rows:
            row_text = " band_rows=" + ", ".join(f"y={y}: {value:.8e}" for y, value in band_rows)
        print(
            f"  {name}: max_abs={max_abs:.8e} loc={_format_location(location)} native={native_value:.8e} reference={reference_value:.8e}{row_text}"
        )
    worst_name, worst_abs, worst_loc, worst_native, worst_reference, _ = rows[0]
    print(
        f"  worst={worst_name} diff={worst_abs:.8e} loc={_format_location(worst_loc)} native={worst_native:.8e} reference={worst_reference:.8e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reference and native recycling outputs timestep-by-timestep on the target-adjacent cells."
    )
    parser.add_argument("--case", required=True, choices=("recycling_1d_one_step", "recycling_dthe_one_step"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument("--reference-workdir", type=Path, help="Use an existing reference workdir instead of running a fresh reference case.")
    parser.add_argument("--reference-override", action="append", default=(), help="Extra override to pass to the reference run. May be repeated.")
    parser.add_argument("--field", action="append", dest="fields", help="Restrict the report to these fields. May be repeated.")
    parser.add_argument("--target-edge", choices=("upper", "lower"), default="upper", help="Which target-adjacent edge to inspect.")
    parser.add_argument("--target-band-width", type=int, default=2, help="How many y cells to include in the target-adjacent band.")
    parser.add_argument("--max-fields", type=int, default=12, help="Maximum number of fields to print per timestep.")
    parser.add_argument("--keep-reference-workdir", action="store_true", help="Keep the temporary reference workdir when the script runs the reference case itself.")
    args = parser.parse_args()

    case, input_path = resolve_reference_case(args.case, reference_root=args.reference_root)
    if args.reference_workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{args.case}-timeline-"))
        reference_result = run_reference_case(
            args.case,
            reference_root=args.reference_root,
            reference_binary=args.reference_binary,
            workdir=workdir,
            extra_overrides=tuple(args.reference_override),
            keep_workdir=args.keep_reference_workdir,
        )
        reference_workdir = Path(reference_result.summary.workdir)
        reference_time_points = tuple(float(value) for value in reference_result.summary.time_points)
    else:
        reference_workdir = args.reference_workdir
        reference_time_points = ()

    reference_dump = reference_workdir / "BOUT.dmp.0.nc"
    if not reference_dump.exists():
        raise FileNotFoundError(f"Reference dump not found: {reference_dump}")

    native_result = None
    native_time_points: tuple[float, ...] = ()
    native_error: str | None = None
    try:
        native_result = run_curated_case(args.case, reference_root=args.reference_root)
        native_time_points = tuple(float(value) for value in native_result.time_points)
    except Exception as exc:  # pragma: no cover - diagnostic path
        native_error = f"{type(exc).__name__}: {exc}"

    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    build_structured_metrics(config, run_config, mesh)

    reference_vars, dump_time_points = _load_series(reference_dump)
    reference_vars = _trim_series(reference_vars, mesh=mesh, trim_x=case.trim_x_guards, trim_y=case.trim_y_guards)
    native_vars = {name: np.asarray(value, dtype=np.float64) for name, value in native_result.variables.items()} if native_result else {}
    compare_fields = _shared_fields(reference_vars, native_vars, tuple(args.fields) if args.fields else ()) if native_result else ()
    reference_diagnostics = _diagnostic_fields(reference_vars)
    native_diagnostics = _diagnostic_fields(native_vars)
    matching_diagnostics = _matching_diagnostics(reference_vars, native_vars) if native_result else ()
    target_rows = _target_band_rows(mesh, target_edge=args.target_edge, target_band_width=args.target_band_width)

    if reference_time_points and native_time_points and len(reference_time_points) != len(native_time_points):
        print(f"warning: reference time points ({len(reference_time_points)}) != native time points ({len(native_time_points)})")
    if dump_time_points and native_time_points and len(dump_time_points) != len(native_time_points):
        print(f"warning: dump time points ({len(dump_time_points)}) != native time points ({len(native_time_points)})")

    print(f"case={args.case}")
    print(f"reference_workdir={reference_workdir}")
    print(f"target_edge={args.target_edge} target_band_width={args.target_band_width} target_rows={target_rows}")
    print(f"reference_diagnostics={len(reference_diagnostics)}")
    if reference_diagnostics:
        print(", ".join(reference_diagnostics))
    if native_error is not None:
        print(f"native_error={native_error}")
    else:
        print(f"native_diagnostics={len(native_diagnostics)}")
        if native_diagnostics:
            print(", ".join(native_diagnostics))
    print(f"matching_diagnostics={len(matching_diagnostics)}")
    if matching_diagnostics:
        print(", ".join(matching_diagnostics))
    print(f"compare_fields={len(compare_fields)}")
    if compare_fields:
        print(", ".join(compare_fields))

    if native_error is not None or not compare_fields:
        if native_error is not None:
            print("native comparison unavailable")
        elif not compare_fields:
            print("no shared fields found")
        return

    time_count = min(len(native_time_points) or len(reference_time_points) or 0, len(reference_time_points) or len(native_time_points) or 0)
    if time_count == 0 and compare_fields:
        sample = next(iter(compare_fields))
        time_count = min(reference_vars[sample].shape[0], native_vars[sample].shape[0])

    for time_index in range(time_count):
        time_label = native_time_points[time_index] if time_index < len(native_time_points) else (
            reference_time_points[time_index] if time_index < len(reference_time_points) else float(time_index)
        )
        print(f"step={time_index} time={time_label:.12e}")
        _print_series_rows(
            header="  state",
            fields=compare_fields,
            reference_vars=reference_vars,
            native_vars=native_vars,
            time_index=time_index,
            target_edge=args.target_edge,
            target_band_width=args.target_band_width,
            max_fields=args.max_fields,
        )
        _print_series_rows(
            header="  diagnostics",
            fields=matching_diagnostics,
            reference_vars=reference_vars,
            native_vars=native_vars,
            time_index=time_index,
            target_edge=args.target_edge,
            target_band_width=args.target_band_width,
            max_fields=min(max(int(args.max_fields), 1), 8),
        )


if __name__ == "__main__":
    main()
