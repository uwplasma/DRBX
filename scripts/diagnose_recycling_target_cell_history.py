#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import advance_recycling_1d_implicit_history
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


DEFAULT_FIELDS = ("Nd+", "NVd+", "Pe", "Pd+", "Nd")


@dataclass(frozen=True)
class TargetCellSeriesSummary:
    field: str
    max_abs_diff: float
    worst_step: int
    worst_time: float
    native_value: float
    reference_value: float
    global_index: tuple[int, int, int]
    trimmed_index: tuple[int, int, int]


def target_cell_indices(
    mesh: StructuredMesh,
    *,
    x_index: int = 0,
    y_offset: int = 0,
    z_index: int = 0,
    target_edge: str = "upper",
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    x_local = int(x_index)
    z_local = int(z_index)
    x_global = mesh.xstart + x_local
    if target_edge == "upper":
        y_global = mesh.yend - int(y_offset)
    else:
        y_global = mesh.ystart + int(y_offset)
    if x_global < mesh.xstart or x_global > mesh.xend:
        raise IndexError(f"x index out of active range: {x_index}")
    if y_global < mesh.ystart or y_global > mesh.yend:
        raise IndexError(f"y offset out of active range: {y_offset}")
    if z_local < 0:
        raise IndexError(f"z index out of range: {z_index}")
    trimmed = (x_local, y_global - mesh.ystart, z_local)
    global_index = (x_global, y_global, z_local)
    return trimmed, global_index


def extract_active_cell_series(
    values: np.ndarray,
    mesh: StructuredMesh,
    *,
    x_index: int = 0,
    y_offset: int = 0,
    z_index: int = 0,
    target_edge: str = "upper",
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected a 4D time series, got shape {array.shape}")
    _, global_index = target_cell_indices(
        mesh,
        x_index=x_index,
        y_offset=y_offset,
        z_index=z_index,
        target_edge=target_edge,
    )
    xg, yg, zg = global_index
    return np.asarray(array[:, xg, yg, zg], dtype=np.float64)


def build_target_cell_summary(
    *,
    field: str,
    reference: np.ndarray,
    native: np.ndarray,
    time_points: np.ndarray,
    mesh: StructuredMesh,
    x_index: int = 0,
    y_offset: int = 0,
    z_index: int = 0,
    target_edge: str = "upper",
) -> TargetCellSeriesSummary:
    reference_series = extract_active_cell_series(
        reference,
        mesh,
        x_index=x_index,
        y_offset=y_offset,
        z_index=z_index,
        target_edge=target_edge,
    )
    native_series = extract_active_cell_series(
        native,
        mesh,
        x_index=x_index,
        y_offset=y_offset,
        z_index=z_index,
        target_edge=target_edge,
    )
    delta = np.abs(native_series - reference_series)
    worst_step = int(np.nanargmax(delta))
    trimmed_index, global_index = target_cell_indices(
        mesh,
        x_index=x_index,
        y_offset=y_offset,
        z_index=z_index,
        target_edge=target_edge,
    )
    return TargetCellSeriesSummary(
        field=field,
        max_abs_diff=float(delta[worst_step]),
        worst_step=worst_step,
        worst_time=float(time_points[worst_step]),
        native_value=float(native_series[worst_step]),
        reference_value=float(reference_series[worst_step]),
        global_index=global_index,
        trimmed_index=trimmed_index,
    )


def _case_input_path(case_name: str, reference_root: Path) -> Path:
    if case_name == "recycling_1d_one_step":
        return reference_root / "tests/integrated/1D-recycling/data/BOUT.inp"
    if case_name == "recycling_dthe_one_step":
        return reference_root / "tests/integrated/1D-recycling-dthe/data/BOUT.inp"
    raise ValueError(f"unsupported case: {case_name}")


def _solver_mode_for_case(case_name: str) -> str:
    return "continuation" if case_name == "recycling_1d_one_step" else "bdf"


def _load_reference_series(path: Path, *, fields: tuple[str, ...]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with Dataset(path) as dataset:
        time_points = np.asarray(dataset.variables["t_array"][:], dtype=np.float64)
        series = {
            field: np.asarray(dataset.variables[field][:], dtype=np.float64)
            for field in fields
            if field in dataset.variables
        }
    return series, time_points


def _native_history_series(
    *,
    case_name: str,
    reference_root: Path,
    timestep: float,
    steps: int,
    solver_mode: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, StructuredMesh]:
    input_path = _case_input_path(case_name, reference_root)
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
        timestep=timestep,
        steps=steps,
        solver_mode=solver_mode if solver_mode is not None else _solver_mode_for_case(case_name),
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=30,
    )
    time_points = np.asarray([index * timestep for index in range(steps + 1)], dtype=np.float64)
    return history.variable_history, time_points, mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track target-adjacent recycling drift over dense output substeps."
    )
    parser.add_argument("--case", default="recycling_1d_one_step", choices=("recycling_1d_one_step", "recycling_dthe_one_step"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument("--reference-workdir", type=Path)
    parser.add_argument("--timestep", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--solver-mode",
        help="Native recycling solver mode to compare against the same reference history.",
    )
    parser.add_argument("--field", action="append", dest="fields")
    parser.add_argument("--x-index", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0, help="Offset from the selected target edge in trimmed active-space.")
    parser.add_argument("--z-index", type=int, default=0)
    parser.add_argument("--target-edge", choices=("upper", "lower"), default="upper")
    parser.add_argument("--print-limit", type=int, default=12)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    fields = tuple(args.fields) if args.fields else DEFAULT_FIELDS
    def _run_dense_reference() -> Path:
        workdir = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{args.case}-target-history-"))
        reference_result = run_reference_case(
            args.case,
            reference_root=args.reference_root,
            reference_binary=args.reference_binary,
            workdir=workdir,
            extra_overrides=(f"nout={args.steps}", f"timestep={args.timestep:g}"),
            keep_workdir=True,
        )
        return Path(reference_result.summary.workdir)

    reference_workdir = _run_dense_reference() if args.reference_workdir is None else args.reference_workdir

    reference_dump = reference_workdir / "BOUT.dmp.0.nc"
    reference_series, reference_time = _load_reference_series(reference_dump, fields=fields)
    native_series, native_time, mesh = _native_history_series(
        case_name=args.case,
        reference_root=args.reference_root,
        timestep=float(args.timestep),
        steps=int(args.steps),
        solver_mode=args.solver_mode,
    )
    if reference_time.shape != native_time.shape:
        reference_workdir = _run_dense_reference()
        reference_dump = reference_workdir / "BOUT.dmp.0.nc"
        reference_series, reference_time = _load_reference_series(reference_dump, fields=fields)
        if reference_time.shape != native_time.shape:
            raise ValueError(f"time axis mismatch: reference {reference_time.shape}, native {native_time.shape}")

    summaries: list[TargetCellSeriesSummary] = []
    for field in fields:
        if field not in reference_series or field not in native_series:
            continue
        summaries.append(
            build_target_cell_summary(
                field=field,
                reference=reference_series[field],
                native=native_series[field],
                time_points=reference_time,
                mesh=mesh,
                x_index=args.x_index,
                y_offset=args.y_offset,
                z_index=args.z_index,
                target_edge=args.target_edge,
            )
        )
    summaries.sort(key=lambda item: item.max_abs_diff, reverse=True)

    trimmed_index, global_index = target_cell_indices(
        mesh,
        x_index=args.x_index,
        y_offset=args.y_offset,
        z_index=args.z_index,
        target_edge=args.target_edge,
    )
    print(f"case={args.case}")
    print(f"native_solver_mode={args.solver_mode or _solver_mode_for_case(args.case)}")
    print(f"reference_workdir={reference_workdir}")
    print(f"cell_trimmed={trimmed_index}")
    print(f"cell_global={global_index}")
    for summary in summaries[: max(int(args.print_limit), 1)]:
        print(
            f"{summary.field}: max_abs={summary.max_abs_diff:.8e} "
            f"worst_step={summary.worst_step} time={summary.worst_time:.8e} "
            f"native={summary.native_value:.8e} reference={summary.reference_value:.8e}"
        )

    if args.json_out is not None:
        payload: dict[str, Any] = {
            "case": args.case,
            "native_solver_mode": args.solver_mode or _solver_mode_for_case(args.case),
            "reference_workdir": str(reference_workdir),
            "cell_trimmed": trimmed_index,
            "cell_global": global_index,
            "fields": [asdict(summary) for summary in summaries],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
