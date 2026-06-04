#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import shutil
import statistics
from time import perf_counter
from typing import Any


def _sanitize_public_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        pass
    for base_name in ("HOME",):
        base_value = os.environ.get(base_name)
        if not base_value:
            continue
        base_path = Path(base_value).expanduser().resolve()
        try:
            return f"~/{resolved.relative_to(base_path).as_posix()}"
        except ValueError:
            pass
    return resolved.as_posix()


def _public_input_path(args: argparse.Namespace, input_path: Path) -> str:
    root = args.reference_root
    if root is None:
        env_root = os.environ.get("JAX_DRB_REFERENCE_ROOT")
        root = Path(env_root) if env_root else None
    if root is not None:
        try:
            return f"<reference-root>/{input_path.resolve().relative_to(root.expanduser().resolve()).as_posix()}"
        except ValueError:
            pass
    return f"<input-path>/{input_path.name}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the real 1D-recycling fixed-layout backward-Euler gate "
            "through the JAX-linearized Newton path."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Hermès reference root. Falls back to JAX_DRB_REFERENCE_ROOT.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Explicit 1D-recycling BOUT.inp. Defaults under --reference-root.",
    )
    parser.add_argument(
        "--case",
        choices=("hydrogen", "dthe"),
        default="hydrogen",
        help="Reference integrated recycling deck to profile when --input-path is not supplied.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs") / "data" / "runtime_profile_artifacts" / "recycling_1d_jax_linearized_gate",
    )
    parser.add_argument("--timestep", type=float, default=1.0e-6)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-nonlinear-iterations", type=int, default=1)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "BOUT.inp override such as 'mesh:ny=100'. May be repeated. "
            "Use this for heavier real-kernel CPU/GPU scaling gates without "
            "copying large input decks into the repository."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run this many unprofiled solves before timing to amortize JAX compilation.",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Run this many timed solves after warmup. The first timed solve is optionally cProfile/JAX-traced.",
    )
    parser.add_argument(
        "--linear-solver-backend",
        choices=("jax_gmres", "lineax_gmres"),
        default="jax_gmres",
        help="Linear solver backend for nontrivial JAX-linearized Newton updates.",
    )
    parser.add_argument("--cprofile-top", type=int, default=40)
    parser.add_argument("--skip-cprofile", action="store_true")
    parser.add_argument("--rss-profile", action="store_true")
    parser.add_argument("--jax-trace", action="store_true")
    parser.add_argument("--device-memory-profile", action="store_true")
    parser.add_argument("--compilation-cache-dir", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    return parser.parse_args()


def _configure_environment(args: argparse.Namespace) -> None:
    if args.compilation_cache_dir is not None:
        cache_dir = args.compilation_cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    if args.xla_dump_dir is not None:
        dump_dir = args.xla_dump_dir.expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        existing = os.environ.get("XLA_FLAGS", "").strip()
        additions = f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
        os.environ["XLA_FLAGS"] = f"{existing} {additions}".strip()


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input_path is not None:
        return args.input_path.expanduser().resolve()
    root = args.reference_root
    if root is None:
        env_root = os.environ.get("JAX_DRB_REFERENCE_ROOT")
        root = Path(env_root) if env_root else None
    if root is None:
        raise SystemExit("--reference-root or JAX_DRB_REFERENCE_ROOT is required.")
    case_dir = "1D-recycling-dthe" if args.case == "dthe" else "1D-recycling"
    return (root.expanduser().resolve() / "tests" / "integrated" / case_dir / "data" / "BOUT.inp").resolve()


def _solver_mode_for_backend(linear_solver_backend: str) -> str:
    return "jax_linearized_lineax" if str(linear_solver_backend) == "lineax_gmres" else "jax_linearized"


def _profile_once(args: argparse.Namespace, input_path: Path) -> tuple[dict[str, Any], float]:
    from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
    from jax_drb.native.mesh import build_structured_mesh
    from jax_drb.native.metrics import build_structured_metrics
    from jax_drb.native.recycling_1d import (
        _build_recycling_runtime_model,
        _build_recycling_state_fields,
        advance_recycling_1d_backward_euler_step,
    )
    from jax_drb.native.units import resolved_dataset_scalars
    from jax_drb.runtime.run_config import RunConfiguration

    config = load_bout_input(input_path)
    if args.override:
        config = apply_bout_overrides(config, args.override)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}
    solver_mode = _solver_mode_for_backend(args.linear_solver_backend)

    started = perf_counter()
    next_fields, _next_integrals, info = advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=float(args.timestep),
        solver_mode=solver_mode,
        residual_tolerance=float(args.residual_tolerance),
        max_nonlinear_iterations=int(args.max_nonlinear_iterations),
    )
    elapsed = perf_counter() - started
    variable_cells = {
        name: int(getattr(value, "size", 0))
        for name, value in next_fields.items()
        if name in runtime_model.field_names
    }
    report = {
        "input_path": _public_input_path(args, input_path),
        "case": str(args.case),
        "solver_mode": solver_mode,
        "linear_solver_backend": str(args.linear_solver_backend),
        "overrides": list(args.override),
        "warmup_runs": int(max(args.warmup_runs, 0)),
        "timestep": float(args.timestep),
        "residual_tolerance": float(args.residual_tolerance),
        "max_nonlinear_iterations": int(args.max_nonlinear_iterations),
        "field_names": list(runtime_model.field_names),
        "feedback_names": list(runtime_model.feedback_names),
        "mesh_active_shape": [
            int(mesh.xend - mesh.xstart + 1),
            int(mesh.yend - mesh.ystart + 1),
            int(mesh.nz),
        ],
        "active_size": int(info.active_size),
        "variable_cell_count": variable_cells,
        "state_size": int(sum(variable_cells.values()) + len(runtime_model.feedback_names)),
        "residual_inf_norm": float(info.residual_inf_norm),
        "nonlinear_iterations": int(info.nonlinear_iterations),
        "linear_iterations": int(info.linear_iterations),
        "linear_solver_status": info.diagnostics.get("linear_solver_status"),
        "linear_solver_success": info.diagnostics.get("linear_solver_success"),
        "diagnostics": dict(info.diagnostics),
    }
    return report, elapsed


def _run_with_optional_profile(args: argparse.Namespace, input_path: Path, jax):
    warmup_elapsed: list[float] = []
    for _ in range(max(int(args.warmup_runs), 0)):
        _, elapsed = _profile_once(args, input_path)
        warmup_elapsed.append(float(elapsed))

    profiler = None if args.skip_cprofile else cProfile.Profile()
    if profiler is not None:
        profiler.enable()
    trace_dir = args.output_dir / "jax_trace" if args.jax_trace else None
    if trace_dir is not None:
        if trace_dir.exists():
            shutil.rmtree(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
    trace_cm = (
        jax.profiler.trace(
            str(trace_dir),
            create_perfetto_link=False,
            create_perfetto_trace=True,
        )
        if trace_dir is not None
        else _NullContext()
    )
    with trace_cm:
        report, elapsed = _profile_once(args, input_path)
    if profiler is not None:
        profiler.disable()
    timed_elapsed = [float(elapsed)]
    timed_residuals = [float(report["residual_inf_norm"])]
    for _ in range(max(int(args.timed_runs), 1) - 1):
        extra_report, extra_elapsed = _profile_once(args, input_path)
        timed_elapsed.append(float(extra_elapsed))
        timed_residuals.append(float(extra_report["residual_inf_norm"]))
    return report, elapsed, profiler, trace_dir, warmup_elapsed, timed_elapsed, timed_residuals


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def main() -> int:
    args = _parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_environment(args)
    input_path = _resolve_input(args)

    import jax

    from jax_drb.runtime.memory import bytes_to_mebibytes, measure_peak_rss

    profile_report, elapsed, profiler, trace_dir, warmup_elapsed, timed_elapsed, timed_residuals = _run_with_optional_profile(
        args, input_path, jax
    )
    rss_payload = None
    rss_elapsed = None
    if args.rss_profile:
        (rss_report, rss_elapsed), rss_measurement = measure_peak_rss(lambda: _profile_once(args, input_path))
        rss_payload = {
            "status": rss_measurement.status,
            "sample_count": int(rss_measurement.sample_count),
            "sampling_interval_seconds": float(rss_measurement.sampling_interval_seconds),
            "run_seconds": float(rss_elapsed),
            "residual_inf_norm": float(rss_report["residual_inf_norm"]),
            "peak_rss_mebibytes": bytes_to_mebibytes(rss_measurement.peak_rss_bytes),
            "peak_rss_delta_mebibytes": bytes_to_mebibytes(rss_measurement.peak_rss_delta_bytes),
        }

    cprofile_path = args.output_dir / "cprofile_top.txt"
    cprofile_binary_path = args.output_dir / "cprofile_stats.pstats"
    if profiler is not None:
        profiler.dump_stats(str(cprofile_binary_path))
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(int(args.cprofile_top))
        cprofile_path.write_text(stream.getvalue(), encoding="utf-8")

    memory_profile_path = None
    if args.device_memory_profile:
        memory_profile_path = args.output_dir / "device_memory_profile.prof"
        jax.profiler.save_device_memory_profile(str(memory_profile_path))

    summary = {
        "case": f"recycling_1d_{args.case}_jax_linearized_gate",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "profiled_run_seconds": float(elapsed),
        "timed_runs": int(max(int(args.timed_runs), 1)),
        "timed_run_seconds": timed_elapsed,
        "timed_run_seconds_median": float(statistics.median(timed_elapsed)),
        "timed_run_residual_inf_norms": timed_residuals,
        "warmup_run_seconds": warmup_elapsed,
        "rss_profile": rss_payload,
        "profile": profile_report,
        "cprofile_top_path": None if profiler is None else _sanitize_public_path(cprofile_path),
        "cprofile_binary_path": None if profiler is None else _sanitize_public_path(cprofile_binary_path),
        "jax_trace_dir": None if trace_dir is None else _sanitize_public_path(trace_dir),
        "device_memory_profile_path": None if memory_profile_path is None else _sanitize_public_path(memory_profile_path),
        "xla_dump_dir": None if args.xla_dump_dir is None else _sanitize_public_path(args.xla_dump_dir),
        "compilation_cache_dir": (
            None if args.compilation_cache_dir is None else _sanitize_public_path(args.compilation_cache_dir)
        ),
        "interpretation": (
            "This gate profiles a real integrated recycling fixed-layout "
            "residual that reaches JAX linearization. The D/T/He mode exercises "
            "the multispecies residual seam used by the adaptive BDF trial solves; "
            "it is still a controlled BE gate, not a full production output-window profile."
        ),
    }
    summary_path = args.output_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path)
    if cprofile_path.exists():
        print(cprofile_path)
    if trace_dir is not None:
        print(trace_dir)
    if memory_profile_path is not None:
        print(memory_profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
