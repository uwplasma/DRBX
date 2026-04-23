#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
from time import perf_counter
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a curated jax_drb case with optional JAX trace, device-memory "
            "snapshot, compilation cache, and XLA dump configuration."
        )
    )
    parser.add_argument("case_name", help="Curated case name understood by run_curated_case().")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Optional Hermes reference root. If omitted, the default discovery logic is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON summaries and profiler outputs. Defaults to profiles/<case_name>.",
    )
    parser.add_argument("--warm-runs", type=int, default=1, help="Number of untimed warm runs before profiling.")
    parser.add_argument("--timed-runs", type=int, default=2, help="Number of timed runs after warmup.")
    parser.add_argument("--cprofile-top", type=int, default=40, help="Number of cProfile rows to write.")
    parser.add_argument(
        "--jax-trace",
        action="store_true",
        help="Collect a JAX profiler trace in TensorBoard/Perfetto-compatible format.",
    )
    parser.add_argument(
        "--device-memory-profile",
        action="store_true",
        help="Capture a JAX device-memory profile after the profiled run.",
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=None,
        help="Optional JAX persistent compilation cache directory.",
    )
    parser.add_argument(
        "--xla-dump-dir",
        type=Path,
        default=None,
        help="Optional XLA dump directory. Adds --xla_dump_to and text HLO dumping to XLA_FLAGS.",
    )
    return parser.parse_args()


def _configure_environment(args: argparse.Namespace) -> None:
    if args.compilation_cache_dir is not None:
        cache_dir = args.compilation_cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    if args.xla_dump_dir is not None:
        dump_dir = args.xla_dump_dir.expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        xla_flags = os.environ.get("XLA_FLAGS", "").strip()
        additions = f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
        os.environ["XLA_FLAGS"] = f"{xla_flags} {additions}".strip()


def _block_result(result: Any) -> None:
    variables = getattr(result, "variables", None)
    if isinstance(variables, dict):
        for value in variables.values():
            blocker = getattr(value, "block_until_ready", None)
            if callable(blocker):
                blocker()


def _time_case(run_curated_case, jax, args: argparse.Namespace, *, trace_dir: Path | None, enable_cprofile: bool):
    profiler = cProfile.Profile() if enable_cprofile else None
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
        if profiler is not None:
            profiler.enable()
        started = perf_counter()
        result = run_curated_case(
            args.case_name,
            reference_root=args.reference_root,
        )
        _block_result(result)
        elapsed = perf_counter() - started
        if profiler is not None:
            profiler.disable()
    return result, elapsed, profiler


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def main() -> int:
    args = _parse_args()
    _configure_environment(args)

    import jax

    from jax_drb.native import run_curated_case

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (Path.cwd() / "profiles" / args.case_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "jax_trace" if args.jax_trace else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    warm_durations: list[float] = []
    for _ in range(max(0, args.warm_runs)):
        _, elapsed, _ = _time_case(run_curated_case, jax, args, trace_dir=None, enable_cprofile=False)
        warm_durations.append(float(elapsed))

    timed_durations: list[float] = []
    profiled_result = None
    profiled_elapsed = None
    profiler = None
    for timed_index in range(max(1, args.timed_runs)):
        use_trace = trace_dir if timed_index == 0 else None
        use_cprofile = timed_index == 0
        profiled_result, elapsed, profiler = _time_case(
            run_curated_case,
            jax,
            args,
            trace_dir=use_trace,
            enable_cprofile=use_cprofile,
        )
        timed_durations.append(float(elapsed))
        if timed_index == 0:
            profiled_elapsed = float(elapsed)

    cprofile_path = output_dir / "cprofile_top.txt"
    if profiler is not None:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(args.cprofile_top)
        cprofile_path.write_text(stream.getvalue(), encoding="utf-8")

    memory_profile_path = None
    if args.device_memory_profile:
        memory_profile_path = output_dir / "device_memory_profile.prof"
        jax.profiler.save_device_memory_profile(str(memory_profile_path))

    summary = {
        "case_name": args.case_name,
        "reference_root": None if args.reference_root is None else str(args.reference_root.expanduser().resolve()),
        "devices": [str(device) for device in jax.devices()],
        "default_backend": jax.default_backend(),
        "warm_run_count": len(warm_durations),
        "warm_run_seconds": warm_durations,
        "timed_run_count": len(timed_durations),
        "timed_run_seconds": timed_durations,
        "profiled_run_seconds": profiled_elapsed,
        "timed_run_mean_seconds": float(sum(timed_durations) / len(timed_durations)),
        "timed_run_min_seconds": float(min(timed_durations)),
        "timed_run_max_seconds": float(max(timed_durations)),
        "cprofile_top_path": str(cprofile_path),
        "jax_trace_dir": None if trace_dir is None else str(trace_dir),
        "device_memory_profile_path": None if memory_profile_path is None else str(memory_profile_path),
        "xla_dump_dir": None if args.xla_dump_dir is None else str(args.xla_dump_dir.expanduser().resolve()),
        "compilation_cache_dir": (
            None if args.compilation_cache_dir is None else str(args.compilation_cache_dir.expanduser().resolve())
        ),
        "compare_variable_count": None if profiled_result is None else len(getattr(profiled_result, "variables", {})),
    }
    summary_path = output_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(summary_path)
    print(cprofile_path)
    if trace_dir is not None:
        print(trace_dir)
    if memory_profile_path is not None:
        print(memory_profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
