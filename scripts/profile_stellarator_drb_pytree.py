#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import time
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the non-axisymmetric 3D PyTree DRB lane with optional RSS, "
            "cProfile, JAX trace, device-memory profile, and XLA dump output."
        )
    )
    parser.add_argument("--nx", type=int, default=18, help="Radial grid points.")
    parser.add_argument("--ny", type=int, default=16, help="Poloidal grid points.")
    parser.add_argument("--nz", type=int, default=32, help="Toroidal grid points.")
    parser.add_argument("--steps", type=int, default=8, help="Explicit transient steps.")
    parser.add_argument("--warm-runs", type=int, default=1, help="Untimed warm profile builds before timed runs.")
    parser.add_argument("--timed-runs", type=int, default=2, help="Timed profile builds.")
    parser.add_argument("--cprofile-top", type=int, default=50, help="Rows to write in the cProfile text summary.")
    parser.add_argument("--skip-cprofile", action="store_true", help="Disable cProfile collection.")
    parser.add_argument("--rss-profile", action="store_true", help="Sample process-tree RSS during timed runs.")
    parser.add_argument("--jax-trace", action="store_true", help="Collect a Perfetto/TensorBoard-compatible JAX trace.")
    parser.add_argument(
        "--device-memory-profile",
        action="store_true",
        help="Capture a JAX device-memory pprof snapshot after the timed runs.",
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=Path(".jax_cache/stellarator_drb_pytree"),
        help="JAX persistent compilation cache directory.",
    )
    parser.add_argument(
        "--xla-dump-dir",
        type=Path,
        default=None,
        help="Optional XLA dump directory. Enables text HLO dumping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiles/stellarator_drb_pytree"),
        help="Output directory for profile summaries.",
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


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _run_campaign(nx: int, ny: int, nz: int, steps: int) -> dict[str, Any]:
    from dkx.validation.stellarator_drb_pytree_campaign import build_stellarator_drb_pytree_campaign

    report, _arrays = build_stellarator_drb_pytree_campaign(nx=nx, ny=ny, nz=nz, steps=steps)
    return report


def _time_one_run(
    *,
    args: argparse.Namespace,
    jax,
    measure_peak_rss,
    trace_dir: Path | None,
    enable_cprofile: bool,
    enable_rss_profile: bool,
) -> tuple[dict[str, Any], float, cProfile.Profile | None, Any | None]:
    profiler = cProfile.Profile() if enable_cprofile else None
    trace_context = (
        jax.profiler.trace(
            str(trace_dir),
            create_perfetto_link=False,
            create_perfetto_trace=True,
        )
        if trace_dir is not None
        else _NullContext()
    )

    def execute() -> dict[str, Any]:
        return _run_campaign(args.nx, args.ny, args.nz, args.steps)

    with trace_context:
        if profiler is not None:
            profiler.enable()
        started = time.perf_counter()
        if enable_rss_profile:
            report, rss_measurement = measure_peak_rss(execute)
        else:
            report = execute()
            rss_measurement = None
        elapsed = time.perf_counter() - started
        if profiler is not None:
            profiler.disable()
    return report, elapsed, profiler, rss_measurement


def main() -> int:
    args = _parse_args()
    _configure_environment(args)

    import jax

    if args.compilation_cache_dir is not None:
        jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache_dir.expanduser().resolve()))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    from dkx.runtime.memory import bytes_to_mebibytes, measure_peak_rss

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "jax_trace" if args.jax_trace else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    warm_seconds: list[float] = []
    for _ in range(max(0, args.warm_runs)):
        _report, elapsed, _profiler, _rss = _time_one_run(
            args=args,
            jax=jax,
            measure_peak_rss=measure_peak_rss,
            trace_dir=None,
            enable_cprofile=False,
            enable_rss_profile=False,
        )
        warm_seconds.append(float(elapsed))

    timed_seconds: list[float] = []
    run_reports: list[dict[str, Any]] = []
    rss_measurements: list[Any] = []
    profiled_run: cProfile.Profile | None = None
    for run_index in range(max(1, args.timed_runs)):
        use_cprofile = run_index == 0 and not args.skip_cprofile
        report, elapsed, profiler, rss_measurement = _time_one_run(
            args=args,
            jax=jax,
            measure_peak_rss=measure_peak_rss,
            trace_dir=trace_dir if run_index == 0 else None,
            enable_cprofile=use_cprofile,
            enable_rss_profile=bool(args.rss_profile and not use_cprofile),
        )
        run_reports.append(report)
        timed_seconds.append(float(elapsed))
        if profiler is not None:
            profiled_run = profiler
        if rss_measurement is not None:
            rss_measurements.append(rss_measurement)

    if args.rss_profile and not rss_measurements:
        report, elapsed, _profiler, rss_measurement = _time_one_run(
            args=args,
            jax=jax,
            measure_peak_rss=measure_peak_rss,
            trace_dir=None,
            enable_cprofile=False,
            enable_rss_profile=True,
        )
        run_reports.append(report)
        if rss_measurement is not None:
            rss_measurements.append(rss_measurement)

    cprofile_text_path = output_dir / "cprofile_top.txt"
    cprofile_stats_path = output_dir / "cprofile_stats.pstats"
    if profiled_run is not None:
        profiled_run.dump_stats(str(cprofile_stats_path))
        stream = io.StringIO()
        stats = pstats.Stats(profiled_run, stream=stream).sort_stats("cumtime")
        stats.print_stats(args.cprofile_top)
        cprofile_text_path.write_text(stream.getvalue(), encoding="utf-8")

    device_memory_profile_path = None
    if args.device_memory_profile:
        device_memory_profile_path = output_dir / "device_memory_profile.prof"
        jax.profiler.save_device_memory_profile(str(device_memory_profile_path))

    rss_payload = [
        {
            "status": item.status,
            "sample_count": int(item.sample_count),
            "sampling_interval_seconds": float(item.sampling_interval_seconds),
            "start_rss_mebibytes": bytes_to_mebibytes(item.start_rss_bytes),
            "end_rss_mebibytes": bytes_to_mebibytes(item.end_rss_bytes),
            "peak_rss_mebibytes": bytes_to_mebibytes(item.peak_rss_bytes),
            "peak_rss_delta_mebibytes": bytes_to_mebibytes(item.peak_rss_delta_bytes),
        }
        for item in rss_measurements
    ]
    peak_rss = [entry["peak_rss_mebibytes"] for entry in rss_payload if entry["peak_rss_mebibytes"] is not None]
    last_report = run_reports[-1]
    summary = {
        "case": "stellarator_drb_pytree_profile",
        "grid": {"nx": int(args.nx), "ny": int(args.ny), "nz": int(args.nz), "steps": int(args.steps)},
        "devices": [str(device) for device in jax.devices()],
        "local_devices": [str(device) for device in jax.local_devices()],
        "local_device_count": int(jax.local_device_count()),
        "default_backend": jax.default_backend(),
        "warm_run_seconds": warm_seconds,
        "timed_run_seconds": timed_seconds,
        "timed_run_min_seconds": float(min(timed_seconds)),
        "timed_run_mean_seconds": float(sum(timed_seconds) / len(timed_seconds)),
        "rss_profile": rss_payload,
        "peak_rss_max_mebibytes": None if not peak_rss else float(max(peak_rss)),
        "campaign_passed": bool(last_report["passed"]),
        "campaign_jvp_relative_error": float(last_report["jvp_relative_error"]),
        "campaign_vmap_serial_linf": float(last_report["vmap_serial_linf"]),
        "campaign_warm_execute_seconds": float(last_report["warm_execute_seconds"]),
        "campaign_batch_sizes": last_report["batch_sizes"],
        "campaign_batch_throughput_cases_per_second": last_report["batch_throughput_cases_per_second"],
        "campaign_pmap_execute_seconds": last_report["pmap_execute_seconds"],
        "compilation_cache_dir": (
            None if args.compilation_cache_dir is None else str(args.compilation_cache_dir.expanduser().resolve())
        ),
        "xla_dump_dir": None if args.xla_dump_dir is None else str(args.xla_dump_dir.expanduser().resolve()),
        "jax_trace_dir": None if trace_dir is None else str(trace_dir),
        "cprofile_top_path": None if not cprofile_text_path.exists() else str(cprofile_text_path),
        "cprofile_stats_path": None if not cprofile_stats_path.exists() else str(cprofile_stats_path),
        "device_memory_profile_path": None if device_memory_profile_path is None else str(device_memory_profile_path),
    }
    summary_path = output_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path)
    if cprofile_text_path.exists():
        print(cprofile_text_path)
    if device_memory_profile_path is not None:
        print(device_memory_profile_path)
    if trace_dir is not None:
        print(trace_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
