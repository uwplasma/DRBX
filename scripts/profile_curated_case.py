#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
from collections.abc import Mapping
import io
import json
import os
from pathlib import Path
import pstats
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
    parts = resolved.parts
    if "hermes-3" in parts:
        index = parts.index("hermes-3")
        suffix = Path(*parts[index + 1 :]).as_posix() if parts[index + 1 :] else ""
        return "<reference-root>" if not suffix else f"<reference-root>/{suffix}"
    return resolved.name


def _sanitize_profile_text(text: str, *, reference_root: Path | None = None) -> str:
    """Remove local absolute paths from cProfile text artifacts."""

    replacements: list[tuple[Path, str]] = [(Path.cwd().resolve(), "<repo-root>")]
    if reference_root is not None:
        replacements.append((reference_root.expanduser().resolve(), "<reference-root>"))
    replacements.append((Path.home().resolve(), "<home>"))
    sanitized = text
    for path, label in replacements:
        sanitized = sanitized.replace(str(path), label)
    return sanitized


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
        help="Optional Hermes reference root. If omitted, JAX_DRB_REFERENCE_ROOT is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON summaries and profiler outputs. Defaults to profiles/<case_name>.",
    )
    parser.add_argument("--warm-runs", type=int, default=1, help="Number of untimed warm runs before profiling.")
    parser.add_argument("--timed-runs", type=int, default=2, help="Number of timed runs after warmup.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Extra BOUT.inp override forwarded to run_curated_case(), for example "
            "'runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp'. May be repeated."
        ),
    )
    parser.add_argument("--cprofile-top", type=int, default=40, help="Number of cProfile rows to write.")
    parser.add_argument(
        "--skip-cprofile",
        action="store_true",
        help="Skip cProfile text/binary dump generation.",
    )
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
        "--rss-profile",
        action="store_true",
        help="Sample process-tree peak RSS during timed runs.",
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
    parser.add_argument(
        "--require-native-diagnostic",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Require native_run_diagnostics[KEY] to stringify exactly to VALUE "
            "after the profiled run. May be repeated."
        ),
    )
    parser.add_argument(
        "--require-min-native-diagnostic",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Require native_run_diagnostics[KEY] to be numeric and at least VALUE "
            "after the profiled run. May be repeated."
        ),
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


def _json_ready_diagnostics(result: Any) -> dict[str, Any]:
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    converted: dict[str, Any] = {}
    for name, value in diagnostics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            converted[str(name)] = value
            continue
        item = getattr(value, "item", None)
        if callable(item):
            try:
                converted[str(name)] = item()
                continue
            except (TypeError, ValueError):
                pass
        converted[str(name)] = str(value)
    return converted


def _parse_key_value_requirement(requirement: str, *, option_name: str) -> tuple[str, str]:
    key, separator, value = requirement.partition("=")
    key = key.strip()
    value = value.strip()
    if not separator or not key or not value:
        raise ValueError(f"{option_name} requires KEY=VALUE, got {requirement!r}")
    return key, value


def _native_diagnostic_gate_errors(
    diagnostics: Mapping[str, Any],
    *,
    exact_requirements: tuple[str, ...] = (),
    minimum_requirements: tuple[str, ...] = (),
) -> list[str]:
    errors: list[str] = []
    for requirement in exact_requirements:
        key, expected = _parse_key_value_requirement(
            requirement, option_name="--require-native-diagnostic"
        )
        if key not in diagnostics:
            errors.append(f"native diagnostics did not report {key!r}")
            continue
        actual = diagnostics[key]
        if str(actual) != expected:
            errors.append(
                f"native diagnostics reported {key}={actual!r}, expected {expected!r}"
            )
    for requirement in minimum_requirements:
        key, minimum_text = _parse_key_value_requirement(
            requirement, option_name="--require-min-native-diagnostic"
        )
        try:
            minimum = float(minimum_text)
        except ValueError:
            errors.append(
                f"--require-min-native-diagnostic {requirement!r} has nonnumeric minimum"
            )
            continue
        if key not in diagnostics:
            errors.append(f"native diagnostics did not report {key!r}")
            continue
        actual = diagnostics[key]
        try:
            actual_value = float(actual)
        except (TypeError, ValueError):
            errors.append(
                f"native diagnostics reported nonnumeric {key}={actual!r}, "
                f"expected at least {minimum:g}"
            )
            continue
        if actual_value < minimum:
            errors.append(
                f"native diagnostics reported {key}={actual_value:g}, "
                f"expected at least {minimum:g}"
            )
    return errors


def _validate_native_diagnostic_requirements(
    *,
    exact_requirements: tuple[str, ...] = (),
    minimum_requirements: tuple[str, ...] = (),
) -> None:
    for requirement in exact_requirements:
        _parse_key_value_requirement(
            requirement, option_name="--require-native-diagnostic"
        )
    for requirement in minimum_requirements:
        _, minimum_text = _parse_key_value_requirement(
            requirement, option_name="--require-min-native-diagnostic"
        )
        try:
            float(minimum_text)
        except ValueError as exc:
            raise ValueError(
                "--require-min-native-diagnostic requires numeric VALUE, "
                f"got {requirement!r}"
            ) from exc


def _time_case(
    run_curated_case,
    jax,
    measure_peak_rss,
    args: argparse.Namespace,
    *,
    trace_dir: Path | None,
    enable_cprofile: bool,
    enable_rss_profile: bool,
):
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
        run_kwargs: dict[str, Any] = {}
        if args.reference_root is not None:
            run_kwargs["reference_root"] = args.reference_root

        def execute_case():
            run_result = run_curated_case(
                args.case_name,
                extra_overrides=tuple(args.override),
                **run_kwargs,
            )
            _block_result(run_result)
            return run_result

        started = perf_counter()
        if enable_rss_profile:
            result, rss_measurement = measure_peak_rss(execute_case)
        else:
            result = execute_case()
            rss_measurement = None
        elapsed = perf_counter() - started
        if profiler is not None:
            profiler.disable()
    return result, elapsed, profiler, rss_measurement


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def main() -> int:
    args = _parse_args()
    try:
        _validate_native_diagnostic_requirements(
            exact_requirements=tuple(args.require_native_diagnostic),
            minimum_requirements=tuple(args.require_min_native_diagnostic),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.reference_root is None:
        env_reference_root = os.environ.get("JAX_DRB_REFERENCE_ROOT")
        if env_reference_root:
            args.reference_root = Path(env_reference_root)
    if args.reference_root is None:
        raise SystemExit("profile_curated_case.py requires --reference-root or JAX_DRB_REFERENCE_ROOT.")
    _configure_environment(args)

    import jax

    from jax_drb.native import run_curated_case
    from jax_drb.runtime.memory import bytes_to_mebibytes, measure_peak_rss

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
        _, elapsed, _, _ = _time_case(
            run_curated_case,
            jax,
            measure_peak_rss,
            args,
            trace_dir=None,
            enable_cprofile=False,
            enable_rss_profile=False,
        )
        warm_durations.append(float(elapsed))

    timed_durations: list[float] = []
    rss_measurements: list[dict[str, object]] = []
    rss_profile_durations: list[float] = []
    profiled_result = None
    profiled_elapsed = None
    profiler = None
    for timed_index in range(max(1, args.timed_runs)):
        use_trace = trace_dir if timed_index == 0 else None
        use_cprofile = (timed_index == 0) and (not args.skip_cprofile)
        result, elapsed, run_profiler, rss_measurement = _time_case(
            run_curated_case,
            jax,
            measure_peak_rss,
            args,
            trace_dir=use_trace,
            enable_cprofile=use_cprofile,
            enable_rss_profile=args.rss_profile and not use_cprofile,
        )
        profiled_result = result
        timed_durations.append(float(elapsed))
        if rss_measurement is not None:
            rss_measurements.append(_rss_measurement_payload(rss_measurement, bytes_to_mebibytes))
            rss_profile_durations.append(float(elapsed))
        if timed_index == 0:
            profiled_elapsed = float(elapsed)
            profiler = run_profiler

    if args.rss_profile and not rss_measurements:
        result, elapsed, _, rss_measurement = _time_case(
            run_curated_case,
            jax,
            measure_peak_rss,
            args,
            trace_dir=None,
            enable_cprofile=False,
            enable_rss_profile=True,
        )
        profiled_result = result
        rss_profile_durations.append(float(elapsed))
        if rss_measurement is not None:
            rss_measurements.append(_rss_measurement_payload(rss_measurement, bytes_to_mebibytes))

    cprofile_path = output_dir / "cprofile_top.txt"
    cprofile_binary_path = output_dir / "cprofile_stats.pstats"
    if profiler is not None:
        cprofile_binary_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(cprofile_binary_path))
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(args.cprofile_top)
        cprofile_path.write_text(
            _sanitize_profile_text(stream.getvalue(), reference_root=args.reference_root),
            encoding="utf-8",
        )
    else:
        cprofile_binary_path = None

    memory_profile_path = None
    if args.device_memory_profile:
        memory_profile_path = output_dir / "device_memory_profile.prof"
        jax.profiler.save_device_memory_profile(str(memory_profile_path))

    peak_rss_values = [
        float(entry["peak_rss_mebibytes"])
        for entry in rss_measurements
        if entry.get("peak_rss_mebibytes") is not None
    ]
    native_diagnostics = _json_ready_diagnostics(profiled_result)
    gate_errors = _native_diagnostic_gate_errors(
        native_diagnostics,
        exact_requirements=tuple(args.require_native_diagnostic),
        minimum_requirements=tuple(args.require_min_native_diagnostic),
    )

    summary = {
        "case_name": args.case_name,
        "extra_overrides": list(args.override),
        "reference_root": None if args.reference_root is None else _sanitize_public_path(args.reference_root),
        "devices": [str(device) for device in jax.devices()],
        "default_backend": jax.default_backend(),
        "warm_run_count": len(warm_durations),
        "warm_run_seconds": warm_durations,
        "timed_run_count": len(timed_durations),
        "timed_run_seconds": timed_durations,
        "rss_profile_enabled": bool(args.rss_profile),
        "rss_profile_run_seconds": rss_profile_durations,
        "timed_run_peak_rss": rss_measurements,
        "timed_run_peak_rss_max_mebibytes": (
            None if not peak_rss_values else max(peak_rss_values)
        ),
        "profiled_run_seconds": profiled_elapsed,
        "timed_run_mean_seconds": float(sum(timed_durations) / len(timed_durations)),
        "timed_run_min_seconds": float(min(timed_durations)),
        "timed_run_max_seconds": float(max(timed_durations)),
        "cprofile_top_path": None if not cprofile_path.exists() else _sanitize_public_path(cprofile_path),
        "cprofile_binary_path": None if cprofile_binary_path is None else _sanitize_public_path(cprofile_binary_path),
        "jax_trace_dir": None if trace_dir is None else _sanitize_public_path(trace_dir),
        "device_memory_profile_path": None if memory_profile_path is None else _sanitize_public_path(memory_profile_path),
        "xla_dump_dir": None if args.xla_dump_dir is None else _sanitize_public_path(args.xla_dump_dir),
        "compilation_cache_dir": (
            None if args.compilation_cache_dir is None else _sanitize_public_path(args.compilation_cache_dir)
        ),
        "compare_variable_count": None if profiled_result is None else len(getattr(profiled_result, "variables", {})),
        "native_run_diagnostics": native_diagnostics,
        "native_diagnostic_gate_errors": gate_errors,
        "native_diagnostic_gate_passed": not gate_errors,
    }
    summary_path = output_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(summary_path)
    if cprofile_path.exists():
        print(cprofile_path)
    if cprofile_binary_path is not None:
        print(cprofile_binary_path)
    if trace_dir is not None:
        print(trace_dir)
    if memory_profile_path is not None:
        print(memory_profile_path)
    if gate_errors:
        for error in gate_errors:
            print(f"[profile-gate] {error}")
        return 2
    return 0


def _rss_measurement_payload(measurement, bytes_to_mebibytes) -> dict[str, object]:
    return {
        "status": measurement.status,
        "sample_count": int(measurement.sample_count),
        "sampling_interval_seconds": float(measurement.sampling_interval_seconds),
        "start_rss_bytes": measurement.start_rss_bytes,
        "end_rss_bytes": measurement.end_rss_bytes,
        "peak_rss_bytes": measurement.peak_rss_bytes,
        "peak_rss_delta_bytes": measurement.peak_rss_delta_bytes,
        "peak_rss_mebibytes": bytes_to_mebibytes(measurement.peak_rss_bytes),
        "peak_rss_delta_mebibytes": bytes_to_mebibytes(measurement.peak_rss_delta_bytes),
    }


if __name__ == "__main__":
    raise SystemExit(main())
