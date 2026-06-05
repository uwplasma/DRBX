#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REFERENCE_ROOT = REPO_ROOT / "tests" / "fixtures" / "reference-root"

REFERENCE_INPUT_RELATIVE_PATHS = {
    "hydrogen": Path("tests") / "integrated" / "1D-recycling" / "data" / "BOUT.inp",
    "dthe": Path("tests") / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
}


def _sanitize_public_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        pass
    home = os.environ.get("HOME")
    if home:
        home_path = Path(home).expanduser().resolve()
        try:
            return f"~/{resolved.relative_to(home_path).as_posix()}"
        except ValueError:
            pass
    return resolved.as_posix()


def _resolve_reference_root(args: argparse.Namespace) -> Path | None:
    if args.reference_root is not None:
        return args.reference_root.expanduser().resolve()
    value = os.environ.get("JAX_DRB_REFERENCE_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return FIXTURE_REFERENCE_ROOT if FIXTURE_REFERENCE_ROOT.exists() else None


def _reference_input_relative_path(case: str) -> Path:
    return REFERENCE_INPUT_RELATIVE_PATHS[case]


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input_path is not None:
        input_path = args.input_path.expanduser().resolve()
        if not input_path.is_file():
            raise SystemExit(f"--input-path {input_path} does not exist or is not a file.")
        return input_path
    root = _resolve_reference_root(args)
    if root is None:
        relative_path = _reference_input_relative_path(args.case)
        raise SystemExit(
            "--reference-root, --input-path, or JAX_DRB_REFERENCE_ROOT is required. "
            f"Expected reference-root input: {relative_path.as_posix()}. "
            "The source checkout normally provides lightweight fixture decks under "
            "tests/fixtures/reference-root; for nonstandard staged decks, pass --input-path /path/to/BOUT.inp."
        )
    relative_path = _reference_input_relative_path(args.case)
    input_path = (root / relative_path).resolve()
    if not input_path.is_file():
        raise SystemExit(
            f"reference root {root} is missing required input {relative_path.as_posix()}; "
            "set --reference-root/JAX_DRB_REFERENCE_ROOT to a root containing that file, "
            "or pass --input-path /path/to/BOUT.inp for nonstandard staged decks."
        )
    return input_path


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile batched residual/JVP throughput for the fixed-layout recycling residual. "
            "This is the differentiability and parallel-throughput gate used before promoting "
            "heavier recycling solves."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--case", choices=("hydrogen", "dthe"), default="dthe")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs") / "data" / "runtime_profile_artifacts" / "recycling_dthe_batched_jvp_gate",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=(1, 4, 16, 64))
    parser.add_argument("--timestep", type=float, default=1.0e-4)
    parser.add_argument("--perturbation-scale", type=float, default=1.0e-6)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--disable-pmap", action="store_true")
    parser.add_argument("--jax-trace", action="store_true")
    parser.add_argument("--device-memory-profile", action="store_true")
    parser.add_argument("--compilation-cache-dir", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-objective-grad-check",
        action="store_true",
        help="Skip the reverse-mode objective-gradient check for bounded GPU throughput runs.",
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
        existing = os.environ.get("XLA_FLAGS", "").strip()
        additions = f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
        os.environ["XLA_FLAGS"] = f"{existing} {additions}".strip()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_environment(args)
    trace_dir = args.output_dir / "jax_trace" if args.jax_trace else None
    if trace_dir is not None:
        if trace_dir.exists():
            shutil.rmtree(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

    import jax

    from jax_drb.validation.recycling_batched_jvp_profile import create_recycling_batched_jvp_profile_package

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
        report = create_recycling_batched_jvp_profile_package(
            _resolve_input(args),
            args.output_dir,
            overrides=tuple(args.override),
            batch_sizes=tuple(args.batch_sizes),
            timestep=float(args.timestep),
            perturbation_scale=float(args.perturbation_scale),
            fd_epsilon=float(args.fd_epsilon),
            timed_runs=int(args.timed_runs),
            enable_pmap=not bool(args.disable_pmap),
            check_objective_grad=not bool(args.skip_objective_grad_check),
        )

    memory_profile_path = None
    if args.device_memory_profile:
        memory_profile_path = args.output_dir / "device_memory_profile.prof"
        jax.profiler.save_device_memory_profile(str(memory_profile_path))

    report = {
        **report,
        "jax_trace_dir": None if trace_dir is None else _sanitize_public_path(trace_dir),
        "device_memory_profile_path": None if memory_profile_path is None else _sanitize_public_path(memory_profile_path),
        "xla_dump_dir": None if args.xla_dump_dir is None else _sanitize_public_path(args.xla_dump_dir),
        "compilation_cache_dir": (
            None if args.compilation_cache_dir is None else _sanitize_public_path(args.compilation_cache_dir)
        ),
    }
    (args.output_dir / "profile_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
