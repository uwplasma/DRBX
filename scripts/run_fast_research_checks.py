#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_COVERAGE_TARGET = "src/jax_drb"


@dataclass(frozen=True)
class PytestSlice:
    name: str
    description: str
    pytest_args: tuple[str, ...]


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    timed_out: bool = False


def default_slices() -> tuple[PytestSlice, ...]:
    return (
        PytestSlice(
            name="runtime_surface",
            description="CLI, runtime logging, restart, and release-surface checks",
            pytest_args=(
                "tests/test_release_surface.py",
                "tests/test_cli_run.py",
                "tests/test_restartable_diffusion_tutorial.py",
            ),
        ),
        PytestSlice(
            name="precision_surface",
            description="runtime precision defaults and overrides in an isolated process",
            pytest_args=(
                "tests/test_runtime_precision.py",
            ),
        ),
        PytestSlice(
            name="portable_parity",
            description="portable payload, diff, compare, and reference-harness helpers",
            pytest_args=(
                "tests/test_parity_arrays.py",
                "tests/test_parity_compare.py",
                "tests/test_parity_diff.py",
                "tests/test_parity_portable.py",
                "tests/test_parity_reference.py",
            ),
        ),
        PytestSlice(
            name="mms_operator",
            description="manufactured-solution and operator-level native checks",
            pytest_args=(
                "tests/test_native_fluid_1d.py",
                "tests/test_native_open_field.py",
                "tests/test_solver_implicit.py",
            ),
        ),
        PytestSlice(
            name="recycling_operator",
            description="recycling operator, guard-state, and blocker-diagnostic checks",
            pytest_args=(
                "-m",
                "not slow",
                "tests/test_native_recycling_1d.py",
                "tests/test_recycling_diagnostics.py",
            ),
        ),
    )


def _slice_map() -> dict[str, PytestSlice]:
    return {slice_.name: slice_ for slice_ in default_slices()}


def resolve_slices(requested_names: tuple[str, ...] | None) -> tuple[PytestSlice, ...]:
    if not requested_names:
        return default_slices()
    mapping = _slice_map()
    resolved: list[PytestSlice] = []
    for name in requested_names:
        if name not in mapping:
            known = ", ".join(sorted(mapping))
            raise ValueError(f"unknown slice {name!r}; expected one of: {known}")
        resolved.append(mapping[name])
    return tuple(resolved)


def build_pytest_command(
    slice_: PytestSlice,
    *,
    python_executable: str,
    with_coverage: bool,
    coverage_append: bool,
    extra_pytest_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    command = [python_executable, "-m", "pytest", "-q", "--maxfail=1", *slice_.pytest_args, *extra_pytest_args]
    if with_coverage:
        command.extend(
            (
                f"--cov={DEFAULT_COVERAGE_TARGET}",
                "--cov-report=term-missing:skip-covered",
            )
        )
        if coverage_append:
            command.append("--cov-append")
    return tuple(command)


def run_checked_command(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> CommandResult:
    start = time.monotonic()
    pythonpath_entries = [str(cwd / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["JAX_DRB_PRECISION"] = "float64"
    env["JAX_ENABLE_X64"] = "true"
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=command,
            returncode=124,
            elapsed_seconds=time.monotonic() - start,
            timed_out=True,
        )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        elapsed_seconds=time.monotonic() - start,
        timed_out=False,
    )


def _print_slice_header(slice_: PytestSlice, *, timeout_seconds: int) -> None:
    print(f"[research-check] {slice_.name}: {slice_.description}")
    print(f"[research-check] timeout={timeout_seconds}s")


def _print_command(command: tuple[str, ...]) -> None:
    print(f"[research-check] command: {' '.join(shlex.quote(part) for part in command)}")


def _run_coverage_report(*, python_executable: str, cwd: Path) -> int:
    report_command = (
        python_executable,
        "-m",
        "coverage",
        "report",
        "--show-missing",
        "--skip-covered",
    )
    result = subprocess.run(report_command, cwd=cwd, check=False)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fast research-grade validation slices with a hard per-slice timeout. "
            "Any slice that exceeds the timeout fails immediately instead of hanging the iteration loop."
        )
    )
    parser.add_argument(
        "--slice",
        dest="slice_names",
        action="append",
        default=[],
        help="Named validation slice to run. Repeat to run multiple slices. Defaults to all fast slices.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Hard timeout per slice in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage collection/reporting and run the slices as plain pytest commands.",
    )
    parser.add_argument(
        "--extra-pytest-arg",
        dest="extra_pytest_args",
        action="append",
        default=[],
        help="Additional pytest argument to append to every slice command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    slices = resolve_slices(tuple(args.slice_names) if args.slice_names else None)
    python_executable = sys.executable
    with_coverage = not args.no_coverage

    if with_coverage:
        subprocess.run((python_executable, "-m", "coverage", "erase"), cwd=repo_root, check=False)

    coverage_append = False
    for slice_ in slices:
        command = build_pytest_command(
            slice_,
            python_executable=python_executable,
            with_coverage=with_coverage,
            coverage_append=coverage_append,
            extra_pytest_args=tuple(args.extra_pytest_args),
        )
        _print_slice_header(slice_, timeout_seconds=args.timeout_seconds)
        _print_command(command)
        if args.dry_run:
            coverage_append = coverage_append or with_coverage
            continue
        result = run_checked_command(command, cwd=repo_root, timeout_seconds=args.timeout_seconds)
        if result.timed_out:
            print(
                f"[research-check] slice {slice_.name} exceeded {args.timeout_seconds}s "
                f"and was terminated after {result.elapsed_seconds:.1f}s",
                file=sys.stderr,
            )
            return result.returncode
        if result.returncode != 0:
            print(
                f"[research-check] slice {slice_.name} failed with exit code {result.returncode} "
                f"after {result.elapsed_seconds:.1f}s",
                file=sys.stderr,
            )
            return result.returncode
        print(f"[research-check] slice {slice_.name} passed in {result.elapsed_seconds:.1f}s")
        coverage_append = coverage_append or with_coverage

    if with_coverage and not args.dry_run:
        report_code = _run_coverage_report(python_executable=python_executable, cwd=repo_root)
        if report_code != 0:
            return report_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
