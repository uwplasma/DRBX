#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_TOTAL_COVERAGE = 95.0
PROMOTED_SOLVER_TESTS = (
    "-m",
    "not slow",
    "tests/test_native_metrics.py",
    "tests/test_native_open_field.py",
    "tests/test_validation_open_field_operator_campaign.py",
    "tests/test_solver_implicit.py",
    "tests/test_native_recycling_1d.py",
    "tests/test_recycling_diagnostics.py",
    "tests/test_native_integrated_2d_recycling.py",
    "tests/test_native_runner_recycling.py",
    "tests/test_native_runner_solver_mode.py",
    "tests/test_native_runner.py",
    "tests/test_parity_arrays.py",
    "tests/test_parity_compare.py",
    "tests/test_parity_diff.py",
    "tests/test_parity_portable.py",
    "tests/test_parity_reference.py",
    "tests/test_cli_run.py",
)
PROMOTED_SOLVER_TARGETS = (
    "src/jax_drb/native/mesh.py",
    "src/jax_drb/native/metrics.py",
    "src/jax_drb/native/open_field.py",
    "src/jax_drb/native/recycling_1d.py",
    "src/jax_drb/native/recycling_rhs_terms.py",
    "src/jax_drb/native/recycling_targets.py",
    "src/jax_drb/native/runner.py",
    "src/jax_drb/native/runner_recycling.py",
    "src/jax_drb/native/runner_solver_mode.py",
    "src/jax_drb/parity/arrays.py",
    "src/jax_drb/parity/compare.py",
    "src/jax_drb/parity/diff.py",
    "src/jax_drb/parity/portable.py",
    "src/jax_drb/parity/reference.py",
    "src/jax_drb/cli.py",
)


def _build_pytest_command(*, python_executable: str) -> list[str]:
    return [
        python_executable,
        "-m",
        "coverage",
        "run",
        "-m",
        "pytest",
        "-q",
        "--maxfail=1",
        *PROMOTED_SOLVER_TESTS,
    ]


def _build_report_command(*, python_executable: str) -> list[str]:
    return [
        python_executable,
        "-m",
        "coverage",
        "report",
        "-m",
        *PROMOTED_SOLVER_TARGETS,
    ]


def _parse_total_coverage(report_text: str) -> float:
    for line in report_text.splitlines():
        if not line.startswith("TOTAL"):
            continue
        match = re.search(r"(\d+)%\s*$", line.strip())
        if match is None:
            break
        return float(match.group(1))
    raise ValueError("Could not parse TOTAL coverage from coverage report output.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the promoted solver/public-surface coverage audit.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=MIN_TOTAL_COVERAGE,
        help=f"Minimum acceptable coverage percentage for the promoted slice (default: {MIN_TOTAL_COVERAGE}).",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Report coverage without failing if the promoted-solver threshold is not yet met.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    args = parser.parse_args()

    python_executable = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    erase_command = [python_executable, "-m", "coverage", "erase"]
    pytest_command = _build_pytest_command(python_executable=python_executable)
    report_command = _build_report_command(python_executable=python_executable)

    if args.dry_run:
        print(" ".join(erase_command))
        print(" ".join(pytest_command))
        print(" ".join(report_command))
        return 0

    subprocess.run(erase_command, cwd=REPO_ROOT, env=env, check=True)
    subprocess.run(pytest_command, cwd=REPO_ROOT, env=env, check=True)
    report = subprocess.run(
        report_command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    print(report.stdout, end="")
    total_coverage = _parse_total_coverage(report.stdout)
    if total_coverage < float(args.threshold):
        message = f"promoted solver coverage gate failed: {total_coverage:.1f}% < required {float(args.threshold):.1f}%"
        if args.audit:
            print(f"{message} (audit mode)")
            return 0
        print(message, file=sys.stderr)
        return 1
    print(f"promoted solver coverage gate passed: {total_coverage:.1f}% >= required {float(args.threshold):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
