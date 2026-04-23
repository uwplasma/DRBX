from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_TOTAL_COVERAGE = 95.0
CLOSEOUT_TESTS = (
    "tests/test_validation_controller_feedback_campaign.py",
    "tests/test_validation_temperature_feedback_campaign.py",
    "tests/test_validation_detachment_controller_campaign.py",
    "tests/test_validation_reactions_collisions_campaign.py",
    "tests/test_validation_impurity_radiation_campaign.py",
    "tests/test_validation_autodiff_diffusion_uncertainty.py",
    "tests/test_validation_open_field_operator_campaign.py",
    "tests/test_validation_tokamak_native_selected_field.py",
    "tests/test_validation_native_3d_runtime_campaign.py",
    "tests/test_validation_native_3d_convergence_campaign.py",
    "tests/test_validation_jax_native_profile_audit.py",
    "tests/test_validation_hermes_comparison_summary.py",
    "tests/test_validation_hermes_capability_audit.py",
    "tests/test_packaging_metadata.py",
    "tests/test_release_surface.py",
)
COVERAGE_TARGETS = (
    "src/jax_drb/validation/controller_feedback_campaign.py",
    "src/jax_drb/validation/temperature_feedback_campaign.py",
    "src/jax_drb/validation/detachment_controller_campaign.py",
    "src/jax_drb/validation/reactions_collisions_campaign.py",
    "src/jax_drb/validation/impurity_radiation_campaign.py",
    "src/jax_drb/validation/autodiff_diffusion_uncertainty.py",
    "src/jax_drb/validation/open_field_operator_campaign.py",
    "src/jax_drb/validation/tokamak_native_selected_field.py",
    "src/jax_drb/validation/native_3d_runtime_campaign.py",
    "src/jax_drb/validation/native_3d_convergence_campaign.py",
    "src/jax_drb/validation/jax_native_profile_audit.py",
    "src/jax_drb/validation/hermes_comparison_summary.py",
    "src/jax_drb/validation/hermes_capability_audit.py",
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
        *CLOSEOUT_TESTS,
    ]


def _build_report_command(*, python_executable: str) -> list[str]:
    return [
        python_executable,
        "-m",
        "coverage",
        "report",
        "-m",
        *COVERAGE_TARGETS,
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
    parser = argparse.ArgumentParser(description="Run the bounded release-closeout coverage slice.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=MIN_TOTAL_COVERAGE,
        help=f"Minimum acceptable coverage percentage for the closeout slice (default: {MIN_TOTAL_COVERAGE}).",
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
        print(
            f"closeout coverage gate failed: {total_coverage:.1f}% < required {float(args.threshold):.1f}%",
            file=sys.stderr,
        )
        return 1
    print(f"closeout coverage gate passed: {total_coverage:.1f}% >= required {float(args.threshold):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
