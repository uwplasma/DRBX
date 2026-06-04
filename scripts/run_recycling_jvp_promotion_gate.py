#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REFERENCE_ROOT = REPO_ROOT / "tests" / "fixtures" / "reference-root"


@dataclass(frozen=True)
class RecyclingJvpGateCase:
    case: str
    fields: tuple[str, ...]
    pairwise_threshold: float
    mode_timeout_seconds: float


GATE_CASES = {
    "recycling_1d_one_step": RecyclingJvpGateCase(
        case="recycling_1d_one_step",
        fields=("Pe", "Nd+", "Pd+"),
        pairwise_threshold=1.0e-5,
        mode_timeout_seconds=150.0,
    ),
    "recycling_dthe_one_step": RecyclingJvpGateCase(
        case="recycling_dthe_one_step",
        fields=("Pe", "Nd+", "Nt+", "Phe+"),
        pairwise_threshold=2.0e-5,
        mode_timeout_seconds=300.0,
    ),
}


def _default_reference_root() -> Path:
    env_value = os.environ.get("JAX_DRB_REFERENCE_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return FIXTURE_REFERENCE_ROOT


def _build_case_command(
    gate_case: RecyclingJvpGateCase,
    *,
    reference_root: Path,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        str(REPO_ROOT / "scripts" / "compare_recycling_transient_modes.py"),
        "--case",
        gate_case.case,
        "--reference-root",
        str(reference_root),
        "--mode",
        "bdf",
        "--mode",
        "bdf_fixed_full_field_jvp",
        "--diagnostics-only",
        "--require-fixed-jvp-diagnostics",
        "--require-bdf-pairwise-max",
        f"{gate_case.pairwise_threshold:.8e}",
        "--mode-timeout-seconds",
        f"{gate_case.mode_timeout_seconds:g}",
    ]
    for field in gate_case.fields:
        command.extend(("--field", field))
    return command


def _selected_cases(case_names: Sequence[str]) -> tuple[RecyclingJvpGateCase, ...]:
    if not case_names:
        return tuple(GATE_CASES.values())
    return tuple(GATE_CASES[name] for name in case_names)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the self-contained recycling BDF fixed-full-field JVP promotion gate. "
            "The gate compares the stable BDF path with the fixed-layout/JVP path and "
            "requires both parity and diagnostic evidence before the JVP path can be "
            "considered for wider promotion."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=_default_reference_root(),
        help=(
            "Reference input root. Defaults to JAX_DRB_REFERENCE_ROOT when set, "
            "otherwise the committed lightweight fixture decks."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(GATE_CASES),
        dest="cases",
        help="Gate case to run. May be repeated. Defaults to all promotion cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    args = parser.parse_args(argv)

    reference_root = args.reference_root.expanduser().resolve()
    if not reference_root.exists():
        parser.error(f"reference root does not exist: {reference_root}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    failures = 0
    for gate_case in _selected_cases(args.cases or ()):
        command = _build_case_command(
            gate_case,
            reference_root=reference_root,
            python_executable=sys.executable,
        )
        print(f"gate_case={gate_case.case}")
        print("command=" + " ".join(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
        if completed.returncode != 0:
            failures += 1
            print(f"gate_failure={gate_case.case} returncode={completed.returncode}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
