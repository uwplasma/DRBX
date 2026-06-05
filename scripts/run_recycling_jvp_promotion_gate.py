#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REFERENCE_ROOT = REPO_ROOT / "tests" / "fixtures" / "reference-root"
GATE_SOLVER_MODES = (
    "bdf",
    "bdf_fixed_full_field_jvp",
    "fixed_bdf2_jax_linearized",
)


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
    output_json: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(REPO_ROOT / "scripts" / "compare_recycling_transient_modes.py"),
        "--case",
        gate_case.case,
        "--reference-root",
        str(reference_root),
        "--diagnostics-only",
        "--require-fixed-jvp-diagnostics",
        "--require-fixed-bdf2-diagnostics",
        "--require-bdf-pairwise-max",
        f"{gate_case.pairwise_threshold:.8e}",
        "--mode-timeout-seconds",
        f"{gate_case.mode_timeout_seconds:g}",
    ]
    for mode in GATE_SOLVER_MODES:
        command.extend(("--mode", mode))
    for field in gate_case.fields:
        command.extend(("--field", field))
    if output_json is not None:
        command.extend(("--output-json", str(output_json)))
    return command


def _selected_cases(case_names: Sequence[str]) -> tuple[RecyclingJvpGateCase, ...]:
    if not case_names:
        return tuple(GATE_CASES.values())
    return tuple(GATE_CASES[name] for name in case_names)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_summary(
    output_dir: Path,
    *,
    reference_root: Path,
    case_reports: list[dict[str, object]],
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "reference_root": str(reference_root),
        "case_reports": _json_ready(case_reports),
        "all_cases_passed": all(
            int(report["returncode"]) == 0 for report in case_reports
        ),
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the self-contained recycling BDF JAX promotion gate. "
            "The gate compares the stable BDF path with the fixed-layout/JVP "
            "SciPy-BDF path and the fixed-layout non-SciPy BDF2 path, then requires "
            "both parity and diagnostic evidence before either JAX-native path can "
            "be considered for wider promotion."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional artifact directory. Each case writes a compare_recycling_transient_modes "
            "JSON report and the wrapper writes summary.json. With --dry-run, summary.json "
            "contains the planned commands without executing expensive solver cases."
        ),
    )
    args = parser.parse_args(argv)

    reference_root = args.reference_root.expanduser().resolve()
    if not reference_root.exists():
        parser.error(f"reference root does not exist: {reference_root}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    failures = 0
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else None
    )
    case_reports: list[dict[str, object]] = []
    for gate_case in _selected_cases(args.cases or ()):
        case_output_json = (
            output_dir / f"{gate_case.case}.json" if output_dir is not None else None
        )
        command = _build_case_command(
            gate_case,
            reference_root=reference_root,
            python_executable=sys.executable,
            output_json=case_output_json,
        )
        print(f"gate_case={gate_case.case}")
        print("command=" + " ".join(command))
        report: dict[str, object] = {
            "case": gate_case.case,
            "fields": list(gate_case.fields),
            "pairwise_threshold": gate_case.pairwise_threshold,
            "mode_timeout_seconds": gate_case.mode_timeout_seconds,
            "command": command,
            "output_json": case_output_json,
            "returncode": 0,
        }
        if args.dry_run:
            case_reports.append(report)
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
        report["returncode"] = int(completed.returncode)
        if case_output_json is not None and case_output_json.exists():
            report["case_report"] = json.loads(
                case_output_json.read_text(encoding="utf-8")
            )
        case_reports.append(report)
        if completed.returncode != 0:
            failures += 1
            print(f"gate_failure={gate_case.case} returncode={completed.returncode}")
    if output_dir is not None:
        summary_path = _write_summary(
            output_dir,
            reference_root=reference_root,
            case_reports=case_reports,
            dry_run=bool(args.dry_run),
        )
        print(f"summary_json={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
