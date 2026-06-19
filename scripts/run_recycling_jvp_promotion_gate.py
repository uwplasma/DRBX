#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REFERENCE_ROOT = REPO_ROOT / "tests" / "fixtures" / "reference-root"
BDF_JVP_GATE_SOLVER_MODES = (
    "bdf",
    "bdf_fixed_full_field_jvp",
)
FIXED_BDF2_GATE_SOLVER_MODES = (
    "fixed_bdf2_jax_linearized",
    "fixed_bdf2_active_array_jax_linearized",
)
EXPERIMENTAL_GATE_SOLVER_MODES = ("bdf_active_array_jvp",)


@dataclass(frozen=True)
class RecyclingJvpGateCase:
    case: str
    fields: tuple[str, ...]
    pairwise_threshold: float
    mode_timeout_seconds: float
    steps: int
    fixed_bdf2_timestep: float | None = None
    fixed_bdf2_max_internal_timestep: float | None = None


GATE_CASES = {
    "recycling_1d_one_step": RecyclingJvpGateCase(
        case="recycling_1d_one_step",
        fields=("Pe", "Nd+", "Pd+"),
        pairwise_threshold=1.0e-5,
        mode_timeout_seconds=300.0,
        steps=2,
        fixed_bdf2_timestep=10.0,
    ),
    "recycling_dthe_one_step": RecyclingJvpGateCase(
        case="recycling_dthe_one_step",
        fields=("Pe", "Nd+", "Nt+", "Phe+"),
        pairwise_threshold=2.0e-5,
        mode_timeout_seconds=600.0,
        steps=2,
        fixed_bdf2_timestep=1.0,
        fixed_bdf2_max_internal_timestep=0.5,
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
    include_active_array_jvp: bool = False,
    gate_phase: str = "bdf_jvp",
    fixed_bdf2_timestep: float | None = None,
    mode_timeout_seconds: float | None = None,
    fixed_bdf2_linear_preconditioner: str | None = None,
    fixed_bdf2_linear_preconditioner_refresh: int | None = None,
    fixed_bdf2_linear_restart: int | None = None,
    fixed_bdf2_linear_maxiter: int | None = None,
    fixed_bdf2_linear_tolerance_factor: float | None = None,
    fixed_bdf2_max_linear_iterations: int | None = None,
    fixed_bdf2_max_linear_operator_calls: int | None = None,
    fixed_bdf2_max_linear_update_residual: float | None = None,
    fixed_bdf2_max_linear_update_relative_residual: float | None = None,
    fixed_bdf2_max_preconditioner_builds: int | None = None,
    fixed_bdf2_max_preconditioner_applies: int | None = None,
    fixed_bdf2_jit_linear_operator: bool = False,
    fixed_bdf2_linear_operator_counting: str | None = None,
    fixed_bdf2_diagnose_linear_update_residual: bool = False,
) -> list[str]:
    resolved_mode_timeout = (
        gate_case.mode_timeout_seconds
        if mode_timeout_seconds is None
        else float(mode_timeout_seconds)
    )
    if resolved_mode_timeout <= 0.0:
        raise ValueError("mode_timeout_seconds must be positive.")
    command = [
        python_executable,
        str(REPO_ROOT / "scripts" / "compare_recycling_transient_modes.py"),
        "--case",
        gate_case.case,
        "--reference-root",
        str(reference_root),
        "--diagnostics-only",
        "--mode-timeout-seconds",
        f"{resolved_mode_timeout:g}",
        "--steps",
        str(gate_case.steps),
    ]
    if gate_phase == "bdf_jvp":
        command.extend(
            (
                "--require-fixed-jvp-diagnostics",
                "--require-bdf-pairwise-max",
                f"{gate_case.pairwise_threshold:.8e}",
            )
        )
        solver_modes = BDF_JVP_GATE_SOLVER_MODES + (
            EXPERIMENTAL_GATE_SOLVER_MODES if include_active_array_jvp else ()
        )
    elif gate_phase == "fixed_bdf2":
        if fixed_bdf2_timestep is None:
            raise ValueError("fixed_bdf2_timestep is required for fixed_bdf2 gates.")
        command.extend(
            (
                "--require-fixed-bdf2-diagnostics",
                "--timestep",
                f"{float(fixed_bdf2_timestep):g}",
            )
        )
        if fixed_bdf2_linear_preconditioner is not None:
            preconditioner_name = str(fixed_bdf2_linear_preconditioner).strip()
            if not preconditioner_name:
                raise ValueError("fixed_bdf2_linear_preconditioner must be nonempty.")
            command.extend(
                (
                    "--override",
                    "runtime:recycling_jax_linear_preconditioner="
                    f"{preconditioner_name}",
                    "--require-fixed-bdf2-linear-preconditioner",
                    preconditioner_name,
                )
            )
            if fixed_bdf2_linear_preconditioner_refresh is not None:
                refresh = int(fixed_bdf2_linear_preconditioner_refresh)
                if refresh <= 0:
                    raise ValueError(
                        "fixed_bdf2_linear_preconditioner_refresh must be positive."
                    )
                command.extend(
                    (
                        "--override",
                        "runtime:recycling_jax_linear_preconditioner_refresh="
                        f"{refresh}",
                    )
                )
        if fixed_bdf2_linear_restart is not None:
            restart = int(fixed_bdf2_linear_restart)
            if restart <= 0:
                raise ValueError("fixed_bdf2_linear_restart must be positive.")
            command.extend(
                (
                    "--override",
                    f"runtime:recycling_jax_linear_restart={restart}",
                )
            )
        if fixed_bdf2_linear_maxiter is not None:
            maxiter = int(fixed_bdf2_linear_maxiter)
            if maxiter <= 0:
                raise ValueError("fixed_bdf2_linear_maxiter must be positive.")
            command.extend(
                (
                    "--override",
                    f"runtime:recycling_jax_linear_maxiter={maxiter}",
                )
            )
        if fixed_bdf2_linear_tolerance_factor is not None:
            tolerance_factor = float(fixed_bdf2_linear_tolerance_factor)
            if tolerance_factor <= 0.0:
                raise ValueError(
                    "fixed_bdf2_linear_tolerance_factor must be positive."
                )
            command.extend(
                (
                    "--override",
                    "runtime:recycling_jax_linear_tolerance_factor="
                    f"{tolerance_factor:g}",
                )
            )
        if gate_case.fixed_bdf2_max_internal_timestep is not None:
            command.extend(
                (
                    "--override",
                    "runtime:recycling_fixed_bdf2_max_internal_timestep="
                    f"{float(gate_case.fixed_bdf2_max_internal_timestep):g}",
                )
            )
        if bool(fixed_bdf2_jit_linear_operator):
            command.extend(
                (
                    "--override",
                    "runtime:recycling_jax_linear_jit_linear_operator=true",
                    "--require-fixed-bdf2-linear-operator-jitted",
                )
            )
        if fixed_bdf2_linear_operator_counting is not None:
            counting_mode = str(fixed_bdf2_linear_operator_counting).strip().lower()
            if counting_mode not in {"instrumented", "direct"}:
                raise ValueError(
                    "fixed_bdf2_linear_operator_counting must be 'instrumented' "
                    "or 'direct'."
                )
            if (
                counting_mode == "direct"
                and fixed_bdf2_max_linear_operator_calls is not None
            ):
                raise ValueError(
                    "fixed_bdf2_linear_operator_counting='direct' disables "
                    "Python-visible operator call counts; do not combine it "
                    "with fixed_bdf2_max_linear_operator_calls."
                )
            command.extend(
                (
                    "--override",
                    "runtime:recycling_jax_linear_operator_counting="
                    f"{counting_mode}",
                )
            )
        if bool(fixed_bdf2_diagnose_linear_update_residual):
            command.extend(
                (
                    "--override",
                    "runtime:recycling_jax_linear_diagnose_update_residual=true",
                )
            )
        if fixed_bdf2_max_linear_iterations is not None:
            max_linear_iterations = int(fixed_bdf2_max_linear_iterations)
            if max_linear_iterations < 0:
                raise ValueError("fixed_bdf2_max_linear_iterations must be nonnegative.")
            command.extend(
                (
                    "--require-fixed-bdf2-max-linear-iterations",
                    str(max_linear_iterations),
                )
            )
        if fixed_bdf2_max_linear_operator_calls is not None:
            max_linear_operator_calls = int(fixed_bdf2_max_linear_operator_calls)
            if max_linear_operator_calls < 0:
                raise ValueError(
                    "fixed_bdf2_max_linear_operator_calls must be nonnegative."
                )
            command.extend(
                (
                    "--require-fixed-bdf2-max-linear-operator-calls",
                    str(max_linear_operator_calls),
                )
            )
        if fixed_bdf2_max_linear_update_residual is not None:
            max_update_residual = float(fixed_bdf2_max_linear_update_residual)
            if not math.isfinite(max_update_residual) or max_update_residual < 0.0:
                raise ValueError(
                    "fixed_bdf2_max_linear_update_residual must be finite and "
                    "nonnegative."
                )
            command.extend(
                (
                    "--require-fixed-bdf2-max-linear-update-residual",
                    f"{max_update_residual:.17g}",
                )
            )
        if fixed_bdf2_max_linear_update_relative_residual is not None:
            max_relative_update_residual = float(
                fixed_bdf2_max_linear_update_relative_residual
            )
            if (
                not math.isfinite(max_relative_update_residual)
                or max_relative_update_residual < 0.0
            ):
                raise ValueError(
                    "fixed_bdf2_max_linear_update_relative_residual must be finite "
                    "and nonnegative."
                )
            command.extend(
                (
                    "--require-fixed-bdf2-max-linear-update-relative-residual",
                    f"{max_relative_update_residual:.17g}",
                )
            )
        if fixed_bdf2_max_preconditioner_builds is not None:
            max_preconditioner_builds = int(fixed_bdf2_max_preconditioner_builds)
            if max_preconditioner_builds < 0:
                raise ValueError(
                    "fixed_bdf2_max_preconditioner_builds must be nonnegative."
                )
            command.extend(
                (
                    "--require-fixed-bdf2-max-preconditioner-builds",
                    str(max_preconditioner_builds),
                )
            )
        if fixed_bdf2_max_preconditioner_applies is not None:
            max_preconditioner_applies = int(fixed_bdf2_max_preconditioner_applies)
            if max_preconditioner_applies < 0:
                raise ValueError(
                    "fixed_bdf2_max_preconditioner_applies must be nonnegative."
                )
            command.extend(
                (
                    "--require-fixed-bdf2-max-preconditioner-applies",
                    str(max_preconditioner_applies),
                )
            )
        solver_modes = FIXED_BDF2_GATE_SOLVER_MODES
    else:
        raise ValueError(f"Unknown recycling JVP gate phase: {gate_phase}")
    for mode in solver_modes:
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
        "--include-active-array-jvp",
        action="store_true",
        help=(
            "Also run the experimental bdf_active_array_jvp bridge. This is not "
            "part of the default promotion gate because current local evidence "
            "shows it can timeout before completing the fixture case."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-only",
        action="store_true",
        help=(
            "Run only the bounded fixed-BDF2 JAX-linearized phase. Use this for "
            "preconditioner and matrix-free performance sweeps when the separate "
            "SciPy-BDF JVP bridge parity gate is not the quantity under test."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-timestep",
        type=float,
        default=None,
        help=(
            "Override the bounded fixed-BDF2 diagnostic timestep. By default only "
            "cases with a validated case-specific bounded timestep run this phase; "
            "cases may also add a validated fixed-BDF2 internal-substep override."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-linear-preconditioner",
        default=None,
        help=(
            "Opt into a JAX-GMRES preconditioner for the fixed-BDF2 promotion phase. "
            "The generated compare command also requires matching preconditioner "
            "diagnostics so the gate fails if the run silently falls back to an "
            "unpreconditioned path."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-linear-preconditioner-refresh",
        type=int,
        default=None,
        help=(
            "When --fixed-bdf2-linear-preconditioner is set, forward this positive "
            "refresh interval as runtime:recycling_jax_linear_preconditioner_refresh. "
            "Use values larger than one to test preconditioner reuse inside each "
            "implicit solve."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-jit-linear-operator",
        action="store_true",
        help=(
            "Forward runtime:recycling_jax_linear_jit_linear_operator=true to the "
            "fixed-BDF2 phase and require fixed_bdf2_linear_operator_jitted_steps "
            "in the compare report. This gates the JAX-compiled matrix-free "
            "Krylov action used by heavier recycling profiles."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-linear-operator-counting",
        choices=("instrumented", "direct"),
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_operator_counting=<mode> to "
            "the fixed-BDF2 phase. Use direct for lower-overhead profiling after "
            "operator-call budgets have already been established with "
            "instrumented diagnostics."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-diagnose-linear-update-residual",
        action="store_true",
        help=(
            "Forward runtime:recycling_jax_linear_diagnose_update_residual=true "
            "to the fixed-BDF2 phase. This adds one extra linearized action per "
            "Newton update and records the achieved J v + r residual for "
            "preconditioner screening."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-linear-restart",
        type=int,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_restart=<n> to the fixed-BDF2 "
            "phase. This is useful for constrained-budget preconditioner "
            "screening before running full heavy profiles."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-linear-maxiter",
        type=int,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_maxiter=<n> to the fixed-BDF2 "
            "phase. Keep this explicit in artifact summaries when testing "
            "whether a preconditioner can preserve accuracy under a smaller "
            "Krylov budget."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-linear-tolerance-factor",
        type=float,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_tolerance_factor=<f> to the "
            "fixed-BDF2 phase. Values larger than one loosen the inner Krylov "
            "solve relative to the nonlinear residual tolerance."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-max-linear-iterations",
        type=int,
        default=None,
        help=(
            "Forward --require-fixed-bdf2-max-linear-iterations to the fixed-BDF2 "
            "compare phase. This should be used only for performance-promotion "
            "campaigns, not correctness-only gates."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-max-linear-operator-calls",
        type=int,
        default=None,
        help=(
            "Forward --require-fixed-bdf2-max-linear-operator-calls to the "
            "fixed-BDF2 compare phase. This gates actual JAX linear-map/JVP "
            "operator work and should be used with preconditioner experiments."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-max-linear-update-residual",
        type=float,
        default=None,
        help=(
            "Forward --require-fixed-bdf2-max-linear-update-residual to the "
            "fixed-BDF2 compare phase. Use with "
            "--fixed-bdf2-diagnose-linear-update-residual."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-max-linear-update-relative-residual",
        type=float,
        default=None,
        help=(
            "Forward --require-fixed-bdf2-max-linear-update-relative-residual to "
            "the fixed-BDF2 compare phase. This gates achieved Krylov update "
            "quality under explicit iteration/operator budgets."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-max-preconditioner-builds",
        type=int,
        default=None,
        help=(
            "Forward --require-fixed-bdf2-max-preconditioner-builds to the fixed-BDF2 "
            "compare phase. Useful for proving dynamic-preconditioner reuse."
        ),
    )
    parser.add_argument(
        "--fixed-bdf2-max-preconditioner-applies",
        type=int,
        default=None,
        help=(
            "Forward --require-fixed-bdf2-max-preconditioner-applies to the fixed-BDF2 "
            "compare phase. Use with operator-call budgets to screen preconditioners "
            "that are applied frequently without reducing Krylov work."
        ),
    )
    parser.add_argument(
        "--mode-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Override the per-mode timeout forwarded to compare_recycling_transient_modes.py. "
            "Useful for bounded experimental active-array JVP probes."
        ),
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
    if args.fixed_bdf2_only and args.include_active_array_jvp:
        parser.error("--include-active-array-jvp cannot be used with --fixed-bdf2-only")

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
        phase_specs: list[tuple[str, float | None]] = []
        if not bool(args.fixed_bdf2_only):
            phase_specs.append(("bdf_jvp", None))
        bounded_timestep = (
            float(args.fixed_bdf2_timestep)
            if args.fixed_bdf2_timestep is not None
            else gate_case.fixed_bdf2_timestep
        )
        if bounded_timestep is not None:
            if float(bounded_timestep) <= 0.0:
                raise ValueError("--fixed-bdf2-timestep must be positive.")
            phase_specs.append(("fixed_bdf2", float(bounded_timestep)))
        elif bool(args.fixed_bdf2_only):
            parser.error(
                f"{gate_case.case} has no bounded fixed-BDF2 timestep; pass "
                "--fixed-bdf2-timestep explicitly."
            )
        for gate_phase, fixed_bdf2_timestep in phase_specs:
            case_output_json = (
                output_dir / f"{gate_case.case}.{gate_phase}.json"
                if output_dir is not None
                else None
            )
            command = _build_case_command(
                gate_case,
                reference_root=reference_root,
                python_executable=sys.executable,
                output_json=case_output_json,
                include_active_array_jvp=bool(args.include_active_array_jvp),
                gate_phase=gate_phase,
                fixed_bdf2_timestep=fixed_bdf2_timestep,
                mode_timeout_seconds=args.mode_timeout_seconds,
                fixed_bdf2_linear_preconditioner=(
                    args.fixed_bdf2_linear_preconditioner
                ),
                fixed_bdf2_linear_preconditioner_refresh=(
                    args.fixed_bdf2_linear_preconditioner_refresh
                ),
                fixed_bdf2_linear_restart=args.fixed_bdf2_linear_restart,
                fixed_bdf2_linear_maxiter=args.fixed_bdf2_linear_maxiter,
                fixed_bdf2_linear_tolerance_factor=(
                    args.fixed_bdf2_linear_tolerance_factor
                ),
                fixed_bdf2_max_linear_iterations=(
                    args.fixed_bdf2_max_linear_iterations
                ),
                fixed_bdf2_max_linear_operator_calls=(
                    args.fixed_bdf2_max_linear_operator_calls
                ),
                fixed_bdf2_max_linear_update_residual=(
                    args.fixed_bdf2_max_linear_update_residual
                ),
                fixed_bdf2_max_linear_update_relative_residual=(
                    args.fixed_bdf2_max_linear_update_relative_residual
                ),
                fixed_bdf2_max_preconditioner_builds=(
                    args.fixed_bdf2_max_preconditioner_builds
                ),
                fixed_bdf2_max_preconditioner_applies=(
                    args.fixed_bdf2_max_preconditioner_applies
                ),
                fixed_bdf2_jit_linear_operator=bool(
                    args.fixed_bdf2_jit_linear_operator
                ),
                fixed_bdf2_linear_operator_counting=(
                    args.fixed_bdf2_linear_operator_counting
                ),
                fixed_bdf2_diagnose_linear_update_residual=bool(
                    args.fixed_bdf2_diagnose_linear_update_residual
                ),
            )
            resolved_mode_timeout = (
                gate_case.mode_timeout_seconds
                if args.mode_timeout_seconds is None
                else float(args.mode_timeout_seconds)
            )
            print(f"gate_case={gate_case.case}")
            print(f"gate_phase={gate_phase}")
            print("command=" + " ".join(command))
            report: dict[str, object] = {
                "case": gate_case.case,
                "phase": gate_phase,
                "fields": list(gate_case.fields),
                "pairwise_threshold": gate_case.pairwise_threshold,
                "mode_timeout_seconds": resolved_mode_timeout,
                "steps": gate_case.steps,
                "include_active_array_jvp": bool(args.include_active_array_jvp),
                "fixed_bdf2_only": bool(args.fixed_bdf2_only),
                "fixed_bdf2_timestep": fixed_bdf2_timestep,
                "fixed_bdf2_linear_preconditioner": (
                    args.fixed_bdf2_linear_preconditioner
                ),
                "fixed_bdf2_linear_preconditioner_refresh": (
                    args.fixed_bdf2_linear_preconditioner_refresh
                ),
                "fixed_bdf2_linear_restart": args.fixed_bdf2_linear_restart,
                "fixed_bdf2_linear_maxiter": args.fixed_bdf2_linear_maxiter,
                "fixed_bdf2_linear_tolerance_factor": (
                    args.fixed_bdf2_linear_tolerance_factor
                ),
                "fixed_bdf2_jit_linear_operator": bool(
                    args.fixed_bdf2_jit_linear_operator
                ),
                "fixed_bdf2_linear_operator_counting": (
                    args.fixed_bdf2_linear_operator_counting
                ),
                "fixed_bdf2_diagnose_linear_update_residual": bool(
                    args.fixed_bdf2_diagnose_linear_update_residual
                ),
                "fixed_bdf2_max_linear_iterations": (
                    args.fixed_bdf2_max_linear_iterations
                ),
                "fixed_bdf2_max_linear_operator_calls": (
                    args.fixed_bdf2_max_linear_operator_calls
                ),
                "fixed_bdf2_max_linear_update_residual": (
                    args.fixed_bdf2_max_linear_update_residual
                ),
                "fixed_bdf2_max_linear_update_relative_residual": (
                    args.fixed_bdf2_max_linear_update_relative_residual
                ),
                "fixed_bdf2_max_preconditioner_builds": (
                    args.fixed_bdf2_max_preconditioner_builds
                ),
                "fixed_bdf2_max_preconditioner_applies": (
                    args.fixed_bdf2_max_preconditioner_applies
                ),
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
                print(
                    f"gate_failure={gate_case.case} phase={gate_phase} "
                    f"returncode={completed.returncode}"
                )
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
