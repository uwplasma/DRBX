#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import os
from pathlib import Path
import pstats
import shutil
import statistics
import sys
from time import perf_counter
from typing import Any, Sequence


def _sanitize_public_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        pass
    for base_name in ("HOME",):
        base_value = os.environ.get(base_name)
        if not base_value:
            continue
        base_path = Path(base_value).expanduser().resolve()
        try:
            return f"~/{resolved.relative_to(base_path).as_posix()}"
        except ValueError:
            pass
    return resolved.as_posix()


def _public_input_path(args: argparse.Namespace, input_path: Path) -> str:
    root = args.reference_root
    if root is None:
        env_root = os.environ.get("JAX_DRB_REFERENCE_ROOT")
        root = Path(env_root) if env_root else None
    if root is not None:
        try:
            return f"<reference-root>/{input_path.resolve().relative_to(root.expanduser().resolve()).as_posix()}"
        except ValueError:
            pass
    return f"<input-path>/{input_path.name}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the real 1D-recycling fixed-layout backward-Euler gate "
            "through the JAX-linearized Newton path."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Hermès reference root. Falls back to JAX_DRB_REFERENCE_ROOT.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Explicit 1D-recycling BOUT.inp. Defaults under --reference-root.",
    )
    parser.add_argument(
        "--case",
        choices=("hydrogen", "dthe"),
        default="hydrogen",
        help="Reference integrated recycling deck to profile when --input-path is not supplied.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs")
        / "data"
        / "runtime_profile_artifacts"
        / "recycling_1d_jax_linearized_gate",
    )
    parser.add_argument("--timestep", type=float, default=1.0e-6)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-nonlinear-iterations", type=int, default=1)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "BOUT.inp override such as 'mesh:ny=100'. May be repeated. "
            "Use this for heavier real-kernel CPU/GPU scaling gates without "
            "copying large input decks into the repository."
        ),
    )
    parser.add_argument(
        "--jit-residual",
        action="store_true",
        help=(
            "Set runtime:recycling_jax_linear_jit_residual=true before profiling. "
            "This is a diagnostic JAX compilation seam, not a production default."
        ),
    )
    parser.add_argument(
        "--jit-linear-operator",
        action="store_true",
        help=(
            "Set runtime:recycling_jax_linear_jit_linear_operator=true before "
            "profiling. This wraps the JVP-derived Krylov action in jax.jit and "
            "is useful for separating compilation/dispatch cost from operator "
            "and preconditioner quality."
        ),
    )
    parser.add_argument(
        "--linear-operator-counting",
        choices=("instrumented", "direct"),
        default=None,
        help=(
            "Set runtime:recycling_jax_linear_operator_counting. The default "
            "instrumented mode records Python-visible operator calls; direct "
            "mode passes the JAX linear operator straight to the Krylov solver "
            "for lower-overhead production-style profiling."
        ),
    )
    parser.add_argument(
        "--active-array-rhs",
        action="store_true",
        help=(
            "Profile the active-array JAX-linearized recycling residual instead "
            "of the fixed-full-field residual. This keeps the same physical "
            "fields and fixed-layout packing but avoids full-field residual "
            "work when collecting CPU/GPU evidence."
        ),
    )
    parser.add_argument(
        "--skip-initial-residual-check",
        action="store_true",
        help=(
            "Set runtime:recycling_jax_linear_check_initial_residual=false before "
            "profiling. This avoids a separate pre-linearization residual call "
            "when the selected deck is known not to start converged."
        ),
    )
    parser.add_argument(
        "--initial-residual-mode",
        choices=("residual", "linearize"),
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_initial_residual_mode=<mode>. "
            "Use 'linearize' to keep the initial convergence check but obtain it "
            "from the first JAX linearization, avoiding the standalone residual "
            "call on known non-converged heavy solves."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run this many unprofiled solves before timing to amortize JAX compilation.",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Run this many timed solves after warmup. The first timed solve is optionally cProfile/JAX-traced.",
    )
    parser.add_argument(
        "--linear-solver-backend",
        choices=("jax_gmres", "lineax_gmres"),
        default="jax_gmres",
        help="Linear solver backend for nontrivial JAX-linearized Newton updates.",
    )
    parser.add_argument(
        "--gmres-solve-method",
        choices=("batched", "incremental"),
        default=None,
        help=(
            "Optional JAX GMRES solve_method override. 'incremental' can "
            "terminate within a restart on CPU; 'batched' is the JAX default "
            "and usually has lower accelerator overhead."
        ),
    )
    parser.add_argument(
        "--linear-restart",
        type=int,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_restart=<n>. This is a "
            "first-class equivalent of --override for reproducible Krylov-budget "
            "sweeps."
        ),
    )
    parser.add_argument(
        "--linear-maxiter",
        type=int,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_maxiter=<n>. The reported "
            "linear-iteration budget is restart * maxiter."
        ),
    )
    parser.add_argument(
        "--linear-tolerance-factor",
        type=float,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_tolerance_factor=<factor>. "
            "The inner Krylov tolerance remains residual_tolerance * factor."
        ),
    )
    parser.add_argument(
        "--line-search-initial-step-scale",
        type=float,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_line_search_initial_step_scale=<s>. "
            "The value must be finite and in (0, 1]. Use this to test whether a "
            "known damped Newton step avoids rejected line-search residual calls."
        ),
    )
    parser.add_argument(
        "--linear-preconditioner",
        default=None,
        help=(
            "Opt into runtime:recycling_jax_linear_preconditioner=<name> for "
            "the profiled JAX-GMRES solve."
        ),
    )
    parser.add_argument(
        "--linear-preconditioner-refresh",
        type=int,
        default=None,
        help=(
            "Forward runtime:recycling_jax_linear_preconditioner_refresh=<n>. "
            "Use positive values larger than one to profile dynamic-preconditioner "
            "reuse inside one implicit solve."
        ),
    )
    parser.add_argument(
        "--require-linear-preconditioner",
        default=None,
        help=(
            "Fail after profiling unless solver diagnostics report this "
            "linear_preconditioner. Dynamic JVP-derived preconditioners must "
            "also report at least one build."
        ),
    )
    parser.add_argument(
        "--require-initial-residual-mode",
        choices=("residual", "linearize"),
        default=None,
        help=(
            "Fail after profiling unless diagnostics.initial_residual_mode "
            "matches this mode. Use this to prove a profile exercised the "
            "safety-preserving linearized initial-residual check instead of "
            "silently falling back to the standalone residual path."
        ),
    )
    parser.add_argument(
        "--require-linear-operator-jitted",
        action="store_true",
        help=(
            "Fail after profiling unless diagnostics.linear_operator_jitted is true. "
            "Use this with --jit-linear-operator when collecting JAX compilation "
            "and same-kernel performance evidence."
        ),
    )
    parser.add_argument(
        "--require-rhs-backend",
        choices=("fixed_full_field_array", "active_array"),
        default=None,
        help=(
            "Fail after profiling unless diagnostics.rhs_backend matches this "
            "fixed-layout residual backend. Use 'active_array' with "
            "--active-array-rhs when collecting promoted output-window or "
            "CPU/GPU profiles."
        ),
    )
    parser.add_argument(
        "--require-max-linear-iterations",
        type=int,
        default=None,
        help=(
            "Fail after profiling when reported linear_iterations exceeds this "
            "nonnegative budget."
        ),
    )
    parser.add_argument(
        "--require-max-residual-inf-norm",
        type=float,
        default=None,
        help=(
            "Fail after profiling when profile.residual_inf_norm exceeds this "
            "finite nonnegative ceiling. Use this with Krylov-budget sweeps so a "
            "shorter run cannot pass by degrading the nonlinear residual."
        ),
    )
    parser.add_argument(
        "--require-max-linear-update-residual",
        type=float,
        default=None,
        help=(
            "Fail after profiling when diagnostics.linear_update_residual_inf_norm "
            "exceeds this finite nonnegative ceiling. Requires "
            "runtime:recycling_jax_linear_diagnose_update_residual=true."
        ),
    )
    parser.add_argument(
        "--require-max-linear-update-relative-residual",
        type=float,
        default=None,
        help=(
            "Fail after profiling when diagnostics.linear_update_relative_residual "
            "exceeds this finite nonnegative ceiling. Use this with explicit "
            "Krylov budgets to screen preconditioner quality."
        ),
    )
    parser.add_argument(
        "--require-max-residual-evaluations",
        type=int,
        default=None,
        help=(
            "Fail after profiling when diagnostics.residual_evaluation_count "
            "exceeds this nonnegative budget."
        ),
    )
    parser.add_argument(
        "--require-max-line-search-trials",
        type=int,
        default=None,
        help=(
            "Fail after profiling when diagnostics.line_search_trial_count "
            "exceeds this nonnegative budget."
        ),
    )
    parser.add_argument(
        "--require-min-linear-operator-calls",
        type=int,
        default=None,
        help=(
            "Fail after profiling when diagnostics.linear_operator_call_count "
            "is below this nonnegative floor. This proves the matrix-free "
            "linearized operator was exercised."
        ),
    )
    parser.add_argument(
        "--require-max-linear-operator-calls",
        type=int,
        default=None,
        help=(
            "Fail after profiling when diagnostics.linear_operator_call_count "
            "exceeds this nonnegative budget."
        ),
    )
    parser.add_argument(
        "--require-min-linear-iterations",
        type=int,
        default=None,
        help=(
            "Fail after profiling when reported linear_iterations is below this "
            "nonnegative floor. Use this to prove the profile exercised the "
            "JAX-linearized Krylov path instead of exiting at the predictor."
        ),
    )
    parser.add_argument(
        "--require-min-linear-solve-count",
        type=int,
        default=None,
        help=(
            "Fail after profiling when diagnostics.linear_solve_count is below "
            "this nonnegative floor. Use this with "
            "--linear-operator-counting=direct, where Python-visible operator "
            "calls and iteration counts are intentionally not instrumented."
        ),
    )
    parser.add_argument(
        "--require-min-nonlinear-iterations",
        type=int,
        default=None,
        help=(
            "Fail after profiling when reported nonlinear_iterations is below this "
            "nonnegative floor. Use this to reject no-op residual-check profiles."
        ),
    )
    parser.add_argument(
        "--require-max-preconditioner-builds",
        type=int,
        default=None,
        help=(
            "Fail after profiling when diagnostics.linear_preconditioner_build_count "
            "exceeds this nonnegative budget."
        ),
    )
    parser.add_argument(
        "--require-max-preconditioner-applies",
        type=int,
        default=None,
        help=(
            "Fail after profiling when diagnostics.linear_preconditioner_apply_count "
            "exceeds this nonnegative budget. Use this with operator-call budgets "
            "to reject preconditioners that are cheap to build but overused."
        ),
    )
    parser.add_argument("--cprofile-top", type=int, default=40)
    parser.add_argument("--skip-cprofile", action="store_true")
    parser.add_argument("--rss-profile", action="store_true")
    parser.add_argument("--jax-trace", action="store_true")
    parser.add_argument("--device-memory-profile", action="store_true")
    parser.add_argument("--compilation-cache-dir", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.linear_preconditioner is not None and not str(
        args.linear_preconditioner
    ).strip():
        raise SystemExit("--linear-preconditioner must be nonempty.")
    if args.linear_preconditioner_refresh is not None:
        if int(args.linear_preconditioner_refresh) <= 0:
            raise SystemExit("--linear-preconditioner-refresh must be positive.")
    if args.require_linear_preconditioner is not None and not str(
        args.require_linear_preconditioner
    ).strip():
        raise SystemExit("--require-linear-preconditioner must be nonempty.")
    required_initial_residual_mode = getattr(
        args, "require_initial_residual_mode", None
    )
    if required_initial_residual_mode is not None:
        mode = str(required_initial_residual_mode).strip().lower().replace("-", "_")
        if mode not in {"residual", "linearize"}:
            raise SystemExit(
                "--require-initial-residual-mode must be 'residual' or 'linearize'."
            )
    initial_residual_mode = getattr(args, "initial_residual_mode", None)
    if initial_residual_mode is not None:
        mode = str(initial_residual_mode).strip().lower().replace("-", "_")
        if mode not in {"residual", "linearize"}:
            raise SystemExit(
                "--initial-residual-mode must be 'residual' or 'linearize'."
            )
    if args.require_max_linear_iterations is not None:
        if int(args.require_max_linear_iterations) < 0:
            raise SystemExit("--require-max-linear-iterations must be nonnegative.")
    if args.linear_restart is not None:
        if int(args.linear_restart) <= 0:
            raise SystemExit("--linear-restart must be positive.")
    if args.linear_maxiter is not None:
        if int(args.linear_maxiter) <= 0:
            raise SystemExit("--linear-maxiter must be positive.")
    if args.linear_tolerance_factor is not None:
        factor = float(args.linear_tolerance_factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise SystemExit("--linear-tolerance-factor must be finite and positive.")
    if args.line_search_initial_step_scale is not None:
        scale = float(args.line_search_initial_step_scale)
        if not math.isfinite(scale) or scale <= 0.0 or scale > 1.0:
            raise SystemExit(
                "--line-search-initial-step-scale must be finite and in (0, 1]."
            )
    if args.require_max_residual_inf_norm is not None:
        ceiling = float(args.require_max_residual_inf_norm)
        if not math.isfinite(ceiling) or ceiling < 0.0:
            raise SystemExit(
                "--require-max-residual-inf-norm must be finite and nonnegative."
            )
    if args.require_max_linear_update_residual is not None:
        ceiling = float(args.require_max_linear_update_residual)
        if not math.isfinite(ceiling) or ceiling < 0.0:
            raise SystemExit(
                "--require-max-linear-update-residual must be finite and nonnegative."
            )
    if args.require_max_linear_update_relative_residual is not None:
        ceiling = float(args.require_max_linear_update_relative_residual)
        if not math.isfinite(ceiling) or ceiling < 0.0:
            raise SystemExit(
                "--require-max-linear-update-relative-residual must be finite "
                "and nonnegative."
            )
    if args.require_max_residual_evaluations is not None:
        if int(args.require_max_residual_evaluations) < 0:
            raise SystemExit(
                "--require-max-residual-evaluations must be nonnegative."
            )
    if args.require_max_line_search_trials is not None:
        if int(args.require_max_line_search_trials) < 0:
            raise SystemExit("--require-max-line-search-trials must be nonnegative.")
    if args.require_min_linear_operator_calls is not None:
        if int(args.require_min_linear_operator_calls) < 0:
            raise SystemExit(
                "--require-min-linear-operator-calls must be nonnegative."
            )
    if args.require_max_linear_operator_calls is not None:
        if int(args.require_max_linear_operator_calls) < 0:
            raise SystemExit(
                "--require-max-linear-operator-calls must be nonnegative."
            )
    if args.linear_operator_counting == "direct" and (
        args.require_min_linear_operator_calls is not None
        or args.require_max_linear_operator_calls is not None
    ):
        raise SystemExit(
            "--linear-operator-counting=direct disables Python-visible operator "
            "call counts; do not combine it with linear-operator call gates."
        )
    if args.require_min_linear_iterations is not None:
        if int(args.require_min_linear_iterations) < 0:
            raise SystemExit("--require-min-linear-iterations must be nonnegative.")
    if args.require_min_linear_solve_count is not None:
        if int(args.require_min_linear_solve_count) < 0:
            raise SystemExit(
                "--require-min-linear-solve-count must be nonnegative."
            )
    if args.require_min_nonlinear_iterations is not None:
        if int(args.require_min_nonlinear_iterations) < 0:
            raise SystemExit(
                "--require-min-nonlinear-iterations must be nonnegative."
            )
    if args.require_max_preconditioner_builds is not None:
        if int(args.require_max_preconditioner_builds) < 0:
            raise SystemExit(
                "--require-max-preconditioner-builds must be nonnegative."
            )
    if args.require_max_preconditioner_applies is not None:
        if int(args.require_max_preconditioner_applies) < 0:
            raise SystemExit(
                "--require-max-preconditioner-applies must be nonnegative."
            )


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


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input_path is not None:
        return args.input_path.expanduser().resolve()
    root = args.reference_root
    if root is None:
        env_root = os.environ.get("JAX_DRB_REFERENCE_ROOT")
        root = Path(env_root) if env_root else None
    if root is None:
        raise SystemExit("--reference-root or JAX_DRB_REFERENCE_ROOT is required.")
    case_dir = "1D-recycling-dthe" if args.case == "dthe" else "1D-recycling"
    return (
        root.expanduser().resolve()
        / "tests"
        / "integrated"
        / case_dir
        / "data"
        / "BOUT.inp"
    ).resolve()


def _solver_mode_for_backend(
    linear_solver_backend: str, *, active_array_rhs: bool = False
) -> str:
    if str(linear_solver_backend) == "lineax_gmres":
        return (
            "active_array_jax_linearized_lineax"
            if bool(active_array_rhs)
            else "jax_linearized_lineax"
        )
    return (
        "active_array_jax_linearized"
        if bool(active_array_rhs)
        else "jax_linearized"
    )


def _effective_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(getattr(args, "override", ()) or ())
    if bool(getattr(args, "jit_residual", False)):
        overrides.append("runtime:recycling_jax_linear_jit_residual=true")
    if bool(getattr(args, "jit_linear_operator", False)):
        overrides.append("runtime:recycling_jax_linear_jit_linear_operator=true")
    operator_counting = getattr(args, "linear_operator_counting", None)
    if operator_counting is not None:
        overrides.append(
            f"runtime:recycling_jax_linear_operator_counting={operator_counting}"
        )
    if bool(getattr(args, "skip_initial_residual_check", False)):
        overrides.append("runtime:recycling_jax_linear_check_initial_residual=false")
    initial_residual_mode = getattr(args, "initial_residual_mode", None)
    if initial_residual_mode is not None:
        mode = str(initial_residual_mode).strip().lower().replace("-", "_")
        if mode not in {"residual", "linearize"}:
            raise ValueError("initial_residual_mode must be 'residual' or 'linearize'.")
        overrides.append(f"runtime:recycling_jax_linear_initial_residual_mode={mode}")
    gmres_solve_method = getattr(args, "gmres_solve_method", None)
    if gmres_solve_method:
        overrides.append(
            f"runtime:recycling_jax_linear_gmres_solve_method={gmres_solve_method}"
        )
    linear_restart = getattr(args, "linear_restart", None)
    if linear_restart is not None:
        restart = int(linear_restart)
        if restart <= 0:
            raise ValueError("linear_restart must be positive.")
        overrides.append(f"runtime:recycling_jax_linear_restart={restart}")
    linear_maxiter = getattr(args, "linear_maxiter", None)
    if linear_maxiter is not None:
        maxiter = int(linear_maxiter)
        if maxiter <= 0:
            raise ValueError("linear_maxiter must be positive.")
        overrides.append(f"runtime:recycling_jax_linear_maxiter={maxiter}")
    tolerance_factor = getattr(args, "linear_tolerance_factor", None)
    if tolerance_factor is not None:
        factor = float(tolerance_factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("linear_tolerance_factor must be finite and positive.")
        overrides.append(f"runtime:recycling_jax_linear_tolerance_factor={factor:.17g}")
    line_search_initial_step_scale = getattr(
        args, "line_search_initial_step_scale", None
    )
    if line_search_initial_step_scale is not None:
        scale = float(line_search_initial_step_scale)
        if not math.isfinite(scale) or scale <= 0.0 or scale > 1.0:
            raise ValueError(
                "line_search_initial_step_scale must be finite and in (0, 1]."
            )
        overrides.append(
            "runtime:recycling_jax_linear_line_search_initial_step_scale="
            f"{scale:.17g}"
        )
    linear_preconditioner = getattr(args, "linear_preconditioner", None)
    if linear_preconditioner is not None:
        name = str(linear_preconditioner).strip()
        if not name:
            raise ValueError("linear_preconditioner must be nonempty.")
        overrides.append(f"runtime:recycling_jax_linear_preconditioner={name}")
    refresh = getattr(args, "linear_preconditioner_refresh", None)
    if refresh is not None:
        refresh_count = int(refresh)
        if refresh_count <= 0:
            raise ValueError("linear_preconditioner_refresh must be positive.")
        overrides.append(
            "runtime:recycling_jax_linear_preconditioner_refresh="
            f"{refresh_count}"
        )
    return overrides


def _canonical_preconditioner_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "parallel_transport": "parallel_line",
        "transport_line": "parallel_line",
        "neutral_diffusion": "neutral_line",
        "neutral_transport": "neutral_line",
        "parallel_momentum": "momentum_line",
        "momentum_transport": "momentum_line",
        "target_line": "sheath_line",
        "target_sheath": "sheath_line",
        "target_sheath_line": "sheath_line",
        "sheath_transport": "sheath_line",
        "line_field_schur": "field_line_schur",
        "transport_field_schur": "field_line_schur",
        "field_transport_schur": "field_line_schur",
        "sheath_schur": "target_schur",
        "plasma_neutral_schur": "neutral_plasma_schur",
    }
    return aliases.get(normalized, normalized)


def _canonical_initial_residual_mode(name: str) -> str:
    normalized = str(name or "residual").strip().lower().replace("-", "_")
    aliases = {
        "": "residual",
        "default": "residual",
        "residual": "residual",
        "standalone": "residual",
        "separate": "residual",
        "linearize": "linearize",
        "linearized": "linearize",
        "linearization": "linearize",
        "jacobian": "linearize",
    }
    if normalized not in aliases:
        raise ValueError(
            "initial residual mode must be 'residual' or 'linearize', "
            f"got {name!r}."
        )
    return aliases[normalized]


def _is_dynamic_preconditioner_name(name: str) -> bool:
    return _canonical_preconditioner_name(name) in {
        "linearized_diag",
        "field_sample_diag",
        "field_sample",
        "sampled_field_diag",
        "field_block_sample",
        "field_sample_block",
        "sampled_field_block",
        "field_split",
        "field_block_feedback_diag",
        "field_feedback_block",
        "feedback_field_block",
        "field_split_feedback",
        "field_diag",
        "field_jacobi",
        "field_diagonal",
        "local_block_diag",
        "block_jacobi",
        "parallel_line",
        "transport_line",
        "neutral_line",
        "neutral_parallel_line",
        "neutral_transport",
        "momentum_line",
        "momentum_parallel_line",
        "momentum_transport",
        "sheath_line",
        "target_line",
        "target_sheath_line",
        "sheath_transport",
        "field_line_schur",
        "target_schur",
        "neutral_plasma_schur",
    }


def _validate_required_linear_preconditioner(
    profile_report: dict[str, Any],
    required_linear_preconditioner: str,
) -> list[str]:
    expected = _canonical_preconditioner_name(required_linear_preconditioner)
    diagnostics = profile_report.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    reported = diagnostics.get("linear_preconditioner")
    reported_name = (
        None if reported is None else _canonical_preconditioner_name(str(reported))
    )
    errors: list[str] = []
    if reported_name != expected:
        errors.append(f"profile did not report linear_preconditioner={expected}")
    if _is_dynamic_preconditioner_name(expected):
        try:
            build_count = int(diagnostics.get("linear_preconditioner_build_count", 0))
        except (TypeError, ValueError):
            build_count = 0
        if build_count <= 0:
            errors.append(
                f"profile did not report any {expected} preconditioner builds"
            )
        try:
            build_seconds = float(
                diagnostics.get("linear_preconditioner_build_seconds", float("nan"))
            )
        except (TypeError, ValueError):
            build_seconds = float("nan")
        if not math.isfinite(build_seconds) or build_seconds < 0.0:
            errors.append(
                "profile did not report finite nonnegative "
                "linear_preconditioner_build_seconds"
            )
    return errors


def _validate_required_initial_residual_mode(
    profile_report: dict[str, Any],
    required_initial_residual_mode: str,
) -> list[str]:
    try:
        expected = _canonical_initial_residual_mode(required_initial_residual_mode)
    except ValueError as exc:
        return [str(exc)]
    diagnostics = profile_report.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    reported = diagnostics.get("initial_residual_mode")
    if reported is None:
        return ["profile did not report diagnostics.initial_residual_mode"]
    try:
        reported_mode = _canonical_initial_residual_mode(str(reported))
    except ValueError:
        return [
            "profile reported invalid diagnostics.initial_residual_mode="
            f"{reported!r}"
        ]
    if reported_mode != expected:
        return [
            "profile reported diagnostics.initial_residual_mode="
            f"{reported_mode}, expected {expected}"
        ]
    return []


def _validate_maximum_integer_value(
    profile_report: dict[str, Any],
    *,
    key: str,
    maximum: int,
    label: str,
    source: dict[str, Any] | None = None,
) -> list[str]:
    if int(maximum) < 0:
        return [f"profile received a negative {label} gate"]
    values = profile_report if source is None else source
    try:
        reported = int(values[key])
    except KeyError:
        return [f"profile did not report {key}"]
    except (TypeError, ValueError):
        return [f"profile did not report an integer {key}"]
    if reported > int(maximum):
        return [f"profile reported {reported} {label}, exceeding {int(maximum)}"]
    return []


def _validate_maximum_float_value(
    profile_report: dict[str, Any],
    *,
    key: str,
    maximum: float,
    label: str,
    source: dict[str, Any] | None = None,
) -> list[str]:
    if not math.isfinite(float(maximum)) or float(maximum) < 0.0:
        return [f"profile received an invalid {label} gate"]
    values = profile_report if source is None else source
    try:
        reported = float(values[key])
    except KeyError:
        return [f"profile did not report {key}"]
    except (TypeError, ValueError):
        return [f"profile did not report a finite {key}"]
    if not math.isfinite(reported):
        return [f"profile did not report a finite {key}"]
    if reported > float(maximum):
        return [
            f"profile reported {reported:.8e} {label}, "
            f"exceeding {float(maximum):.8e}"
        ]
    return []


def _validate_minimum_integer_value(
    profile_report: dict[str, Any],
    *,
    key: str,
    minimum: int,
    label: str,
    source: dict[str, Any] | None = None,
) -> list[str]:
    if int(minimum) < 0:
        return [f"profile received a negative {label} floor"]
    values = profile_report if source is None else source
    try:
        reported = int(values[key])
    except KeyError:
        return [f"profile did not report {key}"]
    except (TypeError, ValueError):
        return [f"profile did not report an integer {key}"]
    if reported < int(minimum):
        return [f"profile reported {reported} {label}, below {int(minimum)}"]
    return []


def _profile_gate_errors(
    profile_report: dict[str, Any], args: argparse.Namespace
) -> list[str]:
    errors: list[str] = []
    required_preconditioner = getattr(args, "require_linear_preconditioner", None)
    if required_preconditioner is not None:
        errors.extend(
            _validate_required_linear_preconditioner(
                profile_report, str(required_preconditioner)
            )
        )
    required_initial_residual_mode = getattr(
        args, "require_initial_residual_mode", None
    )
    if required_initial_residual_mode is not None:
        errors.extend(
            _validate_required_initial_residual_mode(
                profile_report,
                str(required_initial_residual_mode),
            )
        )
    if bool(getattr(args, "require_linear_operator_jitted", False)):
        diagnostics = profile_report.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        if not bool(diagnostics.get("linear_operator_jitted", False)):
            errors.append("profile did not report diagnostics.linear_operator_jitted=true")
    required_rhs_backend = getattr(args, "require_rhs_backend", None)
    if required_rhs_backend is not None:
        diagnostics = profile_report.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        reported_rhs_backend = diagnostics.get("rhs_backend")
        if str(reported_rhs_backend) != str(required_rhs_backend):
            errors.append(
                "profile reported diagnostics.rhs_backend="
                f"{reported_rhs_backend!r}, expected {required_rhs_backend!r}"
            )
    max_linear_iterations = getattr(args, "require_max_linear_iterations", None)
    if max_linear_iterations is not None:
        errors.extend(
            _validate_maximum_integer_value(
                profile_report,
                key="linear_iterations",
                maximum=int(max_linear_iterations),
                label="linear iterations",
            )
        )
    max_residual_inf_norm = getattr(args, "require_max_residual_inf_norm", None)
    if max_residual_inf_norm is not None:
        errors.extend(
            _validate_maximum_float_value(
                profile_report,
                key="residual_inf_norm",
                maximum=float(max_residual_inf_norm),
                label="residual inf-norm",
            )
        )
    diagnostics = profile_report.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    max_linear_update_residual = getattr(
        args, "require_max_linear_update_residual", None
    )
    if max_linear_update_residual is not None:
        errors.extend(
            _validate_maximum_float_value(
                profile_report,
                key="linear_update_residual_inf_norm",
                maximum=float(max_linear_update_residual),
                label="linear-update residual inf-norm",
                source=diagnostics,
            )
        )
    max_linear_update_relative_residual = getattr(
        args, "require_max_linear_update_relative_residual", None
    )
    if max_linear_update_relative_residual is not None:
        errors.extend(
            _validate_maximum_float_value(
                profile_report,
                key="linear_update_relative_residual",
                maximum=float(max_linear_update_relative_residual),
                label="linear-update relative residual",
                source=diagnostics,
            )
        )
    max_residual_evaluations = getattr(
        args, "require_max_residual_evaluations", None
    )
    if max_residual_evaluations is not None:
        errors.extend(
            _validate_maximum_integer_value(
                profile_report,
                key="residual_evaluation_count",
                maximum=int(max_residual_evaluations),
                label="residual evaluations",
                source=diagnostics,
            )
        )
    max_line_search_trials = getattr(args, "require_max_line_search_trials", None)
    if max_line_search_trials is not None:
        errors.extend(
            _validate_maximum_integer_value(
                profile_report,
                key="line_search_trial_count",
                maximum=int(max_line_search_trials),
                label="line-search trials",
                source=diagnostics,
            )
        )
    min_linear_operator_calls = getattr(
        args, "require_min_linear_operator_calls", None
    )
    if min_linear_operator_calls is not None:
        errors.extend(
            _validate_minimum_integer_value(
                profile_report,
                key="linear_operator_call_count",
                minimum=int(min_linear_operator_calls),
                label="linear-operator calls",
                source=diagnostics,
            )
        )
    max_linear_operator_calls = getattr(
        args, "require_max_linear_operator_calls", None
    )
    if max_linear_operator_calls is not None:
        errors.extend(
            _validate_maximum_integer_value(
                profile_report,
                key="linear_operator_call_count",
                maximum=int(max_linear_operator_calls),
                label="linear-operator calls",
                source=diagnostics,
            )
        )
    min_linear_iterations = getattr(args, "require_min_linear_iterations", None)
    if min_linear_iterations is not None:
        errors.extend(
            _validate_minimum_integer_value(
                profile_report,
                key="linear_iterations",
                minimum=int(min_linear_iterations),
                label="linear iterations",
            )
        )
    min_linear_solve_count = getattr(args, "require_min_linear_solve_count", None)
    if min_linear_solve_count is not None:
        errors.extend(
            _validate_minimum_integer_value(
                profile_report,
                key="linear_solve_count",
                minimum=int(min_linear_solve_count),
                label="linear solve attempts",
                source=diagnostics,
            )
        )
    min_nonlinear_iterations = getattr(args, "require_min_nonlinear_iterations", None)
    if min_nonlinear_iterations is not None:
        errors.extend(
            _validate_minimum_integer_value(
                profile_report,
                key="nonlinear_iterations",
                minimum=int(min_nonlinear_iterations),
                label="nonlinear iterations",
            )
        )
    max_preconditioner_builds = getattr(
        args, "require_max_preconditioner_builds", None
    )
    if max_preconditioner_builds is not None:
        errors.extend(
            _validate_maximum_integer_value(
                profile_report,
                key="linear_preconditioner_build_count",
                maximum=int(max_preconditioner_builds),
                label="preconditioner builds",
                source=diagnostics,
            )
        )
    max_preconditioner_applies = getattr(
        args, "require_max_preconditioner_applies", None
    )
    if max_preconditioner_applies is not None:
        errors.extend(
            _validate_maximum_integer_value(
                profile_report,
                key="linear_preconditioner_apply_count",
                maximum=int(max_preconditioner_applies),
                label="preconditioner applies",
                source=diagnostics,
            )
        )
    return errors


def _profile_once(
    args: argparse.Namespace, input_path: Path
) -> tuple[dict[str, Any], float]:
    from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
    from jax_drb.native.mesh import build_structured_mesh
    from jax_drb.native.metrics import build_structured_metrics
    from jax_drb.native.recycling_1d import (
        _build_recycling_runtime_model,
        _build_recycling_state_fields,
        advance_recycling_1d_backward_euler_step,
    )
    from jax_drb.native.units import resolved_dataset_scalars
    from jax_drb.runtime.run_config import RunConfiguration

    config = load_bout_input(input_path)
    overrides = _effective_overrides(args)
    if overrides:
        config = apply_bout_overrides(config, overrides)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}
    solver_mode = _solver_mode_for_backend(
        args.linear_solver_backend,
        active_array_rhs=bool(getattr(args, "active_array_rhs", False)),
    )

    started = perf_counter()
    next_fields, _next_integrals, info = advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=float(args.timestep),
        solver_mode=solver_mode,
        residual_tolerance=float(args.residual_tolerance),
        max_nonlinear_iterations=int(args.max_nonlinear_iterations),
    )
    elapsed = perf_counter() - started
    variable_cells = {
        name: int(getattr(value, "size", 0))
        for name, value in next_fields.items()
        if name in runtime_model.field_names
    }
    report = {
        "input_path": _public_input_path(args, input_path),
        "case": str(args.case),
        "solver_mode": solver_mode,
        "linear_solver_backend": str(args.linear_solver_backend),
        "active_array_rhs_requested": bool(
            getattr(args, "active_array_rhs", False)
        ),
        "overrides": list(overrides),
        "jit_residual_requested": bool(getattr(args, "jit_residual", False)),
        "jit_linear_operator_requested": bool(
            getattr(args, "jit_linear_operator", False)
        ),
        "warmup_runs": int(max(args.warmup_runs, 0)),
        "timestep": float(args.timestep),
        "residual_tolerance": float(args.residual_tolerance),
        "max_nonlinear_iterations": int(args.max_nonlinear_iterations),
        "field_names": list(runtime_model.field_names),
        "feedback_names": list(runtime_model.feedback_names),
        "mesh_active_shape": [
            int(mesh.xend - mesh.xstart + 1),
            int(mesh.yend - mesh.ystart + 1),
            int(mesh.nz),
        ],
        "active_size": int(info.active_size),
        "variable_cell_count": variable_cells,
        "state_size": int(
            sum(variable_cells.values()) + len(runtime_model.feedback_names)
        ),
        "residual_inf_norm": float(info.residual_inf_norm),
        "nonlinear_iterations": int(info.nonlinear_iterations),
        "linear_iterations": int(info.linear_iterations),
        "linear_solve_count": int(info.diagnostics.get("linear_solve_count", 0)),
        "linear_solver_status": info.diagnostics.get("linear_solver_status"),
        "linear_solver_success": info.diagnostics.get("linear_solver_success"),
        "linear_solver_reported_iterations": info.diagnostics.get(
            "linear_solver_reported_iterations"
        ),
        "diagnostics": dict(info.diagnostics),
    }
    return report, elapsed


def _run_with_optional_profile(args: argparse.Namespace, input_path: Path, jax):
    warmup_elapsed: list[float] = []
    for _ in range(max(int(args.warmup_runs), 0)):
        _, elapsed = _profile_once(args, input_path)
        warmup_elapsed.append(float(elapsed))

    profiler = None if args.skip_cprofile else cProfile.Profile()
    if profiler is not None:
        profiler.enable()
    trace_dir = args.output_dir / "jax_trace" if args.jax_trace else None
    if trace_dir is not None:
        if trace_dir.exists():
            shutil.rmtree(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
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
        report, elapsed = _profile_once(args, input_path)
    if profiler is not None:
        profiler.disable()
    timed_elapsed = [float(elapsed)]
    timed_residuals = [float(report["residual_inf_norm"])]
    for _ in range(max(int(args.timed_runs), 1) - 1):
        extra_report, extra_elapsed = _profile_once(args, input_path)
        timed_elapsed.append(float(extra_elapsed))
        timed_residuals.append(float(extra_report["residual_inf_norm"]))
    return (
        report,
        elapsed,
        profiler,
        trace_dir,
        warmup_elapsed,
        timed_elapsed,
        timed_residuals,
    )


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_environment(args)
    input_path = _resolve_input(args)

    import jax

    from jax_drb.runtime.memory import bytes_to_mebibytes, measure_peak_rss

    (
        profile_report,
        elapsed,
        profiler,
        trace_dir,
        warmup_elapsed,
        timed_elapsed,
        timed_residuals,
    ) = _run_with_optional_profile(args, input_path, jax)
    rss_payload = None
    rss_elapsed = None
    if args.rss_profile:
        (rss_report, rss_elapsed), rss_measurement = measure_peak_rss(
            lambda: _profile_once(args, input_path)
        )
        rss_payload = {
            "status": rss_measurement.status,
            "sample_count": int(rss_measurement.sample_count),
            "sampling_interval_seconds": float(
                rss_measurement.sampling_interval_seconds
            ),
            "run_seconds": float(rss_elapsed),
            "residual_inf_norm": float(rss_report["residual_inf_norm"]),
            "peak_rss_mebibytes": bytes_to_mebibytes(rss_measurement.peak_rss_bytes),
            "peak_rss_delta_mebibytes": bytes_to_mebibytes(
                rss_measurement.peak_rss_delta_bytes
            ),
        }

    cprofile_path = args.output_dir / "cprofile_top.txt"
    cprofile_binary_path = args.output_dir / "cprofile_stats.pstats"
    if profiler is not None:
        profiler.dump_stats(str(cprofile_binary_path))
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(int(args.cprofile_top))
        cprofile_path.write_text(stream.getvalue(), encoding="utf-8")

    memory_profile_path = None
    if args.device_memory_profile:
        memory_profile_path = args.output_dir / "device_memory_profile.prof"
        jax.profiler.save_device_memory_profile(str(memory_profile_path))

    gate_errors = _profile_gate_errors(profile_report, args)
    summary = {
        "case": f"recycling_1d_{args.case}_jax_linearized_gate",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "profiled_run_seconds": float(elapsed),
        "timed_runs": int(max(int(args.timed_runs), 1)),
        "timed_run_seconds": timed_elapsed,
        "timed_run_seconds_median": float(statistics.median(timed_elapsed)),
        "timed_run_residual_inf_norms": timed_residuals,
        "warmup_run_seconds": warmup_elapsed,
        "rss_profile": rss_payload,
        "profile": profile_report,
        "gate_requirements": {
            "linear_preconditioner": args.require_linear_preconditioner,
            "initial_residual_mode": args.require_initial_residual_mode,
            "linear_operator_jitted": bool(args.require_linear_operator_jitted),
            "rhs_backend": args.require_rhs_backend,
            "max_linear_iterations": args.require_max_linear_iterations,
            "max_residual_inf_norm": args.require_max_residual_inf_norm,
            "max_residual_evaluations": args.require_max_residual_evaluations,
            "max_line_search_trials": args.require_max_line_search_trials,
            "min_linear_operator_calls": args.require_min_linear_operator_calls,
            "max_linear_operator_calls": args.require_max_linear_operator_calls,
            "min_linear_iterations": args.require_min_linear_iterations,
            "min_linear_solve_count": args.require_min_linear_solve_count,
            "min_nonlinear_iterations": args.require_min_nonlinear_iterations,
            "max_preconditioner_builds": args.require_max_preconditioner_builds,
            "max_preconditioner_applies": args.require_max_preconditioner_applies,
        },
        "gate_passed": not bool(gate_errors),
        "gate_errors": gate_errors,
        "cprofile_top_path": None
        if profiler is None
        else _sanitize_public_path(cprofile_path),
        "cprofile_binary_path": None
        if profiler is None
        else _sanitize_public_path(cprofile_binary_path),
        "jax_trace_dir": None
        if trace_dir is None
        else _sanitize_public_path(trace_dir),
        "device_memory_profile_path": None
        if memory_profile_path is None
        else _sanitize_public_path(memory_profile_path),
        "xla_dump_dir": None
        if args.xla_dump_dir is None
        else _sanitize_public_path(args.xla_dump_dir),
        "compilation_cache_dir": (
            None
            if args.compilation_cache_dir is None
            else _sanitize_public_path(args.compilation_cache_dir)
        ),
        "interpretation": (
            "This gate profiles a real integrated recycling fixed-layout "
            "residual that reaches JAX linearization. The D/T/He mode exercises "
            "the multispecies residual seam used by the adaptive BDF trial solves; "
            "it is still a controlled BE gate, not a full production output-window profile."
        ),
    }
    summary_path = args.output_dir / "profile_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary_path)
    if cprofile_path.exists():
        print(cprofile_path)
    if trace_dir is not None:
        print(trace_dir)
    if memory_profile_path is not None:
        print(memory_profile_path)
    for error in gate_errors:
        print(f"gate_failure: {error}", file=sys.stderr)
    if gate_errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
