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


DEFAULT_TIMEOUT_SECONDS = 7200
REFERENCE_INPUT_RELATIVE_PATHS = {
    "hydrogen": Path("tests") / "integrated" / "1D-recycling" / "data" / "BOUT.inp",
    "dthe": Path("tests") / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
}
REFERENCE_INPUT_LABELS = {
    "hydrogen": "hydrogen recycling",
    "dthe": "D/T/He recycling",
}


@dataclass(frozen=True)
class CampaignCommand:
    name: str
    description: str
    command: tuple[str, ...]
    requires_reference: bool = False
    required_reference_inputs: tuple[str, ...] = ()
    requires_gpu: bool = False


@dataclass(frozen=True)
class CampaignResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    timed_out: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_reference_root() -> Path | None:
    value = os.environ.get("JAX_DRB_REFERENCE_ROOT")
    return None if not value else Path(value).expanduser()


def _reference_root_args(reference_root: Path | None) -> tuple[str, ...]:
    return () if reference_root is None else ("--reference-root", str(reference_root.expanduser()))


def reference_input_relative_path(case: str) -> Path:
    try:
        return REFERENCE_INPUT_RELATIVE_PATHS[case]
    except KeyError as exc:
        known = ", ".join(sorted(REFERENCE_INPUT_RELATIVE_PATHS))
        raise ValueError(f"unknown reference input case {case!r}; expected one of: {known}") from exc


def validate_reference_root(reference_root: Path, required_inputs: tuple[str, ...] = ()) -> None:
    root = reference_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(
            f"reference root {root} is not a directory; set --reference-root or "
            "JAX_DRB_REFERENCE_ROOT to a Hermès reference root."
        )
    for case in required_inputs:
        relative_path = reference_input_relative_path(case)
        input_path = root / relative_path
        if input_path.is_file():
            continue
        label = REFERENCE_INPUT_LABELS.get(case, case)
        raise ValueError(
            f"reference root {root} is missing required {label} input "
            f"{relative_path.as_posix()}; set --reference-root or JAX_DRB_REFERENCE_ROOT "
            "to a root containing that file. For nonstandard staged decks, run the "
            "profile_recycling_jax_linearized_gate.py or profile_recycling_batched_jvp_gate.py "
            "profiler directly with --input-path /path/to/BOUT.inp."
        )


def _campaign_command_map(
    *,
    python_executable: str,
    repo_root: Path,
    reference_root: Path | None,
    output_root: Path,
    fast_timeout_seconds: int,
) -> dict[str, CampaignCommand]:
    scripts = repo_root / "scripts"
    examples = repo_root / "examples" / "engineering"
    output_root = output_root.expanduser()
    reference_args = _reference_root_args(reference_root)
    return {
        "scheduled-fast-research": CampaignCommand(
            name="scheduled-fast-research",
            description="Bounded public research-grade pytest slices that do not require external reference decks.",
            command=(
                python_executable,
                str(scripts / "run_fast_research_checks.py"),
                "--timeout-seconds",
                str(int(fast_timeout_seconds)),
            ),
        ),
        "closeout-coverage": CampaignCommand(
            name="closeout-coverage",
            description="95% closeout coverage gate over the promoted public surface.",
            command=(python_executable, str(scripts / "run_closeout_coverage.py")),
        ),
        "promoted-solver-coverage": CampaignCommand(
            name="promoted-solver-coverage",
            description="95% promoted native-solver and public-surface coverage gate.",
            command=(python_executable, str(scripts / "run_promoted_solver_coverage.py")),
        ),
        "local-cpu-scaling": CampaignCommand(
            name="local-cpu-scaling",
            description="Heavy fixed-work ensemble scaling campaign on local CPU workers.",
            command=(python_executable, str(examples / "local_cpu_scaling_campaign_demo.py")),
        ),
        "atomic-rate-throughput-gate": CampaignCommand(
            name="atomic-rate-throughput-gate",
            description="Batched atomic-rate and autodiff source-kernel throughput gate.",
            command=(
                python_executable,
                str(scripts / "profile_atomic_rate_throughput_gate.py"),
                "--output-dir",
                str(output_root / "runtime_profile_artifacts" / "atomic_rate_throughput_gate_cpu"),
                "--timed-runs",
                "3",
            ),
        ),
        "live-reference": CampaignCommand(
            name="live-reference",
            description="Same-machine native-versus-live-reference validation matrix.",
            command=(
                python_executable,
                str(examples / "hermes_live_rerun_campaign_demo.py"),
                *reference_args,
                "--output-root",
                str(output_root / "hermes_live_rerun_campaign_artifacts"),
            ),
            requires_reference=True,
        ),
        "heavy-recycling-profile": CampaignCommand(
            name="heavy-recycling-profile",
            description="Full production D/T/He recycling one-step cProfile/RSS bundle.",
            command=(
                python_executable,
                str(scripts / "profile_curated_case.py"),
                "recycling_dthe_one_step",
                *reference_args,
                "--output-dir",
                str(output_root / "runtime_profile_artifacts" / "recycling_dthe_one_step"),
                "--warm-runs",
                "0",
                "--timed-runs",
                "1",
                "--cprofile-top",
                "35",
                "--rss-profile",
            ),
            requires_reference=True,
            required_reference_inputs=("dthe",),
        ),
        "dthe-jax-linearized-gate": CampaignCommand(
            name="dthe-jax-linearized-gate",
            description="D/T/He fixed-layout recycling residual through the JAX-linearized gate.",
            command=(
                python_executable,
                str(scripts / "profile_recycling_jax_linearized_gate.py"),
                *reference_args,
                "--case",
                "dthe",
                "--output-dir",
                str(output_root / "runtime_profile_artifacts" / "recycling_dthe_jax_linearized_gate"),
                "--rss-profile",
                "--skip-cprofile",
            ),
            requires_reference=True,
            required_reference_inputs=("dthe",),
        ),
        "dthe-batched-jvp-gate": CampaignCommand(
            name="dthe-batched-jvp-gate",
            description="Batched D/T/He recycling residual/JVP differentiability and throughput gate.",
            command=(
                python_executable,
                str(scripts / "profile_recycling_batched_jvp_gate.py"),
                *reference_args,
                "--case",
                "dthe",
                "--output-dir",
                str(output_root / "runtime_profile_artifacts" / "recycling_dthe_batched_jvp_gate_cpu"),
                "--override",
                "mesh:ny=100",
                "--batch-sizes",
                "1,4,16,64",
                "--timed-runs",
                "3",
            ),
            requires_reference=True,
            required_reference_inputs=("dthe",),
        ),
        "gpu-dthe-jax-linearized-gate": CampaignCommand(
            name="gpu-dthe-jax-linearized-gate",
            description="Large D/T/He fixed-layout recycling residual gate for GPU trace/memory profiling.",
            command=(
                python_executable,
                str(scripts / "profile_recycling_jax_linearized_gate.py"),
                *reference_args,
                "--case",
                "dthe",
                "--output-dir",
                str(output_root / "runtime_profile_artifacts" / "recycling_dthe_jax_linearized_gate_gpu_large"),
                "--override",
                "mesh:ny=400",
                "--timestep",
                "1e-4",
                "--max-nonlinear-iterations",
                "2",
                "--warmup-runs",
                "1",
                "--timed-runs",
                "5",
                "--rss-profile",
                "--skip-cprofile",
                "--jax-trace",
                "--device-memory-profile",
                "--compilation-cache-dir",
                str(repo_root / "tmp" / "jax_cache" / "recycling_dthe_jax_linearized_gate_gpu_large"),
            ),
            requires_reference=True,
            required_reference_inputs=("dthe",),
            requires_gpu=True,
        ),
        "gpu-dthe-full-output-jvp-profile": CampaignCommand(
            name="gpu-dthe-full-output-jvp-profile",
            description="Full D/T/He output-window recycling profile through the fixed-full-field JVP BDF seam.",
            command=(
                python_executable,
                str(scripts / "profile_curated_case.py"),
                "recycling_dthe_one_step",
                *reference_args,
                "--output-dir",
                str(output_root / "runtime_profile_artifacts" / "recycling_dthe_full_output_jvp_gpu"),
                "--override",
                "runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp",
                "--warm-runs",
                "1",
                "--timed-runs",
                "3",
                "--rss-profile",
                "--skip-cprofile",
                "--jax-trace",
                "--device-memory-profile",
                "--compilation-cache-dir",
                str(repo_root / "tmp" / "jax_cache" / "recycling_dthe_full_output_jvp_gpu"),
            ),
            requires_reference=True,
            required_reference_inputs=("dthe",),
            requires_gpu=True,
        ),
        "gpu-dthe-batched-jvp-gate": CampaignCommand(
            name="gpu-dthe-batched-jvp-gate",
            description="Multi-device D/T/He batched residual/JVP throughput gate with pmap parity metadata.",
            command=(
                python_executable,
                str(scripts / "profile_recycling_batched_jvp_gate.py"),
                *reference_args,
                "--case",
                "dthe",
                "--output-dir",
                str(output_root / "runtime_profile_artifacts" / "recycling_dthe_batched_jvp_gate_gpu_pmap"),
                "--override",
                "mesh:ny=200",
                "--batch-sizes",
                "2,4,8,16,32,64,128",
                "--timed-runs",
                "7",
                "--skip-objective-grad-check",
                "--jax-trace",
                "--device-memory-profile",
                "--compilation-cache-dir",
                str(repo_root / "tmp" / "jax_cache" / "recycling_dthe_batched_jvp_gate_gpu_pmap"),
            ),
            requires_reference=True,
            required_reference_inputs=("dthe",),
            requires_gpu=True,
        ),
    }


def expand_campaign_names(requested: tuple[str, ...]) -> tuple[str, ...]:
    names = requested or ("scheduled-fast-research",)
    expanded: list[str] = []
    for name in names:
        if name == "all-local":
            expanded.extend(
                (
                    "scheduled-fast-research",
                    "atomic-rate-throughput-gate",
                    "local-cpu-scaling",
                    "dthe-jax-linearized-gate",
                    "dthe-batched-jvp-gate",
                    "heavy-recycling-profile",
                    "live-reference",
                )
            )
        elif name == "all-ci":
            expanded.extend(("scheduled-fast-research", "closeout-coverage", "promoted-solver-coverage"))
        elif name == "all-gpu":
            expanded.extend(
                (
                    "gpu-dthe-jax-linearized-gate",
                    "gpu-dthe-full-output-jvp-profile",
                    "gpu-dthe-batched-jvp-gate",
                )
            )
        else:
            expanded.append(name)
    return tuple(dict.fromkeys(expanded))


def build_campaign_commands(
    *,
    campaign_names: tuple[str, ...],
    python_executable: str,
    repo_root: Path,
    reference_root: Path | None,
    output_root: Path,
    fast_timeout_seconds: int,
) -> tuple[CampaignCommand, ...]:
    mapping = _campaign_command_map(
        python_executable=python_executable,
        repo_root=repo_root,
        reference_root=reference_root,
        output_root=output_root,
        fast_timeout_seconds=fast_timeout_seconds,
    )
    commands: list[CampaignCommand] = []
    for name in expand_campaign_names(campaign_names):
        try:
            command = mapping[name]
        except KeyError as exc:
            known = ", ".join(sorted((*mapping, "all-ci", "all-gpu", "all-local")))
            raise ValueError(f"unknown campaign {name!r}; expected one of: {known}") from exc
        if command.requires_reference and reference_root is None:
            raise ValueError(f"campaign {name!r} requires --reference-root or JAX_DRB_REFERENCE_ROOT")
        if command.requires_reference and reference_root is not None:
            validate_reference_root(reference_root, command.required_reference_inputs)
        commands.append(command)
    return tuple(commands)


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_campaign_command(command: CampaignCommand, *, cwd: Path, timeout_seconds: int) -> CampaignResult:
    started = time.monotonic()
    env = dict(os.environ)
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else os.pathsep.join((src_path, env["PYTHONPATH"]))
    env["JAX_DRB_PRECISION"] = "float64"
    env["JAX_ENABLE_X64"] = "true"
    if command.requires_gpu:
        env.setdefault("JAX_PLATFORMS", "cuda")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        completed = subprocess.run(command.command, cwd=cwd, env=env, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return CampaignResult(
            name=command.name,
            command=command.command,
            returncode=124,
            elapsed_seconds=time.monotonic() - started,
            timed_out=True,
        )
    return CampaignResult(
        name=command.name,
        command=command.command,
        returncode=int(completed.returncode),
        elapsed_seconds=time.monotonic() - started,
        timed_out=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible research-grade campaign bundles. Hosted CI should use "
            "scheduled-fast-research/all-ci; live reference and heavy profiling "
            "campaigns are intended for local or self-hosted machines with reference decks."
        )
    )
    parser.add_argument(
        "--campaign",
        action="append",
        default=[],
        help=(
            "Campaign to run. Repeat for multiple campaigns. Supported names include "
            "scheduled-fast-research, closeout-coverage, promoted-solver-coverage, "
            "local-cpu-scaling, live-reference, heavy-recycling-profile, "
            "dthe-jax-linearized-gate, dthe-batched-jvp-gate, "
            "gpu-dthe-jax-linearized-gate, gpu-dthe-batched-jvp-gate, "
            "all-ci, all-local, and all-gpu."
        ),
    )
    parser.add_argument("--reference-root", type=Path, default=_default_reference_root())
    parser.add_argument("--output-root", type=Path, default=_repo_root() / "docs" / "data")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--fast-timeout-seconds", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    try:
        commands = build_campaign_commands(
            campaign_names=tuple(args.campaign),
            python_executable=sys.executable,
            repo_root=repo_root,
            reference_root=args.reference_root,
            output_root=args.output_root,
            fast_timeout_seconds=int(args.fast_timeout_seconds),
        )
    except ValueError as exc:
        print(f"[campaign] {exc}", file=sys.stderr)
        return 2

    for command in commands:
        print(f"[campaign] {command.name}: {command.description}")
        if command.requires_reference and args.reference_root is not None:
            if command.required_reference_inputs:
                for case in command.required_reference_inputs:
                    print(
                        "[campaign] reference input: "
                        f"{args.reference_root.expanduser() / reference_input_relative_path(case)}"
                    )
            else:
                print(f"[campaign] reference root: {args.reference_root.expanduser()}")
        if command.requires_gpu:
            print(
                "[campaign] GPU prerequisite: CUDA-visible JAX devices "
                "(for NVIDIA hosts, set JAX_PLATFORMS=cuda and CUDA_VISIBLE_DEVICES as needed)"
            )
        print(f"[campaign] command: {_format_command(command.command)}")
        if args.dry_run:
            continue
        result = run_campaign_command(command, cwd=repo_root, timeout_seconds=int(args.timeout_seconds))
        if result.timed_out:
            print(
                f"[campaign] {result.name} exceeded {args.timeout_seconds}s after {result.elapsed_seconds:.1f}s",
                file=sys.stderr,
            )
            return result.returncode
        if result.returncode != 0:
            print(
                f"[campaign] {result.name} failed with exit code {result.returncode} "
                f"after {result.elapsed_seconds:.1f}s",
                file=sys.stderr,
            )
            return result.returncode
        print(f"[campaign] {result.name} passed in {result.elapsed_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
