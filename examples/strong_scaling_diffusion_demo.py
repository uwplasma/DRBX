from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np

from jax_drb.validation.autodiff_diffusion import build_diffusion_autodiff_setup, compute_strong_scaling_points


@dataclass(frozen=True)
class StrongScalingSettings:
    output_root: Path
    cpu_device_counts: tuple[int, ...]
    gpu_device_counts: tuple[int, ...]
    total_batch: int
    nx: int
    ny: int
    timestep: float
    steps: int
    repeats: int
    remote_host: str
    skip_gpu: bool
    quiet: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure publication-style strong scaling for a gradient-enabled diffusion "
            "objective on CPU and optional remote GPUs."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "strong_scaling_diffusion_artifacts",
    )
    parser.add_argument("--cpu-device-counts", default="1,2,4,8")
    parser.add_argument("--gpu-device-counts", default="1,2")
    parser.add_argument("--total-batch", type=int, default=16)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--timestep", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--remote-host", default="office")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--device-count", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_device_counts(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def build_settings(args: argparse.Namespace) -> StrongScalingSettings:
    return StrongScalingSettings(
        output_root=args.output_root,
        cpu_device_counts=parse_device_counts(args.cpu_device_counts),
        gpu_device_counts=parse_device_counts(args.gpu_device_counts),
        total_batch=args.total_batch,
        nx=args.nx,
        ny=args.ny,
        timestep=args.timestep,
        steps=args.steps,
        repeats=args.repeats,
        remote_host=args.remote_host,
        skip_gpu=args.skip_gpu,
        quiet=args.quiet,
    )


def log(settings: StrongScalingSettings, title: str, mapping: dict[str, Any]) -> None:
    if settings.quiet:
        return
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def _worker_main(args: argparse.Namespace) -> int:
    import jax
    import jax.numpy as jnp
    from jax import jit, pmap, value_and_grad, vmap

    from jax_drb.validation.autodiff_diffusion import objective_for_physical_parameters

    if args.total_batch % args.device_count != 0:
        raise ValueError("total_batch must be divisible by device_count")

    setup = build_diffusion_autodiff_setup(nx=args.nx, ny=args.ny, timestep=args.timestep, steps=args.steps)
    local_device_count = jax.local_device_count()
    if local_device_count != args.device_count:
        raise RuntimeError(f"expected {args.device_count} local devices, found {local_device_count}")

    total_batch = args.total_batch
    local_batch = total_batch // args.device_count
    diffusivity = jnp.linspace(0.22, 0.52, total_batch, dtype=jnp.float64)
    amplitude = jnp.linspace(0.08, 0.24, total_batch, dtype=jnp.float64)
    centers = jnp.linspace(0.30, 0.70, total_batch, dtype=jnp.float64)
    widths = jnp.linspace(0.08, 0.18, total_batch, dtype=jnp.float64)
    global_batch = jnp.stack([diffusivity, amplitude, centers, widths], axis=1)
    def single_sample_loss(parameters: jnp.ndarray) -> jnp.ndarray:
        return objective_for_physical_parameters(parameters, setup, objective_kind="variance")

    def local_batch_value_and_grad(parameters: jnp.ndarray):
        values, gradients = vmap(value_and_grad(single_sample_loss))(parameters)
        return jnp.sum(values), gradients

    if args.device_count == 1:
        compiled = jit(local_batch_value_and_grad)
        warmup = compiled(global_batch)
        jax.block_until_ready(warmup)
    else:
        sharded_batch = global_batch.reshape(args.device_count, local_batch, 4)
        compiled = pmap(local_batch_value_and_grad)
        warmup = compiled(sharded_batch)
        jax.block_until_ready(warmup)

    timings: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        if args.device_count == 1:
            result = compiled(global_batch)
        else:
            result = compiled(sharded_batch)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - started)

    payload = {
        "backend": args.backend,
        "device_count": args.device_count,
        "parallel_kind": "single_device" if args.device_count == 1 else "pmap",
        "total_batch": total_batch,
        "local_batch": local_batch,
        "timings": timings,
        "best_seconds": min(timings),
        "mean_seconds": float(np.mean(timings)),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_local_cpu_worker(script_path: Path, settings: StrongScalingSettings, device_count: int) -> dict[str, Any]:
    if settings.total_batch % device_count != 0:
        raise ValueError("total_batch must be divisible by cpu device_count")
    local_batch = settings.total_batch // device_count
    command = [
        str(_repo_root() / ".venv" / "bin" / "python"),
        str(script_path),
        "--worker",
        "--backend",
        "cpu",
        "--device-count",
        "1",
        "--total-batch",
        str(local_batch),
        "--nx",
        str(settings.nx),
        "--ny",
        str(settings.ny),
        "--timestep",
        str(settings.timestep),
        "--steps",
        str(settings.steps),
        "--repeats",
        str(settings.repeats),
    ]
    processes: list[subprocess.Popen[str]] = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_repo_root() / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    try:
        for _ in range(device_count):
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=_repo_root(),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        worker_results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=300)
            if process.returncode != 0:
                raise RuntimeError(
                    f"cpu scaling worker failed for {device_count} workers:\n{stdout}\n{stderr}"
                )
            worker_results.append(json.loads(stdout.strip().splitlines()[-1]))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    return {
        "backend": "cpu",
        "device_count": device_count,
        "parallel_kind": "process_group",
        "total_batch": settings.total_batch,
        "local_batch": local_batch,
        "best_seconds": max(float(item["best_seconds"]) for item in worker_results),
        "mean_seconds": max(float(item["mean_seconds"]) for item in worker_results),
        "worker_results": worker_results,
    }


def stage_remote_tree(host: str, remote_root: str) -> None:
    command = (
        f"tar czf - src examples/strong_scaling_diffusion_demo.py | "
        f"ssh -o BatchMode=yes {shlex.quote(host)} "
        f"{shlex.quote(f'mkdir -p {remote_root} && tar xzf - -C {remote_root}')}"
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=_repo_root(), capture_output=True, text=True, check=False, timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(f"failed to stage remote scaling tree:\n{completed.stdout}\n{completed.stderr}")


def run_remote_gpu_worker(script_path: Path, settings: StrongScalingSettings, device_count: int, remote_root: str) -> dict[str, Any]:
    visible_devices = ",".join(str(index) for index in range(device_count))
    remote_script = f"{remote_root}/examples/{script_path.name}"
    remote_command = (
        f"cd {shlex.quote(remote_root)} && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)} "
        f"PYTHONPATH={shlex.quote(f'{remote_root}/src')} "
        f"python3 {shlex.quote(remote_script)} "
        f"--worker --backend gpu --device-count {device_count} "
        f"--total-batch {settings.total_batch} --nx {settings.nx} --ny {settings.ny} "
        f"--timestep {settings.timestep} --steps {settings.steps} --repeats {settings.repeats}"
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", settings.remote_host, remote_command],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"gpu scaling worker failed for {device_count} devices:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def cleanup_remote_tree(host: str, remote_root: str) -> None:
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"rm -rf {shlex.quote(remote_root)}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def write_analysis_json(settings: StrongScalingSettings, cpu_points, gpu_points, raw_results: dict[str, Any]) -> Path:
    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "cpu_device_counts": list(settings.cpu_device_counts),
            "gpu_device_counts": list(settings.gpu_device_counts),
            "total_batch": settings.total_batch,
            "nx": settings.nx,
            "ny": settings.ny,
            "timestep": settings.timestep,
            "steps": settings.steps,
            "repeats": settings.repeats,
        },
        "cpu": [point.__dict__ for point in cpu_points],
        "gpu": [point.__dict__ for point in gpu_points],
        "raw_results": raw_results,
    }
    path = data_dir / "strong_scaling_diffusion_analysis.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_publication_plot(settings: StrongScalingSettings, cpu_points, gpu_points) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images_dir = settings.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    for label, points, color in (
        ("CPU (local process group)", cpu_points, "#1d3557"),
        ("GPU (device pmap)", gpu_points, "#d62828"),
    ):
        if not points:
            continue
        devices = [point.device_count for point in points]
        elapsed = [point.elapsed_seconds for point in points]
        speedup = [point.speedup for point in points]
        efficiency = [point.efficiency for point in points]
        axes[0].plot(devices, elapsed, marker="o", linewidth=2.6, color=color, label=label)
        axes[1].plot(devices, speedup, marker="o", linewidth=2.6, color=color, label=f"{label} speedup")
        axes[1].plot(devices, efficiency, marker="s", linewidth=2.0, linestyle="--", color=color, alpha=0.75, label=f"{label} efficiency")

    axes[0].set_xlabel("device count")
    axes[0].set_ylabel("best elapsed time [s]")
    axes[0].set_title("Fixed-workload strong scaling")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    ideal_devices = list(settings.cpu_device_counts)
    if settings.gpu_device_counts:
        ideal_devices = sorted(set(ideal_devices + list(settings.gpu_device_counts)))
    if ideal_devices:
        axes[1].plot(ideal_devices, ideal_devices, color="#555555", linestyle=":", linewidth=1.5, label="ideal speedup")
    axes[1].set_xlabel("device count")
    axes[1].set_ylabel("speedup / efficiency")
    axes[1].set_title("Scaling mode comparison")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    path = images_dir / "strong_scaling_diffusion.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    if args.worker:
        return _worker_main(args)

    settings = build_settings(args)
    script_path = Path(__file__).resolve()

    cpu_results = []
    for device_count in settings.cpu_device_counts:
        result = run_local_cpu_worker(script_path, settings, device_count)
        cpu_results.append(result)
        log(settings, f"CPU scaling {device_count} device(s)", result)
    cpu_points = compute_strong_scaling_points(
        [(int(item["device_count"]), float(item["best_seconds"])) for item in cpu_results],
        backend="cpu",
    )

    gpu_results = []
    remote_root = f"/tmp/jax_drb_strong_scaling_{int(time.time())}"
    if not settings.skip_gpu and settings.gpu_device_counts:
        try:
            stage_remote_tree(settings.remote_host, remote_root)
            for device_count in settings.gpu_device_counts:
                result = run_remote_gpu_worker(script_path, settings, device_count, remote_root)
                gpu_results.append(result)
                log(settings, f"GPU scaling {device_count} device(s)", result)
        finally:
            cleanup_remote_tree(settings.remote_host, remote_root)
    gpu_points = compute_strong_scaling_points(
        [(int(item["device_count"]), float(item["best_seconds"])) for item in gpu_results],
        backend="gpu",
    )

    analysis_path = write_analysis_json(
        settings,
        cpu_points,
        gpu_points,
        raw_results={"cpu": cpu_results, "gpu": gpu_results},
    )
    plot_path = save_publication_plot(settings, cpu_points, gpu_points)
    log(settings, "Strong Scaling Artifacts", {"analysis_json": analysis_path, "plot": plot_path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
