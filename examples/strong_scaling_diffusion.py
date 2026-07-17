"""Fixed-workload strong scaling for a gradient-enabled diffusion objective.

A batch of ``TOTAL_BATCH`` diffusion parameter vectors is evaluated with
``value_and_grad`` (each sample differentiates through a full multi-step
rollout), and the same fixed workload is timed while the device count grows.
Three parallel modes are measured:

1. CPU process group: one single-device worker process per "device", each
   pinned to one BLAS/XLA thread, taking an equal share of the batch;
2. CPU host pmap: one worker process with
   ``--xla_force_host_platform_device_count=<n>`` forced host devices and a
   ``pmap`` over the sharded batch; and
3. optional remote GPUs over SSH (``RUN_REMOTE_GPU``): the source tree is
   staged to ``REMOTE_HOST`` and a ``pmap`` worker runs per GPU count.

Because the XLA host device count must be fixed before JAX initializes, each
measurement re-invokes this script as a subprocess; the single internal
handshake environment variable ``JAX_DRB_STRONG_SCALING_WORKER`` carries the
worker request (backend, device count, batch share) as JSON. All tunables are
the plain constants below.

The script prints per-mode timings and writes (relative to the current working
directory)
``docs/data/strong_scaling_diffusion_artifacts/data/strong_scaling_diffusion_analysis.json``
and
``docs/data/strong_scaling_diffusion_artifacts/images/strong_scaling_diffusion.png``.
Previously measured GPU points in an existing analysis JSON are preserved when
the GPU stage is skipped.

Run from the repository root:

    PYTHONPATH=src python examples/strong_scaling_diffusion.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from jax_drb.validation.autodiff_diffusion import (
    StrongScalingPoint,
    build_diffusion_autodiff_setup,
    compute_strong_scaling_points,
    objective_for_physical_parameters,
)

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/strong_scaling_diffusion_artifacts")  # artifact root (cwd-relative)
CPU_DEVICE_COUNTS = (1, 2, 4)   # laptop-sized sweep; office-scale runs use (1, 2, 4, 8)
TOTAL_BATCH = 16                # fixed global workload; must divide by every device count
NX = 256                        # radial grid points per sample
NY = 32                         # poloidal grid points per sample
TIMESTEP = 3.0                  # rollout output interval
STEPS = 8                       # rollout steps each gradient flows through
REPEATS = 3                     # timed repetitions per point (best time is kept)
RUN_REMOTE_GPU = False          # set True to add remote-GPU points over SSH
GPU_DEVICE_COUNTS = (1, 2)      # GPU counts measured when RUN_REMOTE_GPU is True
REMOTE_HOST = "office"          # SSH host with CUDA GPUs and a python3 + jax install

# Internal worker handshake (single env var, JSON payload). Not a user setting:
# the XLA host device count must be configured before JAX starts, so each
# measurement re-invokes this script with this variable set.
WORKER_ENV_VAR = "JAX_DRB_STRONG_SCALING_WORKER"


# --- worker branch: one timed measurement inside a prepared environment -----------
def run_worker(request: dict[str, Any]) -> None:
    """Time the batched value-and-grad workload and print one JSON line."""

    import jax
    import jax.numpy as jnp
    from jax import jit, pmap, value_and_grad, vmap

    backend = str(request["backend"])
    device_count = int(request["device_count"])
    total_batch = int(request["total_batch"])
    if total_batch % device_count != 0:
        raise ValueError("total_batch must be divisible by device_count")

    setup = build_diffusion_autodiff_setup(nx=NX, ny=NY, timestep=TIMESTEP, steps=STEPS)
    local_device_count = jax.local_device_count()
    if local_device_count != device_count:
        raise RuntimeError(f"expected {device_count} local devices, found {local_device_count}")

    local_batch = total_batch // device_count
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

    if device_count == 1:
        compiled = jit(local_batch_value_and_grad)
        batch = global_batch
    else:
        compiled = pmap(local_batch_value_and_grad)
        batch = global_batch.reshape(device_count, local_batch, 4)
    warmup = compiled(batch)
    jax.block_until_ready(warmup)

    timings: list[float] = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        result = compiled(batch)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - started)

    payload = {
        "backend": backend,
        "device_count": device_count,
        "parallel_kind": "single_device" if device_count == 1 else "pmap",
        "total_batch": total_batch,
        "local_batch": local_batch,
        "timings": timings,
        "best_seconds": min(timings),
        "mean_seconds": float(np.mean(timings)),
    }
    print(json.dumps(payload, sort_keys=True))


if os.environ.get(WORKER_ENV_VAR):
    run_worker(json.loads(os.environ[WORKER_ENV_VAR]))
    sys.exit(0)


# --- parent-side helpers ----------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def worker_env(request: dict[str, Any], extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.update(SINGLE_THREAD_ENV)
    env.update(extra)
    env[WORKER_ENV_VAR] = json.dumps(request)
    return env


def run_local_cpu_worker(device_count: int) -> dict[str, Any]:
    """Process-group mode: one single-device worker process per device."""

    if TOTAL_BATCH % device_count != 0:
        raise ValueError("TOTAL_BATCH must be divisible by cpu device_count")
    local_batch = TOTAL_BATCH // device_count
    env = worker_env(
        {"backend": "cpu", "device_count": 1, "total_batch": local_batch},
        {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"},
    )
    processes: list[subprocess.Popen[str]] = []
    try:
        for _ in range(device_count):
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(SCRIPT_PATH)],
                    cwd=REPO_ROOT,
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
        "total_batch": TOTAL_BATCH,
        "local_batch": local_batch,
        "best_seconds": max(float(item["best_seconds"]) for item in worker_results),
        "mean_seconds": max(float(item["mean_seconds"]) for item in worker_results),
        "worker_results": worker_results,
    }


def run_local_cpu_host_pmap_worker(device_count: int) -> dict[str, Any]:
    """Host-pmap mode: one process with forced XLA host devices and pmap."""

    env = worker_env(
        {"backend": "cpu", "device_count": device_count, "total_batch": TOTAL_BATCH},
        {
            "JAX_DRB_HOST_DEVICE_COUNT": str(device_count),
            "XLA_FLAGS": (
                f"--xla_force_host_platform_device_count={device_count} "
                "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
            ),
        },
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cpu host pmap worker failed for {device_count} devices:\n{completed.stdout}\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload["parallel_kind"] = "host_pmap"
    return payload


def stage_remote_tree(host: str, remote_root: str) -> None:
    command = (
        f"tar czf - src examples/strong_scaling_diffusion.py | "
        f"ssh -o BatchMode=yes {shlex.quote(host)} "
        f"{shlex.quote(f'mkdir -p {remote_root} && tar xzf - -C {remote_root}')}"
    )
    completed = subprocess.run(
        ["bash", "-lc", command], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=300
    )
    if completed.returncode != 0:
        raise RuntimeError(f"failed to stage remote scaling tree:\n{completed.stdout}\n{completed.stderr}")


def run_remote_gpu_worker(device_count: int, remote_root: str) -> dict[str, Any]:
    visible_devices = ",".join(str(index) for index in range(device_count))
    remote_script = f"{remote_root}/examples/{SCRIPT_PATH.name}"
    request = json.dumps({"backend": "gpu", "device_count": device_count, "total_batch": TOTAL_BATCH})
    remote_command = (
        f"cd {shlex.quote(remote_root)} && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)} "
        f"PYTHONPATH={shlex.quote(f'{remote_root}/src')} "
        f"{WORKER_ENV_VAR}={shlex.quote(request)} "
        f"python3 {shlex.quote(remote_script)}"
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, remote_command],
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


def load_existing_gpu_points(output_root: Path) -> list[StrongScalingPoint]:
    """Reload previously measured GPU points so a CPU-only rerun keeps them."""

    analysis_path = output_root / "data" / "strong_scaling_diffusion_analysis.json"
    if not analysis_path.exists():
        return []
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    return [
        StrongScalingPoint(
            backend=str(entry["backend"]),
            device_count=int(entry["device_count"]),
            elapsed_seconds=float(entry["elapsed_seconds"]),
            speedup=float(entry["speedup"]),
            efficiency=float(entry["efficiency"]),
        )
        for entry in payload.get("gpu", [])
    ]


# --- CPU process-group sweep ------------------------------------------------------
print(f"CPU process-group sweep over device counts {CPU_DEVICE_COUNTS} "
      f"(batch {TOTAL_BATCH}, grid {NX}x{NY}, {STEPS} steps, {REPEATS} repeats)...")
cpu_process_group_results = []
for device_count in CPU_DEVICE_COUNTS:
    result = run_local_cpu_worker(device_count)
    cpu_process_group_results.append(result)
    print(f"  {device_count} device(s): best {result['best_seconds']:.3f} s, "
          f"mean {result['mean_seconds']:.3f} s")
cpu_process_group_points = compute_strong_scaling_points(
    [(int(item["device_count"]), float(item["best_seconds"])) for item in cpu_process_group_results],
    backend="cpu",
)

# --- CPU host-pmap sweep ----------------------------------------------------------
print(f"CPU host-pmap sweep over device counts {CPU_DEVICE_COUNTS}...")
cpu_host_pmap_results = []
for device_count in CPU_DEVICE_COUNTS:
    result = run_local_cpu_host_pmap_worker(device_count)
    cpu_host_pmap_results.append(result)
    print(f"  {device_count} device(s): best {result['best_seconds']:.3f} s, "
          f"mean {result['mean_seconds']:.3f} s")
cpu_host_pmap_points = compute_strong_scaling_points(
    [(int(item["device_count"]), float(item["best_seconds"])) for item in cpu_host_pmap_results],
    backend="cpu_host_pmap",
)

# --- optional remote GPU sweep ----------------------------------------------------
gpu_results: list[dict[str, Any]] = []
if RUN_REMOTE_GPU and GPU_DEVICE_COUNTS:
    remote_root = f"/tmp/jax_drb_strong_scaling_{int(time.time())}"
    print(f"staging source tree to {REMOTE_HOST}:{remote_root} for the GPU sweep...")
    try:
        stage_remote_tree(REMOTE_HOST, remote_root)
        for device_count in GPU_DEVICE_COUNTS:
            result = run_remote_gpu_worker(device_count, remote_root)
            gpu_results.append(result)
            print(f"  {device_count} GPU(s): best {result['best_seconds']:.3f} s, "
                  f"mean {result['mean_seconds']:.3f} s")
    finally:
        cleanup_remote_tree(REMOTE_HOST, remote_root)
else:
    print("remote GPU sweep skipped (set RUN_REMOTE_GPU = True to enable it)")
gpu_points = compute_strong_scaling_points(
    [(int(item["device_count"]), float(item["best_seconds"])) for item in gpu_results],
    backend="gpu",
)
if not gpu_points:
    gpu_points = load_existing_gpu_points(OUTPUT_ROOT)
    if gpu_points:
        print(f"kept {len(gpu_points)} previously measured GPU point(s) from the existing analysis JSON")

# --- write the analysis JSON ------------------------------------------------------
data_dir = OUTPUT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)
analysis_path = data_dir / "strong_scaling_diffusion_analysis.json"
persisted_gpu_points = [point.__dict__ for point in gpu_points]
persisted_gpu_results: list[dict[str, Any]] = gpu_results
if not gpu_results and analysis_path.exists():
    existing_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    persisted_gpu_results = list(existing_payload.get("raw_results", {}).get("gpu", []))
analysis_payload = {
    "settings": {
        "cpu_device_counts": list(CPU_DEVICE_COUNTS),
        "gpu_device_counts": list(GPU_DEVICE_COUNTS),
        "total_batch": TOTAL_BATCH,
        "nx": NX,
        "ny": NY,
        "timestep": TIMESTEP,
        "steps": STEPS,
        "repeats": REPEATS,
    },
    "cpu": [point.__dict__ for point in cpu_process_group_points],
    "cpu_host_pmap": [point.__dict__ for point in cpu_host_pmap_points],
    "gpu": persisted_gpu_points,
    "raw_results": {
        "cpu": cpu_process_group_results,
        "cpu_host_pmap": cpu_host_pmap_results,
        "gpu": persisted_gpu_results,
    },
}
analysis_path.write_text(json.dumps(analysis_payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote analysis JSON: {analysis_path}")

# --- summary plot -----------------------------------------------------------------
images_dir = OUTPUT_ROOT / "images"
images_dir.mkdir(parents=True, exist_ok=True)
figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

for label, points, color in (
    ("CPU (local process group)", cpu_process_group_points, "#1d3557"),
    ("CPU (host device pmap)", cpu_host_pmap_points, "#2a9d8f"),
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

ideal_devices = sorted(set(CPU_DEVICE_COUNTS) | (set(GPU_DEVICE_COUNTS) if RUN_REMOTE_GPU else set()))
axes[1].plot(ideal_devices, ideal_devices, color="#555555", linestyle=":", linewidth=1.5, label="ideal speedup")
axes[1].set_xlabel("device count")
axes[1].set_ylabel("speedup / efficiency")
axes[1].set_title("Scaling mode comparison")
axes[1].grid(alpha=0.25)
axes[1].legend(frameon=False, ncol=2)

plot_path = images_dir / "strong_scaling_diffusion.png"
figure.savefig(plot_path, dpi=220)
plt.close(figure)
print(f"wrote scaling plot: {plot_path}")
