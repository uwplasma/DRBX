"""Fixed-workload strong scaling for a gradient-enabled diffusion objective.

A batch of ``TOTAL_BATCH`` diffusion parameter vectors is evaluated with
``value_and_grad`` (each sample differentiates through a full multi-step
rollout) and the same fixed workload is timed while the device count grows, in
three modes:

1. CPU process group: one single-device, single-thread worker process per
   "device", each taking an equal share of the batch;
2. CPU host pmap: one worker process with
   ``--xla_force_host_platform_device_count=<n>`` and a ``pmap`` over the batch;
3. optional remote GPUs over SSH (``RUN_REMOTE_GPU``): ``src`` is staged to
   ``REMOTE_HOST`` and a ``pmap`` worker runs per GPU count.

The XLA host device count must be fixed before JAX starts, so every measurement
runs the short ``WORKER_CODE`` below in a fresh ``python -c`` subprocess. The
default is a laptop-sized sweep, not the office-scale measurement quoted in the
docs (that used ``CPU_DEVICE_COUNTS = (1, 2, 4, 8)``).

Run from the repository root:

    PYTHONPATH=src python examples/strong_scaling_diffusion.py

Writes (relative to the current working directory)
``docs/data/strong_scaling_diffusion_artifacts/data/strong_scaling_diffusion_analysis.json``
and ``.../images/strong_scaling_diffusion.png``. Previously measured GPU points
in an existing analysis JSON are kept when the GPU stage is skipped.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drbx.validation.autodiff_diffusion import StrongScalingPoint, compute_strong_scaling_points

# ---- PARAMETERS ----
OUTPUT_ROOT = Path("docs/data/strong_scaling_diffusion_artifacts")  # artifact root (cwd-relative)
CPU_DEVICE_COUNTS = (1, 2, 4)   # laptop-sized sweep; office-scale runs use (1, 2, 4, 8)
TOTAL_BATCH = 16                # fixed global workload; must divide by every device count
NX, NY = 256, 32                # grid points per sample
TIMESTEP = 3.0                  # rollout output interval
STEPS = 8                       # rollout steps each gradient flows through
REPEATS = 3                     # timed repetitions per point (best time is kept)
RUN_REMOTE_GPU = False          # set True to add remote-GPU points over SSH
GPU_DEVICE_COUNTS = (1, 2)      # GPU counts measured when RUN_REMOTE_GPU is True
REMOTE_HOST = "office"          # SSH host with CUDA GPUs and a python3 + jax install

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_CODE = """
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from jax import jit, pmap, value_and_grad, vmap
from drbx.validation.autodiff_diffusion import build_diffusion_autodiff_setup, objective_for_physical_parameters
r = json.loads(sys.argv[1]); n = r["device_count"]; total = r["total_batch"]
assert jax.local_device_count() == n, f"expected {n} devices, found {jax.local_device_count()}"
setup = build_diffusion_autodiff_setup(nx=r["nx"], ny=r["ny"], timestep=r["timestep"], steps=r["steps"])
batch = jnp.stack([jnp.linspace(a, b, total, dtype=jnp.float64) for a, b in
                   ((0.22, 0.52), (0.08, 0.24), (0.30, 0.70), (0.08, 0.18))], axis=1)
def local(p):
    v, g = vmap(value_and_grad(lambda q: objective_for_physical_parameters(q, setup, objective_kind="variance")))(p)
    return jnp.sum(v), g
fn = jit(local) if n == 1 else pmap(local)
if n > 1: batch = batch.reshape(n, total // n, 4)
jax.block_until_ready(fn(batch))
t = []
for _ in range(r["repeats"]):
    s = time.perf_counter(); jax.block_until_ready(fn(batch)); t.append(time.perf_counter() - s)
print(json.dumps({"backend": r["backend"], "device_count": n, "total_batch": total, "local_batch": total // n,
                  "parallel_kind": "single_device" if n == 1 else "pmap", "timings": t,
                  "best_seconds": min(t), "mean_seconds": float(np.mean(t))}, sort_keys=True))
"""
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
SINGLE_THREAD_XLA = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def request(backend, device_count, total_batch):
    return json.dumps({"backend": backend, "device_count": device_count, "total_batch": total_batch,
                       "nx": NX, "ny": NY, "timestep": TIMESTEP, "steps": STEPS, "repeats": REPEATS})


def child_env(xla_flags):
    return {**os.environ, **THREAD_ENV, "PYTHONPATH": str(REPO_ROOT / "src"), "XLA_FLAGS": xla_flags}


def last_json(proc_out, proc_err, code, what):
    if code != 0:
        raise RuntimeError(f"{what} failed:\n{proc_out}\n{proc_err}")
    return json.loads(proc_out.strip().splitlines()[-1])


def report(label, result):
    print(f"  {label}: best {result['best_seconds']:.3f} s, mean {result['mean_seconds']:.3f} s", flush=True)


wall_start = time.perf_counter()
print(f"strong scaling: batch {TOTAL_BATCH}, grid {NX}x{NY}, {STEPS} steps, {REPEATS} repeats, "
      f"CPU device counts {CPU_DEVICE_COUNTS}")

# --- CPU process group: n single-device workers, each with TOTAL_BATCH / n samples ---
print("CPU process-group sweep...")
cpu_results = []
for n in CPU_DEVICE_COUNTS:
    procs = [subprocess.Popen([sys.executable, "-c", WORKER_CODE, request("cpu", 1, TOTAL_BATCH // n)],
                              cwd=REPO_ROOT, env=child_env(SINGLE_THREAD_XLA), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(n)]
    workers = [last_json(*p.communicate(timeout=300), p.returncode, f"cpu worker ({n})") for p in procs]
    cpu_results.append({"backend": "cpu", "device_count": n, "parallel_kind": "process_group",
                        "total_batch": TOTAL_BATCH, "local_batch": TOTAL_BATCH // n,
                        "best_seconds": max(w["best_seconds"] for w in workers),
                        "mean_seconds": max(w["mean_seconds"] for w in workers), "worker_results": workers})
    report(f"{n} process(es)", cpu_results[-1])

# --- CPU host pmap: one process with n forced host devices ---------------------------
print("CPU host-pmap sweep...")
pmap_results = []
for n in CPU_DEVICE_COUNTS:
    env = child_env(f"--xla_force_host_platform_device_count={n} {SINGLE_THREAD_XLA}")
    done = subprocess.run([sys.executable, "-c", WORKER_CODE, request("cpu", n, TOTAL_BATCH)], cwd=REPO_ROOT,
                          env={**env, "DRBX_HOST_DEVICE_COUNT": str(n)}, capture_output=True, text=True, timeout=300)
    pmap_results.append({**last_json(done.stdout, done.stderr, done.returncode, f"host pmap ({n})"),
                         "parallel_kind": "host_pmap"})
    report(f"{n} host device(s)", pmap_results[-1])

# --- optional remote GPU sweep -------------------------------------------------------
gpu_results = []
if RUN_REMOTE_GPU:
    remote = f"/tmp/drbx_strong_scaling_{int(time.time())}"
    print(f"staging src to {REMOTE_HOST}:{remote} for the GPU sweep...")
    try:
        stage = subprocess.run(["bash", "-c", f"tar czf - src | ssh -o BatchMode=yes {shlex.quote(REMOTE_HOST)} "
                                f"{shlex.quote(f'mkdir -p {remote} && tar xzf - -C {remote}')}"],
                               cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        last_json("{}", stage.stderr, stage.returncode, "remote staging")
        for n in GPU_DEVICE_COUNTS:
            cmd = (f"cd {remote} && CUDA_VISIBLE_DEVICES={','.join(map(str, range(n)))} PYTHONPATH={remote}/src "
                   f"python3 -c {shlex.quote(WORKER_CODE)} {shlex.quote(request('gpu', n, TOTAL_BATCH))}")
            done = subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE_HOST, cmd],
                                  capture_output=True, text=True, timeout=300)
            gpu_results.append(last_json(done.stdout, done.stderr, done.returncode, f"gpu worker ({n})"))
            report(f"{n} GPU(s)", gpu_results[-1])
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE_HOST, f"rm -rf {remote}"], capture_output=True, timeout=120)
else:
    print("remote GPU sweep skipped (set RUN_REMOTE_GPU = True to enable it)")


def points(results, backend):
    return compute_strong_scaling_points([(int(r["device_count"]), float(r["best_seconds"])) for r in results],
                                         backend=backend)


cpu_points, pmap_points, gpu_points = points(cpu_results, "cpu"), points(pmap_results, "cpu_host_pmap"), points(gpu_results, "gpu")
analysis_path = OUTPUT_ROOT / "data" / "strong_scaling_diffusion_analysis.json"
if not gpu_results and analysis_path.exists():  # keep previously measured GPU points
    old = json.loads(analysis_path.read_text(encoding="utf-8"))
    gpu_points = [StrongScalingPoint(**e) for e in old.get("gpu", [])]
    gpu_results = list(old.get("raw_results", {}).get("gpu", []))
    if gpu_points:
        print(f"kept {len(gpu_points)} previously measured GPU point(s) from {analysis_path}")

# --- write the analysis JSON and the plot --------------------------------------------
analysis_path.parent.mkdir(parents=True, exist_ok=True)
analysis_path.write_text(json.dumps({
    "settings": {"cpu_device_counts": list(CPU_DEVICE_COUNTS), "gpu_device_counts": list(GPU_DEVICE_COUNTS),
                 "total_batch": TOTAL_BATCH, "nx": NX, "ny": NY, "timestep": TIMESTEP, "steps": STEPS,
                 "repeats": REPEATS},
    "cpu": [p.__dict__ for p in cpu_points], "cpu_host_pmap": [p.__dict__ for p in pmap_points],
    "gpu": [p.__dict__ for p in gpu_points],
    "raw_results": {"cpu": cpu_results, "cpu_host_pmap": pmap_results, "gpu": gpu_results},
}, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote analysis JSON: {analysis_path}")

figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
for label, pts, color in (("CPU (local process group)", cpu_points, "#1d3557"),
                          ("CPU (host device pmap)", pmap_points, "#2a9d8f"),
                          ("GPU (device pmap)", gpu_points, "#d62828")):
    if not pts:
        continue
    devices = [p.device_count for p in pts]
    axes[0].plot(devices, [p.elapsed_seconds for p in pts], marker="o", linewidth=2.6, color=color, label=label)
    axes[1].plot(devices, [p.speedup for p in pts], marker="o", linewidth=2.6, color=color, label=f"{label} speedup")
    axes[1].plot(devices, [p.efficiency for p in pts], marker="s", linewidth=2.0, linestyle="--", color=color,
                 alpha=0.75, label=f"{label} efficiency")
ideal = sorted(set(CPU_DEVICE_COUNTS) | (set(GPU_DEVICE_COUNTS) if RUN_REMOTE_GPU else set()))
axes[1].plot(ideal, ideal, color="#555555", linestyle=":", linewidth=1.5, label="ideal speedup")
for axis, ylabel, title in ((axes[0], "best elapsed time [s]", "Fixed-workload strong scaling"),
                            (axes[1], "speedup / efficiency", "Scaling mode comparison")):
    axis.set(xlabel="device count", ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
axes[0].legend(frameon=False)
axes[1].legend(frameon=False, ncol=2)
plot_path = OUTPUT_ROOT / "images" / "strong_scaling_diffusion.png"
plot_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(plot_path, dpi=220)
plt.close(figure)
print(f"wrote scaling plot: {plot_path}")
print(f"total wall time: {time.perf_counter() - wall_start:.1f} s")
