"""Strong-scaling demo for the sharded FCI two-field RK4 step.

For each device count the script launches the short ``WORKER_CODE`` below in a
fresh ``python -c`` subprocess (the XLA host device count must be fixed with
``--xla_force_host_platform_device_count=<n>`` before JAX starts). Each worker
advances the same shifted-torus two-field state and prints one JSON line; the
parent checks that all final-state checksums agree, then writes
``output/fci_sharded_strong_scaling/scaling_<platform>.json`` and
``scaling_<platform>.png`` (relative to the current working directory).

The default grid and sweep are a quick laptop preset, not the 36-core
measurement quoted in the docs (that used ``GRID = (256, 128, 32)`` and device
counts up to 16 with ``taskset`` core binding on Linux).

Run from the repository root:

    PYTHONPATH=src python examples/benchmarks/fci_sharded_strong_scaling.py

Host CPU note: wall times are meaningful only up to the number of physical
cores, and simultaneous heavy processes skew them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- PARAMETERS ----
GRID = (128, 64, 16)       # (nx, ny, nz) of the shifted-torus two-field state
STEPS = 10                 # timed RK4 steps per worker (one extra warmup step compiles)
DT = 1.0e-3                # RK4 time step
PLATFORM = "cpu"           # "cpu" (forced host devices + core binding), "auto"
                           # (let JAX pick, accelerators win), or "gpu"/"tpu"
DEVICE_COUNTS = (1, 2, 4)  # device counts to sweep; each needs a SHARD_LAYOUTS entry
SHARD_LAYOUTS = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (4, 2, 1), 16: (4, 4, 1), 32: (4, 4, 2)}
OUTPUT_DIR = Path("output/fci_sharded_strong_scaling")  # artifact directory (cwd-relative)
CHECKSUM_RTOL = 1.0e-10    # relative tolerance for the cross-device checksum gate

REPO_ROOT = Path(__file__).resolve().parents[2]
# The worker reuses the test-suite case builder (tests/fci_sharded_2field_case.py).
WORKER_CODE = """
import json, sys, time
import jax, numpy as np
from drbx.native import Fci2FieldRhsParameters, make_sharded_2field_step
from tests.fci_sharded_2field_case import build_case_geometry, build_initial_state
r = json.loads(sys.argv[1]); n = r["device_count"]
assert len(jax.devices()) >= n, f"requested {n} devices, JAX sees {len(jax.devices())}"
geometry = build_case_geometry(tuple(r["grid"]))
state = build_initial_state(geometry)
step, _ = make_sharded_2field_step(geometry, tuple(r["shards"]), Fci2FieldRhsParameters(rho_star=1.0), None, dt=r["dt"])
state = step(state); jax.block_until_ready(state.density)   # warmup / compile
start = time.perf_counter()
for _ in range(r["steps"]):
    state = step(state)
jax.block_until_ready(state.density)
elapsed = time.perf_counter() - start
# gather to host before summing so every shard is included
checksum = float(np.abs(np.asarray(jax.device_get(state.density))).sum()
                 + np.abs(np.asarray(jax.device_get(state.v_parallel))).sum())
print(json.dumps({"device_count": n, "shard_counts": r["shards"], "steps": r["steps"],
                  "seconds_per_step": elapsed / r["steps"], "checksum": checksum}))
"""


def launch_worker(n):
    env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"}
    env.pop("DRBX_HOST_DEVICE_COUNT", None)
    request = json.dumps({"device_count": n, "shards": SHARD_LAYOUTS[n], "grid": GRID, "steps": STEPS, "dt": DT})
    command = [sys.executable, "-c", WORKER_CODE, request]
    if PLATFORM == "cpu":
        # A single-device CPU program already threads across all cores, so bind
        # one core per device where taskset exists (Linux); on macOS the curve
        # mostly reflects intra-op threading.
        env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={n}".strip()
        env["JAX_PLATFORMS"] = "cpu"
        if shutil.which("taskset"):
            command = ["taskset", "-c", f"0-{n - 1}", *command]
    elif PLATFORM == "auto":
        env.pop("JAX_PLATFORMS", None)
    else:
        env["JAX_PLATFORMS"] = PLATFORM
    done = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        raise RuntimeError(f"worker for {n} devices failed:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
    return json.loads(lines[-1])


# --- sweep device counts --------------------------------------------------------------
wall_start = time.perf_counter()
print(f"FCI two-field sharded RK4 strong scaling: grid {GRID}, {STEPS} steps, dt {DT}, platform {PLATFORM}")
results = []
for n in DEVICE_COUNTS:
    layout = SHARD_LAYOUTS.get(n)
    if layout is None or any(size % count for size, count in zip(GRID, layout)):
        print(f"skipping {n} devices: no layout, or grid {GRID} not divisible by layout {layout}")
        continue
    print(f"running worker with {n} device(s), layout {layout}...", flush=True)
    t0 = time.perf_counter()
    results.append(launch_worker(n))
    print(f"  seconds_per_step={results[-1]['seconds_per_step']:.4f} checksum={results[-1]['checksum']:.12e} "
          f"(worker wall {time.perf_counter() - t0:.1f} s)", flush=True)
if not results:
    raise RuntimeError("no worker produced a result")

reference = float(results[0]["checksum"])
for entry in results:
    if abs(float(entry["checksum"]) - reference) > CHECKSUM_RTOL * max(1.0, abs(reference)):
        raise RuntimeError(f"checksum mismatch for {entry['device_count']} devices: {entry['checksum']!r} vs {reference!r}")
print(f"all checksums agree to {CHECKSUM_RTOL:.1e} (reference {reference:.12e})")

# --- save JSON and plot ---------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
scaling_path = OUTPUT_DIR / f"scaling_{PLATFORM}.json"
scaling_path.write_text(json.dumps({"grid": list(GRID), "steps": STEPS, "dt": DT, "results": results}, indent=2) + "\n")
devices = [int(e["device_count"]) for e in results]
seconds = [float(e["seconds_per_step"]) for e in results]
efficiency = seconds[0] * devices[0] / (devices[-1] * seconds[-1])
plt.figure(figsize=(6.0, 4.5))
plt.loglog(devices, seconds, "o-", label="measured")
plt.loglog(devices, [seconds[0] * devices[0] / d for d in devices], "k--", label="ideal scaling")
plt.xlabel(f"{PLATFORM.upper()} devices")
plt.ylabel("Wall time per RK4 step [s]")
plt.title(f"FCI two-field sharded step, grid {GRID[0]}x{GRID[1]}x{GRID[2]}")
plt.grid(True, which="both", alpha=0.4)
plt.legend()
plt.annotate(f"parallel efficiency at {devices[-1]} devices: {efficiency:.1%}", xy=(devices[-1], seconds[-1]),
             xytext=(0.03, 0.06), textcoords="axes fraction")
plt.tight_layout()
plot_path = OUTPUT_DIR / f"scaling_{PLATFORM}.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"wrote {scaling_path}")
print(f"wrote {plot_path}")
print(f"parallel efficiency at {devices[-1]} devices: {efficiency:.1%}; total wall {time.perf_counter() - wall_start:.1f} s")
