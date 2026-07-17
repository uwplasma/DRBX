"""Strong-scaling demo for the sharded FCI two-field RK4 step.

The script re-invokes itself once per device count because the XLA host device
count must be fixed with ``--xla_force_host_platform_device_count=<n>`` before
JAX is imported; the single internal handshake environment variable
``DRBX_SHARDING_DEMO_WORKER`` carries the requested device count, and the
worker branch is the small guarded block right below the constants. Each worker
advances the same shifted-torus two-field state and prints one JSON line; the
parent verifies that all final-state checksums agree, then writes
``output/fci_sharded_strong_scaling/scaling_<platform>.json`` and
``scaling_<platform>.png`` (relative to the current working directory).

Edit the PARAMETERS constants below (grid, steps, platform, device counts) and
run from the repository root:

    PYTHONPATH=src python examples/benchmarks/fci_sharded_strong_scaling.py

Host CPU note: wall times are meaningful only up to the number of physical
cores, and simultaneous heavy processes skew them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- PARAMETERS -----------------------------------------------------------------
GRID = (256, 128, 16)      # (nx, ny, nz) of the shifted-torus two-field state
STEPS = 10                 # timed RK4 steps per worker (one extra warmup step compiles)
DT = 1.0e-3                # RK4 time step
PLATFORM = "cpu"           # "cpu" (forced host devices + core binding), "auto"
                           # (let JAX pick, accelerators win), or an explicit
                           # accelerator platform such as "gpu"/"tpu"
DEVICE_COUNTS = (1, 2, 4, 8)  # device counts to sweep; each needs a SHARD_LAYOUTS entry
SHARD_LAYOUTS = {
    1: (1, 1, 1),
    2: (2, 1, 1),
    4: (2, 2, 1),
    8: (4, 2, 1),
    16: (4, 4, 1),
    32: (4, 4, 2),
}
OUTPUT_DIR = Path("output/fci_sharded_strong_scaling")  # artifact directory (cwd-relative)
CHECKSUM_RTOL = 1.0e-10    # relative tolerance for the cross-device checksum gate

# Internal worker handshake (do not set by hand): the parent re-invokes this
# script with this env var holding the device count after preparing XLA_FLAGS.
WORKER_ENV_VAR = "DRBX_SHARDING_DEMO_WORKER"


def run_worker(device_count: int) -> None:
    """Advance the sharded two-field state and print one JSON result line."""

    import jax
    import jax.numpy as jnp  # noqa: F401  (kept for parity with the sharded API imports)

    from drbx.native import Fci2FieldRhsParameters, make_sharded_2field_step
    from tests.fci_sharded_2field_case import build_case_geometry, build_initial_state

    shard_counts = SHARD_LAYOUTS[device_count]
    if len(jax.devices()) < device_count:
        raise RuntimeError(
            f"worker requested {device_count} devices but JAX sees {len(jax.devices())}"
        )

    geometry = build_case_geometry(GRID)
    state = build_initial_state(geometry)
    step_fn, _info = make_sharded_2field_step(
        geometry,
        shard_counts,
        Fci2FieldRhsParameters(rho_star=1.0),
        None,
        dt=DT,
    )

    # One warmup step triggers compilation and is excluded from the timing.
    state = step_fn(state)
    jax.block_until_ready(state.density)

    start = time.perf_counter()
    for _ in range(STEPS):
        state = step_fn(state)
    jax.block_until_ready(state.density)
    elapsed = time.perf_counter() - start

    # Gather to host before summing: on real multi-device runs a jnp.sum over
    # the sharded output can reduce a single shard, so assemble the global
    # arrays first.
    density = numpy.asarray(jax.device_get(state.density))
    v_parallel = numpy.asarray(jax.device_get(state.v_parallel))
    checksum = float(numpy.abs(density).sum() + numpy.abs(v_parallel).sum())
    print(
        json.dumps(
            {
                "device_count": device_count,
                "shard_counts": list(shard_counts),
                "steps": STEPS,
                "seconds_per_step": elapsed / STEPS,
                "checksum": checksum,
            }
        )
    )


# --- worker branch: re-invoked subprocesses do one measurement and exit -----------
if os.environ.get(WORKER_ENV_VAR):
    run_worker(int(os.environ[WORKER_ENV_VAR]))
    sys.exit(0)


# --- parent-side helpers ----------------------------------------------------------
def _layout_skip_reason(device_count: int) -> str | None:
    layout = SHARD_LAYOUTS.get(device_count)
    if layout is None:
        return f"no shard layout configured for {device_count} devices"
    for axis, (size, count) in enumerate(zip(GRID, layout)):
        if size % count:
            return (
                f"grid axis {axis} with size {size} is not divisible by "
                f"shard count {count} in layout {layout}"
            )
    return None


def _launch_worker(device_count: int) -> dict[str, object]:
    env = dict(os.environ)
    command = [sys.executable, str(Path(__file__).resolve())]
    if PLATFORM == "cpu":
        # On CPU, XLA intra-op threading already parallelises a single-device
        # program across every core, so forced host devices alone measure
        # nothing (the modern thunk runtime ignores the legacy Eigen thread
        # flags). Where the OS supports it, bind the worker to one core per
        # device so the sweep isolates the domain-decomposition machinery at
        # one core per shard; without taskset (e.g. macOS) the curve mostly
        # reflects intra-op threading and is labelled accordingly.
        existing_flags = env.get("XLA_FLAGS", "")
        env["XLA_FLAGS"] = (
            f"{existing_flags} --xla_force_host_platform_device_count={device_count}"
        ).strip()
        # Pin the platform explicitly: on a host with accelerators the worker
        # would otherwise silently run on them.
        env["JAX_PLATFORMS"] = "cpu"
        if shutil.which("taskset"):
            command = ["taskset", "-c", f"0-{device_count - 1}", *command]
    elif PLATFORM == "auto":
        # Let JAX choose (accelerators win when present); pinning the platform
        # explicitly can change how XLA stages jit constants on some builds.
        env.pop("JAX_PLATFORMS", None)
    else:
        # Real accelerator devices: no forced host platform, no affinity.
        env["JAX_PLATFORMS"] = PLATFORM
    env[WORKER_ENV_VAR] = str(device_count)
    env.pop("DRBX_HOST_DEVICE_COUNT", None)

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker for {device_count} devices failed:\n{completed.stderr[-4000:]}"
        )
    json_lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not json_lines:
        raise RuntimeError(
            f"worker for {device_count} devices produced no JSON line:\n{completed.stdout[-2000:]}"
        )
    return json.loads(json_lines[-1])


def _write_scaling_plot(results: list[dict[str, object]], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devices = [int(entry["device_count"]) for entry in results]
    seconds = [float(entry["seconds_per_step"]) for entry in results]
    ideal = [seconds[0] * devices[0] / count for count in devices]
    efficiency = seconds[0] * devices[0] / (devices[-1] * seconds[-1])

    plt.figure(figsize=(6.0, 4.5))
    plt.loglog(devices, seconds, "o-", label="measured")
    plt.loglog(devices, ideal, "k--", label="ideal scaling")
    plt.xlabel(f"{PLATFORM.upper()} devices")
    plt.ylabel("Wall time per RK4 step [s]")
    plt.title(f"FCI two-field sharded step, grid {GRID[0]}x{GRID[1]}x{GRID[2]}")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.annotate(
        f"parallel efficiency at {devices[-1]} devices: {efficiency:.1%}",
        xy=(devices[-1], seconds[-1]),
        xytext=(0.03, 0.06),
        textcoords="axes fraction",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


# --- parent flow: sweep device counts, verify checksums, save artifacts -----------
results: list[dict[str, object]] = []
for device_count in DEVICE_COUNTS:
    reason = _layout_skip_reason(device_count)
    if reason is not None:
        print(f"skipping {device_count} devices: {reason}")
        continue
    print(f"running worker with {device_count} host device(s)...")
    result = _launch_worker(device_count)
    print(
        f"  devices={result['device_count']} "
        f"layout={tuple(result['shard_counts'])} "
        f"seconds_per_step={result['seconds_per_step']:.4f} "
        f"checksum={result['checksum']:.12e}"
    )
    results.append(result)

if not results:
    raise RuntimeError("no worker produced a result")

reference_checksum = float(results[0]["checksum"])
for entry in results:
    deviation = abs(float(entry["checksum"]) - reference_checksum)
    if deviation > CHECKSUM_RTOL * max(1.0, abs(reference_checksum)):
        raise RuntimeError(
            f"checksum mismatch for {entry['device_count']} devices: "
            f"{entry['checksum']!r} vs {reference_checksum!r}"
        )
print(f"all checksums agree to {CHECKSUM_RTOL:.1e} (reference {reference_checksum:.12e})")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
scaling_path = OUTPUT_DIR / f"scaling_{PLATFORM}.json"
scaling_path.write_text(
    json.dumps(
        {
            "grid": list(GRID),
            "steps": STEPS,
            "dt": DT,
            "results": results,
        },
        indent=2,
    )
    + "\n"
)
plot_path = OUTPUT_DIR / f"scaling_{PLATFORM}.png"
_write_scaling_plot(results, plot_path)
print(f"wrote {scaling_path}")
print(f"wrote {plot_path}")
