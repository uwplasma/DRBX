"""Performance and differentiability benchmark of the JAX turbulence core.

Two measurements that quantify the paper's "fast and differentiable" claim, on
the closed-field-line Hasegawa-Wakatani drift-wave model:

1. Throughput: jit-compiled wall-clock per RK4 step versus grid size, in
   million-cell-updates per second. Every step is an FFT-spectral RHS with a
   dealiased Poisson bracket.
2. Differentiation overhead: the wall-clock ratio of one reverse-mode gradient
   of a diagnostic of the *evolved* turbulence (with respect to the
   transport-drive parameter, through the full multi-step rollout) to one
   forward evaluation. Reverse-mode autodiff through the whole time integration
   costs only a small constant factor.

Run:

    PYTHONPATH=src python examples/benchmarks/performance_benchmark.py

prints per-size throughput and the gradient/forward ratio, then writes
``output/performance/`` (relative to the current working directory) with a
two-panel ``performance.png`` and a ``summary.json``. All timings are
single-CPU, float64; absolute numbers depend on the host, the scalings do not.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from dkx.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_step,
)

# --- PARAMETERS -----------------------------------------------------------------
LENGTH = 2.0 * np.pi * 8.0      # periodic box length
DT = 5.0e-3                     # RK4 time step
ALPHA = 1.0                     # Hasegawa-Wakatani adiabaticity
KAPPA0 = 1.0                    # background density-gradient drive
NU = 1.0e-3                     # hyperviscosity
THROUGHPUT_SIZES = (32, 48, 64, 96, 128, 192)  # grid sizes n for the n x n throughput sweep
THROUGHPUT_STEPS = 100          # RK4 steps per timed run
THROUGHPUT_REPEATS = 5          # timed repetitions per size (mean is reported)
OVERHEAD_N = 64                 # grid size of the differentiation-overhead case
OVERHEAD_STEPS = 200            # rollout steps the gradient differentiates through
OVERHEAD_REPEATS = 5            # timed repetitions of forward and gradient
OUTPUT_DIR = Path("output/performance")   # artifact directory (cwd-relative)


def _seed_state(n, seed=0):
    rng = np.random.default_rng(seed)
    zeta = 1.0e-2 * rng.standard_normal((n, n))
    dens = 1.0e-2 * rng.standard_normal((n, n))
    return jnp.fft.fft2(jnp.asarray(zeta)), jnp.fft.fft2(jnp.asarray(dens))


def _timed(fn, *args, repeats):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeats, out


def throughput_sweep(sizes, steps=THROUGHPUT_STEPS, repeats=THROUGHPUT_REPEATS):
    params = HasegawaWakataniParameters(adiabaticity=ALPHA, gradient=KAPPA0, hyperviscosity=NU)
    rows = []
    for n in sizes:
        grid = hw_grid(n, LENGTH)

        @jax.jit
        def run(z0, m0, grid=grid):
            def body(carry, _):
                z, m = carry
                return hw_step(z, m, grid, params, DT), None

            (zf, mf), _ = jax.lax.scan(body, (z0, m0), None, length=steps)
            return zf, mf

        z0, m0 = _seed_state(n)
        per_run, _ = _timed(run, z0, m0, repeats=repeats)
        per_step = per_run / steps
        mcups = (n * n) / per_step / 1.0e6  # million cell-updates / second
        rows.append({"n": int(n), "ms_per_step": per_step * 1e3, "mcups": mcups})
        print(f"n={n:4d}: {per_step*1e3:8.3f} ms/step   {mcups:8.1f} Mcell-updates/s")
    return rows


def differentiation_overhead(n=OVERHEAD_N, steps=OVERHEAD_STEPS, repeats=OVERHEAD_REPEATS):
    grid = hw_grid(n, LENGTH)
    z0, m0 = _seed_state(n, seed=1)

    def evolved_energy(kappa):
        params = HasegawaWakataniParameters(adiabaticity=ALPHA, gradient=kappa, hyperviscosity=NU)

        def body(carry, _):
            z, m = carry
            return hw_step(z, m, grid, params, DT), None

        (zf, mf), _ = jax.lax.scan(body, (z0, m0), None, length=steps)
        return jnp.real(jnp.sum(jnp.abs(zf) ** 2 + jnp.abs(mf) ** 2)) / (n**4)

    forward = jax.jit(evolved_energy)
    gradient = jax.jit(jax.grad(evolved_energy))
    kappa = jnp.asarray(KAPPA0)

    fwd_time, value = _timed(forward, kappa, repeats=repeats)
    grad_time, grad_value = _timed(gradient, kappa, repeats=repeats)
    ratio = grad_time / fwd_time
    print(f"forward {fwd_time*1e3:.2f} ms | grad {grad_time*1e3:.2f} ms | ratio {ratio:.2f}x "
          f"(steps={steps}, n={n})")
    return {
        "n": n, "steps": steps,
        "forward_ms": fwd_time * 1e3, "grad_ms": grad_time * 1e3, "grad_over_forward": ratio,
        "objective": float(value), "d_objective_d_kappa": float(grad_value),
    }


def plot(sweep, overhead, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ns = np.array([r["n"] for r in sweep])
    mcups = np.array([r["mcups"] for r in sweep])
    axes[0].plot(ns, mcups, "o-", color="#1f77b4")
    axes[0].set_xlabel("grid size n (n x n)")
    axes[0].set_ylabel("million cell-updates / s")
    axes[0].set_title("Hasegawa-Wakatani throughput (jit, 1 CPU, f64)")
    axes[0].grid(True, ls=":", alpha=0.5)

    axes[1].bar(["forward", "gradient\n(reverse-mode)"], [overhead["forward_ms"], overhead["grad_ms"]],
                color=["#1f77b4", "#d62728"])
    axes[1].set_ylabel("wall-clock (ms)")
    axes[1].set_title(f"Differentiating through {overhead['steps']} turbulence steps "
                      f"(n={overhead['n']}): {overhead['grad_over_forward']:.2f}x forward")
    axes[1].grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle("dkx performance: fast turbulence, cheap gradients through the rollout")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# --- run the two measurements and save the artifacts ------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"throughput sweep over grid sizes {THROUGHPUT_SIZES} "
      f"({THROUGHPUT_STEPS} steps per run, {THROUGHPUT_REPEATS} repeats)...")
sweep = throughput_sweep(THROUGHPUT_SIZES)
print(f"differentiation overhead at n={OVERHEAD_N} through {OVERHEAD_STEPS} steps...")
overhead = differentiation_overhead()
summary = {"throughput": sweep, "differentiation": overhead}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
plot(sweep, overhead, OUTPUT_DIR / "performance.png")
print(f"wrote {OUTPUT_DIR / 'performance.png'} and summary.json")
