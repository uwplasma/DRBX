"""Choosing the most efficient differentiation method for turbulence rollouts.

One objective -- a diagnostic of the evolved Hasegawa-Wakatani turbulence after
``STEPS`` RK4 steps, differentiated with respect to the transport-drive
parameter ``kappa`` -- computed three ways:

1. **Reverse mode** (``jax.grad``): one backward sweep, but the scan stores every
   step's state for the sweep, so memory grows with the rollout length. The
   right choice when differentiating with respect to *many* parameters (fields,
   geometry) at once.
2. **Checkpointed reverse** (``jax.grad`` + ``jax.checkpoint`` on the step):
   step internals are recomputed during the backward sweep instead of stored --
   bounded memory for a modest extra compute cost. The right choice when a long
   reverse-mode rollout is memory-bound.
3. **Forward mode** (``jax.jacfwd``): pushes one tangent through the rollout --
   no reverse sweep and no stored trajectory at all. The most efficient choice
   for a *small number* of scalar parameters (here: one).

All three must agree to machine precision (gated in
``tests/test_autodiff_methods.py``); this script times them and writes a
comparison figure + JSON.

Run:

    PYTHONPATH=src python examples/autodiff/differentiation_methods_demo.py
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax_drb.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_step,
)

N = 64
STEPS = 200
DT = 5.0e-3
LENGTH = 2.0 * np.pi * 8.0
REPEATS = 5
OUTPUT_DIR = Path("output/differentiation_methods")


def _seed_state(seed=0):
    rng = np.random.default_rng(seed)
    zeta = jnp.fft.fft2(jnp.asarray(1.0e-2 * rng.standard_normal((N, N))))
    dens = jnp.fft.fft2(jnp.asarray(1.0e-2 * rng.standard_normal((N, N))))
    return zeta, dens


GRID = hw_grid(N, LENGTH)
Z0, M0 = _seed_state()


def evolved_energy(kappa, *, checkpoint=False):
    params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=kappa, hyperviscosity=1.0e-3)

    def step(carry, _):
        z, m = carry
        return hw_step(z, m, GRID, params, DT), None

    body = jax.checkpoint(step) if checkpoint else step
    (zf, mf), _ = jax.lax.scan(body, (Z0, M0), None, length=STEPS)
    return jnp.real(jnp.sum(jnp.abs(zf) ** 2 + jnp.abs(mf) ** 2)) / (N**4)


METHODS = {
    "forward eval": jax.jit(evolved_energy),
    "reverse (grad)": jax.jit(jax.grad(evolved_energy)),
    "reverse + checkpoint": jax.jit(jax.grad(partial(evolved_energy, checkpoint=True))),
    "forward (jacfwd)": jax.jit(jax.jacfwd(evolved_energy)),
}


def _timed(fn, value):
    out = fn(value)
    jax.block_until_ready(out)  # compile
    start = time.perf_counter()
    for _ in range(REPEATS):
        out = fn(value)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / REPEATS, float(out)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kappa = jnp.asarray(1.0)
    results = {}
    for name, fn in METHODS.items():
        seconds, value = _timed(fn, kappa)
        results[name] = {"seconds": seconds, "value": value}
        print(f"{name:22s}  {seconds*1e3:9.2f} ms   value={value:.10e}")

    gradients = [results[k]["value"] for k in ("reverse (grad)", "reverse + checkpoint", "forward (jacfwd)")]
    spread = max(gradients) - min(gradients)
    print(f"gradient agreement spread: {spread:.2e}")
    fastest = min(("reverse (grad)", "reverse + checkpoint", "forward (jacfwd)"),
                  key=lambda k: results[k]["seconds"])
    print(f"fastest gradient method for 1 parameter: {fastest}")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(
        {"n": N, "steps": STEPS, "results": results, "gradient_spread": spread, "fastest": fastest},
        indent=2))

    import matplotlib.pyplot as plt

    names = list(METHODS)
    times_ms = [results[k]["seconds"] * 1e3 for k in names]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar(names, times_ms, color=colors)
    for bar, t in zip(bars, times_ms):
        ax.annotate(f"{t:.0f} ms", (bar.get_x() + bar.get_width() / 2, t),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("wall-clock per call (ms)")
    ax.set_title(f"Differentiating {STEPS} turbulence steps w.r.t. one parameter (n={N}, CPU f64)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "differentiation_methods.png", dpi=170)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'differentiation_methods.png'} and summary.json")


if __name__ == "__main__":
    main()
