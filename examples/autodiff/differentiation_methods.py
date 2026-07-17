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
``tests/test_autodiff_methods.py``). The script prints the compile + timing
progress per method, the gradient values and their spread, and the fastest
method, then writes a comparison figure + JSON.

Run:

    PYTHONPATH=src python examples/autodiff/differentiation_methods.py

writes ``output/differentiation_methods/`` with a bar-chart PNG and a JSON
summary.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dkx.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_step,
)

# ----------------------------------------------------------------------------
# PARAMETERS -- everything you might want to change, in one place.
# ----------------------------------------------------------------------------
N = 64                       # spectral grid points per side (the state is two N x N complex fields)
LENGTH = 2.0 * np.pi * 8.0   # periodic box size (units of the drift scale rho_s)
STEPS = 200                  # RK4 steps in the rollout being differentiated
DT = 5.0e-3                  # timestep (normalized drift time)
ADIABATICITY = 1.0           # HW adiabaticity C (coupling of density and potential)
KAPPA = 1.0                  # background-gradient drive; d(objective)/d(kappa) is what we compare
HYPERVISCOSITY = 1.0e-3      # small-scale dissipation
SEED = 0                     # RNG seed for the small random initial perturbation
INIT_AMPLITUDE = 1.0e-2      # amplitude of the random initial vorticity/density
REPEATS = 5                  # timed calls per method (after one untimed compile call)
OUTPUT_DIR = Path("output/differentiation_methods")  # PNG + JSON summary land here

# ----------------------------------------------------------------------------
# Simulation setup: spectral grid and the random initial state.
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRID = hw_grid(N, LENGTH)
rng = np.random.default_rng(SEED)
Z0 = jnp.fft.fft2(jnp.asarray(INIT_AMPLITUDE * rng.standard_normal((N, N))))  # vorticity
M0 = jnp.fft.fft2(jnp.asarray(INIT_AMPLITUDE * rng.standard_normal((N, N))))  # density


def evolved_energy(kappa, *, checkpoint=False):
    """Objective: total spectral energy after STEPS RK4 steps, as a function of kappa.

    The parameter dataclass is built inside so kappa stays a traced input; set
    ``checkpoint=True`` to wrap each step in ``jax.checkpoint`` (recompute
    instead of store during the backward sweep).
    """

    params = HasegawaWakataniParameters(
        adiabaticity=ADIABATICITY, gradient=kappa, hyperviscosity=HYPERVISCOSITY
    )

    def step(carry, _):
        z, m = carry
        return hw_step(z, m, GRID, params, DT), None

    body = jax.checkpoint(step) if checkpoint else step
    (zf, mf), _ = jax.lax.scan(body, (Z0, M0), None, length=STEPS)
    return jnp.real(jnp.sum(jnp.abs(zf) ** 2 + jnp.abs(mf) ** 2)) / (N**4)


# The four transformations under test: the plain forward evaluation (baseline
# cost of the rollout) and the three ways of getting d(objective)/d(kappa).
METHODS = {
    "forward eval": jax.jit(evolved_energy),
    "reverse (grad)": jax.jit(jax.grad(evolved_energy)),
    "reverse + checkpoint": jax.jit(jax.grad(partial(evolved_energy, checkpoint=True))),
    "forward (jacfwd)": jax.jit(jax.jacfwd(evolved_energy)),
}

# ----------------------------------------------------------------------------
# Timing loop: compile each method once (untimed), then average REPEATS calls.
# ----------------------------------------------------------------------------
print("== Differentiation methods on a Hasegawa-Wakatani rollout ==")
print(f"   grid {N} x {N}, {STEPS} RK4 steps, dt = {DT}, kappa = {KAPPA} "
      f"(one scalar parameter)")
print(f"   timing: 1 compile call + {REPEATS} timed calls per method")

kappa = jnp.asarray(KAPPA)
results = {}
for name, fn in METHODS.items():
    print(f"   compiling {name} ...", flush=True)
    out = fn(kappa)
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(REPEATS):
        out = fn(kappa)
    jax.block_until_ready(out)
    seconds = (time.perf_counter() - start) / REPEATS
    results[name] = {"seconds": seconds, "value": float(out)}
    print(f"   {name:22s}  {seconds * 1e3:9.2f} ms / call   value = {float(out):.10e}")

# ----------------------------------------------------------------------------
# Agreement check and the "which method should I use" verdict.
# ----------------------------------------------------------------------------
gradient_names = ("reverse (grad)", "reverse + checkpoint", "forward (jacfwd)")
gradients = [results[k]["value"] for k in gradient_names]
spread = max(gradients) - min(gradients)
fastest = min(gradient_names, key=lambda k: results[k]["seconds"])
print("== Verdict ==")
print(f"   gradient agreement spread: {spread:.2e} "
      "(all three methods compute the same derivative)")
print(f"   fastest gradient method for 1 parameter: {fastest} "
      f"({results[fastest]['seconds'] * 1e3:.1f} ms; "
      f"forward eval alone costs {results['forward eval']['seconds'] * 1e3:.1f} ms)")

(OUTPUT_DIR / "summary.json").write_text(json.dumps(
    {"n": N, "steps": STEPS, "results": results, "gradient_spread": spread, "fastest": fastest},
    indent=2))
print(f"wrote {OUTPUT_DIR / 'summary.json'}")

# ----------------------------------------------------------------------------
# Figure: wall-clock per call for the four transformations.
# ----------------------------------------------------------------------------
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
print(f"wrote {OUTPUT_DIR / 'differentiation_methods.png'}")
