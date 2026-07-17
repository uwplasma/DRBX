"""Gradient-based detachment control through the differentiable SOL solve.

Divertor detachment sets the exhaust survival of a reactor: the plasma must be
kept at the edge of detachment — a cold (~1 eV), recombining target — without
collapsing further (Dudson et al., PPCF 61, 065008, 2019; Body et al., NME 41,
101819, 2024). Here the control problem is solved with a gradient: find the
upstream density ``n_up`` that places the target electron temperature exactly
at the 1 eV detachment threshold.

Because the whole detaching-SOL model is differentiable, the sensitivity
``d Te_target / d n_up`` comes from **forward-mode autodiff through the entire
stiff, 20,000-step operator-split solve** (implicit conduction, self-limiting
radiation, recycling neutrals) — a single scalar parameter, so forward mode is
the efficient method. The controller is a step-clipped Newton iteration on
``ln Te`` (the temperature falls over a sharp cliff at detachment onset, so raw
Newton steps are clipped to a trust region that contracts whenever the residual
changes sign).

The script prints each controller iterate (n_up, Te_target, the autodiff
sensitivity, the trust region), then traces the Te_target(n_up) context curve
for the figure.

Run:

    PYTHONPATH=src python examples/autodiff/detachment_control.py

writes ``output/detachment_control/detachment_control.png`` and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dkx.native.neutrals import (  # noqa: E402
    DetachmentSolParameters,
    DetachmentSolState,
    PlasmaNormalization,
    detachment_diagnostics,
    detachment_sol_run,
)

# ----------------------------------------------------------------------------
# PARAMETERS -- everything you might want to change, in one place.
# The SOL model matches examples/benchmarks/b6_detachment_rollover.py (hermes-3
# normalized: Tnorm = 50 eV, Nnorm = 1e19 m^-3; time in transit times L / c_s),
# on a coarser grid / shorter relaxation so every controller iterate is cheap.
# ----------------------------------------------------------------------------
NZ = 96                       # parallel cells, midplane (z=0) to target (z=L)
PARALLEL_LENGTH = 30.0        # connection length L in metres
UPSTREAM_POWER = 6.0          # fixed normalized upstream power source
RECYCLING_FRACTION = 0.95     # fraction of the target ion flux recycled as neutrals
NEUTRAL_SEED = 0.05           # initial trace neutral density (normalized)
INITIAL_TEMPERATURE = 0.6     # initial normalized temperature (30 eV) -> pressure = 2 n T
NORMALIZATION = PlasmaNormalization(Tnorm=50.0)  # SOL reference normalization
STEPS = 20000                 # operator-split steps per objective evaluation
DT = 0.25 * (1.0 / NZ) / 3.0  # timestep: CFL 0.25 against the fastest wave (~3 c_s)
TARGET_EV = 1.0               # the detachment threshold Te_target to sit on [eV]
INITIAL_DENSITY = 3.0         # controller start: attached (hot-target) upstream density
MAX_STEP = 0.5                # trust region for the Newton update in n_up
ITERATIONS = 14               # maximum controller iterations
TOLERANCE_EV = 0.05           # stop when |Te_target - TARGET_EV| falls below this
SCAN_DENSITIES = np.linspace(2.5, 6.5, 9)  # context curve Te_target(n_up) for the figure
OUTPUT_DIR = Path("output/detachment_control")  # PNG + JSON summary land here

# ----------------------------------------------------------------------------
# The differentiable objective: upstream density -> relaxed target temperature.
# The parameter dataclass and cold-start state are built inside the function so
# that jax.jacfwd can push the n_up tangent through the *entire* solve.
# ----------------------------------------------------------------------------


def target_temperature(upstream_density):
    """Relax the detaching SOL at ``upstream_density``; return Te_target in eV."""

    params = DetachmentSolParameters(
        parallel_length=PARALLEL_LENGTH,
        upstream_density=upstream_density,   # Dirichlet-pinned at z = 0 every step
        upstream_power=UPSTREAM_POWER,
        recycling_fraction=RECYCLING_FRACTION,
        normalization=NORMALIZATION,
    )
    state = DetachmentSolState(
        ion_density=jnp.full(NZ, upstream_density),                              # flat cold start
        ion_momentum=jnp.zeros(NZ),                                              # plasma at rest
        plasma_pressure=jnp.full(NZ, 2.0 * upstream_density * INITIAL_TEMPERATURE),  # P = 2 n T
        neutral_density=jnp.full(NZ, NEUTRAL_SEED),                              # trace neutrals
    )
    state = detachment_sol_run(state, params, dt=DT, steps=STEPS)
    return detachment_diagnostics(state, params).target_temperature_ev


# One jitted call returns Te_target and d Te_target / d n_up: forward mode
# (jacfwd) is the efficient choice for a single scalar control parameter.
value_and_derivative = jax.jit(
    lambda n: (target_temperature(n), jax.jacfwd(target_temperature)(n))
)

# ----------------------------------------------------------------------------
# Controller loop: trust-region Newton on ln Te. The Newton step
# -(ln Te - ln target) / (d ln Te / d n) is clipped to a trust region that
# contracts whenever the residual changes sign -- the detachment cliff is
# steeper than any local derivative predicts, so an uncontracted step bounces
# across it forever, while the shrinking trust region homes in on the threshold.
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("== Gradient-based detachment control: setup ==")
print(f"   objective: place Te_target at {TARGET_EV} eV by adjusting n_up "
      f"(start {INITIAL_DENSITY}, tolerance {TOLERANCE_EV} eV)")
print(f"   each iterate: {STEPS}-step stiff solve on {NZ} cells + forward-mode "
      f"autodiff sensitivity (first call compiles)")

density = float(INITIAL_DENSITY)
trust = MAX_STEP
previous_sign = None
history = []
for iteration in range(ITERATIONS):
    temperature, derivative = value_and_derivative(jnp.asarray(density))
    temperature, derivative = float(temperature), float(derivative)
    history.append({"iteration": iteration, "n_up": density, "Te_target_ev": temperature})
    print(f"   iter {iteration:2d}: n_up = {density:.4f},  Te_target = {temperature:.4f} eV,  "
          f"dTe/dn_up = {derivative:+.4f} eV (autodiff),  trust = {trust:.3f}")
    residual = np.log(temperature) - np.log(TARGET_EV)
    if abs(temperature - TARGET_EV) < TOLERANCE_EV:
        print(f"   converged: |Te_target - {TARGET_EV} eV| < {TOLERANCE_EV} eV")
        break
    sign = np.sign(residual)
    if previous_sign is not None and sign != previous_sign:
        trust *= 0.5  # crossed the threshold: contract the trust region
    previous_sign = sign
    log_derivative = derivative / temperature
    step = -residual / log_derivative if log_derivative != 0.0 else -sign * trust
    density = max(density + float(np.clip(step, -trust, trust)), 0.5)

print("== Controller result ==")
print(f"   final: n_up = {history[-1]['n_up']:.4f}, "
      f"Te_target = {history[-1]['Te_target_ev']:.4f} eV "
      f"({len(history)} objective evaluations)")

# ----------------------------------------------------------------------------
# Context curve Te_target(n_up) for the figure: the detachment cliff the
# controller walked down.
# ----------------------------------------------------------------------------
print("== Tracing the Te_target(n_up) context curve ==")
curve = []
for n_scan in SCAN_DENSITIES:
    curve.append(float(target_temperature(jnp.asarray(n_scan))))
    print(f"   n_up = {n_scan:.2f}:  Te_target = {curve[-1]:7.4f} eV")

(OUTPUT_DIR / "summary.json").write_text(json.dumps({
    "target_ev": TARGET_EV,
    "history": history,
    "final_n_up": history[-1]["n_up"],
    "final_Te_target_ev": history[-1]["Te_target_ev"],
}, indent=2))
print(f"wrote {OUTPUT_DIR / 'summary.json'}")

# ----------------------------------------------------------------------------
# Figure: controller iterates on the detachment cliff + convergence history.
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
ax = axes[0]
ax.semilogy(SCAN_DENSITIES, curve, "-", color="#9ecae1", lw=2, label="Te_target(n_up)")
trajectory_n = [h["n_up"] for h in history]
trajectory_t = [h["Te_target_ev"] for h in history]
ax.semilogy(trajectory_n, trajectory_t, "o-", color="#d62728", label="controller iterates")
ax.annotate("start", (trajectory_n[0], trajectory_t[0]), textcoords="offset points",
            xytext=(6, 6), fontsize=9)
ax.axhline(TARGET_EV, color="gray", ls=":", label="1 eV detachment threshold")
ax.set_xlabel("upstream density (normalized)")
ax.set_ylabel("target temperature (eV)")
ax.set_title("Newton iterates on the detachment cliff")
ax.legend(fontsize=8), ax.grid(True, which="both", ls=":", alpha=0.4)

ax = axes[1]
ax.semilogy(range(len(history)), [abs(t - TARGET_EV) for t in trajectory_t], "o-", color="#d62728")
ax.set_xlabel("iteration")
ax.set_ylabel("|Te_target - 1 eV|")
ax.set_title("Convergence (derivative by forward-mode autodiff\n"
             f"through the {STEPS:,}-step stiff solve)")
ax.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("Gradient-based detachment control: place the target at the 1 eV threshold")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "detachment_control.png", dpi=170)
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'detachment_control.png'}")
