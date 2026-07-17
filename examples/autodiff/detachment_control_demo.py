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
Newton steps are clipped to a trust region).

Run:

    PYTHONPATH=src python examples/autodiff/detachment_control_demo.py

writes ``output/detachment_control/detachment_control.png`` and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax_drb.native.neutrals import (  # noqa: E402
    DetachmentSolParameters,
    DetachmentSolState,
    detachment_diagnostics,
    detachment_sol_run,
)

NZ = 96
STEPS = 20000
TARGET_EV = 1.0          # the detachment threshold to sit on
MAX_STEP = 0.5           # trust region for the Newton update in n_up
ITERATIONS = 14
OUTPUT_DIR = Path("output/detachment_control")


def target_temperature(upstream_density):
    params = DetachmentSolParameters(upstream_density=upstream_density, upstream_power=6.0)
    state = DetachmentSolState(
        jnp.full(NZ, upstream_density),
        jnp.zeros(NZ),
        jnp.full(NZ, 2.0 * upstream_density * 0.6),
        jnp.full(NZ, 0.05),
    )
    state = detachment_sol_run(state, params, dt=0.25 * (1.0 / NZ) / 3.0, steps=STEPS)
    return detachment_diagnostics(state, params).target_temperature_ev


value_and_derivative = jax.jit(
    lambda n: (target_temperature(n), jax.jacfwd(target_temperature)(n))
)


def control(initial_density=3.0):
    """Trust-region Newton on ln Te.

    The Newton step ``-(ln Te - ln target) / (d ln Te / d n)`` is clipped to a
    trust region that **contracts whenever the residual changes sign** — the
    detachment cliff is steeper than any local derivative predicts, so an
    uncontracted step bounces across it forever, while the shrinking trust
    region homes in on the threshold.
    """

    density = float(initial_density)
    trust = MAX_STEP
    previous_sign = None
    history = []
    for iteration in range(ITERATIONS):
        temperature, derivative = value_and_derivative(jnp.asarray(density))
        temperature, derivative = float(temperature), float(derivative)
        history.append({"iteration": iteration, "n_up": density, "Te_target_ev": temperature})
        print(f"iter {iteration}: n_up={density:.4f}  Te_target={temperature:.4f} eV  "
              f"dTe/dn={derivative:+.4f}  trust={trust:.3f}")
        residual = np.log(temperature) - np.log(TARGET_EV)
        if abs(temperature - TARGET_EV) < 0.05:
            break
        sign = np.sign(residual)
        if previous_sign is not None and sign != previous_sign:
            trust *= 0.5
        previous_sign = sign
        log_derivative = derivative / temperature
        step = -residual / log_derivative if log_derivative != 0.0 else -sign * trust
        density = max(density + float(np.clip(step, -trust, trust)), 0.5)
    return history


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = control()

    # Context curve for the figure.
    scan = np.linspace(2.5, 6.5, 9)
    curve = [float(target_temperature(jnp.asarray(n))) for n in scan]

    (OUTPUT_DIR / "summary.json").write_text(json.dumps({
        "target_ev": TARGET_EV,
        "history": history,
        "final_n_up": history[-1]["n_up"],
        "final_Te_target_ev": history[-1]["Te_target_ev"],
    }, indent=2))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    ax.semilogy(scan, curve, "-", color="#9ecae1", lw=2, label="Te_target(n_up)")
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
    ax.set_title("Convergence (derivative by forward-mode autodiff\nthrough the 20,000-step stiff solve)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Gradient-based detachment control: place the target at the 1 eV threshold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "detachment_control.png", dpi=170)
    plt.close(fig)
    print(f"final: n_up={history[-1]['n_up']:.4f}, Te_target={history[-1]['Te_target_ev']:.4f} eV")
    print(f"wrote {OUTPUT_DIR / 'detachment_control.png'} and summary.json")


if __name__ == "__main__":
    main()
