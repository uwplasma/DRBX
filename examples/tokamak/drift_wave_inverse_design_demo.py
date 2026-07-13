"""Differentiable inverse design through drift-wave turbulence.

Because the Hasegawa-Wakatani model in ``jax_drb.native.hasegawa_wakatani`` is
written entirely in JAX, the gradient of any diagnostic of a turbulence run with
respect to any model parameter is available by automatic differentiation --
through the whole nonlinear time evolution. This example demonstrates two
things no non-differentiable edge code can do directly:

1. transport sensitivities: ``d(final fluctuation energy)/d(gradient drive)``
   and ``.../d(adiabaticity)`` by autodiff;
2. inverse design: gradient descent on the gradient-drive ``kappa`` to hit a
   target fluctuation-energy level.

    PYTHONPATH=src python examples/tokamak/drift_wave_inverse_design_demo.py

writes ``output/drift_wave_inverse_design/`` with a convergence PNG and JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jax_drb.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    potential_from_vorticity,
)

N = 32
LENGTH = 2.0 * np.pi * 5.0
ADIABATICITY = 1.0
HYPERVISCOSITY = 3.0e-2
DT = 5.0e-3
STEPS = 240
TARGET_GRADIENT = 1.3        # the "unknown" drive to recover
START_GRADIENT = 0.6
LEARNING_RATE = 0.15
ITERATIONS = 40
SEED = 3
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "drift_wave_inverse_design"


def _initial_state(grid):
    rng = np.random.default_rng(SEED)
    field = 1.0e-2 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    field[0, 0] = 0.0
    return jnp.array(field), jnp.array(field * 0.7)


def main() -> None:
    grid = hw_grid(N, LENGTH)
    zeta0, density0 = _initial_state(grid)

    def final_energy(kappa):
        params = HasegawaWakataniParameters(
            adiabaticity=ADIABATICITY, gradient=kappa, hyperviscosity=HYPERVISCOSITY
        )
        zeta, density = hw_run(zeta0, density0, grid, params, dt=DT, steps=STEPS)
        phi = potential_from_vorticity(zeta, grid)
        return jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)

    target = float(final_energy(TARGET_GRADIENT))

    # Transport sensitivities by autodiff, checked against finite differences.
    d_energy_d_kappa = float(jax.grad(final_energy)(TARGET_GRADIENT))
    fd = (float(final_energy(TARGET_GRADIENT + 1e-3)) - float(final_energy(TARGET_GRADIENT - 1e-3))) / 2e-3
    print(f"d(energy)/d(kappa) at kappa={TARGET_GRADIENT}: autodiff={d_energy_d_kappa:.4e} FD={fd:.4e}")

    def loss(kappa):
        return (jnp.log(final_energy(kappa)) - jnp.log(target)) ** 2

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    kappa = START_GRADIENT
    history = []
    for iteration in range(ITERATIONS):
        loss_value, grad = value_and_grad(kappa)
        kappa = float(kappa - LEARNING_RATE * grad)
        kappa = float(np.clip(kappa, 0.05, 3.0))
        history.append({"iteration": iteration, "kappa": kappa, "loss": float(loss_value)})
    print(f"recovered kappa={kappa:.4f} (target {TARGET_GRADIENT}); "
          f"loss {history[0]['loss']:.3e} -> {history[-1]['loss']:.3e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "drift_wave_inverse_design.json").write_text(json.dumps({
        "target_gradient": TARGET_GRADIENT,
        "recovered_gradient": kappa,
        "d_energy_d_kappa_autodiff": d_energy_d_kappa,
        "d_energy_d_kappa_fd": fd,
        "history": history,
    }, indent=2))

    iters = [h["iteration"] for h in history]
    fig, (ax_loss, ax_kappa) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax_loss.semilogy(iters, [h["loss"] for h in history], "o-", ms=3)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("inverse design through turbulence")
    ax_loss.grid(alpha=0.3)
    ax_kappa.plot(iters, [h["kappa"] for h in history], "o-", ms=3)
    ax_kappa.axhline(TARGET_GRADIENT, color="k", ls="--", label="target")
    ax_kappa.set_xlabel("iteration")
    ax_kappa.set_ylabel(r"gradient drive $\kappa$")
    ax_kappa.set_title("parameter convergence")
    ax_kappa.legend()
    ax_kappa.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "drift_wave_inverse_design.png", dpi=180)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'drift_wave_inverse_design.png'}")


if __name__ == "__main__":
    main()
