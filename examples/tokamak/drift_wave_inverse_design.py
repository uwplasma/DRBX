"""Differentiable inverse design through drift-wave turbulence.

Physics. The Hasegawa-Wakatani model in ``jax_drb.native.hasegawa_wakatani``
is written entirely in JAX, so the gradient of any diagnostic of a turbulence
run with respect to any model parameter is available by automatic
differentiation -- through the whole nonlinear time evolution. This example
demonstrates two things no non-differentiable edge code can do directly:

1. transport sensitivities: ``d(final fluctuation energy)/d(gradient drive)``
   by autodiff, cross-checked against finite differences;
2. inverse design: gradient descent on the density-gradient drive ``kappa``
   to hit a target fluctuation-energy level produced by an "unknown" drive.

It prints the autodiff/finite-difference sensitivity comparison and the
recovered ``kappa``, and writes ``output/drift_wave_inverse_design/`` with a
convergence PNG and a JSON history. Run it with:

    PYTHONPATH=src python examples/tokamak/drift_wave_inverse_design.py

Edit the PARAMETERS block below to change resolution, targets, or the
optimizer settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jax_drb.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    potential_from_vorticity,
)

# ----------------------------- PARAMETERS ----------------------------------
N = 32                       # grid points per side (small: many optimizer iterations)
LENGTH = 2.0 * np.pi * 5.0   # box side in units of rho_s
ADIABATICITY = 1.0           # alpha: parallel electron response (held fixed here)
HYPERVISCOSITY = 3.0e-2      # nu: grid-scale damping
DT = 5.0e-3                  # RK4 time step
STEPS = 240                  # rollout length differentiated through (t = STEPS * DT)
TARGET_GRADIENT = 1.3        # the "unknown" drive kappa* whose energy we try to match
START_GRADIENT = 0.6         # initial guess for kappa
LEARNING_RATE = 0.15         # gradient-descent step on kappa
ITERATIONS = 40              # optimizer iterations
SEED = 3                     # RNG seed of the initial noise
SEED_AMPLITUDE = 1.0e-2      # rms of the initial real-space noise field
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "drift_wave_inverse_design"

# ----------------------------- SIMULATION SETUP -----------------------------
grid = hw_grid(N, LENGTH)

# Hermitian (real-field) random noise seed: FFT of a real field, mean removed,
# weighted toward large scales. Identical for every optimizer evaluation.
rng = np.random.default_rng(SEED)
noise_hat = np.fft.fft2(rng.standard_normal((N, N)))
noise_hat[0, 0] = 0.0
noise_hat *= np.exp(-np.asarray(grid.k2)) * np.asarray(grid.dealias)
noise_hat *= SEED_AMPLITUDE / np.sqrt(np.mean(np.real(np.fft.ifft2(noise_hat)) ** 2))
zeta0 = jnp.array(noise_hat)
density0 = jnp.array(noise_hat * 0.7)


def final_energy(kappa):
    """Fluctuation energy after a nonlinear rollout, as a function of the drive."""

    params = HasegawaWakataniParameters(
        adiabaticity=ADIABATICITY, gradient=kappa, hyperviscosity=HYPERVISCOSITY
    )
    zeta, density = hw_run(zeta0, density0, grid, params, dt=DT, steps=STEPS)
    phi = potential_from_vorticity(zeta, grid)
    return jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)


# ----------------------------- SENSITIVITY CHECK ----------------------------
# d(energy)/d(kappa) through the whole rollout, autodiff vs finite differences.
target = float(final_energy(TARGET_GRADIENT))
d_energy_d_kappa = float(jax.grad(final_energy)(TARGET_GRADIENT))
fd = (
    float(final_energy(TARGET_GRADIENT + 1e-3)) - float(final_energy(TARGET_GRADIENT - 1e-3))
) / 2e-3
print(
    f"d(energy)/d(kappa) at kappa={TARGET_GRADIENT}: "
    f"autodiff={d_energy_d_kappa:.4e} FD={fd:.4e}"
)

# ----------------------------- OPTIMIZATION LOOP ----------------------------
def loss(kappa):
    return (jnp.log(final_energy(kappa)) - jnp.log(target)) ** 2


value_and_grad = jax.jit(jax.value_and_grad(loss))

kappa = START_GRADIENT
history = []
for iteration in range(ITERATIONS):
    loss_value, grad = value_and_grad(kappa)
    kappa = float(np.clip(kappa - LEARNING_RATE * float(grad), 0.05, 3.0))
    history.append({"iteration": iteration, "kappa": kappa, "loss": float(loss_value)})
    if iteration % 5 == 0 or iteration == ITERATIONS - 1:
        print(
            f"iteration={iteration:3d}  kappa={kappa:.4f}  "
            f"loss={float(loss_value):.3e}  grad={float(grad):+.3e}"
        )
print(
    f"recovered kappa={kappa:.4f} (target {TARGET_GRADIENT}); "
    f"loss {history[0]['loss']:.3e} -> {history[-1]['loss']:.3e}"
)

# ----------------------------- SAVE AND PLOT --------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "drift_wave_inverse_design.json").write_text(
    json.dumps(
        {
            "target_gradient": TARGET_GRADIENT,
            "recovered_gradient": kappa,
            "d_energy_d_kappa_autodiff": d_energy_d_kappa,
            "d_energy_d_kappa_fd": fd,
            "history": history,
        },
        indent=2,
    )
)
print(f"wrote {OUTPUT_DIR / 'drift_wave_inverse_design.json'}")

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
