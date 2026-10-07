"""Tutorial 07 -- differentiable workflows: a gradient through a run, then inverse design.

Every DRBX model is written in JAX, so jax.grad / jax.jacfwd give exact
derivatives of any output with respect to any input.

Part A: E(kappa) is the fluctuation energy after a nonlinear Hasegawa-Wakatani
run (tutorial 02). jax.grad differentiates through all RK4 steps; we check it
against a central finite difference, then run gradient descent on
L(kappa) = (ln E(kappa) - ln E_target)^2 to recover the drive that produced a
target energy.

Part B: T_t(n_up) is the steady target temperature of the SD1D-matched leg
(tutorial 05). Differentiating the pseudo-time iterations would be expensive,
so ``detachment_target_outputs`` uses the implicit-function theorem at the
converged state (one linear solve). One Newton step on ln T_t(n_up) then moves T_t
toward a requested value.

Run from the repository root (page: docs/tutorial/07_differentiable.md):

    PYTHONPATH=src python examples/tutorial/07_differentiable.py
"""

# 1. Imports ------------------------------------------------------------------
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drbx.native.hasegawa_wakatani import (
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    potential_from_vorticity,
)
from drbx.native.neutrals import (
    DetachmentSolParameters,
    detachment_diagnostics,
    detachment_sol_run,
    detachment_target_outputs,
)

# 2. Input parameters -----------------------------------------------------------
N = 24                        # Part A grid
LENGTH = 2.0 * np.pi * 5.0
STEPS = 150                   # RK4 steps differentiated through
DT = 5.0e-3
KAPPA_TRUE = 1.3              # drive that produces the target energy
KAPPA_START = 0.6             # initial guess of the optimizer
LEARNING_RATE = 0.3
ITERATIONS = 40
NY = 200                      # Part B cells
N_UP = 2.0e19                 # Part B upstream density [m^-3]
POWER_FLUX = 5.0e7            # W/m^2
TARGET_EV = 10.0              # requested target temperature
FIGURE = Path("docs/media/tutorial_07_differentiable.png")

# 3. Run -------------------------------------------------------------------------
start = time.perf_counter()
grid = hw_grid(N, LENGTH)
rng = np.random.default_rng(3)
seed = np.fft.fft2(rng.standard_normal((N, N))) * (1.0e-2 / N)
seed[0, 0] = 0.0
zeta0, density0 = jnp.array(seed), jnp.array(0.7 * seed)


def final_energy(kappa):
    params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=kappa, hyperviscosity=3.0e-2)
    zeta, density = hw_run(zeta0, density0, grid, params, dt=DT, steps=STEPS)
    phi = potential_from_vorticity(zeta, grid)
    return jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)


grad = float(jax.grad(final_energy)(1.0))
h = 1.0e-4
fd = (float(final_energy(1.0 + h)) - float(final_energy(1.0 - h))) / (2 * h)
print(f"[07] A: dE/dkappa at kappa=1: autodiff {grad:.8e}, finite difference {fd:.8e}, "
      f"rel. diff {abs(grad / fd - 1):.1e} ({time.perf_counter() - start:.1f}s)")

target = float(final_energy(KAPPA_TRUE))
loss_and_grad = jax.jit(jax.value_and_grad(lambda k: (jnp.log(final_energy(k)) - np.log(target)) ** 2))
kappa, history = KAPPA_START, []
for iteration in range(ITERATIONS):
    loss, g = loss_and_grad(kappa)
    history.append((kappa, float(loss)))
    if iteration % 5 == 0:
        print(f"[07] A: iter={iteration:2d} kappa={kappa:.4f} loss={float(loss):.3e} "
              f"({time.perf_counter() - start:.1f}s)")
    kappa = float(np.clip(kappa - LEARNING_RATE * float(g), 0.05, 3.0))
print(f"[07] A: recovered kappa={kappa:.4f} (true {KAPPA_TRUE})")

t_b = time.perf_counter()
params = DetachmentSolParameters(ny=NY, upstream_density=N_UP, power_flux=POWER_FLUX)
guess = detachment_sol_run(params)
print(f"[07] B: steady state converged, residual {guess.residual:.1e} ({time.perf_counter() - t_b:.1f}s)")


def target_temperature(n_up):
    return detachment_target_outputs(jnp.asarray([n_up, POWER_FLUX]), params, guess)[0]


t_t = float(target_temperature(N_UP))
slope = float(jax.jacfwd(target_temperature)(N_UP))
h = 1.0e-3 * N_UP
fd = (float(target_temperature(N_UP + h)) - float(target_temperature(N_UP - h))) / (2 * h)
print(f"[07] B: T_t={t_t:.3f} eV, dT_t/dn_up implicit {slope:.6e}, finite difference {fd:.6e} "
      f"(rel. diff {abs(slope / fd - 1):.1e})")
n_next = N_UP - np.log(t_t / TARGET_EV) / (slope / t_t)   # Newton step on ln T_t
next_params = DetachmentSolParameters(ny=NY, upstream_density=n_next, power_flux=POWER_FLUX)
check = detachment_sol_run(next_params, guess.state, source_scale=guess.source_scale)
t_next = float(detachment_diagnostics(check.state, next_params, check.source_scale).target_temperature_ev)
print(f"[07] B: one Newton step n_up {N_UP:.3e} -> {n_next:.3e}: T_t {t_t:.2f} -> {t_next:.2f} eV "
      f"(requested {TARGET_EV}) ({time.perf_counter() - t_b:.1f}s)")

# 4. Results ----------------------------------------------------------------------
# A: the gradient through 150 nonlinear RK4 steps matches finite differences, and
#    plain gradient descent finds kappa from the energy alone.
# B: the implicit derivative costs one linear solve and agrees with finite
#    differences; one Newton step lands close to the requested T_t (repeat to converge,
#    see examples/autodiff/detachment_control.py).
print(f"[07] total {time.perf_counter() - start:.1f}s")
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), layout="constrained")
axes[0].semilogy([h[1] + 1e-16 for h in history], "o-", ms=3)
axes[0].set(title="A: inverse design loss", xlabel="iteration")
axes[1].plot([h[0] for h in history], "o-", ms=3, label="kappa")
axes[1].axhline(KAPPA_TRUE, color="k", ls="--", label="true kappa")
axes[1].set(title="A: recovered drive", xlabel="iteration")
axes[1].legend(fontsize=7)
fig.savefig(FIGURE, dpi=80)
print(f"[07] wrote {FIGURE}")
