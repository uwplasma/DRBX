"""Tutorial 03 -- linear dispersion (drbx.linear) vs early growth of the nonlinear run.

Linearize the Hasegawa-Wakatani equations of tutorial 02 about n = phi = 0 for
one Fourier mode exp(i k.x + lambda t):

    lambda (-k^2 phi) = alpha (phi - n)
    lambda n          = -i ky kappa phi + alpha (phi - n)

``drbx.linear.resistive_drift_wave_operator`` builds this 2x2 matrix and
``eigenmodes`` returns lambda = gamma + i omega. The script

1. scans ky (kx = 0) and plots the growth rate gamma(ky) for several alpha,
2. seeds the nonlinear code with a single eigenmode (its self-bracket {phi, phi}
   vanishes, so it must grow at exactly gamma), and
3. seeds random noise and compares the energy growth with 2 gamma_max over the
   modes that fit in the box.

Run from the repository root (page: docs/tutorial/03_linear_vs_nonlinear.md):

    PYTHONPATH=src python examples/tutorial/03_linear_vs_nonlinear.py
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

from drbx.linear import eigenmodes, resistive_drift_wave_operator
from drbx.native.hasegawa_wakatani import (
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    potential_from_vorticity,
)

# 2. Input parameters -----------------------------------------------------------
ALPHAS = (0.1, 1.0, 10.0)     # adiabaticities for the dispersion scan
KAPPA = 1.0                   # density-gradient drive
KY_SCAN = np.linspace(0.05, 3.0, 60)
N = 32                        # nonlinear grid
LENGTH = 2.0 * np.pi * 4.0    # box side: allowed k are multiples of 2 pi / LENGTH = 0.25
ALPHA_RUN = 1.0               # adiabaticity of the nonlinear checks
MODE = (0, 4)                 # (mx, my) of the single-mode check: ky = 4 * 0.25 = 1
DT = 1.0e-2
STEPS_PER_BLOCK = 500
BLOCKS = 16
FIGURE = Path("docs/media/tutorial_03_linear_vs_nonlinear.png")


def growth_and_vector(ky, k2, alpha):
    modes = eigenmodes(resistive_drift_wave_operator(ky, k2, alpha, KAPPA))
    return float(np.real(np.asarray(modes.eigenvalues)[0])), np.asarray(modes.eigenvectors)[:, 0]


# 3. Run -------------------------------------------------------------------------
start = time.perf_counter()
print(f"[03] linear scan: {len(KY_SCAN)} ky values x {len(ALPHAS)} alphas")
scan = {alpha: [growth_and_vector(ky, ky**2, alpha)[0] for ky in KY_SCAN] for alpha in ALPHAS}
for alpha, gammas in scan.items():
    best = int(np.argmax(gammas))
    print(f"[03]   alpha={alpha:5}: gamma_max={gammas[best]:.4f} at ky={KY_SCAN[best]:.2f}")
print(f"[03] linear scan done in {time.perf_counter() - start:.1f}s")

grid = hw_grid(N, LENGTH)
params = HasegawaWakataniParameters(adiabaticity=ALPHA_RUN, gradient=KAPPA, hyperviscosity=0.0)
k0 = 2.0 * np.pi / LENGTH


def energy(zeta, density):
    phi = potential_from_vorticity(zeta, grid)
    return float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2))


def evolve(zeta, density, label):
    times, energies = [0.0], [energy(zeta, density)]
    for block in range(1, BLOCKS + 1):
        zeta, density = hw_run(zeta, density, grid, params, dt=DT, steps=STEPS_PER_BLOCK)
        times.append(block * STEPS_PER_BLOCK * DT)
        energies.append(energy(zeta, density))
        print(f"[03]   {label} step={block * STEPS_PER_BLOCK:5d} t={times[-1]:4.1f} E={energies[-1]:.4e}")
    return np.array(times), np.array(energies)


# (a) single eigenmode: amplitude ~ exp(gamma t), energy ~ exp(2 gamma t)
mx, my = MODE
gamma_mode, vector = growth_and_vector(my * k0, (mx**2 + my**2) * k0**2, ALPHA_RUN)
zeta = np.zeros((N, N), complex)
density = np.zeros((N, N), complex)
zeta[mx, my] = -((mx**2 + my**2) * k0**2) * vector[0] * 1e-6   # zeta_k = -k^2 phi_k
density[mx, my] = vector[1] * 1e-6
print(f"[03] single-mode run (mx, my) = {MODE}, linear gamma = {gamma_mode:.6f}")
t_mode, e_mode = evolve(jnp.array(zeta), jnp.array(density), "single mode")
gamma_measured = 0.5 * np.log(e_mode[-1] / e_mode[0]) / t_mode[-1]

# (b) random noise: after transients the fastest box mode dominates
box_gammas = [growth_and_vector(j * k0, (i**2 + j**2) * k0**2, ALPHA_RUN)[0]
              for i in range(0, N // 3) for j in range(1, N // 3)]
gamma_box = max(box_gammas)
rng = np.random.default_rng(0)
noise = np.fft.fft2(rng.standard_normal((N, N))) * 1e-8
noise[0, 0] = 0.0
noise *= np.asarray(grid.dealias)
print(f"[03] noise run, fastest box mode gamma = {gamma_box:.6f}")
t_noise, e_noise = evolve(jnp.array(noise), jnp.array(noise), "noise")
half = len(t_noise) // 2
gamma_noise = 0.5 * np.polyfit(t_noise[half:], np.log(e_noise[half:]), 1)[0]

# 4. Results ----------------------------------------------------------------------
print(f"[03] single mode: measured gamma = {gamma_measured:.6f}, linear = {gamma_mode:.6f}, "
      f"rel. diff {abs(gamma_measured / gamma_mode - 1):.1e} (exact up to time-step error)")
print(f"[03] noise: late-time gamma = {gamma_noise:.4f}, fastest box mode = {gamma_box:.4f} "
      "(approaches it as the fastest mode outgrows the rest)")
print(f"[03] total {time.perf_counter() - start:.1f}s")

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), layout="constrained")
for alpha, gammas in scan.items():
    axes[0].plot(KY_SCAN, gammas, label=f"alpha={alpha}")
axes[0].plot([my * k0], [gamma_measured], "k*", ms=10, label="nonlinear code, single mode")
axes[0].set(title="linear growth rate (kx = 0)", xlabel="ky rho_s", ylabel="gamma")
axes[0].legend(fontsize=7)
axes[1].semilogy(t_mode, e_mode / e_mode[0], "o", label="single mode (nonlinear code)")
axes[1].semilogy(t_mode, np.exp(2 * gamma_mode * t_mode), "-", label="exp(2 gamma t), linear")
axes[1].semilogy(t_noise, e_noise / e_noise[0], "s", label="random noise")
axes[1].semilogy(t_noise, e_noise[-1] / e_noise[0] * np.exp(2 * gamma_box * (t_noise - t_noise[-1])), "--",
                 label="exp(2 gamma_max t)")
axes[1].set(title=f"energy growth, alpha={ALPHA_RUN}", xlabel="time", ylabel="E / E(0)")
axes[1].legend(fontsize=7)
fig.savefig(FIGURE, dpi=80)
print(f"[03] wrote {FIGURE}")
