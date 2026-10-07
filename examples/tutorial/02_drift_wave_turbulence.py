"""Tutorial 02 -- 2-D drift-wave turbulence (Hasegawa-Wakatani) and its energy diagnostics.

Evolves vorticity zeta = lap(phi) and density n on a doubly periodic plane:

    d zeta/dt = -{phi, zeta} + alpha (phi - n) - nu lap^2 zeta - mu zeta
    d n/dt    = -{phi, n} - kappa d phi/dy + alpha (phi - n) - nu lap^2 n - mu n

for two values of the adiabaticity alpha, from the same small random seed.
Prints step/time/energy/flux every block, then plots the final vorticity of
each case, the fluctuation energy E = <|grad phi|^2 + n^2>, and the radial
particle flux Gamma = <n v_x>.

Run from the repository root (page: docs/tutorial/02_drift_wave_turbulence.md):

    PYTHONPATH=src python examples/tutorial/02_drift_wave_turbulence.py
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
    particle_flux,
    potential_from_vorticity,
)

# 2. Input parameters -----------------------------------------------------------
N = 64                        # grid points per side
LENGTH = 2.0 * np.pi * 6.0    # box side in rho_s (smallest k = 2 pi / LENGTH)
ADIABATICITIES = (0.2, 1.0)   # alpha values to compare (parallel electron coupling)
GRADIENT = 1.0                # kappa: background density-gradient drive
HYPERVISCOSITY = 1.0e-2       # nu: grid-scale damping
FRICTION = 3.0e-2             # mu: large-scale drag (absorbs the inverse cascade)
DT = 5.0e-3                   # RK4 time step
STEPS_PER_BLOCK = 500         # steps between progress prints / diagnostics
BLOCKS = 60                   # total time = BLOCKS * STEPS_PER_BLOCK * DT
SEED_AMPLITUDE = 5.0e-2       # rms of the initial noise
FIGURE = Path("docs/media/tutorial_02_drift_wave_turbulence.png")

# 3. Run -------------------------------------------------------------------------
grid = hw_grid(N, LENGTH)
rng = np.random.default_rng(0)
noise = np.fft.fft2(rng.standard_normal((N, N)))  # FFT of a real field: stays real
noise[0, 0] = 0.0
noise *= np.exp(-np.asarray(grid.k2)) * np.asarray(grid.dealias)
noise *= SEED_AMPLITUDE / np.sqrt(np.mean(np.real(np.fft.ifft2(noise)) ** 2))
print(f"[02] grid {N}x{N}, box {LENGTH:.1f} rho_s, dt={DT}, run to t={BLOCKS * STEPS_PER_BLOCK * DT:g}")

runs = {}
for alpha in ADIABATICITIES:
    params = HasegawaWakataniParameters(adiabaticity=alpha, gradient=GRADIENT,
                                        hyperviscosity=HYPERVISCOSITY, friction=FRICTION)
    zeta, density = jnp.array(noise), jnp.array(noise)
    history = {"t": [], "energy": [], "flux": []}
    start = time.perf_counter()
    print(f"[02] alpha={alpha}: running {BLOCKS} blocks of {STEPS_PER_BLOCK} steps")
    for block in range(1, BLOCKS + 1):
        zeta, density = hw_run(zeta, density, grid, params, dt=DT, steps=STEPS_PER_BLOCK)
        phi = potential_from_vorticity(zeta, grid)
        energy = float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)) / N**4
        flux = float(particle_flux(zeta, density, grid))
        history["t"].append(block * STEPS_PER_BLOCK * DT)
        history["energy"].append(energy)
        history["flux"].append(flux)
        print(f"[02]   alpha={alpha} step={block * STEPS_PER_BLOCK:6d} t={history['t'][-1]:6.1f} "
              f"E={energy:.3e} Gamma={flux:+.3e}  ({time.perf_counter() - start:.1f}s)")
    history["vorticity"] = np.real(np.asarray(jnp.fft.ifft2(zeta)))
    runs[alpha] = history

# 4. Results ----------------------------------------------------------------------
# Linear phase: E grows exponentially (straight line on the log plot). Saturation:
# the E x B nonlinearity transfers energy to other scales and E levels off.
# Gamma > 0 means density is carried outward, down the background gradient.
# Small alpha (hydrodynamic, weakly coupled n and phi): larger eddies and larger flux.
# Large alpha (adiabatic, n ~ phi): weaker drive, smaller flux.
for alpha, history in runs.items():
    tail = max(1, BLOCKS // 4)
    print(f"[02] alpha={alpha}: mean flux over the last {tail} blocks = {np.mean(history['flux'][-tail:]):+.3e}")

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.1), layout="constrained")
for ax, (alpha, history) in zip(axes[:2], runs.items()):
    ax.imshow(history["vorticity"], cmap="RdBu_r", origin="lower")
    ax.set(title=f"vorticity, alpha={alpha}", xticks=[], yticks=[])
for alpha, history in runs.items():
    axes[2].semilogy(history["t"], history["energy"], label=f"alpha={alpha}")
    axes[3].plot(history["t"], history["flux"], label=f"alpha={alpha}")
axes[2].set(title="fluctuation energy E", xlabel="time")
axes[3].set(title="particle flux Gamma", xlabel="time")
axes[3].legend(fontsize=7)
fig.savefig(FIGURE, dpi=80)
print(f"[02] wrote {FIGURE}")
