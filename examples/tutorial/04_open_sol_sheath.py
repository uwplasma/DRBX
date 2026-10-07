"""Tutorial 04 -- open field lines: a SOL flux tube with Bohm sheaths at both targets.

Isothermal plasma flowing along an open field line z in [0, L] (normalized:
speeds in c_s, densities in the initial density):

    dn/dt     + d(n v)/dz           = S_n          (Gaussian source at mid-plane)
    d(nv)/dt  + d(n v^2 + n c_s^2)/dz = 0
    Bohm sheath at z = 0 and z = L:  |v| >= c_s

The steady state is the two-point solution: v = 0 at the mid-plane, Mach 1 at
each target, and (from n v^2 + n c_s^2 = const) n_target = n_upstream / 2.
The FCI sheath/recycling closure then reports target particle and heat loads.

Run from the repository root (page: docs/tutorial/04_open_sol_sheath.md):

    PYTHONPATH=src python examples/tutorial/04_open_sol_sheath.py
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

from drbx.geometry import build_open_slab_geometry
from drbx.native.fci_sheath_recycling import compute_fci_sheath_recycling
from drbx.native.sol_flux_tube import SolFluxTubeParameters, sol_flux_tube_run, sol_flux_tube_source

# 2. Input parameters -----------------------------------------------------------
PARALLEL_LENGTH = 40.0     # connection length L (target to target)
NZ = 200                   # parallel cells
SOURCE_AMPLITUDE = 0.02    # peak of the particle source S_n
SOURCE_WIDTH = 4.0         # its 1/e width
CFL = 0.4                  # dt = CFL dz / (2 c_s)
STEPS = 60000              # RK4 steps (about 60 sound transits L / c_s)
REPORT_EVERY = 6000        # steps between progress lines
TE = TI = 0.5              # target temperatures for the sheath closure (Te + Ti = c_s^2)
RECYCLING_FRACTION = 0.95  # fraction of the target ion flux returned as neutrals
FIGURE = Path("docs/media/tutorial_04_open_sol_sheath.png")

# 3. Run -------------------------------------------------------------------------
start = time.perf_counter()
geometry = build_open_slab_geometry((1, 1, NZ), parallel_length=PARALLEL_LENGTH)
params = SolFluxTubeParameters(sound_speed=1.0, source_amplitude=SOURCE_AMPLITUDE, source_width=SOURCE_WIDTH)
source = sol_flux_tube_source(geometry, params)
density, momentum = jnp.ones(geometry.shape), jnp.zeros(geometry.shape)
dz = float(geometry.spacing.dz[0, 0, 0])
dt = CFL * dz / 2.0
print(f"[04] L={PARALLEL_LENGTH}, {NZ} cells, dt={dt:.3e}, {STEPS} steps = {STEPS * dt:.0f} time units = {STEPS * dt / PARALLEL_LENGTH:.0f} transits L/c_s")
for done in range(REPORT_EVERY, STEPS + 1, REPORT_EVERY):
    previous = density
    density, momentum = sol_flux_tube_run(density, momentum, geometry, params, source, dt=dt, steps=REPORT_EVERY)
    n = np.asarray(density)[0, 0]
    mach = np.asarray(momentum)[0, 0] / n
    residual = float(jnp.max(jnp.abs(density - previous))) / (REPORT_EVERY * dt)
    print(f"[04]   step={done:6d} t={done * dt:7.1f} Mach(targets)={mach[0]:+.3f}/{mach[-1]:+.3f} "
          f"n_t/n_up={n[-1] / n[NZ // 2]:.4f} max|dn/dt|={residual:.1e} ({time.perf_counter() - start:.1f}s)")

# 4. Results ----------------------------------------------------------------------
z = np.asarray(geometry.grid.z.centers)
upstream = n[NZ // 2]
sheath = compute_fci_sheath_recycling(density, jnp.full(geometry.shape, TE), jnp.full(geometry.shape, TI),
                                      geometry.maps, recycling_fraction=RECYCLING_FRACTION)
injected = float(jnp.sum(source) * dz)
print(f"[04] Bohm: Mach at targets {mach[0]:+.4f}, {mach[-1]:+.4f} (two-point: -1, +1 at the target face;\n"
      "[04]   the last cell centre sits half a cell upstream, so it reads slightly below 1)")
print(f"[04] n_target/n_upstream = {n[-1] / upstream:.4f} (two-point: 0.5)")
print(f"[04] particles in (source) = {injected:.4f}, out to both targets = {float(sheath.total_ion_particle_loss):.4f}")
print(f"[04] recycled neutrals = {float(sheath.total_recycled_particle_source):.4f} (= R x target loss), "
      f"target heat load = {float(sheath.total_target_heat_load):.4f}")

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
ax.plot(z, n / upstream, label="n / n_upstream")
ax.plot(z, mach, label="Mach v / c_s")
ax.plot(z, np.asarray(source)[0, 0] / SOURCE_AMPLITUDE, ":", label="source shape")
for level in (0.5, 1.0, -1.0):
    ax.axhline(level, color="gray", lw=0.6, ls="--")
ax.set(title="SOL flux tube steady state (two-point check)", xlabel="z (target at 0 and L)")
ax.legend(fontsize=7)
fig.savefig(FIGURE, dpi=80)
print(f"[04] wrote {FIGURE}; total {time.perf_counter() - start:.1f}s")
