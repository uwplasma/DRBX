"""Open-field-line SOL flux tube: parallel transport to sheath-bounded targets.

The open-field-line counterpart to the closed flux-tube flagships. On the open
slab geometry (field lines terminating on target plates at both ends), the
reduced isothermal SOL model transports an upstream particle source along the
field to the targets, where a Bohm sheath drains it at the sound speed. The run
relaxes to the classic two-point steady state -- the flow accelerates from a
stagnation point to Mach 1 at each target, and the target density is half the
upstream density -- and the kept FCI sheath/recycling closure then reports the
target particle flux, heat load, and recycled-neutral source.

The model evolves density n and parallel momentum m = n v as an isothermal
Euler system along z (Rusanov faces, RK4 in time), with a Gaussian particle
source at the midplane and Bohm outflow (|v| >= c_s) at both targets:

    dn/dt + d(n v)/dz            = S_n
    dm/dt + d(n v^2 + n c_s^2)/dz = 0

The script prints the setup (grid, timestep, source strength), then progress
lines during the relaxation (target Mach number, target/upstream density
ratio, steady-state residual), then the sheath/recycling target accounting.

Run:

    PYTHONPATH=src python examples/sol/open_sol_flux_tube.py

writes ``output/open_sol_flux_tube/`` with a two-panel PNG (parallel profiles +
target sheath diagnostics) and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jax_drb.geometry import build_open_slab_geometry  # noqa: E402
from jax_drb.native.fci_sheath_recycling import compute_fci_sheath_recycling  # noqa: E402
from jax_drb.native.sol_flux_tube import (  # noqa: E402
    SolFluxTubeParameters,
    sol_flux_tube_run,
    sol_flux_tube_source,
)

# ----------------------------------------------------------------------------
# PARAMETERS -- everything you might want to change, in one place.
# The model is isothermal and normalized: lengths in units of the reference
# length, speeds in units of the sound speed c_s, densities in units of the
# initial (upstream) density.
# ----------------------------------------------------------------------------
PARALLEL_LENGTH = 40.0     # connection length L, target plate to target plate (normalized)
SHAPE = (1, 1, 200)        # (nx, ny, nz) grid: one flux tube, 200 parallel cells (raise nz to refine)
SOUND_SPEED = 1.0          # isothermal sound speed c_s (normalized); sets the Bohm outflow speed
SOURCE_AMPLITUDE = 0.02    # peak of the Gaussian upstream particle source S_n
SOURCE_WIDTH = 4.0         # parallel 1/e width of the source (same units as z)
DENSITY_FLOOR = 1.0e-6     # positivity floor applied after each RK4 step
CFL = 0.4                  # timestep safety factor: dt = CFL * dz / (c_s + max|v| estimate)
STEPS = 60000              # total RK4 steps (enough to relax to the two-point steady state)
REPORT_EVERY = 10000       # steps between progress prints during the relaxation
TE = 0.5                   # normalized electron temperature at the target (Te + Ti = c_s^2 = 1)
TI = 0.5                   # normalized ion temperature at the target
RECYCLING_FRACTION = 0.95  # fraction of the target ion flux returned as neutrals (0 = perfect absorber)
OUTPUT_DIR = Path("output/open_sol_flux_tube")  # PNG + JSON summary land here

# ----------------------------------------------------------------------------
# Simulation setup: geometry, model parameters, source, and initial state.
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Open slab: straight field lines along z that terminate on target plates at
# z = 0 and z = L (this is what makes the geometry "open" -- the FCI endpoint
# masks mark exactly those two planes as sheath entrances).
geometry = build_open_slab_geometry(SHAPE, parallel_length=PARALLEL_LENGTH)

# Physics parameters of the reduced isothermal SOL model.
params = SolFluxTubeParameters(
    sound_speed=SOUND_SPEED,
    source_amplitude=SOURCE_AMPLITUDE,
    source_width=SOURCE_WIDTH,
    density_floor=DENSITY_FLOOR,
)

# Gaussian particle source centred at the parallel midplane (the "upstream"
# stagnation point of the two-point model).
source = sol_flux_tube_source(geometry, params)

# Initial state: uniform density, plasma at rest. The Bohm sheath BC (built
# into the RHS, not a settable array) will drain it until source = sheath loss.
density = jnp.ones(geometry.shape)
momentum = jnp.zeros(geometry.shape)

dz = float(geometry.spacing.dz[0, 0, 0])
dt = CFL * dz / (SOUND_SPEED + 1.0)  # fastest wave is |v| + c_s <= 2 c_s near the targets

print("== Open-field-line SOL flux tube: setup ==")
print(f"   grid: nx,ny,nz = {SHAPE},  L_parallel = {PARALLEL_LENGTH} (dz = {dz:.3f})")
print(f"   dt = {dt:.4e} (CFL {CFL}),  {STEPS} RK4 steps = {STEPS * dt:.1f} sound times")
print(f"   source: peak {SOURCE_AMPLITUDE}, width {SOURCE_WIDTH}, "
      f"total {float(jnp.sum(source) * dz):.4f} particles / unit time")
print(f"   Bohm sheath at both targets (|v| >= c_s = {SOUND_SPEED})")

# ----------------------------------------------------------------------------
# Run loop: relax to steady state in chunks, printing the two-point diagnostics.
# The steady-state residual is max |dn/dt| over the domain -- it should fall
# toward zero as source input balances sheath loss.
# ----------------------------------------------------------------------------
print("== Relaxation to the two-point steady state ==")
for start in range(0, STEPS, REPORT_EVERY):
    chunk = min(REPORT_EVERY, STEPS - start)
    previous_density = density
    density, momentum = sol_flux_tube_run(
        density, momentum, geometry, params, source, dt=dt, steps=chunk
    )
    residual = float(jnp.max(jnp.abs(density - previous_density)) / (chunk * dt))
    n_line = np.asarray(density)[0, 0, :]
    mach_line = (np.asarray(momentum)[0, 0, :] / n_line) / SOUND_SPEED
    upstream_density = n_line[len(n_line) // 2]
    print(f"   step {start + chunk:6d}/{STEPS}:  Mach(targets) = "
          f"{mach_line[0]:+.4f} / {mach_line[-1]:+.4f},  "
          f"n_target/n_up = {n_line[-1] / upstream_density:.4f},  "
          f"max|dn/dt| = {residual:.2e}")

z = np.asarray(geometry.grid.z.centers)
n = np.asarray(density)[0, 0, :]
mach = (np.asarray(momentum)[0, 0, :] / n) / SOUND_SPEED
upstream = n[len(n) // 2]
print("== Two-point steady state reached ==")
print(f"   Bohm criterion: Mach at targets = {mach[0]:+.4f}, {mach[-1]:+.4f} (expect -1, +1)")
print(f"   n_target / n_upstream = {n[-1] / upstream:.4f} (two-point prediction: 0.5)")

# ----------------------------------------------------------------------------
# Sheath / recycling target diagnostics on the relaxed state.
# With Te = Ti = 0.5 the sheath sound speed sqrt(Te + Ti) matches c_s = 1, so
# the FCI closure sees the same Bohm flux the transport model drained.
# ----------------------------------------------------------------------------
te = jnp.full(geometry.shape, TE)
ti = jnp.full(geometry.shape, TI)
sheath = compute_fci_sheath_recycling(
    density, te, ti, geometry.maps, recycling_fraction=RECYCLING_FRACTION
)
print(f"== Target sheath accounting (recycling fraction {RECYCLING_FRACTION}) ==")
print(f"   total target ion loss      = {float(sheath.total_ion_particle_loss):.4f} (particles / unit time)")
print(f"   total recycled neutrals    = {float(sheath.total_recycled_particle_source):.4f} "
      f"(= {RECYCLING_FRACTION} x ion loss)")
print(f"   total target heat load     = {float(sheath.total_target_heat_load):.4f} (normalized power)")
print(f"   particle recycling residual = {float(sheath.particle_recycling_residual):.2e} (identity, ~0)")
print(f"   current balance residual    = {float(sheath.current_balance_residual):.2e} (identity, ~0)")

# ----------------------------------------------------------------------------
# Save the JSON summary and the two-panel figure.
# ----------------------------------------------------------------------------
summary = {
    "parallel_length": PARALLEL_LENGTH,
    "mach_at_targets": [float(mach[0]), float(mach[-1])],
    "n_target_over_upstream": float(n[-1] / upstream),
    "total_target_ion_loss": float(sheath.total_ion_particle_loss),
    "total_recycled_source": float(sheath.total_recycled_particle_source),
    "total_target_heat_load": float(sheath.total_target_heat_load),
    "particle_recycling_residual": float(sheath.particle_recycling_residual),
    "current_balance_residual": float(sheath.current_balance_residual),
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"wrote {OUTPUT_DIR / 'summary.json'}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
ax.plot(z, n / upstream, color="#1f77b4", lw=2, label="density  n / n_upstream")
ax.plot(z, mach, color="#d62728", lw=2, label="Mach  v / c_s")
ax.axhline(0.5, color="#1f77b4", ls=":", alpha=0.6)
ax.axhline(1.0, color="#d62728", ls=":", alpha=0.6)
ax.axhline(-1.0, color="#d62728", ls=":", alpha=0.6)
ax.set_xlabel("parallel coordinate z (connection length)")
ax.set_title("SOL two-point profile: stagnation -> Mach 1 at the targets")
ax.legend(loc="center left", fontsize=8)
ax.grid(True, ls=":", alpha=0.4)

ax = axes[1]
ion_flux = np.asarray(sheath.ion_particle_loss)[0, 0, :]
heat = np.asarray(sheath.target_heat_load)[0, 0, :]
recycled = np.asarray(sheath.recycled_particle_source)[0, 0, :]
labels = ["ion flux\nto target", "target\nheat load", "recycled\nneutral source"]
values = [float(ion_flux[-1]), float(heat[-1]), float(recycled[-1])]
ax.bar(labels, values, color=["#1f77b4", "#d62728", "#2ca02c"])
ax.set_ylabel("per-target-cell rate (normalized)")
ax.set_title(f"Bohm sheath at the target (recycling {RECYCLING_FRACTION:.2f})")
ax.grid(True, axis="y", ls=":", alpha=0.4)

fig.suptitle("Open-field-line SOL flux tube: parallel transport to sheath-bounded targets")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "open_sol_flux_tube.png", dpi=180)
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'open_sol_flux_tube.png'}")
