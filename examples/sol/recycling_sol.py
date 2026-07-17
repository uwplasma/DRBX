"""Coupled 1D recycling SOL: neutrals, ionization/recombination, and the onset
of detachment.

Runs the reduced recycling SOL model (plasma + diffusive neutral, hermes-3 AMJUEL
atomic rates) on a flux tube with a prescribed hot-upstream / cold-target
temperature profile. Along z (stagnation midplane at z = 0, target at z = L)
the model evolves the ion density, ion parallel momentum, and neutral density:
neutrals are recycled from the Bohm target flux (fraction R), diffuse along the
field (implicit tridiagonal solve), and are ionized / recombined via the AMJUEL
rate fits, while charge exchange + recombination friction drag the ion flow.

The script first relaxes a reference case and prints per-chunk progress (target
Mach number, target ion flux, neutral cushion), then scans the upstream density
and prints one line per scan point: as the density rises, charge-exchange
friction chokes the flow (target Mach falls toward and below 1) -- the onset of
detachment. The left figure panel shows the steady-state parallel profiles; the
right panel shows the target Mach number across the scan.

Run:

    PYTHONPATH=src python examples/sol/recycling_sol.py

writes ``output/recycling_sol/`` with a two-panel PNG and a JSON summary.
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
    PlasmaNormalization,
    SolRecyclingParameters,
    SolRecyclingState,
    linear_target_temperature_profile,
    sol_recycling_run,
    sol_recycling_step,  # noqa: F401  (the single-step API, if you want a custom loop)
    target_ion_flux,
)

# ----------------------------------------------------------------------------
# PARAMETERS -- everything you might want to change, in one place.
# Fields are hermes-3 normalized: density / Nnorm, temperature / Tnorm,
# velocity / the reference sound speed; time in parallel transit times L / c_s.
# ----------------------------------------------------------------------------
NZ = 200                      # parallel cells, stagnation midplane (z=0) to target (z=L)
PARALLEL_LENGTH = 30.0        # connection length L in metres (sets the atomic-rate scale)
UPSTREAM_EV = 30.0            # prescribed upstream temperature [eV] (hot attached upstream)
TARGET_EV = 1.5               # prescribed target temperature [eV] (cold recycling region)
REFERENCE_DENSITY = 4.0       # upstream density (normalized) for the profile panel
UPSTREAM_SCAN = [1.0, 2.0, 4.0, 8.0, 12.0]  # upstream densities for the detachment-onset scan
RECYCLING_FRACTION = 0.95     # fraction R of the target ion flux returned as neutrals
NEUTRAL_DIFFUSION = 8.0       # neutral parallel diffusion enhancement (kinetic-transport proxy)
NEUTRAL_TEMPERATURE = 0.04    # normalized neutral temperature (~2 eV Franck-Condon)
NEUTRAL_SEED = 0.05           # initial trace neutral density (normalized)
ION_MASS = 2.0                # ion mass in proton masses (deuterium)
NORMALIZATION = PlasmaNormalization()  # Nnorm = 1e19 m^-3, Tnorm = 100 eV (hermes-3 defaults)
STEPS = 60000                 # operator-split steps per relaxation
REPORT_EVERY = 15000          # steps between progress prints
DT = 0.3 * (1.0 / NZ) / 3.0   # timestep: CFL 0.3 against the fastest wave (~3 c_s, normalized)
OUTPUT_DIR = Path("output/recycling_sol")  # PNG + JSON summary land here

# ----------------------------------------------------------------------------
# Simulation setup: prescribed temperature profile and the relaxation driver.
# The temperature closure is *imposed* (quadratic hot upstream -> cold target),
# which sidesteps the stiff conduction/radiation energy balance; the
# self-consistent version is examples/benchmarks/b6_detachment_rollover.py.
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
temperature = linear_target_temperature_profile(
    NZ, upstream_ev=UPSTREAM_EV, target_ev=TARGET_EV, normalization=NORMALIZATION
)
z = (np.arange(NZ) + 0.5) / NZ  # normalized parallel coordinate, target at z = 1
sound_speed = np.sqrt(2.0 * np.asarray(temperature) / ION_MASS)  # local c_s (normalized)


def relax(upstream_density: float, *, verbose: bool) -> tuple[SolRecyclingState, SolRecyclingParameters]:
    """Relax one upstream density to steady state, printing progress if asked.

    All the physics choices are visible here: change the dataclass fields (or the
    PARAMETERS above) to change the model -- e.g. recycling_fraction=0 for a
    perfectly absorbing target, or neutral_diffusion for the neutral mean free path.
    """

    params = SolRecyclingParameters(
        parallel_length=PARALLEL_LENGTH,
        upstream_density=upstream_density,   # Dirichlet-pinned at z = 0 every step
        recycling_fraction=RECYCLING_FRACTION,
        neutral_diffusion=NEUTRAL_DIFFUSION,
        neutral_temperature=NEUTRAL_TEMPERATURE,
        ion_mass=ION_MASS,
        normalization=NORMALIZATION,
    )
    state = SolRecyclingState(
        ion_density=jnp.full(NZ, upstream_density),  # flat start at the upstream density
        ion_momentum=jnp.zeros(NZ),                  # plasma initially at rest
        neutral_density=jnp.full(NZ, NEUTRAL_SEED),  # trace neutral seed everywhere
    )
    for start in range(0, STEPS, REPORT_EVERY):
        chunk = min(REPORT_EVERY, STEPS - start)
        state = sol_recycling_run(state, temperature, params, dt=DT, steps=chunk)
        if verbose:
            density_now = np.asarray(state.ion_density)
            neutral_now = np.asarray(state.neutral_density)
            velocity_now = np.asarray(state.ion_momentum) / (
                ION_MASS * np.maximum(density_now, params.density_floor)
            )
            print(f"   step {start + chunk:6d}/{STEPS}:  target Mach = "
                  f"{velocity_now[-1] / sound_speed[-1]:.4f},  "
                  f"target flux = {float(target_ion_flux(state, temperature, params)):.4f},  "
                  f"n_neutral(target) = {neutral_now[-1]:.3f}")
    return state, params


print("== Coupled recycling SOL: setup ==")
print(f"   grid: {NZ} cells, L_parallel = {PARALLEL_LENGTH} m, dt = {DT:.2e} transit times")
print(f"   prescribed Te: {UPSTREAM_EV} eV upstream -> {TARGET_EV} eV target "
      f"(Tnorm = {NORMALIZATION.Tnorm} eV, Nnorm = {NORMALIZATION.Nnorm:.0e} m^-3)")
print(f"   recycling fraction R = {RECYCLING_FRACTION}, neutral diffusion x{NEUTRAL_DIFFUSION}")

# ----------------------------------------------------------------------------
# Reference run: relax at the reference upstream density and keep the profiles.
# ----------------------------------------------------------------------------
print(f"== Reference relaxation (upstream density {REFERENCE_DENSITY}) ==")
state, params = relax(REFERENCE_DENSITY, verbose=True)
density = np.asarray(state.ion_density)
neutral = np.asarray(state.neutral_density)
mach = (np.asarray(state.ion_momentum)
        / (ION_MASS * np.maximum(density, params.density_floor))) / sound_speed
neutral_cushion = float(neutral[-1] / max(neutral[NZ // 2], 1e-9))
print("== Reference steady state ==")
print(f"   target Mach = {mach[-1]:.4f} (Bohm-attached would be ~1)")
print(f"   neutral cushion: n_neutral(target)/n_neutral(midpoint) = {neutral_cushion:.1f}")
print(f"   target ion flux n c_s = {float(target_ion_flux(state, temperature, params)):.4f}")

# ----------------------------------------------------------------------------
# Detachment-onset scan: raise the upstream density at fixed temperature and
# watch charge-exchange + recombination friction choke the target flow.
# ----------------------------------------------------------------------------
print("== Upstream-density scan (detachment onset) ==")
target_mach = []
target_flux = []
for upstream in UPSTREAM_SCAN:
    scan_state, scan_params = relax(upstream, verbose=False)
    scan_density = np.asarray(scan_state.ion_density)
    scan_velocity = np.asarray(scan_state.ion_momentum) / (
        ION_MASS * np.maximum(scan_density, scan_params.density_floor)
    )
    target_mach.append(float(scan_velocity[-1] / sound_speed[-1]))
    target_flux.append(float(target_ion_flux(scan_state, temperature, scan_params)))
    print(f"   n_up = {upstream:5.1f}:  target Mach = {target_mach[-1]:.4f},  "
          f"target flux = {target_flux[-1]:.4f},  "
          f"n_neutral(target) = {float(np.asarray(scan_state.neutral_density)[-1]):.3f}")
print(f"   Mach falls {target_mach[0]:.3f} -> {target_mach[-1]:.3f} across the scan: "
      "charge-exchange friction chokes the flow (detachment onset)")

# ----------------------------------------------------------------------------
# Save the JSON summary and the two-panel figure.
# ----------------------------------------------------------------------------
summary = {
    "reference_upstream_density": REFERENCE_DENSITY,
    "target_mach_at_reference": float(mach[-1]),
    "neutral_cushion_ratio": neutral_cushion,
    "upstream_scan": UPSTREAM_SCAN,
    "target_mach_scan": target_mach,
    "target_flux_scan": target_flux,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"wrote {OUTPUT_DIR / 'summary.json'}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
ax.plot(z, density / density[0], color="#1f77b4", lw=2, label="ion density n / n_up")
ax.plot(z, neutral / neutral.max(), color="#2ca02c", lw=2, label="neutral density (norm.)")
ax.plot(z, mach, color="#d62728", lw=2, label="Mach  v / c_s")
ax.axhline(1.0, color="#d62728", ls=":", alpha=0.5)
ax.set_xlabel("parallel coordinate z / L (target at z=1)")
ax.set_title(f"Recycling SOL profiles (upstream density = {REFERENCE_DENSITY:g})")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, ls=":", alpha=0.4)

ax = axes[1]
ax.plot(UPSTREAM_SCAN, target_mach, "o-", color="#d62728", label="target Mach")
ax.axhline(1.0, color="gray", ls=":", alpha=0.6)
ax.set_xlabel("upstream density (normalized)")
ax.set_ylabel("target Mach number")
ax.set_title("Detachment onset: friction chokes the flow as density rises")
ax.grid(True, ls=":", alpha=0.4)
ax.legend(fontsize=8)

fig.suptitle("Coupled recycling SOL: neutrals, ionization/recombination, detachment onset")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "recycling_sol.png", dpi=180)
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'recycling_sol.png'}")
