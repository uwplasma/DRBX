"""B6: SD1D detachment benchmark -- target-flux rollover.

Scans the upstream density at fixed upstream power through the self-consistent
detaching SOL model and plots the two detachment signatures: the target ion flux
rising then rolling over, and the target temperature cooling from an attached hot
target into the recombining regime below 1 eV. This is the classic SD1D
detachment picture (Dudson et al., PPCF 61, 065008, 2019).

The model evolves ion density, parallel momentum, plasma pressure, and a
diffusive recycled neutral along the field: hyperbolic transport with a Bohm
sheath at the target, an upstream power source, implicit Spitzer conduction
(kappa ~ T^{5/2}, tridiagonal solve), self-limiting radiative/ionization energy
loss from the AMJUEL fits, and charge-exchange + recombination momentum
friction. The stiff pieces are implicit, so the scan is stable -- and the whole
solve is end-to-end differentiable (see examples/autodiff/detachment_control.py).

For each scan point the script prints the relaxation progress (target Te and
flux per chunk) and a final line; at the end it reports the rollover density
and the detached flux reduction.

Run:

    PYTHONPATH=src python examples/benchmarks/b6_detachment_rollover.py

writes ``output/b6_detachment/`` with a two-panel PNG and a JSON summary.
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
# Fields are hermes-3 normalized (density / Nnorm, temperature / Tnorm); time
# is in parallel transit times L / c_s.
# ----------------------------------------------------------------------------
NZ = 160                      # parallel cells, stagnation midplane (z=0) to target (z=L)
PARALLEL_LENGTH = 30.0        # connection length L in metres (sets the atomic-rate scale)
UPSTREAM_POWER = 6.0          # normalized upstream power source (fixed across the scan)
POWER_WIDTH = 0.2             # parallel width of the upstream power deposition (z/L units)
CONDUCTION_COEFFICIENT = 2.0  # Spitzer kappa0 (normalized); heat conduction ~ kappa0 T^{5/2}
SHEATH_TRANSMISSION = 7.0     # sheath heat transmission gamma (Bohm heat sink at the target)
RECYCLING_FRACTION = 0.95     # fraction of the target ion flux recycled as neutrals
NEUTRAL_DIFFUSION = 8.0       # neutral parallel diffusion enhancement
NEUTRAL_SEED = 0.05           # initial trace neutral density (normalized)
ION_MASS = 2.0                # ion mass in proton masses (deuterium)
NORMALIZATION = PlasmaNormalization(Tnorm=50.0)  # SOL reference: Tnorm = 50 eV, Nnorm = 1e19 m^-3
INITIAL_TEMPERATURE = 0.6     # initial normalized temperature (30 eV) -> pressure = 2 n T
DENSITY_SCAN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 28.0, 40.0]
STEPS = 45000                 # operator-split steps per scan point
REPORT_EVERY = 15000          # steps between progress prints within a scan point
DT = 0.25 * (1.0 / NZ) / 3.0  # timestep: CFL 0.25 against the fastest wave (~3 c_s)
OUTPUT_DIR = Path("output/b6_detachment")  # PNG + JSON summary land here

# ----------------------------------------------------------------------------
# Simulation setup and scan loop. Each scan point builds the parameter
# dataclass and cold-start state explicitly -- change the fields here (or the
# PARAMETERS above) to change the model, e.g. upstream_power for a power scan
# or recycling_fraction for a wall-pumping study.
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("== B6 SD1D detachment benchmark: setup ==")
print(f"   grid: {NZ} cells, L_parallel = {PARALLEL_LENGTH} m, dt = {DT:.2e} transit times, "
      f"{STEPS} steps per point")
print(f"   fixed upstream power = {UPSTREAM_POWER} (normalized), "
      f"Tnorm = {NORMALIZATION.Tnorm} eV, Nnorm = {NORMALIZATION.Nnorm:.0e} m^-3")
print(f"   scanning upstream density over {DENSITY_SCAN}")

flux, temperature, target_density, target_neutral = [], [], [], []
for upstream in DENSITY_SCAN:
    params = DetachmentSolParameters(
        parallel_length=PARALLEL_LENGTH,
        upstream_density=upstream,            # Dirichlet-pinned at z = 0 every step
        upstream_power=UPSTREAM_POWER,
        power_width=POWER_WIDTH,
        conduction_coefficient=CONDUCTION_COEFFICIENT,
        sheath_transmission=SHEATH_TRANSMISSION,
        neutral_diffusion=NEUTRAL_DIFFUSION,
        recycling_fraction=RECYCLING_FRACTION,
        ion_mass=ION_MASS,
        normalization=NORMALIZATION,
    )
    state = DetachmentSolState(
        ion_density=jnp.full(NZ, upstream),                              # flat cold start
        ion_momentum=jnp.zeros(NZ),                                      # plasma at rest
        plasma_pressure=jnp.full(NZ, 2.0 * upstream * INITIAL_TEMPERATURE),  # P = 2 n T
        neutral_density=jnp.full(NZ, NEUTRAL_SEED),                      # trace neutral seed
    )
    print(f"-- n_up = {upstream:5.1f}: relaxing the coupled plasma/neutral/pressure system")
    for start in range(0, STEPS, REPORT_EVERY):
        chunk = min(REPORT_EVERY, STEPS - start)
        state = detachment_sol_run(state, params, dt=DT, steps=chunk)
        progress = detachment_diagnostics(state, params)
        print(f"   step {start + chunk:6d}/{STEPS}:  Te_target = "
              f"{float(progress.target_temperature_ev):7.3f} eV,  "
              f"target flux = {float(progress.target_ion_flux):8.4f}")
    diagnostics = detachment_diagnostics(state, params)
    flux.append(float(diagnostics.target_ion_flux))
    temperature.append(float(diagnostics.target_temperature_ev))
    target_density.append(float(diagnostics.target_density))
    target_neutral.append(float(np.asarray(state.neutral_density)[-1]))
    regime = "attached" if temperature[-1] > 1.0 else "detached (recombining, < 1 eV)"
    print(f"   done: target flux = {flux[-1]:.4f} (n c_s), Te_target = {temperature[-1]:.3f} eV, "
          f"n_target = {target_density[-1]:.3f}, n_neutral(target) = {target_neutral[-1]:.3f} "
          f"-> {regime}")

# ----------------------------------------------------------------------------
# Rollover diagnostics: the SD1D detachment signature is the flux maximum at an
# intermediate density and the target cooling through 1 eV.
# ----------------------------------------------------------------------------
peak_index = int(np.argmax(flux))
detached_reduction = 1.0 - flux[-1] / flux[peak_index]
print("== Detachment rollover summary ==")
print(f"   rollover at upstream density {DENSITY_SCAN[peak_index]} "
      f"(peak target flux {flux[peak_index]:.4f})")
print(f"   flux at the highest density {flux[-1]:.4f}: "
      f"reduced {detached_reduction * 100:.0f}% below the peak")
print(f"   target Te falls {temperature[0]:.2f} eV -> {temperature[-1]:.3f} eV across the scan "
      f"(crosses 1 eV: recombining regime)")

summary = {
    "upstream_density": DENSITY_SCAN,
    "target_flux": flux,
    "target_temperature_ev": temperature,
    "rollover_density": DENSITY_SCAN[peak_index],
    "peak_flux": flux[peak_index],
    "detached_flux_reduction": detached_reduction,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"wrote {OUTPUT_DIR / 'summary.json'}")

# ----------------------------------------------------------------------------
# Figure: target-flux rollover (left) and self-consistent target cooling (right).
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
axes[0].plot(DENSITY_SCAN, flux, "o-", color="#1f77b4")
axes[0].axvline(DENSITY_SCAN[peak_index], color="gray", ls=":", alpha=0.6)
axes[0].annotate("rollover", (DENSITY_SCAN[peak_index], flux[peak_index]),
                 textcoords="offset points", xytext=(10, -4), fontsize=9)
axes[0].set_xlabel("upstream density (normalized)")
axes[0].set_ylabel("target ion flux  n c_s")
axes[0].set_title("Target-flux rollover")
axes[0].grid(True, ls=":", alpha=0.4)

axes[1].semilogy(DENSITY_SCAN, temperature, "o-", color="#d62728")
axes[1].axhline(1.0, color="gray", ls=":", alpha=0.6)
axes[1].annotate("1 eV (recombination)", (DENSITY_SCAN[-1], 1.0),
                 textcoords="offset points", xytext=(-140, 4), fontsize=8)
axes[1].set_xlabel("upstream density (normalized)")
axes[1].set_ylabel("target temperature (eV)")
axes[1].set_title("Self-consistent target cooling")
axes[1].grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("B6 detachment: target-flux rollover and cooling into the recombining regime")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "b6_detachment.png", dpi=180)
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'b6_detachment.png'}")
