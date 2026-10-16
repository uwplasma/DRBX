"""Tutorial 05 -- collisions with neutrals: recycling, AMJUEL rates, and the SD1D divertor leg.

Part A (``sol_recycling_run``): ions on a prescribed hot-upstream / cold-target
temperature profile, coupled to a recycled neutral density n_n. Neutrals come
back from the target (fraction R of the Bohm ion flux), diffuse along the field,
and are ionized / recombined with AMJUEL rates; charge exchange and
recombination drag the ion flow:

    dn_i/dt + d(n_i v)/dz = S_iz - S_rec,      S_iz = <sv>_iz n_n n_e, S_rec = <sv>_rec n_i n_e
    dn_n/dt - d/dz(D_n dn_n/dz) = -(S_iz - S_rec)
    momentum loses the friction (S_cx + S_rec) m v,        S_cx = <sv>_cx n_n n_i

Part B (``detachment_sol_run``): the SD1D-matched 1D divertor leg (Dudson et
al. 2019, hydrogen, 13.6 eV per ionization) with evolved temperature. Raising
the upstream density cools the target; the particle and power ledgers close.

Run from the repository root (page: docs/tutorial/05_neutrals_and_detachment.md):

    PYTHONPATH=src python examples/tutorial/05_neutrals_and_detachment.py
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

from drbx.native.neutrals import (
    DetachmentSolParameters,
    PlasmaNormalization,
    SolRecyclingParameters,
    SolRecyclingState,
    detachment_diagnostics,
    detachment_ledger,
    detachment_sol_run,
    linear_target_temperature_profile,
    sol_recycling_run,
)

# 2. Input parameters -----------------------------------------------------------
# Part A: recycling SOL with a prescribed temperature (hermes-3 normalization:
# density / 1e19 m^-3, temperature / 100 eV, time in L / c_s).
NZ = 200                        # parallel cells, mid-plane (z=0) to target (z=L)
PARALLEL_LENGTH = 30.0          # metres
UPSTREAM_EV, TARGET_EV = 30.0, 1.5
RECYCLING_FRACTION = 0.95
NEUTRAL_DIFFUSION = 8.0        # neutral parallel diffusion enhancement (kinetic-transport proxy)
NEUTRAL_TEMPERATURE = 0.04     # ~4 eV Franck-Condon neutrals (normalized)
UPSTREAM_SCAN_A = (1.0, 4.0, 12.0)   # upstream densities (x 1e19 m^-3)
STEPS_A = 60000
REPORT_EVERY_A = 20000
DT_A = 0.1 / NZ
# Part B: SD1D-matched leg (30 m, 5e7 W/m^2, 99% recycling).
NY_B = 200                      # cells (800 reproduces the published SD1D numbers)
UPSTREAM_SCAN_B = (2.0e19, 3.0e19, 4.0e19, 6.0e19)
FIGURE = Path("docs/media/tutorial_05_neutrals_and_detachment.png")

# 3. Run -------------------------------------------------------------------------
start = time.perf_counter()
norm = PlasmaNormalization()
temperature = linear_target_temperature_profile(NZ, upstream_ev=UPSTREAM_EV, target_ev=TARGET_EV,
                                                normalization=norm)
sound_speed = np.sqrt(np.asarray(temperature))   # (Te + Ti) / m_i with Ti = Te and m_i = 2 m_p
print(f"[05] Part A: recycling SOL, {NZ} cells, Te {UPSTREAM_EV} -> {TARGET_EV} eV, R = {RECYCLING_FRACTION}")
profiles = {}
for n_up in UPSTREAM_SCAN_A:
    params = SolRecyclingParameters(parallel_length=PARALLEL_LENGTH, upstream_density=n_up,
                                    recycling_fraction=RECYCLING_FRACTION, neutral_diffusion=NEUTRAL_DIFFUSION,
                                    neutral_temperature=NEUTRAL_TEMPERATURE, normalization=norm)
    state = SolRecyclingState(ion_density=jnp.full(NZ, n_up), ion_momentum=jnp.zeros(NZ),
                              neutral_density=jnp.full(NZ, 0.05))
    for done in range(REPORT_EVERY_A, STEPS_A + 1, REPORT_EVERY_A):
        state = sol_recycling_run(state, temperature, params, dt=DT_A, steps=REPORT_EVERY_A)
        n_i = np.asarray(state.ion_density)
        mach = np.asarray(state.ion_momentum) / (params.ion_mass * n_i) / sound_speed
        print(f"[05]   n_up={n_up:4.1f} step={done:6d} t={done * DT_A:5.1f} target Mach={mach[-1]:.3f} "
              f"n_neutral(target)={float(state.neutral_density[-1]):.3f} ({time.perf_counter() - start:.1f}s)")
    profiles[n_up] = (n_i, np.asarray(state.neutral_density), mach)

print(f"[05] Part B: SD1D-matched leg, {NY_B} cells, upstream density scan")
rows, previous = [], None
for n_up in UPSTREAM_SCAN_B:
    t_case = time.perf_counter()
    params = DetachmentSolParameters(ny=NY_B, upstream_density=n_up, ionisation_energy=13.6)
    result = detachment_sol_run(params, previous.state if previous else None,
                                source_scale=previous.source_scale if previous else 0.1)
    previous = result   # continuation: start the next density from this steady state
    diag = detachment_diagnostics(result.state, params, result.source_scale)
    ledger = detachment_ledger(result, params)
    rows.append((n_up, float(diag.target_temperature_ev), float(diag.target_ion_flux_area)))
    print(f"[05]   n_up={n_up:.1e}: {result.iterations} iterations, residual {result.residual:.1e}, "
          f"T_target={rows[-1][1]:6.2f} eV, flux={rows[-1][2]:.3e} m^-2 s^-1, "
          f"ledgers particle {ledger['particle_imbalance']:.0e} power {ledger['power_imbalance']:.0e} "
          f"({time.perf_counter() - t_case:.1f}s)")

# 4. Results ----------------------------------------------------------------------
# Part A: more upstream density -> more recycled neutrals near the target -> more
# charge-exchange friction -> the target Mach number drops (detachment onset).
# (At low density the imposed temperature drop accelerates the flow past Mach 1;
# the Bohm condition is the inequality |v| >= c_s.)
# Part B: more upstream density -> the same power is shared by more particles and
# more is spent on ionization -> the target temperature falls (29 -> ~4 eV in SD1D).
print("[05] Part A target Mach: " + ", ".join(f"n_up={k:g}: {v[2][-1]:.3f}" for k, v in profiles.items()))
print("[05] Part B target T:    " + ", ".join(f"{n:.0e}: {t:.2f} eV" for n, t, _ in rows))

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.3), layout="constrained")
z = (np.arange(NZ) + 0.5) / NZ
for n_up, (n_i, n_n, mach) in profiles.items():
    line, = axes[0].plot(z, n_i / n_up, label=f"n_i, n_up={n_up:g}")
    axes[0].plot(z, n_n / n_n.max(), ":", color=line.get_color())
    axes[1].plot(z, mach, color=line.get_color(), label=f"n_up={n_up:g}")
axes[0].set(title="A: ion (solid) and neutral (dotted) density", xlabel="z / L (target at 1)")
axes[0].legend(fontsize=6)
axes[1].set(title="A: Mach number", xlabel="z / L")
axes[1].legend(fontsize=6)
axes[2].plot([r[0] / 1e19 for r in rows], [r[1] for r in rows], "o-")
axes[2].set(title="B: SD1D-matched leg", xlabel="upstream density (1e19 m^-3)", ylabel="target T (eV)")
fig.savefig(FIGURE, dpi=80)
print(f"[05] wrote {FIGURE}; total {time.perf_counter() - start:.1f}s")
