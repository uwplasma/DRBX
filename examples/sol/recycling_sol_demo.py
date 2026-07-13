"""Coupled 1D recycling SOL: neutrals, ionization/recombination, and the onset
of detachment.

Runs the reduced recycling SOL model (plasma + diffusive neutral, hermes-3 AMJUEL
atomic rates) on a flux tube with a prescribed hot-upstream / cold-target
temperature. The left panel shows the steady-state parallel profiles -- the
ion and neutral densities and the flow Mach number -- with the recycled neutral
cushion at the target. The right panel scans the upstream density and plots the
target Mach number: as the density rises, charge-exchange friction chokes the
flow (Mach falls toward and below 1) -- the onset of detachment.

Run:

    PYTHONPATH=src python examples/sol/recycling_sol_demo.py

writes ``output/recycling_sol/`` with a two-panel PNG and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax_drb.native.neutrals import (  # noqa: E402
    SolRecyclingParameters,
    SolRecyclingState,
    linear_target_temperature_profile,
    sol_recycling_run,
    target_ion_flux,
)

NZ = 200
STEPS = 60000
OUTPUT_DIR = Path("output/recycling_sol")


def _relax(upstream_density, temperature):
    params = SolRecyclingParameters(upstream_density=upstream_density, recycling_fraction=0.95, neutral_diffusion=8.0)
    state = SolRecyclingState(jnp.full(NZ, upstream_density), jnp.zeros(NZ), jnp.full(NZ, 0.05))
    dt = 0.3 * (1.0 / NZ) / 3.0
    state = sol_recycling_run(state, temperature, params, dt=dt, steps=STEPS)
    return state, params


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temperature = linear_target_temperature_profile(NZ, upstream_ev=30.0, target_ev=1.5)
    z = (np.arange(NZ) + 0.5) / NZ

    state, params = _relax(4.0, temperature)
    density = np.asarray(state.ion_density)
    neutral = np.asarray(state.neutral_density)
    sound_speed = np.sqrt(2.0 * np.asarray(temperature) / params.ion_mass)
    mach = (np.asarray(state.ion_momentum) / (params.ion_mass * np.maximum(density, params.density_floor))) / sound_speed

    upstream_scan = [1.0, 2.0, 4.0, 8.0, 12.0]
    target_mach = []
    target_flux = []
    for upstream in upstream_scan:
        scan_state, scan_params = _relax(upstream, temperature)
        scan_density = np.asarray(scan_state.ion_density)
        scan_velocity = np.asarray(scan_state.ion_momentum) / (scan_params.ion_mass * np.maximum(scan_density, scan_params.density_floor))
        target_mach.append(float(scan_velocity[-1] / sound_speed[-1]))
        target_flux.append(float(target_ion_flux(scan_state, temperature, scan_params)))

    summary = {
        "reference_upstream_density": 4.0,
        "target_mach_at_reference": float(mach[-1]),
        "neutral_cushion_ratio": float(neutral[-1] / max(neutral[NZ // 2], 1e-9)),
        "upstream_scan": upstream_scan,
        "target_mach_scan": target_mach,
        "target_flux_scan": target_flux,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.plot(z, density / density[0], color="#1f77b4", lw=2, label="ion density n / n_up")
    ax.plot(z, neutral / neutral.max(), color="#2ca02c", lw=2, label="neutral density (norm.)")
    ax.plot(z, mach, color="#d62728", lw=2, label="Mach  v / c_s")
    ax.axhline(1.0, color="#d62728", ls=":", alpha=0.5)
    ax.set_xlabel("parallel coordinate z / L (target at z=1)")
    ax.set_title("Recycling SOL profiles (upstream density = 4)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    ax.plot(upstream_scan, target_mach, "o-", color="#d62728", label="target Mach")
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


if __name__ == "__main__":
    main()
