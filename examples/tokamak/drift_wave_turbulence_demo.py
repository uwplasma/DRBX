"""Tokamak closed-field-line flagship: Hasegawa-Wakatani drift-wave turbulence.

Evolve the JAX-native two-field Hasegawa-Wakatani model from small noise through
the linear drift-wave instability into the onset of nonlinear E x B transport,
and plot the vorticity field, the fluctuation-energy growth, and the outward
particle flux. The linear growth phase is verified against ``jax_drb.linear`` in
``tests/test_hasegawa_wakatani.py`` (benchmark B2).

    PYTHONPATH=src python examples/tokamak/drift_wave_turbulence_demo.py

writes ``output/drift_wave_turbulence/`` with a PNG and a JSON time series.
Edit the constants below to change resolution, drive, or run length.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jax_drb.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    particle_flux,
    potential_from_vorticity,
)

# Numerics note: this fixed-step explicit run resolves the linear drift-wave
# instability and the onset of nonlinear E x B transport in a bounded window.
# The step must satisfy two limits: the stiff adiabatic response
# ``dt < 2.8 (k_min^2)/alpha`` (k_min = 2*pi/LENGTH), and the advective CFL,
# which tightens as the fluctuation amplitude grows. Reaching deep saturated
# turbulence needs CFL-adaptive time stepping (a Phase 7 performance item);
# with a fixed step the window below stays finite while showing the instability.
N = 96
LENGTH = 2.0 * np.pi * 8.0
ADIABATICITY = 1.0
GRADIENT = 1.0
HYPERVISCOSITY = 5.0e-2
DT = 5.0e-3
STEPS_PER_BLOCK = 400
BLOCKS = 32
SEED = 0
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "drift_wave_turbulence"


def main() -> None:
    grid = hw_grid(N, LENGTH)
    params = HasegawaWakataniParameters(
        adiabaticity=ADIABATICITY, gradient=GRADIENT, hyperviscosity=HYPERVISCOSITY
    )
    rng = np.random.default_rng(SEED)
    seed = 1.0e-2 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    seed[0, 0] = 0.0
    zeta = jnp.array(seed.copy())
    density = jnp.array(seed.copy())

    times, energies, fluxes = [], [], []
    for block in range(BLOCKS):
        zeta, density = hw_run(zeta, density, grid, params, dt=DT, steps=STEPS_PER_BLOCK)
        phi = potential_from_vorticity(zeta, grid)
        energy = float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2))
        flux = float(particle_flux(zeta, density, grid))
        times.append(DT * STEPS_PER_BLOCK * (block + 1))
        energies.append(energy)
        fluxes.append(flux)
        print(f"t={times[-1]:7.1f}  energy={energy:.4e}  particle_flux={flux:+.4e}")

    vorticity = np.real(np.asarray(jnp.fft.ifft2(zeta)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "drift_wave_turbulence.json").write_text(
        json.dumps({"time": times, "energy": energies, "particle_flux": fluxes}, indent=2)
    )

    fig, (ax_field, ax_energy, ax_flux) = plt.subplots(1, 3, figsize=(15.0, 4.4))
    im = ax_field.imshow(vorticity, cmap="RdBu_r", origin="lower")
    ax_field.set_title("vorticity")
    ax_field.set_xticks([])
    ax_field.set_yticks([])
    fig.colorbar(im, ax=ax_field, fraction=0.046)
    ax_energy.semilogy(times, energies, "-")
    ax_energy.set_xlabel("time")
    ax_energy.set_ylabel("fluctuation energy")
    ax_energy.set_title("instability growth")
    ax_energy.grid(alpha=0.3)
    ax_flux.plot(times, fluxes, "-")
    ax_flux.set_xlabel("time")
    ax_flux.set_ylabel(r"particle flux $\langle n\, v_x\rangle$")
    ax_flux.set_title("outward transport")
    ax_flux.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "drift_wave_turbulence.png", dpi=180)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'drift_wave_turbulence.png'}")


if __name__ == "__main__":
    main()
