"""Open-field-line SOL flux tube: parallel transport to sheath-bounded targets.

The open-field-line counterpart to the closed flux-tube flagships. On the open
slab geometry (field lines terminating on target plates at both ends), the
reduced isothermal SOL model transports an upstream particle source along the
field to the targets, where a Bohm sheath drains it at the sound speed. The run
relaxes to the classic two-point steady state -- the flow accelerates from a
stagnation point to Mach 1 at each target, and the target density is half the
upstream density -- and the kept FCI sheath/recycling closure then reports the
target particle flux, heat load, and recycled-neutral source.

Run:

    PYTHONPATH=src python examples/sol/open_sol_flux_tube_demo.py

writes ``output/open_sol_flux_tube/`` with a two-panel PNG (parallel profiles +
target sheath diagnostics) and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax_drb.geometry import build_open_slab_geometry  # noqa: E402
from jax_drb.native.fci_sheath_recycling import compute_fci_sheath_recycling  # noqa: E402
from jax_drb.native.sol_flux_tube import (  # noqa: E402
    SolFluxTubeParameters,
    sol_flux_tube_run,
    sol_flux_tube_source,
)

PARALLEL_LENGTH = 40.0
SHAPE = (1, 1, 200)
STEPS = 60000
RECYCLING_FRACTION = 0.95
OUTPUT_DIR = Path("output/open_sol_flux_tube")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geometry = build_open_slab_geometry(SHAPE, parallel_length=PARALLEL_LENGTH)
    params = SolFluxTubeParameters(sound_speed=1.0, source_amplitude=0.02, source_width=4.0)
    source = sol_flux_tube_source(geometry, params)

    dz = float(geometry.spacing.dz[0, 0, 0])
    dt = 0.4 * dz / (params.sound_speed + 1.0)
    density, momentum = sol_flux_tube_run(
        jnp.ones(geometry.shape), jnp.zeros(geometry.shape), geometry, params, source, dt=dt, steps=STEPS
    )

    z = np.asarray(geometry.grid.z.centers)
    n = np.asarray(density)[0, 0, :]
    mach = (np.asarray(momentum)[0, 0, :] / n) / params.sound_speed
    upstream = n[len(n) // 2]

    # Sheath / recycling target diagnostics (Te = Ti = 0.5 -> c_s = 1).
    te = jnp.full(geometry.shape, 0.5)
    ti = jnp.full(geometry.shape, 0.5)
    sheath = compute_fci_sheath_recycling(density, te, ti, geometry.maps, recycling_fraction=RECYCLING_FRACTION)

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
    print(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    main()
