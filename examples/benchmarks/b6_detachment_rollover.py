"""B6: SD1D detachment benchmark -- target-flux rollover.

Scans the upstream density at fixed upstream power through the self-consistent
detaching SOL model and plots the two detachment signatures: the target ion flux
rising then rolling over, and the target temperature cooling from an attached hot
target into the recombining regime below 1 eV. This is the classic SD1D
detachment picture (Dudson et al., PPCF 61, 065008, 2019), here with the
implicit Spitzer conduction and self-limiting radiative loss that keep the stiff
energy balance stable, and end-to-end differentiable.

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
import numpy as np  # noqa: E402

from jax_drb.native.neutrals import (  # noqa: E402
    DetachmentSolParameters,
    DetachmentSolState,
    detachment_diagnostics,
    detachment_sol_run,
)

NZ = 160
STEPS = 45000
UPSTREAM_POWER = 6.0
OUTPUT_DIR = Path("output/b6_detachment")


def _relax(upstream_density):
    params = DetachmentSolParameters(upstream_density=upstream_density, upstream_power=UPSTREAM_POWER)
    state = DetachmentSolState(
        jnp.full(NZ, upstream_density),
        jnp.zeros(NZ),
        jnp.full(NZ, 2.0 * upstream_density * 0.6),
        jnp.full(NZ, 0.05),
    )
    state = detachment_sol_run(state, params, dt=0.25 * (1.0 / NZ) / 3.0, steps=STEPS)
    return detachment_diagnostics(state, params)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    densities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 28.0, 40.0]
    flux, temperature = [], []
    for upstream in densities:
        diagnostics = _relax(upstream)
        flux.append(float(diagnostics.target_ion_flux))
        temperature.append(float(diagnostics.target_temperature_ev))
        print(f"n_up={upstream:5.1f}  target flux={flux[-1]:.4f}  Te_target={temperature[-1]:.3f} eV")

    peak_index = int(np.argmax(flux))
    summary = {
        "upstream_density": densities,
        "target_flux": flux,
        "target_temperature_ev": temperature,
        "rollover_density": densities[peak_index],
        "peak_flux": flux[peak_index],
        "detached_flux_reduction": 1.0 - flux[-1] / flux[peak_index],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(densities, flux, "o-", color="#1f77b4")
    axes[0].axvline(densities[peak_index], color="gray", ls=":", alpha=0.6)
    axes[0].annotate("rollover", (densities[peak_index], flux[peak_index]),
                     textcoords="offset points", xytext=(10, -4), fontsize=9)
    axes[0].set_xlabel("upstream density (normalized)")
    axes[0].set_ylabel("target ion flux  n c_s")
    axes[0].set_title("Target-flux rollover")
    axes[0].grid(True, ls=":", alpha=0.4)

    axes[1].semilogy(densities, temperature, "o-", color="#d62728")
    axes[1].axhline(1.0, color="gray", ls=":", alpha=0.6)
    axes[1].annotate("1 eV (recombination)", (densities[-1], 1.0),
                     textcoords="offset points", xytext=(-140, 4), fontsize=8)
    axes[1].set_xlabel("upstream density (normalized)")
    axes[1].set_ylabel("target temperature (eV)")
    axes[1].set_title("Self-consistent target cooling")
    axes[1].grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("B6 detachment: target-flux rollover and cooling into the recombining regime")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "b6_detachment.png", dpi=180)
    plt.close(fig)
    print(f"rollover at upstream density {densities[peak_index]}; "
          f"detached flux reduced {summary['detached_flux_reduction']*100:.0f}%")
    print(f"wrote {OUTPUT_DIR / 'b6_detachment.png'}")


if __name__ == "__main__":
    main()
