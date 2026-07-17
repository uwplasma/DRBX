"""Linear dispersion benchmarks B2 and B3 for the DRB models.

Diagonalize the reduced drift-wave and shear-Alfven operators from
``dkx.linear`` and compare the eigenfrequencies (and, for the drift wave,
the growth rate) against the analytic dispersion relations. Edit the constants
below and run:

    PYTHONPATH=src python examples/benchmarks/linear_dispersion.py

It prints the B2/B3 error summaries and writes ``linear_dispersion.png`` and
``linear_dispersion.json`` under ``output/linear_dispersion/`` (relative to the
current working directory).

References: shear-Alfven with electron inertia -- Stegmeir et al.,
Phys. Plasmas 26, 052517 (2019); resistive drift wave (Hasegawa-Wakatani) --
Dudson et al., Comput. Phys. Commun. 180, 1467 (2009).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dkx.linear import (  # noqa: E402
    drift_wave_adiabatic_frequency,
    eigenmodes,
    resistive_drift_wave_operator,
    shear_alfven_frequency,
    shear_alfven_operator,
)

# --- PARAMETERS -----------------------------------------------------------------
ALFVEN_SPEED = 3.2e6       # m/s
ELECTRON_SKIN_DEPTH = 1.1e-3   # m
K_PAR = 50.0               # 1/m, fixed parallel wavenumber for the Alfven scan
K_PERP_SCAN = np.linspace(0.0, 1500.0, 40)     # 1/m
DRIFT_KY = 0.5             # normalized poloidal wavenumber
DRIFT_KPERP2 = 0.5**2 + 0.3**2
DRIFT_KAPPA = 1.0          # density-gradient drive
ADIABATICITY_SCAN = np.geomspace(0.1, 100.0, 40)
OUTPUT_DIR = Path("output/linear_dispersion")   # artifact directory (cwd-relative)


def _mode(operator):
    """Dominant growth rate and |frequency| of one linear operator."""

    modes = eigenmodes(operator)
    frequency = float(np.max(np.abs(np.asarray(modes.frequencies))))
    return float(modes.dominant_growth_rate), frequency


# --- B3: shear-Alfven frequency vs k_perp -----------------------------------------
print(f"B3: scanning {len(K_PERP_SCAN)} k_perp points of the shear-Alfven operator...")
alfven_numeric, alfven_analytic = [], []
for k_perp in K_PERP_SCAN:
    _, freq = _mode(shear_alfven_operator(K_PAR, k_perp, ALFVEN_SPEED, ELECTRON_SKIN_DEPTH))
    alfven_numeric.append(freq)
    alfven_analytic.append(float(shear_alfven_frequency(K_PAR, k_perp, ALFVEN_SPEED, ELECTRON_SKIN_DEPTH)))

# --- B2: drift-wave growth rate and frequency vs adiabaticity ---------------------
print(f"B2: scanning {len(ADIABATICITY_SCAN)} adiabaticity points of the drift-wave operator...")
drift_growth, drift_freq = [], []
for alpha in ADIABATICITY_SCAN:
    growth, freq = _mode(resistive_drift_wave_operator(DRIFT_KY, DRIFT_KPERP2, alpha, DRIFT_KAPPA))
    drift_growth.append(growth)
    drift_freq.append(freq)
omega_star = float(drift_wave_adiabatic_frequency(DRIFT_KY, DRIFT_KPERP2, DRIFT_KAPPA))

alfven_rel_err = float(np.max(np.abs(np.array(alfven_numeric) - np.array(alfven_analytic))
                              / np.array(alfven_analytic)))
print(f"B3 shear-Alfven: max relative frequency error vs analytic = {alfven_rel_err:.2e}")
print(f"B2 drift wave: peak growth rate = {max(drift_growth):.3e}, "
      f"omega_star = {omega_star:.3e}")

# --- save the JSON summary --------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "linear_dispersion.json").write_text(json.dumps({
    "alfven": {"k_perp": K_PERP_SCAN.tolist(), "numeric": alfven_numeric,
               "analytic": alfven_analytic, "max_rel_error": alfven_rel_err},
    "drift_wave": {"adiabaticity": ADIABATICITY_SCAN.tolist(), "growth": drift_growth,
                   "frequency": drift_freq, "omega_star": omega_star},
}, indent=2))
print(f"wrote {OUTPUT_DIR / 'linear_dispersion.json'}")

# --- two-panel benchmark figure ---------------------------------------------------
fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(11.0, 4.4))
ax_a.plot(K_PERP_SCAN * ELECTRON_SKIN_DEPTH, np.array(alfven_analytic) / 1e6, "k-", label="analytic")
ax_a.plot(K_PERP_SCAN * ELECTRON_SKIN_DEPTH, np.array(alfven_numeric) / 1e6, "o", ms=3, label="linear solver")
ax_a.set_xlabel(r"$k_\perp d_e$")
ax_a.set_ylabel(r"$\omega$ [Mrad/s]")
ax_a.set_title("B3: shear-Alfven dispersion")
ax_a.legend()
ax_a.grid(alpha=0.3)

ax_d.semilogx(ADIABATICITY_SCAN, drift_growth, "o-", ms=3, label="growth rate")
ax_d.semilogx(ADIABATICITY_SCAN, drift_freq, "s-", ms=3, label="frequency")
ax_d.axhline(omega_star, color="k", ls="--", label=r"$\omega_\ast$ (adiabatic)")
ax_d.set_xlabel(r"adiabaticity $\alpha$")
ax_d.set_ylabel("normalized rate")
ax_d.set_title("B2: resistive drift wave")
ax_d.legend()
ax_d.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "linear_dispersion.png", dpi=200)
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'linear_dispersion.png'}")
