"""B6: SD1D 1D detachment benchmark -- the ``13.6eV`` upstream-density scan.

Solves the SD1D-matched plasma--neutral model (``drbx.native.neutrals.
detachment_sol_model``) to steady state along an upstream-density scan and
compares the target temperature and the target ion flux with the published
SD1D simulations of Dudson et al., PPCF 61, 065008 (2019), ``13.6eV`` scan:
hydrogen only, a fixed 13.6 eV cost per ionisation, no excitation or impurity
radiation, a 30 m leg with the area doubling between the X-point (10 m) and the
target, 5e7 W/m^2 and the particle source over the first 10 m, 99% recycling,
and 800 cells. The SD1D values are the final states of the published dataset.

In this hydrogen-only scan the target cools from ~29 eV to ~3 eV and the target
flux keeps rising: there is no flux rollover without impurity radiation and
excitation (the paper's baseline), which this model does not include.

For each density the script prints the Newton/pseudo-transient iterations, the
converged steady residual, the particle and power ledgers, and the comparison.

Run:

    PYTHONPATH=src python examples/benchmarks/b6_detachment_sd1d.py

writes ``output/b6_detachment/`` with a two-panel PNG and a JSON summary
(about two minutes on a laptop CPU).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drbx.native.neutrals import (  # noqa: E402
    DetachmentSolParameters,
    detachment_diagnostics,
    detachment_ledger,
    detachment_sol_run,
)

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------
NY = 800                 # SD1D resolution (nonuniform, 7 cm upstream -> 4 mm at the target)
IONISATION_ENERGY = 13.6  # eV lost per ionisation (SD1D Eionize)
# SD1D 13.6eV scan: (achieved n_up [m^-3], last-cell T_t [eV], Gamma_t A_t [m^-2 s^-1])
SD1D = [
    (1.699818e19, 29.35, 1.640e24), (2.000355e19, 21.30, 2.231e24), (2.501205e19, 13.50, 3.427e24),
    (2.999828e19, 9.58, 4.691e24), (3.992420e19, 6.22, 6.861e24), (5.000658e19, 4.80, 8.460e24),
    (5.999908e19, 4.07, 9.577e24), (7.521918e19, 3.13, 1.102e25),
]
# The last SD1D point is not a steady state (no PI integral in its restart file,
# earlier simulation time than the 7e19 run it continues from, residual 100x the
# other cases), so its ~10% T_t difference is not a model difference.
OUTPUT_DIR = Path("output/b6_detachment")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("== B6 SD1D 13.6eV scan ==")
rows, previous = [], None
for n_up, t_ref, flux_ref in SD1D:
    params = DetachmentSolParameters(ny=NY, upstream_density=n_up, ionisation_energy=IONISATION_ENERGY)
    start = time.perf_counter()
    result = detachment_sol_run(params, previous.state if previous else None,
                                source_scale=previous.source_scale if previous else 0.1)
    previous = result
    diag = detachment_diagnostics(result.state, params, result.source_scale)
    ledger = detachment_ledger(result, params)
    t_t, flux = float(diag.target_temperature_ev), float(diag.target_ion_flux_area)
    rows.append(dict(n_up=n_up, T_t=t_t, T_t_sd1d=t_ref, flux_area=flux, flux_area_sd1d=flux_ref,
                     mach=float(diag.target_mach), residual=result.residual, iterations=result.iterations,
                     particle_imbalance=ledger["particle_imbalance"], power_imbalance=ledger["power_imbalance"]))
    print(f"   n_up={n_up:.3e}: {result.iterations:3d} its, residual {result.residual:.1e}, "
          f"{time.perf_counter() - start:5.1f}s | T_t {t_t:6.2f} eV (SD1D {t_ref:6.2f}, {100 * (t_t / t_ref - 1):+5.2f}%)"
          f" | Gamma_t A {flux:.3e} (SD1D {flux_ref:.3e}, {100 * (flux / flux_ref - 1):+5.2f}%)"
          f" | ledgers {ledger['particle_imbalance']:.0e}/{ledger['power_imbalance']:.0e}")

(OUTPUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
print(f"wrote {OUTPUT_DIR / 'summary.json'}")

n = np.array([r["n_up"] for r in rows]) / 1e19
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
axes[0].plot(n, [r["T_t"] for r in rows], "-", color="#1f77b4", label="DRBX (SD1D-matched, 800 cells)")
axes[0].plot(n, [r["T_t_sd1d"] for r in rows], "o", mfc="none", color="k", label="SD1D (Dudson 2019)")
axes[0].set_ylabel("target temperature (eV)")
axes[1].plot(n, [r["flux_area"] for r in rows], "-", color="#d62728", label="DRBX")
axes[1].plot(n, [r["flux_area_sd1d"] for r in rows], "o", mfc="none", color="k", label="SD1D")
axes[1].set_ylabel(r"target ion flux $\Gamma_t A_t$ (m$^{-2}$ s$^{-1}$)")
for ax in axes:
    ax.set_xlabel(r"upstream density ($10^{19}$ m$^{-3}$)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8)
fig.suptitle("B6: SD1D 13.6 eV scan (hydrogen only) -- steady target conditions")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "b6_detachment.png", dpi=160)
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'b6_detachment.png'}")
