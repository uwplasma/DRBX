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

writes ``output/b6_detachment/`` with a composite PNG, a GIF of the leg as the
upstream density rises, and a JSON summary (about two minutes on a laptop CPU).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import io  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from PIL import Image  # noqa: E402

from drbx.native.neutrals import (  # noqa: E402
    DetachmentSolParameters,
    detachment_diagnostics,
    detachment_ledger,
    detachment_sol_run,
)
from drbx.native.neutrals.detachment_sol_model import sd1d_grid  # noqa: E402

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
EXTRA_N_UP = (2.25e19, 2.75e19, 3.5e19, 4.5e19, 5.5e19, 6.75e19)  # movie frames without an SD1D reference
OUTPUT_DIR = Path("output/b6_detachment")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("== B6 SD1D 13.6eV scan ==")
scan = sorted([(n, t, f) for n, t, f in SD1D] + [(n, None, None) for n in EXTRA_N_UP])
rows, previous = [], None
for n_up, t_ref, flux_ref in scan:
    params = DetachmentSolParameters(ny=NY, upstream_density=n_up, ionisation_energy=IONISATION_ENERGY)
    start = time.perf_counter()
    result = detachment_sol_run(params, previous.state if previous else None,
                                source_scale=previous.source_scale if previous else 0.1)
    previous = result
    diag = detachment_diagnostics(result.state, params, result.source_scale)
    ledger = detachment_ledger(result, params)
    t_t, flux = float(diag.target_temperature_ev), float(diag.target_ion_flux_area)
    st = result.state
    profile = dict(te=np.asarray(0.5 * st.plasma_pressure / st.ion_density) * params.Tnorm,
                   ni=np.asarray(st.ion_density) * params.Nnorm, nn=np.asarray(st.neutral_density) * params.Nnorm)
    rows.append(dict(n_up=n_up, T_t=t_t, T_t_sd1d=t_ref, flux_area=flux, flux_area_sd1d=flux_ref,
                     mach=float(diag.target_mach), residual=result.residual, iterations=result.iterations,
                     particle_imbalance=ledger["particle_imbalance"], power_imbalance=ledger["power_imbalance"],
                     profile=profile))
    msg = f"   n_up={n_up:.3e}: {result.iterations:3d} its, residual {result.residual:.1e}, {time.perf_counter() - start:5.1f}s | T_t {t_t:6.2f} eV"
    if t_ref is not None:
        msg += (f" (SD1D {t_ref:6.2f}, {100 * (t_t / t_ref - 1):+5.2f}%) | Gamma_t A {flux:.3e} (SD1D {flux_ref:.3e}, "
                f"{100 * (flux / flux_ref - 1):+5.2f}%)")
    print(msg + f" | ledgers {ledger['particle_imbalance']:.0e}/{ledger['power_imbalance']:.0e}")

(OUTPUT_DIR / "summary.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "profile"} for r in rows], indent=2))
print(f"wrote {OUTPUT_DIR / 'summary.json'}")

# ---- composite figure: the flared leg (half-width ~ sqrt(flux-tube area)) coloured by the steady profiles ----
_, area, length = sd1d_grid(DetachmentSolParameters(ny=NY, ionisation_energy=IONISATION_ENERGY))
xlog = np.log10(length[-1] / (length[-1] - length + 0.004))   # log-stretched axis: 0 = upstream, 3.9 = target
radius = np.sqrt(area / area[0])
frac = np.linspace(-1.0, 1.0, 9)
X2, Y2 = np.meshgrid(xlog, frac)
Y2 = Y2 * radius[None, :]
FIELDS = (("te", r"$T_e$ (eV)", "inferno", colors.LogNorm(3.0, 60.0)),
          ("ni", r"$n_i$ (m$^{-3}$)", "viridis", colors.LogNorm(1e19, 1e21)),
          ("nn", r"$n_n$ (m$^{-3}$)", "magma", colors.LogNorm(1e17, 2e21)))
TICKS = {"te": [3, 10, 60], "ni": [1e19, 1e20, 1e21], "nn": [1e17, 1e19, 2e21]}
TICKLABELS = {"te": ["3", "10", "60"], "ni": [r"$10^{19}$", r"$10^{20}$", r"$10^{21}$"], "nn": [r"$10^{17}$", r"$10^{19}$", r"$2\!\times\!10^{21}$"]}
n_all = np.array([r["n_up"] for r in rows]) / 1e19
ref_n = np.array([r["n_up"] for r in rows if r["T_t_sd1d"] is not None]) / 1e19


def draw_frame(k):
    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(9.0, 5.6))
    for i, (key, label, cmap, norm) in enumerate(FIELDS):
        ax = fig.add_axes([0.09, 0.69 - 0.27 * i, 0.38, 0.2])
        vals = np.tile(rows[k]["profile"][key], (len(frac), 1))
        mesh = ax.pcolormesh(X2, Y2, vals, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
        ax.plot(xlog, radius, "k", lw=1.0); ax.plot(xlog, -radius, "k", lw=1.0)
        ax.set_xlim(0, xlog[-1]); ax.set_ylim(-1.15 * radius.max(), 1.15 * radius.max()); ax.set_yticks([])
        ax.set_xticks(np.log10(30.0 / (np.array([10.0, 1.0, 0.1, 0.01]) + 0.004)), ["10", "1", "0.1", "0.01"])
        ax.set_ylabel("leg", fontsize=10)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("distance to the target [m] (log scale)")
        cb = fig.colorbar(mesh, cax=fig.add_axes([0.50, 0.71 - 0.27 * i, 0.013, 0.16]))
        cb.set_ticks(TICKS[key]); cb.set_ticklabels(TICKLABELS[key]); cb.ax.minorticks_off()
        ax.set_title(label, fontsize=10, loc="left")
    for j, (kk, lab, col) in enumerate((("T_t", "target temperature (eV)", "#1f77b4"), ("flux_area", r"target ion flux $\Gamma_t A_t$ (m$^{-2}$ s$^{-1}$)", "#d62728"))):
        ax = fig.add_axes([0.70, 0.58 - 0.43 * j, 0.28, 0.31])
        ax.plot(n_all, [r[kk] for r in rows], "-", color=col, lw=1.6, label="DRBX")
        ax.plot(ref_n, [r[kk + "_sd1d" if kk == "T_t" else "flux_area_sd1d"] for r in rows if r["T_t_sd1d"] is not None],
                "o", mfc="none", color="k", ms=6, label="SD1D (Dudson 2019)")
        ax.plot(n_all[k], rows[k][kk], "o", color=col, ms=9, zorder=5)
        ax.set_ylabel(lab, fontsize=9); ax.grid(True, ls=":", alpha=0.4)
        if j == 0:
            ax.legend(fontsize=9)
        else:
            ax.set_xlabel(r"upstream density ($10^{19}$ m$^{-3}$)")
    fig.suptitle(rf"Divertor leg: $n_{{up}}$ = {n_all[k]:.2f}$\times10^{{19}}$ m$^{{-3}}$, target $T_t$ = {rows[k]['T_t']:.1f} eV "
                 "(1D, hydrogen only)", fontsize=11)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return Image.open(buf).convert("RGB")


frames = [draw_frame(k) for k in range(len(rows))]
frames[-1].quantize(128).save(OUTPUT_DIR / "b6_detachment.png", optimize=True)
quant = [f.quantize(64, dither=Image.Dither.NONE) for f in frames]
quant[0].save(OUTPUT_DIR / "b6_detachment.gif", save_all=True, append_images=quant[1:], duration=450, loop=0, optimize=True)
print(f"wrote {OUTPUT_DIR / 'b6_detachment.png'} and b6_detachment.gif")
