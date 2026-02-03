from __future__ import annotations

"""
07_halpern2013_salpha_ideal_ballooning_map.py

Purpose
-------
Reproduce the *ideal-ballooning* s–alpha diagram from:

  F. D. Halpern et al., "Ideal ballooning modes in the tokamak scrape-off layer",
  Phys. Plasmas 20, 052306 (2013).

In particular, Halpern et al. derive a reduced Sturm–Liouville eigenproblem (their Eq. (16))
for the ideal-ballooning growth rate ĉ as a function of magnetic shear ŝ and ballooning
parameter α. This script computes that diagram using the implementation in:

  `src/jaxdrb/linear/ideal_ballooning.py`

This example is intentionally *separate* from the drift-reduced Braginskii models: it is an
ideal-MHD-like benchmark that is cheap to scan and closely aligned with the published figure.

Run
---
  python examples/3_advanced/07_halpern2013_salpha_ideal_ballooning_map.py

Environment knobs
-----------------
Set `JAXDRB_FAST=0` for a finer grid (slower).

Outputs
-------
Written to `out/3_advanced/halpern2013_salpha_ideal_ballooning_map/`:

  - `halpern2013_fig1_like_panel.png`: main diagram + marginal curve
  - `results.npz`: (ŝ, α, ĉ) arrays
"""

import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.linear.ideal_ballooning import ideal_ballooning_gamma_hat


def main() -> None:
    out_dir = Path("out/3_advanced/halpern2013_salpha_ideal_ballooning_map")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    if fast:
        print("JAXDRB_FAST=1: using a coarse grid for a quick, pedagogic run.", flush=True)
    else:
        print("JAXDRB_FAST=0: using a finer grid (may take a while).", flush=True)

    # Halpern 2013 uses a finite connection-length domain L_h = 2π with Dirichlet boundary conditions.
    Lh = float(2 * np.pi)
    nh = 129 if fast else 257  # interior size ~ nh-2

    # Scan ranges chosen to match the qualitative extent of Halpern Fig. 1.
    shat = np.linspace(-2.0, 2.0, 61 if fast else 161)
    alpha = np.linspace(0.0, 2.5, 71 if fast else 201)

    # NOTE: `eigh_tridiagonal` is batched through vmap.
    def compute_map(shat_grid: jnp.ndarray, alpha_grid: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda s: jax.vmap(lambda a: ideal_ballooning_gamma_hat(shat=s, alpha=a, Lh=Lh, nh=nh))(
                alpha_grid
            )
        )(shat_grid)

    c_hat_map = jax.jit(compute_map)

    print(f"Computing ĉ(ŝ, α) on a {shat.size}×{alpha.size} grid…", flush=True)
    c_hat = np.asarray(c_hat_map(jnp.asarray(shat), jnp.asarray(alpha)))

    # Marginal curve: for each ŝ, pick the smallest α with ĉ>0 (coarse, grid-based estimate).
    alpha_crit = np.full_like(shat, np.nan, dtype=float)
    for i in range(shat.size):
        idx = np.where(c_hat[i, :] > 0.0)[0]
        if idx.size:
            alpha_crit[i] = float(alpha[int(idx[0])])

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(alpha, shat, c_hat, shading="auto", cmap="magma")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\hat{s}$")
    ax.set_title(r"Ideal-ballooning growth rate $\hat{c}(\hat{s},\alpha)$ (Halpern 2013 Eq. 16)")
    cb = fig.colorbar(im, ax=ax, label=r"$\hat{c}$")
    cb.ax.set_ylim(bottom=0.0)

    # Overlay marginal stability curve (coarse).
    ax.plot(alpha_crit, shat, "w--", lw=1.6, label=r"marginal ($\hat{c}>0$)")
    ax.legend(loc="lower right", frameon=True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(shat, alpha_crit, "o-", ms=3)
    ax2.set_xlabel(r"$\hat{s}$")
    ax2.set_ylabel(r"$\alpha_{\mathrm{crit}}$ (grid estimate)")
    ax2.set_title("Marginal stability curve")
    ax2.grid(True, alpha=0.35)

    fig.savefig(out_dir / "halpern2013_fig1_like_panel.png", dpi=240)
    plt.close(fig)

    np.savez(out_dir / "results.npz", shat=shat, alpha=alpha, c_hat=c_hat, alpha_crit=alpha_crit)
    save_json(
        out_dir / "params.json",
        {
            "paper": "Halpern et al. (2013) Phys. Plasmas 20, 052306",
            "equation": "Eq. (16) ideal-ballooning Sturm–Liouville problem (Dirichlet BCs)",
            "Lh": Lh,
            "nh": nh,
            "shat": {"min": float(shat.min()), "max": float(shat.max()), "n": int(shat.size)},
            "alpha": {"min": float(alpha.min()), "max": float(alpha.max()), "n": int(alpha.size)},
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
