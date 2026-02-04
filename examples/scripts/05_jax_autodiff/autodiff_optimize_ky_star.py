"""
autodiff_optimize_ky_star.py

Purpose
-------
Showcase a practical advantage of using JAX: **autodiff-based optimization** of a stability
objective.

Many SOL/edge workflows use the transport proxy:

  (gamma/ky)_max

which is normally obtained by scanning a discrete ky grid.

Here we demonstrate a *continuous* alternative:
- define ky = exp(log_ky) (so ky > 0),
- estimate gamma(ky) using a differentiable initial-value solver,
- maximize max(gamma,0)/ky using gradient-based optimization (optax).

This is not meant to replace careful ky scans in production, but it illustrates how JAX makes
parameter sensitivity and optimization workflows straightforward.

Run
---
  python examples/scripts/05_jax_autodiff/autodiff_optimize_ky_star.py

Outputs
-------
Written to `out/examples/05_jax_autodiff/autodiff_optimize_ky_star/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.growthrate import estimate_growth_rate_jax
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/examples/05_jax_autodiff/autodiff_optimize_ky_star")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # Keep the system small so the optimization is fast.
    nl = 32
    geom = SlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.6, curvature0=0.25)

    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    kx = 0.0
    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)

    def ratio_from_logky(log_ky: jax.Array) -> jax.Array:
        ky = jnp.exp(log_ky)
        matvec = linear_matvec(y_eq, params, geom, kx=kx, ky=ky)
        gr = estimate_growth_rate_jax(matvec, v0, tmax=18.0, dt0=0.02, nsave=120, fit_window=0.5)
        return jnp.maximum(gr.gamma, 0.0) / ky

    def loss(log_ky: jax.Array) -> jax.Array:
        return -ratio_from_logky(log_ky)

    # Optax optimizer over a single scalar.
    opt = optax.adam(learning_rate=0.25)
    log_ky = jnp.log(jnp.array(0.3))
    opt_state = opt.init(log_ky)

    history = []
    print("Optimizing ky to maximize max(gamma,0)/ky…", flush=True)
    for it in range(20):
        val, g = jax.value_and_grad(loss)(log_ky)
        updates, opt_state = opt.update(g, opt_state)
        log_ky = optax.apply_updates(log_ky, updates)

        ky_val = float(jnp.exp(log_ky))
        ratio_val = float(-val)
        history.append((it, ky_val, ratio_val, float(g)))
        print(
            f"[{it:02d}] ky={ky_val:.4f}  (gamma/ky)~{ratio_val:.4e}  dL/dlogky={float(g):+.3e}",
            flush=True,
        )

    ky_opt = float(jnp.exp(log_ky))
    ratio_opt = float(ratio_from_logky(log_ky))

    # Compare against a coarse brute-force scan.
    ky_grid = np.linspace(0.05, 1.2, 24)
    scan = scan_ky(
        params,
        geom,
        ky=ky_grid,
        kx=kx,
        arnoldi_m=30,
        arnoldi_tol=2e-3,
        nev=6,
        do_initial_value=True,
        tmax=18.0,
        dt0=0.02,
        nsave=120,
        verbose=True,
        print_every=4,
        seed=0,
    )
    ratio_scan = np.maximum(scan.gamma_iv, 0.0) / scan.ky  # type: ignore[arg-type]
    ky_star_scan = float(scan.ky[int(np.argmax(ratio_scan))])

    import matplotlib.pyplot as plt

    hist = np.asarray(history, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.plot(scan.ky, ratio_scan, "o-", label="coarse scan (init-value)")
    ax.axvline(
        ky_star_scan,
        color="k",
        alpha=0.25,
        linestyle="--",
        label=rf"scan $k_y^*={ky_star_scan:.3f}$",
    )
    ax.axvline(ky_opt, color="C3", alpha=0.8, linestyle="-", label=rf"opt $k_y={ky_opt:.3f}$")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max(\gamma,0)/k_y$")
    ax.set_title(r"Transport proxy: scan vs autodiff optimization")
    ax.legend()
    fig.savefig(out_dir / "ratio_scan_vs_opt.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    axs[0].plot(hist[:, 0], hist[:, 1], "o-")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel(r"$k_y$")
    axs[0].set_title("Optimized ky trajectory")

    axs[1].plot(hist[:, 0], hist[:, 2], "o-")
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel(r"$\max(\gamma,0)/k_y$")
    axs[1].set_title("Objective value")
    fig.savefig(out_dir / "optimization_history.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        history=hist,
        ky_grid=scan.ky,
        ratio_scan=ratio_scan,
        ky_opt=ky_opt,
        ratio_opt=ratio_opt,
    )
    save_json(
        out_dir / "summary.json",
        {
            "ky_opt": ky_opt,
            "ratio_opt": ratio_opt,
            "ky_star_scan": ky_star_scan,
            "ratio_star_scan": float(np.max(ratio_scan)),
            "note": "ratio computed from initial-value growth-rate estimator; values are qualitative.",
        },
    )
    save_json(
        out_dir / "params.json",
        {
            "geom": {
                "type": "slab",
                "nl": nl,
                "shat": float(geom.shat),
                "curvature0": float(geom.curvature0),
            },
            "kx": kx,
            "params": params.__dict__,
            "optax": {"optimizer": "adam", "learning_rate": 0.25, "n_steps": 20, "init_ky": 0.3},
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
