"""
drb_cold_ion_operator_gate.py

Purpose
-------
Run a strict *operator-level* conservation check on the periodic conservative subset of the
cold-ion DRB field-line model, and generate a reviewer-ready figure.

This complements `drb_cold_ion_conservative_gate.py` by validating the instantaneous RHS operator:
invariant rates computed from `dy = rhs_nonlinear(y)` should be at roundoff-level in the
conservative subset.

Run
---
  python examples/10_verification/drb_cold_ion_operator_gate.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.invariants import cold_ion_invariant_rates_from_rhs, cold_ion_invariants
from jaxdrb.models.params import DRBParams
from jaxdrb.nonlinear.stepper import rk4_step


@dataclass(frozen=True)
class Cfg:
    nl: int = 64
    dt: float = 1.0e-3
    nsteps: int = 2000
    amp: float = 1.0e-2
    nseeds: int = 8
    ky_min: float = 0.12
    ky_max: float = 0.72
    nky: int = 11
    ky_time: float = 0.35


def _params() -> DRBParams:
    return DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        alpha_Te_ohm=1.71,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=True,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )


def main() -> None:
    cfg = Cfg()
    out_dir = Path("out/examples/10_verification/drb_cold_ion_operator_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_mpl_style()

    geom = SlabGeometry.make(nl=cfg.nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(cfg.nl, n0=1.0, Te0=1.0)
    params = _params()
    ky_values = np.linspace(cfg.ky_min, cfg.ky_max, cfg.nky)
    seed_keys = jax.random.split(jax.random.key(0), cfg.nseeds)
    rate_names = ("denergy_dt", "dmass_dt", "dcharge_dt", "dcurrent_dt", "dmomentum_dt")

    print(
        f"Computing operator residuals on {cfg.nseeds} seeds x {cfg.nky} ky points...",
        flush=True,
    )
    rates = np.zeros((cfg.nseeds, cfg.nky, len(rate_names)), dtype=float)
    for i, key in enumerate(seed_keys):
        y = State.random(key, cfg.nl, amplitude=cfg.amp)
        for j, ky in enumerate(ky_values):
            dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=float(ky), eq=eq)
            rr = cold_ion_invariant_rates_from_rhs(
                y, dy, params=params, geom=geom, kx=0.0, ky=float(ky), eq=eq
            )
            rates[i, j, :] = [abs(float(rr[name])) for name in rate_names]
        print(f"  seed {i + 1:02d}/{cfg.nseeds} done", flush=True)

    print("Running finite-time invariant drift check...", flush=True)
    y0 = State.random(jax.random.key(123), cfg.nl, amplitude=cfg.amp)
    keys = ("energy", "mass", "charge", "current", "momentum")

    def rhs_local(t, y):
        return rhs_nonlinear(t, y, params, geom, kx=0.0, ky=cfg.ky_time, eq=eq)

    @jax.jit
    def evolve(y_init):
        inv0 = cold_ion_invariants(y_init, params=params, geom=geom, kx=0.0, ky=cfg.ky_time, eq=eq)
        vec0 = jnp.asarray([inv0[k] for k in keys], dtype=jnp.float64)

        def step(carry, _):
            t, y = carry
            y_next = rk4_step(y, t, cfg.dt, rhs_local)
            inv = cold_ion_invariants(
                y_next, params=params, geom=geom, kx=0.0, ky=cfg.ky_time, eq=eq
            )
            vec = jnp.asarray([inv[k] for k in keys], dtype=jnp.float64)
            return (t + cfg.dt, y_next), vec

        (_, _), hist = jax.lax.scan(
            step, (jnp.asarray(0.0, dtype=jnp.float64), y_init), xs=None, length=cfg.nsteps
        )
        t = cfg.dt * jnp.arange(cfg.nsteps + 1, dtype=jnp.float64)
        return t, jnp.vstack([vec0[None, :], hist])

    t, H = evolve(y0)
    t = np.asarray(t)
    H = np.asarray(H)
    E = H[:, 0]
    E0 = E[0]
    relE = (E - E0) / max(abs(E0), 1e-30)
    drifts = np.abs(H[:, 1:] - H[0:1, 1:])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)
    ax = axs.ravel()

    im0 = ax[0].imshow(
        np.log10(rates[:, :, 0] + 1e-30),
        origin="lower",
        aspect="auto",
        extent=[ky_values[0], ky_values[-1], 0, cfg.nseeds - 1],
        cmap="magma",
    )
    fig.colorbar(im0, ax=ax[0], pad=0.01, label=r"$\log_{10}|dE/dt|$")
    ax[0].set_title("Operator gate: energy-rate residual")
    ax[0].set_xlabel(r"$k_y$")
    ax[0].set_ylabel("seed index")

    mean_rates = np.max(rates[:, :, 1:], axis=2)
    im1 = ax[1].imshow(
        np.log10(mean_rates + 1e-30),
        origin="lower",
        aspect="auto",
        extent=[ky_values[0], ky_values[-1], 0, cfg.nseeds - 1],
        cmap="viridis",
    )
    fig.colorbar(im1, ax=ax[1], pad=0.01, label=r"$\log_{10}\max(|d\langle\cdot\rangle/dt|)$")
    ax[1].set_title("Operator gate: mean-rate residual")
    ax[1].set_xlabel(r"$k_y$")
    ax[1].set_ylabel("seed index")

    ax[2].plot(ky_values, np.max(rates[:, :, 0], axis=0), lw=2.0, label=r"$|dE/dt|$")
    for k, label in enumerate(("mass", "charge", "current", "momentum"), start=1):
        ax[2].plot(ky_values, np.max(rates[:, :, k], axis=0), lw=1.7, label=label)
    ax[2].set_yscale("log")
    ax[2].set_title("Max operator residual vs $k_y$")
    ax[2].set_xlabel(r"$k_y$")
    ax[2].grid(alpha=0.25)
    ax[2].legend(fontsize=8, ncol=2)

    ax[3].plot(t, relE, lw=2.0)
    ax[3].set_title(r"Finite-time: $(E-E_0)/E_0$")
    ax[3].set_xlabel("t")
    ax[3].grid(alpha=0.25)

    for i, label in enumerate(("mass", "charge", "current", "momentum")):
        ax[4].plot(t, drifts[:, i] + 1e-30, lw=1.7, label=label)
    ax[4].set_yscale("log")
    ax[4].set_title("Finite-time: |mean drifts|")
    ax[4].set_xlabel("t")
    ax[4].grid(alpha=0.25)
    ax[4].legend(fontsize=8, ncol=2)

    l = np.asarray(geom.l)
    y_ref = State(
        n=np.asarray(y0.n),
        omega=np.asarray(y0.omega),
        vpar_e=np.asarray(y0.vpar_e),
        vpar_i=np.asarray(y0.vpar_i),
        Te=np.asarray(y0.Te),
    )
    ax[5].plot(l, np.abs(y_ref.n), label="|n|")
    ax[5].plot(l, np.abs(y_ref.omega), label="|omega|")
    ax[5].plot(l, np.abs(y_ref.vpar_e), label="|vpar_e|")
    ax[5].plot(l, np.abs(y_ref.vpar_i), label="|vpar_i|")
    ax[5].plot(l, np.abs(y_ref.Te), label="|Te|")
    ax[5].set_title("Reference state amplitudes (seed)")
    ax[5].set_xlabel("l")
    ax[5].grid(alpha=0.2)
    ax[5].legend(fontsize=8, ncol=2)

    fig.suptitle("Cold-ion DRB strict conservative operator gate", fontsize=13)
    fig.savefig(out_dir / "drb_cold_ion_operator_gate.png", dpi=220)
    plt.close(fig)

    summary = {
        "max_abs_denergy_dt": float(np.max(rates[:, :, 0])),
        "max_abs_dmass_dt": float(np.max(rates[:, :, 1])),
        "max_abs_dcharge_dt": float(np.max(rates[:, :, 2])),
        "max_abs_dcurrent_dt": float(np.max(rates[:, :, 3])),
        "max_abs_dmomentum_dt": float(np.max(rates[:, :, 4])),
        "rel_energy_span": float((np.max(E) - np.min(E)) / max(abs(E0), 1e-30)),
        "rel_energy_end": float(abs(E[-1] - E0) / max(abs(E0), 1e-30)),
        "max_abs_mass_drift": float(np.max(drifts[:, 0])),
        "max_abs_charge_drift": float(np.max(drifts[:, 1])),
        "max_abs_current_drift": float(np.max(drifts[:, 2])),
        "max_abs_momentum_drift": float(np.max(drifts[:, 3])),
    }
    np.savez(
        out_dir / "results.npz",
        ky=ky_values,
        t=t,
        operator_rates=rates,
        invariants=H,
        rate_names=np.asarray(rate_names, dtype=object),
        invariant_names=np.asarray(keys, dtype=object),
    )
    save_json(out_dir / "params.json", {"cfg": cfg.__dict__, "params": params.__dict__, **summary})
    print("Summary:", summary, flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
