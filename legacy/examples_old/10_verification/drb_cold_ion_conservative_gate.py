"""
drb_cold_ion_conservative_gate.py

Purpose
-------
Run a strict conservation check on the **actual field-line DRB equations** (cold-ion model)
in a periodic conservative subset, and produce a reviewer-friendly figure.

What is checked
---------------
For `omega_n=omega_Te=0`, no curvature, no diffusion/sinks/sheath, and finite `me_hat`,
the test tracks:

- quadratic DRB energy functional,
- mean density `<n>` (mass proxy),
- mean vorticity `<omega>` (charge proxy),
- mean parallel current `<j_par>`,
- mean parallel momentum `<v_i + m_e v_e>`.

Run
---
  python examples/10_verification/drb_cold_ion_conservative_gate.py

Output
------
Written to `out/examples/10_verification/drb_cold_ion_conservative_gate/`:

- `drb_cold_ion_conservative_gate.png`
- `history.npz`
- `params.json`
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
from jaxdrb.models.invariants import cold_ion_invariants
from jaxdrb.models.params import DRBParams
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


@dataclass(frozen=True)
class RunCfg:
    nl: int = 64
    kx: float = 0.0
    ky: float = 0.35
    dt: float = 1.0e-3
    nsteps: int = 4000
    seed: int = 101
    amplitude: float = 1.0e-2


def main() -> None:
    cfg = RunCfg()
    out_dir = Path("out/examples/10_verification/drb_cold_ion_conservative_gate")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_mpl_style()
    geom = SlabGeometry.make(nl=cfg.nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(cfg.nl, n0=1.0, Te0=1.0)
    params = DRBParams(
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

    y0 = State.random(jax.random.key(cfg.seed), cfg.nl, amplitude=cfg.amplitude)

    def rhs_local(t: jnp.ndarray, y: State) -> State:
        return rhs_nonlinear(t, y, params, geom, kx=cfg.kx, ky=cfg.ky, eq=eq)

    keys = ("energy", "mass", "charge", "current", "momentum")

    def evolve(y_init: State) -> tuple[jnp.ndarray, jnp.ndarray, State]:
        inv0 = cold_ion_invariants(y_init, params=params, geom=geom, kx=cfg.kx, ky=cfg.ky, eq=eq)
        vec0 = jnp.array([inv0[k] for k in keys], dtype=jnp.float64)

        ys, y_end = diffeqsolve_fixed_steps(
            rhs_local,
            y0=y_init,
            t0=0.0,
            dt=cfg.dt,
            nsteps=cfg.nsteps,
            solver="dopri5",
        )
        vecs = jax.vmap(
            lambda y: jnp.array(
                [
                    cold_ion_invariants(y, params=params, geom=geom, kx=cfg.kx, ky=cfg.ky, eq=eq)[k]
                    for k in keys
                ],
                dtype=jnp.float64,
            )
        )(ys)
        full = jnp.vstack([vec0[None, :], vecs])
        t = cfg.dt * jnp.arange(cfg.nsteps + 1, dtype=jnp.float64)
        return t, full, y_end

    print("Integrating periodic conservative cold-ion DRB subset…", flush=True)
    t, H, y_end = evolve(y0)
    t = np.asarray(t)
    H = np.asarray(H)

    E = H[:, 0]
    E0 = E[0]
    relE = (E - E0) / max(abs(E0), 1e-30)
    spans = np.max(np.abs(H - H[0:1, :]), axis=0)

    print(
        "Max absolute drifts:",
        {k: float(v) for k, v in zip(keys, spans, strict=False)},
        flush=True,
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), constrained_layout=True)
    ax = axes.ravel()
    ax[0].plot(t, relE, lw=2.0, color="#1f77b4")
    ax[0].set_title(r"Relative energy drift $(E-E_0)/E_0$")
    ax[0].set_xlabel("t")
    ax[0].grid(alpha=0.25)

    for i, key in enumerate(keys[1:], start=1):
        drift = np.abs(H[:, i] - H[0, i]) + 1e-30
        ax[i].semilogy(t, drift, lw=1.8)
        ax[i].set_title(f"|{key} drift|")
        ax[i].set_xlabel("t")
        ax[i].grid(alpha=0.25)

    # Final panel: show state amplitudes as a quick mode-shape snapshot.
    l = np.asarray(geom.l)
    y_end = State(
        n=np.asarray(y_end.n),
        omega=np.asarray(y_end.omega),
        vpar_e=np.asarray(y_end.vpar_e),
        vpar_i=np.asarray(y_end.vpar_i),
        Te=np.asarray(y_end.Te),
    )
    ax[5].plot(l, np.abs(y0.n), "--", label="|n| t=0")
    ax[5].plot(l, np.abs(y_end.n), label="|n| t_end")
    ax[5].plot(l, np.abs(y0.omega), "--", label="|omega| t=0")
    ax[5].plot(l, np.abs(y_end.omega), label="|omega| t_end")
    ax[5].set_title("State amplitude reference")
    ax[5].set_xlabel("l")
    ax[5].legend(fontsize=8, ncol=2)
    ax[5].grid(alpha=0.2)

    fig.suptitle("Cold-ion DRB conservative gate (periodic subset)", fontsize=13)
    fig.savefig(out_dir / "drb_cold_ion_conservative_gate.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "history.npz",
        t=t,
        keys=np.asarray(keys, dtype=object),
        values=H,
    )
    save_json(
        out_dir / "params.json",
        {
            "nl": cfg.nl,
            "kx": cfg.kx,
            "ky": cfg.ky,
            "dt": cfg.dt,
            "nsteps": cfg.nsteps,
            "seed": cfg.seed,
            "params": params.__dict__,
            "max_abs_drift": {k: float(v) for k, v in zip(keys, spans, strict=False)},
            "relative_energy_span": float((np.max(E) - np.min(E)) / max(abs(E0), 1e-30)),
        },
    )
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
