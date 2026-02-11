#!/usr/bin/env python3
"""Nonlinear-toggle demonstration for field-line DRB closures.

This example compares three short time evolutions of the cold-ion field-line model:

1) equilibrium-based non-Boussinesq + equilibrium-based Braginskii coefficients,
2) nonlinear non-Boussinesq polarization using n0 + Re[n],
3) (2) + state-dependent Braginskii coefficients using Te0 + Re[Te].

The goal is pedagogic: show how the new nonlinear toggles modify dynamics while keeping
the code path stable, differentiable, and easy to switch on/off.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, phi_from_omega, rhs_nonlinear
from jaxdrb.models.params import DRBParams
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=str,
        default="out/examples/04_closures_transport/nonlinear_flux_tube_toggles",
        help="Output directory.",
    )
    p.add_argument("--nl", type=int, default=96)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--save-stride", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def energy_proxy(y: State) -> float:
    return float(
        0.5
        * jnp.mean(
            jnp.abs(y.n) ** 2
            + jnp.abs(y.omega) ** 2
            + jnp.abs(y.vpar_e) ** 2
            + jnp.abs(y.vpar_i) ** 2
            + jnp.abs(y.Te) ** 2
        )
    )


def integrate_case(
    *,
    y0: State,
    eq: Equilibrium,
    geom,
    params: DRBParams,
    dt: float,
    tmax: float,
    save_stride: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, State]:
    nsteps = int(np.ceil(tmax / dt))
    nchunks = max(1, nsteps // save_stride)
    ts = []
    Es = []
    y = y0
    t = 0.0

    def rhs(t_in: float, y_in: State) -> State:
        return rhs_nonlinear(t_in, y_in, params, geom, kx=0.0, ky=0.35, eq=eq)

    for k in range(nchunks):
        _, y = diffeqsolve_fixed_steps(
            rhs,
            y0=y,
            t0=t,
            dt=dt,
            nsteps=save_stride,
            solver="dopri5",
        )
        t += dt * save_stride
        ts.append(t)
        Es.append(energy_proxy(y))
        print(
            f"[nonlinear-toggles] {name:>20s} chunk {k+1:02d}/{nchunks} "
            f"t={t:6.3f} Eproxy={Es[-1]:9.3e}",
            flush=True,
        )

    return np.asarray(ts), np.asarray(Es), y


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()
    jax.config.update("jax_enable_x64", True)

    geom = SlabGeometry.make(nl=int(args.nl), length=float(2 * np.pi), shat=0.5, curvature0=0.2)
    eq = Equilibrium.constant(int(args.nl), n0=1.2, Te0=0.8)
    y0 = State.random(jax.random.PRNGKey(int(args.seed)), int(args.nl), amplitude=1e-3)

    common = dict(
        omega_n=0.6,
        omega_Te=0.4,
        eta=0.8,
        me_hat=0.1,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        boussinesq=False,
        kperp2_min=1e-6,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
        braginskii_on=True,
        braginskii_eta_on=True,
        braginskii_kappa_e_on=True,
        braginskii_visc_e_on=True,
        braginskii_visc_i_on=True,
        braginskii_Tref=1.0,
        braginskii_T_floor=1e-3,
        braginskii_T_smooth=1e-3,
        chi_par_Te=0.2,
        nu_par_e=0.1,
        nu_par_i=0.1,
    )

    cases = [
        (
            "equilibrium_coeffs",
            DRBParams(
                **common,
                non_boussinesq_perturbed_density_on=False,
                braginskii_state_dependent_on=False,
            ),
        ),
        (
            "nonbouss_state_density",
            DRBParams(
                **common,
                non_boussinesq_perturbed_density_on=True,
                braginskii_state_dependent_on=False,
            ),
        ),
        (
            "full_state_dependent",
            DRBParams(
                **common,
                non_boussinesq_perturbed_density_on=True,
                braginskii_state_dependent_on=True,
            ),
        ),
    ]

    traces = {}
    finals = {}
    for name, params in cases:
        t, e, y_end = integrate_case(
            y0=y0,
            eq=eq,
            geom=geom,
            params=params,
            dt=float(args.dt),
            tmax=float(args.tmax),
            save_stride=int(args.save_stride),
            name=name,
        )
        traces[name] = {"t": t, "E": e}
        finals[name] = y_end

    # Energy proxy panel.
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    ax = axs[0, 0]
    for name, tr in traces.items():
        ax.plot(tr["t"], tr["E"], label=name)
    ax.set_xlabel("t")
    ax.set_ylabel("energy proxy")
    ax.set_title("Short-run nonlinear evolution with closure toggles")
    ax.legend(fontsize=9)

    # Relative drift from baseline.
    base = traces["equilibrium_coeffs"]["E"]
    ax = axs[0, 1]
    for name, tr in traces.items():
        rel = (tr["E"] - base) / (np.maximum(np.abs(base), 1e-30))
        ax.plot(tr["t"], rel, label=name)
    ax.axhline(0.0, color="k", alpha=0.4, lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("relative vs baseline")
    ax.set_title("Toggle impact on energy proxy")

    # Final profiles for most nonlinear case.
    y_end = finals["full_state_dependent"]
    l = np.asarray(geom.l)
    ax = axs[1, 0]
    ax.plot(l, np.real(np.asarray(y_end.n)), label="Re[n]")
    ax.plot(l, np.real(np.asarray(y_end.Te)), label="Re[Te]")
    ax.plot(l, np.real(np.asarray(y_end.omega)), label="Re[omega]")
    ax.set_xlabel("l")
    ax.set_title("Final profiles (full state-dependent toggles)")
    ax.legend(fontsize=9)

    ax = axs[1, 1]
    k2 = np.asarray(geom.kperp2(0.0, 0.35))
    phi = phi_from_omega(
        y_end.omega,
        k2,
        kperp2_min=1e-6,
        boussinesq=False,
        n0=eq.n0,
        n0_min=1e-6,
        n=y_end.n,
        non_boussinesq_perturbed_density_on=True,
    )
    ax.plot(l, np.real(np.asarray(phi)), label="Re[phi]")
    ax.plot(l, np.real(np.asarray(y_end.vpar_e)), label="Re[vpar_e]")
    ax.plot(l, np.real(np.asarray(y_end.vpar_i)), label="Re[vpar_i]")
    ax.set_xlabel("l")
    ax.set_title("Derived potential and parallel flows")
    ax.legend(fontsize=9)

    fig.savefig(out_dir / "nonlinear_toggle_panel.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        l=np.asarray(geom.l),
        t=traces["equilibrium_coeffs"]["t"],
        E_equilibrium_coeffs=traces["equilibrium_coeffs"]["E"],
        E_nonbouss_state_density=traces["nonbouss_state_density"]["E"],
        E_full_state_dependent=traces["full_state_dependent"]["E"],
        n_end=np.asarray(finals["full_state_dependent"].n),
        omega_end=np.asarray(finals["full_state_dependent"].omega),
        Te_end=np.asarray(finals["full_state_dependent"].Te),
    )

    (out_dir / "params.json").write_text(
        json.dumps(
            {
                "dt": float(args.dt),
                "tmax": float(args.tmax),
                "save_stride": int(args.save_stride),
                "geometry": {"nl": int(args.nl), "shat": 0.5, "curvature0": 0.2},
                "equilibrium": {"n0": 1.2, "Te0": 0.8},
                "cases": [
                    {
                        "name": name,
                        "non_boussinesq_perturbed_density_on": params.non_boussinesq_perturbed_density_on,
                        "braginskii_state_dependent_on": params.braginskii_state_dependent_on,
                    }
                    for name, params in cases
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"[nonlinear-toggles] wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
