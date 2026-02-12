"""FCI DRB3D full branch: target/sheath coupling with hot-ion, EM, and neutrals.

This example demonstrates the promoted full-branch model with:

- target-aware open-field-line parallel derivatives,
- Loizu-style linearized sheath/plate closure (`sheath_bc_model="loizu_linear"`),
- hot-ion (`Ti`) and electromagnetic (`psi`) toggles,
- neutral density (`N`) exchange model.

It runs a short nonlinear trajectory and outputs a diagnostics panel suitable for docs/README.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.neutrals import NeutralParams


def _state_add(a: FCIDRB3DFullState, b: FCIDRB3DFullState) -> FCIDRB3DFullState:
    def _opt_add(x, y):
        if x is None and y is None:
            return None
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    return FCIDRB3DFullState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        Ti=_opt_add(a.Ti, b.Ti),
        psi=_opt_add(a.psi, b.psi),
        N=_opt_add(a.N, b.N),
    )


def _random_state(key: jax.Array, shape: tuple[int, int, int], amp: float) -> FCIDRB3DFullState:
    k = jax.random.split(key, 8)
    n0 = amp * jax.random.normal(k[0], shape)
    omega0 = amp * jax.random.normal(k[1], shape)
    vpe0 = amp * jax.random.normal(k[2], shape)
    vpi0 = amp * jax.random.normal(k[3], shape)
    Te0 = 0.18 + amp * jax.random.normal(k[4], shape)
    Ti0 = 0.16 + amp * jax.random.normal(k[5], shape)
    psi0 = amp * jax.random.normal(k[6], shape)
    N0 = 0.25 + amp * jax.random.normal(k[7], shape)
    return FCIDRB3DFullState(
        n=n0,
        omega=omega0,
        vpar_e=vpe0,
        vpar_i=vpi0,
        Te=Te0,
        Ti=Ti0,
        psi=psi0,
        N=N0,
    )


def make_model() -> tuple[FCIDRB3DFullModel, FCIDRB3DFullState]:
    grid = FCISlabGrid.make(
        nx=10,
        ny=10,
        nz=12,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=6.0,
        Bx=0.0,
        By=0.15,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.03,
        me_hat=0.3,
        Dn=3e-4,
        DOmega=3e-4,
        Dvpar=3e-4,
        DTe=3e-4,
        DTi=3e-4,
        Dpsi=3e-4,
        chi_par=4e-4,
        hot_ion_on=True,
        tau_i=0.7,
        em_on=True,
        beta=0.06,
        neutrals_on=True,
        neutrals=NeutralParams(
            enabled=True,
            Dn0=2e-4,
            S0=0.0,
            nu_sink=0.0,
            nu_ion=5e-3,
            nu_rec=3e-3,
            n_background=1.0,
            nu_cx_omega=0.0,
        ),
        sheath_on=True,
        sheath_bc_model="loizu_linear",
        sheath_nu_mom=0.5,
        sheath_nu_particle=0.18,
        sheath_nu_energy=0.1,
        sheath_gamma_e=3.2,
        sheath_gamma_i=3.0,
        sheath_delta=0.0,
        perp_operator="spectral",
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    y0 = _random_state(jax.random.key(42), (grid.nz, grid.nx, grid.ny), amp=5e-4)
    return model, y0


def _energy_for_norm(model: FCIDRB3DFullModel, y: FCIDRB3DFullState) -> float:
    y_norm = FCIDRB3DFullState(
        n=y.n,
        omega=y.omega,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
        Te=y.Te,
        Ti=None,
        psi=None,
        N=None,
    )
    return float(model.energy(y_norm))


def main() -> None:
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="out_fci_drb3d_multiphysics")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--nsteps", type=int, default=140)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, y0 = make_model()
    print("[fci-multiphysics] integrating ...")
    ys, y_end = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=float(args.dt),
        nsteps=int(args.nsteps),
        save_every=int(args.save_every),
        solver="dopri5",
    )

    ts = float(args.dt) * jnp.arange(
        int(args.save_every), int(args.nsteps) + 1, int(args.save_every)
    )
    nsave = int(ts.size)
    energy = []
    particle_plasma = []
    particle_total = []
    sheath_prate = []
    sheath_erate = []
    particle_rate_other = []
    advective_rate = []
    parallel_rate = []

    for i in range(nsave):
        yi = FCIDRB3DFullState(
            n=ys.n[i],
            omega=ys.omega[i],
            vpar_e=ys.vpar_e[i],
            vpar_i=ys.vpar_i[i],
            Te=ys.Te[i],
            Ti=None if ys.Ti is None else ys.Ti[i],
            psi=None if ys.psi is None else ys.psi[i],
            N=None if ys.N is None else ys.N[i],
        )
        pb = model.particle_budget_terms(yi)
        psh, esh = model.sheath_budget_rates(yi)
        energy.append(_energy_for_norm(model, yi))
        particle_plasma.append(float(model.particle_content(yi)))
        particle_total.append(float(model.total_particle_content(yi)))
        sheath_prate.append(float(psh))
        sheath_erate.append(float(esh))
        particle_rate_other.append(float(pb["other"]))
        advective_rate.append(float(pb["advective"]))
        parallel_rate.append(float(pb["parallel"]))

    energy = jnp.asarray(energy)
    particle_plasma = jnp.asarray(particle_plasma)
    particle_total = jnp.asarray(particle_total)
    sheath_prate = jnp.asarray(sheath_prate)
    sheath_erate = jnp.asarray(sheath_erate)
    particle_rate_other = jnp.asarray(particle_rate_other)
    advective_rate = jnp.asarray(advective_rate)
    parallel_rate = jnp.asarray(parallel_rate)

    kz = y_end.n.shape[0] // 2
    n_mid = y_end.n[kz]
    omega_mid = y_end.omega[kz]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8))

    ax = axes[0, 0]
    e0 = jnp.maximum(jnp.abs(energy[0]), 1e-14)
    ax.plot(ts, (energy - energy[0]) / e0, lw=2.2, label=r"$(E-E_0)/|E_0|$")
    ax.set_xlabel("t")
    ax.set_ylabel("relative drift")
    ax.set_title("Total energy drift")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(ts, particle_plasma - particle_plasma[0], lw=2.0, label=r"$\langle n\rangle$")
    ax.plot(ts, particle_total - particle_total[0], lw=2.0, label=r"$\langle n+N\rangle$")
    ax.set_xlabel("t")
    ax.set_ylabel("content drift")
    ax.set_title("Plasma and total particles")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 2]
    ax.plot(ts, parallel_rate, lw=2.0, label="parallel rate")
    ax.plot(ts, advective_rate, lw=2.0, label="advective rate")
    ax.plot(ts, sheath_prate, lw=2.0, label="sheath particle rate")
    ax.plot(ts, sheath_erate, lw=2.0, label="sheath energy rate")
    ax.set_xlabel("t")
    ax.set_ylabel("rate")
    ax.set_title("Target/sheath budget channels")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    im = ax.imshow(n_mid, origin="lower", cmap="coolwarm", aspect="equal")
    ax.set_title("final n (mid-plane)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 1]
    im = ax.imshow(omega_mid, origin="lower", cmap="coolwarm", aspect="equal")
    ax.set_title(r"final $\Omega$ (mid-plane)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 2]
    ax.plot(ts, jnp.abs(particle_rate_other), lw=2.0, label=r"$|\mathrm{other\ particle\ rate}|$")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("rate")
    ax.set_title("Budget residual (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        "FCI DRB3D full model: Loizu-like sheath + hot-ion + EM + neutrals",
        fontsize=14,
    )
    fig.tight_layout()
    out_png = out_dir / "fci_drb3d_full_multiphysics_sheath.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print(f"[fci-multiphysics] wrote {out_png}")
    print(
        "[fci-multiphysics] final diagnostics:",
        f"energy_drift={float((energy[-1]-energy[0])/jnp.maximum(jnp.abs(energy[0]),1e-14)):.3e}",
        f"plasma_particle_drift={float(particle_plasma[-1]-particle_plasma[0]):.3e}",
        f"total_particle_drift={float(particle_total[-1]-particle_total[0]):.3e}",
    )


if __name__ == "__main__":
    main()
