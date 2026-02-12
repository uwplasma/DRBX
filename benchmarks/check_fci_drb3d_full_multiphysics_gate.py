#!/usr/bin/env python3
"""Hard gate for full FCI DRB3D multiphysics target/sheath budget consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.neutrals import NeutralParams


def run_gate() -> dict[str, float]:
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
    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(1203)
    k = jax.random.split(key, 8)
    amp = 5e-4
    y0 = FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=0.18 + amp * jax.random.normal(k[4], shape),
        Ti=0.17 + amp * jax.random.normal(k[5], shape),
        psi=amp * jax.random.normal(k[6], shape),
        N=0.22 + amp * jax.random.normal(k[7], shape),
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
        perp_operator="spectral",
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.002,
        nsteps=140,
        save_every=5,
        solver="dopri5",
    )
    ts = 0.002 * jnp.arange(5, 141, 5)

    energy = []
    total_particles = []
    energy_rates = []
    particle_rates = []
    sheath_prates = []
    sheath_erates = []
    adv_rates = []
    parallel_rates = []
    other_prates = []
    split_residuals = []
    for i, ti in enumerate(ts):
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
        dyi = model.rhs(float(ti), yi)
        pb = model.particle_budget_terms(yi)
        eb = model.energy_budget_terms(yi)
        energy.append(float(model.energy(yi)))
        total_particles.append(float(model.total_particle_content(yi)))
        energy_rates.append(float(model.energy_rate(yi, dyi)))
        particle_rates.append(float(model.total_particle_rate(dyi)))
        sheath_prates.append(float(pb["sheath"]))
        sheath_erates.append(float(model.sheath_budget_rates(yi)[1]))
        adv_rates.append(float(pb["advective"]))
        parallel_rates.append(float(pb["parallel"]))
        other_prates.append(float(pb["other"]))
        split_residuals.append(float(eb["residual"]))

    energy = jnp.asarray(energy)
    total_particles = jnp.asarray(total_particles)
    energy_rates = jnp.asarray(energy_rates)
    particle_rates = jnp.asarray(particle_rates)
    sheath_prates = jnp.asarray(sheath_prates)
    sheath_erates = jnp.asarray(sheath_erates)
    adv_rates = jnp.asarray(adv_rates)
    parallel_rates = jnp.asarray(parallel_rates)
    other_prates = jnp.asarray(other_prates)
    split_residuals = jnp.asarray(split_residuals)

    dt_save = float(ts[1] - ts[0])
    dE_dt_fd = jnp.gradient(energy, dt_save)
    dP_dt_fd = jnp.gradient(total_particles, dt_save)
    rel_e = jnp.sqrt(jnp.mean((dE_dt_fd - energy_rates) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(energy_rates**2)), 1e-12
    )
    rel_p = jnp.sqrt(jnp.mean((dP_dt_fd - particle_rates) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(particle_rates**2)), 1e-12
    )
    return {
        "rel_energy_rate_mismatch": float(rel_e),
        "rel_particle_rate_mismatch": float(rel_p),
        "max_abs_advective_mean_rate": float(jnp.max(jnp.abs(adv_rates))),
        "max_abs_parallel_particle_rate": float(jnp.max(jnp.abs(parallel_rates))),
        "max_abs_other_particle_rate": float(jnp.max(jnp.abs(other_prates))),
        "median_sheath_particle_rate": float(jnp.median(sheath_prates)),
        "median_sheath_energy_rate": float(jnp.median(sheath_erates)),
        "max_abs_split_energy_residual": float(jnp.max(jnp.abs(split_residuals))),
        "is_finite": bool(jnp.isfinite(ys.n).all() and jnp.isfinite(ys.Te).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rel-energy-rate-mismatch", type=float, default=3.5e-2)
    parser.add_argument("--max-rel-particle-rate-mismatch", type=float, default=2.5e-2)
    parser.add_argument("--max-abs-advective-mean-rate", type=float, default=1e-12)
    parser.add_argument("--min-abs-parallel-particle-rate", type=float, default=1e-5)
    parser.add_argument("--max-abs-other-particle-rate", type=float, default=1e-9)
    parser.add_argument("--max-abs-split-energy-residual", type=float, default=5e-12)
    parser.add_argument(
        "--json-out", type=str, default="out/ci/fci_drb3d_full_multiphysics_gate.json"
    )
    args = parser.parse_args()

    metrics = run_gate()
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[fci-drb3d-full-gate] wrote metrics to {out_path}")
    print(json.dumps(metrics, indent=2))

    failures: list[str] = []
    if not metrics["is_finite"]:
        failures.append("solution contains NaN/Inf")
    if metrics["rel_energy_rate_mismatch"] > float(args.max_rel_energy_rate_mismatch):
        failures.append(
            f"rel_energy_rate_mismatch {metrics['rel_energy_rate_mismatch']:.3e} > {args.max_rel_energy_rate_mismatch:.3e}"
        )
    if metrics["rel_particle_rate_mismatch"] > float(args.max_rel_particle_rate_mismatch):
        failures.append(
            f"rel_particle_rate_mismatch {metrics['rel_particle_rate_mismatch']:.3e} > {args.max_rel_particle_rate_mismatch:.3e}"
        )
    if metrics["max_abs_advective_mean_rate"] > float(args.max_abs_advective_mean_rate):
        failures.append(
            f"max_abs_advective_mean_rate {metrics['max_abs_advective_mean_rate']:.3e} > {args.max_abs_advective_mean_rate:.3e}"
        )
    if metrics["max_abs_parallel_particle_rate"] < float(args.min_abs_parallel_particle_rate):
        failures.append(
            f"max_abs_parallel_particle_rate {metrics['max_abs_parallel_particle_rate']:.3e} < {args.min_abs_parallel_particle_rate:.3e}"
        )
    if metrics["max_abs_other_particle_rate"] > float(args.max_abs_other_particle_rate):
        failures.append(
            f"max_abs_other_particle_rate {metrics['max_abs_other_particle_rate']:.3e} > {args.max_abs_other_particle_rate:.3e}"
        )
    if metrics["max_abs_split_energy_residual"] > float(args.max_abs_split_energy_residual):
        failures.append(
            f"max_abs_split_energy_residual {metrics['max_abs_split_energy_residual']:.3e} > {args.max_abs_split_energy_residual:.3e}"
        )
    if metrics["median_sheath_particle_rate"] >= 0.0:
        failures.append(
            f"median_sheath_particle_rate {metrics['median_sheath_particle_rate']:.3e} must be < 0"
        )
    if metrics["median_sheath_energy_rate"] >= 0.0:
        failures.append(
            f"median_sheath_energy_rate {metrics['median_sheath_energy_rate']:.3e} must be < 0"
        )

    if failures:
        raise SystemExit("FCI DRB3D full multiphysics gate failed:\n- " + "\n- ".join(failures))

    print("[fci-drb3d-full-gate] PASS")


if __name__ == "__main__":
    main()
