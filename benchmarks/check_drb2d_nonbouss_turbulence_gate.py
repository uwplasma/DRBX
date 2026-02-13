#!/usr/bin/env python3
"""Non-Boussinesq DRB2D long-time turbulence regression gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxdrb.analysis.turbulence import (
    isotropic_power_spectrum_2d,
    spectrum_loglog_slope,
    zonal_fraction_y,
)
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--Lx", type=float, default=2.0 * jnp.pi)
    p.add_argument("--Ly", type=float, default=2.0 * jnp.pi)
    p.add_argument("--dt", type=float, default=1.0e-2)
    p.add_argument("--tmax", type=float, default=120.0)
    p.add_argument("--save-every", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)

    p.add_argument("--max-abs-tail-dlogE-dt", type=float, default=0.25)
    p.add_argument("--min-zonal-frac", type=float, default=0.02)
    p.add_argument("--max-zonal-frac", type=float, default=0.9)
    p.add_argument("--min-slope", type=float, default=-7.0)
    p.add_argument("--max-slope", type=float, default=-1.0)
    p.add_argument("--slope-kmin", type=float, default=4.0)
    p.add_argument("--slope-kmax", type=float, default=14.0)
    p.add_argument("--tail-frac", type=float, default=0.33)
    p.add_argument("--tail-max-frames", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=float(args.Lx), Ly=float(args.Ly), dealias=False)
    params = DRB2DParams(
        omega_n=0.6,
        omega_Te=0.2,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.6,
        Dn=4e-3,
        DOmega=4e-3,
        DTe=4e-3,
        Dn4=8e-5,
        DOmega4=8e-5,
        DTe4=8e-5,
        mu_zonal_omega=0.12,
        mu_lin_n=0.18,
        mu_lin_omega=0.45,
        mu_lin_Te=0.18,
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        n0=1.0,
        n0_min=0.2,
        n0_max=2.0,
        bracket="arakawa",
        poisson="cg_fd",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
        polarization_cg_maxiter=260,
        polarization_cg_tol=5e-6,
        polarization_preconditioner="spectral_jacobi",
        polarization_precond_shift=1e-6,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(int(args.seed))
    amp = 4e-3
    shape = (grid.nx, grid.ny)
    y0 = DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(int(args.seed) + 1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(int(args.seed) + 2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(int(args.seed) + 3), shape),
        Te=amp * jax.random.normal(jax.random.key(int(args.seed) + 4), shape),
    )

    save_every = float(args.save_every)
    save_ts = jnp.arange(save_every, float(args.tmax) + 1e-12, save_every)
    sol = model.diffeqsolve(
        y0=y0,
        t0=0.0,
        t1=float(args.tmax),
        dt0=float(args.dt),
        save_ts=save_ts,
        solver="tsit5",
        adaptive=True,
        rtol=1e-5,
        atol=1e-8,
        max_steps=400_000,
        progress=False,
    )

    omega_ts = jnp.real(sol.ys.omega)
    finite_ok = bool(jnp.all(jnp.isfinite(omega_ts)))

    E = jax.vmap(model.energy)(sol.ys)
    E = jnp.real(E)
    ts = jnp.asarray(save_ts)
    dlogE_dt = jnp.gradient(jnp.log(jnp.maximum(E, 1e-30)), ts)
    tail = slice(int(2 * dlogE_dt.size / 3), None)
    tail_mean = jnp.mean(dlogE_dt[tail])

    nframes = int(omega_ts.shape[0])
    tail_start = int((1.0 - float(args.tail_frac)) * nframes)
    tail_start = max(0, min(tail_start, nframes - 1))
    tail_idx = jnp.arange(tail_start, nframes)
    if int(tail_idx.size) == 0:
        tail_idx = jnp.asarray([nframes - 1])
    max_frames = max(1, int(args.tail_max_frames))
    if int(tail_idx.size) > max_frames:
        pick = jnp.linspace(0, tail_idx.size - 1, max_frames)
        tail_idx = tail_idx[jnp.round(pick).astype(int)]

    def _omega_fluct(idx):
        w = omega_ts[idx]
        return w - jnp.mean(w)

    zonal_vals = jnp.asarray([zonal_fraction_y(_omega_fluct(i)) for i in tail_idx])
    slopes = []
    for i in tail_idx:
        w = _omega_fluct(i)
        k, Pk = isotropic_power_spectrum_2d(w, Lx=float(args.Lx), Ly=float(args.Ly), nbins=32)
        slopes.append(
            spectrum_loglog_slope(
                k,
                Pk,
                kmin=float(args.slope_kmin),
                kmax=float(args.slope_kmax),
            )
        )
    slope = jnp.mean(jnp.asarray(slopes))
    zonal = jnp.mean(zonal_vals)

    metrics = {
        "finite_ok": bool(finite_ok),
        "tail_mean_dlogE_dt": float(tail_mean),
        "zonal_fraction": float(zonal),
        "spectrum_slope": float(slope),
        "tail_frames": int(tail_idx.size),
    }
    print(
        "[drb2d-nonbouss-gate] "
        f"finite={metrics['finite_ok']} "
        f"tail_mean_dlogE_dt={metrics['tail_mean_dlogE_dt']:.3e} "
        f"zonal={metrics['zonal_fraction']:.3f} "
        f"slope={metrics['spectrum_slope']:.2f}"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))

    failures: list[str] = []
    if not metrics["finite_ok"]:
        failures.append("state contains non-finite values")
    if abs(metrics["tail_mean_dlogE_dt"]) > float(args.max_abs_tail_dlogE_dt):
        failures.append(
            f"|tail_mean_dlogE_dt|={abs(metrics['tail_mean_dlogE_dt']):.3e} "
            f"> {float(args.max_abs_tail_dlogE_dt):.3e}"
        )
    if metrics["zonal_fraction"] < float(args.min_zonal_frac):
        failures.append(
            f"zonal_fraction={metrics['zonal_fraction']:.3f} < {float(args.min_zonal_frac):.3f}"
        )
    if metrics["zonal_fraction"] > float(args.max_zonal_frac):
        failures.append(
            f"zonal_fraction={metrics['zonal_fraction']:.3f} > {float(args.max_zonal_frac):.3f}"
        )
    if metrics["spectrum_slope"] < float(args.min_slope) or metrics["spectrum_slope"] > float(
        args.max_slope
    ):
        failures.append(
            f"spectrum_slope={metrics['spectrum_slope']:.2f} not in "
            f"[{float(args.min_slope):.2f},{float(args.max_slope):.2f}]"
        )
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()

