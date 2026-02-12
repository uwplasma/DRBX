"""DRB2D long-time turbulence regression gate.

This gate exists to prevent the DRB2D nonlinear movie/regression cases from silently
degrading into:

- zonal-collapse (nearly pure ky=0 banded state)
- laminar/overdamped decay
- runaway growth / NaNs

The checks here intentionally use *broad bands* rather than a single sharp target,
because nonlinear turbulence is sensitive to numerics, tolerances, and resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.turbulence import (  # noqa: E402
    isotropic_power_spectrum_2d,
    spectrum_loglog_slope,
    zonal_fraction_y,
)
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState  # noqa: E402
from jaxdrb.nonlinear.grid import Grid2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--Lx", type=float, default=2.0 * jnp.pi)
    p.add_argument("--Ly", type=float, default=2.0 * jnp.pi)
    p.add_argument("--dt", type=float, default=1.0e-2)
    p.add_argument("--tmax", type=float, default=200.0)
    p.add_argument("--save-every", type=float, default=5.0, help="Save cadence in simulation time.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)

    p.add_argument("--max-abs-tail-dlogE-dt", type=float, default=0.1)
    p.add_argument("--min-zonal-frac", type=float, default=0.02)
    p.add_argument("--max-zonal-frac", type=float, default=0.8)
    p.add_argument("--min-slope", type=float, default=-6.0)
    p.add_argument("--max-slope", type=float, default=-1.0)
    p.add_argument("--slope-kmin", type=float, default=4.0)
    p.add_argument("--slope-kmax", type=float, default=14.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=float(args.Lx), Ly=float(args.Ly), dealias=False)
    params = DRB2DParams(
        omega_n=1.0,
        omega_Te=0.35,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.7,
        Dn=3e-3,
        DOmega=3e-3,
        DTe=3e-3,
        Dn4=5e-5,
        DOmega4=5e-5,
        DTe4=5e-5,
        mu_zonal_omega=0.12,
        mu_lin_n=0.12,
        mu_lin_omega=0.35,
        mu_lin_Te=0.12,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(int(args.seed))
    amp = 6e-3
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

    # Basic NaN guard.
    omega_ts = sol.ys.omega
    finite_ok = bool(jnp.all(jnp.isfinite(jnp.real(omega_ts))))

    E = jax.vmap(model.energy)(sol.ys)
    E = jnp.real(E)
    ts = jnp.asarray(save_ts)
    dlogE_dt = jnp.gradient(jnp.log(jnp.maximum(E, 1e-30)), ts)
    tail = slice(int(2 * dlogE_dt.size / 3), None)
    tail_mean = jnp.mean(dlogE_dt[tail])

    omega_last = jnp.real(omega_ts[-1])
    omega_last = omega_last - jnp.mean(omega_last)
    zonal = zonal_fraction_y(omega_last)

    # Use the vorticity spectrum as a regression metric, since it is the dynamically
    # active field in the DRB2D operator and yields a robust inertial-range proxy on
    # coarse grids. (Potential spectra can be substantially steeper and less stable as
    # a CI gate when hyperdiffusion is used to control aliasing/cascade.)
    k, Pk = isotropic_power_spectrum_2d(omega_last, Lx=float(args.Lx), Ly=float(args.Ly), nbins=32)
    slope = spectrum_loglog_slope(
        k,
        Pk,
        kmin=float(args.slope_kmin),
        kmax=float(args.slope_kmax),
    )

    metrics = {
        "finite_ok": bool(finite_ok),
        "tail_mean_dlogE_dt": float(tail_mean),
        "zonal_fraction": float(zonal),
        "spectrum_slope": float(slope),
    }
    print(
        "[drb2d-turbulence-gate] "
        f"finite={metrics['finite_ok']} "
        f"tail_mean_dlogE_dt={metrics['tail_mean_dlogE_dt']:.3e} "
        f"zonal={metrics['zonal_fraction']:.3f} "
        f"slope={metrics['spectrum_slope']:.2f}",
        flush=True,
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
