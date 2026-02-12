"""CI gate for the non-Boussinesq Poisson preconditioner benchmark.

Enforces that both Jacobi and spectral (FFT/circulant) preconditioners
reach reasonable residuals and runtime on a small periodic grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.fd import div_n_grad, inv_div_n_grad_cg
from jaxdrb.nonlinear.grid import Grid2D


def _solve_stats(precond: str) -> tuple[float, float]:
    grid = Grid2D.make(nx=64, ny=64, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    bc = BC2D.periodic()
    x = jnp.linspace(0.0, grid.Lx, grid.nx, endpoint=False)[:, None]
    y = jnp.linspace(0.0, grid.Ly, grid.ny, endpoint=False)[None, :]
    n_coeff = 1.0 + 0.25 * jnp.sin(x) + 0.2 * jnp.cos(2.0 * y)
    n_coeff = jnp.maximum(n_coeff, 0.2)
    phi = jnp.sin(2.0 * x) * jnp.sin(3.0 * y)
    rhs = -div_n_grad(phi, n_coeff, grid.dx, grid.dy, bc)

    def solve():
        return inv_div_n_grad_cg(
            rhs,
            n_coeff=n_coeff,
            dx=grid.dx,
            dy=grid.dy,
            bc=bc,
            maxiter=600,
            tol=1e-10,
            preconditioner=precond,
        )

    solve_jit = jax.jit(solve)
    u = solve_jit()
    jax.block_until_ready(u)

    # Real timing (host).
    import time

    t0 = time.perf_counter()
    u = solve_jit()
    jax.block_until_ready(u)
    t1 = time.perf_counter()

    res = div_n_grad(u, n_coeff, grid.dx, grid.dy, bc) + rhs
    rel = float(jnp.linalg.norm(res.ravel()) / (jnp.linalg.norm(rhs.ravel()) + 1e-12))
    return float(t1 - t0), rel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-time-jacobi", type=float, default=0.2)
    p.add_argument("--max-time-spectral", type=float, default=0.4)
    p.add_argument("--max-res-jacobi", type=float, default=0.5)
    p.add_argument("--max-res-spectral", type=float, default=0.2)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", False)

    time_j, res_j = _solve_stats("jacobi")
    time_s, res_s = _solve_stats("spectral")

    metrics = {
        "jacobi": {"time": time_j, "residual": res_j},
        "spectral": {"time": time_s, "residual": res_s},
        "thresholds": {
            "max_time_jacobi": float(args.max_time_jacobi),
            "max_time_spectral": float(args.max_time_spectral),
            "max_res_jacobi": float(args.max_res_jacobi),
            "max_res_spectral": float(args.max_res_spectral),
        },
    }

    print(
        "[poisson-precond-gate] "
        f"jacobi time={time_j:.3f}s res={res_j:.3e} | "
        f"spectral time={time_s:.3f}s res={res_s:.3e}"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))

    failures = []
    if time_j > args.max_time_jacobi:
        failures.append(f"Jacobi time {time_j:.3f} > {args.max_time_jacobi:.3f}")
    if time_s > args.max_time_spectral:
        failures.append(f"Spectral time {time_s:.3f} > {args.max_time_spectral:.3f}")
    if res_j > args.max_res_jacobi:
        failures.append(f"Jacobi residual {res_j:.3e} > {args.max_res_jacobi:.3e}")
    if res_s > args.max_res_spectral:
        failures.append(f"Spectral residual {res_s:.3e} > {args.max_res_spectral:.3e}")
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
