"""Benchmark SPD Poisson preconditioners for non-Boussinesq polarization.

Compares Jacobi vs spectral (FFT/circulant) preconditioners for the
variable-coefficient SPD solve -∇·(n ∇phi)=Omega.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.fd import div_n_grad, inv_div_n_grad_cg
from jaxdrb.nonlinear.grid import Grid2D


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()
    jax.config.update("jax_enable_x64", False)

    out_dir = Path("out_poisson_precond_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=72, ny=72, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    bc = BC2D.periodic()
    x = jnp.linspace(0.0, grid.Lx, grid.nx, endpoint=False)[:, None]
    y = jnp.linspace(0.0, grid.Ly, grid.ny, endpoint=False)[None, :]

    n_coeff = 1.0 + 0.3 * jnp.sin(x) + 0.2 * jnp.cos(2.0 * y)
    n_coeff = jnp.maximum(n_coeff, 0.2)
    phi_true = jnp.sin(2.0 * x) * jnp.sin(3.0 * y)
    rhs = -div_n_grad(phi_true, n_coeff, grid.dx, grid.dy, bc)

    def run(precond: str) -> tuple[float, float]:
        def solve():
            return inv_div_n_grad_cg(
                rhs,
                n_coeff=n_coeff,
                dx=grid.dx,
                dy=grid.dy,
                bc=bc,
                maxiter=800,
                tol=1e-10,
                preconditioner=precond,
            )

        solve_jit = jax.jit(solve)
        u = solve_jit()
        jax.block_until_ready(u)

        t0 = time.perf_counter()
        u = solve_jit()
        jax.block_until_ready(u)
        t1 = time.perf_counter()

        res = div_n_grad(u, n_coeff, grid.dx, grid.dy, bc) + rhs
        rel = float(jnp.linalg.norm(res.ravel()) / (jnp.linalg.norm(rhs.ravel()) + 1e-12))
        return t1 - t0, rel

    preconds = ["jacobi", "spectral"]
    times = []
    residuals = []
    for p in preconds:
        wall, rel = run(p)
        times.append(wall)
        residuals.append(rel)
        print(f"[poisson-precond] {p}: time={wall:.4f}s residual={rel:.3e}")

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axs[0].bar(preconds, times, color="tab:blue", alpha=0.85)
    axs[0].set_ylabel("wall time (s)")
    axs[0].set_title("Preconditioner runtime")
    axs[0].grid(alpha=0.3, axis="y")

    axs[1].bar(preconds, residuals, color="tab:orange", alpha=0.85)
    axs[1].set_ylabel("relative residual")
    axs[1].set_title("Preconditioner residual")
    axs[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "poisson_preconditioner_bench.png", dpi=220)
    np.savez(
        out_dir / "poisson_preconditioner_bench.npz",
        preconds=np.array(preconds),
        times=np.array(times),
        residuals=np.array(residuals),
    )
    print(f"[poisson-precond] wrote {out_dir}")


if __name__ == "__main__":
    main()
