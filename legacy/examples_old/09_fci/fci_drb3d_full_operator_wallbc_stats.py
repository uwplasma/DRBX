"""FCI DRB3D full model: FD/FV wall-BC comparison and turbulence statistics.

This script runs two short diagnostics:

1) Dirichlet wall relaxation test for perpendicular operators (`fd` and `fv`):
   tracks boundary RMS decay of density perturbations.
2) Periodic-box turbulence-statistics regression run:
   tracks density/vorticity fluctuation levels and zonal fraction.

It produces a publication-style summary panel suitable for docs/README updates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.bc import BC2D
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _boundary_rms(f: jnp.ndarray) -> jnp.ndarray:
    bd = jnp.concatenate(
        [
            f[:, :, 0, :].reshape((f.shape[0], -1)),
            f[:, :, -1, :].reshape((f.shape[0], -1)),
            f[:, :, :, 0].reshape((f.shape[0], -1)),
            f[:, :, :, -1].reshape((f.shape[0], -1)),
        ],
        axis=1,
    )
    return jnp.sqrt(jnp.mean(bd**2, axis=1))


def _random_state(key: jax.Array, shape: tuple[int, int, int], amp: float) -> FCIDRB3DFullState:
    k = jax.random.split(key, 5)
    return FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=amp * jax.random.normal(k[4], shape),
    )


def run_wall_bc_case(perp_operator: str, *, nsteps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    grid = FCISlabGrid.make(
        nx=16,
        ny=16,
        nz=6,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.2,
        Bz=1.0,
        open_field_line=False,
    )
    y0 = _random_state(jax.random.key(11), (grid.nz, grid.nx, grid.ny), amp=1e-2)
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.05,
        Dn=3e-3,
        DOmega=3e-3,
        Dvpar=3e-3,
        DTe=3e-3,
        chi_par=3e-3,
        sheath_on=False,
        perp_operator=perp_operator,  # type: ignore[arg-type]
        perp_bc=BC2D.dirichlet(x=0.0, y=0.0),
        perp_bc_nu=4.0,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.02,
        nsteps=int(nsteps),
        save_every=10,
        solver="dopri5",
    )
    t = 0.02 * jnp.arange(10, int(nsteps) + 1, 10)
    bnd = _boundary_rms(ys.n)
    return t, bnd


def run_turbulence_case(
    *, nsteps: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    grid = FCISlabGrid.make(
        nx=18,
        ny=18,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.25,
        Bz=1.0,
        open_field_line=False,
    )
    shape = (grid.nz, grid.nx, grid.ny)
    y0 = _random_state(jax.random.key(1), shape, amp=5e-4)
    params = FCIDRB3DFullParams(
        omega_n=3.0,
        omega_Te=2.0,
        kappa=1.0,
        alpha=0.45,
        eta_par=0.03,
        Dn=2e-4,
        DOmega=2e-4,
        Dvpar=2e-4,
        DTe=2e-4,
        chi_par=4e-4,
        sheath_on=False,
        perp_operator="spectral",
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.01,
        nsteps=int(nsteps),
        save_every=20,
        solver="dopri5",
    )
    t = 0.01 * jnp.arange(20, int(nsteps) + 1, 20)
    n_fluct = ys.n - jnp.mean(ys.n, axis=(-1, -2), keepdims=True)
    omega_fluct = ys.omega - jnp.mean(ys.omega, axis=(-1, -2), keepdims=True)
    n_rms = jnp.sqrt(jnp.mean(n_fluct**2, axis=(-1, -2, -3)))
    omega_rms = jnp.sqrt(jnp.mean(omega_fluct**2, axis=(-1, -2, -3)))
    omega_zonal = jnp.mean(omega_fluct, axis=-1, keepdims=True)
    zonal_fraction = jnp.sqrt(jnp.mean(omega_zonal**2, axis=(-1, -2, -3))) / jnp.maximum(
        omega_rms, 1e-30
    )
    return t, n_rms, omega_rms, zonal_fraction


def main() -> None:
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="out_fci_drb3d_full_ops")
    parser.add_argument("--wall-nsteps", type=int, default=140)
    parser.add_argument("--turb-nsteps", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_fd, b_fd = run_wall_bc_case("fd", nsteps=int(args.wall_nsteps))
    t_fv, b_fv = run_wall_bc_case("fv", nsteps=int(args.wall_nsteps))
    t_turb, n_rms, omega_rms, zonal_fraction = run_turbulence_case(nsteps=int(args.turb_nsteps))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    ax = axes[0, 0]
    ax.semilogy(t_fd, b_fd, "-o", ms=3, label="FD")
    ax.semilogy(t_fv, b_fv, "-s", ms=3, label="FV")
    ax.set_xlabel("t")
    ax.set_ylabel("boundary RMS(n)")
    ax.set_title("Dirichlet wall relaxation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    decay_fd = float(b_fd[-1] / jnp.maximum(b_fd[0], 1e-30))
    decay_fv = float(b_fv[-1] / jnp.maximum(b_fv[0], 1e-30))
    ax.bar(["FD", "FV"], [decay_fd, decay_fv], color=["tab:blue", "tab:orange"])
    ax.set_ylabel("boundary RMS ratio (end/start)")
    ax.set_title("Wall-BC damping effectiveness")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(t_turb, n_rms, label=r"$\mathrm{rms}(n-\langle n \rangle)$")
    ax.semilogy(t_turb, omega_rms, label=r"$\mathrm{rms}(\Omega-\langle \Omega \rangle)$")
    ax.set_xlabel("t")
    ax.set_ylabel("RMS fluctuation level")
    ax.set_title("Periodic turbulence-statistics run")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(t_turb, zonal_fraction, lw=2.0, label="zonal fraction")
    ax.axhline(0.5, color="k", lw=1.0, ls="--", label="gate upper bound")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathrm{rms}(\Omega_{zonal}) / \mathrm{rms}(\Omega)$")
    ax.set_title("Zonal-collapse guard metric")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("FCI DRB3D full-model: wall BC + turbulence regression diagnostics", fontsize=14)
    fig.tight_layout()
    out_png = out_dir / "fci_drb3d_full_operator_wallbc_stats.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print(f"[fci-drb3d-full] wrote {out_png}")


if __name__ == "__main__":
    main()
