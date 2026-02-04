"""Validate key nonlinear HW2D properties against literature-style checks.

This script focuses on *reviewer-proof* checks that are standard in the
Hasegawa–Wakatani (HW) literature, e.g. Camargo, Biskamp & Scott (1995):

1) A quadratic energy functional

   E = 1/2 ⟨ n^2 + |∇φ|^2 ⟩

   and its time derivative computed by a term-by-term budget, using the identity

   Ė = ⟨ n ∂t n - φ ∂t ω ⟩,

   which holds on periodic domains with ω = ∇⊥² φ.

2) In the ideal (non-driven, non-dissipative) limit, the advection operator should
   conserve quadratic invariants. With Arakawa's Jacobian, this is true to roundoff.

Outputs (in `--out`):
  - `panel_budget.png`: energy/enstrophy time traces + energy budget closure plot
  - `spectrum.png`: isotropic spectra of n and φ at final time
  - `timeseries.npz`: saved time series (t, E, Z, and budget terms)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState, hw2d_random_ic
from jaxdrb.nonlinear.stepper import rk4_step


def isotropic_spectrum(
    field: jnp.ndarray, *, kx: jnp.ndarray, ky: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute an isotropic 1D spectrum from a 2D periodic field.

    Uses a simple radial binning of |f̂(kx,ky)|^2. This is meant for qualitative
    diagnostics/plots (not for high-precision spectral studies).
    """

    fhat = jnp.fft.fft2(field)
    power2d = jnp.abs(fhat) ** 2
    kmag = jnp.sqrt(kx**2 + ky**2)

    kmax = jnp.max(kmag)
    nbins = int(jnp.maximum(16, jnp.floor(jnp.sqrt(field.size))))
    edges = jnp.linspace(0.0, kmax + 1e-12, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Bin by nearest edge interval.
    idx = jnp.clip(jnp.digitize(kmag.ravel(), edges) - 1, 0, nbins - 1)
    sums = jnp.zeros((nbins,), dtype=jnp.float64).at[idx].add(power2d.ravel())
    counts = jnp.zeros((nbins,), dtype=jnp.float64).at[idx].add(1.0)
    spec = sums / jnp.maximum(counts, 1.0)
    return centers, spec


def run_time_series(
    *,
    model: HW2DModel,
    y0: HW2DState,
    dt: float,
    tmax: float,
    stride: int,
) -> tuple[dict[str, jnp.ndarray], HW2DState]:
    """Integrate with fixed-step RK4 and record diagnostics/budgets every `stride` steps."""

    nsteps = int(jnp.ceil(tmax / dt))
    nrec = max(1, nsteps // stride)

    @jax.jit
    def advance_chunk(t: jnp.ndarray, y: HW2DState) -> tuple[jnp.ndarray, HW2DState]:
        def body(i, carry):
            t_, y_ = carry
            y_next = rk4_step(y_, t_, dt, model.rhs)
            return (t_ + dt, y_next)

        t_end, y_end = jax.lax.fori_loop(0, stride, body, (t, y))
        return t_end, y_end

    ts = []
    Es = []
    Zs = []
    budgets = {
        k: []
        for k in [
            "E_dot_adv",
            "E_dot_drive",
            "E_dot_couple",
            "E_dot_diff",
            "E_dot_hyper",
            "E_dot_total",
        ]
    }

    t = jnp.asarray(0.0)
    y = y0
    for i in range(nrec):
        t, y = advance_chunk(t, y)
        diag = model.diagnostics(y)
        budget = model.energy_budget(y)
        if not (
            jnp.isfinite(diag["E"]) & jnp.isfinite(diag["Z"]) & jnp.isfinite(budget["E_dot_total"])
        ):
            raise FloatingPointError(f"Non-finite diagnostics at i={i}, t={float(t):.3f}.")

        ts.append(t)
        Es.append(diag["E"])
        Zs.append(diag["Z"])
        for k in budgets:
            budgets[k].append(budget[k])

        if (i + 1) % max(1, nrec // 10) == 0 or i == 0:
            print(
                f"[hw2d-validate] rec {i + 1}/{nrec} t={float(t):.3f} "
                f"E={float(diag['E']):.3e} Z={float(diag['Z']):.3e} E_dot_total={float(budget['E_dot_total']):+.3e}"
            )

    out = {"t": jnp.stack(ts), "E": jnp.stack(Es), "Z": jnp.stack(Zs)}
    for k, v in budgets.items():
        out[k] = jnp.stack(v)
    return out, y


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--tmax", type=float, default=40.0)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out_hw2d_camargo1995")
    p.add_argument(
        "--ideal", action="store_true", help="Run the ideal (no drive/coupling/dissipation) subset."
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)

    if args.ideal:
        params = HW2DParams(
            kappa=0.0,
            alpha=0.0,
            Dn=0.0,
            DOmega=0.0,
            nu4_n=0.0,
            nu4_omega=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
    else:
        # A stable, turbulence-like regime with mild diffusion + hyperdiffusion.
        params = HW2DParams(
            kappa=1.0,
            alpha=1.0,
            Dn=1e-3,
            DOmega=1e-3,
            nu4_n=1e-6,
            nu4_omega=1e-6,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=True,
            alpha_nonzonal_only=True,
        )

    model = HW2DModel(params=params, grid=grid)
    y0 = hw2d_random_ic(jax.random.key(args.seed), grid, amp=1e-3, include_neutrals=False)

    print(
        f"[hw2d-validate] grid=({grid.nx},{grid.ny}) dt={args.dt} tmax={args.tmax} "
        f"stride={args.stride} ideal={args.ideal} bracket={params.bracket} poisson={params.poisson}"
    )

    series, y_end = run_time_series(
        model=model, y0=y0, dt=float(args.dt), tmax=float(args.tmax), stride=int(args.stride)
    )
    jnp.savez(out_dir / "timeseries.npz", **series)

    t = series["t"]
    E = series["E"]
    Z = series["Z"]
    Edot = series["E_dot_total"]

    # Finite-difference dE/dt to compare to the budget (closure check).
    dE_dt_fd = jnp.gradient(E, t)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    ax.plot(t, E, lw=2, label="E")
    ax.plot(t, Z, lw=2, label="Z")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_title("Integral invariants / diagnostics")
    ax.legend()

    ax = axs[0, 1]
    ax.plot(t, dE_dt_fd, lw=2, label=r"$dE/dt$ (FD)")
    ax.plot(t, Edot, lw=2, label=r"$\dot E$ (budget)")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("t")
    ax.set_title("Energy budget closure")
    ax.legend()

    ax = axs[1, 0]
    for k, lab in [
        ("E_dot_adv", "advection"),
        ("E_dot_drive", "drive"),
        ("E_dot_couple", "coupling"),
        ("E_dot_diff", "diffusion"),
        ("E_dot_hyper", "hyperdiff"),
    ]:
        ax.plot(t, series[k], lw=1.8, label=lab)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("t")
    ax.set_title("Budget term decomposition")
    ax.legend(ncols=2, fontsize=9)

    ax = axs[1, 1]
    # Highlight the key "reviewer plot": advection should not create/destroy energy.
    ax.plot(t, jnp.abs(series["E_dot_adv"]), lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_title(r"$|\dot E_{adv}|$ (Arakawa should be ~ roundoff)")

    fig.suptitle("HW2D validation: invariants and energy budget", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "panel_budget.png", dpi=220)
    plt.close(fig)

    # Final-time spectra (qualitative).
    phi_end = model.phi_from_omega(y_end.omega)
    k, spec_n = isotropic_spectrum(y_end.n, kx=grid.kx, ky=grid.ky)
    _, spec_phi = isotropic_spectrum(phi_end, kx=grid.kx, ky=grid.ky)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    ax.loglog(k[1:], spec_n[1:], lw=2, label=r"$|\hat n|^2$")
    ax.loglog(k[1:], spec_phi[1:], lw=2, label=r"$|\hat \phi|^2$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("binned power")
    ax.set_title("Final-time isotropic spectra (qualitative)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum.png", dpi=220)
    plt.close(fig)

    print(f"[hw2d-validate] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
