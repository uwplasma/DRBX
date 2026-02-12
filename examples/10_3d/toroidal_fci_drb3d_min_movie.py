"""3D toroidal-plane FCI movie for a minimal DRB3D-like model (n, Omega).

This script produces a *true 3D* visualization: the solution is defined on a stack of
toroidal planes (fixed toroidal angle phi). Each plane uses cylindrical coordinates
(R, Z), and we render the field in 3D Cartesian space:

  x = R cos(phi),  y = R sin(phi),  z = Z

The physics model is the minimal 3D DRB-like slab operator in `jaxdrb.fci.drb3d`:

  y = (n, Omega)

with a Hasegawa–Wakatani-like coupling term `alpha*(phi - n)` and a simple curvature/
gradient drive `kappa` used to obtain rapid nonlinear dynamics in a short run.

The goal is a short, reproducible movie that shows:
  linear growth -> nonlinear saturation -> turbulence-like dynamics

This is intentionally a *numerical/geometry* showcase rather than a quantitative device
model. For the full multiphysics DRB3D branch (sheath + hot-ion + EM + neutrals), see
`examples/09_fci/`.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import robust_symmetric_vlim, set_mpl_style
from jaxdrb.fci.builder import EssosToroidalFCIConfig, build_fci_maps_essos_toroidal_planes
from jaxdrb.fci.drb3d import FCIDRB3DModel, FCIDRB3DParams, FCIDRB3DState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


@dataclass(frozen=True)
class AnalyticTokamakLikeField:
    """Simple analytic field with dominant toroidal component and small vertical pitch."""

    R0: float = 1.4
    B0: float = 1.0
    Bz_frac: float = 0.12

    def B(self, xyz: np.ndarray) -> tuple[float, float, float]:
        x, y, z = [float(v) for v in xyz]
        R = float(np.sqrt(x * x + y * y))
        phi = float(np.arctan2(y, x))
        # Cylindrical components.
        BR = 0.0
        Bphi = self.B0 * self.R0 / max(R, 1e-6)
        BZ = self.Bz_frac * self.B0
        # Convert to Cartesian.
        Bx = BR * np.cos(phi) - Bphi * np.sin(phi)
        By = BR * np.sin(phi) + Bphi * np.cos(phi)
        return float(Bx), float(By), float(BZ)


def _mesh_RZ(cfg: EssosToroidalFCIConfig) -> tuple[np.ndarray, np.ndarray]:
    R_axis = cfg.R0 + cfg.dR * np.arange(cfg.nR, dtype=float)
    Z_axis = cfg.Z0 + cfg.dZ * np.arange(cfg.nZ, dtype=float)
    RR, ZZ = np.meshgrid(R_axis, Z_axis, indexing="ij")
    return RR, ZZ


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_toroidal_fci_drb3d_min_movie")
    p.add_argument("--nphi", type=int, default=12)
    p.add_argument("--nR", type=int, default=14)
    p.add_argument("--nZ", type=int, default=14)
    p.add_argument("--dphi", type=float, default=0.12)
    p.add_argument("--R0", type=float, default=1.30)
    p.add_argument("--Z0", type=float, default=-0.18)
    p.add_argument("--dR", type=float, default=0.020)
    p.add_argument("--dZ", type=float, default=0.026)
    p.add_argument(
        "--open-field-line",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether the FCI map is open (hits targets) or periodic in the parallel direction.",
    )
    p.add_argument("--dl-min", type=float, default=2e-2)

    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--nsteps", type=int, default=900)
    p.add_argument("--save-every", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--kappa", type=float, default=0.60)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--Dn", type=float, default=2e-3)
    p.add_argument("--DOmega", type=float, default=2e-3)
    p.add_argument("--sheath-nu", type=float, default=0.10)

    p.add_argument(
        "--rotate-camera",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Rotate the 3D camera during the animation to make the 3D geometry unambiguous.",
    )
    p.add_argument("--azim0", type=float, default=40.0, help="Initial camera azimuth (degrees).")
    p.add_argument(
        "--azim-span",
        type=float,
        default=140.0,
        help="Total azimuth rotation over the full movie (degrees).",
    )

    p.add_argument("--max-wall", type=float, default=45.0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    field = AnalyticTokamakLikeField(R0=1.4, B0=1.0, Bz_frac=0.12)
    cfg = EssosToroidalFCIConfig(
        R0=float(args.R0),
        Z0=float(args.Z0),
        dR=float(args.dR),
        dZ=float(args.dZ),
        nR=int(args.nR),
        nZ=int(args.nZ),
        phi0=0.0,
        dphi=float(args.dphi),
        nphi=int(args.nphi),
        open_field_line=bool(args.open_field_line),
        cell_centered=True,
        periodic_R=False,
        periodic_Z=False,
        periodic_phi=True,
        # Targets are the patch boundary in (R,Z).
        R_min=float(args.R0),
        R_max=float(args.R0 + args.dR * (args.nR - 1)),
        Z_min=float(args.Z0),
        Z_max=float(args.Z0 + args.dZ * (args.nZ - 1)),
    )
    map_fwd, map_bwd, _meta = build_fci_maps_essos_toroidal_planes(
        cfg, field=field, nsub=8, dl_min=float(args.dl_min)
    )
    l = cfg.phi0 + cfg.dphi * jnp.arange(cfg.nphi)
    grid = FCISlabGrid.from_maps(
        x0=cfg.R0,
        y0=cfg.Z0,
        dx=cfg.dR,
        dy=cfg.dZ,
        nx=cfg.nR,
        ny=cfg.nZ,
        l=l,
        map_fwd=map_fwd,
        map_bwd=map_bwd,
        open_field_line=bool(args.open_field_line),
        cell_centered=True,
    )

    params = FCIDRB3DParams(
        kappa=float(args.kappa),
        alpha=float(args.alpha),
        kpar=0.0,
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        bracket="arakawa",
        poisson="spectral",
        boussinesq=True,
        dealias_on=False,
        sheath_nu=float(args.sheath_nu) if bool(args.open_field_line) else 0.0,
    )
    model = FCIDRB3DModel(params=params, grid=grid)

    key = jax.random.key(int(args.seed))
    amp = 6e-3
    shape = (grid.nz, grid.nx, grid.ny)
    y0 = FCIDRB3DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(int(args.seed) + 1), shape),
    )

    print(
        "[toroidal-fci-drb3d-min-movie] "
        f"grid=(nphi={grid.nz},nR={grid.nx},nZ={grid.ny}) dt={args.dt} nsteps={args.nsteps} "
        f"save_every={args.save_every} open={bool(args.open_field_line)}"
    )
    t0 = time.time()
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=float(args.dt),
        nsteps=int(args.nsteps),
        save_every=int(args.save_every),
        solver="dopri5",
    )
    wall = time.time() - t0
    if wall > float(args.max_wall):
        print(
            f"[toroidal-fci-drb3d-min-movie] warning: wall {wall:.1f}s exceeded max-wall={args.max_wall}s"
        )

    # Render omega fluctuations in 3D.
    omega_ts = np.asarray(jax.device_get(ys.omega))
    # Normalize each frame to highlight structure.
    omega_ts = omega_ts - omega_ts.mean(axis=(1, 2, 3), keepdims=True)
    omega_rms = np.sqrt(np.mean(omega_ts**2, axis=(1, 2, 3), keepdims=True))
    omega_plot = omega_ts / (omega_rms + 1e-30)
    vmax = robust_symmetric_vlim(omega_plot, q=0.995)

    RR, ZZ = _mesh_RZ(cfg)
    phis = np.asarray(l)
    # Precompute point cloud coordinates in 3D (constant across time).
    xs = []
    ys3 = []
    zs = []
    for phi in phis:
        xs.append(RR * np.cos(float(phi)))
        ys3.append(RR * np.sin(float(phi)))
        zs.append(ZZ)
    X = np.stack(xs, axis=0).reshape((-1,))
    Y = np.stack(ys3, axis=0).reshape((-1,))
    Z = np.stack(zs, axis=0).reshape((-1,))

    frames = omega_plot.reshape((omega_plot.shape[0], -1))
    nframes = int(frames.shape[0])
    ts = float(args.dt) * float(args.save_every) * np.arange(1, nframes + 1)

    fig = plt.figure(figsize=(5.6, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("FCI DRB3D (toroidal planes): normalized $\\Omega$ fluctuation")
    ax.set_axis_off()
    elev = 18.0
    axim0 = float(args.azim0)
    axim_span = float(args.azim_span)
    ax.view_init(elev=elev, azim=axim0)
    # Use per-plane small "quad patches" in 3D instead of point clouds. This makes the
    # toroidal geometry visually clear in the README and avoids the sparse look of a
    # scatter plot on coarse grids.
    nphi = int(len(phis))
    nx = int(cfg.nR)
    ny = int(cfg.nZ)
    Xg = np.stack(xs, axis=0)
    Yg = np.stack(ys3, axis=0)
    Zg = np.stack(zs, axis=0)
    V0 = frames[0].reshape((nphi, nx, ny))
    # Matplotlib expects facecolors for each quad; we approximate by per-cell coloring.
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    surfs = []
    for k in range(nphi):
        colors = cmap(norm(V0[k]))  # (nx, ny, 4)
        srf = ax.plot_surface(
            Xg[k],
            Yg[k],
            Zg[k],
            rstride=1,
            cstride=1,
            facecolors=colors,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        surfs.append(srf)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cb = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("normalized $\\Omega$")
    txt = ax.text2D(0.04, 0.02, f"t={ts[0]:.2f}", transform=ax.transAxes)

    # Keep the content centered and visible by setting a cubic bounding box.
    xc = 0.5 * (X.min() + X.max())
    yc = 0.5 * (Y.min() + Y.max())
    zc = 0.5 * (Z.min() + Z.max())
    half = 0.55 * float(max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()))
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_zlim(zc - half, zc + half)

    def update(i: int):
        Vi = frames[i].reshape((nphi, nx, ny))
        for k in range(nphi):
            surfs[k].set_facecolors(cmap(norm(Vi[k])).reshape((-1, 4)))
        if bool(args.rotate_camera) and nframes > 1:
            frac = float(i) / float(nframes - 1)
            ax.view_init(elev=elev, azim=axim0 + axim_span * frac)
        txt.set_text(f"t={ts[i]:.2f}")
        return (*surfs, txt)

    ani = animation.FuncAnimation(fig, update, frames=nframes, interval=60, blit=False)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    print(f"[toroidal-fci-drb3d-min-movie] wrote {gif_path}")


if __name__ == "__main__":
    main()
