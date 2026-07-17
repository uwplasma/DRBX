"""3D stellarator renders: cutaway turbulence movie + closed/open field lines.

Two engaging three-dimensional views of the rotating-ellipse stellarator:

1. ``stellarator_3d_turbulence.gif`` -- a cutaway torus whose surface is colored
   by the evolving turbulent density fluctuation from a multi-mode four-field
   run, with poloidal cross-section caps showing the interior and a slowly
   orbiting camera. The rotating elliptical cross-section is visible directly.
2. ``stellarator_3d_field_lines.png`` -- helical field lines on the same
   geometry: closed core field lines wind around the torus indefinitely, while
   scrape-off-layer field lines beyond the limiter radius are open -- they end
   on the toroidal limiter after one transit.

Run:

    PYTHONPATH=src python examples/stellarator/stellarator_3d_render_demo.py

writes ``output/stellarator_3d/``. Both are release-hosted, not committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from jax_drb.geometry import build_rotating_ellipse_geometry, rotating_ellipse_position  # noqa: E402

from stellarator_turbulence_case import run_stellarator_turbulence  # noqa: E402

SHAPE = (16, 32, 16)
STEPS = 120
DT = 2.0e-3
FRAME_STRIDE = 4
R0, ELONGATION, NFP, IOTA = 3.0, 0.35, 1, 0.9
X_MIN, X_MAX = 0.2, 1.0
LIMITER_RADIUS = 0.6
WEDGE = 0.25 * 2.0 * np.pi          # cutaway opening angle
FINE_THETA, FINE_ZETA = 96, 120     # upsampled surface resolution
OUTPUT_DIR = Path("output/stellarator_3d")


def _positions(x, theta, zeta):
    p = np.asarray(rotating_ellipse_position(
        jax.numpy.asarray(x), jax.numpy.asarray(theta), jax.numpy.asarray(zeta),
        r0=R0, elongation=ELONGATION, n_field_periods=NFP))
    return p[..., 0], p[..., 1], p[..., 2]


def _periodic_upsample(field, coarse, fine):
    """Periodic linear interpolation along one axis (last axis of ``field``)."""

    return np.apply_along_axis(
        lambda row: np.interp(fine, coarse, row, period=2.0 * np.pi), -1, field)


def render_turbulence_movie(run, geometry, path):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    x_centers = np.asarray(geometry.grid.x.centers)
    theta_c = np.asarray(geometry.grid.y.centers)
    zeta_c = np.asarray(geometry.grid.z.centers)
    fluctuation = run.density_frames - 1.0

    # Color a substantial outer flux surface (fluctuations carry a radial
    # envelope, so the outermost surfaces are quiet; 0.8 of the radius is both
    # active and visually representative of the rotating ellipse).
    x_target = X_MIN + 0.8 * (X_MAX - X_MIN)
    surface_index = int(np.argmin(np.abs(x_centers - x_target)))
    x_surface = float(x_centers[surface_index])

    theta_f = np.linspace(0.0, 2.0 * np.pi, FINE_THETA)
    zeta_f = np.linspace(WEDGE, 2.0 * np.pi, FINE_ZETA)  # cutaway: skip the wedge
    theta_grid, zeta_grid = np.meshgrid(theta_f, zeta_f, indexing="ij")
    Xs, Ys, Zs = _positions(np.full_like(theta_grid, x_surface), theta_grid, zeta_grid)

    # Cross-section caps at the two wedge faces.
    x_cap = np.linspace(X_MIN, x_surface, 24)
    cap_x, cap_theta = np.meshgrid(x_cap, theta_f, indexing="ij")
    caps = [_positions(cap_x, cap_theta, np.full_like(cap_x, z)) for z in (WEDGE, 2.0 * np.pi)]

    vmax = float(np.abs(fluctuation).max()) or 1.0
    norm = colors.Normalize(-vmax, vmax)
    cmap = cm.RdBu_r

    def _bilinear(field_tz):
        """Upsample a (theta, zeta) field to the fine surface grid, periodically."""

        fine_zeta = _periodic_upsample(field_tz, zeta_c, zeta_f)        # (ny, FZ)
        return _periodic_upsample(fine_zeta.T, theta_c, theta_f).T      # (FT, FZ)

    def cap_colors(frame, z_index):
        field_xt = fluctuation[frame][:, :, z_index]      # (nx, ny)
        fine = _periodic_upsample(field_xt, theta_c, theta_f)   # (nx, FT)
        return cmap(norm(np.stack([np.interp(x_cap, x_centers, fine[:, j]) for j in range(FINE_THETA)], axis=1)))

    cap_z_indices = (int(np.argmin(np.abs(zeta_c - WEDGE))), 0)

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame):
        ax.clear()
        ax.set_axis_off()
        colors_s = _bilinear(fluctuation[frame][surface_index])
        ax.plot_surface(Xs, Ys, Zs, facecolors=cmap(norm(colors_s)), shade=False,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
        for (cx, cy, cz), z_index in zip(caps, cap_z_indices):
            ax.plot_surface(cx, cy, cz, facecolors=cap_colors(frame, z_index), shade=False,
                            rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_box_aspect((1, 1, 0.34))
        ax.set_xlim(-4.15, 4.15), ax.set_ylim(-4.15, 4.15), ax.set_zlim(-1.45, 1.45)
        fig.subplots_adjust(left=0, right=1, bottom=-0.05, top=1.02)
        ax.view_init(elev=24, azim=-55 + 1.2 * frame)
        ax.set_title(f"Stellarator turbulence (rotating ellipse)   t = {run.times[frame]:.3f}",
                     fontsize=10)

    movie = animation.FuncAnimation(fig, draw, frames=len(run.times), interval=110, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=9), dpi=100)
    plt.close(fig)
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


def render_field_lines(path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    # Faint boundary surface for context (with the same cutaway wedge).
    theta_f = np.linspace(0.0, 2.0 * np.pi, 60)
    zeta_f = np.linspace(WEDGE, 2.0 * np.pi, 90)
    tg, zg = np.meshgrid(theta_f, zeta_f, indexing="ij")
    Xb, Yb, Zb = _positions(np.full_like(tg, X_MAX), tg, zg)
    ax.plot_surface(Xb, Yb, Zb, color="lightsteelblue", alpha=0.10, shade=False,
                    rstride=2, cstride=2, linewidth=0, antialiased=False)

    # Closed core field lines: theta = theta0 + iota * zeta, three toroidal turns.
    zeta_line = np.linspace(0.0, 6.0 * np.pi, 900)
    for theta0 in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False):
        X, Y, Z = _positions(np.full_like(zeta_line, 0.45), theta0 + IOTA * zeta_line, zeta_line)
        ax.plot(X, Y, Z, color="#1f77b4", lw=0.9, alpha=0.9)

    # Open SOL field lines: one transit, ending on the limiter at zeta = 0 / 2 pi.
    zeta_open = np.linspace(0.0, 2.0 * np.pi, 300)
    for theta0 in np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False):
        X, Y, Z = _positions(np.full_like(zeta_open, 0.82), theta0 + IOTA * zeta_open, zeta_open)
        ax.plot(X, Y, Z, color="#d62728", lw=1.3)
        ax.scatter(X[[0, -1]], Y[[0, -1]], Z[[0, -1]], color="#d62728", s=14)

    # The limiter: a translucent annulus at the zeta = 0 poloidal plane.
    x_ann = np.linspace(LIMITER_RADIUS, X_MAX, 8)
    ta, xa = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, 60), x_ann, indexing="ij")
    Xl, Yl, Zl = _positions(xa, ta, np.zeros_like(xa))
    ax.plot_surface(Xl, Yl, Zl, color="dimgray", alpha=0.45, shade=False, linewidth=0)

    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=26, azim=-50)
    ax.set_xlim(-3.4, 3.4), ax.set_ylim(-3.4, 3.4), ax.set_zlim(-1.5, 1.5)
    ax.set_title("Rotating-ellipse stellarator: closed core field lines (blue) and\n"
                 "open scrape-off-layer field lines ending on the limiter (red)",
                 fontsize=10, y=0.92)
    fig.subplots_adjust(left=0, right=1, bottom=-0.12, top=1.05)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_field_lines(OUTPUT_DIR / "stellarator_3d_field_lines.png")

    geometry = build_rotating_ellipse_geometry(SHAPE, r0=R0, elongation=ELONGATION,
                                               n_field_periods=NFP, iota=IOTA)
    cache = OUTPUT_DIR / "frames.npz"
    if cache.exists():
        data = np.load(cache)
        from stellarator_turbulence_case import TurbulenceRun
        run = TurbulenceRun(data["density"], data["omega"], data["times"],
                            data["content"], data["flux"])
        print("loaded cached turbulence frames")
    else:
        run = run_stellarator_turbulence(geometry, steps=STEPS, dt=DT, seed=1,
                                         frame_stride=FRAME_STRIDE)
        np.savez_compressed(cache, density=run.density_frames, omega=run.omega_frames,
                            times=run.times, content=run.particle_content, flux=run.target_flux)
        print("turbulence run done")
    render_turbulence_movie(run, geometry, OUTPUT_DIR / "stellarator_3d_turbulence.gif")


if __name__ == "__main__":
    main()
