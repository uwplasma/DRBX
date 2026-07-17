"""Stellarator turbulence movies: closed and open field lines.

Runs the multi-mode seeded four-field interchange model on the rotating-ellipse
stellarator flux tube twice -- once with all field lines closed, once with a
toroidal limiter opening the outer flux surfaces into a scrape-off layer drained
by a Bohm sheath -- and renders a compressed GIF of the density fluctuations in
two rotating physical cross-sections for each, plus a summary figure comparing
particle content and limiter flux.

Run:

    PYTHONPATH=src python examples/stellarator/stellarator_turbulence_demo.py

writes ``output/stellarator_turbulence/`` with two GIFs and a PNG. Movies are
release-hosted, not committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from jax_drb.geometry import build_rotating_ellipse_geometry, rotating_ellipse_position  # noqa: E402

from stellarator_turbulence_case import run_stellarator_turbulence  # noqa: E402

SHAPE = (20, 32, 12)
STEPS = 144
DT = 2.0e-3
FRAME_STRIDE = 4
LIMITER_RADIUS = 0.6
R0, ELONGATION, NFP = 3.0, 0.35, 1
OUTPUT_DIR = Path("output/stellarator_turbulence")


UPSAMPLE = 4  # smooth (non-pixelated) rendering: interpolate the coarse grid


def _fine_axes(geometry):
    x_coarse = np.asarray(geometry.grid.x.centers)
    theta_coarse = np.asarray(geometry.grid.y.centers)
    x_fine = np.linspace(x_coarse[0], x_coarse[-1], len(x_coarse) * UPSAMPLE)
    theta_fine = np.linspace(0.0, 2.0 * np.pi, len(theta_coarse) * UPSAMPLE + 1)
    return x_coarse, theta_coarse, x_fine, theta_fine


def _upsample_plane(field_xt, x_coarse, theta_coarse, x_fine, theta_fine):
    """Periodic interpolation in theta, linear in x, onto the fine grid."""

    fine_theta = np.apply_along_axis(
        lambda row: np.interp(theta_fine, theta_coarse, row, period=2.0 * np.pi), 1, field_xt)
    return np.apply_along_axis(lambda col: np.interp(x_fine, x_coarse, col), 0, fine_theta)


def _physical_plane_fine(geometry, zeta, x_fine, theta_fine):
    xx, tt = np.meshgrid(x_fine, theta_fine, indexing="ij")
    position = np.asarray(rotating_ellipse_position(
        jax.numpy.asarray(xx), jax.numpy.asarray(tt), jax.numpy.asarray(zeta),
        r0=R0, elongation=ELONGATION, n_field_periods=NFP))
    return np.hypot(position[..., 0], position[..., 1]), position[..., 2]


def save_movie(run, geometry, title, path):
    """One row of four toroidal cross-sections, smoothly interpolated."""

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    nz = geometry.shape[2]
    z_indices = [0, nz // 4, nz // 2, (3 * nz) // 4]
    zeta_values = [float(geometry.grid.z.centers[k]) for k in z_indices]
    fluctuation = run.density_frames - 1.0
    vmax = float(np.abs(fluctuation).max()) or 1.0
    x_coarse, theta_coarse, x_fine, theta_fine = _fine_axes(geometry)

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5), constrained_layout=True)
    meshes = []
    for ax, z_index, zeta in zip(axes, z_indices, zeta_values):
        R, Z = _physical_plane_fine(geometry, zeta, x_fine, theta_fine)
        field = _upsample_plane(fluctuation[0][:, :, z_index], x_coarse, theta_coarse, x_fine, theta_fine)
        mesh = ax.pcolormesh(R, Z, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"zeta = {zeta:.2f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        meshes.append((mesh, z_index))
    fig.colorbar(meshes[-1][0], ax=axes, shrink=0.85, label="density fluctuation")
    suptitle = fig.suptitle(f"{title}   t = 0.000")

    def update(frame_index):
        for mesh, z_index in meshes:
            field = _upsample_plane(fluctuation[frame_index][:, :, z_index],
                                    x_coarse, theta_coarse, x_fine, theta_fine)
            mesh.set_array(field.ravel())
        suptitle.set_text(f"{title}   t = {run.times[frame_index]:.3f}")
        return [m for m, _ in meshes]

    movie = animation.FuncAnimation(fig, update, frames=len(run.times), interval=100, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=10), dpi=88)
    plt.close(fig)
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    closed_geometry = build_rotating_ellipse_geometry(SHAPE, r0=R0, elongation=ELONGATION,
                                                      n_field_periods=NFP)
    open_geometry = build_rotating_ellipse_geometry(SHAPE, r0=R0, elongation=ELONGATION,
                                                    n_field_periods=NFP, limiter_radius=LIMITER_RADIUS)

    from stellarator_turbulence_case import TurbulenceRun

    def cached_run(name, geometry, **kwargs):
        cache = OUTPUT_DIR / f"{name}_frames.npz"
        if cache.exists():
            data = np.load(cache)
            print(f"loaded cached {name} frames")
            return TurbulenceRun(data["density"], data["omega"], data["times"],
                                 data["content"], data["flux"])
        run = run_stellarator_turbulence(geometry, steps=STEPS, dt=DT, seed=1,
                                         frame_stride=FRAME_STRIDE, **kwargs)
        np.savez_compressed(cache, density=run.density_frames, omega=run.omega_frames,
                            times=run.times, content=run.particle_content, flux=run.target_flux)
        print(f"{name} run done")
        return run

    closed = cached_run("closed", closed_geometry)
    open_run = cached_run("open", open_geometry, sheath_sink=True)

    save_movie(closed, closed_geometry, "Stellarator turbulence (closed field lines)",
               OUTPUT_DIR / "stellarator_turbulence_closed.gif")
    save_movie(open_run, open_geometry, "Stellarator SOL turbulence (open field lines)",
               OUTPUT_DIR / "stellarator_turbulence_open.gif")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(closed.times, closed.particle_content / closed.particle_content[0],
                 label="closed", color="#1f77b4")
    axes[0].plot(open_run.times, open_run.particle_content / open_run.particle_content[0],
                 label="open (limiter SOL)", color="#d62728")
    axes[0].set_xlabel("time"), axes[0].set_ylabel("particle content (norm.)")
    axes[0].set_title("Open field lines drain to the limiter")
    axes[0].legend(fontsize=8), axes[0].grid(True, ls=":", alpha=0.4)
    axes[1].plot(open_run.times, open_run.target_flux, color="#d62728")
    axes[1].set_xlabel("time"), axes[1].set_ylabel("total limiter ion flux")
    axes[1].set_title("Bohm sheath flux at the limiter")
    axes[1].grid(True, ls=":", alpha=0.4)
    fig.suptitle("Stellarator turbulence: closed vs open field lines (rotating ellipse)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stellarator_turbulence_summary.png", dpi=170)
    plt.close(fig)

    (OUTPUT_DIR / "summary.json").write_text(json.dumps({
        "steps": STEPS, "dt": DT, "shape": list(SHAPE), "limiter_radius": LIMITER_RADIUS,
        "closed_content_change": float(closed.particle_content[-1] - closed.particle_content[0]),
        "open_content_change": float(open_run.particle_content[-1] - open_run.particle_content[0]),
        "final_limiter_flux": float(open_run.target_flux[-1]),
        "closed_omega_max": float(np.abs(closed.omega_frames[-1]).max()),
    }, indent=2))
    print(f"wrote {OUTPUT_DIR / 'stellarator_turbulence_summary.png'} and summary.json")


if __name__ == "__main__":
    main()
