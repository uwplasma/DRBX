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


def _physical_plane(geometry, z_index):
    x = np.asarray(geometry.grid.x.centers)
    theta = np.asarray(geometry.grid.y.centers)
    zeta = float(geometry.grid.z.centers[z_index])
    xx, tt = np.meshgrid(x, theta, indexing="ij")
    position = np.asarray(rotating_ellipse_position(
        jax.numpy.asarray(xx), jax.numpy.asarray(tt), jax.numpy.asarray(zeta),
        r0=R0, elongation=ELONGATION, n_field_periods=NFP))
    return np.hypot(position[..., 0], position[..., 1]), position[..., 2], zeta


def save_movie(run, geometry, title, path):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    z_indices = (SHAPE[2] // 4, (3 * SHAPE[2]) // 4)
    fluctuation = run.density_frames - 1.0
    vmax = float(np.abs(fluctuation).max()) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    meshes = []
    for ax, z_index in zip(axes, z_indices):
        R, Z, zeta = _physical_plane(geometry, z_index)
        mesh = ax.pcolormesh(R, Z, fluctuation[0][:, :, z_index], cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(f"zeta = {zeta:.2f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        meshes.append((mesh, z_index))
    suptitle = fig.suptitle(f"{title}  t = 0.000")
    fig.tight_layout()

    def update(frame_index):
        for mesh, z_index in meshes:
            mesh.set_array(fluctuation[frame_index][:, :, z_index].ravel())
        suptitle.set_text(f"{title}  t = {run.times[frame_index]:.3f}")
        return [m for m, _ in meshes]

    movie = animation.FuncAnimation(fig, update, frames=len(run.times), interval=100, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=10), dpi=90)
    plt.close(fig)
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    closed_geometry = build_rotating_ellipse_geometry(SHAPE, r0=R0, elongation=ELONGATION,
                                                      n_field_periods=NFP)
    open_geometry = build_rotating_ellipse_geometry(SHAPE, r0=R0, elongation=ELONGATION,
                                                    n_field_periods=NFP, limiter_radius=LIMITER_RADIUS)

    closed = run_stellarator_turbulence(closed_geometry, steps=STEPS, dt=DT, seed=1,
                                        frame_stride=FRAME_STRIDE)
    print("closed run done")
    open_run = run_stellarator_turbulence(open_geometry, steps=STEPS, dt=DT, seed=1,
                                          sheath_sink=True, frame_stride=FRAME_STRIDE)
    print("open run done")

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
