from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jaxdrb.io import load_config


def _find_field(data: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    if field in data:
        arr = np.asarray(data[field])
    elif f"snapshots_{field}" in data:
        arr = np.asarray(data[f"snapshots_{field}"])
    else:
        raise ValueError(f"Missing '{field}' or 'snapshots_{field}' in output.")
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D snapshots (t,nz,nx,ny), got shape {arr.shape}")
    return arr


def _nearest_idx(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 3D tokamak turbulence movie with poloidal/toroidal cuts."
    )
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--config", required=True, help="Input TOML")
    parser.add_argument("--field", default="snapshots_n", help="4D snapshot field key")
    parser.add_argument("--out", default="docs/figures/tokamak_sol_3d_movie.gif", help="Output GIF")
    parser.add_argument("--time-stride", type=int, default=6, help="Frame stride in time")
    parser.add_argument(
        "--skip-fraction", type=float, default=0.3, help="Skip initial fraction of frames"
    )
    parser.add_argument(
        "--fluct",
        choices=("none", "mean", "zonal"),
        default="zonal",
        help="Fluctuation mode for coloring.",
    )
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric color limits")
    parser.add_argument("--range-tail", action="store_true", help="Use only tail for color limits")
    parser.add_argument("--tail-fraction", type=float, default=0.3, help="Tail fraction for limits")
    parser.add_argument(
        "--phi-cut-1", type=float, default=-np.pi / 6.0, help="First toroidal cut angle [rad]"
    )
    parser.add_argument(
        "--phi-cut-2", type=float, default=np.pi / 6.0, help="Second toroidal cut angle [rad]"
    )
    parser.add_argument(
        "--theta-cut", type=float, default=np.pi, help="Poloidal angle for toroidal cut [rad]"
    )
    parser.add_argument("--x-step", type=int, default=1, help="Radial decimation")
    parser.add_argument("--z-step", type=int, default=1, help="Poloidal decimation")
    parser.add_argument("--y-step", type=int, default=1, help="Toroidal decimation")
    parser.add_argument("--fps", type=int, default=10, help="GIF framerate")
    parser.add_argument("--dpi", type=int, default=120, help="Render DPI")
    args = parser.parse_args()

    cfg = load_config(args.config)
    geom = cfg.data.get("geometry", {})
    coeff_path = geom.get("coeff_path") or geom.get("coefficients")
    if coeff_path is None:
        raise ValueError("3D tokamak movie requires geometry coeff_path.")
    coeffs = np.load(coeff_path)
    if "Rxy" not in coeffs or "Zxy" not in coeffs:
        raise ValueError("Coefficient file must include Rxy and Zxy.")

    R = np.asarray(coeffs["Rxy"]).T  # (nz, nx)
    Z = np.asarray(coeffs["Zxy"]).T  # (nz, nx)
    z_coord = np.asarray(coeffs["z"]).reshape(-1)
    theta_scale = float(geom.get("theta_scale", 1.0))
    theta = z_coord / max(theta_scale, 1e-8)
    ny = int(geom.get("ny", R.shape[1]))
    Ly = float(geom.get("Ly", 2.0 * np.pi))
    phi = np.linspace(0.0, Ly, ny, endpoint=False) * (2.0 * np.pi / max(Ly, 1e-12))

    data = np.load(args.input)
    frames = _find_field(data, args.field)
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    if args.fluct == "mean":
        frames = frames - frames.mean(axis=(1, 2, 3), keepdims=True)
    elif args.fluct == "zonal":
        frames = frames - frames.mean(axis=3, keepdims=True)

    t0 = int(min(max(frames.shape[0] - 1, 0), args.skip_fraction * frames.shape[0]))
    frames = frames[t0 :: max(args.time_stride, 1)]
    if frames.shape[0] == 0:
        raise ValueError("No frames selected after skip/stride.")

    range_frames = frames
    if args.range_tail and frames.shape[0] > 1:
        frac = float(np.clip(args.tail_fraction, 0.05, 1.0))
        i0 = int((1.0 - frac) * frames.shape[0])
        range_frames = frames[i0:]
    vmin, vmax = np.percentile(range_frames, [2.0, 98.0])
    if args.symmetric:
        vmax = float(max(abs(vmin), abs(vmax)))
        vmin = -vmax
    norm = plt.Normalize(vmin=float(vmin), vmax=float(vmax))
    cmap = plt.get_cmap("coolwarm")

    sx = max(int(args.x_step), 1)
    sy = max(int(args.y_step), 1)
    sz = max(int(args.z_step), 1)

    R_sub = R[::sz, ::sx]
    Z_sub = Z[::sz, ::sx]
    phi_sub = phi[::sy]

    iy1 = _nearest_idx(phi, float(args.phi_cut_1))
    iy2 = _nearest_idx(phi, float(args.phi_cut_2))
    iz = _nearest_idx(theta, float(args.theta_cut))

    phi1 = float(phi[iy1])
    phi2 = float(phi[iy2])

    X_phi1 = R_sub * np.cos(phi1)
    Y_phi1 = R_sub * np.sin(phi1)
    X_phi2 = R_sub * np.cos(phi2)
    Y_phi2 = R_sub * np.sin(phi2)

    R_tor = R[iz, ::sx]
    Z_tor = Z[iz, ::sx]
    Phi_g, R_g = np.meshgrid(phi_sub, R_tor, indexing="xy")
    Z_g = np.repeat(Z_tor[:, None], phi_sub.size, axis=1)
    X_tor = R_g * np.cos(Phi_g)
    Y_tor = R_g * np.sin(Phi_g)

    x_all = np.concatenate([X_phi1.ravel(), X_phi2.ravel(), X_tor.ravel()])
    y_all = np.concatenate([Y_phi1.ravel(), Y_phi2.ravel(), Y_tor.ravel()])
    z_all = np.concatenate([Z_sub.ravel(), Z_sub.ravel(), Z_g.ravel()])
    pad = 0.05 * max(np.ptp(x_all), np.ptp(y_all), np.ptp(z_all), 1e-6)

    images: list[Image.Image] = []
    times = np.asarray(data.get("times", np.arange(frames.shape[0], dtype=float)))
    if times.size >= t0 + frames.shape[0] * max(args.time_stride, 1):
        times = times[t0 :: max(args.time_stride, 1)][: frames.shape[0]]
    else:
        times = np.arange(frames.shape[0], dtype=float)

    for i, frame in enumerate(frames):
        fig = plt.figure(figsize=(9.5, 7.5), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        vals_phi1 = frame[::sz, ::sx, iy1]
        vals_phi2 = frame[::sz, ::sx, iy2]
        vals_tor = frame[iz, ::sx, ::sy]

        ax.plot_surface(
            X_phi1,
            Y_phi1,
            Z_sub,
            facecolors=cmap(norm(vals_phi1)),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.95,
        )
        ax.plot_surface(
            X_phi2,
            Y_phi2,
            Z_sub,
            facecolors=cmap(norm(vals_phi2)),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.95,
        )
        ax.plot_surface(
            X_tor,
            Y_tor,
            Z_g,
            facecolors=cmap(norm(vals_tor)),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.85,
        )

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(args.field.replace("snapshots_", ""))

        ax.set_title(
            f"Tokamak turbulence cuts: "
            f"$\\phi={phi1:+.2f}, {phi2:+.2f}$ rad, "
            f"$\\theta={theta[iz]:+.2f}$ rad, "
            f"t={times[i]:.3f}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(x_all.min() - pad, x_all.max() + pad)
        ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
        ax.set_zlim(z_all.min() - pad, z_all.max() + pad)
        ax.set_box_aspect((np.ptp(x_all), np.ptp(y_all), np.ptp(z_all)))
        ax.view_init(elev=20, azim=35 + 0.8 * i)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images.append(Image.fromarray(rgba[:, :, :3]))
        plt.close(fig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(args.fps, 1)))
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
