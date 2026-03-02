from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from jaxdrb.io import load_config

from plot_utils import maybe_lowpass


def _extract_frames(
    arr: np.ndarray, target_shape: tuple[int, int] | None, y_index: int | None
) -> np.ndarray:
    if arr.ndim == 3:
        # (t, nx, ny) plane
        return arr
    if arr.ndim != 4:
        raise ValueError(f"Unsupported snapshot shape: {arr.shape}")

    # Try all possible slice axes for (t, a, b, c) and match plotting grid.
    if target_shape is not None:
        axes = [1, 3, 2] if y_index is not None else [1, 2, 3]
        for axis in axes:
            idx = y_index if (y_index is not None and axis == axes[0]) else arr.shape[axis] // 2
            idx = int(np.clip(idx, 0, arr.shape[axis] - 1))
            frames = np.take(arr, idx, axis=axis)
            shape2 = frames.shape[1:]
            if shape2 == target_shape:
                return frames
            if shape2[::-1] == target_shape:
                return frames.transpose(0, 2, 1)

    # Fallback to legacy behavior.
    y0 = y_index if y_index is not None else arr.shape[-1] // 2
    return arr[:, :, :, int(np.clip(y0, 0, arr.shape[-1] - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a poloidal (R,Z) GIF movie.")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--config", required=True, help="Path to input TOML")
    parser.add_argument("--field", default="snapshots_n", help="Snapshot field key")
    parser.add_argument("--out", default="docs/figures/poloidal_movie.gif", help="Output GIF")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--y-index", type=int, default=None, help="Toroidal index for 3D fields")
    parser.add_argument(
        "--separatrix",
        type=float,
        default=None,
        help="Optional separatrix radius (same units as Lx).",
    )
    parser.add_argument(
        "--fluct",
        default="none",
        choices=("none", "mean", "zonal"),
        help="Subtract mean or zonal mean before plotting.",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Optional low-pass fraction (0-1) for smoother visuals.",
    )
    parser.add_argument(
        "--skip-fraction",
        type=float,
        default=0.0,
        help="Skip this fraction of early frames (0-1).",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use symmetric color limits about zero.",
    )
    parser.add_argument(
        "--range-tail",
        action="store_true",
        help="Compute color limits from the last portion of frames.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.2,
        help="Fraction of final frames to use for color limits.",
    )
    parser.add_argument(
        "--field-scale",
        type=float,
        default=1.0,
        help="Multiply the plotted field by this factor.",
    )
    parser.add_argument(
        "--range-scale",
        type=float,
        default=0.7,
        help="Scale factor applied to vmin/vmax for contrast.",
    )
    parser.add_argument(
        "--interp-grid",
        type=int,
        default=320,
        help="Interpolation grid resolution for smoother poloidal movies (0 to disable).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    geom = cfg.data.get("geometry", {})
    kind = str(geom.get("kind", "plane")).lower()
    R0 = float(geom.get("R0", 3.0))
    r_minor = float(geom.get("r_minor", geom.get("Lx", 1.0)))
    Lx = float(geom.get("Lx", r_minor))

    coeff_path = geom.get("coeff_path") or geom.get("coefficients")
    coeffs = None
    if kind.startswith("axisymmetric") and coeff_path is not None:
        try:
            coeffs = np.load(coeff_path)
        except OSError:
            coeffs = None

    if kind == "plane":
        nx = int(geom.get("nx", 64))
        ny = int(geom.get("ny", 64))
        Ly = float(geom.get("Ly", 2.0 * np.pi))
        x = np.linspace(0.0, Lx, nx, endpoint=True)
        theta = np.linspace(0.0, Ly, ny, endpoint=False)
    else:
        nx = int(geom.get("nx", 64))
        nz = int(geom.get("nz", 64))
        theta_scale = float(geom.get("theta_scale", 1.0))
        Lz = float(geom.get("Lz", 2.0 * np.pi * theta_scale))
        x = np.linspace(0.0, Lx, nx, endpoint=True)
        if coeffs is not None and "z" in coeffs:
            z = np.asarray(coeffs["z"]).reshape(-1)
            nz = int(z.size)
        else:
            z = np.linspace(-0.5 * Lz, 0.5 * Lz, nz, endpoint=True)
        theta = z / max(theta_scale, 1e-8)

    if coeffs is not None and ("Rxy" in coeffs) and ("Zxy" in coeffs):
        R = np.asarray(coeffs["Rxy"]).T
        Z = np.asarray(coeffs["Zxy"]).T
    else:
        r = x * (r_minor / max(Lx, 1e-8))
        R = R0 + r[None, :] * np.cos(theta[:, None])
        Z = r[None, :] * np.sin(theta[:, None])

    data = np.load(args.input)
    if args.field not in data:
        raise ValueError(f"Missing '{args.field}' in output.")
    frames = _extract_frames(
        np.asarray(data[args.field]), target_shape=R.shape, y_index=args.y_index
    )
    frames = frames[:: max(args.stride, 1)]
    if args.skip_fraction > 0.0:
        start = int(max(0, min(frames.shape[0] - 1, args.skip_fraction * frames.shape[0])))
        frames = frames[start:]

    if kind == "plane":
        frames = frames.transpose(0, 2, 1)

    if args.fluct != "none":
        if args.fluct == "mean":
            frames = frames - frames.mean(axis=(1, 2), keepdims=True)
        elif args.fluct == "zonal":
            frames = frames - frames.mean(axis=2, keepdims=True)
    if args.lowpass:
        frames = np.stack([maybe_lowpass(frame, args.lowpass) for frame in frames], axis=0)

    frames = frames * float(args.field_scale)
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    tri = mtri.Triangulation(R.ravel(), Z.ravel())
    interp_grid = int(args.interp_grid)
    if interp_grid > 0:
        r_lin = np.linspace(R.min(), R.max(), interp_grid)
        z_lin = np.linspace(Z.min(), Z.max(), interp_grid)
        Rg, Zg = np.meshgrid(r_lin, z_lin)
    else:
        r_lin = None
        z_lin = None
        Rg = None
        Zg = None

    range_frames = frames
    if args.range_tail and frames.shape[0] > 1:
        frac = min(max(float(args.tail_fraction), 0.05), 1.0)
        start = int(max(0, frames.shape[0] * (1.0 - frac)))
        range_frames = frames[start:]
    vmin, vmax = np.percentile(range_frames, [2.0, 98.0])
    if args.symmetric:
        vmax = float(max(abs(vmin), abs(vmax)))
        vmin = -vmax
    scale = float(args.range_scale)
    vmin = vmin * scale
    vmax = vmax * scale

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if interp_grid > 0:
        interp = mtri.LinearTriInterpolator(tri, frames[0].ravel())
        vals = interp(Rg, Zg)
        vals = np.ma.masked_invalid(vals)
        im = ax.imshow(
            vals,
            origin="lower",
            extent=(r_lin.min(), r_lin.max(), z_lin.min(), z_lin.max()),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
    else:
        im = ax.tripcolor(
            tri,
            frames[0].ravel(),
            shading="gouraud",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
    ax.set_title(args.field.replace("snapshots_", ""))
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect("equal")
    if coeffs is not None and "mask_open" in coeffs:
        mask_open = np.asarray(coeffs["mask_open"]).T
        try:
            ax.contour(
                R,
                Z,
                mask_open,
                levels=[0.5],
                colors="white",
                linewidths=1.0,
                linestyles="--",
            )
        except Exception:
            pass
    elif args.separatrix is not None:
        sep = float(args.separatrix)
        ax.plot(R0 + sep * np.cos(theta), sep * np.sin(theta), color="white", lw=1.2, ls="--")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(args.field.replace("snapshots_", ""))

    try:
        from matplotlib import animation

        def update(i):
            if interp_grid > 0:
                interp = mtri.LinearTriInterpolator(tri, frames[i].ravel())
                vals = interp(Rg, Zg)
                vals = np.ma.masked_invalid(vals)
                im.set_data(vals)
            else:
                im.set_array(frames[i].ravel())
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=frames.shape[0], blit=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer="pillow", fps=12)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
