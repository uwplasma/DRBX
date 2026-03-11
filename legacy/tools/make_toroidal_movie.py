from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from jaxdrb.io import load_config


def _extract_frames(arr: np.ndarray, y_index: int | None) -> np.ndarray:
    if arr.ndim == 4:
        # (t, nz, nx, ny) field-aligned
        return arr
    raise ValueError(f"Unsupported snapshot shape: {arr.shape}")


def _lowpass_2d(field: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0.0 or frac >= 1.0:
        return field
    nx, ny = field.shape
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    k2 = kxg**2 + kyg**2
    k2_max = float(np.max(k2))
    if k2_max <= 0.0:
        return field
    mask = k2 <= (frac**2) * k2_max
    fhat = np.fft.fftn(field)
    fhat *= mask
    return np.fft.ifftn(fhat).real


def _lowpass_3d(field: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0.0 or frac >= 1.0:
        return field
    nz, nx, ny = field.shape
    kz = np.fft.fftfreq(nz)
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kz_g, kx_g, ky_g = np.meshgrid(kz, kx, ky, indexing="ij")
    k2 = kz_g**2 + kx_g**2 + ky_g**2
    k2_max = float(np.max(k2))
    if k2_max <= 0.0:
        return field
    mask = k2 <= (frac**2) * k2_max
    fhat = np.fft.fftn(field)
    fhat *= mask
    return np.fft.ifftn(fhat).real


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a two-panel toroidal geometry GIF (poloidal + toroidal cut)."
    )
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--config", required=True, help="Path to input TOML")
    parser.add_argument("--field", default="snapshots_n", help="Snapshot field key")
    parser.add_argument("--out", default="docs/figures/toroidal_movie.gif", help="Output GIF")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--y-index", type=int, default=None, help="Toroidal index")
    parser.add_argument("--z-index", type=int, default=None, help="Poloidal index")
    parser.add_argument(
        "--toroidal-theta",
        type=float,
        default=None,
        help="Poloidal angle (radians) to select for the toroidal cut.",
    )
    parser.add_argument(
        "--toroidal-theta2",
        type=float,
        default=None,
        help="Optional second poloidal angle (radians) for an inboard/outboard comparison.",
    )
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
        help="Interpolation grid resolution for smoother poloidal plots (0 to disable).",
    )
    parser.add_argument(
        "--toroidal-mode",
        choices=("polar", "rphi"),
        default="polar",
        help="Display toroidal cut in polar (annulus) or R-phi plane.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    geom = cfg.data.get("geometry", {})
    kind = str(geom.get("kind", "salpha")).lower()
    if kind == "plane":
        raise ValueError("Toroidal movie requires a 3D field-aligned geometry.")

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
    nx = int(geom.get("nx", 64))
    ny = int(geom.get("ny", 64))
    nz = int(geom.get("nz", 64))
    Ly = float(geom.get("Ly", 2.0 * np.pi))
    theta_scale = float(geom.get("theta_scale", 1.0))
    Lz = float(geom.get("Lz", 2.0 * np.pi * theta_scale))

    x = np.linspace(0.0, Lx, nx, endpoint=True)
    y = np.linspace(0.0, Ly, ny, endpoint=False)
    if coeffs is not None and "z" in coeffs:
        z = np.asarray(coeffs["z"]).reshape(-1)
        nz = int(z.size)
    else:
        z = np.linspace(-0.5 * Lz, 0.5 * Lz, nz, endpoint=True)
    theta = z / max(theta_scale, 1e-8)
    phi = y * (2.0 * np.pi / max(Ly, 1e-8))

    data = np.load(args.input)
    if args.field not in data:
        raise ValueError(f"Missing '{args.field}' in output.")
    frames = _extract_frames(np.asarray(data[args.field]), args.y_index)
    frames = frames[:: max(args.stride, 1)]
    if args.skip_fraction > 0.0:
        start = int(max(0, min(frames.shape[0] - 1, args.skip_fraction * frames.shape[0])))
        frames = frames[start:]

    if args.fluct != "none":
        if args.fluct == "mean":
            frames = frames - frames.mean(axis=(1, 2, 3), keepdims=True)
        elif args.fluct == "zonal":
            frames = frames - frames.mean(axis=3, keepdims=True)
    frames = frames * float(args.field_scale)

    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    if coeffs is not None and ("Rxy" in coeffs) and ("Zxy" in coeffs):
        R = np.asarray(coeffs["Rxy"]).T
        Z = np.asarray(coeffs["Zxy"]).T
    else:
        r = x * (r_minor / max(Lx, 1e-8))
        R = R0 + r[None, :] * np.cos(theta[:, None])
        Z = r[None, :] * np.sin(theta[:, None])
    tri = mtri.Triangulation(R.ravel(), Z.ravel())
    interp_grid = int(args.interp_grid)
    if interp_grid > 0:
        r_lin = np.linspace(R.min(), R.max(), interp_grid)
        z_lin = np.linspace(Z.min(), Z.max(), interp_grid)
        Rg_pol, Zg_pol = np.meshgrid(r_lin, z_lin)
    else:
        r_lin = None
        z_lin = None
        Rg_pol = None
        Zg_pol = None

    range_frames = frames
    if args.range_tail and frames.shape[0] > 1:
        frac = min(max(float(args.tail_fraction), 0.05), 1.0)
        start = int(max(0, frames.shape[0] * (1.0 - frac)))
        range_frames = frames[start:]
    energy = np.mean(range_frames**2, axis=(1, 2, 3))
    if energy.size > 1:
        cutoff = np.percentile(energy, 70.0)
        range_frames = range_frames[energy >= cutoff]
    # Estimate range from representative poloidal+toroidal cuts with max variance.
    y_var = np.var(range_frames, axis=(0, 1, 2))
    y_idx = int(np.argmax(y_var)) if args.y_index is None else int(args.y_index)
    if args.z_index is not None:
        z_idx = int(args.z_index)
    elif args.toroidal_theta is not None:
        z_idx = int(np.argmin(np.abs(theta - float(args.toroidal_theta))))
    elif coeffs is not None and ("Rxy" in coeffs):
        r_mean = np.mean(R, axis=1)
        z_idx = int(np.argmax(r_mean))
    else:
        z_var = np.var(range_frames, axis=(0, 2, 3))
        z_idx = int(np.argmax(z_var))
    z_idx2 = None
    if args.toroidal_theta2 is not None:
        z_idx2 = int(np.argmin(np.abs(theta - float(args.toroidal_theta2))))
    elif coeffs is not None and ("Rxy" in coeffs):
        r_mean = np.mean(R, axis=1)
        z_idx2 = int(np.argmin(r_mean))
    pol_samples = range_frames[:, :, :, y_idx]
    tor_samples = range_frames[:, z_idx]
    tor_samples2 = range_frames[:, z_idx2] if z_idx2 is not None else None
    if args.lowpass:
        pol_samples = np.stack(
            [_lowpass_2d(frame, float(args.lowpass)) for frame in pol_samples], axis=0
        )
        tor_samples = np.stack(
            [_lowpass_2d(frame, float(args.lowpass)) for frame in tor_samples], axis=0
        )
        if tor_samples2 is not None:
            tor_samples2 = np.stack(
                [_lowpass_2d(frame, float(args.lowpass)) for frame in tor_samples2], axis=0
            )
    sample_parts = [pol_samples.ravel(), tor_samples.ravel()]
    if tor_samples2 is not None:
        sample_parts.append(tor_samples2.ravel())
    sample = np.concatenate(sample_parts)
    vmin, vmax = np.percentile(sample, [2.0, 98.0])
    if args.symmetric:
        vmax = float(max(abs(vmin), abs(vmax)))
        vmin = -vmax
    cmap = "coolwarm" if args.symmetric else "viridis"
    scale = float(args.range_scale)
    vmin = vmin * scale
    vmax = vmax * scale

    theta0 = theta[z_idx]
    if coeffs is not None and ("Rxy" in coeffs):
        R_tor = R[z_idx, :]
    else:
        r = x * (r_minor / max(Lx, 1e-8))
        R_tor = R0 + r * np.cos(theta0)
    Phi, Rg = np.meshgrid(phi, R_tor, indexing="xy")
    toroidal_slice2 = frames[0, z_idx2] if z_idx2 is not None else None
    rmin = float(np.min(R_tor))
    rmax = float(np.max(R_tor))
    if z_idx2 is not None:
        if coeffs is not None and ("Rxy" in coeffs):
            R_tor2_preview = R[z_idx2, :]
        else:
            r = x * (r_minor / max(Lx, 1e-8))
            R_tor2_preview = R0 + r * np.cos(theta[z_idx2])
        rmin = float(min(rmin, np.min(R_tor2_preview)))
        rmax = float(max(rmax, np.max(R_tor2_preview)))

    if toroidal_slice2 is None:
        fig = plt.figure(figsize=(12.0, 5.0), constrained_layout=True)
        ax0 = fig.add_subplot(1, 2, 1)
    else:
        fig = plt.figure(figsize=(16.0, 5.0), constrained_layout=True)
        ax0 = fig.add_subplot(1, 3, 1)
    poloidal = frames[0, :, :, y_idx]
    toroidal = frames[0, z_idx]
    if args.lowpass:
        poloidal = _lowpass_2d(poloidal, float(args.lowpass))
        toroidal = _lowpass_2d(toroidal, float(args.lowpass))
        if toroidal_slice2 is not None:
            toroidal_slice2 = _lowpass_2d(toroidal_slice2, float(args.lowpass))
    if interp_grid > 0:
        interp = mtri.LinearTriInterpolator(tri, poloidal.ravel())
        vals = interp(Rg_pol, Zg_pol)
        vals = np.ma.masked_invalid(vals)
        im0 = ax0.imshow(
            vals,
            origin="lower",
            extent=(r_lin.min(), r_lin.max(), z_lin.min(), z_lin.max()),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
    else:
        im0 = ax0.pcolormesh(R, Z, poloidal, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax0.set_title(f"poloidal cut ({args.field.replace('snapshots_', '')})")
    ax0.set_xlabel("R")
    ax0.set_ylabel("Z")
    ax0.set_aspect("equal")
    if coeffs is not None and "mask_open" in coeffs:
        mask_open = np.asarray(coeffs["mask_open"]).T
        try:
            ax0.contour(
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
        ax0.plot(R0 + sep * np.cos(theta), sep * np.sin(theta), color="white", lw=1.2, ls="--")

    if args.toroidal_mode == "polar":
        ax1 = fig.add_subplot(1, 2 if toroidal_slice2 is None else 3, 2, projection="polar")
        im1 = ax1.pcolormesh(Phi, Rg, toroidal, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title("toroidal cut (outboard)")
        ax1.set_rmin(rmin)
        ax1.set_rmax(rmax)
        ax1.set_ylim(rmin, rmax)
        ax1.set_facecolor("white")
        ax1.set_yticklabels([])
        ax1.grid(alpha=0.3)
        ax2 = None
        if toroidal_slice2 is not None:
            theta2 = theta[z_idx2]
            if coeffs is not None and ("Rxy" in coeffs):
                R_tor2 = R[z_idx2, :]
            else:
                r = x * (r_minor / max(Lx, 1e-8))
                R_tor2 = R0 + r * np.cos(theta2)
            Phi2, Rg2 = np.meshgrid(phi, R_tor2, indexing="xy")
            ax2 = fig.add_subplot(1, 3, 3, projection="polar")
            im2 = ax2.pcolormesh(
                Phi2,
                Rg2,
                toroidal_slice2,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax2.set_title("toroidal cut (inboard)")
            ax2.set_rmin(rmin)
            ax2.set_rmax(rmax)
            ax2.set_ylim(rmin, rmax)
            ax2.set_facecolor("white")
            ax2.set_yticklabels([])
            ax2.grid(alpha=0.3)
    else:
        ax1 = fig.add_subplot(1, 2 if toroidal_slice2 is None else 3, 2)
        im1 = ax1.pcolormesh(Phi, Rg, toroidal, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title("toroidal cut (outboard)")
        ax1.set_xlabel("toroidal angle")
        ax1.set_ylabel("R")
        ax1.set_ylim(rmin, rmax)
        ax2 = None
        if toroidal_slice2 is not None:
            theta2 = theta[z_idx2]
            if coeffs is not None and ("Rxy" in coeffs):
                R_tor2 = R[z_idx2, :]
            else:
                r = x * (r_minor / max(Lx, 1e-8))
                R_tor2 = R0 + r * np.cos(theta2)
            Phi2, Rg2 = np.meshgrid(phi, R_tor2, indexing="xy")
            ax2 = fig.add_subplot(1, 3, 3)
            im2 = ax2.pcolormesh(
                Phi2, Rg2, toroidal_slice2, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax2.set_title("toroidal cut (inboard)")
            ax2.set_xlabel("toroidal angle")
            ax2.set_ylabel("R")
            ax2.set_ylim(rmin, rmax)
    axes = [ax0, ax1] if ax2 is None else [ax0, ax1, ax2]
    cbar = fig.colorbar(im0, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(args.field.replace("snapshots_", ""))

    try:
        from matplotlib import animation

        def update(i):
            pol = frames[i, :, :, y_idx]
            tor = frames[i, z_idx]
            tor2 = frames[i, z_idx2] if z_idx2 is not None else None
            if args.lowpass:
                pol = _lowpass_2d(pol, float(args.lowpass))
                tor = _lowpass_2d(tor, float(args.lowpass))
                if tor2 is not None:
                    tor2 = _lowpass_2d(tor2, float(args.lowpass))
            if interp_grid > 0:
                interp = mtri.LinearTriInterpolator(tri, pol.ravel())
                vals = interp(Rg_pol, Zg_pol)
                vals = np.ma.masked_invalid(vals)
                im0.set_data(vals)
            else:
                im0.set_array(pol.ravel())
            im1.set_array(tor.ravel())
            artists = [im0, im1]
            if tor2 is not None and ax2 is not None:
                im2.set_array(tor2.ravel())
                artists.append(im2)
            return tuple(artists)

        anim = animation.FuncAnimation(fig, update, frames=frames.shape[0], blit=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer="pillow", fps=12)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
