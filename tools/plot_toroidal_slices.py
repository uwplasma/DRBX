from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from jaxdrb.io import load_config

from plot_utils import maybe_lowpass


def _latest_snapshot(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key in data:
        return np.asarray(data[key])
    if f"snapshot_{key}" in data:
        return np.asarray(data[f"snapshot_{key}"])
    if f"snapshots_{key}" in data:
        arr = np.asarray(data[f"snapshots_{key}"])
        return arr[-1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot toroidal geometry slices (poloidal + toroidal cut)."
    )
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--config", required=True, help="Path to input TOML")
    parser.add_argument("--field", default="n", help="Field name")
    parser.add_argument("--out", default="docs/figures/toroidal_slices.png", help="Output image")
    parser.add_argument("--y-index", type=int, default=None, help="Toroidal index")
    parser.add_argument("--z-index", type=int, default=None, help="Poloidal index")
    parser.add_argument(
        "--toroidal-theta",
        type=float,
        default=None,
        help="Poloidal angle (radians) to select for the toroidal cut.",
    )
    parser.add_argument(
        "--separatrix",
        type=float,
        default=None,
        help="Optional separatrix radius (same units as Lx).",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Optional low-pass fraction (0-1) for smoother visuals.",
    )
    parser.add_argument(
        "--fluct",
        default="none",
        choices=("none", "mean", "zonal"),
        help="Subtract mean or zonal mean before plotting.",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use symmetric color limits about zero.",
    )
    parser.add_argument(
        "--field-scale",
        type=float,
        default=1.0,
        help="Multiply the plotted field by this factor.",
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
        raise ValueError("Toroidal slices require a 3D field-aligned geometry.")

    R0 = float(geom.get("R0", 3.0))
    r_minor = float(geom.get("r_minor", geom.get("Lx", 1.0)))
    Lx = float(geom.get("Lx", r_minor))
    nx = int(geom.get("nx", 64))
    ny = int(geom.get("ny", 64))
    nz = int(geom.get("nz", 64))
    Ly = float(geom.get("Ly", 2.0 * np.pi))
    theta_scale = float(geom.get("theta_scale", 1.0))
    Lz = float(geom.get("Lz", 2.0 * np.pi * theta_scale))

    x = np.linspace(0.0, Lx, nx, endpoint=True)
    y = np.linspace(0.0, Ly, ny, endpoint=False)
    z = np.linspace(-0.5 * Lz, 0.5 * Lz, nz, endpoint=True)
    theta = z / max(theta_scale, 1e-8)
    phi = y * (2.0 * np.pi / max(Ly, 1e-8))

    data = np.load(args.input)
    field = _latest_snapshot(data, args.field)
    if field is None:
        raise ValueError(f"Missing snapshots for '{args.field}'.")
    if field.ndim != 3:
        raise ValueError(f"Expected 3D field, got {field.shape}.")
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    if args.fluct == "mean":
        field = field - np.mean(field)
    elif args.fluct == "zonal":
        field = field - np.mean(field, axis=2, keepdims=True)
    field = field * float(args.field_scale)

    if args.y_index is None:
        y_var = np.var(field, axis=(0, 1))
        y_idx = int(np.argmax(y_var))
    else:
        y_idx = int(args.y_index)
    if args.z_index is None:
        if args.toroidal_theta is not None:
            z_idx = int(np.argmin(np.abs(theta - float(args.toroidal_theta))))
        else:
            z_var = np.var(field, axis=(1, 2))
            z_idx = int(np.argmax(z_var))
    else:
        z_idx = int(args.z_index)
    # pick time of max energy if a time axis exists
    poloidal_slice = field[:, :, y_idx]
    toroidal_slice = field[z_idx]

    poloidal_slice = maybe_lowpass(poloidal_slice, args.lowpass)
    toroidal_slice = maybe_lowpass(toroidal_slice, args.lowpass)

    r = x * (r_minor / max(Lx, 1e-8))
    R = R0 + r[None, :] * np.cos(theta[:, None])
    Z = r[None, :] * np.sin(theta[:, None])
    tri = mtri.Triangulation(R.ravel(), Z.ravel())
    interp_grid = int(args.interp_grid)
    if interp_grid > 0:
        r_lin = np.linspace(R.min(), R.max(), interp_grid)
        z_lin = np.linspace(Z.min(), Z.max(), interp_grid)
        Rg_pol, Zg_pol = np.meshgrid(r_lin, z_lin)
        interp = mtri.LinearTriInterpolator(tri, poloidal_slice.ravel())
        poloidal_interp = interp(Rg_pol, Zg_pol)
        poloidal_interp = np.ma.masked_invalid(poloidal_interp)
    else:
        r_lin = None
        z_lin = None
        Rg_pol = None
        Zg_pol = None
        poloidal_interp = None

    theta0 = theta[z_idx]
    R_tor = R0 + r * np.cos(theta0)
    Phi, Rg = np.meshgrid(phi, R_tor, indexing="xy")

    concat = np.concatenate([poloidal_slice.ravel(), toroidal_slice.ravel()])
    vmin, vmax = np.percentile(concat, [2.0, 98.0])
    if args.symmetric:
        vmax = float(max(abs(vmin), abs(vmax)))
        vmin = -vmax
    cmap = "coolwarm" if args.symmetric else "viridis"

    fig = plt.figure(figsize=(12.0, 5.0), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    if interp_grid > 0:
        im0 = ax0.imshow(
            poloidal_interp,
            origin="lower",
            extent=(r_lin.min(), r_lin.max(), z_lin.min(), z_lin.max()),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
    else:
        im0 = ax0.tripcolor(
            tri, poloidal_slice.ravel(), shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
        )
    ax0.set_title("poloidal cut")
    ax0.set_xlabel("R")
    ax0.set_ylabel("Z")
    ax0.set_aspect("equal")
    if args.separatrix is not None:
        sep = float(args.separatrix)
        ax0.plot(R0 + sep * np.cos(theta), sep * np.sin(theta), color="white", lw=1.2, ls="--")

    if args.toroidal_mode == "polar":
        ax1 = fig.add_subplot(1, 2, 2, projection="polar")
        im1 = ax1.pcolormesh(
            Phi,
            Rg,
            toroidal_slice,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax1.set_title("toroidal cut")
        rmin = float(R0 - r_minor)
        rmax = float(R0 + r_minor)
        ax1.set_rmin(rmin)
        ax1.set_rmax(rmax)
        ax1.set_ylim(rmin, rmax)
        ax1.set_facecolor("white")
        ax1.set_yticklabels([])
        ax1.grid(alpha=0.3)
    else:
        ax1 = fig.add_subplot(1, 2, 2)
        im1 = ax1.pcolormesh(
            Phi, Rg, toroidal_slice, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax1.set_title("toroidal cut")
        ax1.set_xlabel("toroidal angle")
        ax1.set_ylabel("R")
        ax1.set_ylim(float(R0 - r_minor), float(R0 + r_minor))

    fig.colorbar(im0, ax=[ax0, ax1], fraction=0.03, pad=0.02)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
