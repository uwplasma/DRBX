from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from plot_utils import maybe_lowpass

from jaxdrb.io import load_config


def _latest_snapshot(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key in data:
        return np.asarray(data[key])
    if f"snapshot_{key}" in data:
        return np.asarray(data[f"snapshot_{key}"])
    if f"snapshots_{key}" in data:
        arr = np.asarray(data[f"snapshots_{key}"])
        return arr[-1]
    return None


def _slice_midplane(arr: np.ndarray, y_index: int | None) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        idx = y_index if y_index is not None else arr.shape[-1] // 2
        return arr[:, :, idx]
    if arr.ndim == 4:
        idx = y_index if y_index is not None else arr.shape[-1] // 2
        return arr[:, :, idx]
    raise ValueError(f"Unsupported snapshot shape: {arr.shape}")


def _sol_masks(
    x: np.ndarray, *, xs: float, width: float, open_left: bool
) -> tuple[np.ndarray, np.ndarray]:
    width = max(width, 1e-8)
    if open_left:
        mask_open = 0.5 * (1.0 - np.tanh((x - xs) / width))
    else:
        mask_open = 0.5 * (1.0 + np.tanh((x - xs) / width))
    mask_closed = 1.0 - mask_open
    return mask_closed, mask_open


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a poloidal (R,Z) slice.")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--config", required=True, help="Path to input TOML")
    parser.add_argument("--field", default="n", help="Field name (n, Te, omega, phi)")
    parser.add_argument("--out", default="docs/figures/poloidal_plane.png", help="Output image")
    parser.add_argument("--y-index", type=int, default=None, help="Binormal index")
    parser.add_argument(
        "--separatrix",
        type=float,
        default=None,
        help="Optional separatrix radius (same units as Lx).",
    )
    parser.add_argument(
        "--equilibrium",
        default="none",
        choices=("none", "add", "only"),
        help="Use SOL equilibrium profile (none, add, only).",
    )
    parser.add_argument(
        "--overlay-mask",
        action="store_true",
        help="Overlay open/closed mask boundary.",
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
        "--fluct-scale",
        type=float,
        default=1.0,
        help="Scale factor for fluctuations when overlaying equilibrium.",
    )
    parser.add_argument(
        "--field-scale",
        type=float,
        default=1.0,
        help="Multiply the plotted field by this factor.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap name to use for plotting.",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use symmetric color limits about zero.",
    )
    parser.add_argument(
        "--interp-grid",
        type=int,
        default=200,
        help="Interpolation grid resolution for smoother poloidal plots (0 to disable).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    geom = cfg.data.get("geometry", {})
    kind = str(geom.get("kind", "plane")).lower()
    R0 = float(geom.get("R0", 3.0))
    r_minor = float(geom.get("r_minor", geom.get("Lx", 1.0)))
    Lx = float(geom.get("Lx", r_minor))

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
        z = np.linspace(-0.5 * Lz, 0.5 * Lz, nz, endpoint=True)
        theta = z / max(theta_scale, 1e-8)

    data = np.load(args.input)
    field = _latest_snapshot(data, args.field)
    if field is None:
        raise ValueError(f"Missing snapshots for '{args.field}'.")
    field = _slice_midplane(field, args.y_index)
    if kind == "plane" and field.ndim == 2:
        field = field.T
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    if args.fluct == "mean":
        field = field - np.mean(field)
    elif args.fluct == "zonal":
        field = field - np.mean(field, axis=1, keepdims=True)
    field = maybe_lowpass(field, args.lowpass)
    field = float(args.field_scale) * field

    sol_cfg = cfg.data.get("physics", {})
    sol_on = bool(sol_cfg.get("sol_on", False))
    eq_field = None
    if args.equilibrium != "none" and sol_on and args.field in ("n", "Te"):
        xs = float(sol_cfg.get("sol_xs", 0.7))
        width = float(sol_cfg.get("sol_width", 0.05))
        open_left = bool(sol_cfg.get("sol_open_left", False))
        mask_closed, mask_open = _sol_masks(x[:, None], xs=xs, width=width, open_left=open_left)
        if args.field == "n":
            core = float(sol_cfg.get("sol_n_core", 1.0))
            sol = float(sol_cfg.get("sol_n_sol", 0.2))
        else:
            core = float(sol_cfg.get("sol_Te_core", 1.0))
            sol = float(sol_cfg.get("sol_Te_sol", 0.2))
        eq_profile = sol + (core - sol) * mask_closed
        eq_field = np.repeat(eq_profile.T, theta.size, axis=0)
        if args.equilibrium == "only":
            field = eq_field
        else:
            field = eq_field + float(args.fluct_scale) * field

    r = x * (r_minor / max(Lx, 1e-8))
    R = R0 + r[None, :] * np.cos(theta[:, None])
    Z = r[None, :] * np.sin(theta[:, None])

    # Ensure field orientation matches (theta, x) grid.
    if field.shape != R.shape:
        if field.shape == (R.shape[1], R.shape[0]):
            field = field.T
        else:
            raise ValueError(
                f"Field shape {field.shape} does not match grid {R.shape} and cannot be reconciled."
            )

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if args.symmetric:
        amp = float(np.nanpercentile(np.abs(field), 98.0))
        if amp <= 0.0:
            amp = 1.0
        vmin, vmax = -amp, amp
    else:
        vmin, vmax = np.nanpercentile(field, [2.0, 98.0])
        if np.isclose(vmin, vmax):
            span = float(max(abs(vmin), 1.0))
            vmin, vmax = -span, span
    tri = mtri.Triangulation(R.ravel(), Z.ravel())
    interp_grid = int(args.interp_grid)
    if interp_grid > 0:
        r_lin = np.linspace(R.min(), R.max(), interp_grid)
        z_lin = np.linspace(Z.min(), Z.max(), interp_grid)
        Rg, Zg = np.meshgrid(r_lin, z_lin)
        interp = mtri.LinearTriInterpolator(tri, field.ravel())
        vals = interp(Rg, Zg)
        vals = np.ma.masked_invalid(vals)
        im = ax.imshow(
            vals,
            origin="lower",
            extent=(r_lin.min(), r_lin.max(), z_lin.min(), z_lin.max()),
            cmap=str(args.cmap),
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
    else:
        im = ax.tripcolor(
            tri,
            field.ravel(),
            shading="gouraud",
            cmap=str(args.cmap),
            vmin=vmin,
            vmax=vmax,
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Poloidal {args.field}")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect("equal")
    if args.separatrix is not None:
        sep = float(args.separatrix)
        ax.plot(R0 + sep * np.cos(theta), sep * np.sin(theta), color="white", lw=1.2, ls="--")
    if args.overlay_mask and sol_on:
        xs = float(sol_cfg.get("sol_xs", 0.7))
        width = float(sol_cfg.get("sol_width", 0.05))
        open_left = bool(sol_cfg.get("sol_open_left", False))
        _, mask_open = _sol_masks(x[:, None], xs=xs, width=width, open_left=open_left)
        mask_open = np.repeat(mask_open.T, theta.size, axis=0)
        ax.contour(
            R,
            Z,
            mask_open,
            levels=[0.5],
            colors="white",
            linewidths=1.0,
            linestyles=":",
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
