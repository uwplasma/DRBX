from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.benchmarking import load_bundle_npz

FIELDS = ("n", "Te", "omega", "phi")


def _as_plane(a: np.ndarray, plane: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if plane == "xy":
            return arr[:, :, arr.shape[2] // 2]
        if plane == "xz":
            return arr[:, arr.shape[1] // 2, :]
    raise ValueError(f"Unsupported snapshot shape {arr.shape} for plane={plane}.")


def _pick_snapshot(bundle, field: str, *, fluct: bool, plane: str) -> np.ndarray:
    key = f"{field}_fluct_last" if fluct else f"{field}_last"
    if key not in bundle.snapshots:
        key = f"{field}_last"
    return _as_plane(bundle.snapshots[key], plane)


def _shared_symmetric_range(a: np.ndarray, b: np.ndarray, q: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate(
        [np.ravel(np.asarray(a, dtype=np.float64)), np.ravel(np.asarray(b, dtype=np.float64))]
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    scale = float(np.percentile(np.abs(vals), q))
    if scale <= 0.0:
        scale = 1.0
    return -scale, scale


def _resample2d(field: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(field, dtype=np.float64)
    nx_out, ny_out = out_shape
    x_src = np.linspace(0.0, 1.0, src.shape[0])
    y_src = np.linspace(0.0, 1.0, src.shape[1])
    x_out = np.linspace(0.0, 1.0, nx_out)
    y_out = np.linspace(0.0, 1.0, ny_out)

    tmp = np.empty((nx_out, src.shape[1]), dtype=np.float64)
    for j in range(src.shape[1]):
        tmp[:, j] = np.interp(x_out, x_src, src[:, j])

    out = np.empty((nx_out, ny_out), dtype=np.float64)
    for i in range(nx_out):
        out[i, :] = np.interp(y_out, y_src, tmp[i, :])
    return out


def _load_poloidal_grid(
    coeff_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    if not coeff_path:
        return None
    coeffs = np.load(coeff_path)
    if "Rxy" not in coeffs or "Zxy" not in coeffs:
        return None
    mask_open = None
    if "mask_open" in coeffs:
        mask_open = np.asarray(coeffs["mask_open"], dtype=np.float64)
    return (
        np.asarray(coeffs["Rxy"], dtype=np.float64),
        np.asarray(coeffs["Zxy"], dtype=np.float64),
        mask_open,
    )


def _plot_snapshot(
    ax, data: np.ndarray, *, plane: str, poloidal_grid, title: str, vmin: float, vmax: float
) -> None:
    if plane == "xz" and poloidal_grid is not None:
        rxy, zxy, mask_open = poloidal_grid
        vals = np.asarray(data, dtype=np.float64)
        r = rxy
        z = zxy
        if r.shape != vals.shape:
            if r.T.shape == vals.shape:
                r = r.T
                z = z.T
            elif r.shape == vals.T.shape:
                vals = vals.T
            elif r.T.shape == vals.T.shape:
                r = r.T
                z = z.T
                vals = vals.T
            else:
                vals = _resample2d(vals, r.shape)
        im = ax.pcolormesh(r, z, vals, shading="gouraud", cmap="coolwarm", vmin=vmin, vmax=vmax)
        if mask_open is not None:
            mask = mask_open
            if mask.shape != r.shape:
                if mask.T.shape == r.shape:
                    mask = mask.T
                else:
                    mask = _resample2d(mask, r.shape)
            ax.contour(r, z, mask, levels=[0.5], colors="white", linewidths=0.9, linestyles="--")
        ax.set_aspect("equal")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
    else:
        im = ax.imshow(
            np.asarray(data).T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.set_title(title)
    return im


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-field Hermes vs jax_drb benchmark overview with snapshots and time traces."
    )
    parser.add_argument("--hermes", required=True, help="Hermes bundle npz.")
    parser.add_argument("--jax", required=True, help="jax_drb bundle npz.")
    parser.add_argument("--out", required=True, help="Output PNG.")
    parser.add_argument("--plane", choices=("xy", "xz"), default="xz")
    parser.add_argument(
        "--coeff-path", default="", help="Optional coefficient file for poloidal plotting."
    )
    parser.add_argument("--snapshot-mode", choices=("fluct", "total"), default="fluct")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    hermes = load_bundle_npz(args.hermes)
    jax = load_bundle_npz(args.jax)
    fluct = args.snapshot_mode == "fluct"
    poloidal_grid = _load_poloidal_grid(args.coeff_path)

    fig, axes = plt.subplots(len(FIELDS), 4, figsize=(20.0, 18.0))
    h_times = np.asarray(hermes.times_norm, dtype=np.float64)
    j_times = np.asarray(jax.times_norm, dtype=np.float64)

    for row, field in enumerate(FIELDS):
        h_snap = _pick_snapshot(hermes, field, fluct=fluct, plane=args.plane)
        j_snap = _pick_snapshot(jax, field, fluct=fluct, plane=args.plane)
        vmin, vmax = _shared_symmetric_range(h_snap, j_snap)

        im = _plot_snapshot(
            axes[row, 0],
            h_snap,
            plane=args.plane,
            poloidal_grid=poloidal_grid,
            title=f"Hermes {field}",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.03)

        im = _plot_snapshot(
            axes[row, 1],
            j_snap,
            plane=args.plane,
            poloidal_grid=poloidal_grid,
            title=f"jax_drb {field}",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.03)

        j_for_diff = j_snap if h_snap.shape == j_snap.shape else _resample2d(j_snap, h_snap.shape)
        diff = j_for_diff - h_snap
        dvmin, dvmax = _shared_symmetric_range(diff, diff)
        im = _plot_snapshot(
            axes[row, 2],
            diff,
            plane=args.plane,
            poloidal_grid=poloidal_grid,
            title=f"{field} diff",
            vmin=dvmin,
            vmax=dvmax,
        )
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.03)

        h_rms = np.asarray(hermes.diagnostics.get(f"rms_{field}_fluct", []), dtype=np.float64)
        j_rms = np.asarray(jax.diagnostics.get(f"rms_{field}_fluct", []), dtype=np.float64)
        ax = axes[row, 3]
        if h_rms.size:
            ax.plot(h_times[: h_rms.size], h_rms, "--", lw=2.0, label="Hermes")
        if j_rms.size:
            ax.plot(j_times[: j_rms.size], j_rms, "-", lw=1.8, label="jax_drb")
        if h_rms.size and j_rms.size:
            j_interp = np.interp(h_times[: h_rms.size], j_times[: j_rms.size], j_rms)
            rel_end = float((j_interp[-1] - h_rms[-1]) / (abs(h_rms[-1]) + 1e-12))
            ax.text(
                0.02,
                0.95,
                f"end rel={rel_end:+.3e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        ax.set_title(f"{field} fluctuation RMS")
        ax.set_xlabel("t")
        ax.set_ylabel("RMS")
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(frameon=False, fontsize=9)

    suffix = "fluct" if fluct else "total"
    fig.suptitle(
        f"Hermes vs jax_drb long-window overview ({suffix}, plane={args.plane})", fontsize=16
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi))
    print(f"Saved overview panel: {out}")


if __name__ == "__main__":
    main()
