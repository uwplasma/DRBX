import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _permute_axes(data, axes: str, canonical: str):
    axes = axes.lower()
    canonical = canonical.lower()
    if len(axes) != data.ndim:
        raise ValueError(f"axes length {len(axes)} != data.ndim {data.ndim}")
    if len(canonical) != data.ndim:
        raise ValueError(f"canonical length {len(canonical)} != data.ndim {data.ndim}")
    if sorted(axes) != sorted(canonical):
        raise ValueError(f"axes {axes} and canonical {canonical} must contain same labels")
    perm = [axes.index(c) for c in canonical]
    return data.transpose(perm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to npz with fields")
    p.add_argument("--var", default="n", help="Field name in npz (e.g. n, omega)")
    p.add_argument("--outdir", default="out_jaxdrb")
    p.add_argument("--axes", default="zxy", help="Spatial axis order in file, e.g. zxy")
    p.add_argument("--canonical", default="zxy", help="Target axis order for analysis")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.file)
    if args.var not in data:
        raise ValueError(f"Variable {args.var} not found in {args.file}")

    arr = data[args.var]
    t = data["t"] if "t" in data else None

    if arr.ndim == 3:
        field = _permute_axes(arr, args.axes, args.canonical)
    elif arr.ndim == 4:
        field = _permute_axes(arr[-1], args.axes, args.canonical)
    else:
        raise ValueError(f"Expected 3D or 4D field, got shape {arr.shape}")

    zmid = field.shape[0] // 2
    slice_xy = field[zmid, :, :]
    ax0, ax1, ax2 = args.canonical.lower()

    stats = {
        "mean": float(np.mean(field)),
        "rms": float(np.sqrt(np.mean(field ** 2))),
        "min": float(np.min(field)),
        "max": float(np.max(field)),
    }
    print("Stats:", stats)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(slice_xy, origin="lower", aspect="auto")
    ax.set_title(f"jax_drb {args.var} {ax0}={zmid}")
    ax.set_xlabel(f"{ax2} index")
    ax.set_ylabel(f"{ax1} index")
    fig.colorbar(im, ax=ax)
    out_png = outdir / f"jaxdrb_{args.var}_{ax0}{zmid}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    if t is not None and arr.ndim == 4:
        series = np.mean(arr.reshape(arr.shape[0], -1), axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, series, marker="o", lw=1)
        ax.set_title(f"jax_drb mean {args.var}")
        ax.set_xlabel("t")
        ax.set_ylabel(f"mean {args.var}")
        fig.tight_layout()
        out_png = outdir / f"jaxdrb_mean_{args.var}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
