import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def _list_steps(group):
    steps = [k for k in group.keys() if k.isdigit()]
    return sorted(steps)


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


def load_var3d(f, var, step=None):
    g = f["data"]["var3d"][var]
    steps = _list_steps(g)
    if not steps:
        raise ValueError(f"No 3D steps found for {var}")
    if step is None:
        step = steps[-1]
    if step not in g:
        raise ValueError(f"Step {step} not in {var}; available: {steps[:5]}...")
    data = g[step][...]
    return data, step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to HDF5 file")
    p.add_argument("--var", default="theta")
    p.add_argument("--step", default=None)
    p.add_argument("--outdir", default="out_hdf5")
    p.add_argument("--axes", default="zxy", help="Spatial axis order in file, e.g. zxy")
    p.add_argument("--canonical", default="zxy", help="Target axis order for analysis")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.file, "r") as f:
        t = None
        if "var0d" in f["data"] and "time" in f["data"]["var0d"]:
            t = f["data"]["var0d"]["time"][...]

        data3d, step = load_var3d(f, args.var, args.step)
        data3d = _permute_axes(data3d, args.axes, args.canonical)

        zmid = data3d.shape[0] // 2
        slice_xy = data3d[zmid, :, :]
        ax0, ax1, ax2 = args.canonical.lower()

        stats = {
            "mean": float(np.mean(data3d)),
            "rms": float(np.sqrt(np.mean(data3d ** 2))),
            "min": float(np.min(data3d)),
            "max": float(np.max(data3d)),
        }

        print(f"Loaded {args.var} step {step} shape {data3d.shape}")
        print("Stats:", stats)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(slice_xy, origin="lower", aspect="auto")
        ax.set_title(f"HDF5 {args.var} step {step} {ax0}={zmid}")
        ax.set_xlabel(f"{ax2} index")
        ax.set_ylabel(f"{ax1} index")
        fig.colorbar(im, ax=ax)
        out_png = outdir / f"hdf5_{args.var}_step_{step}_{ax0}{zmid}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        if t is not None:
            series_name = f"glob{args.var}"
            if series_name in f["data"]["var0d"]:
                series = f["data"]["var0d"][series_name][...]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(t, series, marker="o", lw=1)
                ax.set_title(f"HDF5 {series_name}")
                ax.set_xlabel("t")
                ax.set_ylabel(series_name)
                fig.tight_layout()
                out_png = outdir / f"hdf5_{series_name}.png"
                fig.savefig(out_png, dpi=150)
                plt.close(fig)


if __name__ == "__main__":
    main()
