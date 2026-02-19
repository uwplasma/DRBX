import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


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
    p.add_argument("--file", required=True, help="Path to NetCDF file")
    p.add_argument("--var", default="Ne")
    p.add_argument("--outdir", default="out_netcdf")
    p.add_argument("--axes", default="xyz", help="Spatial axis order in file, e.g. xyz")
    p.add_argument("--canonical", default="zxy", help="Target axis order for analysis")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with Dataset(args.file, "r") as ds:
        if "t" in ds.variables:
            t = ds.variables["t"][:]
        elif "t_array" in ds.variables:
            t = ds.variables["t_array"][:]
        else:
            t = None

        if args.var not in ds.variables:
            raise ValueError(f"Variable {args.var} not in file")

        var = ds.variables[args.var]
        data = var[:]
        if data.ndim != 4:
            raise ValueError(f"Expected 4D variable, got shape {data.shape}")

        last = data[-1]
        last = _permute_axes(last, args.axes, args.canonical)
        zmid = last.shape[0] // 2
        slice_xy = last[zmid, :, :]
        ax0, ax1, ax2 = args.canonical.lower()

        stats = {
            "mean": float(np.mean(last)),
            "rms": float(np.sqrt(np.mean(last ** 2))),
            "min": float(np.min(last)),
            "max": float(np.max(last)),
        }

        print(f"Loaded {args.var} shape {data.shape}")
        print("Stats (last step):", stats)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(slice_xy, origin="lower", aspect="auto")
        ax.set_title(f"NetCDF {args.var} last step {ax0}={zmid}")
        ax.set_xlabel(f"{ax2} index")
        ax.set_ylabel(f"{ax1} index")
        fig.colorbar(im, ax=ax)
        out_png = outdir / f"netcdf_{args.var}_last_{ax0}{zmid}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        if t is not None:
            series = np.mean(data.reshape(data.shape[0], -1), axis=1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(t, series, marker="o", lw=1)
            ax.set_title(f"NetCDF mean {args.var}")
            ax.set_xlabel("t")
            ax.set_ylabel(f"mean {args.var}")
            fig.tight_layout()
            out_png = outdir / f"netcdf_mean_{args.var}.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
