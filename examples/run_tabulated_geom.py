from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.cli.main import main  # noqa: E402
from jaxdrb.geometry.slab import SlabGeometry  # noqa: E402


def write_example_geometry(path: Path, *, nl: int = 64, shat: float = 0.8, curvature0: float = 0.2) -> None:
    geom = SlabGeometry.make(nl=nl, shat=shat, curvature0=curvature0)
    gxx, gxy, gyy = geom.metric_components()

    l = np.asarray(geom.l)
    curv_x = np.zeros_like(l)
    curv_y = curvature0 * np.cos(l)
    dpar_factor = np.ones_like(l)

    np.savez(
        path,
        l=l,
        gxx=np.asarray(gxx),
        gxy=np.asarray(gxy),
        gyy=np.asarray(gyy),
        curv_x=curv_x,
        curv_y=curv_y,
        dpar_factor=dpar_factor,
    )


if __name__ == "__main__":
    out = Path("out_tabulated_example")
    geom_file = out / "geom.npz"
    out.mkdir(parents=True, exist_ok=True)
    write_example_geometry(geom_file, nl=32, shat=0.8, curvature0=0.2)

    args = [
        "--geom",
        "tabulated",
        "--geom-file",
        str(geom_file),
        "--ky-min",
        "0.05",
        "--ky-max",
        "0.8",
        "--nky",
        "12",
        "--out",
        str(out),
        "--nl",
        "32",
        "--tmax",
        "20.0",
        "--nsave",
        "150",
    ]
    sys.argv = ["jaxdrb-scan", *args]
    main()
