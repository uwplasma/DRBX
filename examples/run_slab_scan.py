from __future__ import annotations

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.cli.main import main


if __name__ == "__main__":
    out = Path("out_slab_example")
    args = [
        "--geom",
        "slab",
        "--ky-min",
        "0.05",
        "--ky-max",
        "1.0",
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
