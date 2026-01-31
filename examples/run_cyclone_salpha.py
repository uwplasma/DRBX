from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.cli.main import main  # noqa: E402

if __name__ == "__main__":
    out = Path("out_cyclone_salpha_example")
    args = [
        "--geom",
        "salpha",
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
        "--q",
        "1.4",
        "--R0",
        "1.0",
        "--epsilon",
        "0.18",
        "--shat",
        "0.796",
        "--alpha",
        "0.0",
        "--tmax",
        "20.0",
        "--nsave",
        "150",
    ]
    sys.argv = ["jaxdrb-scan", *args]
    main()
