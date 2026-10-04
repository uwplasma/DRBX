"""Write the HSX n = 32 nodal-metric test fixture (one-off).

Extracts rings {8, 9, 15, 16, 30, 31}, all theta, eta planes {0, 1, 2, 31} from the full raw-grid extract
(keys ``n, raw_ids, h, jac, B, K, pts``) into ``tests/data/p09_nodal_metric_n32_excerpt.npz``.

    python scripts/p09_nodal/make_metric_excerpt.py --source /path/to/hsx_n32_extract.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "tests" / "data" / "p09_nodal_metric_n32_excerpt.npz"
RINGS = (8, 9, 15, 16, 30, 31)
PLANES = (0, 1, 2, 31)
KEYS = ("raw_ids", "h", "jac", "B", "K", "pts")


def make_excerpt(source: Path, out: Path = DEFAULT_OUT) -> Path:
    with np.load(source, allow_pickle=False) as z:
        n = int(z["n"])
        if n != 32:
            raise ValueError(f"expected an n = 32 extract, got n = {n}")
        raw_ids = np.asarray(z["raw_ids"])
        if not np.array_equal(raw_ids, np.arange(n**3)):
            raise ValueError("the extract is expected to hold every raw cell in order")
        ii, jj, kk = np.meshgrid(RINGS, np.arange(n), PLANES, indexing="ij")
        ids = ((ii * n + jj) * n + kk).ravel()
        payload = {key: np.asarray(z[key])[ids] for key in KEYS}
    payload["n"] = np.array(n)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = make_excerpt(args.source, args.out)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
