from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .map import FCIBilinearMap


def save_fci_maps_npz(
    path: str | Path,
    *,
    map_fwd: FCIBilinearMap,
    map_bwd: FCIBilinearMap,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Save a pair of FCI maps to a compressed ``.npz`` file.

    The on-disk format is intentionally simple and explicit so it can be generated
    by external tools. Arrays are stored under:

    - ``fwd_ix, fwd_iy, fwd_w, fwd_dl, fwd_hit, fwd_dl_hit``
    - ``bwd_ix, bwd_iy, bwd_w, bwd_dl, bwd_hit, bwd_dl_hit``

    where the optional ``*_hit`` and ``*_dl_hit`` arrays can be omitted.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def maybe(a):
        if a is None:
            return None
        return np.asarray(a)

    arrays: dict[str, Any] = {
        "format_version": np.asarray(1, dtype=np.int32),
        "fwd_ix": np.asarray(map_fwd.ix),
        "fwd_iy": np.asarray(map_fwd.iy),
        "fwd_w": np.asarray(map_fwd.w),
        "fwd_dl": np.asarray(map_fwd.dl),
        "bwd_ix": np.asarray(map_bwd.ix),
        "bwd_iy": np.asarray(map_bwd.iy),
        "bwd_w": np.asarray(map_bwd.w),
        "bwd_dl": np.asarray(map_bwd.dl),
    }
    if map_fwd.hit is not None:
        arrays["fwd_hit"] = maybe(map_fwd.hit)
    if map_fwd.dl_hit is not None:
        arrays["fwd_dl_hit"] = maybe(map_fwd.dl_hit)
    if map_bwd.hit is not None:
        arrays["bwd_hit"] = maybe(map_bwd.hit)
    if map_bwd.dl_hit is not None:
        arrays["bwd_dl_hit"] = maybe(map_bwd.dl_hit)

    if meta is not None:
        arrays["meta_json"] = np.asarray(json.dumps(meta), dtype=object)

    np.savez_compressed(path, **arrays)
    return path


def load_fci_maps_npz(path: str | Path) -> tuple[FCIBilinearMap, FCIBilinearMap, dict[str, Any]]:
    """Load a pair of FCI maps from a ``.npz`` file."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        version = int(data.get("format_version", 0))
        if version not in (0, 1):
            raise ValueError(f"Unsupported FCI map format_version={version} in {path}")

        meta_json = data.get("meta_json", None)
        meta: dict[str, Any] = {}
        if meta_json is not None:
            meta = json.loads(str(meta_json))

        def opt(name: str):
            return data[name] if name in data.files else None

        map_fwd = FCIBilinearMap(
            ix=data["fwd_ix"],
            iy=data["fwd_iy"],
            w=data["fwd_w"],
            dl=data["fwd_dl"],
            hit=opt("fwd_hit"),
            dl_hit=opt("fwd_dl_hit"),
        )
        map_bwd = FCIBilinearMap(
            ix=data["bwd_ix"],
            iy=data["bwd_iy"],
            w=data["bwd_w"],
            dl=data["bwd_dl"],
            hit=opt("bwd_hit"),
            dl_hit=opt("bwd_dl_hit"),
        )
        return map_fwd, map_bwd, meta
