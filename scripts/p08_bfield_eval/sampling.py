"""Owner sampling by region and magnetic-knot class (pure numpy except :func:`owner_centroid_phi`).

* **Region** (:data:`REGIONS`): disjoint owner classes built from the P06N ``regional_masks`` plus the ``axis_core`` mask
  (owners containing a raw cell of radius index 0, the ``owner_has(radius == 0)`` rule of
  ``p06n_field_derived_global.core.regional_masks``).  Priority order ``axis_core > physical_wall > last_two_layers >
  transition > aggregate > ordinary``: an owner belongs to the first class whose mask contains it.
  ``last_two_layers`` is ``transverse_last_two_layers & ~physical_wall``.
* **Knot class** (:data:`KNOTS`): the cylindrical toroidal angle ``phi`` of the owner centroid (volume-weighted circular
  mean of its raw-cell midpoint angles) against the MAKEGRID toroidal planes ``phi0 + k * dphi``: ``plane`` when the distance
  to the nearest plane is ``< 0.2 dphi``, ``mid`` when it is within ``0.2 dphi`` of a plane midpoint, otherwise
  unclassified (``-1``, never sampled).
* :func:`draw_sample`: ``per_cell`` owners per (region, knot) cell, ``np.random.default_rng(seed)`` without replacement.
"""
from __future__ import annotations

import numpy as np

REGIONS = ("axis_core", "physical_wall", "last_two_layers", "transition", "aggregate", "ordinary")
KNOTS = ("plane", "mid")
KNOT_TOLERANCE = 0.2


def classify_knot(phi, phi0: float, dphi: float, tolerance: float = KNOT_TOLERANCE) -> np.ndarray:
    """``(m,)`` int8: ``0`` plane, ``1`` mid, ``-1`` unclassified, from angles ``phi`` (radians, any branch) against the planes
    ``phi0 + k * dphi`` (``k`` any integer; ``dphi`` must divide the field period)."""
    if not dphi > 0.0:
        raise ValueError("dphi must be positive")
    frac = np.mod((np.asarray(phi, dtype=np.float64) - float(phi0)) / float(dphi), 1.0)      # position inside a plane cell
    near_plane = np.minimum(frac, 1.0 - frac) < tolerance
    near_mid = np.abs(frac - 0.5) < tolerance
    out = np.full(frac.shape, -1, dtype=np.int8)
    out[near_plane] = 0
    out[near_mid & ~near_plane] = 1
    return out


def axis_core_mask(t) -> np.ndarray:
    """Owners containing a raw cell of radius index 0 (``regional_masks``' ``owner_has(radius == 0)``; raw ids are C-ordered
    ``(radius, theta, eta)``)."""
    n = int(t.n)
    radius = np.arange(n ** 3) // (n * n)
    return np.bincount(np.asarray(t.ro), weights=(radius == 0).astype(np.float64), minlength=len(t.vol)) > 0


def region_codes(masks: dict, axis_core: np.ndarray) -> np.ndarray:
    """``(n_owners,)`` int8 codes into :data:`REGIONS` from the ``regional_masks`` dict and the ``axis_core`` mask."""
    candidates = {
        "axis_core": np.asarray(axis_core, dtype=bool),
        "physical_wall": np.asarray(masks["physical_wall"], dtype=bool),
        "last_two_layers": np.asarray(masks["transverse_last_two_layers"], dtype=bool),
        "transition": np.asarray(masks["transition"], dtype=bool),
        "aggregate": np.asarray(masks["aggregate"], dtype=bool),
        "ordinary": np.asarray(masks["ordinary"], dtype=bool),
    }
    code = np.full(len(candidates["axis_core"]), -1, dtype=np.int8)
    remaining = np.ones(len(code), dtype=bool)
    for index, name in enumerate(REGIONS):
        take = candidates[name] & remaining
        code[take] = index
        remaining &= ~take
    return code


def owner_centroid_phi(t, position, chunk: int = 16384) -> np.ndarray:
    """``(n_owners,)`` cylindrical angle of each owner centroid: the volume-weighted circular mean of the angles of its raw-cell
    midpoints.  ``position(points (Q, 3)) -> (Q, 3)`` is the Cartesian position of logical points (the metric evaluator's
    ``position``; independent of the magnetic field)."""
    points = np.asarray(t.pts, dtype=np.float64)
    owner = np.asarray(t.ro)
    weight = np.asarray(t.rv, dtype=np.float64)
    nvol = len(t.vol)
    cos_sum, sin_sum = np.zeros(nvol), np.zeros(nvol)
    for start in range(0, len(points), chunk):
        xyz = np.asarray(position(points[start:start + chunk]), dtype=np.float64)
        angle = np.arctan2(xyz[:, 1], xyz[:, 0])
        w, o = weight[start:start + chunk], owner[start:start + chunk]
        cos_sum += np.bincount(o, weights=w * np.cos(angle), minlength=nvol)
        sin_sum += np.bincount(o, weights=w * np.sin(angle), minlength=nvol)
    return np.arctan2(sin_sum, cos_sum)


def plane_grid(evaluator) -> tuple[float, float]:
    """``(phi0, dphi)`` of a MAKEGRID B evaluator's toroidal planes (``evaluator.phi`` over one field period)."""
    phi = np.asarray(evaluator.phi, dtype=np.float64)
    dphi = float(phi[1] - phi[0])
    period = float(evaluator.period)
    count = period / dphi
    if abs(count - round(count)) > 1e-6 or abs(round(count) - len(phi)) > 0:
        raise ValueError(f"toroidal planes do not tile the period: {len(phi)} planes, period/dphi = {count}")
    return float(phi[0]), dphi


def draw_sample(region: np.ndarray, knot: np.ndarray, per_cell: int, seed: int) -> tuple[np.ndarray, dict]:
    """``(owners sorted, counts)``: ``per_cell`` owners per (region, knot) cell without replacement (fewer when unavailable).
    One generator ``default_rng(seed)`` is consumed cell by cell in the order ``REGIONS x KNOTS``; ``counts`` is
    ``{"region|knot": {"available": a, "drawn": d}}``."""
    rng = np.random.default_rng(int(seed))
    region, knot = np.asarray(region), np.asarray(knot)
    chosen, counts = [], {}
    for r, rname in enumerate(REGIONS):
        for k, kname in enumerate(KNOTS):
            pool = np.flatnonzero((region == r) & (knot == k))
            take = min(int(per_cell), len(pool))
            picked = rng.choice(pool, size=take, replace=False) if take else np.empty(0, dtype=np.int64)
            chosen.append(np.asarray(picked, dtype=np.int64))
            counts[f"{rname}|{kname}"] = {"available": int(len(pool)), "drawn": int(take)}
    owners = np.sort(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)
    return owners, counts
