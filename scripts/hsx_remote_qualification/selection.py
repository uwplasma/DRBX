"""Versioned, geometry-only tie handling for the clean HSX campaign.

This deliberately does not reproduce historical platform-specific donor IDs.
Close distances form anchored groups (not a non-transitive pairwise comparator).
The owner flat ID orders each group before the pool cutoff is taken.
"""
from __future__ import annotations

import copy
import numpy as np

VERSION = "hsx-cubic-selection-v3"
EPS_FACTOR = 128.0
POLICY = {
    "version": VERSION,
    "distance_tolerance": "128*eps*max(max(abs(owner_xy),abs(query_xy))**2, dr**2)",
    "distance_groups": "ascending radius squared; group anchored at its smallest radius; owner-flat-ID order within group",
    "sector_rule": "all eight boundaries snap within 128*eps in normalized octant coordinates; counterclockwise half-open sectors",
    "pool_rule": "include every candidate within cutoff tolerance before canonical truncation",
}


def tied_order(radius2, ids, tolerance):
    radius2 = np.asarray(radius2)
    ids = np.asarray(ids)
    order = np.lexsort((ids, radius2))
    if not np.all(np.isfinite(radius2)) or tolerance < 0:
        raise ValueError("invalid distance data")
    # Most gaps are well separated. Only sort runs containing possible ties.
    close = np.flatnonzero(np.diff(radius2[order]) <= tolerance)
    cursor = 0
    for start in close:
        if start < cursor:
            continue
        stop = int(start) + 1
        anchor = radius2[order[start]]
        while stop < len(order) and radius2[order[stop]] - anchor <= tolerance:
            stop += 1
        if stop > start + 1:
            order[start:stop] = order[start:stop][np.argsort(ids[order[start:stop]], kind="stable")]
        cursor = stop
    return order


def sectors(displacement):
    angle = np.mod(np.arctan2(displacement[..., 1], displacement[..., 0]), 2 * np.pi)
    octant = angle / (np.pi / 4)
    nearest = np.rint(octant)
    snapped = np.where(np.abs(octant - nearest) <= EPS_FACTOR * np.finfo(float).eps, nearest, octant)
    return np.floor(snapped).astype(np.int64) % 8


def distance_tolerance(context, point):
    # Fixed per-query scale, independent of candidate ordering/pool size.
    if not hasattr(context, "_v3_coordinate_extent"):
        context._v3_coordinate_extent = float(np.max(np.abs(context.arrays["owner_centroid_xy"])))
    extent = max(context._v3_coordinate_extent,
                 float(np.max(np.abs(point))), float(context.dr))
    return EPS_FACTOR * np.finfo(float).eps * extent**2


def nearest_pool(context, plane, xy, *, exact=False, pool_count=32):
    compact = context.plane_indices[plane]
    coordinates = context.arrays["owner_centroid_xy"]
    ids = context.arrays["owner_flat_ids"]
    count = min(pool_count, len(compact))
    result = np.empty((len(xy), count), dtype=np.int64)
    expanded = 0
    for row, point in enumerate(xy):
        tol = distance_tolerance(context, point)
        if exact or count == len(compact):
            candidate = compact
        else:
            # Query a superset, then explicitly include the entire near-cutoff shell.
            distance, _ = context.plane_trees[plane].query(point, k=count)
            radius = np.sqrt(float(np.atleast_1d(distance)[-1])**2 + 4 * tol)
            local = context.plane_trees[plane].query_ball_point(point, np.nextafter(radius, np.inf))
            candidate = compact[np.asarray(local, dtype=np.int64)]
            expanded += int(len(candidate) > count)
            if len(candidate) < count:
                raise RuntimeError("candidate query omitted required neighbors")
        radius2 = np.sum((coordinates[candidate] - point)**2, axis=1)
        order = tied_order(radius2, ids[candidate], tol)
        result[row] = candidate[order[:count]]
    return result, expanded


def sector_select(context, pool, xy, planar_scale, count):
    coordinates = context.arrays["owner_centroid_xy"]
    ids = context.arrays["owner_flat_ids"]
    result = []
    for row, candidate in enumerate(pool):
        displacement = (coordinates[candidate] - xy[row]) / planar_scale[row]
        sector = sectors(displacement)
        radius2 = np.sum(displacement**2, axis=1)
        tol = distance_tolerance(context, xy[row]) / planar_scale[row]**2
        groups = []
        for octant in range(8):
            indices = np.flatnonzero(sector == octant)
            order = tied_order(radius2[indices], ids[candidate[indices]], tol)
            groups.append(indices[order])
        ordered = [groups[s][rank] for rank in range(max(map(len, groups)))
                   for s in range(8) if rank < len(groups[s])]
        result.append(candidate[np.asarray(ordered[:count], dtype=np.int64)])
    return np.asarray(result)


def install(cubic):
    """Install only in this campaign's isolated interpreter, never edit v1/v2."""
    cubic.POLICY = copy.deepcopy(cubic.POLICY)
    cubic.POLICY["sector_boundary_tie"] = POLICY["sector_rule"]
    cubic.POLICY["portable_selection"] = POLICY
    cubic._nearest_pool = nearest_pool
    cubic._sector_select = sector_select
