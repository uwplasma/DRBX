"""Synthetic perpendicular-operator world for the plan / state / operator tests (P08 step 2b).

A small periodic-structured grid (``n = 7``, owners aggregating three consecutive raw ids) with the real
``FaceCensus``, random-but-shaped row objects of every kind the row artifact holds (R1 cells, R2 common,
R3 sides, their Neumann companions, R4 integrated faces; conditioned rows carry trace/wall points drawn
from small pools so the plan's global point tables actually deduplicate), random geometry arrays and
smooth analytic boundary data. Rows are returned in the ``p_shared.owner_closure.build_owner_rows``
dictionary format, so the same world feeds ``lower_perpendicular_plan_from_rows`` and, through
``pack_owner_rows`` + ``save_row_artifact``, the artifact path; the host references use the
``scripts/p_shared`` applications on the same row objects.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow
from drbx.geometry.fci_perpendicular_neumann_trace import NeumannPointRows
from drbx.geometry.fci_perpendicular_reconstruction import PointRows
from drbx.stencils.census import FaceCensus
from drbx.stencils.geometry_arrays import SCHEMA, GeometryArrays, _ARRAY_FIELDS
from drbx.stencils.loader import LoaderGrid
from drbx.stencils.operator_plan import face_row_selection, p07_row_selection

N = 7
NF = 3
FAMILY_NAMES = ("singleton",)


def _points(rng, pool, count):
    return pool[rng.choice(len(pool), size=count, replace=count > len(pool))]


def _point_row(rng, n_owners, q, conditioned, donor_pool, width):
    d = int(width)
    donors = np.sort(rng.choice(n_owners, size=d, replace=False)).astype(np.int64)
    value = 0.3 * rng.normal(size=(q, d))
    gradient = 0.3 * rng.normal(size=(q, 3, d))
    donor_pts = _points(rng, donor_pool, d) if conditioned else np.empty((0, 3))
    target_pts = rng.uniform(size=(q, 3))
    return PointRows(donors, value, gradient, bool(conditioned), donor_pts, target_pts,
                     {"family": "quartic_wall" if conditioned else "singleton"})


def _neumann_row(rng, n_owners, wall_pool):
    d = int(rng.integers(8, 15))
    donors = np.sort(rng.choice(n_owners, size=d, replace=False)).astype(np.int64)
    idx = rng.choice(len(wall_pool), size=28, replace=False)
    return NeumannPointRows(donors, 0.3 * rng.normal(size=d), 0.3 * rng.normal(size=(3, d)), wall_pool[idx],
                            0.1 * rng.normal(size=28), 0.1 * rng.normal(size=(3, 28)), 1.0, 0.0)


def make_world(*, seed: int = 0, n: int = N, owners=None, raw_to_owner=None) -> SimpleNamespace:
    """The full synthetic grid, or (``owners``) the bounded closure of those owners.

    ``raw_to_owner`` (default: three consecutive raw ids per owner) replaces the aggregation. With it every row's
    donors are drawn from the eta planes within two planes of its entity's owner planes (the reach of the real rows),
    so the world can be eta-sharded (``drbx.native.fci_perpendicular_sharding``).

    Returns a namespace with ``n, census, grid, geometry, raw_volume, owner_volume, n_owners, raw_ids,
    face_rows, p07_rows, row_index, neumann_index`` (``build_owner_rows`` format).
    """
    rng = np.random.default_rng(seed)
    ro = np.arange(n ** 3) // 3 if raw_to_owner is None else np.asarray(raw_to_owner)
    n_owners = int(ro.max()) + 1
    owner_plane = np.zeros(n_owners, dtype=np.int64)
    owner_plane[ro] = np.arange(n ** 3) % n
    near_cache = {}

    def near(planes):
        """Owners within two eta planes of ``planes`` (all owners without ``raw_to_owner``)."""
        if raw_to_owner is None:
            return n_owners
        key = frozenset(int(k) for k in planes)
        if key not in near_cache:
            allowed = {(k + d) % n for k in key for d in range(-2, 3)}
            near_cache[key] = np.flatnonzero(np.isin(owner_plane, sorted(allowed)))
        return near_cache[key]

    def face_planes(ridx):
        return [owner_plane[o] for o in (census.owner_lo[ridx], census.owner_hi[ridx]) if o >= 0]
    census = FaceCensus.build(n, ro)
    raw_volume = rng.uniform(0.5, 1.5, size=n ** 3)
    owner_volume = np.bincount(ro, weights=raw_volume)
    eta_centers = (np.arange(n) + 0.5) * 2 * np.pi / n
    grid = LoaderGrid.from_arrays(n=n, raw_to_owner=ro, eta_centers=eta_centers)

    raw_ids = np.arange(n ** 3, dtype=np.int64)
    face_rows = face_row_selection(census)
    p07_rows = p07_row_selection(census)
    if owners is not None:
        owners = np.asarray(sorted(set(int(o) for o in owners)), dtype=np.int64)
        raw_ids = np.flatnonzero(np.isin(ro, owners)).astype(np.int64)
        incident = np.flatnonzero(np.isin(census.owner_lo, owners) | np.isin(census.owner_hi, owners))
        face_rows = np.intersect1d(incident, face_rows)
        p07_rows = np.intersect1d(incident, p07_row_selection(census))   # keeps incident collapsed r=0 faces

    donor_pool = rng.uniform(size=(60, 3))
    wall_pool = rng.uniform(size=(45, 3))
    row_index, neumann_index = {}, {}
    for raw in raw_ids:
        conditioned = rng.random() < 0.3
        pool = near([raw % n])
        row_index[("R1", int(raw))] = _point_row(rng, pool, 1, conditioned, donor_pool, rng.integers(5, 25))
        if conditioned:
            neumann_index[("R1", int(raw), 0)] = _neumann_row(rng, pool, wall_pool)
    for ridx in face_rows:
        conditioned = rng.random() < 0.3
        pool = near(face_planes(ridx))
        row_index[("R2", int(ridx))] = _point_row(rng, pool, 9, conditioned, donor_pool, rng.integers(5, 25))
        if conditioned:
            for q in range(9):
                neumann_index[("R2", int(ridx), q)] = _neumann_row(rng, pool, wall_pool)
        any_side = False
        for side, raw_side in enumerate((census.raw_lo[ridx], census.raw_hi[ridx])):
            if raw_side < 0:
                continue
            side_conditioned = rng.random() < 0.3
            any_side |= side_conditioned
            row_index[("R3", int(ridx) * 2 + side)] = _point_row(rng, pool, 9, side_conditioned, donor_pool,
                                                                 rng.integers(5, 25))
        if any_side:
            for q in range(9):
                neumann_index[("R3", int(ridx), q)] = _neumann_row(rng, pool, wall_pool)
    for ridx in p07_rows:
        pid, family = int(census.p07_id[ridx]), int(census.family[ridx])
        target = rng.uniform(size=(9, 3))
        if family == 0:
            row = IntegratedFaceRow(np.empty(0, np.int64), np.empty(0), False, np.empty((0, 3)), target,
                                    np.empty(0), np.empty((0, 2)), family)
        else:
            d = int(rng.integers(5, 30))
            donors = np.sort(rng.choice(near(face_planes(ridx)), size=d, replace=False)).astype(np.int64)
            weights = 0.3 * rng.normal(size=d)
            if family in (1, 2, 4):
                row = IntegratedFaceRow(donors, weights, True, _points(rng, donor_pool, d), target, -weights,
                                        0.3 * rng.normal(size=(9, 2)), family)
                for q in range(9):
                    neumann_index[("R4", pid, q)] = _neumann_row(rng, near(face_planes(ridx)), wall_pool)
            else:
                row = IntegratedFaceRow(donors, weights, False, np.empty((0, 3)), target, np.empty(0),
                                        np.empty((0, 2)), family)
        row_index[pid] = row

    R, Fc = len(raw_ids), len(face_rows)
    raw = dict(
        raw_points=rng.uniform(size=(R, 3)), p05_raw_h=rng.normal(size=(R, 3)),
        p05_raw_jacobian=rng.uniform(0.5, 2.0, size=R), p06_raw_J=rng.uniform(0.5, 2.0, size=R),
        p06_raw_B=rng.uniform(0.5, 2.0, size=R), p06_raw_K=rng.normal(size=(R, 3)),
        p06_raw_weight=rng.uniform(0.01, 0.1, size=R), p07_raw_tensor=rng.normal(size=(R, 3, 3)),
        p07_raw_divergence=rng.normal(size=(R, 3)))
    face = dict(
        face_points=rng.uniform(size=(Fc, 9, 3)), p05_face_h=rng.normal(size=(Fc, 9, 3)),
        p05_face_jacobian=rng.uniform(0.5, 2.0, size=(Fc, 9)), p06_face_J=rng.uniform(0.5, 2.0, size=(Fc, 9)),
        p06_face_B=rng.uniform(0.5, 2.0, size=(Fc, 9)), p06_face_K=rng.normal(size=(Fc, 9, 3)),
        p06_face_weight=rng.uniform(0.01, 0.1, size=(Fc, 9)), p07_face_tensor=rng.normal(size=(Fc, 9, 3, 3)))
    arrays = {**raw, **face}
    assert set(arrays) == set(_ARRAY_FIELDS)
    geometry = GeometryArrays(schema=SCHEMA, identity=GeometryArrays._compute_identity(arrays), **arrays)
    return SimpleNamespace(n=n, census=census, grid=grid, geometry=geometry, raw_volume=raw_volume,
                           owner_volume=owner_volume, n_owners=n_owners, raw_ids=raw_ids, face_rows=face_rows,
                           p07_rows=p07_rows, row_index=row_index, neumann_index=neumann_index)


def lower_world(world, include=("cells", "faces", "p07")):
    from drbx.stencils.operator_plan import lower_perpendicular_plan_from_rows
    return lower_perpendicular_plan_from_rows(
        world.row_index, world.neumann_index, grid=world.grid, census=world.census, geometry=world.geometry,
        raw_volume=world.raw_volume, owner_volume=world.owner_volume, raw_ids=world.raw_ids,
        face_rows=world.face_rows, p07_rows=world.p07_rows, include=include)


class Boundary:
    """Smooth analytic boundary data: ``dirichlet(points) -> (value (Q, F), gradient (Q, 3, F))`` and
    ``normal(points) -> g_N (Q, F)`` (physical-normal derivative along a fixed smooth direction)."""

    def __init__(self, n_fields: int = NF, seed: int = 11):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(size=(3, n_fields))
        self.b = rng.normal(size=n_fields)
        self.n_fields = n_fields

    def dirichlet(self, points):
        points = np.asarray(points, dtype=np.float64)
        phase = points @ self.A + self.b
        return np.sin(phase), np.cos(phase)[:, None, :] * self.A[None]

    def direction(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.stack([1.0 + 0.1 * points[:, 0], 0.2 * points[:, 1], -0.3 * points[:, 2]], axis=1)

    def normal(self, points):
        _, g = self.dirichlet(points)
        return np.einsum("qa,qaf->qf", self.direction(points), g)


# --------------------------------------------------------------------------
# Host references (p_shared.apply / replay_units) on the row objects of a world or a real closure
# --------------------------------------------------------------------------

def host_cells(world, ov, boundary):
    """Host D and N reconstruction of every raw cell of ``world.raw_ids`` via ``p_shared.apply`` on the row objects."""
    from p_shared import apply as A
    nf = ov.shape[1]
    R = len(world.raw_ids)
    vd, gd = np.empty((R, nf)), np.empty((R, 3, nf))
    vn, gn = np.empty((R, nf)), np.empty((R, 3, nf))
    for c, raw in enumerate(world.raw_ids):
        row = world.row_index[("R1", int(raw))]
        v, g = A.apply_point_row(row, ov, boundary.dirichlet)
        vd[c], gd[c] = v[0], g[0]
        if row.boundary_conditioned:
            nrow = world.neumann_index[("R1", int(raw), 0)]
            vn[c], gn[c] = A.apply_neumann_row(nrow, ov, boundary.normal(nrow.boundary_points))
        else:
            vn[c], gn[c] = vd[c], gd[c]
    return (vd, gd), (vn, gn)


def host_faces(world, ov, boundary):
    """Host D/N common value+gradient and lower/upper side values of every face row, with the replay's own side
    helpers (``replay_units._batch_side_values`` / ``_fix_side_neumann_exterior_fallback``)."""
    import p_shared.replay_units as ru
    from p_shared import apply as A
    census, n = world.census, world.n
    rows = world.face_rows
    nf = ov.shape[1]
    Fc = len(rows)
    common = [world.row_index[("R2", int(r))] for r in rows]
    lower = [world.row_index.get(("R3", int(r) * 2)) for r in rows]
    upper = [world.row_index.get(("R3", int(r) * 2 + 1)) for r in rows]
    exists_lo, exists_hi = census.raw_lo[rows] >= 0, census.raw_hi[rows] >= 0
    points = [r.trace_target_points for r in common]

    def neumann_rows(request, ridx):
        got = [world.neumann_index.get((request, int(ridx), q)) for q in range(9)]
        return None if all(g is None for g in got) else got

    cn = [neumann_rows("R2", r) for r in rows]
    sn = [neumann_rows("R3", r) for r in rows]
    batch = ru._batch_dirichlet_neumann(common, boundary.dirichlet, boundary.normal, neumann_rows_by_row=cn)
    lower_d, lower_n = ru._batch_side_values(lower, exists_lo, points, ov, boundary.dirichlet, boundary.normal,
                                             neumann_rows_by_row=sn)
    upper_d, upper_n = ru._batch_side_values(upper, exists_hi, points, ov, boundary.dirichlet, boundary.normal,
                                             neumann_rows_by_row=sn)
    ru._fix_side_neumann_exterior_fallback(lower_n, upper_n, exists_lo, exists_hi)
    out = {k: np.empty(s) for k, s in (("cvd", (Fc, 9, nf)), ("cgd", (Fc, 9, 3, nf)), ("cvn", (Fc, 9, nf)),
                                        ("cgn", (Fc, 9, 3, nf)), ("ld", (Fc, 9, nf)), ("ln", (Fc, 9, nf)),
                                        ("ud", (Fc, 9, nf)), ("un", (Fc, 9, nf)))}
    for i, row in enumerate(common):
        cv, cg = A.apply_point_row_precomputed(
            row, ov, donor_trace_value=batch["donor_trace_value"][i],
            target_trace_value=batch["target_trace_value"][i], target_trace_gradient=batch["target_trace_gradient"][i])
        out["cvd"][i], out["cgd"][i] = cv, cg
        if row.boundary_conditioned:
            pairs = [A.apply_neumann_row(nr, ov, bd) for nr, bd in zip(batch["nrows_of"][i], batch["boundary_data_of"][i])]
            out["cvn"][i] = np.stack([p[0] for p in pairs]); out["cgn"][i] = np.stack([p[1] for p in pairs])
        else:
            out["cvn"][i], out["cgn"][i] = cv, cg
        out["ld"][i], out["ln"][i] = lower_d[i], lower_n[i]
        out["ud"][i], out["un"][i] = upper_d[i], upper_n[i]
    return out


