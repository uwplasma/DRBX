"""Field-independent row builder over the P08 step-1 face census (task 4,
"Builder" half -- see ``work/p08_step1_consolidation_design_20260928/
design.md`` sections 2 ("Row requests per operator"), 3 ("Row artifact") and
6 (task 4)).

This module enumerates *every* row request the accepted campaigns issue --
R1 raw-midpoint cell rows, R2 face-common rows, R3 face-side rows, and R4
P07 integrated face rows -- exactly as the frozen callers build them
(``scripts/p05n_field_derived_global/core.py``'s ``batched_cell_values`` /
``batched_face_common_gradient`` / ``batched_side_values``,
``scripts/p06n_field_derived_global/rows.py``'s ``batched_face_common_value``,
and ``scripts/p07n_field_derived_global/core.py``'s ``face_chunk``), and
packages the resulting host row objects (``PointRows`` / ``NeumannPointRows``
/ ``IntegratedFaceRow``) together with the tags ``drbx.stencils.artifact``'s
``pack_point_rows`` / ``pack_neumann_rows`` / ``pack_integrated_rows`` need.

Nothing here is field-dependent (no catalogue field, trace, or owner value
ever enters); nothing here imports from ``scripts/`` -- only the existing
``drbx.geometry`` builders (``fci_perpendicular_reconstruction``,
``fci_perpendicular_neumann_trace``, ``fci_perpendicular_integrated_rows``)
and this package's own ``census``/``geometry_arrays``/``artifact`` modules.
Row geometry is built once per (request, key, points, fixed_anchor) and
reused across every physical field/case/BC variant a host later applies it
to -- the "Sharing" rule of design section 2: R1 serves P05 and P06 (both
consume the same ``PointRows.value``/``.gradient``), R2/R3 serve P05 and P06
over the unified (P06 superset) census, and R2's/R3's Neumann companion row
is built once per face and reused by both its lower and upper side (see
``build_r3_side_rows``'s docstring for why that is bitwise, not approximate).

Entity-id encoding (this module's own bookkeeping, not shared with any
frozen campaign):

* R1: the raw flat cell index ``i*n*n+j*n+k`` (0..n**3-1).
* R2: the :class:`~drbx.stencils.census.FaceCensus` row index (the canonical
  P06/P07-diffusion face id; 0..3*(n+1)*n**2-1).
* R3: ``census_row_index*2 + side`` (``side`` 0 = lower/left cell, 1 =
  upper/right cell), so both sides of one face pack into the same chunk
  without colliding.
* R4: the P07 topology-census face id (``FaceCensus.p07_id``; 0..num_p07-1).
* Neumann rows are tagged by ``source`` (``'R1'``, ``'R2'``, ``'R3'``, or
  ``'R4'``) plus the *originating* request's own entity id above, so a
  caller can always trace a Neumann row back to the point/face/side that
  needed it. R3's Neumann row (shared by both sides -- see
  ``build_r3_side_rows``) is tagged once, at the face's own census row
  index (not the side-doubled R3 point-row id).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .census import NO_ID, FaceCensus
from .geometry_arrays import GeometryArrays, GeometryProvider

from drbx.geometry.fci_perpendicular_reconstruction import (
    PointRowContext, PointRows, StructuredReconstruction)
from drbx.geometry.fci_perpendicular_neumann_trace import (
    NeumannPointRows, prepare_neumann_point_rows)
from drbx.geometry.fci_perpendicular_integrated_rows import (
    IntegratedFaceRow, contract_face_tensor, prepare_integrated_face_rows)

__all__ = [
    "PointRowRequest", "NeumannRowRequest", "IntegratedRowRequest",
    "face_kind", "census_face_keys", "build_geometry_arrays",
    "build_r1_cell_rows", "build_r2_face_rows", "build_r3_side_rows",
    "build_r4_p07_rows", "p07_row_selection",
]

# Neumann radial degree per boundary family -- design section 2/6: "Neumann
# radial degree 3 for cell/transverse/side, 4 for wall radial faces."
_CELL_NEUMANN_DEGREE = 3
_SIDE_NEUMANN_DEGREE = 3
# P07 topology-census family -> Neumann radial degree, for R4's families
# 1 (physical_wall_quartic_BC), 2 (adjacent_radial_quartic_BC, also a radial
# "quartic_wall" face by drbx.geometry's own family() rule) and
# 4 (boundary_transverse_cubic_BC) -- mirrors
# scripts/p07n_field_derived_global/core.py's face_chunk: "degree=3 if
# fam[j]==4 else 4".
_P07_NEUMANN_DEGREE = {1: 4, 2: 4, 4: 3}


@dataclass(frozen=True)
class PointRowRequest:
    """One tagged ``PointRows`` (R1 cell / R2 face-common / R3 face-side)."""

    request: str          # 'R1' | 'R2' | 'R3'
    entity_id: int
    bc_variant: str        # '' (unconditioned) | 'D' (boundary-conditioned)
    radial_degree: int      # the D-lift degree (3 or 4); 0 where unconditioned
    row: PointRows
    #: the 1-D factors the row was built from (``PointFactors``; unconditioned singleton / ringwise /
    #: centered_radial rows only, or ``PairedFactors`` for a ``cell_stencil="symmetric"`` R1 cell row), captured when a
    #: builder is called with ``capture_factors=True``
    factors: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class NeumannRowRequest:
    """One tagged ``NeumannPointRows`` (the N-variant companion of a
    boundary-conditioned R1/R2/R3/R4 row)."""

    source: str            # 'R1' | 'R2' | 'R3' | 'R4'
    entity_id: int          # the originating request's own entity id
    quad_node: int
    radial_degree: int
    row: NeumannPointRows


@dataclass(frozen=True)
class IntegratedRowRequest:
    """One tagged ``IntegratedFaceRow`` (R4 P07 integrated face row)."""

    entity_id: int          # the P07 topology-census face id
    row: IntegratedFaceRow


def face_kind(n: int, key) -> str:
    """The four-way family ``S.rows``/``prepare_integrated_face_rows`` route
    boundary handling on, re-derived bitwise from
    ``drbx.geometry._fci_perpendicular_point_primitives.family`` (itself an
    AST-identical copy of ``scripts/p07_combined_global/kernels.py``'s
    ``family``): radial (``axis == 0``) faces are ``'quartic_wall'`` at
    ``i >= n-2`` else ``'centered_radial'``; transverse faces are
    ``'boundary_transverse'`` at ``i >= n-2`` else ``'interior_transverse'``.
    The collapsed r=0 face (``axis == 0, i == 0``) is never passed here by
    this module (see the census-row filters below); callers that need to
    recognize it should check ``FaceCensus.collapsed_r0`` directly.
    """
    axis, i = int(key[0]), int(key[1])
    if axis == 0:
        return "quartic_wall" if i >= n - 2 else "centered_radial"
    return "boundary_transverse" if i >= n - 2 else "interior_transverse"


def census_face_keys(census: FaceCensus, row_indices) -> np.ndarray:
    """The ``(axis, i, j, k)`` key of every requested census row, as ``int64``."""
    return census.keys()[np.asarray(row_indices, dtype=np.int64)]


def p07_row_selection(census: FaceCensus) -> np.ndarray:
    """Census row indices covering every P07 topology-census face exactly
    once: rows with a valid ``p07_id`` (owner-boundary faces), excluding the
    periodic alias slots (theta/eta slot ``n``), which carry the *same*
    ``p07_id`` as their slot-0 counterpart (see ``census.py``'s
    ``_gather_dense``/``rep_j``/``rep_k`` remap) and would otherwise be
    processed twice.
    """
    mask = (census.p07_id != NO_ID) & ~census.legacy_alias_slots
    return np.flatnonzero(mask)


def build_geometry_arrays(provider: GeometryProvider, context: PointRowContext, *,
                          raw_ids, face_row_indices, census: FaceCensus, face_order: int = 3) -> GeometryArrays:
    """Build the field-independent ``GeometryArrays`` for a batch of raw
    cells and census faces, in exactly the shape ``GeometryArrays.build``
    needs (design section 4: "plus GeometryArrays via a supplied
    GeometryProvider"). ``face_order`` (3, or 2) is the P05/P06 face-node rule of ``GeometryArrays.build``."""
    n = context.n
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    raw_keys = np.array(np.unravel_index(raw_ids, (n, n, n))).T.astype(np.int64)
    face_keys = census_face_keys(census, face_row_indices)
    if face_order == 3:
        return GeometryArrays.build(provider, faces=context.faces, raw_keys=raw_keys, face_keys=face_keys)
    return GeometryArrays.build(provider, faces=context.faces, raw_keys=raw_keys, face_keys=face_keys,
                                face_order=face_order)


# ---------------------------------------------------------------------------
# R1: raw-midpoint cell rows.
# ---------------------------------------------------------------------------
def build_r1_cell_rows(S: StructuredReconstruction, context: PointRowContext, raw_ids, *,
                       normal_coefficients, patch_cache,
                       capture_factors: bool = False) -> tuple[list[PointRowRequest], list[NeumannRowRequest]]:
    """R1: ``S.rows(key, points, location='cell')`` at every raw midpoint.

    Mirrors ``p05n_field_derived_global.core.batched_cell_values``: interior
    cells (radial index < n-2) get one unconditioned row; boundary cells
    (radial index >= n-2) get a D-conditioned ``PointRows`` (the row's own
    boundary-map reconstruction, applied at runtime with the Dirichlet
    trace) plus one shared Neumann companion row (radial degree 3, "target-
    anchored" -- ``prepare_neumann_point_rows`` batched over every boundary
    point in this call, exactly as ``batched_cell_values`` batches its own
    ``bpoints``).

    ``capture_factors`` (here and in ``build_r2_face_rows`` / ``build_r3_side_rows``) additionally records,
    on each returned ``PointRowRequest.factors``, the factors an unconditioned singleton / ringwise /
    centered_radial row was built from (``StructuredReconstruction.rows_with_factors``; for a
    ``cell_stencil="symmetric"`` cell row ``1/2 (A + B)`` the ``PairedFactors`` of its two parts); the rows are
    bitwise the same either way.
    """
    n = context.n
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    points = context.pts[raw_ids]
    keys = np.array(np.unravel_index(raw_ids, (n, n, n))).T
    boundary_mask = keys[:, 0] >= n - 2

    point_rows: list[PointRowRequest] = []
    for j in range(len(raw_ids)):
        key = tuple(int(v) for v in keys[j])
        if capture_factors:
            row, factors = S.rows_with_factors(key, points[j:j + 1], location="cell")
        else:
            row, factors = S.rows(key, points[j:j + 1], location="cell"), None
        if bool(row.boundary_conditioned) != bool(boundary_mask[j]):
            raise ValueError(f"unexpected boundary conditioning at raw cell {key}")
        family = row.diagnostics.get("family")
        degree = _CELL_NEUMANN_DEGREE if row.boundary_conditioned else 0
        if row.boundary_conditioned and family != "boundary_transverse":
            raise ValueError(f"R1 boundary cell {key} has unexpected family {family!r}")
        bc_variant = "D" if row.boundary_conditioned else ""
        point_rows.append(PointRowRequest("R1", int(raw_ids[j]), bc_variant, degree, row, factors))

    neumann_rows: list[NeumannRowRequest] = []
    if np.any(boundary_mask):
        bpoints = points[boundary_mask]
        bids = raw_ids[boundary_mask]
        nrows = prepare_neumann_point_rows(context, bpoints, normal_coefficients=normal_coefficients,
                                            radial_degree=_CELL_NEUMANN_DEGREE, patch_cache=patch_cache)
        for local, nrow in enumerate(nrows):
            neumann_rows.append(NeumannRowRequest("R1", int(bids[local]), 0, _CELL_NEUMANN_DEGREE, nrow))
    return point_rows, neumann_rows


# ---------------------------------------------------------------------------
# R2: face-common rows.
# ---------------------------------------------------------------------------
def build_r2_face_rows(S: StructuredReconstruction, context: PointRowContext, census: FaceCensus,
                       row_indices, face_points_by_row, *,
                       normal_coefficients, patch_cache,
                       capture_factors: bool = False) -> tuple[list[PointRowRequest], list[NeumannRowRequest]]:
    """R2: ``S.rows(key, points, location='face')`` at each census face's 9
    q3 nodes. ``row_indices``/``face_points_by_row`` must already exclude
    the collapsed r=0 face and the legacy alias slots (design section 2:
    "collapsed r=0 and alias slots excluded; internal faces included" --
    internal-to-owner faces are *not* filtered here, only the two families
    above are).

    Mirrors ``p05n_field_derived_global.core.batched_face_common_gradient``
    / ``p06n_field_derived_global.rows.batched_face_common_value``: both
    read the *same* ``PointRows`` (its ``.value`` and ``.gradient`` are both
    always populated), so one row per face serves both P05's gradient
    consumer and P06's value consumer. Boundary faces (``quartic_wall``
    degree 4, ``boundary_transverse`` degree 3) additionally get a Neumann
    companion row.
    """
    n = context.n
    keys = census.keys()
    row_indices = np.asarray(row_indices, dtype=np.int64)
    point_rows: list[PointRowRequest] = []
    neumann_rows: list[NeumannRowRequest] = []
    for local, ridx in enumerate(row_indices):
        key = tuple(int(v) for v in keys[ridx])
        axis, i = key[0], key[1]
        if axis == 0 and i == 0:
            raise ValueError("R2 must not receive the collapsed r=0 census row")
        points = np.asarray(face_points_by_row[local], dtype=np.float64)
        if capture_factors:
            row, factors = S.rows_with_factors(key, points, location="face")
        else:
            row, factors = S.rows(key, points, location="face"), None
        kind = face_kind(n, key)
        boundary = kind in ("quartic_wall", "boundary_transverse")
        if bool(row.boundary_conditioned) != boundary:
            raise ValueError(f"unexpected boundary conditioning at census row {int(ridx)} key={key}")
        degree = 4 if kind == "quartic_wall" else 3 if kind == "boundary_transverse" else 0
        bc_variant = "D" if boundary else ""
        point_rows.append(PointRowRequest("R2", int(ridx), bc_variant, degree, row, factors))
        if boundary:
            nrows = prepare_neumann_point_rows(context, points, normal_coefficients=normal_coefficients,
                                                radial_degree=degree, patch_cache=patch_cache)
            for q, nrow in enumerate(nrows):
                neumann_rows.append(NeumannRowRequest("R2", int(ridx), q, degree, nrow))
    return point_rows, neumann_rows


# ---------------------------------------------------------------------------
# R3: face-side rows.
# ---------------------------------------------------------------------------
def build_r3_side_rows(S: StructuredReconstruction, context: PointRowContext, census: FaceCensus,
                       row_indices, face_points_by_row, *,
                       normal_coefficients, patch_cache,
                       capture_factors: bool = False) -> tuple[list[PointRowRequest], list[NeumannRowRequest]]:
    """R3: ``S.side_rows(key, points)`` (lower, upper) at the same nodes as R2.

    Mirrors ``p05n_field_derived_global.core.batched_side_values``: each
    side's own D-conditioned ``PointRows`` comes from its own boundary-map
    reconstruction (``fixed_anchor=True``, "as R1" -- conditioning depends
    on the *side cell's* radial index, independent of the face's own
    family). The Neumann companion, however, is built from
    ``prepare_neumann_point_rows(context, points, ...)`` with the face's own
    (unshifted) q3 nodes -- ``batched_side_values`` passes that identical
    ``points`` array to both its left and right ``cell_value`` calls, with
    no side-dependent shift, so the *result* is bitwise identical for lower
    and upper; this module therefore builds it once per face and tags it at
    the face's own census row index (not doubled per side), reused by
    whichever side(s) need it.
    """
    keys = census.keys()
    row_indices = np.asarray(row_indices, dtype=np.int64)
    point_rows: list[PointRowRequest] = []
    neumann_rows: list[NeumannRowRequest] = []
    for local, ridx in enumerate(row_indices):
        key = tuple(int(v) for v in keys[ridx])
        if key[0] == 0 and key[1] == 0:
            raise ValueError("R3 must not receive the collapsed r=0 census row")
        points = np.asarray(face_points_by_row[local], dtype=np.float64)
        if capture_factors:
            (lower, lower_factors), (upper, upper_factors) = S.side_rows_with_factors(key, points)
        else:
            (lower, upper), (lower_factors, upper_factors) = S.side_rows(key, points), (None, None)
        any_conditioned = False
        for side_code, row, factors in ((0, lower, lower_factors), (1, upper, upper_factors)):
            if row is None:
                continue
            entity_id = int(ridx) * 2 + side_code
            bc_variant = "D" if row.boundary_conditioned else ""
            degree = _SIDE_NEUMANN_DEGREE if row.boundary_conditioned else 0
            point_rows.append(PointRowRequest("R3", entity_id, bc_variant, degree, row, factors))
            any_conditioned = any_conditioned or bool(row.boundary_conditioned)
        if any_conditioned:
            nrows = prepare_neumann_point_rows(context, points, normal_coefficients=normal_coefficients,
                                                radial_degree=_SIDE_NEUMANN_DEGREE, patch_cache=patch_cache)
            for q, nrow in enumerate(nrows):
                neumann_rows.append(NeumannRowRequest("R3", int(ridx), q, _SIDE_NEUMANN_DEGREE, nrow))
    return point_rows, neumann_rows


# ---------------------------------------------------------------------------
# R4: P07 integrated face rows.
# ---------------------------------------------------------------------------
def build_r4_p07_rows(context: PointRowContext, census: FaceCensus, p07_row_indices,
                      face_points_by_row, face_weight_by_row, face_tensor_by_row, *,
                      normal_coefficients, patch_cache,
                      inner_support: str = "profile7") -> tuple[list[IntegratedRowRequest], list[NeumannRowRequest]]:
    """R4: ``prepare_integrated_face_rows`` at every P07-census face.

    ``p07_row_indices`` (census row indices; see :func:`p07_row_selection`),
    ``face_points_by_row`` (Q3 nodes), ``face_weight_by_row`` (q3 weights,
    the same shared quadrature primitive as R2/R3) and ``face_tensor_by_row``
    (the geometry-only P07 tensor at those nodes) must share the same row
    order and count. ``inner_support`` (``"profile7"`` default, or ``"fixed_radius"``) is the inner donor
    support of the P07 rows (see ``prepare_integrated_face_rows``). The integrand is P07's frozen normal-row convention,
    ``w_q * T[q, axis, :]`` (``contract_face_tensor``).

    Families 1 (physical_wall_quartic_BC), 2 (adjacent_radial_quartic_BC)
    and 4 (boundary_transverse_cubic_BC) are boundary-conditioned
    (``IntegratedFaceRow.boundary_conditioned``); mirroring
    ``scripts/p07n_field_derived_global/core.py``'s ``face_chunk`` (its
    ``neumann[j] = prepare_neumann_point_rows(c, p[j], ..., radial_degree=
    degree, patch_cache=patch_cache)``, ``degree = 3 if fam[j] == 4 else 4``),
    this also builds each such face's Neumann companion rows (design section
    3: "R4 also stores the precontracted P07N Neumann rows" -- stored here
    as the uncontracted per-q3-node ``NeumannPointRows``, since the actual
    contraction with a field's owner values is a runtime, field-dependent
    step out of this module's field-independent scope; see this module's
    docstring for the ``source='R4'`` entity-id convention).
    """
    keys = census.keys()
    p07_row_indices = np.asarray(p07_row_indices, dtype=np.int64)
    face_keys = keys[p07_row_indices]
    family = census.family[p07_row_indices].astype(np.int64)
    p07_ids = census.p07_id[p07_row_indices]
    points = np.asarray(face_points_by_row, dtype=np.float64)
    weights = np.asarray(face_weight_by_row, dtype=np.float64)
    tensor = np.asarray(face_tensor_by_row, dtype=np.float64)
    integrand = contract_face_tensor(weights, tensor, face_keys[:, 0])

    if inner_support == "profile7":
        rows = prepare_integrated_face_rows(context, face_keys, family, points, integrand)
    else:
        rows = prepare_integrated_face_rows(context, face_keys, family, points, integrand,
                                            inner_support=inner_support)
    result = [IntegratedRowRequest(int(p07_ids[j]), rows[j]) for j in range(len(rows))]

    neumann_rows: list[NeumannRowRequest] = []
    for j, fam in enumerate(family):
        degree = _P07_NEUMANN_DEGREE.get(int(fam))
        if degree is None:
            continue
        if not rows[j].boundary_conditioned:
            raise ValueError(f"P07 family {int(fam)} face expected boundary-conditioned, got unconditioned")
        nrows = prepare_neumann_point_rows(context, points[j], normal_coefficients=normal_coefficients,
                                            radial_degree=degree, patch_cache=patch_cache)
        for q, nrow in enumerate(nrows):
            neumann_rows.append(NeumannRowRequest("R4", int(p07_ids[j]), q, degree, nrow))
    return result, neumann_rows
