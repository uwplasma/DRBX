"""Per-field-and-boundary-condition row selector for the P05N Neumann bracket.

Every interior row, and every Dirichlet-field row (in any layer), is the
accepted ``StructuredReconstruction.rows``/``side_rows`` path, unchanged.
Neumann-field rows in the last two cell layers (or the equivalent boundary
face families) are replaced by ``prepare_neumann_point_rows``, built from the
*same* fixed-support patch (donors, radial nodes, angular stencils) as the
accepted boundary rows -- only the wall datum treatment differs (physical-
normal elimination instead of a prescribed trace lift).

Callers supply small closures rather than a field name directly, so this
module is field-catalogue-agnostic: it is exercised both by the new P05N
field-derived catalogue and, in all-Dirichlet mode, by a straight column
slice of the accepted P05 ``boundary_trace`` (for the bit-level replay
check against the accepted campaign).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from p07_combined_global import kernels as pk
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows


@dataclass
class RowDiagnostics:
    boundary_conditioned: bool
    neumann: bool
    condition: Optional[float] = None
    constraint_residual: Optional[float] = None
    family: Optional[str] = None
    donor_ids: Optional[np.ndarray] = None


def _wrap_single_field_trace(trace_fn: Callable):
    """(v,g) -> (v[:,None], g[:,:,None]) so PointRows.apply sees one column."""
    def trace(q):
        v, g = trace_fn(q)
        v = np.asarray(v); g = np.asarray(g)
        return v[:, None], g[:, :, None]
    return trace


def cell_rows(t, S, owner_column, keys, points, *, bc, dirichlet_trace=None, normal_coefficients=None,
              context=None, patch_cache=None, normal_data=None, radial_degree=3):
    """(value, gradient, [RowDiagnostics]) for a batch of raw-midpoint cell targets.

    ``keys`` is (Q,3) int cell addresses; ``points`` is (Q,3). Interior cells
    (radial index < n-2) always take the accepted, BC-independent path.
    Boundary cells route through ``bc`` in {'dirichlet','neumann'}.
    """
    n = t.n
    keys = np.asarray(keys, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    Q = len(points)
    value = np.empty(Q); gradient = np.empty((Q, 3))
    diagnostics = []
    boundary_mask = keys[:, 0] >= n - 2
    interior_idx = np.flatnonzero(~boundary_mask)
    boundary_idx = np.flatnonzero(boundary_mask)
    if len(interior_idx):
        for j in interior_idx:
            row = S.rows(tuple(int(v) for v in keys[j]), points[j:j + 1], location="cell")
            if row.boundary_conditioned:
                raise ValueError(f"interior cell unexpectedly boundary-conditioned key={tuple(keys[j])}")
            v, g = row.apply(owner_column[:, None], None)
            value[j] = v[0, 0]; gradient[j] = g[0, :, 0]
            diagnostics.append(RowDiagnostics(False, False, family=row.diagnostics.get("family"), donor_ids=row.donor_ids))
    for j in boundary_idx:
        key = tuple(int(v) for v in keys[j])
        row = S.rows(key, points[j:j + 1], location="cell")
        if not row.boundary_conditioned:
            raise ValueError(f"boundary cell unexpectedly unconditioned key={key}")
        if bc == "dirichlet":
            trace = _wrap_single_field_trace(dirichlet_trace)
            v, g = row.apply(owner_column[:, None], trace)
            value[j] = v[0, 0]; gradient[j] = g[0, :, 0]
            diagnostics.append(RowDiagnostics(True, False, family=row.diagnostics.get("family"), donor_ids=row.donor_ids))
        elif bc == "neumann":
            nrow = prepare_neumann_point_rows(context, points[j:j + 1], normal_coefficients=normal_coefficients,
                                               radial_degree=radial_degree, patch_cache=patch_cache)[0]
            g_n = np.asarray(normal_data(nrow.boundary_points))
            value[j] = nrow.value @ owner_column[nrow.donor_ids] + nrow.boundary_value @ g_n
            gradient[j] = nrow.gradient @ owner_column[nrow.donor_ids] + nrow.boundary_gradient @ g_n
            diagnostics.append(RowDiagnostics(True, True, condition=float(nrow.condition),
                                               constraint_residual=float(nrow.constraint_residual),
                                               family="neumann_boundary_transverse", donor_ids=nrow.donor_ids))
        else:
            raise ValueError(f"unknown boundary condition {bc!r}")
    return value, gradient, diagnostics


def face_common_gradient(t, S, owner_column, key, points, *, bc, dirichlet_trace=None,
                          normal_coefficients=None, context=None, patch_cache=None, normal_data=None):
    """Common-row gradient at a batch of quadrature points on one face.

    ``key`` is the single (axis,i,j,k) face address shared by all points.
    Regular/centered-radial (non-boundary) faces always take the accepted,
    BC-independent path. Boundary families ('quartic_wall' degree 4,
    'boundary_transverse' degree 3) route through ``bc``.
    """
    n = t.n
    axis, i = int(key[0]), int(key[1])
    points = np.asarray(points, dtype=np.float64)
    Q = len(points)
    if axis == 0 and i == 0:
        row = S.rows(tuple(int(v) for v in key), points, location="face")
        v, g = row.apply(owner_column[:, None], None)
        return g[:, :, 0], [RowDiagnostics(False, False, family="collapsed_r0", donor_ids=row.donor_ids) for _ in range(Q)]
    kind = pk.family(n, key)
    boundary = kind in ("quartic_wall", "boundary_transverse")
    if not boundary:
        row = S.rows(tuple(int(v) for v in key), points, location="face")
        v, g = row.apply(owner_column[:, None], None)
        return g[:, :, 0], [RowDiagnostics(False, False, family=kind, donor_ids=row.donor_ids) for _ in range(Q)]
    if bc == "dirichlet":
        row = S.rows(tuple(int(v) for v in key), points, location="face")
        trace = _wrap_single_field_trace(dirichlet_trace)
        v, g = row.apply(owner_column[:, None], trace)
        return g[:, :, 0], [RowDiagnostics(True, False, family=kind, donor_ids=row.donor_ids) for _ in range(Q)]
    if bc != "neumann":
        raise ValueError(f"unknown boundary condition {bc!r}")
    degree = 4 if kind == "quartic_wall" else 3
    nrows = prepare_neumann_point_rows(context, points, normal_coefficients=normal_coefficients,
                                        radial_degree=degree, patch_cache=patch_cache)
    gradient = np.empty((Q, 3)); diagnostics = []
    for q, row in enumerate(nrows):
        g_n = np.asarray(normal_data(row.boundary_points))
        gradient[q] = row.gradient @ owner_column[row.donor_ids] + row.boundary_gradient @ g_n
        diagnostics.append(RowDiagnostics(True, True, condition=float(row.condition),
                                           constraint_residual=float(row.constraint_residual),
                                           family=f"neumann_{kind}", donor_ids=row.donor_ids))
    return gradient, diagnostics


def side_values(t, S, owner_column, key, points, *, bc, dirichlet_trace=None,
                 normal_coefficients=None, context=None, patch_cache=None, normal_data=None):
    """(lower_value, upper_value, boundary_conditioned_pair) at a batch of face points.

    Mirrors ``StructuredReconstruction.side_rows``: both sides are 'cell'
    location, so a side is boundary-conditioned solely by its own radial
    cell index (>= n-2), independent of the face's own family. The physical
    wall exterior (no upper side) falls back to the prescribed trace for a
    Dirichlet field, and to the lower side's own recovered value for a
    Neumann field (the adopted wall-exterior contract: no manufactured wall
    state; the exterior *is* the recovered trace).
    """
    n = t.n
    axis, *ijk = (int(v) for v in key)
    left = list(ijk); left[axis] -= 1
    right = list(ijk)
    points = np.asarray(points, dtype=np.float64)
    Q = len(points)

    def cell_value(cell, exists):
        if not exists:
            return None, None
        cell = cell.copy()
        cell[1] %= n; cell[2] %= n
        conditioned = cell[0] >= n - 2
        cellkey = tuple(cell)
        row = S.rows(cellkey, points, location="cell", fixed_anchor=True)
        if bool(row.boundary_conditioned) != conditioned:
            raise ValueError(f"side BC-conditioning mismatch key={cellkey} expected={conditioned}")
        if not conditioned:
            v, _ = row.apply(owner_column[:, None], None)
            return v[:, 0], False
        if bc == "dirichlet":
            trace = _wrap_single_field_trace(dirichlet_trace)
            v, _ = row.apply(owner_column[:, None], trace)
            return v[:, 0], True
        if bc != "neumann":
            raise ValueError(f"unknown boundary condition {bc!r}")
        nrows = prepare_neumann_point_rows(context, points, normal_coefficients=normal_coefficients,
                                            radial_degree=3, patch_cache=patch_cache)
        v = np.empty(Q)
        for q, nrow in enumerate(nrows):
            g_n = np.asarray(normal_data(nrow.boundary_points))
            v[q] = nrow.value @ owner_column[nrow.donor_ids] + nrow.boundary_value @ g_n
        return v, True

    left_exists = not (axis == 0 and left[0] < 0)
    right_exists = not (axis == 0 and right[0] >= n)
    lv, lb = cell_value(np.array(left), left_exists)
    rv, rb = cell_value(np.array(right), right_exists)
    if lv is None and rv is None:
        raise ValueError("a face cannot be exterior on both sides")
    if lv is None:
        # Collapsed r=0 interior seam only; never reached near the wall.
        if bc == "dirichlet":
            lv, _ = dirichlet_trace(points)
        else:
            lv = rv.copy()
        lb = rb
    if rv is None:
        if bc == "dirichlet":
            rv, _ = dirichlet_trace(points)
        else:
            rv = lv.copy()
        rb = lb
    return lv, rv, (lb, rb)
