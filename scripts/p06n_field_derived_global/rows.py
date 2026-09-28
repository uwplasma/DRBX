"""P06N-specific addition to the P05N/P07N batched row primitives.

P06N reuses, unchanged, every catalogue-agnostic row-building primitive it
needs from ``p05n_field_derived_global.core`` (concurrently developed by
another agent; imported here read-only, never modified):

- ``batched_cell_values`` -- both BC-variant (value, gradient) at raw-midpoint
  cell targets, batched over an arbitrary field-column axis. Used for P06's
  q1 volume term (``numerics._continuum_terms`` needs value *and* gradient).
- ``batched_side_values`` -- both BC-variant (lower, upper) face-side values.
  Used for P06's q3 characteristic term's left/right states.

P06's q3 term needs one thing P05N's row primitives never computed: the
face's own common-row *value* (not gradient) -- ``numerics._compute_faces``
builds its central/left/right characteristic states entirely from
``rows.apply_value``, never a face-common gradient (P05N's bracket needs the
opposite: a common gradient, never a face value). This module adds exactly
that one function, ``batched_face_common_value``, mirroring
``core.batched_face_common_gradient`` line for line (same ``S.rows``/
``prepare_neumann_point_rows`` calls, same donor sets, same family routing
via ``p07_combined_global.kernels.family``) but calling ``row.apply_value``/
the value-only half of the Neumann row arithmetic instead of ``row.apply``.
``PointRows.apply`` and ``PointRows.apply_value`` compute the identical
``self.value @ data`` (plus the identical boundary lift ``value += gt``) --
see ``scripts/perpendicular_structured/reconstruction.py`` -- so this is not
a numerically distinct code path, only the (cheaper) one that skips the
unused gradient contraction.
"""
from __future__ import annotations

import numpy as np

from p07_combined_global import kernels as pk
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows


def _cached_normal_data(normal_data_fn, bc_cache, boundary_points):
    """Memoize ``normal_data_fn`` by wall-node coordinates.

    Duplicates ``p05n_field_derived_global.core._cached_normal_data`` (a
    private helper of that module) rather than importing a leading-underscore
    name across a package boundary; the eight-line memoization pattern is
    documented there and mirrored exactly here.
    """
    key = boundary_points.tobytes()
    cached = bc_cache.get(key)
    if cached is None:
        cached = normal_data_fn(boundary_points)
        bc_cache[key] = cached
    return cached


def batched_face_common_value(t, S, owner_values, key, points, *, normal_coefficients, ctx, patch_cache,
                                dirichlet_trace_fn, normal_data_fn, bc_cache=None, donor_accumulator=None):
    """Both BC-variant common-row VALUE at a batch of quadrature points on one face.

    Returns ``dict(dirichlet=(Q,F), neumann=(Q,F), family=str, boundary=bool,
    [condition, residual])``, exactly mirroring
    ``core.batched_face_common_gradient``'s return shape/contract but for the
    row's value instead of its gradient.
    """
    if bc_cache is None:
        bc_cache = {}
    n = t.n
    axis, i = int(key[0]), int(key[1])
    points = np.asarray(points, dtype=np.float64)
    Q = len(points)
    F = owner_values.shape[1]
    keytuple = tuple(int(v) for v in key)

    def track(ids):
        if donor_accumulator is not None:
            donor_accumulator.update(int(x) for x in ids)

    if axis == 0 and i == 0:
        row = S.rows(keytuple, points, location="face")
        v = np.asarray(row.apply_value(owner_values, None))
        track(row.donor_ids)
        return dict(dirichlet=v, neumann=v.copy(), family="collapsed_r0", boundary=False)
    kind = pk.family(n, key)
    boundary = kind in ("quartic_wall", "boundary_transverse")
    if not boundary:
        row = S.rows(keytuple, points, location="face")
        v = np.asarray(row.apply_value(owner_values, None))
        track(row.donor_ids)
        return dict(dirichlet=v, neumann=v.copy(), family=kind, boundary=False)
    row = S.rows(keytuple, points, location="face")
    v_d = np.asarray(row.apply_value(owner_values, dirichlet_trace_fn))
    track(row.donor_ids)
    degree = 4 if kind == "quartic_wall" else 3
    nrows = prepare_neumann_point_rows(ctx, points, normal_coefficients=normal_coefficients,
                                        radial_degree=degree, patch_cache=patch_cache)
    v_n = np.empty((Q, F))
    condition = np.empty(Q)
    residual = np.empty(Q)
    for q, nrow in enumerate(nrows):
        g_bc = _cached_normal_data(normal_data_fn, bc_cache, nrow.boundary_points)
        v_n[q] = nrow.value @ owner_values[nrow.donor_ids] + nrow.boundary_value @ g_bc
        condition[q] = nrow.condition
        residual[q] = nrow.constraint_residual
        track(nrow.donor_ids)
    return dict(dirichlet=v_d, neumann=v_n, family=kind, boundary=True, condition=condition, residual=residual)
