"""Batched P05N field-derived kernels: the fast path validated against the
per-owner oracle in ``p05n_field_derived_global.operator``/``rows``.

Do not import this package's own directory first on ``sys.path`` (it shadows
the stdlib ``operator`` -- see ``operator.py``'s module docstring). Callers
put ``DRBX/scripts`` on ``sys.path`` and import ``p05n_field_derived_global.core``.

Design (see the campaign task spec / README.md for the full rationale):

* Row geometry (``StructuredReconstruction.rows`` / ``prepare_neumann_point_rows``)
  is built once per target point (or once per face, batched over that face's
  quadrature points) -- independent of which catalogue field it is applied to.
* Every catalogue physical field is then applied to that one row through a
  single matrix product: ``owner_values`` carries one column per physical
  field (shape ``(owners, fields)``), and ``PointRows.apply``/the Neumann row
  arithmetic are already generic over that trailing field axis (unchanged
  from ``rows.py``/``fci_perpendicular_neumann_trace.py``).
* ``ref._metric`` is called once per chunk over every raw midpoint / face
  quadrature point in that chunk, not once per (owner, pairing).
* ``drbx.native.fci_perpendicular_face_corrections.p05_scalar_face_jump`` and
  ``p05_direct_midpoint_global.direct_operator.direct_pair_actions`` already
  vectorize over an arbitrary list of (generator, transported) index pairs
  into a shared trailing field axis, so every catalogue pairing (and its
  matched-Dirichlet diagnostic) is evaluated in one call per chunk.

This module is catalogue-agnostic at the kernel level (``batched_cell_values``,
``batched_face_common_gradient``, ``batched_side_values`` take an explicit
``owner_values`` matrix and trace/normal-data callables); the P05N ten-role
frozen catalogue tables below (``ROLES``, ``PAIRINGS``, ...) are the specific
instantiation used by ``campaign.py``, and are also reused, with a different
(8-field, all-Dirichlet) instantiation, for the accepted-P05-campaign replay
preflight check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p07_combined_global import kernels as pk
from p07_combined_global import topology as pt
from perpendicular_structured.reconstruction import StructuredReconstruction, load_context
from p07_diffusion_global import numerics as refnum
from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
from drbx.native.fci_perpendicular_face_corrections import p05_scalar_face_jump
from p05_direct_midpoint_global.direct_operator import point_bracket, direct_pair_actions

from p05n_field_derived_global import fields as p05n_fields
from p05n_field_derived_global import operator as p05n_operator

# ---------------------------------------------------------------------------
# Frozen P05N catalogues. configuration.json's catalogue_reference selects one;
# each is vendored in this package directory and checked by SHA-256.
#   p05n_catalogue.json         frozen_v1 (campaign 05be9063)
#   p05n_upwind_catalogue.json  upwind_v1 (fields outside the reconstruction's
#                               exactness space, so the live U - A jump is exercised)
# Role names carry their boundary condition as a _N / _D suffix.
# ---------------------------------------------------------------------------
_CATALOGUE_TABLES = {
    "p05n_catalogue.json": dict(
        names=p05n_fields.NAMES,
        roles={
            "b1_N": ("field_b1", "neumann"), "b1_D": ("field_b1", "dirichlet"),
            "e3_N": ("field_e3", "neumann"), "e3_D": ("field_e3", "dirichlet"),
            "e12_N": ("field_e12", "neumann"), "e12_D": ("field_e12", "dirichlet"),
            "b2_N": ("heldout_field_b2", "neumann"), "b2_D": ("heldout_field_b2", "dirichlet"),
            "ztg_D": ("zero_trace_generator", "dirichlet"),
            "constant_D": ("constant", "dirichlet"),
        },
        pairings={
            "a_main1": ("b1_N", "e3_N"),
            "a_main2": ("e12_N", "b1_N"),
            "a_heldout": ("b2_N", "e12_N"),
            "a_control1": ("b1_N", "constant_D"),
            "a_control2": ("constant_D", "e3_N"),
            "b_main1": ("b1_D", "e3_N"),
            "b_main2": ("ztg_D", "e12_N"),
            "b_heldout": ("b2_D", "b1_N"),
            "b_control1": ("b1_D", "constant_D"),
            "b_control2": ("constant_D", "e3_N"),
        }),
    "p05n_upwind_catalogue.json": dict(
        names=("field_b1", "field_e3", "constant", "rich_a", "rich_f", "heldout_rich_g"),
        roles={
            "b1_N": ("field_b1", "neumann"), "b1_D": ("field_b1", "dirichlet"),
            "e3_N": ("field_e3", "neumann"), "e3_D": ("field_e3", "dirichlet"),
            "ra_N": ("rich_a", "neumann"), "ra_D": ("rich_a", "dirichlet"),
            "rf_N": ("rich_f", "neumann"), "rf_D": ("rich_f", "dirichlet"),
            "rg_N": ("heldout_rich_g", "neumann"), "rg_D": ("heldout_rich_g", "dirichlet"),
            "constant_D": ("constant", "dirichlet"),
        },
        pairings={
            "a_main1": ("ra_N", "rf_N"),
            "a_main2": ("rf_N", "ra_N"),
            "a_heldout": ("ra_N", "rg_N"),
            "a_main_regression": ("b1_N", "e3_N"),
            "a_control1": ("ra_N", "constant_D"),
            "a_control2": ("constant_D", "rf_N"),
            "b_main1": ("ra_D", "rf_N"),
            "b_heldout": ("ra_D", "rg_N"),
            "b_control1": ("ra_D", "constant_D"),
        }),
}
CATALOGUE_REFERENCE = json.loads((Path(__file__).resolve().parent / "configuration.json").read_text())["catalogue_reference"]
_TABLE = _CATALOGUE_TABLES[CATALOGUE_REFERENCE]

NAMES = tuple(_TABLE["names"])  # physical fields of the active catalogue, fixed column order for owner_values
NAME_INDEX = {name: i for i, name in enumerate(NAMES)}

# role name -> (physical field name, boundary condition)
ROLES = dict(_TABLE["roles"])
ROLE_NAMES = tuple(sorted(ROLES))
ROLE_INDEX = {r: i for i, r in enumerate(ROLE_NAMES)}
ROLE_PHYSICAL_INDEX = np.array([NAME_INDEX[ROLES[r][0]] for r in ROLE_NAMES], dtype=np.int64)
ROLE_BC = {r: ROLES[r][1] for r in ROLE_NAMES}

# Every role's Dirichlet counterpart (self-mapped if already Dirichlet), used
# to build the matched-Dirichlet diagnostic D for every catalogue pairing.
DIRICHLET_COUNTERPART = {r: (r[:-2] + "_D" if r.endswith("_N") else r) for r in ROLES}

# pairing name -> (generator role, transported role); mirrors the active
# catalogue's pairings exactly. 'constant' is treated as Dirichlet-role
# throughout (documented in README.md).
PAIRINGS = dict(_TABLE["pairings"])
PAIR_NAMES = tuple(sorted(PAIRINGS))  # fixed order
MAIN_PAIRS = tuple(p for p in PAIR_NAMES if "main" in p)
HELDOUT_PAIRS = tuple(p for p in PAIR_NAMES if "heldout" in p)
CONTROL_PAIRS = tuple(p for p in PAIR_NAMES if "control" in p)
CONSTANT_PAIRS = tuple(p for p in PAIR_NAMES if "constant_D" in PAIRINGS[p])
GATED_PAIRS = MAIN_PAIRS + HELDOUT_PAIRS  # N-R order gate scope (nonconstant main + held-out, both candidates)

# index pairs into ROLE_NAMES (reconstructed-role axis), one entry per PAIR_NAMES
_N_PAIR_INDEX = [(ROLE_INDEX[PAIRINGS[p][0]], ROLE_INDEX[PAIRINGS[p][1]]) for p in PAIR_NAMES]
_D_PAIR_INDEX = [(ROLE_INDEX[DIRICHLET_COUNTERPART[PAIRINGS[p][0]]],
                  ROLE_INDEX[DIRICHLET_COUNTERPART[PAIRINGS[p][1]]]) for p in PAIR_NAMES]
# First 10 columns: candidate N (as declared). Next 10: matched-Dirichlet diagnostic D.
ACTION_PAIR_INDEX = tuple(_N_PAIR_INDEX) + tuple(_D_PAIR_INDEX)

# index pairs into NAMES (exact physical-field axis), for the analytic reference R.
# R depends only on the physical fields in a pairing, not on the BC role, so
# e.g. R('a_main1') == R('b_main1') by construction (both are (field_b1,field_e3)).
R_PAIR_INDEX = [(int(ROLE_PHYSICAL_INDEX[ROLE_INDEX[PAIRINGS[p][0]]]),
                 int(ROLE_PHYSICAL_INDEX[ROLE_INDEX[PAIRINGS[p][1]]])) for p in PAIR_NAMES]


def context(t):
    return PointRowContext.from_arrays(faces=t.faces, centers=t.centers, raw_to_owner=t.ro,
                                        raw_volume=t.rv, owner_volume=t.vol,
                                        owner_centroid_xy=t.g.owner_centroid_xy, eta_period=t.g.eta_period,
                                        dr=t.g.dr, dtheta=t.g.dtheta, deta=t.g.deta)


def load(input_root, sidecar, n):
    t = load_context(n, input_root)
    ref = refnum.reference(sidecar, verify_hashes=False)
    return t, ref


# ---------------------------------------------------------------------------
# P05N six-physical-field batched trace/normal-data/exact-gradient helpers.
# ---------------------------------------------------------------------------
def dirichlet_trace_all(ref, q, period):
    """(v, g) stacked over NAMES; v shape (Q,6), g shape (Q,3,6)."""
    q = np.asarray(q, dtype=np.float64)
    Q = len(q)
    v = np.empty((Q, len(NAMES)))
    g = np.empty((Q, 3, len(NAMES)))
    for j, name in enumerate(NAMES):
        vv, gg = p05n_fields.dirichlet_trace(ref, q, name, period)
        v[:, j] = vv
        g[:, :, j] = gg
    return v, g


def normal_data_all(ref, q, period):
    """(Q,6) g_N stacked over NAMES.

    Calls ``ref._metric`` exactly once (via ``p05n_fields.normal``), not once
    per field: ``p05n_fields.normal_data`` recomputes the physical normal
    per field, which would silently reintroduce the very per-(owner,field)
    ``ref._metric`` redundancy this campaign batches away (found by direct
    profiling: this one fix took the boundary-face kernels from ~0.66s/face
    to the sub-0.1s/face range reported in the timing measurement).
    """
    q = np.asarray(q, dtype=np.float64)
    a = p05n_fields.normal(ref, q)
    out = np.empty((len(q), len(NAMES)))
    for j, name in enumerate(NAMES):
        _, g, _ = p05n_fields.evaluate(ref, q, name, period)
        out[:, j] = np.einsum("qa,qa->q", a, g)
    return out


def exact_grad_all(ref, q, period):
    """(Q,3,6) exact analytic gradient stacked over NAMES."""
    q = np.asarray(q, dtype=np.float64)
    Q = len(q)
    g = np.empty((Q, 3, len(NAMES)))
    for j, name in enumerate(NAMES):
        _, gg, _ = p05n_fields.evaluate(ref, q, name, period)
        g[:, :, j] = gg
    return g


def memoize_single_target_cardinal():
    """Memoize the frozen trigonometric cardinal for single-target calls, exactly.

    ``StructuredReconstruction._tensor`` evaluates
    ``p07_combined_global.kernels.cardinal(nodes, [theta])`` once per point,
    layer and eta plane, and the same (nodes, theta) pair recurs across
    quadrature points, eta planes, the common/side rows of a face and
    neighbouring faces. Each call costs about 1 ms (a longdouble loop over
    ``np.delete``), and these calls dominated the interior face cost. The cache
    key is the exact bytes and dtype of both arguments, and a hit returns the
    array the original function produced for those inputs, so results are
    bitwise identical. Only this process's module attribute is replaced; the
    shared source is unchanged. Multi-target calls (the cached ring fits) are
    passed through.
    """
    original = pk.cardinal
    if getattr(original, "_p05n_memo", False):
        return
    cache = {}

    def cardinal(nodes, targets):
        nodes_array = np.asarray(nodes)
        target_array = np.asarray(targets)
        if target_array.size != 1:
            return original(nodes, targets)
        key = (nodes_array.dtype.str, nodes_array.tobytes(), target_array.dtype.str, target_array.tobytes())
        hit = cache.get(key)
        if hit is None:
            if len(cache) > 500_000:
                cache.clear()
            value, derivative = original(nodes, targets)
            value.setflags(write=False); derivative.setflags(write=False)
            hit = cache[key] = (value, derivative)
        return hit

    cardinal._p05n_memo = True
    cardinal._p05n_original = original
    pk.cardinal = cardinal


memoize_single_target_cardinal()


class WallLattice:
    """Physical normal and field-derived g_N on the fixed Neumann wall lattice.

    ``prepare_neumann_point_rows`` places every wall node at
    ``(1, t.centers[1][i], t.centers[2][k])``, so all Neumann wall data live on
    one n x n lattice. Both are evaluated here once per grid (one metric call)
    and looked up by exact coordinate. Before, they were evaluated per patch
    and per target, and those metric calls dominated the wall cost. A query
    point that is not bitwise a lattice node raises an error.
    """

    def __init__(self, t, ref, period, names=None):
        self.names = tuple(NAMES if names is None else names)
        theta = np.asarray(t.centers[1], dtype=np.float64)
        eta = np.asarray(t.centers[2], dtype=np.float64)
        self._theta_index = {float(v): i for i, v in enumerate(theta)}
        self._eta_index = {float(v): k for k, v in enumerate(eta)}
        self._eta_count = len(eta)
        self.points = np.column_stack((np.ones(len(theta) * len(eta)),
                                       np.repeat(theta, len(eta)), np.tile(eta, len(theta))))
        self.a = p05n_fields.normal(ref, self.points)
        if names is None:
            self.g_n = normal_data_all(ref, self.points, period)
        else:
            gradients = [p05n_fields.evaluate(ref, self.points, name, period)[1] for name in self.names]
            self.g_n = np.stack([np.einsum("qa,qa->q", self.a, g) for g in gradients], axis=1)

    def index(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3 or np.any(q[:, 0] != 1.0):
            raise ValueError("wall-lattice query must be (Q,3) points with u == 1")
        try:
            return np.fromiter((self._theta_index[float(x)] * self._eta_count + self._eta_index[float(y)]
                                for x, y in q[:, 1:]), dtype=np.int64, count=len(q))
        except KeyError as exc:
            raise ValueError("wall-lattice query point is not a lattice node") from exc

    def normal(self, q):
        return self.a[self.index(q)]

    def normal_data(self, q):
        return self.g_n[self.index(q)]


def live_observations(t, values):
    return p05n_operator.live_observations(t, values)


# ---------------------------------------------------------------------------
# Generic (catalogue-agnostic) batched row kernels.
#
# ``owner_values`` has shape (owners, F): one column per catalogue field
# (physical field for the P05N six-field catalogue; the accepted P05 8-field
# catalogue for the all-Dirichlet replay check). ``dirichlet_trace_fn(q)`` ->
# (v (Q,F), g (Q,3,F)); ``normal_data_fn(q)`` -> (Q,F).
# ---------------------------------------------------------------------------
def _cached_normal_data(normal_data_fn, bc_cache, boundary_points):
    """Memoize ``normal_data_fn`` by wall-node coordinates.

    Many quadrature/cell targets sharing one Neumann patch (same theta/eta
    cardinal nodes) share the identical 28 ``boundary_points``; recomputing
    ``normal_data_fn`` (a ``ref._metric`` call) for each target instead of
    once per unique wall-node set would silently reintroduce the very
    per-target ``ref._metric`` redundancy this campaign batches away.
    Mirrors ``scripts/p07n_field_derived_global/core.py``'s ``bc_cache``.
    """
    key = boundary_points.tobytes()
    cached = bc_cache.get(key)
    if cached is None:
        cached = normal_data_fn(boundary_points)
        bc_cache[key] = cached
    return cached


def batched_cell_values(t, S, owner_values, keys, points, *, normal_coefficients, ctx, patch_cache,
                         dirichlet_trace_fn, normal_data_fn, radial_degree=3, bc_cache=None,
                         donor_accumulator=None):
    """Both BC-variant (value, gradient) at raw-midpoint cell targets, batched over fields.

    Mirrors ``rows.cell_rows`` exactly (same ``S.rows``/``prepare_neumann_point_rows``
    calls, same donor sets), generalized from one field per call to the full
    ``owner_values`` field axis, and computing both the Dirichlet-role and
    Neumann-role reconstruction in one pass (interior cells: identical, one
    reconstruction; boundary cells: both variants, one shared
    ``prepare_neumann_point_rows`` call across every boundary point in this chunk).

    ``donor_accumulator``, when given a ``set``, is updated with every owner id
    used as a donor by any row built in this call (both BC variants). Used
    only by ``preflight_fixtures/extract.py`` to compute the closure of owner
    ids a fixture must carry; never touched by the campaign/preflight path.
    """
    n = t.n
    keys = np.asarray(keys, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    Q = len(points)
    F = owner_values.shape[1]
    value_d = np.empty((Q, F)); grad_d = np.empty((Q, 3, F))
    value_n = np.empty((Q, F)); grad_n = np.empty((Q, 3, F))
    boundary_conditioned = np.zeros(Q, dtype=bool)
    condition = np.zeros(Q); residual = np.zeros(Q)
    if bc_cache is None:
        bc_cache = {}
    boundary_mask = keys[:, 0] >= n - 2
    interior_idx = np.flatnonzero(~boundary_mask)
    boundary_idx = np.flatnonzero(boundary_mask)
    for j in interior_idx:
        row = S.rows(tuple(int(v) for v in keys[j]), points[j:j + 1], location="cell")
        if row.boundary_conditioned:
            raise ValueError(f"interior cell unexpectedly boundary-conditioned key={tuple(keys[j])}")
        v, g = row.apply(owner_values, None)
        value_d[j] = value_n[j] = v[0]
        grad_d[j] = grad_n[j] = g[0]
        if donor_accumulator is not None:
            donor_accumulator.update(int(x) for x in row.donor_ids)
    for j in boundary_idx:
        key = tuple(int(v) for v in keys[j])
        row = S.rows(key, points[j:j + 1], location="cell")
        if not row.boundary_conditioned:
            raise ValueError(f"boundary cell unexpectedly unconditioned key={key}")
        v, g = row.apply(owner_values, dirichlet_trace_fn)
        value_d[j] = v[0]; grad_d[j] = g[0]
        boundary_conditioned[j] = True
        if donor_accumulator is not None:
            donor_accumulator.update(int(x) for x in row.donor_ids)
    if len(boundary_idx):
        bpoints = points[boundary_idx]
        nrows = prepare_neumann_point_rows(ctx, bpoints, normal_coefficients=normal_coefficients,
                                            radial_degree=radial_degree, patch_cache=patch_cache)
        for local, nrow in enumerate(nrows):
            j = boundary_idx[local]
            g_n = _cached_normal_data(normal_data_fn, bc_cache, nrow.boundary_points)
            value_n[j] = nrow.value @ owner_values[nrow.donor_ids] + nrow.boundary_value @ g_n
            grad_n[j] = nrow.gradient @ owner_values[nrow.donor_ids] + nrow.boundary_gradient @ g_n
            condition[j] = nrow.condition
            residual[j] = nrow.constraint_residual
            if donor_accumulator is not None:
                donor_accumulator.update(int(x) for x in nrow.donor_ids)
    return dict(dirichlet=(value_d, grad_d), neumann=(value_n, grad_n),
                boundary_conditioned=boundary_conditioned, condition=condition, residual=residual)


def batched_face_common_gradient(t, S, owner_values, key, points, *, normal_coefficients, ctx, patch_cache,
                                  dirichlet_trace_fn, normal_data_fn, bc_cache=None, donor_accumulator=None):
    """Both BC-variant common-row gradient at a batch of quadrature points on one face.

    Mirrors ``rows.face_common_gradient``, generalized to the full field axis.
    See ``batched_cell_values`` for ``donor_accumulator``.
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
        _, g = row.apply(owner_values, None)
        track(row.donor_ids)
        return dict(dirichlet=g, neumann=g, family="collapsed_r0", boundary=False)
    kind = pk.family(n, key)
    boundary = kind in ("quartic_wall", "boundary_transverse")
    if not boundary:
        row = S.rows(keytuple, points, location="face")
        _, g = row.apply(owner_values, None)
        track(row.donor_ids)
        return dict(dirichlet=g, neumann=g, family=kind, boundary=False)
    row = S.rows(keytuple, points, location="face")
    _, g_d = row.apply(owner_values, dirichlet_trace_fn)
    track(row.donor_ids)
    degree = 4 if kind == "quartic_wall" else 3
    nrows = prepare_neumann_point_rows(ctx, points, normal_coefficients=normal_coefficients,
                                        radial_degree=degree, patch_cache=patch_cache)
    g_n = np.empty((Q, 3, F))
    condition = np.empty(Q); residual = np.empty(Q)
    for q, nrow in enumerate(nrows):
        g_bc = _cached_normal_data(normal_data_fn, bc_cache, nrow.boundary_points)
        g_n[q] = nrow.gradient @ owner_values[nrow.donor_ids] + nrow.boundary_gradient @ g_bc
        condition[q] = nrow.condition
        residual[q] = nrow.constraint_residual
        track(nrow.donor_ids)
    return dict(dirichlet=g_d, neumann=g_n, family=kind, boundary=True, condition=condition, residual=residual)


def batched_side_values(t, S, owner_values, key, points, *, normal_coefficients, ctx, patch_cache,
                         dirichlet_trace_fn, normal_data_fn, bc_cache=None, donor_accumulator=None):
    """Both BC-variant (lower, upper) side values at a batch of face points.

    Mirrors ``rows.side_values``, generalized to the full field axis, per BC
    variant independently (the wall-exterior fallback -- prescribed trace for
    Dirichlet, recovered-trace copy for Neumann -- is applied separately for
    each variant, since a role's own BC only determines which variant a given
    catalogue pairing actually uses). See ``batched_cell_values`` for
    ``donor_accumulator``.
    """
    if bc_cache is None:
        bc_cache = {}
    n = t.n
    axis, *ijk = (int(v) for v in key)
    left = list(ijk); left[axis] -= 1
    right = list(ijk)
    points = np.asarray(points, dtype=np.float64)
    Q = len(points)
    F = owner_values.shape[1]

    def cell_value(cell, exists):
        if not exists:
            return None
        cell = cell.copy(); cell[1] %= n; cell[2] %= n
        conditioned = cell[0] >= n - 2
        cellkey = tuple(int(v) for v in cell)
        row = S.rows(cellkey, points, location="cell", fixed_anchor=True)
        if bool(row.boundary_conditioned) != conditioned:
            raise ValueError(f"side BC-conditioning mismatch key={cellkey} expected={conditioned}")
        if not conditioned:
            v, _ = row.apply(owner_values, None)
            if donor_accumulator is not None:
                donor_accumulator.update(int(x) for x in row.donor_ids)
            return dict(dirichlet=v, neumann=v.copy(), conditioned=False)
        vd, _ = row.apply(owner_values, dirichlet_trace_fn)
        if donor_accumulator is not None:
            donor_accumulator.update(int(x) for x in row.donor_ids)
        nrows = prepare_neumann_point_rows(ctx, points, normal_coefficients=normal_coefficients,
                                            radial_degree=3, patch_cache=patch_cache)
        vn = np.empty((Q, F))
        for q, nrow in enumerate(nrows):
            g_bc = _cached_normal_data(normal_data_fn, bc_cache, nrow.boundary_points)
            vn[q] = nrow.value @ owner_values[nrow.donor_ids] + nrow.boundary_value @ g_bc
            if donor_accumulator is not None:
                donor_accumulator.update(int(x) for x in nrow.donor_ids)
        return dict(dirichlet=vd, neumann=vn, conditioned=True)

    left_exists = not (axis == 0 and left[0] < 0)
    right_exists = not (axis == 0 and right[0] >= n)
    L = cell_value(np.array(left), left_exists)
    R = cell_value(np.array(right), right_exists)
    if L is None and R is None:
        raise ValueError("a face cannot be exterior on both sides")
    trace_v = None
    if L is None or R is None:
        trace_v, _ = dirichlet_trace_fn(points)
    if L is None:
        L = dict(dirichlet=trace_v, neumann=R["neumann"].copy(), conditioned=R["conditioned"])
    if R is None:
        R = dict(dirichlet=trace_v, neumann=L["neumann"].copy(), conditioned=L["conditioned"])
    return dict(lower=L, upper=R)


def _select_role_matrix(bc_dict, role_names, role_bc, role_physical_index, is_gradient=None):
    """Gather a (..., roles) array from a {'dirichlet': X, 'neumann': X} pair of
    (..., physical_fields) arrays, picking each role's own BC variant.

    ``is_gradient`` is accepted (and ignored) for call-site readability only:
    the trailing-axis gather is the same shape operation for a value array
    (..., fields) or a gradient array (..., 3, fields).
    """
    d = bc_dict["dirichlet"]; n = bc_dict["neumann"]
    out = np.empty(d.shape[:-1] + (len(role_names),))
    for r, role in enumerate(role_names):
        src = d if role_bc[role] == "dirichlet" else n
        out[..., r] = src[..., role_physical_index[r]]
    return out


# ---------------------------------------------------------------------------
# Chunk-level compute: observations / raw (centered+reference) / faces (jump).
# ---------------------------------------------------------------------------
def observation_chunk(t, ref, raw_ids, period):
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    q = t.pts[raw_ids]
    values = np.column_stack([p05n_fields.evaluate(ref, q, name, period)[0] for name in NAMES])
    return dict(ids=raw_ids, owner_ids=t.ro[raw_ids], numerator=t.rv[raw_ids, None] * values)


def raw_chunk(t, S, ref, ctx, raw_ids, owner_values, normal_coefficients, patch_cache, period,
              normal_data_override=None):
    """Centered N/D actions and the analytic reference R at every raw midpoint in the chunk."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    points = t.pts[raw_ids]
    keys = np.array(np.unravel_index(raw_ids, (t.n,) * 3)).T
    owner_ids = t.ro[raw_ids]

    def dirichlet_trace_fn(q):
        return dirichlet_trace_all(ref, q, period)

    def normal_data_fn(q):
        if normal_data_override is not None:
            return normal_data_override(q)
        return normal_data_all(ref, q, period)

    bc_cache = {}
    cellvals = batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                    ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=dirichlet_trace_fn,
                                    normal_data_fn=normal_data_fn, bc_cache=bc_cache)
    grad_roles = _select_role_matrix(dict(dirichlet=cellvals["dirichlet"][1], neumann=cellvals["neumann"][1]),
                                      ROLE_NAMES, ROLE_BC, ROLE_PHYSICAL_INDEX, is_gradient=True)

    metric = ref._metric(points)
    h = metric["bcov"] / metric["B"][:, None]
    jac = np.abs(metric["J"])

    actions, antisymmetry = direct_pair_actions(h, jac, grad_roles, ACTION_PAIR_INDEX)
    N = actions[:, :len(PAIR_NAMES)]
    D = actions[:, len(PAIR_NAMES):]

    exact_grad = exact_grad_all(ref, points, period)
    R = np.empty((len(points), len(PAIR_NAMES)))
    for col, (a, b) in enumerate(R_PAIR_INDEX):
        R[:, col] = point_bracket(h, jac, exact_grad[:, :, a], exact_grad[:, :, b])
    exact_input_action, _ = direct_pair_actions(h, jac, exact_grad, R_PAIR_INDEX)
    exact_input_defect = np.max(np.abs(R - exact_input_action), axis=0) if len(points) else np.zeros(len(PAIR_NAMES))

    condition_max = float(np.max(cellvals["condition"], initial=0.0))
    residual_max = float(np.max(cellvals["residual"], initial=0.0))
    return dict(ids=raw_ids, owner_ids=owner_ids, raw_volume=t.rv[raw_ids],
                N=N, D=D, R=R, antisymmetry=antisymmetry, exact_input_defect=exact_input_defect,
                condition_max=condition_max, residual_max=residual_max,
                boundary_conditioned=cellvals["boundary_conditioned"])


def face_chunk(t, S, ref, ctx, face_ids, endpoints, owner_values, normal_coefficients, patch_cache, period,
               order=3, normal_data_override=None):
    """Live jump N/D per face, plus per-face-family zero-check bookkeeping."""
    face_ids = np.asarray(face_ids, dtype=np.int64)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    keys = pt.decode(t.n, face_ids)
    Fcount = len(face_ids)
    P = len(PAIR_NAMES)
    N = np.zeros((Fcount, P)); D = np.zeros((Fcount, P))
    # Face kind (physical_wall/radial_n_minus_1/transverse_last_two_layers/None)
    # is cheap to recompute from (n, key) and is not saved (avoids an
    # allow_pickle=False-incompatible object array in the chunk NPZ).
    condition_max = 0.0; residual_max = 0.0

    def dirichlet_trace_fn(q):
        return dirichlet_trace_all(ref, q, period)

    def normal_data_fn(q):
        if normal_data_override is not None:
            return normal_data_override(q)
        return normal_data_all(ref, q, period)

    bc_cache = {}
    # One quadrature call and one metric call for the whole chunk; per-face
    # metric calls on 9 points were the dominant face cost (numpy overhead).
    all_points, all_weights = pk.num.quadrature(t.faces, keys, order, face=True)
    # The collapsed r=0 radial faces (skipped below) sit at u=0, where the
    # ordinary metric is singular, so they are excluded from the batch.
    evaluated = ~((keys[:, 0] == 0) & (keys[:, 1] == 0))
    all_h = np.full(all_points.shape, np.nan)
    if np.any(evaluated):
        quad = all_points[evaluated]
        all_metric = ref._metric(quad.reshape(-1, 3))
        all_h[evaluated] = (all_metric["bcov"] / all_metric["B"][:, None]).reshape(quad.shape)
    for row in range(Fcount):
        key = keys[row]
        axis = int(key[0])
        if axis == 0 and int(key[1]) == 0:
            continue  # collapsed r0: never an owner-boundary face, but skip defensively
        points = all_points[row]; weights = all_weights[row]
        common = batched_face_common_gradient(t, S, owner_values, key, points,
                                               normal_coefficients=normal_coefficients, ctx=ctx,
                                               patch_cache=patch_cache, dirichlet_trace_fn=dirichlet_trace_fn,
                                               normal_data_fn=normal_data_fn, bc_cache=bc_cache)
        side = batched_side_values(t, S, owner_values, key, points, normal_coefficients=normal_coefficients,
                                    ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=dirichlet_trace_fn,
                                    bc_cache=bc_cache,
                                    normal_data_fn=normal_data_fn)
        if common["boundary"]:
            condition_max = max(condition_max, float(np.max(common["condition"])))
            residual_max = max(residual_max, float(np.max(common["residual"])))
        grad_roles = _select_role_matrix(common, ROLE_NAMES, ROLE_BC, ROLE_PHYSICAL_INDEX, is_gradient=True)
        lower_roles = _select_role_matrix(side["lower"], ROLE_NAMES, ROLE_BC, ROLE_PHYSICAL_INDEX, is_gradient=False)
        upper_roles = _select_role_matrix(side["upper"], ROLE_NAMES, ROLE_BC, ROLE_PHYSICAL_INDEX, is_gradient=False)

        h = all_h[row]
        jump = np.asarray(p05_scalar_face_jump(grad_roles[None], lower_roles[None], upper_roles[None],
                                                h[None], weights[None], np.array([axis]),
                                                np.array(ACTION_PAIR_INDEX)))[0]
        N[row] = jump[:P]
        D[row] = jump[P:]
    return dict(ids=face_ids, endpoints=endpoints, N=N, D=D,
                condition_max=condition_max, residual_max=residual_max)


def _face_kind(n, key):
    """Mirrors ``operator._face_kind``: the three zero-jump-checked families."""
    axis, i = int(key[0]), int(key[1])
    if axis == 0:
        if i == n:
            return "physical_wall"
        if i == n - 1:
            return "radial_n_minus_1"
        return None
    if i >= n - 2:
        return "transverse_last_two_layers"
    return None


def encode_face_id(n, key):
    """Face-id in the same periodic scheme as ``p07_combined_global.topology.census``/``decode``."""
    axis, i, j, k = (int(v) for v in key)
    if axis == 0:
        return i * n * n + j * n + k
    if axis == 1:
        return (n + 1) * n * n + i * n * n + j * n + k
    return (n + 1) * n * n + n ** 3 + i * n * n + j * n + k


def global_faces(n, input_root, output_dir):
    """Every owner-boundary face, globally, via the frozen p07n topology census."""
    pt.census(n, str(input_root), str(output_dir))
    with np.load(str(output_dir) + f"/N{n}.topology.npz") as z:
        return z["face_ids"].copy(), z["endpoints"].copy()
