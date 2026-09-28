"""P06N batched chunk kernels: the fast path validated against the per-owner
oracle in ``p06n_field_derived_global.operator`` (``owner_q1``/``owner_q3``,
``check_omega_independence``) by the campaign's ``verify-equivalence`` command.

Reuses, unchanged, by import:

* the accepted P06 curvature algebra from ``p06_structured_global.numerics``
  (``_continuum_terms``, ``_face_geometry``, ``_principal_matrix``,
  ``_absolute_action``, ``_curvature_bc_characteristic_wall_states``,
  ``_cell_quadrature``, ``base._face_keys``/``_face_count``/``_face_incidence``,
  ``_periodic_duplicate_face``, ``TAU``/``FLOOR``) -- never modified;
* the catalogue-agnostic batched row primitives from
  ``p05n_field_derived_global.core`` (``batched_cell_values``,
  ``batched_side_values``, ``StructuredReconstruction``, ``load_context``,
  ``WallLattice``) and this package's own ``rows.batched_face_common_value``;
* this package's own frozen oracle, ``operator.py`` (``Case``, ``Role``,
  ``make_role``, ``owner_all_faces`` for the deduplicated/accepted census
  choice, ``live_observations``).

Design, mirroring the batching strategy ``p05n_field_derived_global.core``
uses (see its module docstring) and the task spec:

* Row geometry (``S.rows``/``prepare_neumann_point_rows``) is built once per
  raw-cell chunk (q1) or once per face (q3, batched over that face's
  quadrature points) -- independent of which of the 14 case/state-variants
  (7 catalogue cases, each with its matched-Dirichlet diagnostic D) it is
  applied to. Every catalogue case then only *selects*, per physical-field
  column, which BC variant (dirichlet/neumann) its role uses.
* ``ref._metric``/``ref._curvature`` (via ``curvature_geometry``/
  ``_face_geometry``) is called once per chunk, not once per (case, target).

Import as ``from p06n_field_derived_global import core`` with
``DRBX/scripts`` (not this package's own directory) on ``sys.path`` (this
package's ``operator.py`` shadows the stdlib ``operator`` module -- see its
docstring).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from p07_combined_global.kernels import num
from p07_combined_global import topology as pt
from perpendicular_structured.reconstruction import StructuredReconstruction, load_context
from perpendicular_structured.reference_geometry import curvature_geometry
from p07_diffusion_global import numerics as refnum
from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext

from p05n_field_derived_global import core as p05n_core

import p06_structured_global.numerics as p06numerics

from p06n_field_derived_global import fields as p06n_fields
from p06n_field_derived_global import rows as p06n_rows
from p06n_field_derived_global import operator as p06n_operator
from p06n_field_derived_global.operator import (  # noqa: F401
    Case, Role, make_role, owner_q1, owner_q3, owner_all_faces, owner_values_matrix,
    live_observations, STATE_NAMES, TAU, FLOOR,
)

HERE = Path(__file__).resolve().parent

STATE_SLOTS = STATE_NAMES  # ("n", "Te", "Ti", "omega", "phi")
EQUATIONS = ("density", "Te", "Ti", "omega")   # material/remainder/total 4-vector order
GATED_EQUATIONS = ("density", "Te", "Ti")      # phi never gets its own equation; omega reported, not gated
TERMS = ("material", "remainder", "total")
CANDIDATES = ("centered", "U")


# ---------------------------------------------------------------------------
# Frozen catalogue parsing.
# ---------------------------------------------------------------------------
def _parse_field_spec(spec: str):
    """``'name'`` -> (name, 'neumann'); ``'name:dirichlet'`` -> (name, 'dirichlet')."""
    if ":" in spec:
        name, bc = spec.split(":", 1)
        if bc != "dirichlet":
            raise ValueError(f"unknown field-spec suffix {spec!r}")
        return name, "dirichlet"
    return spec, "neumann"


def load_catalogue(path=None):
    path = Path(path) if path is not None else HERE / "p06n_catalogue.json"
    doc = json.loads(path.read_text())
    if tuple(doc["state_order"]) != STATE_SLOTS:
        raise ValueError("catalogue state order changed")
    cases = {}
    for name, slots in doc["cases"].items():
        cases[name] = tuple(_parse_field_spec(slots[slot]) for slot in STATE_SLOTS)
    return doc, cases


CATALOGUE_DOC, CASES = load_catalogue()
CASE_NAMES = tuple(sorted(CASES))  # fixed alphabetical order
NONCONSTANT_CASES = tuple(c for c in CASE_NAMES if not c.startswith("control_"))
CONTROL_CASES = tuple(c for c in CASE_NAMES if c.startswith("control_"))
MAIN_CASES = tuple(c for c in CASE_NAMES if c.startswith("main_"))
HELDOUT_CASES = tuple(c for c in CASE_NAMES if c.startswith("heldout_"))
DIRICHLET_RICH_CASES = tuple(c for c in CASE_NAMES if c == "dirichlet_rich")
GATED_CASES = MAIN_CASES + HELDOUT_CASES + DIRICHLET_RICH_CASES  # every nonconstant case, per the frozen catalogue gate


def matched_dirichlet_spec(spec):
    """Every slot's own field, at Dirichlet BC -- the D diagnostic (see module docstring)."""
    return tuple((field, "dirichlet") for field, _bc in spec)


# Variant table: one entry per (case, is_D) pair, fixed order. Variant name
# 'name' is the candidate N case; 'name:D' is its matched-Dirichlet diagnostic.
VARIANT_NAMES = tuple(CASE_NAMES) + tuple(f"{c}:D" for c in CASE_NAMES)
VARIANT_SPEC = {name: CASES[name] for name in CASE_NAMES}
VARIANT_SPEC.update({f"{c}:D": matched_dirichlet_spec(CASES[c]) for c in CASE_NAMES})

# Physical fields referenced anywhere in the catalogue (N or D variant), fixed order.
NAMES = tuple(sorted({field for spec in VARIANT_SPEC.values() for field, _bc in spec}))
NAME_INDEX = {name: i for i, name in enumerate(NAMES)}
VARIANT_FIELD_INDEX = {name: np.array([NAME_INDEX[f] for f, _bc in spec], dtype=np.int64)
                        for name, spec in VARIANT_SPEC.items()}
VARIANT_BC = {name: tuple(bc for _f, bc in spec) for name, spec in VARIANT_SPEC.items()}
VARIANT_IS_NEUMANN = {name: np.array([bc == "neumann" for bc in bcs], dtype=bool)
                       for name, bcs in VARIANT_BC.items()}


class FieldTables:
    """The field axis and variant selection the batched kernels read.

    The default is the frozen catalogue. The accepted-P06 replay gate passes
    tables built from the accepted P06 states instead, so the replay runs
    through exactly the batched kernels the campaign uses.
    ``evaluate(ref, q, name, period)`` returns (value, gradient, Hessian).
    """

    def __init__(self, variant_spec, evaluate):
        self.variant_spec = dict(variant_spec)
        self.variant_names = tuple(self.variant_spec)
        self.names = tuple(sorted({f for spec in self.variant_spec.values() for f, _bc in spec}))
        index = {name: i for i, name in enumerate(self.names)}
        self.field_index = {k: np.array([index[f] for f, _bc in spec], dtype=np.int64)
                            for k, spec in self.variant_spec.items()}
        self.is_neumann = {k: np.array([bc == "neumann" for _f, bc in spec], dtype=bool)
                           for k, spec in self.variant_spec.items()}
        self.evaluate = evaluate


CATALOGUE_TABLES = FieldTables(VARIANT_SPEC, p06n_fields.evaluate)
assert CATALOGUE_TABLES.names == NAMES and CATALOGUE_TABLES.variant_names == VARIANT_NAMES


# ---------------------------------------------------------------------------
# Grid / context loading (mirrors p05n_field_derived_global.core exactly).
# ---------------------------------------------------------------------------
def context(t):
    return PointRowContext.from_arrays(faces=t.faces, centers=t.centers, raw_to_owner=t.ro,
                                        raw_volume=t.rv, owner_volume=t.vol,
                                        owner_centroid_xy=t.g.owner_centroid_xy, eta_period=t.g.eta_period,
                                        dr=t.g.dr, dtheta=t.g.dtheta, deta=t.g.deta)


def load(input_root, sidecar, n):
    t = load_context(n, input_root)
    ref = refnum.reference(sidecar, verify_hashes=False)
    return t, ref


def dirichlet_trace_all(ref, q, period, tables=None):
    """(v (Q,F), g (Q,3,F)) stacked over the table's field axis."""
    tables = CATALOGUE_TABLES if tables is None else tables
    q = np.asarray(q, dtype=np.float64)
    Q = len(q)
    v = np.empty((Q, len(tables.names)))
    g = np.empty((Q, 3, len(tables.names)))
    for j, name in enumerate(tables.names):
        vv, gg, _ = tables.evaluate(ref, q, name, period)
        v[:, j] = vv
        g[:, :, j] = gg
    return v, g


def normal_data_all(ref, q, period, tables=None):
    """(Q,F) g_N stacked over NAMES (one physical-normal call, not one per field;
    see p05n_field_derived_global.core.normal_data_all's docstring for why this matters)."""
    tables = CATALOGUE_TABLES if tables is None else tables
    q = np.asarray(q, dtype=np.float64)
    a = p06n_fields.normal(ref, q)
    out = np.empty((len(q), len(tables.names)))
    for j, name in enumerate(tables.names):
        _, g, _ = tables.evaluate(ref, q, name, period)
        out[:, j] = np.einsum("qa,qa->q", a, g)
    return out


class WallLattice:
    """Physical normal and field-derived g_N on the fixed Neumann wall lattice,
    for the full P06N catalogue field set (including this package's own
    ``rich_c``/``heldout_rich_h``, which ``p05n_field_derived_global.core.WallLattice``
    cannot serve: its ``names=`` branch calls ``p05n_fields.evaluate`` directly,
    which does not know these fields). Mirrors that class's once-per-grid
    metric call exactly, via ``p06n_fields`` (a strict superset of p05n's
    fields plus the shifted variants and the two rich P06N additions), so the
    behavior for every field common to both is identical.
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
        self.a = p06n_fields.normal(ref, self.points)
        gradients = [p06n_fields.evaluate(ref, self.points, name, period)[1] for name in self.names]
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


def observations_all(t, ref, period):
    """Owner-averaged live observations, one column per NAMES field."""
    values = np.column_stack([p06n_fields.evaluate(ref, t.pts, name, period)[0] for name in NAMES])
    return live_observations(t, values)


def _select(dirichlet_arr: np.ndarray, neumann_arr: np.ndarray, field_index: np.ndarray, is_neumann: np.ndarray,
            axis: int) -> np.ndarray:
    """Gather a (..., 5) array from full-NAMES (..., F) dirichlet/neumann arrays,
    picking each of a variant's 5 slots' own field column and BC."""
    d = np.take(dirichlet_arr, field_index, axis=axis)
    n = np.take(neumann_arr, field_index, axis=axis)
    shape = [1] * d.ndim
    shape[axis] = len(field_index)
    sel = is_neumann.reshape(shape)
    return np.where(sel, n, d)


def _cell_geometry_shim(t):
    return SimpleNamespace(x_faces=t.faces[0], y_faces=t.faces[1], z_faces=t.faces[2])


# ---------------------------------------------------------------------------
# q1 (raw / material+remainder) chunk kernel.
# ---------------------------------------------------------------------------
def raw_chunk(t, S, ref, ctx, raw_ids, owner_values, normal_coefficients, patch_cache, period,
              normal_data_override=None, variants=None, tables=None):
    """Every variant's candidate (material, remainder, total) and exact R at
    every raw midpoint in the chunk, plus shared diagnostics.

    Row/patch construction (``batched_cell_values``) happens exactly once for
    this chunk, over the full ``NAMES`` field axis and both BC variants;
    each of ``variants`` then only selects its own 5 columns/BCs and calls
    the accepted ``_continuum_terms`` (candidate, then exact).
    """
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    points = t.pts[raw_ids]
    keys = np.array(np.unravel_index(raw_ids, (t.n,) * 3)).T
    owner_ids = t.ro[raw_ids]

    tables = CATALOGUE_TABLES if tables is None else tables
    variants = tables.variant_names if variants is None else tuple(variants)

    def dirichlet_trace_fn(q):
        return dirichlet_trace_all(ref, q, period, tables)

    def normal_data_fn(q):
        if normal_data_override is not None:
            return normal_data_override(q)
        return normal_data_all(ref, q, period, tables)

    bc_cache = {}
    cellvals = p05n_core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                              ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=dirichlet_trace_fn,
                                              normal_data_fn=normal_data_fn, radial_degree=3, bc_cache=bc_cache)
    value_d, grad_d = cellvals["dirichlet"]
    value_n, grad_n = cellvals["neumann"]

    shim = _cell_geometry_shim(t)
    cell_points, cell_weights = p06numerics._cell_quadrature(shim, keys, 1)
    if not np.allclose(cell_points.reshape(-1, 3), points, atol=0.0, rtol=0.0):
        raise ValueError("q1 raw-midpoint mismatch between StructuredReconstruction and p06 cell quadrature")
    weights = cell_weights.reshape(-1)

    prepared = curvature_geometry(ref, points)
    jacobian = np.asarray(prepared.J)
    bmag = np.asarray(prepared.B)
    evolution_weight = weights * jacobian / np.maximum(bmag, 1.0e-30)

    Q = len(points)
    V = len(variants)
    material = np.empty((V, Q, 4)); remainder = np.empty((V, Q, 4)); total = np.empty((V, Q, 4))
    r_material = np.empty((V, Q, 4)); r_remainder = np.empty((V, Q, 4)); r_total = np.empty((V, Q, 4))
    exact_input_defect = np.zeros((V, Q, 4))
    closure_max = 0.0
    exact_cache = {}

    for vi, name in enumerate(variants):
        field_index = tables.field_index[name]
        is_neumann = tables.is_neumann[name]
        value = _select(value_d, value_n, field_index, is_neumann, axis=1)      # (Q,5)
        gradient = _select(grad_d, grad_n, field_index, is_neumann, axis=2)     # (Q,3,5)
        candidate_values = value.T
        candidate_gradients = gradient.transpose(2, 0, 1)
        candidate = p06numerics._continuum_terms(candidate_values, candidate_gradients, prepared)
        material[vi], remainder[vi], total[vi] = candidate[0], candidate[1], candidate[2]
        closure_max = max(closure_max, float(np.max(np.abs(candidate[0] + candidate[1] - candidate[2]))))

        spec = tables.variant_spec[name]
        key = tuple(f for f, _bc in spec)
        cached = exact_cache.get(key)
        if cached is None:
            exact_values = np.empty_like(candidate_values)
            exact_gradients = np.empty_like(candidate_gradients)
            for j, (field, _bc) in enumerate(spec):
                ev, eg, _ = tables.evaluate(ref, points, field, period)
                exact_values[j] = ev
                exact_gradients[j] = eg
            exact = p06numerics._continuum_terms(exact_values, exact_gradients, prepared)
            exact_replay = p06numerics._continuum_terms(exact_values, exact_gradients, prepared)
            cached = (exact, np.abs(exact_replay[2] - exact[2]))
            exact_cache[key] = cached
        exact, defect = cached
        r_material[vi], r_remainder[vi], r_total[vi] = exact[0], exact[1], exact[2]
        exact_input_defect[vi] = defect

    has_neumann = any(np.any(tables.is_neumann[name]) for name in variants)
    condition_max = float(np.max(cellvals["condition"])) if has_neumann and len(cellvals["condition"]) else 0.0
    residual_max = float(np.max(cellvals["residual"])) if has_neumann and len(cellvals["residual"]) else 0.0

    return dict(ids=raw_ids, owner_ids=owner_ids, evolution_weight=evolution_weight,
                material=material, remainder=remainder, total=total,
                R_material=r_material, R_remainder=r_remainder, R_total=r_total,
                exact_input_defect=exact_input_defect, closure_max=closure_max,
                condition_max=condition_max, residual_max=residual_max, variants=list(variants))


def observation_chunk(t, ref, raw_ids, period):
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    q = t.pts[raw_ids]
    values = np.column_stack([p06n_fields.evaluate(ref, q, name, period)[0] for name in NAMES])
    return dict(ids=raw_ids, owner_ids=t.ro[raw_ids], numerator=t.rv[raw_ids, None] * values)


# ---------------------------------------------------------------------------
# q3 (face / characteristic correction) chunk kernel.
# ---------------------------------------------------------------------------
def face_chunk(t, S, ref, ctx, keys, owner_values, normal_coefficients, patch_cache, period,
               order=3, normal_data_override=None, variants=None, dedupe=True, tables=None):
    """Every variant's characteristic material-correction contribution
    (lower, upper) for a batch of faces, in the accepted P06 face-index
    space, plus their owner endpoints.

    ``keys`` is an (F,4) array of (axis, i, j, k) accepted-scheme face keys
    (see ``p06_structured_global.numerics.base._face_keys`` /
    ``owner_all_faces``'s docstring). Collapsed r=0 faces (axis 0, i == 0)
    must already be excluded by the caller (they are never an owner-boundary
    face; see ``p05n_field_derived_global.core.face_chunk``'s identical
    convention). ``dedupe`` drops the periodic-duplicate face (theta/eta slot
    ``n``) from the owner scatter (``lower_valid``/``upper_valid``), matching
    the campaign's own (deduplicated) topology census; ``dedupe=False``
    reproduces the accepted campaign's saved artifact (the gate-(a) replay).
    """
    keys = np.asarray(keys, dtype=np.int64)
    n = t.n
    F = len(keys)
    if F and np.any((keys[:, 0] == 0) & (keys[:, 1] == 0)):
        raise ValueError("face_chunk must not receive the collapsed r=0 face")

    tables = CATALOGUE_TABLES if tables is None else tables
    variants = tables.variant_names if variants is None else tuple(variants)

    def dirichlet_trace_fn(q):
        return dirichlet_trace_all(ref, q, period, tables)

    def normal_data_fn(q):
        if normal_data_override is not None:
            return normal_data_override(q)
        return normal_data_all(ref, q, period, tables)

    bc_cache = {}
    V = len(variants)
    correction_lo = np.zeros((V, F, 4))
    correction_hi = np.zeros((V, F, 4))
    zero_face_max = {}
    wall_exterior_defect_max = 0.0
    wall_faces_seen = 0
    condition_max = 0.0
    residual_max = 0.0

    face_points = []
    face_weights = []
    for row in range(F):
        p, w = num.quadrature(t.faces, keys[row:row + 1], order, face=True)
        face_points.append(p[0]); face_weights.append(w[0])
    if F:
        all_points = np.concatenate(face_points, axis=0)
        all_J, all_B, all_K = p06numerics._face_geometry(ref, all_points)
    offset = 0

    for row in range(F):
        key = keys[row]
        axis = int(key[0])
        points, weights = face_points[row], face_weights[row]
        Q = len(points)
        J, B, K = all_J[offset:offset + Q], all_B[offset:offset + Q], all_K[offset:offset + Q]
        offset += Q

        central = p06n_rows.batched_face_common_value(t, S, owner_values, key, points,
                                                        normal_coefficients=normal_coefficients, ctx=ctx,
                                                        patch_cache=patch_cache, dirichlet_trace_fn=dirichlet_trace_fn,
                                                        normal_data_fn=normal_data_fn, bc_cache=bc_cache)
        sides = p05n_core.batched_side_values(t, S, owner_values, key, points,
                                               normal_coefficients=normal_coefficients, ctx=ctx,
                                               patch_cache=patch_cache, dirichlet_trace_fn=dirichlet_trace_fn,
                                               normal_data_fn=normal_data_fn, bc_cache=bc_cache)
        if central.get("boundary"):
            condition_max = max(condition_max, float(np.max(central["condition"])))
            residual_max = max(residual_max, float(np.max(central["residual"])))

        normal_vec = J * K[:, axis] / np.maximum(B * B, 1.0e-30)
        is_wall = axis == 0 and int(key[1]) == n
        kind = p06n_operator._face_kind(n, key)

        for vi, name in enumerate(variants):
            field_index = tables.field_index[name]
            is_neumann = tables.is_neumann[name]
            central_v = _select(central["dirichlet"], central["neumann"], field_index, is_neumann, axis=1)
            lower_v = _select(sides["lower"]["dirichlet"], sides["lower"]["neumann"], field_index, is_neumann, axis=1)
            upper_v = _select(sides["upper"]["dirichlet"], sides["upper"]["neumann"], field_index, is_neumann, axis=1)

            state_central = central_v[:, :4].copy()
            state_left = lower_v[:, :4].copy()
            state_right = upper_v[:, :4].copy()

            if is_wall:
                import jax.numpy as jnp
                interior = state_central
                exterior, working, _fb = p06numerics._curvature_bc_characteristic_wall_states(
                    jnp.asarray(interior), jnp.asarray(interior), jnp.asarray(B), TAU,
                    jnp.asarray(normal_vec), interior_on_right=False, positivity_floor=FLOOR,
                )
                state_left = interior
                state_right = np.asarray(exterior)
                state_central = np.asarray(working)
                wall_exterior_defect_max = max(wall_exterior_defect_max, float(np.max(np.abs(state_right - interior))))
                if vi == 0:
                    wall_faces_seen += 1

            matrix = p06numerics._principal_matrix(state_central, B)
            flux_matrix = -normal_vec[..., None, None] * matrix
            jump = state_right - state_left
            absolute, _fb = p06numerics._absolute_action(flux_matrix, jump)
            material = np.einsum("qij,qj->qi", flux_matrix, jump)
            dplus = 0.5 * (material + absolute)
            dminus = 0.5 * (material - absolute)
            correction_lo[vi, row] = -np.sum(weights[:, None] * dminus, axis=0)
            correction_hi[vi, row] = -np.sum(weights[:, None] * dplus, axis=0)
            if kind is not None:
                zero_face_max[kind] = max(zero_face_max.get(kind, 0.0), float(np.max(np.abs(state_right - state_left))))

    lower_raw, lower_valid, upper_raw, upper_valid = p06numerics._face_incidence(n, keys) if F else (
        np.empty(0, np.int64), np.empty(0, bool), np.empty(0, np.int64), np.empty(0, bool))
    if dedupe and F:
        dup = p06numerics._periodic_duplicate_face(n, keys)
        lower_valid = lower_valid & ~dup
        upper_valid = upper_valid & ~dup
    ro_flat = t.ro.ravel()
    lo_owner = np.where(lower_valid, ro_flat[np.clip(lower_raw, 0, len(ro_flat) - 1)], -1)
    hi_owner = np.where(upper_valid, ro_flat[np.clip(upper_raw, 0, len(ro_flat) - 1)], -1)

    return dict(keys=keys, lo=lo_owner, hi=hi_owner, correction_lo=correction_lo, correction_hi=correction_hi,
                zero_face_max=zero_face_max, wall_exterior_defect_max=wall_exterior_defect_max,
                wall_faces_seen=wall_faces_seen, condition_max=condition_max, residual_max=residual_max,
                variants=list(variants))


# ---------------------------------------------------------------------------
# Accepted-scheme face id space / global census helpers.
# ---------------------------------------------------------------------------
def all_face_ids(n):
    """Every accepted-scheme face id, excluding the collapsed r=0 face."""
    ids = np.arange(p06numerics._face_count(n), dtype=np.int64)
    keys = p06numerics.base._face_keys(n, ids)
    keep = ~((keys[:, 0] == 0) & (keys[:, 1] == 0))
    return ids[keep]


def face_keys_for_ids(n, ids):
    return p06numerics.base._face_keys(n, np.asarray(ids, dtype=np.int64))


def owner_face_union(t, owners, dedupe=False):
    """Every (key, lo, hi) face incident to any of ``owners``, deduplicated
    across owners (a face shared by two selected owners is listed once)."""
    seen = {}
    for owner in owners:
        for key, lo, hi in owner_all_faces(t, owner, dedupe=dedupe):
            seen[key] = (lo, hi)
    keys = np.array(sorted(seen), dtype=np.int64) if seen else np.empty((0, 4), dtype=np.int64)
    lo = np.array([seen[tuple(k)][0] if seen[tuple(k)][0] is not None else -1 for k in keys], dtype=np.int64)
    hi = np.array([seen[tuple(k)][1] if seen[tuple(k)][1] is not None else -1 for k in keys], dtype=np.int64)
    return keys, lo, hi


def _face_kind(n, key):
    return p06n_operator._face_kind(n, key)


def regional_masks(t):
    """Owner region masks, mirroring p05n_field_derived_global.campaign.regional_masks."""
    n = t.n
    count = np.bincount(t.ro, minlength=len(t.vol))
    rawcount = count[t.ro].reshape((n, n, n))
    trans = np.zeros((n, n, n), bool)
    diff = rawcount[1:] != rawcount[:-1]
    trans[1:] |= diff; trans[:-1] |= diff
    transition_owner = np.bincount(t.ro, weights=trans.reshape(-1), minlength=len(t.vol)) > 0
    radius = np.zeros((n, n, n), np.int64); radius[:] = np.arange(n)[:, None, None]

    def owner_has(mask3d):
        return np.bincount(t.ro, weights=mask3d.reshape(-1), minlength=len(t.vol)) > 0

    axis_core = owner_has(radius == 0)
    wall = owner_has(radius == n - 1)
    last_two_layers = owner_has(radius >= n - 2)
    transition = transition_owner & ~axis_core & ~wall
    aggregate = (count > 1) & ~axis_core & ~wall & ~transition
    ordinary = ~(axis_core | wall | transition | aggregate)
    return {"physical_wall": wall, "transverse_last_two_layers": last_two_layers, "transition": transition,
            "aggregate": aggregate, "ordinary": ordinary, "interior": ~(axis_core | wall | last_two_layers)}
