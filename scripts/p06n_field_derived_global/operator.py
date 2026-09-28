"""P06N owner action: q1 midpoint volume term plus q3 characteristic face term.

Reuses the accepted P06 curvature algebra unchanged, by import, from
``p06_structured_global.numerics`` (never modified, never re-derived here):
``_continuum_terms`` (the q1 material/remainder/total expression, built on
``curvature_principal_matrix`` from ``drbx.native.fci_curvature_production_flux``,
which has no omega column -- see the design doc and ``check_omega_independence``
below), ``_face_geometry`` (K/B/J at face points, including its one-sided
finite-difference curl at the exact physical wall -- kept unchanged per the
task spec), ``_principal_matrix``/``_absolute_action`` (the q3 flux Jacobian
and its characteristic absolute-value action), ``TAU``/``FLOOR``, and
``_curvature_bc_characteristic_wall_states`` (the wall solve, called with
``boundary_trace = interior`` exactly as accepted -- see
``owner_q3``: the adopted recovered-trace contract, design.md section 3).

Row construction is field-catalogue-agnostic and reused, by import, from
``p05n_field_derived_global.core`` (``batched_cell_values``,
``batched_side_values``) and this package's own ``rows.batched_face_common_value``
(the one primitive P05N never needed -- see ``rows.py``'s module docstring).
Both BC variants (Dirichlet-lift and Neumann-eliminated) are built once per
target from one shared row/patch, for every physical-field column at once
(the owner-values matrix carries one column per field); each catalogue
pairing then only *selects*, per column, which variant its role uses --
no repeated ``ref._metric``/``ref._curvature``/row construction per field or
per pairing.

Import as ``from p06n_field_derived_global import operator`` with
``DRBX/scripts`` (not this package's own directory) on ``sys.path``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from p07_combined_global.kernels import num
from p05n_field_derived_global import core as p05n_core
from p05n_field_derived_global.operator import Role, live_observations, owner_raw, owner_boundary_faces  # noqa: F401

from p06n_field_derived_global import rows as p06n_rows

import p06_structured_global.numerics as p06numerics

TAU = p06numerics.TAU
FLOOR = p06numerics.FLOOR

# The 5-column state order P06's own algebra fixes throughout: (n, Te, Ti, omega, phi).
STATE_NAMES = ("n", "Te", "Ti", "omega", "phi")


@dataclass
class Case:
    """One P06N evaluation case: a Role (name/bc/evaluate/normal_data) per state slot.

    ``omega`` never affects the action (see ``check_omega_independence``); it
    still needs *some* smooth Role for row bookkeeping (constant-shape value
    and gradient), per the design doc ("set it to any smooth field").
    """
    n: Role
    Te: Role
    Ti: Role
    omega: Role
    phi: Role

    def roles(self) -> Tuple[Role, ...]:
        return (self.n, self.Te, self.Ti, self.omega, self.phi)

    def names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self.roles())

    def bcs(self) -> Tuple[str, ...]:
        return tuple(r.bc for r in self.roles())


def make_role(field_module, ref, period, field_name: str, bc: str, normal_data=None) -> Role:
    """Build a Role. ``normal_data``, when given, overrides the default
    per-call ``field_module.normal_data(ref, q, field_name, period)`` (which
    calls ``ref._metric`` fresh every time) -- pass a
    ``p05n_field_derived_global.core.WallLattice``-backed lookup instead
    (see run_grid.py's ``build_pairings``): every Neumann wall node is a
    fixed lattice point, so the metric/normal-derivative data can be
    evaluated once per grid and looked up, not recomputed per patch.
    """
    def evaluate(q, _name=field_name):
        v, g, _ = field_module.evaluate(ref, q, _name, period)
        return v, g
    if bc == "neumann" and normal_data is None:
        normal_data = lambda q, _name=field_name: field_module.normal_data(ref, q, _name, period)  # noqa: E731
    return Role(name=field_name, bc=bc, evaluate=evaluate, normal_data=normal_data if bc == "neumann" else None)


def _dirichlet_trace_fn(roles: Sequence[Role]):
    """(v (Q,F), g (Q,3,F)) -- every role's own ``evaluate``, used as its Dirichlet trace."""
    def trace(q):
        q = np.asarray(q, dtype=np.float64)
        Qn = len(q)
        v = np.empty((Qn, len(roles)))
        g = np.empty((Qn, 3, len(roles)))
        for j, role in enumerate(roles):
            vv, gg = role.evaluate(q)
            v[:, j] = vv
            g[:, :, j] = gg
        return v, g
    return trace


def _normal_data_fn(roles: Sequence[Role]):
    """(Q,F) g_N -- only meaningful for Neumann roles; harmless (unused) elsewhere."""
    def normal(q):
        q = np.asarray(q, dtype=np.float64)
        out = np.zeros((len(q), len(roles)))
        for j, role in enumerate(roles):
            if role.bc == "neumann":
                out[:, j] = role.normal_data(q)
        return out
    return normal


def owner_all_faces(t, owner, *, dedupe=False):
    """Every face incident to any raw member of ``owner``, with the exact face
    census and multiplicity ``p06_structured_global.numerics._preflight`` uses
    -- not ``p05n_field_derived_global.operator.owner_boundary_faces``'s
    convention, which differs in two ways found by the check-1 replay:

    ``dedupe=True`` drops the periodic-duplicate face key itself (theta face
    slot ``n`` / eta face slot ``n``, i.e. ``key[2] == n`` for axis 1 or
    ``key[3] == n`` for axis 2 -- exactly ``p06_structured_global.numerics``'s
    current, already-fixed ``_periodic_duplicate_face`` predicate), for the
    deduplicated topology census the campaign's own reduction uses. The
    default (``False``) reproduces the accepted campaign's saved artifact
    (counts the seam correction twice), for the gate-(a) replay only.

    1. It drops internal aggregation seams (``lo == hi``). Accepted P06 does
       not: it adds ``correction[...,side]`` to ``raw_owner[that side]``
       whenever it matches the target owner, on *both* sides independently,
       so an internal face contributes
       ``correction[...,0] + correction[...,1] == -sum(w*material)`` to that
       one owner. (Excluding these, as ``owner_boundary_faces`` does,
       undercounted every agglomerated owner -- worst for heavily-aggregated
       axis owners with many same-owner theta/eta neighbors, e.g. N32 owner
       0, an axis-ring owner spanning all 32 theta cells at one eta index.)
    2. Accepted P06's face id scheme (``p07_diffusion_global.numerics.face_indices``)
       allocates ``n+1`` theta/eta face slots (0..n) per ring WITHOUT
       wrapping the ``own index + 1`` upper face modulo ``n``:
       ``_preflight`` builds its face set from every selected raw cell's own
       index and (unwrapped) own-index-plus-one, via
       ``p06_structured_global.numerics.base._face_index``. For a periodic
       ring this makes face slot ``n`` a *distinct id* from slot 0, even
       though ``_face_incidence`` (``(m-1)%n``, ``m%n``) assigns it the
       *identical* (lo, hi) owner pair as slot 0 -- so the one physical
       wraparound face between the ring's last and first cell is visited
       (and its correction counted) through two distinct ids, not
       deduplicated. Reproduced verbatim here (only the *key*, used for
       dict-dedup, is left unwrapped for axes 1/2; the ``grid`` lookup for
       lo/hi still uses ``%n``, exactly as ``_face_incidence`` does), so this
       traversal's face count and per-owner sum match accepted's exactly.
    """
    n = t.n
    grid = t.ro.reshape(n, n, n)
    raw_ids = np.flatnonzero(t.ro == int(owner))
    ijk = np.array(np.unravel_index(raw_ids, (n, n, n))).T
    seen = {}
    for i, j, k in ijk:
        i, j, k = int(i), int(j), int(k)
        for i_face in (i, i + 1):
            if i_face == 0:
                continue
            lo = int(grid[i_face - 1, j, k]) if i_face - 1 >= 0 else None
            hi = int(grid[i_face, j, k]) if i_face < n else None
            seen[(0, i_face, j, k)] = (lo, hi)
        for j_face in (j, j + 1):
            lo = int(grid[i, (j_face - 1) % n, k])
            hi = int(grid[i, j_face % n, k])
            seen[(1, i, j_face, k)] = (lo, hi)
        for k_face in (k, k + 1):
            lo = int(grid[i, j, (k_face - 1) % n])
            hi = int(grid[i, j, k_face % n])
            seen[(2, i, j, k_face)] = (lo, hi)
    if dedupe:
        seen = {key: lohi for key, lohi in seen.items()
                if not ((key[0] == 1 and key[2] == n) or (key[0] == 2 and key[3] == n))}
    return [(key, lo, hi) for key, (lo, hi) in seen.items()]


def owner_values_matrix(t, ref, period, roles: Sequence[Role]):
    """Owner-averaged raw-volume projection, one column per role's physical field."""
    values = np.column_stack([role.evaluate(t.pts)[0] for role in roles])
    return live_observations(t, values)


def _select(dirichlet_arr: np.ndarray, neumann_arr: np.ndarray, bcs: Sequence[str], axis: int) -> np.ndarray:
    sel = np.array([bc == "neumann" for bc in bcs])
    shape = [1] * dirichlet_arr.ndim
    shape[axis] = len(bcs)
    sel = sel.reshape(shape)
    return np.where(sel, neumann_arr, dirichlet_arr)


def owner_q1(t, S, ref, owner, case: Case, owner_values: np.ndarray, *, normal_coefficients, context, patch_cache,
             donor_accumulator=None):
    """q1 volume term at raw midpoints, owner-reduced: (N, R, exact-input O==R defect, diagnostics)."""
    raw_ids, points, keys = owner_raw(t, owner)
    roles = case.roles()
    trace_fn = _dirichlet_trace_fn(roles)
    normal_fn = _normal_data_fn(roles)
    result = p05n_core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                            ctx=context, patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                            normal_data_fn=normal_fn, radial_degree=3,
                                            donor_accumulator=donor_accumulator)
    vd, gd = result["dirichlet"]
    vn, gn = result["neumann"]
    bcs = case.bcs()
    value = _select(vd, vn, bcs, axis=1)          # (Q,5)
    gradient = _select(gd, gn, bcs, axis=2)       # (Q,3,5)

    from perpendicular_structured.reference_geometry import curvature_geometry
    prepared = curvature_geometry(ref, points)

    # q1 weight: the accepted P06 raw-cell coordinate-quadrature weight
    # (Delta_u*Delta_theta*Delta_eta at cell_order=1, from the SAME
    # p06_structured_global.numerics._cell_quadrature the accepted campaign
    # uses) times J -- NOT ``t.rv`` (P05N/P07N's own, independently
    # computed, raw physical volume; verified numerically to differ from
    # weights*J by ~1-2% for wall-adjacent cells, which would silently fail
    # the tight Check-1 replay tolerance). A tiny shim exposes ``t.faces``
    # (identical face arrays either loader uses) under the attribute names
    # ``_cell_quadrature`` expects.
    from types import SimpleNamespace as _SimpleNamespace
    _shim = _SimpleNamespace(x_faces=t.faces[0], y_faces=t.faces[1], z_faces=t.faces[2])
    _cell_points, _cell_weights = p06numerics._cell_quadrature(_shim, keys, 1)
    if not np.allclose(_cell_points.reshape(-1, 3), points, atol=0.0, rtol=0.0):
        raise ValueError("q1 raw-midpoint mismatch between StructuredReconstruction and p06 cell quadrature")
    weights = _cell_weights.reshape(-1)

    candidate_values = value.T                       # (Q,5) -> (5,Q)
    candidate_gradients = gradient.transpose(2, 0, 1)  # (Q,3,5) -> (5,Q,3), matching _continuum_terms's (f,p,d)
    candidate = p06numerics._continuum_terms(candidate_values, candidate_gradients, prepared)

    exact_values = np.empty_like(candidate_values)
    exact_gradients = np.empty_like(candidate_gradients)
    for j, role in enumerate(roles):
        ev, eg = role.evaluate(points)
        exact_values[j] = ev
        exact_gradients[j] = eg
    exact = p06numerics._continuum_terms(exact_values, exact_gradients, prepared)

    jacobian = np.asarray(prepared.J)
    bmag = np.asarray(prepared.B)
    evolution_weight = weights * jacobian / np.maximum(bmag, 1.0e-30)
    weight_sum = float(np.sum(evolution_weight))

    def reduce(term):
        return np.sum(evolution_weight[:, None] * term, axis=0) / weight_sum

    N = {term: reduce(candidate[i]) for i, term in enumerate(("material", "remainder", "total"))}
    R = {term: reduce(exact[i]) for i, term in enumerate(("material", "remainder", "total"))}

    # Check 7 (exact-input O==R): a second, independent call to the same
    # accepted _continuum_terms with the identical exact (value, gradient),
    # mirroring work/p06n_exact_input_screen_20260927/report.md's own
    # methodology exactly (two independent calls, not a reused array) --
    # documented there as a structural-identity check, not distinguishing
    # numerical discretizations.
    exact_replay = p06numerics._continuum_terms(exact_values, exact_gradients, prepared)
    exact_input_defect = float(np.max(np.abs(exact_replay[2] - exact[2])))
    closure_max = float(np.max(np.abs(candidate[0] + candidate[1] - candidate[2])))

    # batched_cell_values already reports the Neumann row condition/residual
    # per point (0 for non-boundary points); it is a property of the shared
    # row/patch, not of which role happens to read it, so no per-role rebuild
    # is needed -- just gate on whether this case has any Neumann role at all.
    has_neumann = any(role.bc == "neumann" for role in roles)
    condition_max = float(np.max(result["condition"])) if has_neumann else 0.0
    residual_max = float(np.max(result["residual"])) if has_neumann else 0.0

    return dict(N=N, R=R, exact_input_defect=exact_input_defect, closure_max=closure_max,
                condition_max=condition_max, residual_max=residual_max, evolution_volume=weight_sum,
                raw_count=len(raw_ids), value=value, gradient=gradient)


_FACE_QUADRATURE_ORDER = 3


def _face_kind(n, key):
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


def owner_q3(t, S, ref, owner, case: Case, owner_values: np.ndarray, *, normal_coefficients, context, patch_cache,
             order=_FACE_QUADRATURE_ORDER, face_report=None, dedupe=False, donor_accumulator=None):
    """q3 characteristic face term, owner-reduced material correction (state order n,Te,Ti,omega only -- phi never
    enters the face term; see the module docstring / accepted numerics.py: ``state_central=central[state,:4]``).

    ``dedupe`` selects the face census (see ``owner_all_faces``): ``False``
    (default) reproduces the accepted campaign's saved artifact exactly
    (gate-(a) replay); ``True`` is the deduplicated topology census the
    campaign's own reduction uses.

    Returns (material_correction (4,), zero_face_max {kind: max|jump|}, wall_exterior_defect_max).
    """
    faces = owner_all_faces(t, owner, dedupe=dedupe)
    roles = case.roles()
    trace_fn = _dirichlet_trace_fn(roles)
    normal_fn = _normal_data_fn(roles)
    bcs = case.bcs()
    n = t.n

    material_correction = np.zeros(4)
    zero_face_max: Dict[str, float] = {}
    wall_exterior_defect_max = 0.0
    wall_faces_seen = 0

    # One batched ref._metric/_curvature call for every face-quadrature point
    # of this owner at once (via _face_geometry), not one call per face --
    # ref._metric is the dominant cost, and per-face calls reintroduce
    # exactly the redundancy p05n_field_derived_global.core.WallLattice was
    # built to eliminate on the wall side.
    face_points = []
    face_weights = []
    for key, lo, hi in faces:
        points, weights = num.quadrature(t.faces, np.array([key]), order, face=True)
        face_points.append(points[0])
        face_weights.append(weights[0])
    if face_points:
        all_points = np.concatenate(face_points, axis=0)
        all_J, all_B, all_K = p06numerics._face_geometry(ref, all_points)
    offset = 0

    for (key, lo, hi), points, weights in zip(faces, face_points, face_weights):
        axis = int(key[0])
        Q = len(points)
        J, B, K = all_J[offset:offset + Q], all_B[offset:offset + Q], all_K[offset:offset + Q]
        offset += Q

        central = p06n_rows.batched_face_common_value(t, S, owner_values, key, points,
                                                        normal_coefficients=normal_coefficients, ctx=context,
                                                        patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                                        normal_data_fn=normal_fn, donor_accumulator=donor_accumulator)
        sides = p05n_core.batched_side_values(t, S, owner_values, key, points,
                                               normal_coefficients=normal_coefficients, ctx=context,
                                               patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                               normal_data_fn=normal_fn, donor_accumulator=donor_accumulator)

        central_v = _select(central["dirichlet"], central["neumann"], bcs, axis=1)
        lower_v = _select(sides["lower"]["dirichlet"], sides["lower"]["neumann"], bcs, axis=1)
        upper_v = _select(sides["upper"]["dirichlet"], sides["upper"]["neumann"], bcs, axis=1)

        normal_vec = J * K[:, axis] / np.maximum(B * B, 1.0e-30)

        state_central = central_v[:, :4].copy()   # n, Te, Ti, omega -- phi excluded
        state_left = lower_v[:, :4].copy()
        state_right = upper_v[:, :4].copy()

        is_wall = axis == 0 and int(key[1]) == n
        if is_wall:
            interior = state_central
            exterior, working, _fallback = p06numerics._curvature_bc_characteristic_wall_states(
                jnp.asarray(interior), jnp.asarray(interior), jnp.asarray(B), TAU,
                jnp.asarray(normal_vec), interior_on_right=False, positivity_floor=FLOOR,
            )
            state_left = interior
            state_right = np.asarray(exterior)
            state_central = np.asarray(working)
            wall_exterior_defect_max = max(wall_exterior_defect_max, float(np.max(np.abs(state_right - interior))))
            wall_faces_seen += 1

        matrix = p06numerics._principal_matrix(state_central, B)
        flux_matrix = -normal_vec[..., None, None] * matrix
        jump = state_right - state_left
        absolute, _fallback_count = p06numerics._absolute_action(flux_matrix, jump)
        material = np.einsum("qij,qj->qi", flux_matrix, jump)
        dplus = 0.5 * (material + absolute)
        dminus = 0.5 * (material - absolute)

        contribution_lo = -np.sum(weights[:, None] * dminus, axis=0)
        contribution_hi = -np.sum(weights[:, None] * dplus, axis=0)
        if lo == owner:
            material_correction += contribution_lo
        if hi == owner:
            material_correction += contribution_hi

        kind = _face_kind(n, key)
        if kind is not None:
            zero_face_max[kind] = max(zero_face_max.get(kind, 0.0), float(np.max(np.abs(state_right - state_left))))
        if face_report is not None:
            face_report.append(dict(key=[int(v) for v in key], kind=kind, lo=lo, hi=hi,
                                     side_defect=float(np.max(np.abs(state_right - state_left)))))

    return dict(material_correction=material_correction, zero_face_max=zero_face_max,
                wall_exterior_defect_max=wall_exterior_defect_max, wall_faces_seen=wall_faces_seen)


def check_omega_independence(t, S, ref, owner, case: Case, owner_values: np.ndarray, *, normal_coefficients,
                              context, patch_cache, alternate_omega: Role) -> float:
    """Max |action difference| when omega's Role is swapped for an unrelated smooth field.

    Confirms numerically the structural claim (design.md / this module's
    docstring): ``curvature_principal_matrix`` has no omega column, so
    omega's curvature never reaches ``material`` (q1), and ``_principal_matrix``
    (q3) never reads state[...,3] either.
    """
    q1_a = owner_q1(t, S, ref, owner, case, owner_values, normal_coefficients=normal_coefficients,
                     context=context, patch_cache=patch_cache)
    # Rebuild owner_values with the omega column replaced by the alternate role's
    # live observation (only the omega column differs).
    idx = STATE_NAMES.index("omega")
    alt_values = owner_values.copy()
    alt_values[:, idx] = live_observations(t, alternate_omega.evaluate(t.pts)[0][:, None])[:, 0]
    swapped_case = Case(n=case.n, Te=case.Te, Ti=case.Ti, omega=alternate_omega, phi=case.phi)
    q1_b = owner_q1(t, S, ref, owner, swapped_case, alt_values, normal_coefficients=normal_coefficients,
                     context=context, patch_cache=patch_cache)
    q3_a = owner_q3(t, S, ref, owner, case, owner_values, normal_coefficients=normal_coefficients,
                     context=context, patch_cache=patch_cache)
    q3_b = owner_q3(t, S, ref, owner, swapped_case, alt_values, normal_coefficients=normal_coefficients,
                     context=context, patch_cache=patch_cache)
    defect_q1 = max(float(np.max(np.abs(q1_a["N"][term] - q1_b["N"][term]))) for term in ("material", "remainder", "total"))
    defect_q3 = float(np.max(np.abs(q3_a["material_correction"] - q3_b["material_correction"])))
    return max(defect_q1, defect_q3)
