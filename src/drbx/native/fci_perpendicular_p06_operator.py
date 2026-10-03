"""P06 / P06N / P06-legacy curvature operator on the row artifact (P08 step 2b, E4).

Design: ``work/p08_step2b_operator_assembly_design_20260929/design.md`` sections 2.3 and 5.

The host spec is ``scripts/p_shared/replay_units.py`` (``_cells_unit_core``, ``_faces_unit_core``,
``reduce_grid``) with ``scripts/p_shared/apply.py`` (``p06_q1_terms``, ``p06_q3_correction``); the JAX
kernels of ``fci_perpendicular_face_corrections`` are reused and were checked to reproduce them:

* q1: ``p06_midpoint_material_remainder`` on ``(raw, 1, 5)`` states with the evolution weight
  ``w = weight * J / max(B, 1e-30)`` returns ``w * material`` / ``w * remainder`` per raw cell (a sum over
  one q1 node), which is the host's ``_sparse_scatter(term, evolution_weight, owner_ids)`` term. The
  owner sum divided by ``evolution_volume`` (the host's ``max(., 1e-300)`` guard) is the mean.
* q3: ``p06_characteristic_face_correction`` returns ``-sum_q w_q dminus`` / ``-sum_q w_q dplus``
  (lower/upper numerators) with the host's wall characteristic solve, ``lower = central`` and
  ``upper = exterior`` at wall faces, and the spectral fallback of ``_absolute_action``. The only
  difference from the host is the eigen-solver (XLA vs LAPACK ``geev`` through numpy; ``|A| v`` does not
  depend on the eigenvector normalization) and ``n_safe = max(n, 1e-30)`` in the principal matrix
  (the host divides by ``n``): they agree for every state with ``n > 1e-30``.

A **state** is five consecutive field columns ``(n, Te, Ti, omega, phi)``. ``fields`` is ``(n_owners, F)``
with per-column ``field_kinds`` (``"dirichlet"`` / ``"neumann"``, static); ``groups`` ``(V, 5)`` picks the
five columns of each of ``V`` states out of the ``F`` columns, so several P06N variants share the
reconstruction of a physical column:

* P06-legacy, one campaign field: ``F = 5``, all ``"dirichlet"``, ``groups=None`` (outputs have no ``V``
  axis), ``face_multiplier`` = ``campaign_fields.legacy_seam_multiplier(keys)``.
* P06N: :func:`p06n_layout` builds ``(columns, kinds, groups)`` from the adapter ``Reconstruction`` objects
  (unique ``(physical column, kind)`` pairs, each reconstructed once); the caller feeds
  ``fields = owner_values[:, columns]`` and ``bc = bc_columns(bc_all, columns)``.

Every function is one jitted computation with static kinds, so an eager call and a call inside an outer
``jax.jit`` agree bitwise (the internal-jit pattern of ``fci_perpendicular_p07_operator``).

The owner action is **not linear** in the fields (the principal matrix depends on the state, the q3 part
on the eigensystem): use ``jax.jvp``. The q1 numerators are linear in the gradients at fixed values
(``p06_q1_state_numerators``). The q3 JVP is valid away from eigenvalue crossings and sign changes
(``fci_perpendicular_face_corrections._absolute_matrix_action_jvp``).

``absolute_method`` (static, default ``"closed_form"``; ``"lapack4"`` reproduces the frozen campaigns bitwise) switches that eigensystem for the faster ``"block_lapack"`` /
``"closed_form"`` evaluations of the same ``|M| jump`` (``ABSOLUTE_METHODS``); the default is bitwise unchanged.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_face_corrections import (
    ABSOLUTE_METHODS, _validated_absolute_method, p06_characteristic_face_correction,
    p06_midpoint_material_remainder, scatter_p06_characteristic)
from drbx.native.fci_perpendicular_reconstruction_state import (
    BoundaryData, cell_state, face_state, normalize_kinds)
from drbx.stencils.operator_plan import PerpendicularPlan

__all__ = [
    "ABSOLUTE_METHODS", "TAU", "FLOOR", "EVOLUTION_VOLUME_FLOOR", "P06Action", "P06FaceNumerators", "P06RawNumerators",
    "bc_columns", "p06_action", "p06_action_from_state", "p06_q1_raw_numerators", "p06_q1_state_numerators",
    "p06_q3_face_numerators", "p06n_layout"]

#: the campaign constants of ``scripts/p06_structured_global/numerics.py`` (``TAU``, ``FLOOR``)
TAU = 1.0
FLOOR = 1.0e-12
#: ``reduce_grid``'s guard ``evolution_volume_safe = max(evolution_volume, 1e-300)``
EVOLUTION_VOLUME_FLOOR = 1.0e-300


class P06RawNumerators(NamedTuple):
    """Raw-cell evolution-weighted numerators, ``(V, R, 4)`` (``(R, 4)`` if ``groups is None``)."""

    material: object
    remainder: object

    @property
    def total(self):
        return self.material + self.remainder


class P06FaceNumerators(NamedTuple):
    """Per-face lower/upper q3 numerators ``(V, Fc, 4)`` (``face_multiplier`` applied) and per-state counters ``(V,)``."""

    lower: object
    upper: object
    spectral_fallback: object     # q3 absolute-matrix spectral fallbacks (host ``_fb``)
    floor_hits: object            # working-state thermodynamic entries at/below the positivity floor
    wall_fallback: object         # wall-solve fallbacks at wall faces


class P06Action(NamedTuple):
    """Owner-level P06 terms, ``(V, n_owners, 4)`` (no ``V`` axis if ``groups is None``).

    ``material``, ``remainder``, ``total`` are the q1 evolution-weighted owner means; ``correction`` is the
    q3 characteristic correction divided by the q1 evolution volume, i.e. the term the operator adds to
    ``material`` (host ``u_material = material + correction / evolution_volume``). ``*_numerator`` are the
    undivided owner sums exactly as the host's ``_sparse_scatter`` pairs hold them; the frozen P06N
    oracle stores ``correction`` undivided (``correction_numerator``). ``evolution_volume`` is
    ``plan.cells.evolution_volume`` (shared, ``(n_owners,)``).
    """

    material: object
    remainder: object
    total: object
    correction: object
    material_numerator: object
    remainder_numerator: object
    correction_numerator: object
    evolution_volume: object


# --------------------------------------------------------------------------
# Layout helpers (P06N variants over shared physical columns)
# --------------------------------------------------------------------------

def p06n_layout(reconstructions: Sequence) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """``(columns (Fu,), kinds (Fu,), groups (V, 5))`` for a list of ``Reconstruction``-like objects.

    Each object has ``columns`` (five physical column indices) and ``field_kinds`` (five kinds). The
    unique ``(column, kind)`` pairs are ordered by first appearance; ``groups[v, j]`` is the position of
    variant ``v``'s ``j``-th field in that unique list. Feed ``fields = owner_values[:, columns]``,
    ``field_kinds = kinds`` and ``bc = bc_columns(bc, columns)``.
    """
    index: dict = {}
    columns, kinds, groups = [], [], []
    for rec in reconstructions:
        row = []
        if len(rec.columns) != 5 or len(rec.field_kinds) != 5:
            raise ValueError("a P06 state needs five field columns (n, Te, Ti, omega, phi)")
        for column, kind in zip(np.asarray(rec.columns).tolist(), rec.field_kinds):
            key = (int(column), str(kind))
            if key not in index:
                index[key] = len(columns)
                columns.append(int(column)); kinds.append(str(kind))
            row.append(index[key])
        groups.append(row)
    return np.asarray(columns, dtype=np.int64), tuple(kinds), np.asarray(groups, dtype=np.int32)


def bc_columns(bc: BoundaryData, columns) -> BoundaryData:
    """``BoundaryData`` restricted to (and repeated over) the physical field ``columns``."""
    columns = np.asarray(columns, dtype=np.int64)
    take = lambda a, axis: None if a is None else jnp.take(jnp.asarray(a), columns, axis=axis)
    return BoundaryData(take(bc.dirichlet_value, -1), take(bc.dirichlet_tangential, -1), take(bc.neumann_normal, -1))


def _groups_array(groups, n_fields: int):
    """``(groups (V, 5) int32, squeeze)``."""
    if groups is None:
        if n_fields != 5:
            raise ValueError(f"groups=None needs exactly five field columns, got {n_fields}")
        return jnp.arange(5, dtype=jnp.int32)[None], True
    g = jnp.asarray(groups, dtype=jnp.int32)
    if g.ndim != 2 or g.shape[1] != 5:
        raise ValueError("groups must be (V, 5)")
    return g, False


class _Plan:
    """A minimal ``plan`` view (``.cells`` / ``.faces`` only): the state functions take a plan, while this
    module's jitted bodies receive the sub-plans as pytrees."""

    def __init__(self, cells=None, faces=None):
        self.cells = cells
        self.faces = faces


# --------------------------------------------------------------------------
# q1: raw-cell numerators
# --------------------------------------------------------------------------

def _q1_core(value, gradient, bmag, curvature, weight, tau):
    """``(V, R, 5)`` values / ``(V, R, 3, 5)`` gradients -> weighted material / remainder ``(V, R, 4)``."""
    def one(v, g):
        return p06_midpoint_material_remainder(v[:, None, :], jnp.swapaxes(g, -1, -2)[:, None], bmag[:, None],
                                               curvature[:, None], weight[:, None], tau)
    return jax.vmap(one)(value, gradient)


@jax.jit
def _p06_q1_state_numerators(bmag, curvature, weight, value, gradient, tau):
    return _q1_core(value, gradient, bmag, curvature, weight, tau)


def p06_q1_state_numerators(bmag, curvature_vector, evolution_weight, value, gradient, *, tau=TAU):
    """Raw-cell numerators from prepared states.

    ``value (V, R, 5)``, ``gradient (V, R, 3, 5)`` (field order ``n, Te, Ti, omega, phi``; gradient in the
    row-application layout), ``bmag (R,)``, ``curvature_vector (R, 3)`` (``K``), ``evolution_weight (R,)``.
    Returns ``(material, remainder)`` ``(V, R, 4)``, each ``w * term`` (the host
    ``_sparse_scatter`` summand). Linear in ``gradient`` at fixed ``value``.
    """
    return P06RawNumerators(*_p06_q1_state_numerators(
        jnp.asarray(bmag), jnp.asarray(curvature_vector), jnp.asarray(evolution_weight),
        jnp.asarray(value), jnp.asarray(gradient), tau))


def _q1_from_state(cells, cell_value, cell_gradient, groups, tau):
    """Cell ``value (R, F)`` / ``gradient (R, 3, F)`` -> weighted material / remainder ``(V, R, 4)`` per group."""
    value = jnp.moveaxis(cell_value[:, groups], 1, 0)                  # (V, R, 5)
    gradient = jnp.moveaxis(cell_gradient[:, :, groups], 2, 0)         # (V, R, 3, 5)
    return _q1_core(value, gradient, cells.B, cells.K, cells.evolution_weight, tau)


@partial(jax.jit, static_argnames=("kinds",))
def _p06_q1_raw(cells, fields, bc, groups, tau, *, kinds):
    state = cell_state(_Plan(cells=cells), fields, bc, kinds)
    return _q1_from_state(cells, state.value, state.gradient, groups, tau)


def p06_q1_raw_numerators(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, groups=None, *,
                          tau=TAU) -> P06RawNumerators:
    """Raw-cell q1 numerators ``(V, R, 4)`` (cells in ``plan.cells.raw_ids`` order): ``w * material`` and
    ``w * remainder`` with ``w = plan.cells.evolution_weight``."""
    fields = jnp.asarray(fields)
    kinds = normalize_kinds(field_kinds, fields.shape[1])
    g, squeeze = _groups_array(groups, fields.shape[1])
    material, remainder = _p06_q1_raw(plan.cells, fields, bc, g, tau, kinds=kinds)
    if squeeze:
        material, remainder = material[0], remainder[0]
    return P06RawNumerators(material, remainder)


# --------------------------------------------------------------------------
# q3: face numerators
# --------------------------------------------------------------------------

def _q3_core(faces, common, lower, upper, tau, floor, multiplier, absolute_method="closed_form"):
    """States ``(V, Fc, Qf, 4)`` -> ``(lower, upper (V, Fc, 4), spectral, floor, wall counters (V,))``."""
    axis = faces.axis.astype(jnp.int32)
    K_axis = jnp.take_along_axis(faces.K, axis[:, None, None], axis=-1)[..., 0]
    normal = faces.J * K_axis / jnp.maximum(faces.B * faces.B, 1e-30)      # J K_axis / B^2, as the host

    wall_faces = getattr(faces, "wall_faces", None)       # the wall solve runs on the wall faces only

    def one(c, lo, up):
        return p06_characteristic_face_correction(c, lo, up, faces.B, normal, faces.weight, faces.wall,
                                                  faces.collapsed, tau=tau, positivity_floor=floor,
                                                  wall_faces=wall_faces, absolute_method=absolute_method)
    lo_num, up_num, spectral, floor_hits, wall_fallback = jax.vmap(one)(common, lower, upper)
    m = multiplier[None, :, None]
    return lo_num * m, up_num * m, spectral, floor_hits, wall_fallback


def _q3_from_state(faces, face_value, lower, upper, groups, tau, floor, multiplier, absolute_method="closed_form"):
    """Face ``value`` / ``lower`` / ``upper`` ``(Fc, Qf, F)`` -> q3 numerators and counters per group."""
    def pick(x):                                                            # (Fc, Qf, F) -> (V, Fc, Qf, 4)
        return jnp.moveaxis(x[..., groups[:, :4]], 2, 0)
    return _q3_core(faces, pick(face_value), pick(lower), pick(upper), tau, floor, multiplier, absolute_method)


@partial(jax.jit, static_argnames=("kinds", "absolute_method"))
def _p06_q3_faces(faces, fields, bc, groups, tau, floor, multiplier, *, kinds, absolute_method="closed_form"):
    state = face_state(_Plan(faces=faces), fields, bc, kinds, gradients=False)
    return _q3_from_state(faces, state.value, state.lower, state.upper, groups, tau, floor, multiplier,
                          absolute_method)


def _multiplier(plan_faces, face_multiplier):
    if face_multiplier is None:
        return jnp.asarray(plan_faces.face_multiplier)
    m = jnp.asarray(face_multiplier)
    if m.shape != (len(plan_faces.census_row),):
        raise ValueError(f"face_multiplier must have shape ({len(plan_faces.census_row)},), got {m.shape}")
    return m


def p06_q3_face_numerators(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, groups=None, *,
                           tau=TAU, positivity_floor=FLOOR, face_multiplier=None,
                           absolute_method="closed_form") -> P06FaceNumerators:
    """q3 lower/upper numerators per face, ``(V, Fc, 4)`` in ``plan.faces.census_row`` order, times
    ``face_multiplier`` (default ``plan.faces.face_multiplier``, ones; the harness sets 2 on the legacy seam
    faces), plus per-state counters. These are the host's ``corr_lo`` / ``corr_hi`` per face."""
    fields = jnp.asarray(fields)
    kinds = normalize_kinds(field_kinds, fields.shape[1])
    g, squeeze = _groups_array(groups, fields.shape[1])
    out = _p06_q3_faces(plan.faces, fields, bc, g, tau, positivity_floor, _multiplier(plan.faces, face_multiplier),
                        kinds=kinds, absolute_method=_validated_absolute_method(absolute_method))
    if squeeze:
        out = tuple(x[0] for x in out)
    return P06FaceNumerators(*out)


# --------------------------------------------------------------------------
# Owner action
# --------------------------------------------------------------------------

def _action_from_state_core(cells, faces, cell_value, cell_gradient, face_value, lower, upper, groups, tau, floor,
                            multiplier, absolute_method="closed_form", face_groups=None):
    """The owner arithmetic after the reconstruction: 7 arrays of :class:`P06Action` and the q3 counters.

    ``face_groups``: the groups as columns of a pruned face state (``face_state(value_columns=...)``); only its
    first four columns (the evolved fields) are read. Default: ``groups``."""
    n_owners = len(cells.evolution_volume)
    mat_raw, rem_raw = _q1_from_state(cells, cell_value, cell_gradient, groups, tau)
    seg = lambda x: jax.vmap(lambda a: jax.ops.segment_sum(a, cells.raw_owner, num_segments=n_owners))(x)
    material_num, remainder_num = seg(mat_raw), seg(rem_raw)
    lo_num, up_num, spectral, floor_hits, wall_fallback = _q3_from_state(
        faces, face_value, lower, upper, groups if face_groups is None else face_groups, tau, floor, multiplier,
        absolute_method)
    ones = jnp.ones((n_owners,), dtype=lo_num.dtype)
    correction_num = jax.vmap(lambda lo, up: scatter_p06_characteristic(
        lo, up, faces.lower_owner, faces.upper_owner, ones))(lo_num, up_num)
    ev = jnp.maximum(cells.evolution_volume, EVOLUTION_VOLUME_FLOOR)[:, None]
    material, remainder = material_num / ev, remainder_num / ev
    owner = (material, remainder, material + remainder, correction_num / ev,
             material_num, remainder_num, correction_num)
    return owner, (spectral, floor_hits, wall_fallback)


@partial(jax.jit, static_argnames=("kinds", "absolute_method"))
def _p06_action(cells, faces, fields, bc, groups, tau, floor, multiplier, *, kinds, absolute_method="closed_form"):
    cs = cell_state(_Plan(cells=cells), fields, bc, kinds)
    fs = face_state(_Plan(faces=faces), fields, bc, kinds, gradients=False)
    return _action_from_state_core(cells, faces, cs.value, cs.gradient, fs.value, fs.lower, fs.upper, groups, tau,
                                   floor, multiplier, absolute_method)[0]


@partial(jax.jit, static_argnames=("absolute_method",))
def _p06_action_from_state(cells, faces, cell_value, cell_gradient, face_value, lower, upper, groups, tau, floor,
                           multiplier, absolute_method="closed_form"):
    return _action_from_state_core(cells, faces, cell_value, cell_gradient, face_value, lower, upper, groups, tau,
                                   floor, multiplier, absolute_method)


def p06_action(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, groups=None, *, tau=TAU,
               positivity_floor=FLOOR, face_multiplier=None, absolute_method="closed_form") -> P06Action:
    """P06 owner terms of ``fields`` ``(n_owners, F)``; see the module docstring.

    ``groups`` ``(V, 5)`` selects the five columns of each state (``None``: ``F == 5``, no ``V`` axis in
    the result); ``field_kinds`` is per column (a string broadcasts). ``tau`` / ``positivity_floor`` are
    the campaign constants. ``face_multiplier`` ``(Fc,)`` (default ``plan.faces.face_multiplier``) scales
    the q3 numerators before the scatter (2 on P06-legacy's duplicated seam faces). Needs a plan lowered
    with ``cells`` and ``faces``. ``bc`` at the plan's point tables, columns matching ``fields``.

    ``absolute_method`` (static; default ``"closed_form"``; ``"lapack4"`` is bitwise the campaign's 4x4 ``eig``) selects the evaluation of the
    q3 absolute-matrix action: ``"block_lapack"`` (3x3 block ``eig``) or ``"closed_form"`` (cubic root and Sylvester
    projector, no LAPACK); see :func:`drbx.native.fci_perpendicular_face_corrections.p06_characteristic_face_correction`.
    """
    if plan.cells is None or plan.faces is None:
        raise ValueError("p06_action needs a plan lowered with include cells and faces")
    fields = jnp.asarray(fields)
    kinds = normalize_kinds(field_kinds, fields.shape[1])
    g, squeeze = _groups_array(groups, fields.shape[1])
    out = _p06_action(plan.cells, plan.faces, fields, bc, g, tau, positivity_floor,
                      _multiplier(plan.faces, face_multiplier), kinds=kinds,
                      absolute_method=_validated_absolute_method(absolute_method))
    if squeeze:
        out = tuple(x[0] for x in out)
    material, remainder, total, correction, mat_num, rem_num, corr_num = out
    return P06Action(material, remainder, total, correction, mat_num, rem_num, corr_num,
                     jnp.asarray(plan.cells.evolution_volume))


def p06_action_from_state(plan: PerpendicularPlan, cell_value, cell_gradient, face_value, face_lower, face_upper,
                          groups=None, *, tau=TAU, positivity_floor=FLOOR, face_multiplier=None,
                          return_counters: bool = False, absolute_method="closed_form"):
    """The operator arithmetic of :func:`p06_action` on an already-reconstructed state, for callers that share
    one ``cell_state`` / ``face_state`` between operators (P08 step 3, the combined perpendicular RHS).

    ``cell_value (R, F)`` / ``cell_gradient (R, 3, F)`` are ``cell_state(...)`` value and gradient, and
    ``face_value`` / ``face_lower`` / ``face_upper`` ``(Fc, Qf, F)`` the ``face_state(..., gradients=False)`` common
    value and side values, all with the per-column kinds already applied; ``groups`` picks the five columns
    of each state exactly as in :func:`p06_action`. Fed the states :func:`p06_action` reconstructs from the same
    fields it returns the same arrays (it is the same arithmetic; bitwise on the tested plans). With
    ``return_counters`` the result is ``(P06Action, (spectral_fallback, floor_hits, wall_fallback))``, the
    per-state q3 counters of :class:`P06FaceNumerators`.
    """
    if plan.cells is None or plan.faces is None:
        raise ValueError("p06_action_from_state needs a plan lowered with include cells and faces")
    cell_value = jnp.asarray(cell_value)
    g, squeeze = _groups_array(groups, cell_value.shape[1])
    owner, counters = _p06_action_from_state(
        plan.cells, plan.faces, cell_value, jnp.asarray(cell_gradient), jnp.asarray(face_value),
        jnp.asarray(face_lower), jnp.asarray(face_upper), g, tau, positivity_floor,
        _multiplier(plan.faces, face_multiplier), _validated_absolute_method(absolute_method))
    if squeeze:
        owner = tuple(x[0] for x in owner)
        counters = tuple(x[0] for x in counters)
    material, remainder, total, correction, mat_num, rem_num, corr_num = owner
    action = P06Action(material, remainder, total, correction, mat_num, rem_num, corr_num,
                       jnp.asarray(plan.cells.evolution_volume))
    return (action, counters) if return_counters else action
