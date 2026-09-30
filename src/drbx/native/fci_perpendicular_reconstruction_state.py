"""Boundary data and the per-cell / per-face reconstruction state of the perpendicular operators (P08 step 2b, E3).

Design: ``work/p08_step2b_operator_assembly_design_20260929/design.md`` sections 2.2 and 5.

The state is what every operator (P05 bracket, P06 curvature, P07 diffusion) needs of a field set: the
reconstructed value and gradient at the raw midpoints and the q3 face nodes, with the host replay's
boundary rules (``scripts/p_shared/replay_units.py``, ``_batch_dirichlet_neumann`` /
``_batch_side_values`` / ``_fix_side_neumann_exterior_fallback``; ``apply.py`` row applications):

* **Dirichlet (D)** reconstruction: the Dirichlet-lift rows (``apply_source_rows``): donors minus the
  prescribed trace at the donor queries, plus the trace at the target, plus the two tangential trace
  derivatives (theta, eta) in gradient components 1 and 2.
* **Neumann (N)** reconstruction: at a *conditioned* row the stored Neumann row
  (``value @ donors + boundary_value @ g_N``, gradient likewise); at an unconditioned row it equals D.
* **Faces**, side values. D: the side row's D value; a face without that side row takes the prescribed
  trace at the common row's target points. N: a conditioned side row takes the *face's* R3 Neumann
  value (one Neumann set per face serves both sides), an unconditioned side row its D value, and a
  missing side copies the other side's N value (the exterior rule: it is never the trace).

``BoundaryData`` holds the prescribed data at the plan's global point tables
(``plan.dirichlet_points`` / ``plan.neumann_points``): ``dirichlet_value (Qd, F)``,
``dirichlet_tangential (Qd, 2, F)`` (the theta and eta components of the trace gradient) and
``neumann_normal (Qn, F)`` (the physical-normal derivative ``g_N``). Parts a kind does not need may be
``None``: ``neumann_normal`` for all-Dirichlet fields.

``field_kinds`` is a static tuple, one ``"dirichlet"`` / ``"neumann"`` per field column (or a single
string broadcast to all fields). Operators that mix reconstructions per role build role columns from
the physical fields (the same physical field may appear as a D column and an N column); the
``*_pair`` functions return the D and N reconstructions of every column so a caller can select
freely (P05N pairs the N and D counterparts of each role).

Every function is a jitted computation of (plan, fields, boundary data) with static kinds, so an
eager call and a call inside an outer ``jax.jit`` agree bitwise (the internal-jit pattern of
``apply_source_rows``).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_neumann_rows import apply_neumann_point_rows
from drbx.native.fci_perpendicular_point_rows import BoundaryArrays
from drbx.native.fci_perpendicular_source_rows import apply_source_rows
from drbx.stencils.operator_plan import PerpendicularPlan

__all__ = [
    "BoundaryData", "CellState", "FaceState", "cell_state", "cell_state_pair", "face_state",
    "face_state_pair", "normalize_kinds", "zero_boundary_data", "boundary_data_from_callables"]

DIRICHLET, NEUMANN = "dirichlet", "neumann"


class BoundaryData(NamedTuple):
    """Prescribed data at the plan's boundary point tables."""

    dirichlet_value: object | None          # (Qd, F)
    dirichlet_tangential: object | None     # (Qd, 2, F)  theta, eta components of the trace gradient
    neumann_normal: object | None           # (Qn, F)


class CellState(NamedTuple):
    value: object | None                    # (R, F)
    gradient: object | None                 # (R, 3, F)


class FaceState(NamedTuple):
    value: object | None                    # (Fc, 9, F) common row
    gradient: object | None                 # (Fc, 9, 3, F)
    lower: object                           # (Fc, 9, F) lower side value
    upper: object                           # (Fc, 9, F)


def normalize_kinds(field_kinds, n_fields: int) -> tuple[str, ...]:
    """A per-field tuple of ``"dirichlet"`` / ``"neumann"`` (a single string is broadcast)."""
    if isinstance(field_kinds, str):
        field_kinds = (field_kinds,) * n_fields
    kinds = tuple(field_kinds)
    if len(kinds) != n_fields:
        raise ValueError(f"field_kinds has {len(kinds)} entries for {n_fields} fields")
    bad = set(kinds) - {DIRICHLET, NEUMANN}
    if bad:
        raise ValueError(f"unknown field kinds: {sorted(bad)}")
    return kinds


def zero_boundary_data(plan: PerpendicularPlan, n_fields: int, dtype=np.float64) -> BoundaryData:
    """All-zero boundary data (the linear part of an operator: ``action(f, bc=0)``)."""
    qd, qn = len(plan.dirichlet_points), len(plan.neumann_points)
    return BoundaryData(jnp.zeros((qd, n_fields), dtype), jnp.zeros((qd, 2, n_fields), dtype),
                        jnp.zeros((qn, n_fields), dtype))


def boundary_data_from_callables(plan: PerpendicularPlan, dirichlet: Callable | None = None,
                                 normal: Callable | None = None) -> BoundaryData:
    """Evaluate host callables once at the plan's points.

    ``dirichlet(points (Q, 3)) -> (value (Q, F), gradient (Q, 3, F))`` (the host ``trace`` convention;
    the tangential data is ``gradient[:, 1:]``) and ``normal(points (Q, 3)) -> g_N (Q, F)``. A missing
    callable leaves that part ``None``. Points are passed in one batch.
    """
    dv = dt = nn = None
    if dirichlet is not None and len(plan.dirichlet_points):
        value, gradient = dirichlet(np.asarray(plan.dirichlet_points, dtype=np.float64))
        dv = jnp.asarray(value)
        dt = jnp.asarray(np.asarray(gradient)[:, 1:, :])
    if normal is not None and len(plan.neumann_points):
        nn = jnp.asarray(normal(np.asarray(plan.neumann_points, dtype=np.float64)))
    return BoundaryData(dv, dt, nn)


def _dirichlet_arrays(bc: BoundaryData, needed: bool) -> BoundaryArrays | None:
    if not needed:
        return None
    if bc.dirichlet_value is None or bc.dirichlet_tangential is None:
        raise ValueError("the plan has conditioned rows: BoundaryData needs dirichlet_value and "
                         "dirichlet_tangential")
    return BoundaryArrays(bc.dirichlet_value, bc.dirichlet_tangential)


def _neumann_data(bc: BoundaryData):
    if bc.neumann_normal is None:
        raise ValueError("Neumann fields need BoundaryData.neumann_normal")
    return bc.neumann_normal


def _select(is_neumann, neumann, dirichlet):
    """``where(kind == neumann, N, D)`` over the trailing field axis (arrays or ``None``)."""
    if dirichlet is None:
        return None
    return jnp.where(is_neumann, neumann, dirichlet)


def _mask(kinds: tuple[str, ...]):
    """Static field-kind mask, broadcast against the trailing field axis of ``(..., F)`` arrays."""
    return jnp.asarray([k == NEUMANN for k in kinds])


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------

def _cell_pair_core(cells, fields, bc, need_neumann: bool, values: bool, gradients: bool):
    barr = _dirichlet_arrays(bc, cells.rows.n_boundary_queries > 0)
    v, g = apply_source_rows(cells.rows, fields, barr, values=values, gradients=gradients)
    value_d = v[cells.value_slot] if values else None
    grad_d = g[cells.gradient_slot] if gradients else None
    if not need_neumann or cells.neumann is None:
        return CellState(value_d, grad_d), CellState(value_d, grad_d)
    nv, ng = apply_neumann_point_rows(cells.neumann.payload(0), fields, _neumann_data(bc))
    value_n = value_d.at[cells.neumann_cell].set(nv) if values else None
    grad_n = grad_d.at[cells.neumann_cell].set(ng) if gradients else None
    return CellState(value_d, grad_d), CellState(value_n, grad_n)


@partial(jax.jit, static_argnames=("need_neumann", "values", "gradients"))
def _cell_pair(cells, fields, bc, *, need_neumann, values, gradients):
    return _cell_pair_core(cells, fields, bc, need_neumann, values, gradients)


@partial(jax.jit, static_argnames=("kinds", "values", "gradients"))
def _cell_state(cells, fields, bc, *, kinds, values, gradients):
    need = NEUMANN in kinds
    d, n = _cell_pair_core(cells, fields, bc, need, values, gradients)
    if not need:
        return d
    if DIRICHLET not in kinds:
        return n
    m = _mask(kinds)
    return CellState(_select(m, n.value, d.value), _select(m, n.gradient, d.gradient))


def cell_state_pair(plan: PerpendicularPlan, fields, bc: BoundaryData, *, values: bool = True,
                    gradients: bool = True) -> tuple[CellState, CellState]:
    """``(dirichlet, neumann)`` :class:`CellState` of every field column (they coincide at unconditioned cells)."""
    return _cell_pair(plan.cells, jnp.asarray(fields), bc, need_neumann=True, values=values, gradients=gradients)


def cell_state(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, *, values: bool = True,
               gradients: bool = True) -> CellState:
    """Per-raw-cell reconstruction: ``value (R, F)``, ``gradient (R, 3, F)`` (cells in ``plan.cells.raw_ids`` order).

    Dirichlet columns: the Dirichlet-lift rows. Neumann columns: the Neumann rows at the conditioned
    cells (``+ g_N`` terms), the Dirichlet rows elsewhere. ``fields`` is ``(n_owners, F)``.
    """
    fields = jnp.asarray(fields)
    kinds = normalize_kinds(field_kinds, fields.shape[1])
    return _cell_state(plan.cells, fields, bc, kinds=kinds, values=values, gradients=gradients)


# --------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------

def _face_pair_core(faces, fields, bc, need_neumann: bool, gradients: bool):
    Fc, nq = faces.common_value_slot.shape
    nf = fields.shape[1]
    barr = _dirichlet_arrays(bc, faces.rows.n_boundary_queries > 0)
    v, g = apply_source_rows(faces.rows, fields, barr, values=True, gradients=gradients)
    cv = v[faces.common_value_slot]
    cg = g[faces.common_gradient_slot] if gradients else None
    lv, uv = v[faces.lower_slot], v[faces.upper_slot]
    lp, up = faces.lower_present[:, None, None], faces.upper_present[:, None, None]
    if faces.has_missing_side:
        if bc.dirichlet_value is None:
            raise ValueError("faces without a side row need BoundaryData.dirichlet_value (the side fallback)")
        fb = jnp.asarray(bc.dirichlet_value)[faces.fallback_query]
        lower_d, upper_d = jnp.where(lp, lv, fb), jnp.where(up, uv, fb)
    else:
        lower_d, upper_d = lv, uv
    dirichlet = FaceState(cv, cg, lower_d, upper_d)
    if not need_neumann:
        return dirichlet, dirichlet
    data = _neumann_data(bc)
    cvn, cgn = cv, cg
    if faces.common_neumann is not None:
        nv, ng = apply_neumann_point_rows(faces.common_neumann.payload(0), fields, data)
        target = faces.common_neumann_target
        cvn = cv.reshape(Fc * nq, nf).at[target].set(nv).reshape(Fc, nq, nf)
        if gradients:
            cgn = cg.reshape(Fc * nq, 3, nf).at[target].set(ng).reshape(Fc, nq, 3, nf)
    own_l, own_u = lv, uv
    if faces.side_neumann is not None:
        sv, _sg = apply_neumann_point_rows(faces.side_neumann.payload(0), fields, data)
        wall_value = jnp.zeros((Fc * nq, nf), dtype=sv.dtype).at[faces.side_neumann_target].set(sv)
        wall_value = wall_value.reshape(Fc, nq, nf)
        own_l = jnp.where(faces.lower_conditioned[:, None, None], wall_value, lv)
        own_u = jnp.where(faces.upper_conditioned[:, None, None], wall_value, uv)
    lower_n = jnp.where(lp, own_l, own_u)
    upper_n = jnp.where(up, own_u, own_l)
    return dirichlet, FaceState(cvn, cgn, lower_n, upper_n)


@partial(jax.jit, static_argnames=("need_neumann", "gradients"))
def _face_pair(faces, fields, bc, *, need_neumann, gradients):
    return _face_pair_core(faces, fields, bc, need_neumann, gradients)


@partial(jax.jit, static_argnames=("kinds", "gradients"))
def _face_state(faces, fields, bc, *, kinds, gradients):
    need = NEUMANN in kinds
    d, n = _face_pair_core(faces, fields, bc, need, gradients)
    if not need:
        return d
    if DIRICHLET not in kinds:
        return n
    m = _mask(kinds)
    return FaceState(jnp.where(m, n.value, d.value),
                     None if d.gradient is None else jnp.where(m, n.gradient, d.gradient),
                     jnp.where(m, n.lower, d.lower), jnp.where(m, n.upper, d.upper))


def face_state_pair(plan: PerpendicularPlan, fields, bc: BoundaryData, *,
                    gradients: bool = True) -> tuple[FaceState, FaceState]:
    """``(dirichlet, neumann)`` :class:`FaceState` of every field column.

    The Neumann state differs from the Dirichlet one only through conditioned common rows (value and
    gradient from the R2 Neumann rows), conditioned side rows (the face's R3 Neumann value) and
    missing sides (the other side's N value instead of the trace).
    """
    return _face_pair(plan.faces, jnp.asarray(fields), bc, need_neumann=True, gradients=gradients)


def face_state(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, *,
               gradients: bool = True) -> FaceState:
    """Per-face q3 reconstruction in ``plan.faces.census_row`` order.

    ``value (Fc, 9, F)`` / ``gradient (Fc, 9, 3, F)`` at the common row's nodes and the ``lower`` /
    ``upper`` side values ``(Fc, 9, F)``, per field column by ``field_kinds`` (see the module docstring
    for the missing-side rules). ``gradients=False`` skips the gradient outputs (P06 needs values only).
    """
    fields = jnp.asarray(fields)
    kinds = normalize_kinds(field_kinds, fields.shape[1])
    return _face_state(plan.faces, fields, bc, kinds=kinds, gradients=gradients)
