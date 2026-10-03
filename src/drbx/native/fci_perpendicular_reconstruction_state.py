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
    value: object | None                    # (Fc, Qf, F) common row
    gradient: object | None                 # (Fc, Qf, 3, F)
    lower: object                           # (Fc, Qf, F) lower side value
    upper: object                           # (Fc, Qf, F)


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


def _pick_columns(x, cols: tuple[int, ...]):
    """``x[..., cols]`` as a slice when ``cols`` is a contiguous run (``None`` and the identity pass through)."""
    if x is None or cols == tuple(range(x.shape[-1])):
        return x
    if cols == tuple(range(cols[0], cols[0] + len(cols))):
        return x[..., cols[0]:cols[0] + len(cols)]
    return x[..., np.asarray(cols, dtype=np.int32)]


def _pick_boundary(bc: BoundaryData, cols: tuple[int, ...]) -> BoundaryData:
    return BoundaryData(*(_pick_columns(None if x is None else jnp.asarray(x), cols) for x in bc))


def _set_columns(x, rows, positions: tuple[int, ...], values):
    """``x.at[rows, ..., positions].set(values)`` (``rows=None``: all leading entries) in place, one slice
    write per contiguous run of the static, ascending ``positions``; ``values`` has ``len(positions)`` columns."""
    k = 0
    while k < len(positions):
        j = k
        while j + 1 < len(positions) and positions[j + 1] == positions[j] + 1:
            j += 1
        index = (..., slice(positions[k], positions[j] + 1))
        x = x.at[index if rows is None else (rows,) + index].set(values[..., k:j + 1])
        k = j + 1
    return x


def _neumann_positions(kinds: tuple[str, ...], cols: tuple[int, ...]) -> tuple[int, ...]:
    """Positions within ``cols`` of the Neumann-kind columns."""
    return tuple(i for i, c in enumerate(cols) if kinds[c] == NEUMANN)


def _face_columns_core(faces, fields, bc, kinds, value_columns, gradient_columns):
    """The pruned per-face reconstruction: values (common row, lower, upper) of ``value_columns`` and the common-row
    gradient of ``gradient_columns`` (static tuples of column indices; the outputs follow their order).

    Dirichlet rows run for the requested parts only (values and gradients in separate passes when the column sets
    differ). The Neumann rows are applied to the Neumann-kind columns of each part only (value-only for the values,
    gradient-only for the gradient) and overwrite the Dirichlet state in place, so no per-kind select is needed.
    """
    Fc, nq = faces.common_value_slot.shape
    nv, ng = len(value_columns), len(gradient_columns)
    f_v, bc_v = _pick_columns(fields, value_columns), _pick_boundary(bc, value_columns)
    one_pass = bool(ng) and gradient_columns == value_columns
    if one_pass:
        v, g = apply_source_rows(faces.rows, f_v, _dirichlet_arrays(bc_v, faces.rows.n_boundary_queries > 0),
                                 values=True, gradients=True)
    else:
        v, _ = apply_source_rows(faces.rows, f_v, _dirichlet_arrays(bc_v, faces.rows.n_boundary_queries > 0),
                                 values=True, gradients=False)
        g = None
        if ng:
            f_g, bc_g = _pick_columns(fields, gradient_columns), _pick_boundary(bc, gradient_columns)
            _, g = apply_source_rows(faces.rows, f_g, _dirichlet_arrays(bc_g, faces.rows.n_boundary_queries > 0),
                                     values=False, gradients=True)
    cv = v[faces.common_value_slot]
    cg = g[faces.common_gradient_slot] if ng else None
    lv, uv = v[faces.lower_slot], v[faces.upper_slot]
    lp, up = faces.lower_present[:, None, None], faces.upper_present[:, None, None]
    if faces.has_missing_side:
        if bc.dirichlet_value is None:
            raise ValueError("faces without a side row need BoundaryData.dirichlet_value (the side fallback)")
        fb = bc_v.dirichlet_value[faces.fallback_query]
        lower, upper = jnp.where(lp, lv, fb), jnp.where(up, uv, fb)
    else:
        lower, upper = lv, uv
    pos_v = _neumann_positions(kinds, value_columns)
    pos_g = _neumann_positions(kinds, gradient_columns)
    if not pos_v and not pos_g:
        return FaceState(cv, cg, lower, upper)
    data = _neumann_data(bc)

    def neumann(rows, cols, **part):
        return apply_neumann_point_rows(rows.payload(0), _pick_columns(fields, cols), _pick_columns(data, cols),
                                        **part)

    if faces.common_neumann is not None:
        target = faces.common_neumann_target
        if pos_v:
            nvv, _ = neumann(faces.common_neumann, tuple(value_columns[i] for i in pos_v), gradients=False)
            cv = _set_columns(cv.reshape(Fc * nq, nv), target, pos_v, nvv).reshape(Fc, nq, nv)
        if pos_g:
            _, ngg = neumann(faces.common_neumann, tuple(gradient_columns[i] for i in pos_g), values=False)
            cg = _set_columns(cg.reshape(Fc * nq, 3, ng), target, pos_g, ngg).reshape(Fc, nq, 3, ng)
    if pos_v:
        own_l, own_u = _pick_columns(lv, pos_v), _pick_columns(uv, pos_v)
        if faces.side_neumann is not None:
            sv, _ = neumann(faces.side_neumann, tuple(value_columns[i] for i in pos_v), gradients=False)
            wall_value = jnp.zeros((Fc * nq, len(pos_v)), dtype=sv.dtype).at[faces.side_neumann_target].set(sv)
            wall_value = wall_value.reshape(Fc, nq, len(pos_v))
            own_l = jnp.where(faces.lower_conditioned[:, None, None], wall_value, own_l)
            own_u = jnp.where(faces.upper_conditioned[:, None, None], wall_value, own_u)
        lower = _set_columns(lower, None, pos_v, jnp.where(lp, own_l, own_u))
        upper = _set_columns(upper, None, pos_v, jnp.where(up, own_u, own_l))
    return FaceState(cv, cg, lower, upper)


@partial(jax.jit, static_argnames=("kinds", "value_columns", "gradient_columns"))
def _face_state(faces, fields, bc, *, kinds, value_columns, gradient_columns):
    return _face_columns_core(faces, fields, bc, kinds, value_columns, gradient_columns)


def face_state_pair(plan: PerpendicularPlan, fields, bc: BoundaryData, *,
                    gradients: bool = True) -> tuple[FaceState, FaceState]:
    """``(dirichlet, neumann)`` :class:`FaceState` of every field column.

    The Neumann state differs from the Dirichlet one only through conditioned common rows (value and
    gradient from the R2 Neumann rows), conditioned side rows (the face's R3 Neumann value) and
    missing sides (the other side's N value instead of the trace).
    """
    return _face_pair(plan.faces, jnp.asarray(fields), bc, need_neumann=True, gradients=gradients)


def _column_tuple(columns, n_fields: int, name: str, allow_empty: bool = False) -> tuple[int, ...]:
    cols = tuple(int(c) for c in columns)
    if (not cols and not allow_empty) or any(not 0 <= c < n_fields for c in cols):
        raise ValueError(f"{name} must be {'' if allow_empty else 'non-empty '}column indices in [0, {n_fields}), got {tuple(columns)}")
    return cols


def face_state(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, *,
               gradients: bool = True, value_columns=None, gradient_columns=None) -> FaceState:
    """Per-face q3 reconstruction in ``plan.faces.census_row`` order.

    ``value (Fc, Qf, F)`` / ``gradient (Fc, Qf, 3, F)`` at the common row's nodes and the ``lower`` /
    ``upper`` side values ``(Fc, Qf, F)``, per field column by ``field_kinds`` (see the module docstring
    for the missing-side rules). ``gradients=False`` skips the gradient outputs (P06 needs values only).

    ``value_columns`` / ``gradient_columns`` (static sequences of column indices; ``None`` is every column) prune
    the work to the columns a caller reads: ``value`` / ``lower`` / ``upper`` then hold only ``value_columns``
    and ``gradient`` only ``gradient_columns``, in the order given (the last axis is the position in that
    tuple). The Neumann rows are applied to the Neumann-kind columns of each part only, value-only for the
    values and gradient-only for the gradient. A column's entries equal the full call's up to roundoff (the
    contraction kernels depend on the column count). ``gradient_columns`` needs ``gradients=True``.
    """
    fields = jnp.asarray(fields)
    nf = fields.shape[1]
    kinds = normalize_kinds(field_kinds, nf)
    vcols = tuple(range(nf)) if value_columns is None else _column_tuple(value_columns, nf, "value_columns")
    if not gradients:
        if gradient_columns is not None:
            raise ValueError("gradient_columns needs gradients=True")
        gcols = ()
    else:
        gcols = tuple(range(nf)) if gradient_columns is None else _column_tuple(gradient_columns, nf,
                                                                                 "gradient_columns")
    return _face_state(plan.faces, fields, bc, kinds=kinds, value_columns=vcols, gradient_columns=gcols)
