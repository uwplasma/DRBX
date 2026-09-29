"""Host (NumPy) application of stencil rows and geometry arrays to owner values.

Implements design task 5 ("Host apply.py") of
``work/p08_step1_consolidation_design_20260928/design.md`` (see its section 2
"Row requests per operator", section 4 "Module layout and API", and section 6
task 5: "Compare against the per-owner oracles (p05n rows/operator, p06n
owner_q1/q3, p07n face_chunk)"). This module contributes exactly one new
thing: the *generic* application of a already-prepared row (``PointRows`` /
``NeumannPointRows`` / ``IntegratedFaceRow`` -- from ``drbx.geometry``, or the
same objects expanded from a ``drbx.stencils.artifact`` CSR chunk) to a batch
of per-field owner values. That application is the identical dense
contraction every accepted campaign already performs inline at its own row
objects (``PointRows.apply``/``apply_value`` in
``scripts/perpendicular_structured/reconstruction.py``; the
``nrow.value @ owner_values[nrow.donor_ids] + nrow.boundary_value @ g_bc``
pattern in ``p05n_field_derived_global.core.batched_cell_values``,
``p06n_field_derived_global.rows.batched_face_common_value``, and
``p07n_field_derived_global.core.face_chunk``); it is written fresh here
because the *package* row dataclasses
(``drbx.geometry.fci_perpendicular_reconstruction.PointRows``,
``fci_perpendicular_neumann_trace.NeumannPointRows``,
``fci_perpendicular_integrated_rows.IntegratedFaceRow``) are plain, method-less
dataclasses -- exactly the objects a ``drbx.stencils.artifact`` CSR chunk
expands back into.

Every *operator-specific* numerical action past that point is reused, by
import, from the frozen packages this design pins as oracles -- never
re-derived:

* P05 centered direct midpoint bracket:
  ``p05_direct_midpoint_global.direct_operator`` (``point_bracket``,
  ``direct_pair_actions``, ``project_raw_to_owners``).
* P05 live U-A jump: ``drbx.native.fci_perpendicular_face_corrections``
  (``p05_scalar_face_jump``, ``scatter_p05_jump``), exactly as
  ``p05n_field_derived_global.core.face_chunk`` uses them.
* P06 q1 material/remainder and q3 characteristic correction:
  ``p06_structured_global.numerics`` (``_continuum_terms``,
  ``_principal_matrix``, ``_absolute_action``, ``TAU``, ``FLOOR``) and
  ``drbx.native.fci_operators._curvature_bc_characteristic_wall_states``,
  exactly as ``p06n_field_derived_global.core.raw_chunk``/``face_chunk`` use
  them (the wall solve under the recovered-trace contract:
  ``boundary_trace = interior``). The owner-level scatter reuses
  ``drbx.native.fci_perpendicular_face_corrections.scatter_p06_characteristic``
  (add both sides independently, unlike P05/P07's antisymmetric jump/flux --
  see ``p06n_field_derived_global.campaign.py``'s own
  ``np.add.at(correction[vi], lo, ...); np.add.at(correction[vi], hi, ...)``).
* P07 integrated face flux and owner scatter:
  ``drbx.geometry.fci_perpendicular_integrated_rows.contract_face_tensor``
  (the q3 weighted-tensor integrand) and
  ``drbx.native.fci_perpendicular_integrated_rows.scatter_integrated_face_flux``
  (lower minus, upper plus, divided by stored owner volume), exactly as
  ``p07n_field_derived_global.core.face_chunk`` computes N/D (a regular
  ``IntegratedFaceRow`` gives N == D directly; families 1/2/4 compute N by
  contracting the common-gradient Neumann row -- the same R2-N request P05N/
  P06N use -- against the integrand, and D via the same ``IntegratedFaceRow``
  with its ``value_loading``/``tangential_loading``).

Census choice (``dedupe="dedupe"`` vs. ``"legacy"``, design section 3's
``legacy_alias_slots``): this module does not itself select a census -- the
caller decides which P06 slot-space rows/owners to include (via
``drbx.stencils.census.FaceCensus.dedupe_mask``) before calling the q3/flux
functions here. ``dedupe=True`` drops the periodic theta/eta alias slot
(``FaceCensus.legacy_alias_slots``); ``dedupe=False`` keeps it, reproducing
the accepted (pre-fix) P06 campaign's saved artifact.

Boundary data (Dirichlet ``g``/``dg``, Neumann ``g_N``) are always supplied by
the caller, per field, via a ``trace(points) -> (value (Q, F), gradient
(Q, 3, F))`` callable (the same convention ``dirichlet_trace_all`` uses in
p05n/p06n/p07n's own ``core.py``) or a precomputed ``(28, F)``/``(Q, F)``
array for Neumann wall-lattice data -- this module never evaluates an MMS
field itself.

Import as ``from p_shared import apply`` with ``DRBX/scripts`` (not this
package's own directory) on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent  # .../DRBX/scripts
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p05_direct_midpoint_global.direct_operator import (  # noqa: E402
    point_bracket as p05_bracket,
    direct_pair_actions as p05_pair_actions,
    project_raw_to_owners as p05_project_to_owners,
)
from drbx.native.fci_perpendicular_face_corrections import (  # noqa: E402
    p05_scalar_face_jump,
    scatter_p05_jump,
    scatter_p06_characteristic,
)
import p06_structured_global.numerics as p06numerics  # noqa: E402
from drbx.native.fci_operators import _curvature_bc_characteristic_wall_states  # noqa: E402
from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor  # noqa: E402
from drbx.native.fci_perpendicular_integrated_rows import scatter_integrated_face_flux  # noqa: E402

TAU = p06numerics.TAU
FLOOR = p06numerics.FLOOR

TraceFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Generic per-row application: PointRows / NeumannPointRows / IntegratedFaceRow
# to a (owners, fields) matrix. Bitwise the same contraction the frozen
# campaigns already perform inline (see the module docstring); written fresh
# here because the package row dataclasses carry no methods of their own.
# ---------------------------------------------------------------------------
def apply_point_row(row, owner_values: np.ndarray, trace: TraceFn | None = None):
    """(value, gradient) for one ``PointRows`` row, mirroring
    ``scripts/perpendicular_structured/reconstruction.py``'s ``PointRows.apply``.

    ``value`` has shape ``(Q, F)``; ``gradient`` has shape ``(Q, 3, F)``,
    where ``Q = len(row.trace_target_points)`` and ``F`` is
    ``owner_values.shape[1]``. A boundary-conditioned row (R2/R3 at the wall
    or the last transverse layers) requires ``trace``: the prescribed
    Dirichlet trace/gradient at both the row's donor-side trace points (for
    the lift-and-restore) and its own target points.
    """
    owner_values = np.asarray(owner_values)
    data = owner_values[row.donor_ids]
    if data.ndim != 2:
        raise ValueError("owner_values must have shape (owners, fields)")
    if row.boundary_conditioned:
        if trace is None:
            raise ValueError("a prescribed Dirichlet trace callback is required for a conditioned row")
        donor_trace_value, _donor_trace_gradient = trace(row.trace_donor_points)
        target_trace_value, target_trace_gradient = trace(row.trace_target_points)
        data = data - donor_trace_value
    value = row.value @ data
    gradient = np.einsum("qad,df->qaf", row.gradient, data)
    if row.boundary_conditioned:
        value = value + target_trace_value
        gradient = gradient.copy()
        gradient[:, 1:] += target_trace_gradient[:, 1:]
    return value, gradient


def batch_evaluate(fn: Callable, groups: Sequence[np.ndarray]):
    """Call ``fn`` (a pure *pointwise* map, e.g. a Dirichlet ``trace`` or a
    Neumann ``normal_data_fn`` callback -- never a reduction across its own
    batch axis) **once**, on the concatenation of every array in ``groups``,
    splitting the return value(s) back into one entry per input group.

    This is the fix for the P08 step-1 replay's dominant cost: profiling one
    real boundary-conditioned N32 cells unit found essentially all wall time
    (>3 s/row) inside the oracle physics ``trace``/``normal_data_fn``
    callbacks themselves (``MetricEvaluator.evaluate_prepared`` and its
    finite-difference derivative machinery), called once (or, for a face's
    28-point Neumann boundary data, once *per quadrature node*) per row in a
    Python loop -- never amortized. Because every such callback is pointwise
    (each output row depends only on the matching input point, independent
    of how many other points share the call), concatenating every row's
    query points into one array and calling the callback once produces
    bit-identical per-point results while paying its (dominant, largely
    row-count-independent) per-call cost only once per unit instead of once
    per row -- see the task report's before/after per-unit timing.

    ``groups`` may contain zero-length arrays (skipped) or be empty (returns
    an empty list); ``fn``'s return may be a single ``(Q, ...)`` array or a
    tuple of such arrays (e.g. ``(value, gradient)``).
    """
    counts = [int(np.shape(g)[0]) if len(np.shape(g)) else 0 for g in groups]
    total = sum(counts)
    if total == 0:
        return [None] * len(groups)
    concat = np.concatenate([np.asarray(g, dtype=np.float64) for g, c in zip(groups, counts) if c], axis=0)
    result = fn(concat)
    is_tuple = isinstance(result, tuple)
    parts = result if is_tuple else (result,)
    out = []
    offset = 0
    for c in counts:
        if c == 0:
            out.append((None,) * len(parts) if is_tuple else None)
            continue
        piece = tuple(p[offset:offset + c] for p in parts) if is_tuple else parts[0][offset:offset + c]
        out.append(piece)
        offset += c
    return out


def apply_point_row_precomputed(row, owner_values: np.ndarray, *, donor_trace_value=None,
                                target_trace_value=None, target_trace_gradient=None):
    """Exactly :func:`apply_point_row`'s arithmetic, but taking the
    (possibly batch-evaluated -- see :func:`batch_evaluate`) Dirichlet trace
    outputs directly instead of a ``trace`` callback this function would
    otherwise call itself. Bit-identical to ``apply_point_row(row,
    owner_values, trace)`` where ``donor_trace_value, target_trace_value,
    target_trace_gradient`` are exactly what that call's own
    ``trace(row.trace_donor_points)``/``trace(row.trace_target_points)``
    would have returned."""
    owner_values = np.asarray(owner_values)
    data = owner_values[row.donor_ids]
    if row.boundary_conditioned:
        if donor_trace_value is None or target_trace_value is None or target_trace_gradient is None:
            raise ValueError("precomputed Dirichlet trace values are required for a conditioned row")
        data = data - donor_trace_value
    value = row.value @ data
    gradient = np.einsum("qad,df->qaf", row.gradient, data)
    if row.boundary_conditioned:
        value = value + target_trace_value
        gradient = gradient.copy()
        gradient[:, 1:] += target_trace_gradient[:, 1:]
    return value, gradient


def apply_point_row_value_precomputed(row, owner_values: np.ndarray, *, donor_trace_value=None,
                                      target_trace_value=None):
    """The value half of :func:`apply_point_row_precomputed` (mirrors
    :func:`apply_point_row_value`)."""
    owner_values = np.asarray(owner_values)
    data = owner_values[row.donor_ids]
    if row.boundary_conditioned:
        if donor_trace_value is None or target_trace_value is None:
            raise ValueError("precomputed Dirichlet trace values are required for a conditioned row")
        data = data - donor_trace_value
    value = row.value @ data
    if row.boundary_conditioned:
        value = value + target_trace_value
    return value


def apply_point_row_value(row, owner_values: np.ndarray, trace: TraceFn | None = None):
    """The value half of :func:`apply_point_row`, without the unused gradient
    contraction -- mirrors ``PointRows.apply_value``."""
    owner_values = np.asarray(owner_values)
    data = owner_values[row.donor_ids]
    if data.ndim != 2:
        raise ValueError("owner_values must have shape (owners, fields)")
    if row.boundary_conditioned:
        if trace is None:
            raise ValueError("a prescribed Dirichlet trace callback is required for a conditioned row")
        donor_trace_value, _ = trace(row.trace_donor_points)
        target_trace_value, _ = trace(row.trace_target_points)
        data = data - donor_trace_value
    value = row.value @ data
    if row.boundary_conditioned:
        value = value + target_trace_value
    return value


def apply_neumann_row(row, owner_values: np.ndarray, boundary_data: np.ndarray):
    """(value, gradient) for one ``NeumannPointRows`` row.

    ``boundary_data`` is the caller-supplied ``g_N`` array at
    ``row.boundary_points`` (``(28, F)``), matching the
    ``nrow.value @ owner_values[nrow.donor_ids] + nrow.boundary_value @ g_bc``
    pattern every accepted campaign's batched kernel already performs
    (``p05n_field_derived_global.core.batched_cell_values``,
    ``p06n_field_derived_global.rows.batched_face_common_value``,
    ``p07n_field_derived_global.core.face_chunk``).
    """
    owner_values = np.asarray(owner_values)
    boundary_data = np.asarray(boundary_data)
    donor_values = owner_values[row.donor_ids]
    value = row.value @ donor_values + row.boundary_value @ boundary_data
    gradient = row.gradient @ donor_values + row.boundary_gradient @ boundary_data
    return value, gradient


def apply_neumann_row_value(row, owner_values: np.ndarray, boundary_data: np.ndarray):
    """The value half of :func:`apply_neumann_row` (mirrors ``rows.apply_value``'s
    Neumann branch, e.g. ``p06n_field_derived_global.rows.batched_face_common_value``)."""
    owner_values = np.asarray(owner_values)
    boundary_data = np.asarray(boundary_data)
    donor_values = owner_values[row.donor_ids]
    return row.value @ donor_values + row.boundary_value @ boundary_data


def apply_integrated_row(row, owner_values: np.ndarray, trace: TraceFn | None = None):
    """The one shared, oriented integrated flux for one ``IntegratedFaceRow``.

    Mirrors ``p07n_field_derived_global.core.face_chunk``'s D computation
    (``row.weights @ owner_values[row.donor_ids] + row.value_loading @ trace
    + tangential loading``), which also equals its N for every non-conditioned
    (family 0/3/5/6/7) row -- see :func:`p07_neumann_face_flux` for the
    conditioned-family (1/2/4) N path, which never uses this row's own weights.
    """
    owner_values = np.asarray(owner_values)
    value = row.weights @ owner_values[row.donor_ids]
    if row.boundary_conditioned:
        if trace is None:
            raise ValueError("a prescribed Dirichlet trace callback is required for a conditioned row")
        donor_trace_value, _ = trace(row.trace_donor_points)
        _, target_trace_gradient = trace(row.trace_target_points)
        value = value + row.value_loading @ donor_trace_value
        value = value + np.einsum("qa,qaf->f", row.tangential_loading, target_trace_gradient[:, 1:])
    return value


def apply_integrated_row_precomputed(row, owner_values: np.ndarray, *, donor_trace_value=None,
                                     target_trace_gradient=None):
    """Exactly :func:`apply_integrated_row`'s arithmetic, taking the
    (batch-evaluated -- see :func:`batch_evaluate`) Dirichlet trace outputs
    directly instead of calling a ``trace`` callback itself."""
    owner_values = np.asarray(owner_values)
    value = row.weights @ owner_values[row.donor_ids]
    if row.boundary_conditioned:
        if donor_trace_value is None or target_trace_gradient is None:
            raise ValueError("precomputed Dirichlet trace values are required for a conditioned row")
        value = value + row.value_loading @ donor_trace_value
        value = value + np.einsum("qa,qaf->f", row.tangential_loading, target_trace_gradient[:, 1:])
    return value


# ---------------------------------------------------------------------------
# P05: centered direct midpoint bracket, and the live U-A jump.
# ---------------------------------------------------------------------------
def p05_centered_gradients(rows: Sequence, owner_values: np.ndarray, trace: TraceFn | None = None) -> np.ndarray:
    """Stack :func:`apply_point_row`'s gradient over a sequence of R1 cell
    rows (one per raw midpoint) into the ``(points, 3, roles)`` array
    ``p05_pair_actions``/``direct_pair_actions`` expects.

    ``rows`` holds one ``PointRows`` per (point, role) pair already, in the
    same flattened layout ``p05n_field_derived_global.core.raw_chunk`` builds
    via ``_select_role_matrix`` -- this helper only exists to assemble that
    array from individually-applied rows; every entry of the stack is
    :func:`apply_point_row`'s own gradient (index 0 of its target axis, since
    each row here targets exactly one point).
    """
    gradients = []
    for row in rows:
        _, gradient = apply_point_row(row, owner_values, trace)
        gradients.append(gradient[0])
    return np.stack(gradients, axis=0)


def p05_centered_action(h, jacobian, role_gradients, pairs):
    """The centered direct-midpoint bracket action for every (generator,
    transported) pair, reusing ``direct_pair_actions`` unchanged."""
    return p05_pair_actions(h, jacobian, role_gradients, pairs)


def p05_face_jump(common_gradient, lower_value, upper_value, h_covariant_over_b,
                  quadrature_weight, face_axis, pairs):
    """The frozen P05 q3 live jump, reusing ``p05_scalar_face_jump`` unchanged."""
    return p05_scalar_face_jump(common_gradient, lower_value, upper_value, h_covariant_over_b,
                                quadrature_weight, face_axis, pairs)


def p05_scatter_jump(face_jump, lower_owner, upper_owner, owner_volume):
    """P05's owner scatter (lower plus, upper minus), reusing ``scatter_p05_jump``."""
    return scatter_p05_jump(face_jump, lower_owner, upper_owner, owner_volume)


# ---------------------------------------------------------------------------
# P06: q1 material/remainder, and q3 characteristic face correction.
# ---------------------------------------------------------------------------
def p06_q1_terms(value: np.ndarray, gradient: np.ndarray, prepared):
    """Per-point (material, remainder, total, ...) q1 terms, reusing
    ``p06_structured_global.numerics._continuum_terms`` unchanged.

    ``value``/``gradient`` are the row-application output shape
    (``(Q, 5)``/``(Q, 3, 5)``, field order ``n, Te, Ti, omega, phi``);
    ``_continuum_terms`` itself wants the transposed ``(5, Q)``/``(5, Q, 3)``
    layout -- the same transposes
    ``p06n_field_derived_global.core.raw_chunk`` applies
    (``value.T`` / ``gradient.transpose(2, 0, 1)``) before calling it.
    """
    candidate_values = np.asarray(value).T
    candidate_gradients = np.asarray(gradient).transpose(2, 0, 1)
    return p06numerics._continuum_terms(candidate_values, candidate_gradients, prepared)


def p06_owner_weighted_sum(term: np.ndarray, evolution_weight: np.ndarray) -> np.ndarray:
    """Sum a q1 term over quadrature nodes with the evolution (q1 J/B) weight
    -- the raw-midpoint contribution an owner-level aggregation later divides
    by its own summed ``evolution_weight`` (design section 2's "raw midpoints
    with evolution weight"; see ``p06n_field_derived_global.campaign.py``'s
    own ``np.add.at(material[vi], oid, w[:, None] * z["material"][vi])``)."""
    term = np.asarray(term)
    weight = np.asarray(evolution_weight)
    return np.einsum("q...,q->...", term, weight)


def p06_q3_correction(state_central, state_lower, state_upper, bmag, normal, weight,
                      *, wall: bool = False, tau: float = TAU, positivity_floor: float = FLOOR):
    """One face's characteristic correction (correction_lo, correction_hi),
    reusing ``_principal_matrix``/``_absolute_action`` unchanged; at ``wall``
    faces also reuses ``_curvature_bc_characteristic_wall_states`` under the
    recovered-trace contract (``boundary_trace = interior``), exactly as
    ``p06n_field_derived_global.core.face_chunk`` computes one face's
    ``correction_lo[vi, row]``/``correction_hi[vi, row]``.

    ``state_central``/``state_lower``/``state_upper`` are ``(Q, 4)``
    ``(n, Te, Ti, omega)`` states at this face's ``Q`` quadrature nodes;
    ``bmag``/``normal``/``weight`` are ``(Q,)``. Returns
    ``(correction_lo, correction_hi, spectral_fallback_count)``.
    """
    import jax.numpy as jnp

    # Cast to the canonical float64 dtype object explicitly: on some platforms
    # (e.g. arm64, where numpy's longdouble and float64 share an itemsize) an
    # array can carry a non-canonical float64-like dtype that reprs
    # identically to "float64" but that JAX's dtype table rejects. A plain
    # ``np.asarray(x)`` preserves whatever dtype object the array already
    # has, so the dtype is forced here instead.
    state_central = np.asarray(state_central, dtype=np.float64)
    state_lower = np.asarray(state_lower, dtype=np.float64)
    state_upper = np.asarray(state_upper, dtype=np.float64)
    bmag = np.asarray(bmag, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if wall:
        interior = state_central
        exterior, working, _wall_fallback = _curvature_bc_characteristic_wall_states(
            jnp.asarray(interior), jnp.asarray(interior), jnp.asarray(bmag), tau,
            jnp.asarray(normal), interior_on_right=False, positivity_floor=positivity_floor,
        )
        state_lower = interior
        state_upper = np.asarray(exterior)
        state_central = np.asarray(working)
    matrix = p06numerics._principal_matrix(state_central, bmag)
    flux_matrix = -normal[..., None, None] * matrix
    jump = state_upper - state_lower
    absolute, fallback = p06numerics._absolute_action(flux_matrix, jump)
    material = np.einsum("qij,qj->qi", flux_matrix, jump)
    dplus = 0.5 * (material + absolute)
    dminus = 0.5 * (material - absolute)
    correction_lo = -np.sum(weight[:, None] * dminus, axis=0)
    correction_hi = -np.sum(weight[:, None] * dplus, axis=0)
    return correction_lo, correction_hi, fallback


def p06_scatter_correction(lower_numerator, upper_numerator, lower_owner, upper_owner, evolution_volume):
    """P06's owner scatter: add both sides independently (not an antisymmetric
    jump), reusing ``scatter_p06_characteristic`` unchanged."""
    return scatter_p06_characteristic(lower_numerator, upper_numerator, lower_owner, upper_owner, evolution_volume)


# ---------------------------------------------------------------------------
# P07: integrated face flux (regular and conditioned families), owner scatter.
# ---------------------------------------------------------------------------
def p07_face_flux(row, owner_values: np.ndarray, trace: TraceFn | None = None):
    """The regular-family (0/3/5/6/7) integrated flux, and every family's D:
    an alias of :func:`apply_integrated_row`."""
    return apply_integrated_row(row, owner_values, trace)


def p07_neumann_face_flux(rows: Sequence, owner_values: np.ndarray, boundary_data: Sequence[np.ndarray],
                          integrand: np.ndarray) -> np.ndarray:
    """The conditioned-family (1/2/4) N flux: contract each q3 node's
    common-gradient Neumann row (the R2-N request) against the P07
    weighted-tensor integrand, and sum over nodes -- matching
    ``p07n_field_derived_global.core.face_chunk``'s
    ``N[j] = np.einsum('qa,qak->k', integ[j], restored)`` where
    ``restored[q] = nrow.gradient @ owner_values[nrow.donor_ids]
    + nrow.boundary_gradient @ bc``.

    ``rows`` is one ``NeumannPointRows`` per quadrature node (``Q``, in the
    same order as ``integrand``'s leading axis); ``boundary_data`` is the
    matching per-node ``g_N`` array (``(28, F)`` each), e.g. from
    :func:`p07_neumann_boundary_data`. ``integrand`` is ``(Q, 3)``, e.g. from
    ``contract_face_tensor(weight, tensor, face_axis)[face]``.
    """
    integrand = np.asarray(integrand)
    if integrand.shape[0] != len(rows) or len(boundary_data) != len(rows):
        raise ValueError("one integrand row and one boundary_data array are required per Neumann node")
    total = None
    for q, row in enumerate(rows):
        _, gradient = apply_neumann_row(row, owner_values, boundary_data[q])
        contribution = np.einsum("a,af->f", integrand[q], gradient)
        total = contribution if total is None else total + contribution
    return total


def p07_scatter_flux(face_flux, lower_owner, upper_owner, owner_volume):
    """P07's owner scatter: lower minus, upper plus, divided by stored owner
    volume, reusing ``scatter_integrated_face_flux`` unchanged (duck-typing
    its ``IntegratedFacePayload`` argument down to the three fields it
    actually reads)."""
    from types import SimpleNamespace

    payload = SimpleNamespace(lower_owner=np.asarray(lower_owner), upper_owner=np.asarray(upper_owner),
                              owner_volume=np.asarray(owner_volume))
    return scatter_integrated_face_flux(payload, face_flux)


__all__ = [
    "apply_point_row", "apply_point_row_value",
    "apply_point_row_precomputed", "apply_point_row_value_precomputed", "batch_evaluate",
    "apply_neumann_row", "apply_neumann_row_value",
    "apply_integrated_row", "apply_integrated_row_precomputed",
    "p05_bracket", "p05_pair_actions", "p05_project_to_owners",
    "p05_centered_gradients", "p05_centered_action",
    "p05_face_jump", "p05_scatter_jump",
    "p06_q1_terms", "p06_owner_weighted_sum", "p06_q3_correction", "p06_scatter_correction",
    "p07_face_flux", "p07_neumann_face_flux", "p07_scatter_flux", "contract_face_tensor",
    "TAU", "FLOOR",
]
