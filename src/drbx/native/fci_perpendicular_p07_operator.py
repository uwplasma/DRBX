"""P07 diffusion/polarization operator on the row artifact: integrated face flux + owner scatter (P08 step 2b, E3).

Design: ``work/p08_step2b_operator_assembly_design_20260929/design.md`` section 2.3.

``p07_action(plan, fields, bc, field_kinds)`` returns the ``(n_owners, F)`` owner action
(lower face minus, upper face plus, divided by the owner volume; the collapsed r=0 face, family 0,
carries zero flux):

* unconditioned faces: the integrated rows (``apply_integrated_face_rows``);
* conditioned faces (families 1, 2, 4): Dirichlet fields use the Dirichlet lift of the integrated row
  (``value_loading``, ``tangential_loading``); Neumann fields use the Neumann restoration
  ``sum_q integrand[q] . gradient_N[q]`` (``apply_neumann_integrated_face_rows``). The default plan stores these rows
  as one donor union per face with the integrand folded into the weights (``IntegratedNeumannRows``); the per-row layout
  (``lower_perpendicular_plan(neumann_layout="rows")``) agrees to rounding.

Plain P07 is ``field_kinds = "dirichlet"`` for every field (replay fix 2: the frozen campaign's own
Dirichlet lift at conditioned faces); P07N is ``"neumann"`` for every field, and its D variant
``"dirichlet"``. The action is affine in ``fields`` and, for a fixed ``BoundaryData``, linear in
``bc`` too; with ``bc`` zero it is the linear part (JVP = apply of the tangent with zero data).

``columns`` (optional, static) restricts the call to the first ``columns`` columns of ``fields`` / ``bc`` /
``field_kinds`` (the output then has that many columns); the combined RHS passes the four field columns so that the
rows do not run on the ``phi`` column. The restricted columns are materialized contiguously
(``optimization_barrier``). The result is bitwise equal to the leading columns of the full-width call only for some
widths (the contraction kernel XLA selects depends on the column count: measured on the N32 plan, 4 columns equal the
leading four of any 5..8-column call, 2 columns equal those of a 3-column call, a single column or two Neumann columns
against a wider call do not); anything else agrees to rounding.

The body is a single jitted computation with static ``field_kinds``: an eager call and a call inside an
outer ``jax.jit`` agree bitwise.
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from drbx.native.fci_perpendicular_integrated_rows import (
    IntegratedFacePayload, apply_integrated_face_rows, scatter_integrated_face_flux)
from drbx.native.fci_perpendicular_neumann_rows import apply_neumann_integrated_face_rows
from drbx.native.fci_perpendicular_point_rows import BoundaryArrays
from drbx.native.fci_perpendicular_reconstruction_state import (
    NEUMANN, BoundaryData, _mask, normalize_kinds)
from drbx.stencils.operator_plan import PerpendicularPlan

__all__ = ["p07_action", "p07_face_flux"]


def _first_columns(fields, bc, kinds, columns):
    """The first ``columns`` columns of ``fields`` (contiguous), ``bc`` and ``kinds``."""
    if columns is None or columns == fields.shape[1]:
        return fields, bc, kinds
    cut = lambda a: None if a is None else a[..., :columns]
    return (jax.lax.optimization_barrier(fields[:, :columns]),
            BoundaryData(cut(bc.dirichlet_value), cut(bc.dirichlet_tangential), cut(bc.neumann_normal)),
            kinds[:columns])


def _face_flux_core(p07, fields, bc, kinds, columns=None):
    fields, bc, kinds = _first_columns(fields, bc, kinds, columns)
    rows = p07.rows
    barr = None
    if rows.boundary_query_count:
        if bc.dirichlet_value is None or bc.dirichlet_tangential is None:
            raise ValueError("the plan has conditioned P07 faces: BoundaryData needs dirichlet_value and "
                             "dirichlet_tangential")
        barr = BoundaryArrays(bc.dirichlet_value, bc.dirichlet_tangential)
    payload = IntegratedFacePayload(rows.batches, rows.lower_owner, rows.upper_owner, p07.owner_volume,
                                    rows.boundary_query_count, rows.face_count)
    flux = apply_integrated_face_rows(payload, fields, barr)               # Dirichlet lift / plain rows
    if NEUMANN in kinds and p07.neumann is not None:
        if bc.neumann_normal is None:
            raise ValueError("Neumann fields need BoundaryData.neumann_normal")
        restored = apply_neumann_integrated_face_rows(p07.neumann.payload(0), fields, bc.neumann_normal,
                                                      p07.integrand)      # (Fn, F)
        flux_n = flux.at[p07.neumann_face].set(restored)
        flux = flux_n if all(k == NEUMANN for k in kinds) else jnp.where(_mask(kinds), flux_n, flux)
    return jnp.where((p07.family != 0)[:, None], flux, 0.0)


@partial(jax.jit, static_argnames=("kinds", "columns"))
def _p07_face_flux(p07, fields, bc, *, kinds, columns):
    return _face_flux_core(p07, fields, bc, kinds, columns)


@partial(jax.jit, static_argnames=("kinds", "columns"))
def _p07_action(p07, fields, bc, *, kinds, columns):
    flux = _face_flux_core(p07, fields, bc, kinds, columns)
    payload = SimpleNamespace(lower_owner=p07.rows.lower_owner, upper_owner=p07.rows.upper_owner,
                              owner_volume=p07.owner_volume)
    return scatter_integrated_face_flux(payload, flux)


def _columns(columns, n_fields: int):
    if columns is None:
        return None
    columns = int(columns)
    if not 1 <= columns <= n_fields:
        raise ValueError(f"columns must be in 1..{n_fields}, got {columns}")
    return None if columns == n_fields else columns


def p07_face_flux(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, columns=None):
    """Oriented integrated flux ``(Fp, F)`` of every P07 face of ``plan.p07`` (before the owner scatter).

    ``columns``: compute only the first ``columns`` columns (output ``(Fp, columns)``)."""
    fields = jnp.asarray(fields)
    return _p07_face_flux(plan.p07, fields, bc, kinds=normalize_kinds(field_kinds, fields.shape[1]),
                          columns=_columns(columns, fields.shape[1]))


def p07_action(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, columns=None):
    """P07 owner action ``(n_owners, F)`` of ``fields`` ``(n_owners, F)``; see the module docstring.

    ``columns``: compute only the first ``columns`` columns (output ``(n_owners, columns)``)."""
    fields = jnp.asarray(fields)
    return _p07_action(plan.p07, fields, bc, kinds=normalize_kinds(field_kinds, fields.shape[1]),
                       columns=_columns(columns, fields.shape[1]))
