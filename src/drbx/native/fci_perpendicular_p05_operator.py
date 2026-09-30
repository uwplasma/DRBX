"""P05 / P05N bracket operator on the row artifact: centered midpoint bracket + live q3 upwind jump
(P08 step 2b, E5).

Design: ``work/p08_step2b_operator_assembly_design_20260929/design.md`` sections 2.3 and 5.4 (E5 API note in
section 7). Host spec: ``scripts/p_shared/replay_units.py`` (``_cells_unit_core`` / ``_faces_unit_core``
P05 and P05N blocks, ``reduce_grid`` divisions).

The operator is the sum of two owner terms kept separate for the gate:

* **centered**: the cell gradients (``cell_state``, gradients only) through the E1 kernels
  ``pair_actions`` (``-((h x grad a) . grad b) / |J|`` at the raw midpoints, ``h = plan.cells.h``,
  ``J = plan.cells.jac``) and the raw-volume projection onto owners;
* **jump**: ``p05_scalar_face_jump`` of the common-row gradient and the two side values (``face_state``,
  ``h = plan.faces.h``, ``weight = plan.faces.weight`` the q3 weight, ``axis``) over the faces selected by
  ``jump_mask`` (default ``plan.faces.p07_valid``: valid p07 id and not collapsed), scattered lower plus,
  upper minus (owner ``-1`` dropped).

Both are returned normalized (``/ owner_volume``, the host's ``reduce_grid`` division) and as the
unnormalized numerators the host stores per unit: ``centered_numerator[o] = sum_{raw in o}
raw_volume * action`` (host ``p05_centered`` / ``p05n_*_raw_*``) and ``jump_numerator[o] = sum over
faces (lower +, upper -) of the face jump`` (host ``p05_live_jump_owner_num`` / ``p05n_*_face_*``);
``owner = numerator / owner_volume`` bitwise. ``face_jump`` ``(Fc, P)`` is the per-face oriented jump
(zero outside ``jump_mask``) in ``plan.faces`` order; the host's ``p05_live_jump_values`` are
``face_jump[jump_mask]`` keyed by ``plan.faces.p07_id[jump_mask]``.

Plain P05 is ``field_kinds = "dirichlet"`` for every field (P05 is Dirichlet-only; missing sides take the
prescribed trace at the common row's target points). **P05N** reconstructs each *role* in its own kind:
a role is a physical column of ``fields`` plus a D/N kind (``adapter.reconstructions["role"]``: ``columns
= role_physical_index``, ``field_kinds`` per role), and the N and D actions are pair brackets of that *one*
role state, over ``n_pair_index`` and ``d_pair_index`` (indices into the role columns). :func:`p05n_action`
takes exactly those (``columns`` selects the role columns of ``fields`` and of the boundary data, so
``fields`` and ``bc`` stay in the physical layout the campaign adapter produces; with ``columns=None`` the
inputs are already role columns) and evaluates the concatenated pair list ``n_pair_index + d_pair_index`` in
one pass (the host's ``action_pair_index``), splitting at ``P``: ``raw_N``/``raw_D`` from the cells,
``face_N``/``face_D`` from the faces (all faces of the plan: pass ``jump_mask`` to restrict).

Everything is one jitted computation of ``(cells, faces, fields, bc, mask)`` with static kinds/pairs/columns:
an eager call and a call inside an outer ``jax.jit`` agree bitwise. The action is bilinear in the
reconstructed gradients (affine in ``fields``); the jump is differentiable away from sign changes of the
advection speed (``|speed|``).
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_face_corrections import p05_scalar_face_jump, scatter_p05_jump
from drbx.native.fci_perpendicular_midpoint_bracket import pair_actions, project_raw_to_owners
from drbx.native.fci_perpendicular_reconstruction_state import (
    BoundaryData, cell_state, face_state, normalize_kinds)
from drbx.stencils.operator_plan import PerpendicularPlan

__all__ = ["P05Terms", "P05NTerms", "p05_terms", "p05_terms_from_state", "p05_action", "p05_face_jump",
           "p05n_action"]


class P05Terms(NamedTuple):
    """All P05 outputs for a pair list of length ``P`` (``n`` owners, ``Fc`` plan faces)."""

    centered_owner: object        # (n, P) normalized centered action
    jump_owner: object            # (n, P) normalized live-jump action
    centered_numerator: object    # (n, P) sum raw_volume * action (host ``p05_centered`` before / vol)
    jump_numerator: object        # (n, P) lower +, upper - face-jump scatter (host ``*_owner_num``)
    face_jump: object             # (Fc, P) per-face oriented jump, zero outside the mask
    antisymmetry: object          # scalar max |ab + ba| diagnostic (detached)


class P05NTerms(NamedTuple):
    """P05N outputs: N and D catalogues of ``P`` pairs each (numerators as host ``p05n_*_raw_*`` / ``*_face_*``)."""

    raw_N: object                 # (n, P) normalized centered, N pairs
    raw_D: object
    face_N: object                # (n, P) normalized jump, N pairs
    face_D: object
    raw_N_numerator: object
    raw_D_numerator: object
    face_N_numerator: object
    face_D_numerator: object
    face_jump_N: object           # (Fc, P) per-face jump, zero outside the mask
    face_jump_D: object
    antisymmetry: object


def _static_pairs(pairs, n_fields: int) -> tuple:
    out = tuple((int(a), int(b)) for a, b in pairs)
    if not out:
        raise ValueError("P05 needs at least one (generator, transported) pair")
    for a, b in out:
        if not (0 <= a < n_fields and 0 <= b < n_fields):
            raise ValueError("pair index is outside the field axis")
    return out


def _select_boundary_columns(bc: BoundaryData, columns) -> BoundaryData:
    """Boundary data restricted/repeated to ``columns`` of its trailing field axis (``None`` parts stay)."""
    cols = np.asarray(columns, dtype=np.int32)
    return BoundaryData(None if bc.dirichlet_value is None else jnp.asarray(bc.dirichlet_value)[..., cols],
                        None if bc.dirichlet_tangential is None else jnp.asarray(bc.dirichlet_tangential)[..., cols],
                        None if bc.neumann_normal is None else jnp.asarray(bc.neumann_normal)[..., cols])


def _from_state_core(cells, faces, gradient, common_gradient, lower, upper, mask, pairs) -> P05Terms:
    ones = jnp.ones_like(cells.owner_volume)
    raw_action, antisymmetry = pair_actions(cells.h, cells.jac, gradient, pairs)
    # dividing by one is exact: the projection returns the raw numerator
    centered_num = project_raw_to_owners(raw_action, cells.raw_volume, cells.raw_owner, ones)
    jump = p05_scalar_face_jump(common_gradient, lower, upper, faces.h, faces.weight, faces.axis, pairs)
    jump = jnp.where(mask[:, None], jump, 0.0)
    jump_num = scatter_p05_jump(jump, faces.lower_owner, faces.upper_owner, jnp.ones_like(faces.owner_volume))
    return P05Terms(centered_num / cells.owner_volume[:, None], jump_num / faces.owner_volume[:, None],
                    centered_num, jump_num, jump, antisymmetry)


def _p05_core(cells, faces, fields, bc, mask, kinds, pairs) -> P05Terms:
    ns = SimpleNamespace(cells=cells, faces=faces)
    cs = cell_state(ns, fields, bc, kinds, values=False)
    fs = face_state(ns, fields, bc, kinds, gradients=True)
    return _from_state_core(cells, faces, cs.gradient, fs.gradient, fs.lower, fs.upper, mask, pairs)


@partial(jax.jit, static_argnames=("pairs",))
def _p05_from_state(cells, faces, gradient, common_gradient, lower, upper, mask, *, pairs):
    return _from_state_core(cells, faces, gradient, common_gradient, lower, upper, mask, pairs)


@partial(jax.jit, static_argnames=("kinds", "pairs"))
def _p05_terms(cells, faces, fields, bc, mask, *, kinds, pairs):
    return _p05_core(cells, faces, fields, bc, mask, kinds, pairs)


@partial(jax.jit, static_argnames=("kinds", "pairs", "columns", "n_pairs"))
def _p05n_terms(cells, faces, fields, bc, mask, *, kinds, pairs, columns, n_pairs):
    if columns is not None:
        cols = np.asarray(columns, dtype=np.int32)
        fields = fields[:, cols]
        bc = _select_boundary_columns(bc, cols)
    t = _p05_core(cells, faces, fields, bc, mask, kinds, pairs)
    p = n_pairs
    return P05NTerms(t.centered_owner[:, :p], t.centered_owner[:, p:], t.jump_owner[:, :p], t.jump_owner[:, p:],
                     t.centered_numerator[:, :p], t.centered_numerator[:, p:],
                     t.jump_numerator[:, :p], t.jump_numerator[:, p:],
                     t.face_jump[:, :p], t.face_jump[:, p:], t.antisymmetry)


def _need_parts(plan: PerpendicularPlan):
    if plan.cells is None or plan.faces is None:
        raise ValueError("the P05 operator needs a plan lowered with include cells and faces")
    return plan.cells, plan.faces


def _mask(plan_faces, jump_mask):
    if jump_mask is None:
        return jnp.asarray(plan_faces.p07_valid, dtype=bool)
    m = jnp.asarray(jump_mask, dtype=bool)
    if m.shape != (len(plan_faces.census_row),):
        raise ValueError("jump_mask must have one entry per plan face")
    return m


def p05_terms(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, pairs: Sequence[tuple[int, int]], *,
              jump_mask=None) -> P05Terms:
    """Centered and jump owner terms with numerators and per-face jumps; see the module docstring.

    ``fields`` is ``(n_owners, F)``, ``pairs`` a static sequence of ``(generator, transported)`` column
    indices, ``field_kinds`` per column (``"dirichlet"`` for plain P05).
    """
    cells, faces = _need_parts(plan)
    fields = jnp.asarray(fields)
    kinds = normalize_kinds(field_kinds, fields.shape[1])
    return _p05_terms(cells, faces, fields, bc, _mask(faces, jump_mask), kinds=kinds,
                      pairs=_static_pairs(pairs, fields.shape[1]))


def p05_terms_from_state(plan: PerpendicularPlan, gradient, common_gradient, lower, upper,
                         pairs: Sequence[tuple[int, int]], *, jump_mask=None) -> P05Terms:
    """The operator arithmetic on an already-reconstructed state (what :func:`p05_terms` does after the
    reconstruction), for callers that share the state between operators or substitute another one.

    ``gradient`` ``(R, 3, F)`` (``cell_state(...).gradient``), ``common_gradient`` ``(Fc, Qf, 3, F)`` and the side
    values ``lower`` / ``upper`` ``(Fc, Qf, F)`` (``face_state``), ``pairs`` indexing the last axis. For P05N pass
    the role-selected arrays and ``n_pair_index + d_pair_index`` and split the pair axis at ``P``.
    """
    cells, faces = _need_parts(plan)
    gradient = jnp.asarray(gradient)
    return _p05_from_state(cells, faces, gradient, jnp.asarray(common_gradient), jnp.asarray(lower),
                           jnp.asarray(upper), _mask(faces, jump_mask), pairs=_static_pairs(pairs, gradient.shape[2]))


def p05_action(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, pairs: Sequence[tuple[int, int]], *,
               jump_mask=None):
    """``(centered_owner, jump_owner)``, each ``(n_owners, len(pairs))``; the P05 operator is their sum."""
    t = p05_terms(plan, fields, bc, field_kinds, pairs, jump_mask=jump_mask)
    return t.centered_owner, t.jump_owner


def p05_face_jump(plan: PerpendicularPlan, fields, bc: BoundaryData, field_kinds, pairs: Sequence[tuple[int, int]], *,
                  jump_mask=None):
    """Per-face oriented jump ``(Fc, len(pairs))`` in ``plan.faces`` order (zero outside ``jump_mask``); keyed
    by ``plan.faces.p07_id`` these are the host's ``p05_live_jump_values``."""
    return p05_terms(plan, fields, bc, field_kinds, pairs, jump_mask=jump_mask).face_jump


def p05n_action(plan: PerpendicularPlan, fields, bc: BoundaryData, role_kinds, n_pair_index, d_pair_index, *,
                columns=None, jump_mask=None) -> P05NTerms:
    """P05N ``raw_N``, ``raw_D`` (centered) and ``face_N``, ``face_D`` (jump) owner terms.

    ``columns`` / ``role_kinds`` are the campaign ``Reconstruction("role")`` (``role_physical_index`` and the
    per-role kinds); ``n_pair_index`` / ``d_pair_index`` are the role-index pairs of the N and D actions
    (equal length ``P``). ``fields`` ``(n_owners, F_phys)`` and ``bc`` are in the physical layout and are
    gathered to the role columns inside; ``columns=None`` means ``fields`` / ``bc`` already are role columns.
    ``jump_mask`` defaults to every face of the plan (the host's P05N face domain).
    """
    cells, faces = _need_parts(plan)
    fields = jnp.asarray(fields)
    if columns is None:
        n_roles = fields.shape[1]
    else:
        columns = tuple(int(c) for c in columns)
        if any(not 0 <= c < fields.shape[1] for c in columns):
            raise ValueError("role column outside the field axis")
        n_roles = len(columns)
    kinds = normalize_kinds(role_kinds, n_roles)
    if len(n_pair_index) != len(d_pair_index):
        raise ValueError("n_pair_index and d_pair_index must have the same length")
    p = len(n_pair_index)
    pairs = _static_pairs(tuple(n_pair_index) + tuple(d_pair_index), n_roles)
    mask = _mask(faces, np.ones(len(faces.census_row), dtype=bool) if jump_mask is None else jump_mask)
    return _p05n_terms(cells, faces, fields, bc, mask, kinds=kinds, pairs=pairs, columns=columns, n_pairs=p)
