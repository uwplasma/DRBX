"""Combined perpendicular RHS of the four fields (n, Te, Ti, omega) on the row artifact (P08 step 3, tasks 3.2-3.3).

Design: ``work/p08_step3_combined_rhs_design_20260930/design.md`` (section 2 mapping, section 3 API, section 4 gates).
An opt-in verification path: it is not wired into ``LocalFciDrbEBRhs``.

For every field ``g`` in ``fields`` and the prescribed potential ``phi`` the call returns, per owner,

* ``poisson_bracket``: ``P05(phi, g) / rho_star`` (centered midpoint bracket plus the live q3 upwind jump;
  ``P05(a, b) = -[a, b]`` so this is production's ``-[phi, g] / rho_star``, with
  ``[f, g] = b_cov . (grad f x grad g) / (J B)`` and ``b_cov`` the covariant components of the unit vector ``b``);
* ``curvature``: P06 on the five-column state ``(n, Te, Ti, omega, phi)``: the q1 ``total``
  (material + remainder, the split is kept in ``detail``) plus the q3 ``correction / evolution_volume``;
* ``perpendicular_diffusion``: ``-D_g * P07(g)``. P07 (``p07_action``: integrated q3 face flux of ``J P_perp grad g``,
  scatter lower -, upper +, over the owner volume) is the POSITIVE operator ``-div(P_perp grad g)`` (the frozen
  P07 / P07N convention), so the RHS term is ``-D_g P07(g) = +D_g div(P_perp grad g)``, as production's
  ``_field_perp_diffusion`` (``owner_result = -coefficient * positive_owner``);
* ``total``: their sum, in that order.

Everything is named columns. The columns of the shared reconstruction are ``perpendicular_columns(fields, terms,
raw_pairs)`` = the requested fields, then (for ``curvature``) the four curvature fields that are still missing, then
the columns named in ``raw_pairs``, and ``phi`` last. ``field_kinds`` gives one ``"dirichlet"`` / ``"neumann"`` per
column (a string broadcasts; a mapping is keyed by column name, missing names are ``"dirichlet"``), and ``bc`` is
one :class:`BoundaryData` whose trailing axis follows that column order (``bc_columns`` picks it out of a wider
one). A physical field a campaign reconstructs both ways is therefore fed as two named columns. Vi and Ve are out
of scope: ``fields`` must be a subset of ``FIELDS`` (curvature exists only for these four).

One ``cell_state`` and one ``face_state`` over all columns feed P05 (``p05_terms_from_state``) and P06
(``p06_action_from_state``); P07 uses its own integrated rows on the field columns. The whole call is one jitted
computation with static names / terms / kinds and the plan as an argument, so an eager call and a call inside an
outer ``jax.jit`` agree bitwise. The result is differentiable in ``state``, ``phi``, ``bc`` and ``params`` away from
the branch points of the operators (P06 q3 eigenvalue crossings, P05 advection-speed sign changes).

``raw_pairs``: explicit ``(generator, transported)`` column-name pairs evaluated through the same shared state with
P05; their unscaled ``P05Terms`` (centered / jump split, per-face jumps) are returned as ``PerpendicularTerms
.raw_pairs``. It is what reaches the frozen campaigns' own pairs, which are not all ``(phi, g)``.
"""
from __future__ import annotations

import dataclasses
from functools import partial
from typing import Mapping, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_face_corrections import _validated_absolute_method
from drbx.native.fci_perpendicular_p05_operator import P05Terms, p05_terms_from_state
from drbx.native.fci_perpendicular_p06_operator import (
    FLOOR, TAU, _action_from_state_core, bc_columns)
from drbx.native.fci_perpendicular_p07_operator import p07_action
from drbx.native.fci_perpendicular_reconstruction_state import (
    BoundaryData, cell_state, face_state, normalize_kinds)
from drbx.stencils.operator_plan import PerpendicularPlan

__all__ = [
    "FIELDS", "PHI", "TERM_NAMES", "PerpendicularParams", "PerpendicularTerms", "perpendicular_columns",
    "perpendicular_rhs"]

#: the fields of step 3, in the order of the P06 state / curvature output (Vi and Ve are not enabled)
FIELDS = ("density", "Te", "Ti", "vorticity")
PHI = "phi"
#: production's term names for ``bracket`` / ``curvature`` / ``diffusion``
TERM_NAMES = {"bracket": "poisson_bracket", "curvature": "curvature", "diffusion": "perpendicular_diffusion"}
_TERM_ORDER = tuple(TERM_NAMES)
_CURVATURE_INDEX = {name: i for i, name in enumerate(FIELDS)}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PerpendicularParams:
    """Coefficients of the combined RHS (pytree; differentiable).

    ``rho_star`` divides the bracket only, ``tau`` enters the curvature only, ``diffusion`` maps a field name to
    its ``D_perp`` (needed for every field when ``diffusion`` is requested), ``positivity_floor`` is the P06 q3
    thermodynamic floor. ``absolute_method`` is the static (not differentiable, part of the jit key) evaluation of the
    P06 q3 absolute-matrix action: ``"lapack4"`` (default, the campaign's 4x4 ``eig``), ``"block_lapack"`` or
    ``"closed_form"`` (see ``fci_perpendicular_face_corrections.ABSOLUTE_METHODS``). Being part of ``params`` it
    reaches the sharded RHS as well.
    """

    rho_star: object = 1.0
    tau: object = TAU
    diffusion: Mapping = dataclasses.field(default_factory=dict)
    positivity_floor: object = FLOOR
    absolute_method: str = dataclasses.field(default="lapack4", metadata=dict(static=True))


class PerpendicularTerms(NamedTuple):
    """Per-field owner arrays ``(n_owners,)`` (all dicts keyed by field name; ``None`` if not requested).

    * ``terms[field][term]``: ``term`` in ``poisson_bracket`` / ``curvature`` / ``perpendicular_diffusion``;
    * ``total[field]``: the sum of the requested terms;
    * ``detail[field]``: ``bracket_centered``, ``bracket_jump`` (already divided by ``rho_star``),
      ``curvature_material``, ``curvature_remainder``, ``curvature_q1`` (= ``total`` of P06: material + remainder),
      ``curvature_correction`` (q3, divided by the q1 evolution volume); ``curvature == q1 + correction``;
    * ``raw_pairs``: :class:`P05Terms` of the explicit ``raw_pairs`` (unscaled, pair axis in the order given);
    * ``diagnostics``: P06 q3 counters (``spectral_fallback``, ``floor_hits``, ``wall_fallback``) and the P05
      bracket ``antisymmetry`` when the corresponding term ran.
    """

    terms: dict
    total: dict
    detail: dict
    raw_pairs: Optional[P05Terms]
    diagnostics: dict


def perpendicular_columns(fields: Sequence[str] = FIELDS, terms: Sequence[str] = _TERM_ORDER,
                          raw_pairs: Optional[Sequence[tuple[str, str]]] = None) -> tuple[str, ...]:
    """Column names of the shared state (and of ``field_kinds`` / ``bc``): fields, curvature extras, raw-pair
    columns, ``phi`` last. The leading ``len(fields)`` columns are the requested fields, in that order."""
    fields, terms, pairs = _validate(fields, terms, raw_pairs)
    names = list(fields)
    if "curvature" in terms:
        names += [f for f in FIELDS if f not in names]
    for a, b in pairs:
        names += [x for x in (a, b) if x != PHI and x not in names]
    return tuple(names) + (PHI,)


def _validate(fields, terms, raw_pairs):
    fields = tuple(fields)
    if len(set(fields)) != len(fields):
        raise ValueError("fields must be distinct")
    if not fields:
        raise ValueError("at least one field is required")
    bad = [f for f in fields if f not in FIELDS]
    if bad:
        raise ValueError(f"unsupported fields {bad}: the step-3 RHS covers {FIELDS} (Vi/Ve are out of scope)")
    terms = tuple(terms)
    unknown = sorted(set(terms) - set(_TERM_ORDER))
    if unknown or len(set(terms)) != len(terms):
        raise ValueError(f"terms must be distinct members of {_TERM_ORDER}, got {terms}")
    terms = tuple(t for t in _TERM_ORDER if t in terms)
    pairs = () if raw_pairs is None else tuple((str(a), str(b)) for a, b in raw_pairs)
    if not terms and not pairs:
        raise ValueError("nothing to compute: no terms and no raw_pairs")
    return fields, terms, pairs


def _kinds(field_kinds, columns) -> tuple[str, ...]:
    if isinstance(field_kinds, str):
        return normalize_kinds(field_kinds, len(columns))
    if isinstance(field_kinds, Mapping):
        extra = sorted(set(field_kinds) - set(columns))
        if extra:
            raise ValueError(f"field_kinds names columns that are not in the layout {columns}: {extra}")
        return normalize_kinds(tuple(field_kinds.get(c, "dirichlet") for c in columns), len(columns))
    return normalize_kinds(tuple(field_kinds), len(columns))


@partial(jax.jit, static_argnames=("columns", "fields", "terms", "kinds", "pairs"))
def _rhs(plan, state, phi, bc, params, jump_mask, face_multiplier, *, columns, fields, terms, kinds, pairs):
    nf = len(fields)
    col = {name: i for i, name in enumerate(columns)}
    stacked = jnp.stack([state[name] for name in columns[:-1]] + [phi], axis=1)          # (n_owners, ncol)
    need_bracket, need_curv = "bracket" in terms, "curvature" in terms
    out: dict = {}
    diagnostics: dict = {}

    if need_bracket or need_curv or pairs:
        cs = cell_state(plan, stacked, bc, kinds, values=need_curv, gradients=True)
        fs = face_state(plan, stacked, bc, kinds, gradients=need_bracket or bool(pairs))
    if need_bracket or pairs:
        index = [(col[PHI], col[f]) for f in fields] * need_bracket + [(col[a], col[b]) for a, b in pairs]
        t = p05_terms_from_state(plan, cs.gradient, fs.gradient, fs.lower, fs.upper, index, jump_mask=jump_mask)
        if need_bracket:
            out["bracket_centered"] = t.centered_owner[:, :nf] / params.rho_star
            out["bracket_jump"] = t.jump_owner[:, :nf] / params.rho_star
            out["bracket"] = (t.centered_owner[:, :nf] + t.jump_owner[:, :nf]) / params.rho_star
            diagnostics["antisymmetry"] = t.antisymmetry
        if pairs:
            k = nf * need_bracket
            out["raw_pairs"] = P05Terms(t.centered_owner[:, k:], t.jump_owner[:, k:], t.centered_numerator[:, k:],
                                        t.jump_numerator[:, k:], t.face_jump[:, k:], t.antisymmetry)
    if need_curv:
        groups = jnp.asarray([[col[f] for f in FIELDS] + [col[PHI]]], dtype=jnp.int32)
        owner, (spectral, floor_hits, wall_fallback) = _action_from_state_core(
            plan.cells, plan.faces, cs.value, cs.gradient, fs.value, fs.lower, fs.upper, groups, params.tau,
            params.positivity_floor, face_multiplier, params.absolute_method)
        material, remainder, q1, correction = (x[0] for x in owner[:4])
        pick = np.asarray([_CURVATURE_INDEX[f] for f in fields])
        out["curvature_material"] = material[:, pick]
        out["curvature_remainder"] = remainder[:, pick]
        out["curvature_q1"] = q1[:, pick]
        out["curvature_correction"] = correction[:, pick]
        out["curvature"] = q1[:, pick] + correction[:, pick]
        diagnostics.update(spectral_fallback=spectral[0], floor_hits=floor_hits[0], wall_fallback=wall_fallback[0])
    if "diffusion" in terms:
        # P07 runs on every column of the state and the field block is sliced from its output. The integrated-row
        # sums cancel strongly (a changed summation order moves owner values by ~1e-13 relative), and XLA reorders
        # them when the operand is a strided slice (or a column ``take`` of the boundary data) whose layout was
        # chosen for the other consumers; fed the whole array it reproduces the standalone ``p07_action`` bitwise.
        action = p07_action(plan, jax.lax.optimization_barrier(stacked), bc, kinds)[:, :nf]
        d = jnp.stack([jnp.asarray(params.diffusion[f]) for f in fields])
        out["diffusion"] = -d[None, :] * action
    total = None
    for term in terms:
        total = out[term] if total is None else total + out[term]
    out["total"] = total
    return out, diagnostics


def perpendicular_rhs(plan: PerpendicularPlan, state: Mapping[str, object], phi, bc: BoundaryData, field_kinds,
                      params: PerpendicularParams, *, fields: Sequence[str] = FIELDS,
                      terms: Sequence[str] = _TERM_ORDER,
                      raw_pairs: Optional[Sequence[tuple[str, str]]] = None, jump_mask=None,
                      face_multiplier=None) -> PerpendicularTerms:
    """The perpendicular RHS terms of ``fields`` at ``state`` and the prescribed potential ``phi``.

    ``state`` maps column names to ``(n_owners,)`` owner values and must hold every column of
    :func:`perpendicular_columns` except ``phi`` (extra entries are ignored); ``phi`` is ``(n_owners,)``.
    ``bc`` and ``field_kinds`` follow the column order (see the module docstring), ``terms`` picks among
    ``bracket`` / ``curvature`` / ``diffusion``. ``jump_mask`` (default ``plan.faces.p07_valid``) and
    ``face_multiplier`` (default ``plan.faces.face_multiplier``, ones) are the P05 / P06 options of the same name.
    The plan needs ``cells`` and ``faces`` for the bracket / curvature / ``raw_pairs`` and ``p07`` for the diffusion.
    """
    fields, terms, pairs = _validate(fields, terms, raw_pairs)
    _validated_absolute_method(params.absolute_method)
    columns = perpendicular_columns(fields, terms, pairs)
    missing = [c for c in columns[:-1] if c not in state]
    if missing:
        raise KeyError(f"state is missing the columns {missing}; the layout is {columns}")
    needs_states = "bracket" in terms or "curvature" in terms or bool(pairs)
    if needs_states and (plan.cells is None or plan.faces is None):
        raise ValueError("the bracket / curvature / raw_pairs need a plan lowered with cells and faces")
    if "diffusion" in terms:
        if plan.p07 is None:
            raise ValueError("the diffusion needs a plan lowered with p07")
        absent = [f for f in fields if f not in params.diffusion]
        if absent:
            raise ValueError(f"params.diffusion has no coefficient for {absent}")
    diffusion = {f: params.diffusion[f] for f in fields} if "diffusion" in terms else {}
    params = dataclasses.replace(params, diffusion=diffusion)
    kinds = _kinds(field_kinds, columns)
    if needs_states:
        jump_mask = jnp.asarray(plan.faces.p07_valid, dtype=bool) if jump_mask is None else jnp.asarray(
            jump_mask, dtype=bool)
        face_multiplier = jnp.asarray(plan.faces.face_multiplier) if face_multiplier is None else jnp.asarray(
            face_multiplier)
    else:
        jump_mask = face_multiplier = None
    used = {c: jnp.asarray(state[c]) for c in columns[:-1]}
    out, diagnostics = _rhs(plan, used, jnp.asarray(phi), bc, params, jump_mask, face_multiplier, columns=columns,
                            fields=fields, terms=terms, kinds=kinds, pairs=pairs)

    def per_field(key):
        return {f: out[key][:, i] for i, f in enumerate(fields)}

    term_arrays = {t: per_field(t) for t in terms}
    by_field = {f: {TERM_NAMES[t]: term_arrays[t][f] for t in terms} for f in fields}
    detail_keys = [k for k in ("bracket_centered", "bracket_jump", "curvature_material", "curvature_remainder",
                               "curvature_q1", "curvature_correction") if k in out]
    detail = {f: {k: out[k][:, i] for k in detail_keys} for i, f in enumerate(fields)}
    total = per_field("total") if terms else {}
    return PerpendicularTerms(by_field, total, detail, out.get("raw_pairs"), diagnostics)
