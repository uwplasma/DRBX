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

One ``cell_state`` and one ``face_state`` feed P05 (``p05_terms_from_state``) and P06 (``_action_from_state_core``); the
face state is pruned to the columns the operators read (values of the transported / evolved fields, the gradient of
the bracket's generator columns only); P07 uses its own integrated rows on the field columns. The whole call is one jitted
computation with static names / terms / kinds and the plan as an argument, so an eager call and a call inside an
outer ``jax.jit`` agree bitwise. The result is differentiable in ``state``, ``phi``, ``bc`` and ``params`` away from
the branch points of the operators (P06 q3 eigenvalue crossings, P05 advection-speed sign changes).

``raw_pairs``: explicit ``(generator, transported)`` column-name pairs evaluated through the same shared state with
P05; their unscaled ``P05Terms`` (centered / jump split, per-face jumps) are returned as ``PerpendicularTerms
.raw_pairs``. It is what reaches the frozen campaigns' own pairs, which are not all ``(phi, g)``.

``PerpendicularParams.wall_transport="characteristic"`` changes the bracket's radial cell gradient of the
Dirichlet-kind fields at the outer wall cells where the E x B transport leaves the domain
(:func:`_characteristic_wall_gradient`); the explicit ``raw_pairs``, P06 and P07 always read the unmodified state, and the
default ``"dirichlet"`` is the unmodified RHS (bitwise).
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
    "FIELDS", "PHI", "TERM_NAMES", "WALL_TRANSPORTS", "RHO_STAR_CONVENTIONS", "PerpendicularParams", "PerpendicularTerms",
    "perpendicular_columns", "perpendicular_rhs"]

#: the fields of step 3, in the order of the P06 state / curvature output (Vi and Ve are not enabled)
FIELDS = ("density", "Te", "Ti", "vorticity")
PHI = "phi"
#: production's term names for ``bracket`` / ``curvature`` / ``diffusion``
TERM_NAMES = {"bracket": "poisson_bracket", "curvature": "curvature", "diffusion": "perpendicular_diffusion"}
_TERM_ORDER = tuple(TERM_NAMES)
_CURVATURE_INDEX = {name: i for i, name in enumerate(FIELDS)}
#: choices of ``PerpendicularParams.wall_transport``
WALL_TRANSPORTS = ("dirichlet", "characteristic")
#: choices of ``PerpendicularParams.rho_star_convention``
RHO_STAR_CONVENTIONS = ("single-length", "legacy-bracket-only")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PerpendicularParams:
    """Coefficients of the combined RHS (pytree; differentiable).

    ``rho_star`` divides the bracket only (``rho_star_convention="legacy-bracket-only"``, the default) or, under
    ``rho_star_convention="single-length"`` (``rho_star = rho_s0 / L_ref``), multiplies the bracket and every curvature
    output (``curvature``, ``curvature_material``, ``curvature_remainder``, ``curvature_q1``, ``curvature_correction``);
    ``tau`` enters the curvature only, ``diffusion`` maps a field name to
    its ``D_perp`` (needed for every field when ``diffusion`` is requested), ``positivity_floor`` is the P06 q3
    thermodynamic floor. ``absolute_method`` is the static (not differentiable, part of the jit key) evaluation of the
    P06 q3 absolute-matrix action: ``"lapack4"`` (the campaign's 4x4 ``eig``; pin it to reproduce frozen campaigns bitwise) or
    ``"closed_form"`` (see ``fci_perpendicular_face_corrections.ABSOLUTE_METHODS``). Being part of ``params`` it
    reaches the sharded RHS as well.

    ``wall_transport`` is the static closure of the bracket's radial cell gradient at the outer wall, one of
    :data:`WALL_TRANSPORTS`. ``"dirichlet"`` (the default) is the plan's cell rows as they are: the Dirichlet-kind fields'
    rows on the outer rings are one-sided cubics through the pinned wall value, which over-specifies a field the E x B
    transport carries into the wall (a growing near-wall mode). ``"characteristic"`` replaces, at the wall cells
    (rings ``n - 2`` and ``n - 1``) where the transport is outward (``U^u < 0``, see :func:`_characteristic_wall_gradient`),
    the radial component of the cell gradient of the Dirichlet-kind fields in the ``(phi, g)`` bracket by the interior
    one-sided cubic on the cell's own radial column (no wall value); the value, the other components, P06, the face states,
    the diffusion, the explicit ``raw_pairs`` and the potential solve are unchanged. It needs the plan's ``wall_cells`` /
    ``wall_donors`` / ``wall_weights`` (:class:`~drbx.stencils.operator_plan.CellPlan`: full rings next to the wall) and
    assumes ``rho_star > 0`` (only its sign matters there, in either convention); it is inert without the bracket. Being part
    of ``params`` it reaches the sharded RHS as well.

    ``rho_star_convention`` is the static (part of the jit key) placement of ``rho_star``, one of :data:`RHO_STAR_CONVENTIONS`.
    ``"legacy-bracket-only"`` (the default; the P08 step-5/6 campaigns and their tests pin it) is the bracket divided by
    ``rho_star`` and the curvature free of it, executed literally. ``"single-length"`` is the one-length normalization
    (``rho_star = rho_s0 / L_ref``): the bracket times ``rho_star`` and the whole curvature times ``rho_star``; the
    diffusion is unchanged. At ``rho_star = 1`` the two are bitwise equal. This RHS takes ``phi`` as given and has no
    potential solve or vorticity-to-potential map (the polarization ``rho_star**2 lap(phi + tau p_i) = Omega`` belongs to
    the caller); ``raw_pairs`` are unscaled in both conventions.
    """

    rho_star: object = 1.0
    tau: object = TAU
    diffusion: Mapping = dataclasses.field(default_factory=dict)
    positivity_floor: object = FLOOR
    absolute_method: str = dataclasses.field(default="closed_form", metadata=dict(static=True))
    wall_transport: str = dataclasses.field(default="dirichlet", metadata=dict(static=True))
    rho_star_convention: str = dataclasses.field(default="legacy-bracket-only", metadata=dict(static=True))


class PerpendicularTerms(NamedTuple):
    """Per-field owner arrays ``(n_owners,)`` (all dicts keyed by field name; ``None`` if not requested).

    * ``terms[field][term]``: ``term`` in ``poisson_bracket`` / ``curvature`` / ``perpendicular_diffusion``;
    * ``total[field]``: the sum of the requested terms;
    * ``detail[field]``: ``bracket_centered``, ``bracket_jump`` (already divided by ``rho_star``, or multiplied under ``rho_star_convention="single-length"``),
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


def _validated_wall_transport(value) -> str:
    if value not in WALL_TRANSPORTS:
        raise ValueError(f"wall_transport must be one of {WALL_TRANSPORTS}, got {value!r}")
    return value


def _validated_rho_star_convention(value) -> str:
    if value not in RHO_STAR_CONVENTIONS:
        raise ValueError(f"rho_star_convention must be one of {RHO_STAR_CONVENTIONS}, got {value!r}")
    return value


def _check_wall_transport(plan, params, terms) -> None:
    """Host-side check that ``params.wall_transport`` can act on ``plan`` (a plan or the stacked plan of a sharded one):
    the characteristic closure needs the wall column stencils of the cell plan when the bracket is requested."""
    if (_validated_wall_transport(params.wall_transport) == "characteristic" and "bracket" in terms
            and plan.cells is not None and plan.cells.wall_cells is None):
        raise ValueError(
            "wall_transport='characteristic' needs the wall column stencils of the plan (CellPlan.wall_cells / "
            "wall_donors / wall_weights), which exist only if every column cell of the wall cells on the radial rings "
            "n-4 .. n-1 is its owner's only raw cell (full rings); this plan has agglomerated or ownerless column cells. "
            "Use wall_transport='dirichlet' or a plan with full rings next to the wall.")


def _characteristic_wall_gradient(cells, gradient, stacked, phi_column: int, field_columns: tuple):
    """``gradient`` ``(R, 3, F)`` with its radial component (0, ``d/du``) replaced in the columns ``field_columns`` at the
    wall cells where the E x B transport leaves the domain (the ``wall_transport="characteristic"`` closure).

    The bracket is ``dg/dt = +U . grad g / (|J| rho_star)`` with ``U = -h x grad phi`` (see ``pair_actions``), so ``g`` is
    advected with ``V = -U / (|J| rho_star)``: the transport is outward at the outer wall where ``V^u > 0``, i.e.
    ``U^u < 0`` (``rho_star > 0``; ``h`` is ``cells.h`` and ``grad phi`` the cell gradient of column ``phi_column``).
    The Dirichlet-kind cell rows of the outer rings ``n - 2`` and ``n - 1`` are one-sided cubics through the rings
    ``n - 3 .. n - 1`` and the pinned wall value; where the transport carries the field into the wall that stencil leans
    downwind and the pinned value over-specifies the problem (a growing near-wall mode). A characteristic (upwind)
    closure takes the information from the interior instead: there ``d/du`` of the field is the one-sided cubic on the
    cell's own radial column, rings ``n - 4 .. n - 1``, without the wall value, i.e. ``(1, -6, 3, 2) / (6 du)`` on the
    owners of the rings ``(n - 4, n - 3, n - 2, n - 1)`` for a cell of ring ``n - 2`` and ``(-2, 9, -18, 11) / (6 du)``
    for ring ``n - 1`` (``du = 1 / n``; the weights are ``cells.wall_weights``, the Lagrange derivative at the cell's
    centre, the owners ``cells.wall_donors`` and the cells ``cells.wall_cells``, see ``operator_plan.wall_column_rows``).
    Where the transport enters the domain the plan's row (with the wall value, the physical inflow data) is kept. Only
    the radial component switches, by a hard ``where``; the bracket multiplies it by ``U^u``, so the RHS stays continuous
    (an ``|U^u|`` kink at ``U^u = 0``) and the derivative is that of the selected branch. The value, the theta / eta
    components and every other column are not touched (P06 reads the unmodified gradient).

    ``stacked`` ``(n_owners, F)`` holds the owner values the column stencils read. A ``wall_cells`` entry outside
    ``[0, R)`` is the padding of a sharded plan and does nothing (the scatter drops it; the gathers are clipped).
    """
    if not field_columns:
        return gradient
    cols = np.asarray(field_columns, dtype=np.int32)
    wall = cells.wall_cells
    safe = jnp.minimum(wall, gradient.shape[0] - 1)
    outward = -jnp.cross(cells.h[safe], gradient[safe, :, phi_column])[:, 0] < 0.0          # U^u < 0  <=>  V^u > 0
    interior = jnp.sum(cells.wall_weights[:, :, None] * stacked[:, cols][cells.wall_donors], axis=1)      # (W, ncol)
    current = gradient[safe[:, None], 0, cols[None, :]]
    chosen = jnp.where(outward[:, None], interior, current)
    return gradient.at[wall[:, None], 0, cols[None, :]].set(chosen, mode="drop")


@partial(jax.jit, static_argnames=("columns", "fields", "terms", "kinds", "pairs"))
def _rhs(plan, state, phi, bc, params, jump_mask, face_multiplier, *, columns, fields, terms, kinds, pairs):
    nf = len(fields)
    single_length = params.rho_star_convention == "single-length"
    col = {name: i for i, name in enumerate(columns)}
    stacked = jnp.stack([state[name] for name in columns[:-1]] + [phi], axis=1)          # (n_owners, ncol)
    need_bracket, need_curv = "bracket" in terms, "curvature" in terms
    out: dict = {}
    diagnostics: dict = {}

    if need_bracket or need_curv or pairs:
        index = ([(col[PHI], col[f]) for f in fields] * need_bracket + [(col[a], col[b]) for a, b in pairs]
                 if need_bracket or pairs else [])
        # The face state is pruned to what its consumers read: P05's jump reads the common-row gradient of the
        # generator columns and the side values of the transported columns; P06's q3 reads the common-row value and
        # the side values of the four evolved fields. Everything else (e.g. phi's face value, the gradient of the
        # evolved fields) is never used.
        gcols = tuple(sorted({a for a, _ in index}))
        vcols = tuple(sorted({b for _, b in index} | ({col[f] for f in FIELDS} if need_curv else set())))
        cs = cell_state(plan, stacked, bc, kinds, values=need_curv, gradients=True)
        fs = face_state(plan, stacked, bc, kinds, gradients=bool(gcols), value_columns=vcols,
                        gradient_columns=gcols or None)
    if need_bracket or pairs:
        vpos = {c: i for i, c in enumerate(vcols)}
        face_index = [(gcols.index(a), vpos[b]) for a, b in index]
        # ``wall_transport="characteristic"`` changes the radial cell gradient of the Dirichlet-kind fields in the (phi, g)
        # bracket only; P06 (below) and the explicit ``raw_pairs`` keep ``cs.gradient``
        grad_bracket = cs.gradient
        if need_bracket and params.wall_transport == "characteristic":
            grad_bracket = _characteristic_wall_gradient(
                plan.cells, cs.gradient, stacked, col[PHI], tuple(col[f] for f in fields if kinds[col[f]] == "dirichlet"))
        nb = nf * need_bracket
        split = bool(pairs) and grad_bracket is not cs.gradient         # the raw pairs then take a call of their own
        t = p05_terms_from_state(plan, grad_bracket, fs.gradient, fs.lower, fs.upper, index[:nb] if split else index,
                                 jump_mask=jump_mask, face_pairs=face_index[:nb] if split else face_index)
        t_raw, k = t, nb
        if split:
            t_raw = p05_terms_from_state(plan, cs.gradient, fs.gradient, fs.lower, fs.upper, index[nb:],
                                         jump_mask=jump_mask, face_pairs=face_index[nb:])
            k = 0
        if need_bracket:
            if single_length:
                out["bracket_centered"] = t.centered_owner[:, :nf] * params.rho_star
                out["bracket_jump"] = t.jump_owner[:, :nf] * params.rho_star
                out["bracket"] = (t.centered_owner[:, :nf] + t.jump_owner[:, :nf]) * params.rho_star
            else:
                out["bracket_centered"] = t.centered_owner[:, :nf] / params.rho_star
                out["bracket_jump"] = t.jump_owner[:, :nf] / params.rho_star
                out["bracket"] = (t.centered_owner[:, :nf] + t.jump_owner[:, :nf]) / params.rho_star
            diagnostics["antisymmetry"] = t.antisymmetry
        if pairs:
            out["raw_pairs"] = P05Terms(t_raw.centered_owner[:, k:], t_raw.jump_owner[:, k:],
                                        t_raw.centered_numerator[:, k:], t_raw.jump_numerator[:, k:],
                                        t_raw.face_jump[:, k:], t_raw.antisymmetry)
    if need_curv:
        groups = jnp.asarray([[col[f] for f in FIELDS] + [col[PHI]]], dtype=jnp.int32)
        face_groups = jnp.asarray([[vcols.index(col[f]) for f in FIELDS] + [0]], dtype=jnp.int32)
        owner, (spectral, floor_hits, wall_fallback) = _action_from_state_core(
            plan.cells, plan.faces, cs.value, cs.gradient, fs.value, fs.lower, fs.upper, groups, params.tau,
            params.positivity_floor, face_multiplier, params.absolute_method, face_groups=face_groups)
        material, remainder, q1, correction = (x[0] for x in owner[:4])
        if single_length:
            material, remainder, q1, correction = (params.rho_star * x for x in (material, remainder, q1, correction))
        pick = np.asarray([_CURVATURE_INDEX[f] for f in fields])
        out["curvature_material"] = material[:, pick]
        out["curvature_remainder"] = remainder[:, pick]
        out["curvature_q1"] = q1[:, pick]
        out["curvature_correction"] = correction[:, pick]
        out["curvature"] = q1[:, pick] + correction[:, pick]
        diagnostics.update(spectral_fallback=spectral[0], floor_hits=floor_hits[0], wall_fallback=wall_fallback[0])
    if "diffusion" in terms:
        # P07 reads only the field block (the first ``nf`` columns; the curvature extras, raw-pair columns and phi are
        # never used), so for the four-field layout it runs on those four columns. The integrated-row sums cancel
        # strongly (a changed summation order moves owner values by ~1e-13 relative), and the contraction kernel XLA
        # picks depends on the column count: four columns are bitwise equal to the leading four of any wider call
        # (checked for 5..8 columns, Dirichlet and Neumann kinds), whereas 1 to 3 columns are not (the last column
        # of an odd-width call and the narrow Neumann kernels round differently). Every other field subset therefore
        # keeps the full width, and the whole array enters through a barrier: XLA also reorders the sums when the
        # operand is a strided slice whose layout was chosen for the other consumers.
        action = p07_action(plan, jax.lax.optimization_barrier(stacked), bc, kinds,
                            columns=nf if nf == len(FIELDS) else None)[:, :nf]
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
    ``params.wall_transport="characteristic"`` (the bracket's outflow wall closure, see :class:`PerpendicularParams`)
    needs the plan's wall column stencils (``plan.cells.wall_cells`` / ``wall_donors`` / ``wall_weights``, built by the
    lowering when the wall columns are full rings) and raises ``ValueError`` without them; it does not touch
    ``raw_pairs``. ``params.rho_star_convention`` places ``rho_star`` (see :class:`PerpendicularParams`).
    """
    fields, terms, pairs = _validate(fields, terms, raw_pairs)
    _validated_absolute_method(params.absolute_method)
    _validated_wall_transport(params.wall_transport)
    _validated_rho_star_convention(params.rho_star_convention)
    columns = perpendicular_columns(fields, terms, pairs)
    missing = [c for c in columns[:-1] if c not in state]
    if missing:
        raise KeyError(f"state is missing the columns {missing}; the layout is {columns}")
    needs_states = "bracket" in terms or "curvature" in terms or bool(pairs)
    if needs_states and (plan.cells is None or plan.faces is None):
        raise ValueError("the bracket / curvature / raw_pairs need a plan lowered with cells and faces")
    _check_wall_transport(plan, params, terms)
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
