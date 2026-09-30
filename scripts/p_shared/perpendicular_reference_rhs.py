"""Host continuum reference of the production perpendicular RHS terms (P08 step 3, gate G3.3).

Design: ``work/p08_step3_combined_rhs_design_20260930/design.md`` (section 1 production expressions,
section 2 mapping, section 4 G3.3).  Given a manufactured 5-field state ``(n, Te, Ti, omega, phi)``
(exact values and gradients supplied by a *state adapter*) and the reference geometry ``env.ref``, this
module computes, **per term and per owner**, the continuum value of the four fields' production
perpendicular RHS terms

* ``poisson_bracket``  ``-[phi, g] / rho_star`` (``[f, g] = b_cov.(grad f x grad g) / (J B)``, plain phi
  is the generator for every field ``g``);
* ``curvature``        ``(1/B) M(q) K.grad q  +  coeff K.grad(phi + tau Ti)`` with
  ``coeff = (-2n, -4Te/3, -4Ti/3, 0)/B`` (material + remainder; the split is also returned);
* ``perpendicular_diffusion``  ``D_f div(P_perp grad f)``, ``P_perp = g^ij - b^i b^j``;

as **owner-level references built exactly like the frozen campaigns' own MMS references**, so a
discretization error measured against them (``N - R``) is directly comparable to the qualified
operators' ``N - R``:

bracket
    P05N ``raw_R`` (``replay_units._cells_unit_core``): exact gradients at the raw-cell midpoints,
    ``p05_direct_midpoint_global.direct_operator.point_bracket``, projected to owners by raw volume over
    owner volume.  ``point_bracket(h, |J|, grad a, grad b) = -h.(grad a x grad b)/|J| = -[a, b]``
    with ``h = b_cov / B``; hence ``-[phi, g] = point_bracket(., grad phi, grad g)``.  The live q3
    face-jump reference is zero for exact continuous fields.
curvature
    P06N ``raw_R_material/remainder/total`` (``p06_structured_global.numerics._continuum_terms`` with
    exact values/gradients at the raw midpoints, evolution-weighted owner means: weight
    ``q1_weight * J / max(B, 1e-30)``).  The q3 correction reference is zero.
diffusion
    P07N ``O_q3`` (``replay_units._p07_unit_core``): exact gradients contracted with the q3
    weighted-tensor integrand per census face (``contract_face_tensor`` of the ``J P_perp`` tensor and
    the 9-node face quadrature), scattered lower ``-``, upper ``+``, divided by the owner volume.  Family-0
    (collapsed ``r = 0``) faces carry zero.

Only the given owner set is ever evaluated: the raw cells of the owners, and the ``p07`` census faces
incident to them (exactly the sets :func:`p_shared.owner_closure.build_owner_rows` uses).  Nothing here
reads a grid-sized geometry array beyond the ``env.t`` context arrays that ``build_environment`` already
holds, and no row/stencil artifact is built.

Conventions (verified against the code, see :func:`production_formula_check`)
* ``env.ref._metric(q)["bcov"]`` holds the covariant components of the **unit** vector ``b = B/|B|``
  (``bcov = g_cov . (B_contra/|B|)``), so the design's ``b_cov`` in ``b_cov.(grad f x grad g)/(J B)`` is
  that quantity and ``h = bcov / B`` is what the campaigns feed to the bracket.
* ``B`` is normalised by the reference field ``B0``; ``tau`` enters only the curvature terms.

Independent check
    :func:`production_formula_check` re-derives the bracket and curvature terms *directly from the
    design's production expressions* (own cross product with the un-normalised ``J``,
    ``curvature_principal_matrix`` as production calls it, the production remainder coefficients, and the
    continuum-reduced rows of design section 1) and reports the maximum relative difference against the
    campaign-style constructions above at the same raw midpoints.

Import as ``from p_shared import perpendicular_reference_rhs`` with ``DRBX/scripts`` on ``sys.path``
(never a directory holding a stray ``operator.py``).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p_shared import owner_closure as oc                               # noqa: E402
from p_shared import replay_units as ru                                # noqa: E402
from p_shared import campaign_fields as cf                             # noqa: E402
from p_shared import provider as pshared_provider                      # noqa: E402
from drbx.stencils import builder as stencil_builder                    # noqa: E402
from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor  # noqa: E402
from drbx.native.fci_curvature_production_flux import curvature_principal_matrix  # noqa: E402

__all__ = [
    "FIELDS", "SLOTS", "TERMS", "ReferenceParams", "OwnerSupport", "owner_support", "P06NState",
    "p06n_state", "bracket_reference", "curvature_reference", "diffusion_reference", "reference_rhs",
    "production_formula_pointwise", "production_formula_check", "diffusion_midpoint_check",
    "verify_against_oracles",
]

#: The four gated fields, in production's names, and the state slots they occupy.
FIELDS = ("density", "Te", "Ti", "vorticity")
#: The 5-column manufactured state order (P06N ``state_order``); ``phi`` is the prescribed generator.
SLOTS = ("n", "Te", "Ti", "omega", "phi")
#: Production's term names (``fci_drb_EB_rhs.RHS_TERM_NAMES``) of the three perpendicular terms.
TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion")

_PHI = 4


# ---------------------------------------------------------------------------
# Parameters.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceParams:
    """``rho_star`` (bracket only), ``tau`` (curvature only) and the per-field perpendicular diffusion
    coefficients ``D_f`` (a mapping field -> D; missing fields default to 1)."""
    rho_star: float = 1.0
    tau: float = 1.0
    diffusion: Mapping[str, float] = field(default_factory=dict)

    def D(self, name: str) -> float:
        return float(self.diffusion.get(name, 1.0))


@contextmanager
def _patched_tau(tau: float):
    """``p06_structured_global.numerics._continuum_terms`` reads the module constant ``TAU``; the campaign
    construction is reused unchanged by setting it for the duration of the call."""
    import p06_structured_global.numerics as p06numerics

    old = p06numerics.TAU
    p06numerics.TAU = float(tau)
    try:
        yield
    finally:
        p06numerics.TAU = old


# ---------------------------------------------------------------------------
# Owner support: exactly the raw cells / p07 faces the given owners need.
# ---------------------------------------------------------------------------
@dataclass
class OwnerSupport:
    """Bounded geometry-independent support of an owner set (ids only; geometry is computed lazily)."""
    owners: np.ndarray
    raw_ids: np.ndarray
    raw_owner: np.ndarray
    raw_volume: np.ndarray
    points: np.ndarray
    raw_keys: np.ndarray
    p07_rows: np.ndarray
    p07_keys: np.ndarray
    p07_family: np.ndarray
    p07_lower: np.ndarray
    p07_upper: np.ndarray
    owner_volume: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False)

    def raw_geometry(self, env):
        """``(h, |J|, J, B, bcov, K, evolution_weight)`` at the raw midpoints (computed once)."""
        if "raw" not in self._cache:
            from perpendicular_structured.reference_geometry import curvature_geometry
            from p07_diffusion_global.numerics import quadrature

            metric = env.ref._metric(self.points)
            h = metric["bcov"] / metric["B"][:, None]
            prepared = curvature_geometry(env.ref, self.points)
            _pts, q1_weight = quadrature(env.t.faces, self.raw_keys, 1, face=False)
            weight = q1_weight.reshape(-1)
            evolution_weight = weight * np.asarray(prepared.J) / np.maximum(np.asarray(prepared.B), 1.0e-30)
            self._cache["raw"] = {"h": h, "jac": np.abs(metric["J"]), "J": np.asarray(metric["J"]),
                                  "B": np.asarray(metric["B"]), "bcov": np.asarray(metric["bcov"]),
                                  "prepared": prepared, "evolution_weight": evolution_weight}
        return self._cache["raw"]


def owner_support(env, owners, built: Optional[dict] = None) -> OwnerSupport:
    """The bounded support of ``owners`` (a sequence of distinct owner ids, or a ``build_owner_rows``
    dict, whose ``"owners"`` are used).  ``built`` (optional) supplies ``raw_ids``/``p07_row_indices`` from
    an already-built closure; otherwise they are derived exactly as ``build_owner_rows`` does."""
    if isinstance(owners, OwnerSupport):
        return owners
    if isinstance(owners, Mapping):
        built = owners
        owners = built["owners"]
    if owners is None or isinstance(owners, str):
        raise ValueError("perpendicular_reference_rhs never evaluates a full grid: pass an explicit owner set")
    owner_arr = np.asarray(list(owners), dtype=np.int64)
    if len(np.unique(owner_arr)) != len(owner_arr):
        raise ValueError("owner ids must be distinct")
    t = env.t
    census = env.census
    if built is not None:
        raw_ids = np.asarray(built["raw_ids"], dtype=np.int64)
        p07_rows = np.asarray(built["p07_row_indices"], dtype=np.int64)
    else:
        raw_ids = np.flatnonzero(np.isin(t.ro, owner_arr)).astype(np.int64)
        incident = oc.incident_census_rows(census, [int(o) for o in owner_arr])
        face_rows = np.intersect1d(incident, ru.face_row_selection(census))
        p07_rows = np.intersect1d(face_rows, stencil_builder.p07_row_selection(census))
    n = int(t.n)
    raw_keys = np.array(np.unravel_index(raw_ids, (n, n, n))).T.astype(np.int64)
    return OwnerSupport(
        owners=owner_arr, raw_ids=raw_ids, raw_owner=np.asarray(t.ro[raw_ids], dtype=np.int64),
        raw_volume=np.asarray(t.rv[raw_ids], dtype=np.float64), points=np.asarray(t.pts[raw_ids], dtype=np.float64),
        raw_keys=raw_keys, p07_rows=p07_rows, p07_keys=census.keys()[p07_rows],
        p07_family=np.asarray(census.family[p07_rows]), p07_lower=np.asarray(census.owner_lo[p07_rows]),
        p07_upper=np.asarray(census.owner_hi[p07_rows]), owner_volume=np.asarray(t.vol[owner_arr], dtype=np.float64))


def _gather(pair, owners: np.ndarray) -> np.ndarray:
    """Read a sparse ``(unique_owner_ids, values)`` scatter at ``owners`` (absent owners read exactly 0)."""
    uniq, values = pair
    uniq = np.asarray(uniq, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros((len(owners),) + values.shape[1:], dtype=np.float64)
    if len(uniq) == 0:
        return out
    pos = np.minimum(np.searchsorted(uniq, owners), len(uniq) - 1)
    hit = uniq[pos] == owners
    out[hit] = values[pos[hit]]
    return out


# ---------------------------------------------------------------------------
# State adapters.  A state adapter exposes
#   values_gradients(points) -> (values (5, Q), gradients (5, Q, 3))      [slots n, Te, Ti, omega, phi]
# ---------------------------------------------------------------------------
class P06NState:
    """A P06N catalogue variant as a manufactured 5-field state (exact values, gradients, Hessians).

    ``variant`` is any ``tables.variant_names`` entry (``main_phi_neumann``, ``main_phi_dirichlet``,
    ``heldout_*``, ``dirichlet_rich``, ``control_*`` and their ``:D`` counterparts); the boundary-kind
    suffix only selects the field for the ``phi`` slot -- the exact fields themselves are kind-free."""

    def __init__(self, adapter: "cf.P06NAdapter", variant: str = "main_phi_neumann"):
        self.adapter = adapter
        self.variant = variant
        self.fields = tuple(name for name, _bc in adapter.variant_spec[variant])

    def _evaluate(self, points):
        pts = np.asarray(points, dtype=np.float64)
        return [self.adapter.evaluate_exact(pts, name) for name in self.fields]

    def values_gradients(self, points):
        ev = self._evaluate(points)
        return np.stack([e[0] for e in ev]), np.stack([e[1] for e in ev])

    def values_gradients_hessians(self, points):
        ev = self._evaluate(points)
        return np.stack([e[0] for e in ev]), np.stack([e[1] for e in ev]), np.stack([e[2] for e in ev])


def p06n_state(env, variant: str = "main_phi_neumann") -> P06NState:
    """The P06N variant ``variant`` on ``env``'s reference (the adapter never reads owner values)."""
    return P06NState(cf.P06NAdapter(env.ref, env.t.g.eta_period, None), variant)


# ---------------------------------------------------------------------------
# The three campaign-style owner references.
# ---------------------------------------------------------------------------
def bracket_reference(env, support: OwnerSupport, gradients: np.ndarray, pairs: Sequence) -> np.ndarray:
    """P05N ``raw_R``: ``(n_owners, len(pairs))`` = raw-volume-weighted owner mean of
    ``point_bracket(h, |J|, grad a, grad b) = -[a, b]`` at the raw midpoints.  ``gradients`` is the exact
    ``(Q, 3, F)`` gradient array at ``support.points``; ``pairs`` are column-index pairs ``(a, b)``."""
    from p05_direct_midpoint_global.direct_operator import point_bracket

    geo = support.raw_geometry(env)
    action = np.empty((len(support.raw_ids), len(pairs)))
    for col, (a, b) in enumerate(pairs):
        action[:, col] = point_bracket(geo["h"], geo["jac"], gradients[:, :, a], gradients[:, :, b])
    pair = ru._sparse_scatter(action, support.raw_volume, support.raw_owner)
    return _gather(pair, support.owners) / support.owner_volume[:, None]


def curvature_reference(env, support: OwnerSupport, values: np.ndarray, gradients: np.ndarray,
                        tau: float = 1.0) -> dict:
    """P06N ``raw_R_{material,remainder,total}``: ``{term: (n_owners, 4)}`` evolution-weighted owner means
    of ``_continuum_terms`` at the raw midpoints.  ``values`` is ``(5, Q)`` and ``gradients`` ``(5, Q, 3)``
    in the order ``n, Te, Ti, omega, phi``."""
    import p06_structured_global.numerics as p06numerics

    geo = support.raw_geometry(env)
    with _patched_tau(tau):
        material, remainder, total = p06numerics._continuum_terms(values, gradients, geo["prepared"])[:3]
    weight = geo["evolution_weight"]
    _u, denom = ru._sparse_scatter(weight[:, None], None, support.raw_owner)
    denominator = np.maximum(_gather((_u, denom[:, 0]), support.owners), 1e-300)
    out = {}
    for label, arr in (("material", material), ("remainder", remainder), ("total", total)):
        pair = ru._sparse_scatter(arr, weight, support.raw_owner)
        out[label] = _gather(pair, support.owners) / denominator[:, None]
    return out


def diffusion_reference(env, support: OwnerSupport, exact_gradients: Callable[[np.ndarray], np.ndarray],
                        *, positive_operator: bool = False) -> np.ndarray:
    """P07N ``O_q3`` construction: ``(n_owners, F)``, the exact gradient contracted with the q3
    weighted-tensor integrand per census face, scattered over the owner volume.

    **Sign.**  The frozen P07/P07N action (and production's ``_positive_polarization_action``) is the
    *positive* operator ``-div(P_perp grad f)``: it scatters the face flux "lower owner -, upper owner +"
    (``owner_lo`` is the cell *below* the face, so this is minus the net outflow).  With
    ``positive_operator=True`` this function returns exactly that oracle convention (``O_q3``); by
    default it returns the production diffusion operator ``+div(P_perp grad f)`` (bitwise ``-O_q3``),
    which is what ``D_f div(P_perp grad f)`` needs.  ``exact_gradients(points (Q, 3)) -> (Q, F, 3)`` is
    never called at a family-0 (collapsed ``r = 0``) face."""
    keys, family = support.p07_keys, support.p07_family
    non_collapsed = np.flatnonzero(family != 0)
    if non_collapsed.size == 0:
        probe = np.asarray(exact_gradients(support.points[:1]))
        return np.zeros((len(support.owners), probe.shape[1]))
    sel_keys = keys[non_collapsed]
    points, weight = pshared_provider._quadrature(env.t.faces, sel_keys, 3, face=True)
    tensor = env.ref._perpendicular_flux_tensor(points.reshape(-1, 3)).reshape(len(non_collapsed), 9, 3, 3)
    integrand = contract_face_tensor(weight, tensor, sel_keys[:, 0])
    grads = np.asarray(exact_gradients(points.reshape(-1, 3)))
    grads = grads.reshape(len(non_collapsed), 9, grads.shape[1], 3)
    face_O = np.zeros((len(keys), grads.shape[2]))
    face_O[non_collapsed] = np.einsum("fqa,fqka->fk", integrand, grads)
    pair = ru._sparse_scatter_signed(face_O, support.p07_lower, support.p07_upper, -1.0, +1.0)
    o_q3 = _gather(pair, support.owners) / support.owner_volume[:, None]
    return o_q3 if positive_operator else -o_q3


# ---------------------------------------------------------------------------
# The combined reference.
# ---------------------------------------------------------------------------
def reference_rhs(env, owners, state, params: ReferenceParams, fields: Sequence[str] = FIELDS,
                  *, terms: Sequence[str] = TERMS, support: Optional[OwnerSupport] = None) -> dict:
    """``{field: {term: (n_owners,) array}}`` at ``owners`` (order preserved).

    Per field: ``poisson_bracket`` (``-[phi, g]/rho_star``), ``curvature`` (material + remainder) with
    ``curvature_material`` / ``curvature_remainder`` alongside, ``perpendicular_diffusion``
    (``D_f div(P_perp grad f)``), and ``total`` = the sum of the requested main terms.  ``state`` is a
    state adapter (:class:`P06NState`)."""
    support = support if support is not None else owner_support(env, owners)
    index = {name: i for i, name in enumerate(FIELDS)}
    for name in fields:
        if name not in index:
            raise KeyError(f"unknown field {name!r}; fields are {FIELDS}")
    result = {name: {} for name in fields}
    need_raw = "poisson_bracket" in terms or "curvature" in terms
    if need_raw:
        values, gradients = state.values_gradients(support.points)
    if "poisson_bracket" in terms:
        g_all = np.moveaxis(np.asarray(gradients), 0, -1)               # (Q, 3, 5)
        pairs = [(_PHI, index[name]) for name in fields]
        bracket = bracket_reference(env, support, g_all, pairs) / float(params.rho_star)
        for col, name in enumerate(fields):
            result[name]["poisson_bracket"] = bracket[:, col]
    if "curvature" in terms:
        curv = curvature_reference(env, support, values, gradients, params.tau)
        for name in fields:
            i = index[name]
            result[name]["curvature"] = curv["total"][:, i]
            result[name]["curvature_material"] = curv["material"][:, i]
            result[name]["curvature_remainder"] = curv["remainder"][:, i]
    if "perpendicular_diffusion" in terms:
        def gradients_fn(pts):
            _v, g = state.values_gradients(pts)
            return np.moveaxis(np.asarray(g)[:4], 0, 1)                    # (Q, 4, 3)

        diff = diffusion_reference(env, support, gradients_fn)
        for name in fields:
            result[name]["perpendicular_diffusion"] = params.D(name) * diff[:, index[name]]
    for name in fields:
        parts = [result[name][t_] for t_ in TERMS if t_ in result[name]]
        result[name]["total"] = np.sum(parts, axis=0) if parts else np.zeros(len(support.owners))
    return result


# ---------------------------------------------------------------------------
# Independent pointwise check of the production formulas (design section 1).
# ---------------------------------------------------------------------------
def production_formula_pointwise(env, points: np.ndarray, values: np.ndarray, gradients: np.ndarray,
                                 params: ReferenceParams) -> dict:
    """The bracket and curvature terms at ``points`` written *directly from design section 1*, using only
    the reference metric/B/K and production's ``curvature_principal_matrix`` (no campaign helper):

    * ``poisson_bracket[f] = -(b_cov . (grad phi x grad g)) / (J B) / rho_star`` (signed ``J``);
    * ``curvature[f]`` = ``(1/B) (M(q) . (K.grad q))_f + coeff_f K.grad(phi + tau Ti)``,
      ``coeff = (-2n, -4Te/3, -4Ti/3, 0)/B``;
    * ``curvature_reduced[f]``: the continuum-reduced rows of design section 1 (a third, M-free
      derivation): ``n: (2/B)[C(n Te) - n C(phi)]``, ``Te: (4Te/3B)[C(p_e)/n + 5/2 C(Te) - C(phi)]``,
      ``Ti: (4Ti/3B)[C(p_e)/n - C(phi)] - (10 tau Ti/3B) C(Ti)``, ``omega: (2B/n) C(n (Te + tau Ti))``.

    ``values`` is ``(5, Q)``, ``gradients`` ``(5, Q, 3)`` (n, Te, Ti, omega, phi).  Returns
    ``{term: {field: (Q,) array}}``."""
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    metric = env.ref._metric(points)
    J = np.asarray(metric["J"], dtype=np.float64)
    B = np.asarray(metric["B"], dtype=np.float64)
    bcov = np.asarray(metric["bcov"], dtype=np.float64)
    K = np.asarray(env.ref._curvature(points), dtype=np.float64)
    tau = float(params.tau)
    n, te, ti = values[0], values[1], values[2]

    def C(grad):                                    # C(f) = K . grad f
        return np.einsum("pd,pd->p", K, grad)

    out: dict = {"poisson_bracket": {}, "curvature": {}, "curvature_reduced": {}}
    for i, name in enumerate(FIELDS):
        cross = np.cross(gradients[_PHI], gradients[i])
        bracket = np.einsum("pd,pd->p", bcov, cross) / (J * B)            # [phi, g]
        out["poisson_bracket"][name] = -bracket / float(params.rho_star)

    matrix = np.asarray(curvature_principal_matrix(n, te, ti, B, tau))    # (Q, 4, 4)
    Cq = np.stack([C(gradients[i]) for i in range(4)], axis=-1)           # (Q, 4)
    material = np.matmul(matrix, Cq[..., None])[..., 0] / B[:, None]
    C_psi = C(gradients[_PHI]) + tau * C(gradients[2])
    coeff = np.stack([-2.0 * n / B, -4.0 * te / (3.0 * B), -4.0 * ti / (3.0 * B), np.zeros_like(n)], axis=-1)
    total = material + coeff * C_psi[:, None]
    for i, name in enumerate(FIELDS):
        out["curvature"][name] = total[:, i]

    grad_pe = te[:, None] * gradients[0] + n[:, None] * gradients[1]      # grad(n Te)
    C_pe = C(grad_pe)
    C_n, C_te, C_ti, C_phi = C(gradients[0]), C(gradients[1]), C(gradients[2]), C(gradients[_PHI])
    grad_pt = (te + tau * ti)[:, None] * gradients[0] + n[:, None] * (gradients[1] + tau * gradients[2])
    reduced = np.stack([
        (2.0 / B) * (C_pe - n * C_phi),
        (4.0 * te / (3.0 * B)) * (C_pe / n + 2.5 * C_te - C_phi),
        (4.0 * ti / (3.0 * B)) * (C_pe / n - C_phi) - (10.0 * tau * ti / (3.0 * B)) * C_ti,
        (2.0 * B / n) * C(grad_pt),
    ], axis=-1)
    for i, name in enumerate(FIELDS):
        out["curvature_reduced"][name] = reduced[:, i]
    out["J_min"] = float(np.min(J))
    return out


def _rel(a: np.ndarray, b: np.ndarray, scale: Optional[float] = None) -> tuple[float, float]:
    """``(max |a - b|, max |a - b| / scale)``; ``scale`` defaults to ``max |b|``.  Callers pass the largest
    magnitude of the *term* over all fields, so a field whose term vanishes identically (e.g. the Te
    bracket of a state with ``grad phi = grad Te``) is not judged against its own roundoff-level scale."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if scale is None:
        scale = float(np.max(np.abs(b))) if b.size else 0.0
    return diff, (diff / scale if scale > 0.0 else (0.0 if diff == 0.0 else float("inf")))


def production_formula_check(env, owners, state, params: ReferenceParams,
                             support: Optional[OwnerSupport] = None) -> dict:
    """Prove sign/coefficient agreement between the design's production expressions and the
    campaign-style constructions of this module.

    Pointwise, at the raw midpoints of ``owners``: the independent production-formula evaluation
    (:func:`production_formula_pointwise`) versus (a) ``point_bracket`` (the P05N R construction) for the
    bracket, (b) ``_continuum_terms`` total (the P06N R construction) for the curvature, and (c) for the
    curvature, the continuum-reduced rows of design section 1.  Owner-level: the independent pointwise
    values projected with the same weights versus :func:`reference_rhs`.  Every entry is
    ``{"max_abs", "max_rel"}`` with ``max_rel = max|diff| / (largest |reference| of that term over the four
    fields)``; ``max_rel`` at the top
    level is the maximum over everything."""
    import p06_structured_global.numerics as p06numerics
    from p05_direct_midpoint_global.direct_operator import point_bracket

    support = support if support is not None else owner_support(env, owners)
    geo = support.raw_geometry(env)
    values, gradients = state.values_gradients(support.points)
    values = np.asarray(values); gradients = np.asarray(gradients)
    indep = production_formula_pointwise(env, support.points, values, gradients, params)

    report: dict = {"pointwise": {}, "owner_level": {}}
    worst = 0.0
    # campaign-style pointwise references
    with _patched_tau(params.tau):
        camp_total = p06numerics._continuum_terms(values, gradients, geo["prepared"])[2]
    camp_bracket = {name: point_bracket(geo["h"], geo["jac"], gradients[_PHI], gradients[i]) / float(params.rho_star)
                    for i, name in enumerate(FIELDS)}
    camp = {"poisson_bracket": camp_bracket, "curvature": {name: camp_total[:, i] for i, name in enumerate(FIELDS)}}
    # term-level scales (largest magnitude of the term over all four fields)
    scale = {term: max(float(np.max(np.abs(camp[term][name]))) for name in FIELDS) for term in camp}
    for name in FIELDS:
        for term in ("poisson_bracket", "curvature"):
            d, r = _rel(indep[term][name], camp[term][name], scale[term])
            report["pointwise"][f"{name}:{term}:vs_campaign"] = {"max_abs": d, "max_rel": r}
            worst = max(worst, r)
        d, r = _rel(indep["curvature_reduced"][name], indep["curvature"][name], scale["curvature"])
        report["pointwise"][f"{name}:curvature:reduced_vs_M"] = {"max_abs": d, "max_rel": r}
        worst = max(worst, r)
        d, r = _rel(indep["curvature_reduced"][name], camp["curvature"][name], scale["curvature"])
        report["pointwise"][f"{name}:curvature:reduced_vs_campaign"] = {"max_abs": d, "max_rel": r}
        worst = max(worst, r)

    # owner level: project the independent pointwise values with the campaign weights
    rhs = reference_rhs(env, support.owners, state, params, terms=("poisson_bracket", "curvature"), support=support)
    weight = geo["evolution_weight"]
    _u, denom = ru._sparse_scatter(weight[:, None], None, support.raw_owner)
    denominator = np.maximum(_gather((_u, denom[:, 0]), support.owners), 1e-300)
    for name in FIELDS:
        pair = ru._sparse_scatter(indep["poisson_bracket"][name][:, None], support.raw_volume, support.raw_owner)
        proj_b = _gather(pair, support.owners)[:, 0] / support.owner_volume
        pair = ru._sparse_scatter(indep["curvature"][name][:, None], weight, support.raw_owner)
        proj_c = _gather(pair, support.owners)[:, 0] / denominator
        for term, proj in (("poisson_bracket", proj_b), ("curvature", proj_c)):
            owner_scale = max(float(np.max(np.abs(rhs[f][term]))) for f in FIELDS)
            d, r = _rel(proj, rhs[name][term], owner_scale)
            report["owner_level"][f"{name}:{term}"] = {"max_abs": d, "max_rel": r}
            worst = max(worst, r)
    report["max_rel"] = worst
    report["J_min"] = indep["J_min"]
    report["n_points"] = int(len(support.points))
    return report


def diffusion_midpoint_check(env, owners, state: P06NState, params: ReferenceParams,
                             support: Optional[OwnerSupport] = None) -> dict:
    """Sign/coefficient sanity of the diffusion reference at *discretization* level (not roundoff): the
    owner mean, by raw volume, of the continuum pointwise ``J^-1 d_i[J P^ij d_j f]``
    (``ref._perpendicular_operator`` with the exact gradient and Hessian, i.e. ``+div(P_perp grad f)``) at
    the raw midpoints versus this module's face-flux diffusion reference.  Returns
    ``{field: {"max_abs", "max_rel", "sign_agreement"}}`` (``max_rel`` scaled by the largest magnitude of
    that field's reference; ``sign_agreement`` True when the pointwise mean has the same sign at every owner whose reference
    magnitude is at least a quarter of the field's largest one)."""
    support = support if support is not None else owner_support(env, owners)
    _v, g, H = state.values_gradients_hessians(support.points)
    unit = ReferenceParams(rho_star=1.0, tau=params.tau, diffusion={name: 1.0 for name in FIELDS})
    o_q3 = reference_rhs(env, support.owners, state, unit, terms=("perpendicular_diffusion",), support=support)
    out = {}
    for i, name in enumerate(FIELDS):
        pointwise = env.ref._perpendicular_operator(support.points, np.asarray(g[i]), np.asarray(H[i]))
        pair = ru._sparse_scatter(np.asarray(pointwise)[:, None], support.raw_volume, support.raw_owner)
        proj = _gather(pair, support.owners)[:, 0] / support.owner_volume
        ref_val = o_q3[name]["perpendicular_diffusion"]
        d, r = _rel(proj, ref_val)
        dominant = np.abs(ref_val) >= 0.25 * np.max(np.abs(ref_val))
        out[name] = {"max_abs": d, "max_rel": r,
                     "sign_agreement": bool(np.all(np.sign(proj[dominant]) == np.sign(ref_val[dominant])))}
    return out


# ---------------------------------------------------------------------------
# Machinery verification against the frozen campaigns' own saved R arrays.
# ---------------------------------------------------------------------------
def verify_against_oracles(env, owners, paths: dict, *, support: Optional[OwnerSupport] = None,
                           p06n_variants: Optional[Sequence[str]] = None) -> list[dict]:
    """Reproduce, at ``owners``, the frozen oracles' own R arrays with this module's constructions on the
    campaigns' *own* catalogue states: P06N ``R_material/R_remainder/R_total`` (every variant in
    ``p06n_variants``, default all), P07N ``O_q3`` and P05N ``R`` (frozen catalogue).  Each row is
    ``{"campaign", "term", "max_abs", "max_rel", "ratio_to_oracle_NR", "pass"}`` (``max_rel`` scaled by the
    largest saved value; the ratio is to the oracle's own archived ``|N - R|``, as
    :func:`p_shared.owner_closure.compare_to_oracle`)."""
    support = support if support is not None else owner_support(env, owners)
    n = int(env.t.n)
    owner_arr = support.owners
    rows: list[dict] = []

    def add(campaign, term, replay, saved, archived_error):
        row = oc._row(campaign, term, replay, saved, archived_error)
        scale = float(np.max(np.abs(saved))) if np.size(saved) else 0.0
        row["max_rel"] = row["max_abs"] / scale if scale > 0 else (0.0 if row["max_abs"] == 0 else 1e18)
        rows.append(row)

    # P06N ------------------------------------------------------------------------------------------
    import p06n_field_derived_global.core as p06n_core

    tables = p06n_core.CATALOGUE_TABLES
    variants = tuple(p06n_variants) if p06n_variants is not None else tables.variant_names
    root = paths["p05n_p06n_upwind"] / "p06n"
    with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
        saved = {label: z[label][:, owner_arr].copy() for label in
                 ("total", "R_material", "R_remainder", "R_total")}
    adapter = cf.P06NAdapter(env.ref, env.t.g.eta_period, None)
    for name in variants:
        vi = tables.variant_names.index(name)
        state = P06NState(adapter, name)
        values, gradients = state.values_gradients(support.points)
        got = curvature_reference(env, support, values, gradients, tau=p06n_core.TAU)
        archived = saved["total"][vi] - saved["R_total"][vi]
        for label, key in (("material", "R_material"), ("remainder", "R_remainder"), ("total", "R_total")):
            add("P06N", f"raw_{key}[{name}]", got[label], saved[key][vi], archived)

    # P07N ------------------------------------------------------------------------------------------
    p07n_dir = paths["p07n"]
    with np.load(p07n_dir / f"N{n}.global.npz", allow_pickle=False) as z:
        saved_O = z["O_q3"][owner_arr].copy(); saved_nmo = z["N_minus_O"][owner_arr].copy()
    p07n = cf.P07NAdapter(env.ref, env.t.g.eta_period, np.zeros((1, 1)))
    got_O = diffusion_reference(env, support, p07n.exact_gradients, positive_operator=True)
    add("P07N", "global_O_q3", got_O, saved_O, saved_nmo)

    # P05N (frozen catalogue) -----------------------------------------------------------------------
    root = paths["p05n_frozen"]
    with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
        saved_N = z["N"][owner_arr].copy(); saved_R = z["R"][owner_arr].copy()
    p05n = cf.P05NAdapter("p05n_frozen", env.ref, env.t.g.eta_period, np.zeros((1, 1)))
    grads = p05n.exact_gradient(support.points)
    got_R = bracket_reference(env, support, grads, p05n.r_pair_index)
    add("P05N", "raw_R", got_R, saved_R, saved_N - saved_R)
    return rows
