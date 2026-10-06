"""Combined perpendicular RHS of (n, Te, Ti, omega) on synthetic rows (P08 step 3, tasks 3.2-3.3; fast).

The bounded closure of the shared synthetic world (``tests.perpendicular_synthetic``: real census, random shaped
rows incl. conditioned rows, Neumann rows, a wall face with a missing side) is made thermodynamically admissible
(positive convex value weights, positive fields and trace) as in ``test_perpendicular_p06_operator``. G3.1: the
combined terms equal the separate operator calls (``p05_terms`` with pairs ``(phi, g)`` scaled by ``1/rho_star``,
``p06_action`` total + correction, ``-D * p07_action``, P07 being the positive operator -div(P_perp grad f)); G3.4: eager == jit (to roundoff), JVP vs central finite difference.
Owner arrays are compared at the closure owners (the owners whose raw cells are all in the bounded world; the
others carry a partial q1 evolution volume).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_p05_operator import p05_terms
from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action, p06_action_from_state
from drbx.native.fci_perpendicular_p07_operator import p07_action
from drbx.native.fci_perpendicular_reconstruction_state import (
    boundary_data_from_callables, cell_state, face_state)
from drbx.native.fci_perpendicular_rhs import (
    FIELDS, PHI, PerpendicularParams, PerpendicularTerms, perpendicular_columns, perpendicular_rhs)
from tests.perpendicular_synthetic import Boundary, lower_world, make_world

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
NPHYS = 8
D, N = "dirichlet", "neumann"
RHO, DIFF = 0.7, {"density": 1.1, "Te": 0.3, "Ti": 0.5, "vorticity": 2.0}
KINDS = dict(density=N, Te=N, Ti=D, vorticity=D, phi=D)
PRODUCTION = ("poisson_bracket", "curvature", "perpendicular_diffusion")


class PositiveBoundary(Boundary):
    """Trace ``1 + 0.3 sin(phase)`` (positive), physical-normal data from its gradient."""

    def dirichlet(self, points):
        v, g = super().dirichlet(points)
        return 1.0 + 0.3 * v, 0.3 * g


def _admissible(world):
    for key, row in list(world.row_index.items()):
        if hasattr(row, "value") and getattr(row, "value").ndim == 2 and hasattr(row, "gradient"):
            a = np.abs(row.value)
            world.row_index[key] = dataclasses.replace(row, value=a / a.sum(axis=-1, keepdims=True))
    for key, row in list(world.neumann_index.items()):
        a = np.abs(row.value)
        world.neumann_index[key] = dataclasses.replace(row, value=a / a.sum(), boundary_value=0.02 * row.boundary_value)
    return world


@pytest.fixture(scope="module")
def world():
    return _admissible(make_world(owners=OWNERS))


@pytest.fixture(scope="module")
def plan(world):
    return lower_world(world)


@pytest.fixture(scope="module")
def owner_fields(world):
    return 1.0 + 0.4 * np.random.default_rng(5).uniform(size=(world.n_owners, NPHYS))


@pytest.fixture(scope="module")
def bc_all(plan):
    b = PositiveBoundary(NPHYS)
    return boundary_data_from_callables(plan, b.dirichlet, b.normal)


@pytest.fixture(scope="module")
def bc5(bc_all):
    return bc_columns(bc_all, np.arange(5))                       # columns (density, Te, Ti, vorticity, phi)


@pytest.fixture(scope="module")
def state(owner_fields):
    return {name: owner_fields[:, i] for i, name in enumerate(FIELDS)}


@pytest.fixture(scope="module")
def phi(owner_fields):
    return owner_fields[:, 4]


@pytest.fixture(scope="module")
def params():
    return PerpendicularParams(rho_star=RHO, tau=1.0, diffusion=dict(DIFF))


@pytest.fixture(scope="module")
def rhs(plan, state, phi, bc5, params):
    return perpendicular_rhs(plan, state, phi, bc5, KINDS, params)


@pytest.fixture(scope="module")
def separate(plan, owner_fields, bc5, params):
    """The separate operator calls with the same coefficients (5 columns in the combined layout)."""
    f = owner_fields[:, :5]
    kinds = tuple(KINDS[c] for c in (*FIELDS, PHI))
    t = p05_terms(plan, f, bc5, kinds, [(4, i) for i in range(4)])
    a = p06_action(plan, f, bc5, kinds)
    p7 = p07_action(plan, f[:, :4], bc_columns(bc5, np.arange(4)), kinds[:4])
    d = np.asarray([DIFF[n] for n in FIELDS])
    return dict(bracket=(t.centered_owner + t.jump_owner) / RHO, centered=t.centered_owner / RHO,
                jump=t.jump_owner / RHO, q1=a.total, correction=a.correction, material=a.material,
                remainder=a.remainder, curvature=a.total + a.correction, diffusion=-d[None] * p7,
                p05=t, p06=a, kinds=kinds)


def _max_diff(a, b, rows=OWNERS):
    return float(np.max(np.abs(np.asarray(a)[rows] - np.asarray(b)[rows])))


def _scale(a, rows=OWNERS):
    return float(np.max(np.abs(np.asarray(a)[rows])))


def test_columns_layout_and_names():
    assert perpendicular_columns() == (*FIELDS, PHI)
    assert perpendicular_columns(("Te",), ("bracket",)) == ("Te", PHI)
    assert perpendicular_columns(("Te",), ("curvature",)) == ("Te", "density", "Ti", "vorticity", PHI)
    assert perpendicular_columns(("Ti",), ("bracket",), [("Ti", "extra"), ("phi", "Ti")]) == ("Ti", "extra", PHI)


def test_output_structure_uses_production_term_names(rhs, world):
    assert isinstance(rhs, PerpendicularTerms)
    assert set(rhs.terms) == set(FIELDS) == set(rhs.total)
    for f in FIELDS:
        assert tuple(rhs.terms[f]) == PRODUCTION
        assert rhs.total[f].shape == (world.n_owners,)
        # the sum is fused with the terms inside the jit (FMA contraction): equal to a few ulp, not always bitwise
        terms = rhs.terms[f]
        by_hand = terms["poisson_bracket"] + terms["curvature"] + terms["perpendicular_diffusion"]
        assert _max_diff(rhs.total[f], by_hand) <= 1e-14 * _scale(by_hand)
        d = rhs.detail[f]
        assert _max_diff(d["bracket_centered"] + d["bracket_jump"], terms["poisson_bracket"]) <= 1e-14 * _scale(
            terms["poisson_bracket"])
        assert _max_diff(d["curvature_q1"] + d["curvature_correction"], terms["curvature"]) <= 1e-14 * _scale(
            terms["curvature"])
    assert rhs.raw_pairs is None
    assert set(rhs.diagnostics) == {"antisymmetry", "spectral_fallback", "floor_hits", "wall_fallback"}
    # the synthetic world exercises every term, incl. a non-trivial q3 correction
    assert min(_scale(rhs.terms[f][t]) for f in FIELDS for t in PRODUCTION) > 1e-3
    assert min(_scale(rhs.detail[f]["curvature_correction"]) for f in FIELDS) > 1e-6


def test_g31_consistency_with_the_separate_operator_calls(rhs, separate):
    """G3.1: every term of every field equals the separate operator call (bitwise on this platform for the three
    terms; the split ``detail`` arrays may differ by an ulp where XLA fuses the division by ``rho_star``)."""
    term_diffs, detail_diffs = {}, {}
    for i, f in enumerate(FIELDS):
        for key, ref in (("poisson_bracket", separate["bracket"]), ("curvature", separate["curvature"]),
                         ("perpendicular_diffusion", separate["diffusion"])):
            term_diffs[(f, key)] = _max_diff(rhs.terms[f][key], np.asarray(ref)[:, i])
        for key, ref in (("bracket_centered", separate["centered"]), ("bracket_jump", separate["jump"]),
                         ("curvature_material", separate["material"]), ("curvature_remainder", separate["remainder"]),
                         ("curvature_q1", separate["q1"]), ("curvature_correction", separate["correction"])):
            detail_diffs[(f, key)] = _max_diff(rhs.detail[f][key], np.asarray(ref)[:, i])
    scale = max(_scale(rhs.terms[f][t]) for f in FIELDS for t in PRODUCTION)
    print("\nG3.1 synthetic: max |combined - separate|, terms:", max(term_diffs.values()),
          "details:", max(detail_diffs.values()), "(term scale %.3g)" % scale)
    assert max(term_diffs.values()) <= 1e-14 * scale, term_diffs
    assert max(detail_diffs.values()) <= 1e-14 * scale, detail_diffs


def test_terms_are_selectable_and_fields_are_a_subset(plan, state, phi, bc5, params, rhs, separate):
    """Term / field subsets agree with the full call (the shared state is built from the requested columns)."""
    sub = perpendicular_rhs(plan, state, phi, bc5, KINDS, params, fields=("density", "Te", "Ti", "vorticity"),
                            terms=("bracket",))
    for f in FIELDS:
        assert tuple(sub.terms[f]) == ("poisson_bracket",)
        assert _max_diff(sub.terms[f]["poisson_bracket"], rhs.terms[f]["poisson_bracket"], slice(None)) <= 1e-14
        assert set(sub.detail[f]) == {"bracket_centered", "bracket_jump"}
        assert _max_diff(sub.total[f], sub.terms[f]["poisson_bracket"], slice(None)) == 0.0
    diff_only = perpendicular_rhs(plan, {k: state[k] for k in ("Ti", "vorticity")}, phi,
                                  bc_columns(bc5, [2, 3, 4]), {"Ti": D, "vorticity": D},
                                  params, fields=("Ti", "vorticity"), terms=("diffusion",))
    assert set(diff_only.terms) == {"Ti", "vorticity"}
    # reorder: Ti first, then vorticity; the diffusion columns are the leading state columns
    for name in ("Ti", "vorticity"):
        i = FIELDS.index(name)
        assert _max_diff(diff_only.terms[name]["perpendicular_diffusion"], np.asarray(separate["diffusion"])[:, i],
                         slice(None)) <= 1e-13 * _scale(separate["diffusion"][:, i], slice(None))
    layout = perpendicular_columns(("Te",), ("curvature",))         # (Te, density, Ti, vorticity, phi)
    bc_te = bc_columns(bc5, [(*FIELDS, PHI).index(c) for c in layout])
    curv = perpendicular_rhs(plan, state, phi, bc_te, KINDS, params, fields=("Te",), terms=("curvature",))
    assert _max_diff(curv.terms["Te"]["curvature"], np.asarray(separate["curvature"])[:, 1]) <= 1e-13 * _scale(
        separate["curvature"][:, 1])
    assert set(curv.terms["Te"]) == {"curvature"} and "bracket_jump" not in curv.detail["Te"]


def test_raw_pairs_reach_arbitrary_pairs_of_the_shared_state(plan, state, phi, bc5, bc_all, params, owner_fields,
                                                             separate):
    pairs = [("density", "Te"), ("phi", "Ti"), ("vorticity", "density")]
    out = perpendicular_rhs(plan, state, phi, bc5, KINDS, params, terms=("bracket",), raw_pairs=pairs)
    idx = [(0, 1), (4, 2), (3, 0)]
    ref = p05_terms(plan, owner_fields[:, :5], bc5, separate["kinds"], idx)
    raw = out.raw_pairs
    assert raw.centered_owner.shape == (len(phi), 3) and raw.face_jump.shape[1] == 3
    for name in ("centered_owner", "jump_owner", "centered_numerator", "jump_numerator", "face_jump"):
        assert _max_diff(getattr(raw, name), getattr(ref, name), slice(None)) <= 1e-14 * max(
            1.0, _scale(getattr(ref, name), slice(None))), name
    # the (phi, g) bracket columns are unaffected by the extra pairs
    for f in FIELDS:
        np.testing.assert_allclose(np.asarray(out.terms[f]["poisson_bracket"]),
                                   np.asarray(perpendicular_rhs(plan, state, phi, bc5, KINDS, params,
                                                                terms=("bracket",)).terms[f]["poisson_bracket"]),
                                   rtol=0, atol=1e-14)
    # raw pairs alone (no terms), with a column that is not a step-3 field
    extra = dict(state, extra=np.asarray(owner_fields[:, 5]))
    bc3 = bc_columns(bc_all, [0, 5, 4])                       # layout (density, extra, phi)
    only = perpendicular_rhs(plan, extra, phi, bc3, {"extra": N}, params, fields=("density",), terms=(),
                             raw_pairs=[("phi", "extra")])
    assert only.terms == {"density": {}} and only.total == {}
    ref2 = p05_terms(plan, owner_fields[:, [0, 5, 4]], bc3, (D, N, D), [(2, 1)])
    np.testing.assert_allclose(np.asarray(only.raw_pairs.centered_owner), np.asarray(ref2.centered_owner), atol=1e-14)


@pytest.mark.parametrize("kinds", [tuple(KINDS[c] for c in (*FIELDS, PHI)), (N,) * 5, (D,) * 5, (D, N, N, D, N)])
@pytest.mark.parametrize("value_columns, gradient_columns", [
    ((0, 1, 2, 3), (4,)), ((0, 1, 2, 3), None), ((4, 1), (1, 4)), ((3, 0, 1), (0, 2, 3)), ((2,), (2,)), ((0, 1, 3), ())])
def test_pruned_face_state_equals_the_full_state_columns(plan, owner_fields, bc5, kinds, value_columns,
                                                          gradient_columns):
    """``face_state(value_columns=, gradient_columns=)`` holds the full call's columns, in the order given (roundoff:
    the contraction kernels depend on the column count)."""
    f = jnp.asarray(owner_fields[:, :5])
    full = face_state(plan, f, bc5, kinds)
    gradients = gradient_columns != ()
    pruned = face_state(plan, f, bc5, kinds, gradients=gradients, value_columns=value_columns,
                        gradient_columns=gradient_columns if gradients else None)
    vcols = list(value_columns)
    gcols = list(range(5)) if gradient_columns is None else list(gradient_columns)
    for name in ("value", "lower", "upper"):
        a, b = np.asarray(getattr(pruned, name)), np.asarray(getattr(full, name))[..., vcols]
        assert a.shape == b.shape
        assert np.max(np.abs(a - b)) <= 1e-14 * np.max(np.abs(b))
    if gradients:
        a, b = np.asarray(pruned.gradient), np.asarray(full.gradient)[..., gcols]
        assert a.shape == b.shape
        assert np.max(np.abs(a - b)) <= 1e-14 * np.max(np.abs(b))
    else:
        assert pruned.gradient is None


def test_pruned_face_state_validates_columns(plan, owner_fields, bc5):
    f = jnp.asarray(owner_fields[:, :5])
    kinds = tuple(KINDS[c] for c in (*FIELDS, PHI))
    for kw in (dict(value_columns=()), dict(value_columns=(5,)), dict(gradient_columns=(-1,)),
               dict(gradients=False, gradient_columns=(0,))):
        with pytest.raises(ValueError):
            face_state(plan, f, bc5, kinds, **kw)


def test_params_scale_the_terms_they_own(plan, state, phi, bc5, params, rhs):
    half = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, rho_star=2 * RHO))
    doubled = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(
        params, diffusion={k: 2 * v for k, v in DIFF.items()}))
    for f in FIELDS:
        np.testing.assert_array_equal(np.asarray(half.terms[f]["poisson_bracket"]) * 2,
                                      np.asarray(rhs.terms[f]["poisson_bracket"]))
        np.testing.assert_array_equal(np.asarray(half.terms[f]["curvature"]), np.asarray(rhs.terms[f]["curvature"]))
        np.testing.assert_array_equal(np.asarray(doubled.terms[f]["perpendicular_diffusion"]),
                                      2 * np.asarray(rhs.terms[f]["perpendicular_diffusion"]))
    tau2 = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, tau=2.0), terms=("curvature",))
    assert _max_diff(tau2.terms["Ti"]["curvature"], rhs.terms["Ti"]["curvature"]) > 1e-3


def test_p06_from_state_equals_p06_action_bitwise(plan, owner_fields, bc5, separate):
    kinds = separate["kinds"]
    f = jnp.asarray(owner_fields[:, :5])
    cs = cell_state(plan, f, bc5, kinds)
    fs = face_state(plan, f, bc5, kinds, gradients=False)
    action, counters = p06_action_from_state(plan, cs.value, cs.gradient, fs.value, fs.lower, fs.upper,
                                             return_counters=True)
    ref = p06_action(plan, f, bc5, kinds)
    for a, b in zip(action, ref):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert len(counters) == 3
    groups = np.asarray([[0, 1, 2, 3, 4], [4, 1, 2, 3, 0]])
    multiplier = np.where(np.arange(len(plan.faces.census_row)) % 4 == 1, 2.0, 1.0)
    a2 = p06_action_from_state(plan, cs.value, cs.gradient, fs.value, fs.lower, fs.upper, groups,
                               face_multiplier=multiplier)
    r2 = p06_action(plan, f, bc5, kinds, groups, face_multiplier=multiplier)
    for a, b in zip(a2, r2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_g34_eager_equals_jit(plan, state, phi, bc5, params, rhs):
    def call(p, s, ph, b, prm):
        return perpendicular_rhs(p, s, ph, b, KINDS, prm, raw_pairs=[("density", "Te")])

    jitted = jax.jit(call)(plan, {k: jnp.asarray(v) for k, v in state.items()}, jnp.asarray(phi), bc5, params)
    eager = call(plan, state, phi, bc5, params)
    la, lb = jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(jitted)
    assert len(la) == len(lb) > 20
    for a, b in zip(la, lb):
        # Roundoff, not bitwise, since the face state is pruned to the columns P05 / P06 read (P09 fix 1): with a
        # second generator column (the extra pair) XLA fuses the pruned face producers into the P05 jump
        # differently in a standalone call and inside an outer jit (observed: <= 1 ulp of the bracket jump).
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14 * max(float(np.max(np.abs(b))), 1e-300))
    for f in FIELDS:                                                # the extra pair does not change the columns
        assert _max_diff(eager.terms[f]["poisson_bracket"], rhs.terms[f]["poisson_bracket"], slice(None)) <= 1e-14


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def test_g34_jvp_matches_central_finite_difference(plan, state, phi, bc5, params):
    rng = np.random.default_rng(9)
    x = jnp.stack([jnp.asarray(state[f]) for f in FIELDS] + [jnp.asarray(phi)], axis=1)
    tangent = jnp.asarray(rng.normal(size=x.shape))
    rows = np.asarray(OWNERS)

    def fun(z):
        out = perpendicular_rhs(plan, {f: z[:, i] for i, f in enumerate(FIELDS)}, z[:, 4], bc5, KINDS, params)
        return {f: {t: a[rows] for t, a in out.terms[f].items()} for f in FIELDS}

    _, jvp = jax.jvp(fun, (x,), (tangent,))
    h = 1e-6
    plus, minus = fun(x + h * tangent), fun(x - h * tangent)
    errors = {}
    for f in FIELDS:
        for t in PRODUCTION:
            fd = (plus[f][t] - minus[f][t]) / (2 * h)
            errors[(f, t)] = _rel(jvp[f][t], fd)
            assert float(jnp.max(jnp.abs(jvp[f][t]))) > 0
    print("\nG3.4 synthetic JVP vs central FD (h = 1e-6), max relative error:", max(errors.values()))
    assert max(errors.values()) <= 1e-6, errors


def test_gradient_wrt_params(plan, state, phi, bc5, params):
    """The params pytree is differentiable: bracket ~ 1/rho_star, diffusion linear in D."""
    def bracket_sum(rho):
        return perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, rho_star=rho),
                                 terms=("bracket",)).terms["Te"]["poisson_bracket"].sum()
    g = jax.grad(bracket_sum)(0.7)
    assert abs(float(g) + float(bracket_sum(0.7)) / 0.7) <= 1e-10 * abs(float(g))


def test_input_errors(plan, state, phi, bc5, params):
    with pytest.raises(ValueError, match="out of scope"):
        perpendicular_rhs(plan, state, phi, bc5, KINDS, params, fields=("density", "Vi"))
    with pytest.raises(ValueError, match="terms"):
        perpendicular_rhs(plan, state, phi, bc5, KINDS, params, terms=("bracket", "polarization"))
    with pytest.raises(ValueError, match="nothing to compute"):
        perpendicular_rhs(plan, state, phi, bc5, KINDS, params, terms=())
    with pytest.raises(KeyError, match="vorticity"):
        perpendicular_rhs(plan, {k: v for k, v in state.items() if k != "vorticity"}, phi, bc5, KINDS, params)
    with pytest.raises(ValueError, match="no coefficient"):
        perpendicular_rhs(plan, state, phi, bc5, KINDS, PerpendicularParams(rho_star=1.0, diffusion={"Te": 1.0}))
    with pytest.raises(ValueError, match="not in the layout"):
        perpendicular_rhs(plan, state, phi, bc5, {"Vi": D}, params)
    with pytest.raises(ValueError, match="cells and faces"):
        perpendicular_rhs(lower_world_p07_only(), state, phi, bc5, KINDS, params)


def lower_world_p07_only():
    return lower_world(_admissible(make_world(owners=OWNERS)), include=("p07",))


@pytest.mark.parametrize("method", ("closed_form",))
def test_absolute_method_selector_reaches_the_curvature_and_keeps_the_default_bitwise(
        plan, state, phi, bc5, params, rhs, owner_fields, method):
    # the default is "closed_form"; "lapack4" (the campaign's 4x4 eig) is the reference the others must match
    default = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, absolute_method="closed_form"))
    for f in FIELDS:
        assert np.array_equal(np.asarray(rhs.total[f]), np.asarray(default.total[f]), equal_nan=True)
    ref = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, absolute_method="lapack4"))
    other = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, absolute_method=method))
    for f in FIELDS:
        assert _max_diff(other.total[f], ref.total[f]) <= 1e-12 * _scale(ref.total[f])
        assert _max_diff(other.detail[f]["curvature_correction"], ref.detail[f]["curvature_correction"]) \
            <= 1e-12 * max(_scale(ref.detail[f]["curvature_correction"]), 1e-300)
    assert int(other.diagnostics["spectral_fallback"]) == int(ref.diagnostics["spectral_fallback"]) == 0
    # the operator functions take the same static option, with the same default
    kinds = tuple(KINDS[c] for c in (*FIELDS, PHI))
    a = p06_action(plan, owner_fields[:, :5], bc5, kinds, absolute_method="lapack4")
    b = p06_action(plan, owner_fields[:, :5], bc5, kinds, absolute_method=method)
    assert _max_diff(b.correction, a.correction) <= 1e-12 * _scale(a.correction)
    assert np.array_equal(np.asarray(p06_action(plan, owner_fields[:, :5], bc5, kinds).correction),
                          np.asarray(p06_action(plan, owner_fields[:, :5], bc5, kinds,
                                                absolute_method="closed_form").correction))


def test_absolute_method_is_validated(plan, state, phi, bc5, params):
    with pytest.raises(ValueError, match="absolute_method"):
        perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, absolute_method="svd"))


# ----------------------------------------------------------------------------------------------- rho_star convention
def _all_outputs(res):
    """Every owner array of a result (terms, totals, details) keyed by a path."""
    out = {}
    for f in FIELDS:
        out.update({(f, "terms", k): np.asarray(v) for k, v in res.terms[f].items()})
        out[(f, "total")] = np.asarray(res.total[f])
        out.update({(f, "detail", k): np.asarray(v) for k, v in res.detail[f].items()})
    return out


def test_rho_star_convention_default_is_legacy_and_bitwise(plan, state, phi, bc5, params, rhs):
    assert PerpendicularParams().rho_star_convention == "legacy-bracket-only"
    assert params.rho_star_convention == "legacy-bracket-only"
    explicit = perpendicular_rhs(plan, state, phi, bc5, KINDS,
                                 dataclasses.replace(params, rho_star_convention="legacy-bracket-only"))
    ref, got = _all_outputs(rhs), _all_outputs(explicit)
    assert ref.keys() == got.keys()
    for key in ref:
        assert np.array_equal(ref[key], got[key], equal_nan=True), key


def test_single_length_equals_legacy_bitwise_at_rho_star_one(plan, state, phi, bc5, params):
    legacy = perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, rho_star=1.0))
    single = perpendicular_rhs(plan, state, phi, bc5, KINDS,
                               dataclasses.replace(params, rho_star=1.0, rho_star_convention="single-length"))
    ref, got = _all_outputs(legacy), _all_outputs(single)
    assert ref.keys() == got.keys()
    for key in ref:
        assert np.array_equal(ref[key], got[key], equal_nan=True), key
    assert min(_scale(got[(f, "terms", t)]) for f in FIELDS for t in PRODUCTION) > 1e-3


def test_single_length_homogeneity_in_rho_star(plan, state, phi, bc5, params):
    """Bracket and curvature outputs (and every split detail) are rho_star times those at rho_star = 1; the diffusion is
    unchanged. The legacy bracket is 1 / rho_star times, its curvature unchanged."""
    one = dataclasses.replace(params, rho_star=1.0, rho_star_convention="single-length")
    ref = _all_outputs(perpendicular_rhs(plan, state, phi, bc5, KINDS, one))
    for rho in (0.05, 0.7, 3.0):
        got = _all_outputs(perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(one, rho_star=rho)))
        legacy = _all_outputs(perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(
            one, rho_star=rho, rho_star_convention="legacy-bracket-only")))
        for f in FIELDS:
            for term, factor, lfactor in (("poisson_bracket", rho, 1.0 / rho), ("curvature", rho, 1.0),
                                          ("perpendicular_diffusion", 1.0, 1.0)):
                a, b, c = ref[(f, "terms", term)], got[(f, "terms", term)], legacy[(f, "terms", term)]
                assert _max_diff(b, factor * a) <= 1e-13 * _scale(a) * max(factor, 1.0), (f, term, rho)
                assert _max_diff(c, lfactor * a) <= 1e-13 * _scale(a) * max(lfactor, 1.0), (f, term, rho, "legacy")
            for key in ("bracket_centered", "bracket_jump", "curvature_material", "curvature_remainder", "curvature_q1",
                        "curvature_correction"):
                factor = rho
                a, b = ref[(f, "detail", key)], got[(f, "detail", key)]
                assert _max_diff(b, factor * a) <= 1e-13 * max(_scale(a), 1e-300) * max(factor, 1.0), (f, key, rho)
            d = got[(f, "detail", "curvature_q1")] + got[(f, "detail", "curvature_correction")]
            assert _max_diff(d, got[(f, "terms", "curvature")]) <= 1e-13 * _scale(got[(f, "terms", "curvature")])
            tot = sum(got[(f, "terms", t)] for t in PRODUCTION)
            assert _max_diff(got[(f, "total")], tot) <= 1e-13 * max(_scale(tot), 1.0)
    # raw pairs are unscaled in either convention
    pair = [("phi", "Te")]
    cols = perpendicular_columns(("Te",), ("bracket",), pair)
    bc_pair = bc_columns(bc5, [(*FIELDS, PHI).index(c) for c in cols])
    kinds = {c: KINDS[c] for c in cols}
    r1 = perpendicular_rhs(plan, state, phi, bc_pair, kinds, one, fields=("Te",), terms=("bracket",), raw_pairs=pair)
    r2 = perpendicular_rhs(plan, state, phi, bc_pair, kinds, dataclasses.replace(one, rho_star=0.3), fields=("Te",),
                           terms=("bracket",), raw_pairs=pair)
    assert np.array_equal(np.asarray(r1.raw_pairs.centered_owner), np.asarray(r2.raw_pairs.centered_owner))


def test_rho_star_convention_is_validated_and_static(plan, state, phi, bc5, params):
    with pytest.raises(ValueError, match="rho_star_convention"):
        perpendicular_rhs(plan, state, phi, bc5, KINDS, dataclasses.replace(params, rho_star_convention="gbs"))
    leaves, treedef = jax.tree_util.tree_flatten(dataclasses.replace(params, rho_star_convention="single-length"))
    assert "single-length" not in leaves
    assert treedef != jax.tree_util.tree_flatten(params)[1]


def test_single_length_gradient_wrt_rho_star(plan, state, phi, bc5, params):
    def bracket_sum(rho):
        p = dataclasses.replace(params, rho_star=rho, rho_star_convention="single-length")
        return perpendicular_rhs(plan, state, phi, bc5, KINDS, p, terms=("bracket",)).terms["Te"]["poisson_bracket"].sum()
    g = jax.grad(bracket_sum)(0.7)
    assert abs(float(g) - float(bracket_sum(0.7)) / 0.7) <= 1e-10 * abs(float(g))
