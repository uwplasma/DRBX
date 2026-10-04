"""Tests for ``scripts/p_shared/perpendicular_reference_rhs.py`` -- the host continuum reference of the
production perpendicular RHS terms (P08 step 3, gate G3.3).

Fully synthetic tests of the pure pieces (owner projection weights, sign conventions, the design section 1
algebra) on a fake geometry. The ``slow`` real-geometry tests on the owner closure at N32/N48/N64 (against the frozen
oracles' own R arrays and the host ``assemble_owner_terms`` R terms) were dropped on 4 October 2026 with the
oracle arrays retired on 2 October.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import perpendicular_reference_rhs as prr  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixtures.
# ---------------------------------------------------------------------------
class FakeRef:
    """A reference whose geometry is a fixed per-point table (the query points are ignored)."""

    def __init__(self, J, B, bcov, K, tensor=None):
        self._J, self._B, self._bcov, self._K, self._tensor = J, B, bcov, K, tensor

    def _metric(self, q):
        return {"J": self._J, "B": self._B, "bcov": self._bcov}

    def _curvature(self, q):
        return self._K

    def _perpendicular_flux_tensor(self, q):
        return np.broadcast_to(self._tensor, (len(q), 3, 3)).copy()


def _rng_state(rng, q):
    """A positive (5, Q) state and (5, Q, 3) gradients (n, Te, Ti, omega, phi)."""
    values = np.vstack([1.0 + 0.2 * rng.random((3, q)), rng.normal(size=(2, q))])
    gradients = rng.normal(size=(5, q, 3))
    return values, gradients


def _fake_setup(rng, tau=1.0):
    """Two owners (ids 5 and 7) with two raw cells each; a fully prefilled geometry cache."""
    q = 4
    J = 0.5 + rng.random(q)
    B = 0.8 + 0.4 * rng.random(q)
    bcov = rng.normal(size=(q, 3))
    K = rng.normal(size=(q, 3))
    ref = FakeRef(J, B, bcov, K, tensor=2.0 * np.eye(3))
    env = SimpleNamespace(ref=ref)
    weight = 0.5 + rng.random(q)
    prepared = SimpleNamespace(J=J, B=B, K=K)
    support = prr.OwnerSupport(
        owners=np.array([7, 5]), raw_ids=np.arange(q), raw_owner=np.array([5, 5, 7, 7]),
        raw_volume=np.array([1.0, 3.0, 2.0, 2.0]), points=np.zeros((q, 3)), raw_keys=np.zeros((q, 3), dtype=np.int64),
        p07_rows=np.zeros(0, dtype=np.int64), p07_keys=np.zeros((0, 4), dtype=np.int64),
        p07_family=np.zeros(0, dtype=np.int64), p07_lower=np.zeros(0, dtype=np.int64),
        p07_upper=np.zeros(0, dtype=np.int64), owner_volume=np.array([4.0, 4.0]))
    support._cache["raw"] = {"h": bcov / B[:, None], "jac": np.abs(J), "J": J, "B": B, "bcov": bcov,
                             "prepared": prepared, "evolution_weight": weight * J / np.maximum(B, 1e-30)}
    values, gradients = _rng_state(rng, q)

    class State:
        def values_gradients(self, points):
            index = np.arange(len(points)) % q                              # any point set: tile the q states
            return values[:, index], gradients[:, index]

    return env, support, State(), values, gradients, J, B, bcov, K, weight


# ---------------------------------------------------------------------------
# Pure pieces.
# ---------------------------------------------------------------------------
def test_reference_params_default_diffusion_is_one_and_lookup_is_per_field():
    params = prr.ReferenceParams(rho_star=0.5, tau=2.0, diffusion={"Te": 0.25})
    assert params.D("Te") == 0.25
    assert params.D("density") == 1.0


def test_patched_tau_sets_and_restores_the_p06_module_constant():
    import p06_structured_global.numerics as p06numerics

    before = p06numerics.TAU
    with prr._patched_tau(3.5):
        assert p06numerics.TAU == 3.5
    assert p06numerics.TAU == before
    with pytest.raises(RuntimeError):
        with prr._patched_tau(9.0):
            raise RuntimeError
    assert p06numerics.TAU == before


def test_gather_reads_absent_owners_as_zero_in_requested_order():
    pair = (np.array([3, 5, 9]), np.array([[3.0, 30.0], [5.0, 50.0], [9.0, 90.0]]))
    got = prr._gather(pair, np.array([9, 4, 3]))
    np.testing.assert_array_equal(got, [[9.0, 90.0], [0.0, 0.0], [3.0, 30.0]])
    empty = prr._gather((np.zeros(0, dtype=np.int64), np.zeros((0, 2))), np.array([1, 2]))
    np.testing.assert_array_equal(empty, np.zeros((2, 2)))


def test_rel_uses_the_given_scale_and_is_finite_for_a_vanishing_term():
    d, r = prr._rel(np.array([1e-16]), np.array([0.0]), 10.0)   # a vanishing term judged against the term scale
    assert d == 1e-16 and r == pytest.approx(1e-17)
    assert prr._rel(np.array([0.0]), np.array([0.0])) == (0.0, 0.0)


def test_owner_support_refuses_a_full_grid_and_duplicate_owners():
    with pytest.raises(ValueError):
        prr.owner_support(None, None)
    with pytest.raises(ValueError):
        prr.owner_support(None, "all")
    with pytest.raises(ValueError):
        prr.owner_support(SimpleNamespace(), [3, 3])


def test_bracket_reference_is_the_raw_volume_weighted_owner_mean_of_minus_the_bracket():
    rng = np.random.default_rng(1)
    env, support, _state, _v, gradients, J, B, bcov, _K, _w = _fake_setup(rng)
    g = np.moveaxis(gradients, 0, -1)                                     # (Q, 3, 5)
    got = prr.bracket_reference(env, support, g, [(4, 0), (4, 3)])
    for col, target in enumerate((0, 3)):
        # -[phi, g] = -b_cov.(grad phi x grad g) / (J B), owner mean by raw volume
        point = -np.einsum("qd,qd->q", bcov, np.cross(gradients[4], gradients[target])) / (np.abs(J) * B)
        expected_5 = (1.0 * point[0] + 3.0 * point[1]) / 4.0
        expected_7 = (2.0 * point[2] + 2.0 * point[3]) / 4.0
        np.testing.assert_allclose(got[:, col], [expected_7, expected_5], rtol=1e-13, atol=1e-15)   # owners [7, 5]


def test_reference_rhs_bracket_and_curvature_are_scaled_by_rho_star_and_match_the_design_formulas():
    rng = np.random.default_rng(2)
    env, support, state, values, gradients, J, B, bcov, K, weight = _fake_setup(rng)
    params = prr.ReferenceParams(rho_star=0.37, tau=1.0)
    rhs = prr.reference_rhs(env, support.owners, state, params, terms=("poisson_bracket", "curvature"), support=support)
    indep = prr.production_formula_pointwise(env, support.points, values, gradients, params)
    w_e = weight * J / B
    for name in prr.FIELDS:
        b = indep["poisson_bracket"][name]
        c = indep["curvature"][name]
        np.testing.assert_allclose(rhs[name]["poisson_bracket"],
                                   [(2 * b[2] + 2 * b[3]) / 4.0, (b[0] + 3 * b[1]) / 4.0], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(rhs[name]["curvature"],
                                   [(w_e[2] * c[2] + w_e[3] * c[3]) / (w_e[2] + w_e[3]),
                                    (w_e[0] * c[0] + w_e[1] * c[1]) / (w_e[0] + w_e[1])], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(rhs[name]["curvature"],
                                   rhs[name]["curvature_material"] + rhs[name]["curvature_remainder"],
                                   rtol=1e-13, atol=1e-15)
    # the omega row has no remainder; Ti's remainder coefficient is -4 Ti/(3B)
    np.testing.assert_array_equal(rhs["vorticity"]["curvature_remainder"], 0.0)
    # rho_star only rescales the bracket
    rhs2 = prr.reference_rhs(env, support.owners, state, prr.ReferenceParams(rho_star=0.74, tau=1.0),
                             terms=("poisson_bracket", "curvature"), support=support)
    np.testing.assert_allclose(rhs2["Te"]["poisson_bracket"], rhs["Te"]["poisson_bracket"] / 2.0, rtol=1e-14)
    np.testing.assert_array_equal(rhs2["Te"]["curvature"], rhs["Te"]["curvature"])


@pytest.mark.parametrize("tau", [1.0, 0.4, 2.5])
def test_production_formula_pointwise_material_plus_remainder_equals_the_reduced_continuum_rows(tau):
    """Design section 1's four reduced continuum rows follow from ``(1/B) M K.grad q + coeff K.grad(phi + tau Ti)``
    for *any* phi (the Ti column's polarization response cancels the remainder's ``tau Ti`` part) -- pure algebra."""
    rng = np.random.default_rng(3)
    env, support, _state, values, gradients, *_ = _fake_setup(rng)
    out = prr.production_formula_pointwise(env, support.points, values, gradients, prr.ReferenceParams(tau=tau))
    for name in prr.FIELDS:
        scale = np.max(np.abs(out["curvature"][name]))
        np.testing.assert_allclose(out["curvature_reduced"][name], out["curvature"][name], rtol=0, atol=1e-13 * scale)


def test_production_formula_pointwise_bracket_sign_on_a_hand_computed_case():
    """b_cov = z-hat, J = B = 1: -[phi, g] = -(phi_x g_y - phi_y g_x); sign and rho_star as in the design."""
    env = SimpleNamespace(ref=FakeRef(J=np.ones(1), B=np.ones(1), bcov=np.array([[0.0, 0.0, 1.0]]),
                                      K=np.zeros((1, 3))))
    values = np.ones((5, 1))
    gradients = np.zeros((5, 1, 3))
    gradients[4, 0] = [2.0, 0.0, 0.0]          # grad phi = 2 x-hat
    gradients[0, 0] = [0.0, 3.0, 0.0]          # grad n   = 3 y-hat
    out = prr.production_formula_pointwise(env, np.zeros((1, 3)), values, gradients,
                                           prr.ReferenceParams(rho_star=2.0))
    # grad phi x grad n = 6 z-hat  -> [phi, n] = 6  -> RHS = -6/rho_star = -3
    np.testing.assert_allclose(out["poisson_bracket"]["density"], [-3.0])
    np.testing.assert_allclose(out["poisson_bracket"]["Te"], [0.0])


def test_production_formula_pointwise_curvature_density_row_matches_the_hand_formula():
    K = np.array([[1.0, 2.0, 0.5]])
    env = SimpleNamespace(ref=FakeRef(J=np.ones(1), B=np.array([2.0]), bcov=np.zeros((1, 3)), K=K))
    values = np.array([[1.5], [0.8], [0.6], [0.0], [0.0]])
    rng = np.random.default_rng(4)
    gradients = rng.normal(size=(5, 1, 3))
    out = prr.production_formula_pointwise(env, np.zeros((1, 3)), values, gradients, prr.ReferenceParams(tau=1.0))
    C = lambda i: float(K[0] @ gradients[i, 0])
    n, te = 1.5, 0.8
    expected = (2.0 / 2.0) * (te * C(0) + n * C(1) - n * C(4))            # (2/B)[C(n Te) - n C(phi)]
    np.testing.assert_allclose(out["curvature"]["density"], [expected], rtol=1e-13)


def _patched_quadrature(monkeypatch, n_faces_seen):
    def fake_quadrature(faces, keys, order, face=False):
        keys = np.asarray(keys)
        n_faces_seen.append(len(keys))
        return np.zeros((len(keys), 9, 3)), np.full((len(keys), 9), 1.0 / 9.0)

    monkeypatch.setattr(prr.pshared_provider, "_quadrature", fake_quadrature)


def test_diffusion_reference_scatter_sign_dirichlet_free_and_never_calls_a_collapsed_face(monkeypatch):
    """Two faces on axis 0 (tensor 2*I, unit area): face A (5 below, 7 above), face B a wall face (7 below,
    exterior above, dropped) and a collapsed family-0 face that must never be queried."""
    seen = []
    _patched_quadrature(monkeypatch, seen)
    env = SimpleNamespace(ref=FakeRef(J=np.ones(1), B=np.ones(1), bcov=np.zeros((1, 3)), K=np.zeros((1, 3)),
                                      tensor=2.0 * np.eye(3)), t=SimpleNamespace(faces=None))
    support = prr.OwnerSupport(
        owners=np.array([5, 7]), raw_ids=np.zeros(0, dtype=np.int64), raw_owner=np.zeros(0, dtype=np.int64),
        raw_volume=np.zeros(0), points=np.zeros((1, 3)), raw_keys=np.zeros((0, 3), dtype=np.int64),
        p07_rows=np.arange(3), p07_keys=np.array([[0, 3, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0]]),
        p07_family=np.array([5, 1, 0]), p07_lower=np.array([5, 7, -1]), p07_upper=np.array([7, -1, 5]),
        owner_volume=np.array([2.0, 4.0]))
    calls = []

    def exact_gradients(points):
        calls.append(len(points))
        g = np.zeros((len(points), 1, 3))
        g[:, 0, 0] = 3.0                                                   # df/dx = 3
        return g

    # flux through each unit-area face along +axis 0: (2 I) . (3, 0, 0) -> 6
    div = prr.diffusion_reference(env, support, exact_gradients)                      # production: +div
    pos = prr.diffusion_reference(env, support, exact_gradients, positive_operator=True)   # oracle: -div
    # owner 5: +6 out of its upper face (A) -> +6/2 ; owner 7: +6 out of its upper (wall) face, -6 into
    # its lower face (A) -> 0
    np.testing.assert_allclose(div[:, 0], [6.0 / 2.0, 0.0], atol=1e-15)
    np.testing.assert_array_equal(pos, -div)
    assert seen == [2, 2] and set(calls) == {18}                            # only the two non-collapsed faces


def test_diffusion_reference_returns_zero_when_no_face_contributes():
    env = SimpleNamespace(ref=None, t=SimpleNamespace(faces=None))
    support = prr.OwnerSupport(
        owners=np.array([1, 2]), raw_ids=np.zeros(0, dtype=np.int64), raw_owner=np.zeros(0, dtype=np.int64),
        raw_volume=np.zeros(0), points=np.zeros((1, 3)), raw_keys=np.zeros((0, 3), dtype=np.int64),
        p07_rows=np.zeros(0, dtype=np.int64), p07_keys=np.zeros((0, 4), dtype=np.int64),
        p07_family=np.zeros(0, dtype=np.int64), p07_lower=np.zeros(0, dtype=np.int64),
        p07_upper=np.zeros(0, dtype=np.int64), owner_volume=np.ones(2))
    out = prr.diffusion_reference(env, support, lambda pts: np.zeros((len(pts), 4, 3)))
    np.testing.assert_array_equal(out, np.zeros((2, 4)))


def test_reference_rhs_diffusion_scales_with_D_and_total_sums_the_requested_terms(monkeypatch):
    seen = []
    _patched_quadrature(monkeypatch, seen)
    rng = np.random.default_rng(5)
    env, support, state, *_ = _fake_setup(rng)
    env.t = SimpleNamespace(faces=None)
    env.ref._tensor = np.eye(3)
    support.p07_rows = np.arange(1); support.p07_keys = np.array([[0, 3, 0, 0]])
    support.p07_family = np.array([5]); support.p07_lower = np.array([5]); support.p07_upper = np.array([7])
    params = prr.ReferenceParams(rho_star=1.3, tau=1.0, diffusion={"density": 0.5, "Te": 2.0, "Ti": 0.0, "vorticity": 1.0})
    rhs = prr.reference_rhs(env, support.owners, state, params, support=support)
    unit = prr.ReferenceParams(rho_star=1.3, tau=1.0)
    rhs1 = prr.reference_rhs(env, support.owners, state, unit, terms=("perpendicular_diffusion",), support=support)
    np.testing.assert_allclose(rhs["density"]["perpendicular_diffusion"], 0.5 * rhs1["density"]["perpendicular_diffusion"])
    np.testing.assert_allclose(rhs["Te"]["perpendicular_diffusion"], 2.0 * rhs1["Te"]["perpendicular_diffusion"])
    np.testing.assert_array_equal(rhs["Ti"]["perpendicular_diffusion"], 0.0)
    for name in prr.FIELDS:
        np.testing.assert_allclose(rhs[name]["total"], rhs[name]["poisson_bracket"] + rhs[name]["curvature"]
                                   + rhs[name]["perpendicular_diffusion"], rtol=1e-14, atol=1e-15)
    with pytest.raises(KeyError):
        prr.reference_rhs(env, support.owners, state, params, fields=("Vi",), support=support)


def test_production_formula_check_agrees_to_roundoff_on_the_fake_geometry():
    rng = np.random.default_rng(6)
    env, support, state, *_ = _fake_setup(rng)
    report = prr.production_formula_check(env, support.owners, state, prr.ReferenceParams(rho_star=0.6, tau=1.7),
                                          support=support)
    assert report["max_rel"] < 1e-13
    assert report["J_min"] > 0.0
    assert len(report["pointwise"]) == 4 * 4 and len(report["owner_level"]) == 4 * 2
