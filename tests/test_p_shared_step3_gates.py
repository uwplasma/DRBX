"""Fast tests of the P08 step-3 gate harness (``scripts/p_shared/step3_gates.py``, gates G3.2 / G3.3).

The real N32/N48/N64 gates live in ``tests/test_p_shared_step3_gates_real.py`` (slow).  Here: the pure pieces (column
naming of a campaign's pairs, feeding the combined call, sparse pairs, the G3.3 metrics, observed orders and flags) and
the glue that drives ``perpendicular_rhs`` for the bracket / curvature / diffusion campaigns, on the shared synthetic
perpendicular world (``tests.perpendicular_synthetic``): each glue function must reproduce the separate operator call
of the same campaign layout.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_p05_operator import p05_terms                     # noqa: E402
from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action          # noqa: E402
from drbx.native.fci_perpendicular_p07_operator import p07_action                      # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import (                        # noqa: E402
    BoundaryData, boundary_data_from_callables)
from drbx.native.fci_perpendicular_rhs import FIELDS, PHI                              # noqa: E402
from p_shared import step3_gates as sg                                                 # noqa: E402
from tests.perpendicular_synthetic import Boundary, lower_world, make_world            # noqa: E402

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
NPHYS = 8
D, N = "dirichlet", "neumann"


# ---------------------------------------------------------------------------
# pair_layout / layout_feed / sparse_pair
# ---------------------------------------------------------------------------
def test_pair_layout_names_the_first_pair_phi_density_and_orders_the_columns():
    lay = sg.pair_layout([(0, 1), (0, 2), (3, 2)])
    assert dict(lay.names) == {0: PHI, 1: "density", 2: "c2", 3: "c3"}
    assert lay.raw_pairs == ((PHI, "density"), (PHI, "c2"), ("c3", "c2"))
    assert lay.columns == ("density", "c2", "c3", PHI)
    assert lay.source == (1, 2, 3, 0)                         # campaign column feeding each layout column
    assert lay.columns[-1] == PHI and lay.source[-1] == 0


def test_pair_layout_keeps_only_used_columns_and_allows_phi_as_a_transported_column():
    lay = sg.pair_layout([(5, 9), (7, 5)])                    # column 5 is the generator (phi) and is transported
    assert lay.raw_pairs == ((PHI, "density"), ("c7", PHI))
    assert lay.columns == ("density", "c7", PHI) and lay.source == (9, 7, 5)
    dup = sg.pair_layout([(1, 4), (1, 4), (4, 1)])            # repeated pairs stay (P05N catalogues repeat them)
    assert dup.raw_pairs == ((PHI, "density"), (PHI, "density"), ("density", PHI))
    assert dup.columns == ("density", PHI)


def test_pair_layout_refuses_empty_and_self_first_pair():
    with pytest.raises(ValueError):
        sg.pair_layout([])
    with pytest.raises(ValueError):
        sg.pair_layout([(2, 2), (2, 3)])


def test_layout_feed_selects_and_orders_values_kinds_and_boundary_columns():
    values = np.arange(12.0).reshape(3, 4)
    bc = BoundaryData(np.arange(2 * 3 * 4.0).reshape(2, 3, 4), None, np.arange(5 * 4.0).reshape(5, 4))
    kinds = (D, N, D, N)
    lay = sg.pair_layout([(0, 1), (0, 2), (3, 2)])
    state, phi, bc_l, kinds_l = sg.layout_feed(lay.columns, lay.source, values, bc, kinds)
    assert list(state) == ["density", "c2", "c3"]
    np.testing.assert_array_equal(state["density"], values[:, 1])
    np.testing.assert_array_equal(state["c3"], values[:, 3])
    np.testing.assert_array_equal(phi, values[:, 0])
    assert kinds_l == {"density": N, "c2": D, "c3": N, PHI: D}
    np.testing.assert_array_equal(np.asarray(bc_l.dirichlet_value), np.asarray(bc.dirichlet_value)[..., [1, 2, 3, 0]])
    np.testing.assert_array_equal(np.asarray(bc_l.neumann_normal), np.asarray(bc.neumann_normal)[..., [1, 2, 3, 0]])
    assert bc_l.dirichlet_tangential is None


def test_sparse_pair_and_host_only_part():
    uniq = np.array([1, 3])
    u, v = sg.sparse_pair(np.arange(10.0).reshape(5, 2), uniq)
    np.testing.assert_array_equal(u, uniq)
    np.testing.assert_array_equal(v, [[2.0, 3.0], [6.0, 7.0]])
    out = {"cells": {"p05n_frozen_raw_R": 1, "p05n_frozen_raw_N": 2, "p06n_raw_R_total": 3, "p06n_raw_total": 4,
                     "q1_evolution_volume": 5},
           "faces": {"p05n_frozen_face_N": 6}, "p07": {"p07n_global_O_q3": 7, "p07n_global_N": 8}}
    assert sg.host_only_part(out) == {"cells": {"p05n_frozen_raw_R": 1, "p06n_raw_R_total": 3},
                                      "p07": {"p07n_global_O_q3": 7}}


# ---------------------------------------------------------------------------
# G3.3 metrics, orders, flags
# ---------------------------------------------------------------------------
def test_term_metrics_are_volume_weighted_and_relative_to_the_reference():
    vol = np.array([1.0, 3.0])
    ref = np.array([2.0, 4.0])
    mine = ref * 1.01
    m = sg.term_metrics(mine, ref, vol)
    l2_ref = math.sqrt((1 * 4 + 3 * 16) / 4)
    assert m["ref_l2"] == pytest.approx(l2_ref) and m["l2"] == pytest.approx(0.01 * l2_ref)
    assert m["rel_l2"] == pytest.approx(0.01) and m["rel_max"] == pytest.approx(0.01)
    assert m["max_abs"] == pytest.approx(0.04) and m["ref_max"] == pytest.approx(4.0)
    assert m["fit_scale"] == pytest.approx(1.01) and m["finite"] and not m["degenerate_reference"]
    flipped = sg.term_metrics(-ref, ref, vol)                       # a sign error: O(1) error, fit -1
    assert flipped["fit_scale"] == pytest.approx(-1.0) and flipped["rel_l2"] == pytest.approx(2.0)
    scaled = sg.term_metrics(2 * ref, ref, vol)
    assert scaled["fit_scale"] == pytest.approx(2.0) and scaled["rel_l2"] == pytest.approx(1.0)


def test_term_metrics_use_the_field_scale_for_an_identically_zero_reference():
    vol = np.array([1.0, 1.0])
    m = sg.term_metrics(np.array([1e-3, -1e-3]), np.array([1e-16, 0.0]), vol, fallback_l2=2.0, fallback_max=4.0)
    assert m["degenerate_reference"] and m["fit_scale"] is None
    assert m["rel_l2"] == pytest.approx(1e-3 / 2.0) and m["rel_max"] == pytest.approx(1e-3 / 4.0)
    plain = sg.term_metrics(np.array([1.0, 1.0]), np.array([0.0, 0.0]), vol)       # no fallback: undefined
    assert plain["rel_l2"] is None and not plain["degenerate_reference"]
    bad = sg.term_metrics(np.array([np.nan, 1.0]), np.array([1.0, 1.0]), vol)
    assert not bad["finite"] and bad["rel_l2"] is None


def test_observed_order():
    assert sg.observed_order(4.0, 1.0, 32, 64) == pytest.approx(2.0)
    assert sg.observed_order(1.0, 1.0, 32, 48) == pytest.approx(0.0)
    assert sg.observed_order(1.0, 0.5, 48, 64) == pytest.approx(math.log(2) / math.log(64 / 48))
    for a, b in ((None, 1.0), (1.0, None), (0.0, 1.0), (1.0, 0.0)):
        assert sg.observed_order(a, b, 32, 64) is None


def _entry(ns, errors, finite=None):
    return {"n": list(ns), "rel_l2": list(errors), "finite": finite or [True] * len(ns)}


def test_flag_orders():
    assert sg.flag_orders(_entry([32, 48, 64], [1e-2, 1e-3, 1e-4])) == []
    assert "O(1)" in sg.flag_orders(_entry([32, 48], [1.5, 1.0e-1]))
    flags = sg.flag_orders(_entry([32, 48, 64], [1e-3, 1.5e-3, 2e-3]))                  # grows, but by less than 2x
    assert flags == ["not_decreasing_32_to_48", "not_decreasing_48_to_64"]
    flags = sg.flag_orders(_entry([32, 48], [1e-3, 3e-3]))
    assert "grows_more_than_2x_32_to_48" in flags and "not_decreasing_32_to_48" in flags
    assert sg.flag_orders(_entry([32, 48], [None, 1e-3])) == ["non_finite_or_undefined"]
    assert sg.flag_orders(_entry([32, 48], [1e-3, float("nan")])) == ["non_finite_or_undefined"]
    assert sg.flag_orders(_entry([32, 48], [1e-2, 1e-3], finite=[True, False])) == ["non_finite_or_undefined"]


def _fake_g33_report(n, scale):
    metrics = lambda e: {"rel_l2": e, "rel_max": e, "l2": e, "max_abs": e, "fit_scale": 1.0, "finite": True}
    fields = {f: {t: metrics(scale * (i + 1)) for i, t in enumerate(sg.G33_TERMS)} for f in FIELDS}
    return {"results": {"main_phi_dirichlet": {"fields": fields}}}


def test_observed_orders_over_grids():
    reports = {32: _fake_g33_report(32, 1e-2), 48: _fake_g33_report(48, 1e-2 * (32 / 48) ** 3),
               64: _fake_g33_report(64, 1e-2 * (32 / 64) ** 3)}
    orders = sg.observed_orders(reports)
    e = orders["main_phi_dirichlet"]["Te"]["curvature"]
    assert e["n"] == [32, 48, 64]
    assert e["order_rel_l2"] == pytest.approx([3.0, 3.0])
    assert e["order_l2"] == pytest.approx([3.0, 3.0]) and e["order_max_abs"] == pytest.approx([3.0, 3.0])
    assert all(t["flags"] == [] for f in orders["main_phi_dirichlet"].values() for t in f.values())
    reports[64] = _fake_g33_report(64, 1e-2)                                          # stalls: not decreasing at 64
    flagged = sg.observed_orders(reports)["main_phi_dirichlet"]["density"]["total"]
    assert "not_decreasing_48_to_64" in flagged["flags"]


def test_g33_constants_are_the_reported_parameters():
    assert sg.G33_PARAMS["rho_star"] == 0.05 and sg.G33_PARAMS["tau"] == 1.0
    assert set(sg.G33_PARAMS["diffusion"]) == set(FIELDS) and set(sg.G33_PARAMS["diffusion"].values()) == {1e-2}
    assert sg.G33_TERMS == ("poisson_bracket", "curvature", "perpendicular_diffusion", "total")
    assert sg.G33_VARIANTS == ("main_phi_dirichlet", "main_phi_neumann")


# ---------------------------------------------------------------------------
# glue against the separate operators on the synthetic world
# ---------------------------------------------------------------------------
class PositiveBoundary(Boundary):
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
def closure():
    world = _admissible(make_world(owners=OWNERS))
    return SimpleNamespace(plan=lower_world(world), n_owners=world.n_owners)


@pytest.fixture(scope="module")
def values(closure):
    return 1.0 + 0.4 * np.random.default_rng(5).uniform(size=(closure.n_owners, NPHYS))


@pytest.fixture(scope="module")
def bc(closure):
    b = PositiveBoundary(NPHYS)
    return boundary_data_from_callables(closure.plan, b.dirichlet, b.normal)


def _close(a, b, rel=1e-11):
    a, b = np.asarray(a)[OWNERS], np.asarray(b)[OWNERS]
    scale = max(float(np.max(np.abs(b))), 1e-300)
    assert float(np.max(np.abs(a - b))) <= rel * scale


def test_bracket_glue_reproduces_the_campaign_pairs_of_p05_terms(closure, values, bc):
    kinds = (D, N, D, N, D, D, N, D)
    pairs = [(0, 1), (0, 2), (3, 2), (5, 1), (6, 0)]
    raw, mismatch, _diag = sg._bracket(closure, values, bc, kinds, pairs, None)
    ref = p05_terms(closure.plan, values, bc, kinds, pairs)
    assert mismatch == 0.0                                   # the bracket term is exactly the first raw pair
    for name in ("centered_numerator", "jump_numerator", "centered_owner", "jump_owner"):
        _close(getattr(raw, name), getattr(ref, name))
    np.testing.assert_allclose(np.asarray(raw.face_jump), np.asarray(ref.face_jump), atol=1e-11 * float(
        np.max(np.abs(np.asarray(ref.face_jump)))))


def test_bracket_glue_honours_the_jump_mask(closure, values, bc):
    kinds = (D,) * NPHYS
    pairs = [(0, 1), (2, 3)]
    ones = np.ones(len(closure.plan.faces.census_row), dtype=bool)
    raw, _m, _d = sg._bracket(closure, values, bc, kinds, pairs, ones)
    ref = p05_terms(closure.plan, values, bc, kinds, pairs, jump_mask=ones)
    _close(raw.jump_numerator, ref.jump_numerator)


def test_curvature_glue_reproduces_p06_action(closure, values, bc):
    kinds5 = (N, N, D, D, N)
    bc5 = bc_columns(bc, np.arange(5))
    arrays, curvature, diag = sg._curvature(closure, values[:, :5], bc5, kinds5)
    act = p06_action(closure.plan, values[:, :5], bc5, kinds5)
    _close(arrays["q1"], act.total)
    _close(arrays["correction"], act.correction)
    _close(arrays["material"], act.material)
    _close(arrays["remainder"], act.remainder)
    _close(curvature, np.asarray(act.total) + np.asarray(act.correction))
    assert set(diag) == {"spectral_fallback", "floor_hits", "wall_fallback"}


def test_curvature_glue_applies_the_face_multiplier(closure, values, bc):
    kinds5 = (D,) * 5
    bc5 = bc_columns(bc, np.arange(5))
    multiplier = np.where(np.arange(len(closure.plan.faces.census_row)) % 3 == 0, 2.0, 1.0)
    arrays, _c, _d = sg._curvature(closure, values[:, :5], bc5, kinds5, face_multiplier=multiplier)
    act = p06_action(closure.plan, values[:, :5], bc5, kinds5, face_multiplier=multiplier)
    _close(arrays["correction"], act.correction)
    plain, _c, _d = sg._curvature(closure, values[:, :5], bc5, kinds5)
    assert float(np.max(np.abs(np.asarray(plain["correction"]) - np.asarray(arrays["correction"])))) > 0.0


@pytest.mark.parametrize("ncols", [4, 5, 8])
def test_diffusion_glue_is_the_positive_p07_action_in_blocks_of_four(closure, values, bc, ncols):
    kinds = tuple([N, D] * 4)[:ncols]
    bc_n = bc_columns(bc, np.arange(ncols))
    action = sg._diffusion_action(closure, values[:, :ncols], bc_n, kinds)
    ref = p07_action(closure.plan, values[:, :ncols], bc_n, kinds)
    assert action.shape == (closure.n_owners, ncols)
    _close(action, ref)
