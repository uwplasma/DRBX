"""Fast unit tests of the JAX owner-closure harness helpers (``scripts/p_shared/jax_replay.py``, P08 step 2b, E6).

The real N32/N48/N64 closure gates live in ``tests/test_p_shared_jax_replay_real.py`` (removed 4 October 2026: it needed the oracle arrays retired on 2 October) (slow). Here: the
uniform tolerance policy, the owner-space term normalization, the host-format helpers (``uniq`` /
sparse pairs / structure check), the one-ulp perturbation and conditioning floors, and the P07 owner-numerator
conversion on the synthetic perpendicular world.
"""
from __future__ import annotations

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

from p_shared import jax_replay as jr  # noqa: E402


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
def test_policy_constants_are_the_design_values():
    assert jr.NONCANCELLATION_REL_TOL == 1e-11
    assert jr.CANCELLATION_FLOOR_FACTOR == 10.0 and jr.CANCELLATION_REL_TOL == 1e-8
    assert jr.FLOOR_SEEDS == (0, 1) and 2.0e-16 < jr.ULP < 2.5e-16
    assert jr.CANCELLATION_TERMS >= {"p05_live_jump_owner_num", "p05n_frozen_face_N", "p05n_upwind_face_D",
                                     "p06n_faces_correction", "p06legacy_faces_correction"}


def test_classify_term_and_campaign():
    cls = jr.classify_term
    assert cls("cells.p05_centered") == "operator"
    assert cls("cells.p05n_frozen_raw_N") == "operator"
    assert cls("cells.p05n_frozen_raw_R") == "host_only"
    assert cls("cells.p06n_raw_R_total") == "host_only"
    assert cls("cells.p06n_raw_material") == "operator"
    assert cls("cells.p06legacy_raw_centered.total") == "operator"
    assert cls("faces.p05_live_jump_values") == "cancellation"
    assert cls("faces.p05n_upwind_face_D") == "cancellation"
    assert cls("faces.p06n_faces_correction") == "cancellation"
    assert cls("faces.p06legacy_faces_correction") == "cancellation"
    assert cls("p07.p07n_global_O_q3") == "host_only"
    assert cls("p07.p07n_global_N") == "operator"
    assert jr.term_campaign("cells.p05n_upwind_raw_N") == "p05n_upwind"
    assert jr.term_campaign("faces.p05_live_jump_owner_num") == "p05"
    assert jr.term_campaign("faces.p06legacy_faces_correction") == "p06_legacy"
    assert jr.term_campaign("p07.p07n_global_D") == "p07n"
    assert jr.term_campaign("p07.p07_global_N") == "p07"


def test_noncancellation_policy_is_relative_to_scale():
    ok = jr.evaluate_policy("cells.p05_centered", 0.9e-11 * 40.0, 40.0, None)
    bad = jr.evaluate_policy("cells.p05_centered", 1.1e-11 * 40.0, 40.0, None)
    assert ok["pass"] and not bad["pass"]
    assert ok["max_rel_to_scale"] == pytest.approx(0.9e-11)
    # a floor, if supplied, is reported but never relaxes a non-cancellation term
    assert not jr.evaluate_policy("cells.p05_centered", 1.1e-11 * 40.0, 40.0, 1.0)["pass"]


def test_cancellation_policy_is_the_conditioning_floor():
    scale, floor = 1e-3, 1e-13
    name = "faces.p05n_frozen_face_N"
    # within 10x the floor
    row = jr.evaluate_policy(name, 9.9 * floor, scale, floor)
    assert row["pass"] and row["pass_floor_clause"] and row["pass_rel_clause"] and not row["floor_limited"]
    assert row["over_floor"] == pytest.approx(9.9)
    # beyond 10x the floor
    assert not jr.evaluate_policy(name, 10.1 * floor, scale, floor)["pass"]
    # within the floor although above 1e-8 of scale (the floor itself is): passes, flagged floor_limited
    row = jr.evaluate_policy(name, 2e-11, 1e-3, 5e-11)
    assert row["pass"] and row["pass_floor_clause"] and not row["pass_rel_clause"] and row["floor_limited"]
    # no floor measured: a cancellation term cannot pass
    assert not jr.evaluate_policy(name, 0.0, scale, None)["pass"]


def test_policy_zero_scale_and_finite_sentinels():
    assert jr.evaluate_policy("cells.p05_centered", 0.0, 0.0, None)["pass"]
    row = jr.evaluate_policy("cells.p05_centered", 1e-3, 0.0, None)
    assert not row["pass"] and np.isfinite(row["max_rel_to_scale"])
    row = jr.evaluate_policy("faces.p05n_frozen_face_N", 1e-3, 1.0, 0.0)
    assert not row["pass"] and np.isfinite(row["over_floor"])
    import json
    json.dumps(row, allow_nan=False)


def test_compare_terms_scales_over_all_variants_and_flags_missing_terms():
    host = {"cells.p06n_raw_material": [np.array([[1.0, 2.0]]), np.array([[1e-9, 0.0]])],
            "faces.p06n_faces_correction": [np.array([[1e-3]])], "cells.p05_centered": [np.array([[5.0]])]}
    jax_ = {"cells.p06n_raw_material": [np.array([[1.0, 2.0 + 1e-12]]), np.array([[1e-9 + 1e-12, 0.0]])],
            "faces.p06n_faces_correction": [np.array([[1e-3 + 1e-14]])]}
    rows = {r["term"]: r for r in jr.compare_terms(host, jax_, {"faces.p06n_faces_correction": 5e-15})}
    # the control variant's 1e-12 error is judged against the scale of the largest variant (2.0), not its own 1e-9
    assert rows["cells.p06n_raw_material"]["scale"] == 2.0
    assert rows["cells.p06n_raw_material"]["max_abs"] == pytest.approx(1e-12)
    assert rows["cells.p06n_raw_material"]["pass"]
    assert rows["faces.p06n_faces_correction"]["pass"]                       # 1e-14 <= 10 * 5e-15, 1e-11 of scale
    assert not rows["cells.p05_centered"]["pass"] and rows["cells.p05_centered"]["note"]  # absent on the JAX side


def test_conditioning_floors_take_max_over_seeds():
    nominal = {"t": [np.array([1.0, 2.0])], "u": [np.array([0.0])]}
    seeds = [{"t": [np.array([1.0 + 1e-15, 2.0])], "u": [np.array([2e-16])]},
             {"t": [np.array([1.0, 2.0 - 3e-15])], "u": [np.array([-1e-16])]}]
    floors = jr.conditioning_floors(nominal, seeds)
    assert floors["t"] == pytest.approx(3e-15) and floors["u"] == pytest.approx(2e-16)


def test_perturbation_is_deterministic_one_ulp_scale_and_preserves_none():
    from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData
    x = np.linspace(1.0, 2.0, 5000).reshape(1000, 5)
    a = jr.perturb_array(x, np.random.default_rng(0))
    b = jr.perturb_array(x, np.random.default_rng(0))
    c = jr.perturb_array(x, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    rel = np.abs(a / x - 1.0)
    assert 1e-17 < rel.mean() < 5e-16 and rel.max() < 2e-15
    bc = jr.perturb_boundary(BoundaryData(x, None, x[:, :2]), np.random.default_rng(3))
    assert bc.dirichlet_tangential is None and bc.dirichlet_value.shape == x.shape
    assert not np.array_equal(bc.dirichlet_value, x)


# ---------------------------------------------------------------------------
# host-format helpers
# ---------------------------------------------------------------------------
def test_touched_pair_and_iter_pairs():
    uniq = jr._touched(np.array([3, -1, 1]), np.array([1, 5, -1]))
    np.testing.assert_array_equal(uniq, [1, 3, 5])
    assert uniq.dtype == np.int64
    dense = np.arange(12.0).reshape(6, 2)
    u, v = jr._pair(dense, uniq)
    np.testing.assert_array_equal(v, dense[[1, 3, 5]])
    out = {"a": (u, v), "b": [(u, v), (u, v)], "c": {"x": {"t": (u, v)}, "y": {"t": (u, v)}}, "d": 1.5,
           "e": np.zeros(3)}
    paths = [p for p, _ in jr.iter_pairs(out)]
    assert paths == [("a",), ("b", 0), ("b", 1), ("c", "x", "t"), ("c", "y", "t")]


def test_pairs_mismatches_reports_keys_uniq_and_shape():
    u = np.array([0, 2])
    pair = lambda k=2: (u, np.zeros((len(u), k)))
    host = {"cells": {"a": pair(), "b": [pair(), pair()], "s": 1.0}, "faces": {"f": {"x": pair()}}, "p07": {}}
    same = {"cells": {"a": pair(), "b": [pair(), pair()], "s": 2.0}, "faces": {"f": {"x": pair()}}, "p07": {}}
    assert jr.pairs_mismatches(host, same) == []
    other = {"cells": {"a": (np.array([0, 3]), np.zeros((2, 2))), "b": [pair(), pair(3)], "extra": pair()},
             "faces": {"f": {"x": pair()}}, "p07": {}}
    problems = jr.pairs_mismatches(host, other)
    assert any("cells.s: missing" in p for p in problems)
    assert any("cells.extra: extra" in p for p in problems)
    assert any("cells.a: uniq differs" in p for p in problems)
    assert any("cells.b.1: value shape" in p for p in problems)


def test_normalized_terms_divides_like_compare_to_oracle():
    vol = np.array([2.0, 4.0, 8.0, 16.0])
    owners = np.array([1, 3])
    u = np.array([0, 1, 3])
    pair = lambda scale: (u, np.array([[1.0], [2.0], [4.0]]) * scale)
    ev_num = (u, np.array([10.0, 20.0, 40.0]))
    out = {
        "cells": {"q1_evolution_volume": ev_num, "p05_centered": pair(1.0), "p05_antisymmetry_max": 1e-15,
                  "p06n_raw_material": [pair(1.0), pair(2.0)],
                  "p06legacy_raw_centered": {"f0": {"material": pair(1.0), "remainder": pair(1.0), "total": pair(2.0)}}},
        "faces": {"p05_live_jump_p07ids": np.array([7, 3]), "p05_live_jump_values": np.array([[70.0], [30.0]]),
                  "p05_live_jump_owner_num": pair(1.0), "p06n_faces_correction": [pair(1.0)],
                  "p06legacy_faces_correction": {"f0": pair(1.0)}},
        "p07": {"p07_global_N": pair(1.0)},
    }
    terms = jr.normalized_terms(out, vol=vol, owners=owners)
    np.testing.assert_allclose(terms["cells.q1_evolution_volume"][0], [20.0, 40.0])
    np.testing.assert_allclose(terms["cells.p05_centered"][0], [[2.0 / 4.0], [4.0 / 16.0]])   # / owner volume
    np.testing.assert_allclose(terms["cells.p06n_raw_material"][1], [[4.0 / 20.0], [8.0 / 40.0]])  # / evolution
    np.testing.assert_allclose(terms["cells.p06legacy_raw_centered.total"][0], [[4.0 / 20.0], [8.0 / 40.0]])
    np.testing.assert_allclose(terms["faces.p06n_faces_correction"][0], [[2.0 / 20.0], [4.0 / 40.0]])
    np.testing.assert_allclose(terms["faces.p06legacy_faces_correction"][0], [[2.0 / 20.0], [4.0 / 40.0]])
    np.testing.assert_allclose(terms["p07.p07_global_N"][0], [[2.0 / 4.0], [4.0 / 16.0]])
    np.testing.assert_array_equal(terms["faces.p05_live_jump_values"][0], [[30.0], [70.0]])   # sorted by p07 id
    assert "cells.p05_antisymmetry_max" not in terms and "faces.p05_live_jump_p07ids" not in terms


def test_normalized_terms_guards_the_evolution_volume_like_reduce_grid():
    vol = np.array([1.0, 1.0])
    u = np.array([0, 1])
    out = {"cells": {"q1_evolution_volume": (u, np.array([0.0, 2.0])),
                     "p06n_raw_total": [(u, np.array([[0.0], [4.0]]))]}, "faces": {}, "p07": {}}
    t = jr.normalized_terms(out, vol=vol, owners=np.array([0, 1]))
    np.testing.assert_allclose(t["cells.p06n_raw_total"][0], [[0.0], [2.0]])


# ---------------------------------------------------------------------------
# P07 owner numerator on the synthetic world
# ---------------------------------------------------------------------------
def test_p07_owner_numerator_is_the_unnormalized_action_and_uniq_matches_touched_owners():
    from drbx.native.fci_perpendicular_p07_operator import p07_action
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables
    from tests.perpendicular_synthetic import NF, Boundary, lower_world, make_world

    world = make_world(owners=[0, 1, 2, 40, 96, 99, 100, 114])
    plan = lower_world(world)
    boundary = Boundary()
    bc = boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal)
    fields = np.random.default_rng(5).normal(size=(world.n_owners, NF))
    fake = SimpleNamespace(plan=plan)
    for kinds in ("dirichlet", "neumann", ("neumann", "dirichlet", "neumann")):
        numerator = jr.JaxOwnerClosure._p07_numerator(fake, fields, bc, kinds)
        action = np.asarray(p07_action(plan, fields, bc, kinds))
        np.testing.assert_allclose(numerator / world.owner_volume[:, None], action, rtol=1e-13, atol=1e-13)
        touched = jr._touched(plan.p07.rows.lower_owner, plan.p07.rows.upper_owner)
        untouched = np.setdiff1d(np.arange(world.n_owners), touched)
        assert np.all(numerator[untouched] == 0.0)
        assert np.all(np.any(numerator[touched] != 0.0, axis=1))
