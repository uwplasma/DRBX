"""Inner donor-support option (C0 ``"profile7"`` historic, C1 ``"last_aggregate"``, C3 ``"fixed_radius"`` default; P08 bundle item 3).

Design: ``work/p08_donor_support_c1_20260930/design.md``.  C1 builds the coupled quartic through the last agglomerated
ring: ``StructuredReconstruction._rows`` dispatches to ``_coupled`` when ``profile[rid].min() < 7`` (C0) **or**
``anchor_ring <= last`` (C1 only), ``last`` = first full ring minus one, ``anchor_ring`` = ``i`` for cells and
theta/eta faces and ``i - 1`` for radial faces.  The same rule turns the P07 singleton/ringwise faces (families 5, 6)
into coupled rows (family 7).

* synthetic (fast): the option defaults and threading, the policy/identity keys, ``last`` and the dispatch of
  ``_rows`` on a fake grid, the P07 dispatch;
* one bounded real N32 check (skipped without the HSX inputs): owner rows at C0 and C1 for four owners around the
  switch, every row C1 does not re-dispatch bitwise equal to C0, re-dispatched rows report ``coupled_quartic``.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]              # .../DRBX
WORKSPACE = REPO.parent                                 # .../HSX drbx
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.geometry import fci_perpendicular_integrated_rows as integrated  # noqa: E402
from drbx.geometry.fci_perpendicular_reconstruction import (  # noqa: E402
    INNER_SUPPORTS, StructuredReconstruction)
from p_shared.inner_support import (  # noqa: E402
    DEFAULT_INNER_SUPPORT, INNER_SUPPORT_CHOICES, check_inner_support)


# ---------------------------------------------------------------------------
# Option module, defaults and threading
# ---------------------------------------------------------------------------
def test_inner_support_module():
    assert DEFAULT_INNER_SUPPORT == "fixed_radius"                    # C3 adopted 30 September 2026
    assert INNER_SUPPORT_CHOICES == ("profile7", "last_aggregate", "any_aggregate", "fixed_radius") == INNER_SUPPORTS
    assert check_inner_support("last_aggregate") == "last_aggregate"
    with pytest.raises(ValueError):
        check_inner_support("last")


def test_inner_support_option_is_threaded_with_default_fixed_radius():
    from drbx.stencils import builder
    from p_shared import build_artifact, jax_replay, owner_closure, replay_support, step3_gates

    targets = [replay_support.build_environment, owner_closure.run_owner_closure_check,
               jax_replay.run_jax_owner_closure_check, step3_gates.build_setup, step3_gates.run_g32,
               step3_gates.run_g33, build_artifact.build_policy, build_artifact.build_identity,
               build_artifact.run_full_build, build_artifact._init_worker]
    for fn in targets:
        parameter = inspect.signature(fn).parameters["inner_support"]
        assert parameter.default == "fixed_radius", fn
    for fn in (builder.build_r4_p07_rows, integrated.prepare_integrated_face_rows):   # package API: historic default
        assert inspect.signature(fn).parameters["inner_support"].default == "profile7", fn
    assert inspect.signature(StructuredReconstruction.__init__).parameters["inner_support"].kind is \
        inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(StructuredReconstruction.__init__).parameters["inner_support"].default == "profile7"
    assert replay_support.Environment.__dataclass_fields__["inner_support"].default == "fixed_radius"
    assert step3_gates.Step3Setup.__dataclass_fields__["inner_support"].default == "fixed_radius"
    argv = ["--n", "32", "--input-root", ".", "--sidecar", "s", "--output", "o", "--workers", "1"]
    assert build_artifact.parse_args(argv).inner_support == "fixed_radius"
    assert build_artifact.parse_args(argv + ["--inner-support", "last_aggregate"]).inner_support == "last_aggregate"
    with pytest.raises(ValueError):
        StructuredReconstruction(_fake_t(), inner_support="bogus")


def test_build_policy_and_identity_record_inner_support_only_when_not_profile7(monkeypatch):
    from p_shared import build_artifact as ba

    assert ba.build_policy("fd", "q3", "profile7") == ba.POLICY
    assert ba.build_policy("fd", "q3") == {**ba.POLICY, "inner_support": "fixed_radius"}
    c1 = ba.build_policy("fd", "q3", "last_aggregate")
    assert c1 == {**ba.POLICY, "inner_support": "last_aggregate"}
    assert ba.build_policy("autodiff", "q2", "last_aggregate") == {**ba.build_policy("autodiff", "q2"),
                                                                   "inner_support": "last_aggregate"}
    assert "inner_support" not in ba.POLICY
    with pytest.raises(ValueError):
        ba.build_policy("fd", "q3", "bogus")

    monkeypatch.setattr(ba, "_geometry_component_hashes", lambda root, n: {"geometry": "g"})
    monkeypatch.setattr(ba, "_sidecar_component_hashes", lambda path: {"sidecar": "s"})
    kw = dict(n=32, input_root=Path("."), sidecar_path=Path("."), curvature="fd", face_quadrature="q3")
    c0_id = ba.build_identity(**kw, inner_support="profile7")
    assert ba.build_identity(**kw) == ba.build_identity(**kw, inner_support="fixed_radius") != c0_id
    assert c0_id["policy"] == ba.POLICY and set(c0_id["source_hashes"]) == set(ba.SOURCE_FILES)
    c1_id = ba.build_identity(**kw, inner_support="last_aggregate")
    assert c1_id != c0_id and c1_id["policy"]["inner_support"] == "last_aggregate"
    assert set(c1_id["source_hashes"]) == set(ba.SOURCE_FILES) | set(ba.INNER_SUPPORT_SOURCE_FILES)
    for rel in ba.INNER_SUPPORT_SOURCE_FILES:
        assert (REPO / rel).is_file()


# ---------------------------------------------------------------------------
# ``last`` and the dispatch of ``_rows`` on a fake agglomerated grid
# ---------------------------------------------------------------------------
N = 16
OWNERS_PER_RING = [1, 2, 4, 4, 8, 8] + [16] * 10           # rings 0..5 agglomerated, ring 6 is the first full ring
LAST = 5


def _fake_t(n=N, per_ring=OWNERS_PER_RING):
    ro = np.zeros((n, n, n), dtype=np.int64)
    offset = 0
    for i, m in enumerate(per_ring):
        ro[i] = offset + (np.arange(n) // (n // m))[:, None]
        offset += m
    centers = [(np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) * 2 * np.pi / n, (np.arange(n) + 0.5) * 2 * np.pi / n]
    faces = [np.arange(n + 1) / n, np.arange(n + 1) * 2 * np.pi / n, np.arange(n + 1) * 2 * np.pi / n]
    ro = ro.ravel()
    return SimpleNamespace(n=n, ro=ro, rv=np.ones(ro.size), vol=np.ones(ro.max() + 1), pts=np.zeros((ro.size, 3)),
                           order=np.arange(ro.size), starts=np.arange(ro.max() + 2), centers=centers, faces=faces,
                           g=SimpleNamespace(eta_period=2 * np.pi))


def _tagged(service):
    service._coupled = lambda *a, **k: "coupled"
    service._singleton = lambda *a, **k: "singleton"
    service._tensor = lambda *a, **k: "ringwise"
    return service


def test_last_and_the_dispatch_rule():
    c0 = _tagged(StructuredReconstruction(_fake_t()))
    c1 = _tagged(StructuredReconstruction(_fake_t(), inner_support="last_aggregate"))
    assert c0.inner_support == "profile7" and c1.last == c0.last == LAST
    assert list(c0.profile) == OWNERS_PER_RING
    p = np.zeros((1, 3))

    def kinds(i, location, axis=1):
        key = (i, 3, 3) if location == "cell" else (axis, i, 3, 3)
        return c0._rows(key, p, location, False, None), c1._rows(key, p, location, False, None)

    # cells and theta/eta faces are anchored on ring i; C0: coupled while a layer i-1..i+2 has < 7 owners
    for location, axis in (("cell", 1), ("face", 1), ("face", 2)):
        for i in range(1, N - 2):
            got0, got1 = kinds(i, location, axis)
            want0 = "coupled" if min(OWNERS_PER_RING[i - 1:i + 3]) < 7 else (
                "singleton" if min(OWNERS_PER_RING[i - 1:i + 3]) == N else "ringwise")
            assert got0 == want0, (location, axis, i)
            assert got1 == ("coupled" if (got0 == "coupled" or i <= LAST) else got0), (location, axis, i)
    # radial faces: anchored on ring i - 1 (layers i-2..i+1); face i = 6 sits between ring 5 (last) and ring 6
    for i in range(1, N - 6):
        got0, got1 = kinds(i, "face", axis=0)
        layers = [max(0, l) if l >= 0 else -l - 1 for l in range(i - 2, i + 2)]
        want0 = "coupled" if min(OWNERS_PER_RING[l] for l in layers) < 7 else (
            "singleton" if min(OWNERS_PER_RING[l] for l in layers) == N and max(OWNERS_PER_RING[l] for l in layers) == N
            else "ringwise")
        assert got0 == want0, i
        assert got1 == ("coupled" if (got0 == "coupled" or i - 1 <= LAST) else got0), i
    assert kinds(LAST + 1, "face", axis=0)[1] == "coupled" and kinds(LAST + 2, "face", axis=0)[1] != "coupled"
    assert kinds(LAST, "cell")[1] == "coupled" and kinds(LAST + 1, "cell")[1] != "coupled"
    assert StructuredReconstruction.anchor_ring(0, 7) == 6 and StructuredReconstruction.anchor_ring(1, 7) == 7
    assert StructuredReconstruction.anchor_ring(0, 7, "cell") == 7


def test_last_without_a_full_ring_is_the_outermost_ring():
    t = _fake_t(per_ring=[2] * N)
    assert StructuredReconstruction(t).last == N - 1


# ---------------------------------------------------------------------------
# P07 (R4): families 5 / 6 become the coupled family 7 at C1
# ---------------------------------------------------------------------------
def test_p07_rows_dispatch_follows_inner_support(monkeypatch):
    calls = []

    def stub(name):
        def build(*args):
            calls.append(name)
            return np.array([1, 2], dtype=np.int64), np.array([0.5, 0.25])
        return build

    monkeypatch.setattr(integrated, "_singleton", stub("singleton"))
    monkeypatch.setattr(integrated, "_ringwise", lambda t, s, k, p, i: stub("ringwise")(t, k, p, i))
    monkeypatch.setattr(integrated, "_coupled", lambda t, s, k, p, i: stub("coupled")(t, k, p, i))
    context = _fake_t()
    # (axis, i, j, k): radial face 6 (anchor ring 5 = last), radial face 7 (anchor ring 6), theta faces on rings 5 and 6
    keys = np.array([[0, 6, 3, 3], [0, 7, 3, 3], [1, 5, 3, 3], [1, 6, 3, 3], [2, 4, 3, 3]], dtype=np.int64)
    family = np.array([6, 6, 5, 5, 6])
    points = np.zeros((5, 9, 3)); integrand = np.zeros((5, 9, 3))
    rows0 = integrated.prepare_integrated_face_rows(context, keys, family, points, integrand)
    assert calls == ["ringwise", "ringwise", "singleton", "singleton", "ringwise"]
    assert [r.family for r in rows0] == [6, 6, 5, 5, 6]
    calls.clear()
    rows1 = integrated.prepare_integrated_face_rows(context, keys, family, points, integrand,
                                                    inner_support="last_aggregate")
    assert calls == ["coupled", "ringwise", "coupled", "singleton", "coupled"]
    assert [r.family for r in rows1] == [7, 6, 7, 5, 7]
    # an explicit "profile7" is the default, and families 0-4 / 7 are never touched
    calls.clear()
    rows2 = integrated.prepare_integrated_face_rows(context, keys, family, points, integrand,
                                                    inner_support="profile7")
    assert [r.family for r in rows2] == [6, 6, 5, 5, 6] and calls == ["ringwise", "ringwise", "singleton", "singleton", "ringwise"]
    calls.clear()
    rows3 = integrated.prepare_integrated_face_rows(context, keys[:1], np.array([7]), points[:1], integrand[:1],
                                                    inner_support="last_aggregate")
    assert calls == ["coupled"] and rows3[0].family == 7


# ---------------------------------------------------------------------------
# Real, bounded N32 owner-closure check (four owners around the switch; skipped without the HSX inputs)
# ---------------------------------------------------------------------------
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N32 = 32
_have_inputs = (GEOMETRY / f"{N32}x{N32}x{N32}" / "base_geometry.npz").is_file() and SIDECAR.is_file()
needs_inputs = pytest.mark.skipif(not _have_inputs, reason="HSX N32 geometry/sidecar inputs are unavailable")


@pytest.fixture(scope="module")
def real():
    from p_shared import owner_closure as oc
    from p_shared.replay_support import build_environment

    out = {}
    for name in ("default", "fixed_radius", "profile7", "last_aggregate"):
        kwargs = {} if name == "default" else {"inner_support": name}
        env = build_environment(n=N32, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="fd",
                                face_quadrature="q2", **kwargs)
        if not out:
            last = env.S.last
            # rings inside the agglomerated region (5), the last one, the first full ring, and the first full stencil
            owners = sorted({int(env.t.ro[(i * N32 + N32 // 2) * N32 + N32 // 2]) for i in (5, last, last + 1, last + 2)})
        provider = oc.load_provider_for_env(SIDECAR, curvature="fd", face_quadrature="q2")
        out[name] = dict(env=env, built=oc.build_owner_rows(env, owners, provider=provider))
    out["owners"] = owners
    return out


def _same_point_rows(a, b):
    return (np.array_equal(a.donor_ids, b.donor_ids) and np.array_equal(a.value, b.value)
            and np.array_equal(a.gradient, b.gradient) and a.boundary_conditioned == b.boundary_conditioned
            and np.array_equal(a.trace_target_points, b.trace_target_points)
            and np.array_equal(a.trace_donor_points, b.trace_donor_points))


def _same_integrated(a, b):
    return (np.array_equal(a.donor_ids, b.donor_ids) and np.array_equal(a.weights, b.weights)
            and a.boundary_conditioned == b.boundary_conditioned and a.family == b.family)


@needs_inputs
def test_real_default_is_the_explicit_fixed_radius(real):
    a, b = real["default"], real["fixed_radius"]
    assert a["env"].inner_support == b["env"].inner_support == "fixed_radius"
    assert a["built"]["inner_support"] == "fixed_radius" and a["env"].S.inner_support == "fixed_radius"
    ka, kb = a["built"]["row_index"], b["built"]["row_index"]
    assert set(ka) == set(kb)
    for key in ka:
        same = _same_integrated(ka[key], kb[key]) if isinstance(key, int) else _same_point_rows(ka[key], kb[key])
        assert same, key


@needs_inputs
def test_real_c1_redispatches_only_rows_anchored_at_or_before_the_last_ring(real):
    c0, c1 = real["profile7"], real["last_aggregate"]
    S = c1["env"].S
    assert S.last == 10 and c1["built"]["inner_support"] == "last_aggregate"
    census = c1["env"].census
    keys = census.keys()
    k0, k1 = c0["built"]["row_index"], c1["built"]["row_index"]
    assert set(k0) == set(k1)
    changed = {"R1": 0, "R2": 0, "R3": 0, "R4": 0}
    same = {"R1": 0, "R2": 0, "R3": 0, "R4": 0}
    for key in k0:
        if isinstance(key, int):                                        # R4: P07 id -> census row
            rows = np.flatnonzero((census.p07_id == key) & ~census.legacy_alias_slots)
            axis, i = int(keys[rows[0]][0]), int(keys[rows[0]][1])
            kind, anchor = "R4", StructuredReconstruction.anchor_ring(axis, i)
            a, b = k0[key], k1[key]
            identical = _same_integrated(a, b)
            if a.family in (5, 6) and anchor <= S.last:
                assert b.family == 7 and not identical, key
                changed[kind] += 1
            else:
                assert identical, key
                same[kind] += 1
            continue
        request, entity = key
        if request == "R1":
            anchor = entity // (N32 * N32)
        else:
            row = entity if request == "R2" else entity // 2
            axis, i = int(keys[row][0]), int(keys[row][1])
            anchor = StructuredReconstruction.anchor_ring(axis, i)
        a, b = k0[key], k1[key]
        if request == "R3":                     # side rows are anchored on the adjacent cell: identical or coupled
            if _same_point_rows(a, b):
                same[request] += 1
            else:
                assert b.diagnostics["family"] == "coupled_quartic" and a.diagnostics["family"] != "coupled_quartic", key
                changed[request] += 1
            continue
        if anchor <= S.last and a.diagnostics["family"] not in ("coupled_quartic",) and not a.boundary_conditioned \
                and a.diagnostics["family"] not in ("centered_radial", "collapsed_r0"):
            assert b.diagnostics["family"] == "coupled_quartic", key
            assert not _same_point_rows(a, b), key
            changed[request] += 1
        else:
            assert _same_point_rows(a, b), (key, a.diagnostics["family"], b.diagnostics["family"])
            same[request] += 1
    assert changed["R1"] > 0 and changed["R2"] > 0 and changed["R3"] > 0 and changed["R4"] > 0, changed
    assert same["R1"] > 0 and same["R2"] > 0, same
    # the P07 Neumann rows and the geometry are unaffected
    n0, n1 = c0["built"]["neumann_index"], c1["built"]["neumann_index"]
    assert set(n0) == set(n1)
    np.testing.assert_array_equal(c0["built"]["geometry"].face_points, c1["built"]["geometry"].face_points)
