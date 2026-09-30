"""Tests for ``scripts/p_shared/campaign_fields.py`` -- the module-level
campaign field adapters refactored out of ``replay_units``'s closures (P08
step 2b, task E2).

Two layers, matching this repo's convention: fast synthetic tests (metadata
tables, exact ``wall_cache`` keys including the deliberate omissions, and the
callback plumbing/layout against small fake field evaluators), and ``slow``
real-N32-geometry tests asserting each adapter's ``dirichlet``/``normal``
equals the pre-refactor host closure output *bitwise* on real points (the
closures are reproduced verbatim below as the reference).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import campaign_fields as cf  # noqa: E402

N = 32
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"


def _owner(fields: int, owners: int = 5) -> np.ndarray:
    return np.arange(owners * fields, dtype=np.float64).reshape(owners, fields)


# ---------------------------------------------------------------------------
# Synthetic: metadata, role/variant tables, exact wall-cache keys.
# ---------------------------------------------------------------------------
def test_wall_keys_are_exact_including_deliberate_normal_omissions():
    p05 = cf.P05Adapter(None, _owner(8))
    assert p05.wall_keys == {"cells": cf.WallKeys("p05_cells_trace", None), "faces": cf.WallKeys("p05_faces_trace", None)}
    assert p05.normal is None and not p05.has_normal

    for name in ("p05n_frozen", "p05n_upwind"):
        a = cf.P05NAdapter(name, None, 1.0, _owner(len(cf.P05NAdapter(name, None, 1.0, _owner(1)).names)))
        # cells-role normal key omitted on purpose (raw_N shifted ~2e-16 at N64 when cached)
        assert a.wall_keys["cells"] == cf.WallKeys(f"{name}_cells_trace", None)
        assert a.wall_keys["faces"] == cf.WallKeys(f"{name}_faces_trace", f"{name}_faces_normal")
        assert a.has_normal and callable(a.normal)

    p06n = cf.P06NAdapter(None, 1.0, _owner(5))
    assert p06n.wall_keys["cells"] == cf.WallKeys("p06n_cells_trace", None)  # omitted on purpose
    assert p06n.wall_keys["faces"] == cf.WallKeys("p06n_faces_trace", "p06n_faces_normal")

    legacy = cf.P06LegacyAdapter(None, np.zeros((4, 5, 7)))
    assert tuple(legacy.fields) == legacy.field_names
    for field_name, a in legacy:
        assert a.wall_keys == {"cells": cf.WallKeys(f"p06legacy_cells_trace_{field_name}", None),
                               "faces": cf.WallKeys(f"p06legacy_faces_trace_{field_name}", None)}
        assert not a.has_normal and a.normal is None

    assert cf.P07Adapter(None, _owner(4)).wall_keys == {"p07": cf.WallKeys("p07_p07_trace", "p07_p07_normal")}
    assert cf.P07NAdapter(None, 1.0, _owner(5)).wall_keys == {"p07": cf.WallKeys("p07n_p07_trace", "p07n_p07_normal")}


def test_p05_and_p07_metadata():
    p05 = cf.P05Adapter(None, _owner(8))
    assert p05.pairs == ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (7, 4), (7, 5))
    assert p05.field_kinds == ("dirichlet",) * 8
    assert list(p05.reconstructions) == ["D"]

    p07 = cf.P07Adapter(None, _owner(4))
    assert p07.dirichlet_fields == (0, 1, 2, 3)            # every field: D lift at conditioned faces
    assert p07.field_kinds == ("dirichlet",) * 4 and p07.need_D is False
    p07n = cf.P07NAdapter(None, 1.0, _owner(5))
    assert p07n.dirichlet_fields == ()                     # none: all Neumann-restored
    assert p07n.field_kinds == ("neumann",) * 5 and p07n.need_D is True
    assert p07n.reconstructions["N"].field_kinds == ("neumann",) * 5
    assert p07n.reconstructions["D"].field_kinds == ("dirichlet",) * 5
    assert p07.radial_degree_by_family == p07n.radial_degree_by_family == {1: 4, 2: 4, 4: 3}
    assert len(p07n.names) == 5


@pytest.mark.parametrize("name", ["p05n_frozen", "p05n_upwind"])
def test_p05n_role_and_pair_tables(name):
    a = cf.P05NAdapter(name, None, 1.0, _owner(1))
    a = cf.P05NAdapter(name, None, 1.0, _owner(len(a.names)))
    R = len(a.role_names)
    P = len(a.pair_names)
    assert list(a.role_names) == sorted(a.role_names) and list(a.pair_names) == sorted(a.pair_names)
    assert a.role_physical_index.shape == (R,) and a.role_physical_index.dtype == np.int64
    assert a.role_is_neumann.shape == (R,) and a.role_is_neumann.dtype == bool
    assert len(a.n_pair_index) == len(a.d_pair_index) == len(a.r_pair_index) == P
    assert a.action_pair_index == tuple(a.n_pair_index) + tuple(a.d_pair_index)
    for (na, nb), (da, db), (ra, rb) in zip(a.n_pair_index, a.d_pair_index, a.r_pair_index):
        # R pair = physical columns of the N pair's roles; D pair = the *_D counterparts of the N roles
        assert (ra, rb) == (int(a.role_physical_index[na]), int(a.role_physical_index[nb]))
        for n_role, d_role in ((na, da), (nb, db)):
            src = a.role_names[n_role]
            expected = src[:-2] + "_D" if src.endswith("_N") else src
            assert a.role_names[d_role] == expected
    # Neumann kinds follow role_bc, and the reconstruction feeds owner_values[:, role_physical_index]
    rec = a.reconstructions["role"]
    np.testing.assert_array_equal(rec.columns, a.role_physical_index)
    assert rec.field_kinds == tuple("neumann" if a.role_bc[r] == "neumann" else "dirichlet" for r in a.role_names)
    assert a.field_kinds is None
    assert any(k == "neumann" for k in rec.field_kinds)


def test_p06n_variant_tables():
    a = cf.P06NAdapter(None, 1.0, _owner(1))
    assert set(a.reconstructions) == set(a.variant_names)
    for v in a.variant_names:
        rec = a.reconstructions[v]
        np.testing.assert_array_equal(rec.columns, a.field_index[v])
        assert rec.field_kinds == tuple("neumann" if b else "dirichlet" for b in a.is_neumann[v])
        assert len(rec.columns) == 5                        # 5 primitives per variant
    assert cf.Q3_FIELDS == 4 and a.field_kinds is None


def test_legacy_seam_multiplier_rule():
    keys = np.array([[1, 3, 0, 4], [1, 3, 1, 4], [2, 3, 4, 0], [2, 3, 4, 1], [0, 3, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]])
    np.testing.assert_array_equal(cf.legacy_seam_double_mask(keys),
                                  [True, False, True, False, False, True, True])
    np.testing.assert_array_equal(cf.legacy_seam_multiplier(keys), [2, 1, 2, 1, 1, 2, 2])
    assert cf.legacy_seam_multiplier(keys).dtype == np.float64


def test_p06_legacy_per_field_owner_values_are_transposed_fields():
    ov = np.random.default_rng(0).normal(size=(4, 5, 7))
    legacy = cf.P06LegacyAdapter(None, ov)
    assert legacy.time_value == 0.37
    for fi, (name, a) in enumerate(legacy):
        np.testing.assert_array_equal(a.owner_values, ov[fi].T)
        assert a.owner_values.shape == (7, 5) and a.time_value == 0.37
        assert a.reconstructions["D"].field_kinds == ("dirichlet",) * 5


def test_build_adapter_dispatch():
    assert isinstance(cf.build_adapter("p05", ref=None, period=1.0, owner_values=_owner(8)), cf.P05Adapter)
    assert isinstance(cf.build_adapter("p05n_upwind", ref=None, period=1.0, owner_values=_owner(30)), cf.P05NAdapter)
    assert isinstance(cf.build_adapter("p06n", ref=None, period=1.0, owner_values=_owner(5)), cf.P06NAdapter)
    assert isinstance(cf.build_adapter("p06_legacy", ref=None, period=1.0, owner_values=np.zeros((4, 5, 3))),
                      cf.P06LegacyAdapter)
    assert isinstance(cf.build_adapter("p07", ref=None, period=1.0, owner_values=_owner(4)), cf.P07Adapter)
    assert isinstance(cf.build_adapter("p07n", ref=None, period=1.0, owner_values=_owner(5)), cf.P07NAdapter)
    with pytest.raises(KeyError):
        cf.build_adapter("nope", ref=None, period=1.0, owner_values=_owner(1))


# ---------------------------------------------------------------------------
# Synthetic: callback plumbing/layout against fake field evaluators.
# ---------------------------------------------------------------------------
def _fake_normal(ref, q):
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[:, 0] + 1.0, 2.0 * q[:, 1], -q[:, 2]], axis=1)


def _fake_grad(q, nf):
    """(Q, 3, nf) deterministic gradient."""
    q = np.asarray(q, dtype=np.float64)
    base = np.stack([q[:, 0], q[:, 1] ** 2, q[:, 2] - q[:, 0]], axis=1)          # (Q, 3)
    return base[:, :, None] * (1.0 + np.arange(nf))[None, None, :]


@pytest.fixture
def pts():
    return np.random.default_rng(1).uniform(0.1, 0.9, size=(6, 3))


def test_p05_dirichlet_forwards_float64_points(pts):
    seen = {}

    def boundary_trace(ref, p):
        seen["p"] = p
        return "value", "gradient"

    a = cf.P05Adapter(None, _owner(8))
    a._k = SimpleNamespace(boundary_trace=boundary_trace)
    assert a.dirichlet(pts.astype(np.float32).tolist()) == ("value", "gradient")
    assert seen["p"].dtype == np.float64 and seen["p"].shape == (6, 3)


def test_p05n_dirichlet_and_normal_layout(monkeypatch, pts):
    a = cf.P05NAdapter("p05n_frozen", "REF", 2.5, _owner(1))
    nf = len(a.names)

    def fake_eval(name, ref, points, period):
        assert ref == "REF" and period == 2.5
        j = a.names.index(name)
        g = _fake_grad(points, nf)[:, :, j]
        return np.asarray(points)[:, 0] * (j + 1), g, None

    monkeypatch.setattr(cf, "_p05n_evaluate", fake_eval)
    monkeypatch.setattr("p07n_field_derived_global.fields.normal", _fake_normal)
    v, g = a.dirichlet(pts)
    assert v.shape == (6, nf) and g.shape == (6, 3, nf)
    np.testing.assert_array_equal(g, _fake_grad(pts, nf))
    np.testing.assert_array_equal(v, pts[:, [0]] * (1.0 + np.arange(nf))[None, :])
    gn = a.normal(pts)
    np.testing.assert_array_equal(gn, np.einsum("qa,qaf->qf", _fake_normal(None, pts), _fake_grad(pts, nf)))
    np.testing.assert_array_equal(a.exact_gradient(pts), g)


def test_p06n_normal_is_normal_derivative_of_dirichlet_gradient(monkeypatch, pts):
    a = cf.P06NAdapter("REF", 3.0, _owner(1))

    def fake_trace_all(tables, ref, q, period):
        assert tables is a.tables and ref == "REF" and period == 3.0
        return np.asarray(q)[:, :1] * np.ones((1, 5)), _fake_grad(q, 5)

    monkeypatch.setattr(cf, "_tables_trace_all", fake_trace_all)
    monkeypatch.setattr("p07n_field_derived_global.fields.normal", _fake_normal)
    v, g = a.dirichlet(pts)
    assert v.shape == (6, 5) and g.shape == (6, 3, 5)
    np.testing.assert_array_equal(a.normal(pts), np.einsum("qa,qaf->qf", _fake_normal(None, pts), g))


def test_p06_legacy_trace_layout(pts):
    ov = np.zeros((4, 5, 7))
    legacy = cf.P06LegacyAdapter("REF", ov)
    name, a = next(iter(legacy))
    calls = []

    def fake_eval_fields(field, ref, points, time_value):
        calls.append((field, ref, time_value))
        v5 = np.arange(5)[:, None] + points[:, 0][None, :]                # (5, Q)
        g5 = np.arange(5)[:, None, None] * np.ones((1, len(points), 3))    # (5, Q, 3)
        return v5, g5

    a._p06numerics = SimpleNamespace(_evaluate_fields=fake_eval_fields)
    v, g = a.dirichlet(pts)
    assert calls == [(name, "REF", 0.37)]
    assert v.shape == (6, 5) and g.shape == (6, 3, 5)
    np.testing.assert_array_equal(g[:, 0, :], np.broadcast_to(np.arange(5), (6, 5)))


def test_p07_gradient_axis_swap_at_source(monkeypatch, pts):
    a = cf.P07Adapter("REF", _owner(4))
    raw = np.random.default_rng(2).normal(size=(6, 4, 3))                 # (Q, fields, 3), as p07_fields returns

    def fake_fields(ref, points):
        assert ref == "REF"
        return np.asarray(points)[:, :1] * np.ones((1, 4)), raw, None

    a._fields = fake_fields
    monkeypatch.setattr("p07n_field_derived_global.fields.normal", _fake_normal)
    v, g = a.dirichlet(pts)
    assert g.shape == (6, 3, 4)
    np.testing.assert_array_equal(g, np.swapaxes(raw, 1, 2))
    np.testing.assert_array_equal(a.normal(pts), np.einsum("qa,qaf->qf", _fake_normal(None, pts), g))


def test_p07n_trace_normal_and_exact_gradients(pts):
    a = cf.P07NAdapter("REF", 4.0, _owner(5))
    names = ("a", "b", "c", "d", "e")

    def evaluate(ref, q, name, period):
        assert ref == "REF" and period == 4.0
        j = names.index(name)
        return q[:, 0] * (j + 1), _fake_grad(q, 5)[:, :, j], None

    a._fields = SimpleNamespace(NAMES=names, evaluate=evaluate, normal=_fake_normal)
    v, g = a.dirichlet(pts)
    assert v.shape == (6, 5) and g.shape == (6, 3, 5)
    np.testing.assert_array_equal(g, _fake_grad(pts, 5))
    np.testing.assert_array_equal(a.normal(pts), np.einsum("qa,qaf->qf", _fake_normal(None, pts), g))
    eg = a.exact_gradients(pts)                                           # (Q, F, 3), field axis 1
    np.testing.assert_array_equal(eg, np.swapaxes(g, 1, 2))


# ---------------------------------------------------------------------------
# Real N32 geometry (slow): each adapter's dirichlet/normal equals the
# pre-refactor host closure output, bitwise. The closures below are the
# original ``replay_units`` closures, reproduced verbatim as the reference.
# ---------------------------------------------------------------------------
def _geometry_available(n: int) -> bool:
    directory = GEOMETRY / f"{n}x{n}x{n}"
    return (directory / "base_geometry.npz").is_file() and (directory / "rlp_topology.npz").is_file()


needs_geometry = pytest.mark.skipif(not (_geometry_available(N) and SIDECAR.is_file()),
                                    reason="HSX N32 geometry/sidecar inputs are unavailable")


@pytest.fixture(scope="module")
def real_env():
    from p_shared.replay_support import build_environment

    return build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR)


@pytest.fixture(scope="module")
def real_points(real_env):
    """A few real points: interior cell centres, a few exact wall-lattice nodes (u == 1) and
    off-lattice random points (deterministic)."""
    t = real_env.t
    ids = np.array([t.n ** 3 // 2, t.n ** 3 // 3, 12345, 20001])
    interior = np.asarray(t.pts[ids], dtype=np.float64)
    lattice = np.asarray(real_env.wall_cache.lattice_points[[0, 37, 511]], dtype=np.float64)
    rng = np.random.default_rng(7)
    span = np.asarray(t.pts[:, 1:].max(axis=0))
    rand = np.column_stack([rng.uniform(0.3, 0.95, 3), rng.uniform(0.1, 0.9, 3) * span[0], rng.uniform(0.1, 0.9, 3) * span[1]])
    return np.vstack([interior, lattice, rand])


def _bitwise(a, b):
    a = np.asarray(a); b = np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.all(np.isfinite(a))
    assert a.tobytes() == b.tobytes()


@needs_geometry
@pytest.mark.slow
def test_p05_adapter_matches_host_closure_bitwise(real_env, real_points):
    import p05_structured_global.numerics as k

    a = cf.P05Adapter(real_env.ref, _owner(8))
    v, g = a.dirichlet(real_points)
    rv, rg = k.boundary_trace(real_env.ref, np.asarray(real_points, dtype=np.float64))
    _bitwise(v, rv); _bitwise(g, rg)
    assert g.shape == (len(real_points), 3, 8)


@needs_geometry
@pytest.mark.slow
@pytest.mark.parametrize("name", ["p05n_frozen", "p05n_upwind"])
def test_p05n_adapter_matches_host_closures_bitwise(real_env, real_points, name):
    import p05n_field_derived_global.core as p05n_core
    from p07n_field_derived_global.fields import normal as p07n_normal
    from p_shared.replay_support import _p05n_evaluate

    period = real_env.t.g.eta_period
    table = p05n_core._CATALOGUE_TABLES[cf.P05N_CATALOGUE_FILES[name]]
    names = tuple(table["names"])
    a = cf.P05NAdapter(name, real_env.ref, period, _owner(len(names)))
    assert a.names == names

    def dirichlet_trace_fn(q, names=names, period=period):                # verbatim original closure
        q = np.asarray(q, dtype=np.float64)
        v = np.empty((len(q), len(names))); g = np.empty((len(q), 3, len(names)))
        for j, nm in enumerate(names):
            vv, gg, _ = _p05n_evaluate(nm, real_env.ref, q, period)
            v[:, j] = vv; g[:, :, j] = gg
        return v, g

    def normal_data_fn(q, names=names, period=period):                    # verbatim original closure
        av = p07n_normal(real_env.ref, np.asarray(q, dtype=np.float64))
        outv = np.empty((len(q), len(names)))
        for j, nm in enumerate(names):
            _, gg, _ = _p05n_evaluate(nm, real_env.ref, q, period)
            outv[:, j] = np.einsum("qa,qa->q", av, gg)
        return outv

    v, g = a.dirichlet(real_points)
    rv, rg = dirichlet_trace_fn(real_points)
    _bitwise(v, rv); _bitwise(g, rg)
    _bitwise(a.normal(real_points), normal_data_fn(real_points))
    _bitwise(a.exact_gradient(real_points), rg)


@needs_geometry
@pytest.mark.slow
def test_p06n_adapter_matches_host_closures_bitwise(real_env, real_points):
    import p06n_field_derived_global.core as p06n_core
    from p07n_field_derived_global.fields import normal as p07n_normal
    from p_shared.replay_support import _tables_trace_all

    period = real_env.t.g.eta_period
    tables = p06n_core.CATALOGUE_TABLES
    a = cf.P06NAdapter(real_env.ref, period, _owner(len(tables.names)))

    def dirichlet_trace_fn(q):                                            # verbatim original closure
        return _tables_trace_all(tables, real_env.ref, q, period)

    def normal_data_fn(q):                                                # verbatim original closure
        av = p07n_normal(real_env.ref, np.asarray(q, dtype=np.float64))
        _v, gg = dirichlet_trace_fn(q)
        return np.einsum("qa,qaf->qf", av, gg)

    v, g = a.dirichlet(real_points)
    rv, rg = dirichlet_trace_fn(real_points)
    _bitwise(v, rv); _bitwise(g, rg)
    _bitwise(a.normal(real_points), normal_data_fn(real_points))
    field = tables.names[0]
    for got, want in zip(a.evaluate_exact(real_points, field), tables.evaluate(real_env.ref, real_points, field, period)):
        _bitwise(got, want)


@needs_geometry
@pytest.mark.slow
def test_p06_legacy_adapter_matches_host_closure_bitwise(real_env, real_points):
    import p06_structured_global.numerics as p06numerics

    ov = np.zeros((len(p06numerics.FIELD_NAMES), 5, 3))
    legacy = cf.P06LegacyAdapter(real_env.ref, ov)
    assert legacy.field_names == tuple(p06numerics.FIELD_NAMES)
    for field_name, a in legacy:
        v5, g5 = p06numerics._evaluate_fields(field_name, real_env.ref,      # verbatim original closure
                                              np.asarray(real_points, dtype=np.float64), 0.37)
        rv, rg = v5.T, np.moveaxis(g5, 0, -1)
        v, g = a.dirichlet(real_points)
        _bitwise(v, rv); _bitwise(g, rg)
        assert v.shape == (len(real_points), 5) and g.shape == (len(real_points), 3, 5)


@needs_geometry
@pytest.mark.slow
def test_p07_adapter_matches_host_closures_bitwise(real_env, real_points):
    from p07_diffusion_global.numerics import fields as p07_fields
    from p07n_field_derived_global.fields import normal as p07n_normal

    a = cf.P07Adapter(real_env.ref, _owner(4))

    def trace_fn(q):                                                      # verbatim original closure
        v, g, _h = p07_fields(real_env.ref, np.asarray(q, dtype=np.float64))
        return v, np.swapaxes(g, 1, 2)

    def normal_data_fn(q):                                                # verbatim original closure
        av = p07n_normal(real_env.ref, np.asarray(q, dtype=np.float64))
        _v, g, _h = p07_fields(real_env.ref, np.asarray(q, dtype=np.float64))
        g = np.swapaxes(g, 1, 2)
        return np.einsum("qa,qaf->qf", av, g)

    v, g = a.dirichlet(real_points)
    rv, rg = trace_fn(real_points)
    _bitwise(v, rv); _bitwise(g, rg)
    assert g.shape == (len(real_points), 3, 4)
    _bitwise(a.normal(real_points), normal_data_fn(real_points))


@needs_geometry
@pytest.mark.slow
def test_p07n_adapter_matches_host_closures_bitwise(real_env, real_points):
    import p07n_field_derived_global.fields as p07n_fields

    period = real_env.t.g.eta_period
    a = cf.P07NAdapter(real_env.ref, period, _owner(len(p07n_fields.NAMES)))

    def trace_fn(q):                                                      # verbatim original closure
        q = np.asarray(q, dtype=np.float64)
        v = np.column_stack([p07n_fields.evaluate(real_env.ref, q, nm, period)[0] for nm in p07n_fields.NAMES])
        g = np.stack([p07n_fields.evaluate(real_env.ref, q, nm, period)[1] for nm in p07n_fields.NAMES], axis=-1)
        return v, g

    def normal_data_fn(q):                                                # verbatim original closure
        q = np.asarray(q, dtype=np.float64)
        av = p07n_fields.normal(real_env.ref, q)
        outv = np.empty((len(q), len(p07n_fields.NAMES)))
        for j, nm in enumerate(p07n_fields.NAMES):
            _, gg, _ = p07n_fields.evaluate(real_env.ref, q, nm, period)
            outv[:, j] = np.einsum("qa,qa->q", av, gg)
        return outv

    v, g = a.dirichlet(real_points)
    rv, rg = trace_fn(real_points)
    _bitwise(v, rv); _bitwise(g, rg)
    _bitwise(a.normal(real_points), normal_data_fn(real_points))
    ref_eg = np.stack([p07n_fields.evaluate(real_env.ref, real_points, nm, period)[1] for nm in p07n_fields.NAMES], axis=1)
    _bitwise(a.exact_gradients(real_points), ref_eg)
