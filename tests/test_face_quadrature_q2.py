"""P05/P06 face-quadrature option (q3 default, q2 new; P08 operator-change bundle, item 2; fast).

Design: ``work/p08_q2_face_quadrature_20260930/design.md`` section Q2.2.  ``face_quadrature="q3"`` (nine nodes per
face) is bitwise the historic behaviour; ``"q2"`` (2x2 Gauss, four nodes) applies to the P05 upwind jump and the P06
characteristic side correction only, P07 stays q3.

* synthetic: ``GeometryArrays`` at order 2/3 (shapes, the P07 q3 set, identity, save/load), the provider's
  ``face_node_weight(order=2)``, the plan (``Qf`` inference, mixed-count rejection, ``plan.faces.nodes``), the
  build policy/identity, the option threading and the provider/environment mismatch check;
* one real, bounded N32 owner-closure smoke (skipped without the HSX inputs): rows and plans at q3 and q2 for the
  same three owners, JAX operators on both.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]              # .../DRBX
WORKSPACE = REPO.parent                                 # .../HSX drbx
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.stencils.geometry_arrays import (  # noqa: E402
    SCHEMA, GeometryArrays, _ARRAY_FIELDS, build_face_geometry_arrays)
from p_shared.face_quadrature import DEFAULT_FACE_QUADRATURE  # noqa: E402
from drbx.stencils.operator_plan import FACE_NODE_COUNTS, lower_perpendicular_plan_from_rows  # noqa: E402
from tests.perpendicular_synthetic import Boundary, lower_world, make_world  # noqa: E402

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]


# ---------------------------------------------------------------------------
# Synthetic provider: ``face_node_weight(order)`` with distinct nodes per order (so q2 != a subset of q3 by accident).
# ---------------------------------------------------------------------------
class _Provider:
    calls: list

    def __init__(self):
        self.calls = []

    def p05_metric(self, points):
        points = np.asarray(points, dtype=np.float64)
        return 0.5 * points, 1.0 + points[:, 0]

    def p06_curvature(self, points):
        points = np.asarray(points, dtype=np.float64)
        return 2.0 + points[:, 0], 1.0 + 0.1 * points[:, 1], np.stack([points[:, 0], -points[:, 1], points[:, 2]], axis=-1)

    def p06_face_curvature(self, points):
        return self.p06_curvature(points)

    def p07_perpendicular_tensor(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.eye(3)[None] * (1.0 + points[:, 0] + 0.5 * points[:, 2])[:, None, None]

    def p07_perpendicular_tensor_and_divergence(self, points):
        return self.p07_perpendicular_tensor(points), 0.25 * np.asarray(points, dtype=np.float64)

    def raw_cell_weight(self, faces, keys):
        keys = np.asarray(keys, dtype=np.int64)
        lo = np.stack([np.asarray(faces[a])[keys[:, a]] for a in range(3)], axis=-1)
        hi = np.stack([np.asarray(faces[a])[keys[:, a] + 1] for a in range(3)], axis=-1)
        return (0.5 * (lo + hi))[:, None, :], np.prod(hi - lo, axis=-1)[:, None]

    def face_node_weight(self, faces, keys, order=3):
        self.calls.append(order)
        base, weight = self.raw_cell_weight(faces, keys)
        nodes = order * order
        offsets = 0.01 * (np.arange(nodes) + 1.0)[None, :, None] * np.array([1.0, -0.5, 0.25])[None, None, :]
        return np.repeat(base, nodes, axis=1) + offsets, np.repeat(weight, nodes, axis=1) / nodes


class _LegacyProvider(_Provider):
    """A provider from before the option: ``face_node_weight(faces, keys)`` has no ``order`` parameter."""

    def face_node_weight(self, faces, keys):
        return _Provider.face_node_weight(self, faces, keys, 3)


def _inputs(n=6, n_raw=10, n_faces=5, seed=1):
    rng = np.random.default_rng(seed)
    faces = tuple(np.sort(rng.uniform(0.0, 1.0, size=n + 1)) for _ in range(3))
    raw_keys = np.column_stack([rng.integers(0, n, n_raw) for _ in range(3)])
    face_keys = np.column_stack([rng.integers(0, 3, n_faces)] + [rng.integers(0, n, n_faces) for _ in range(3)])
    return faces, raw_keys, face_keys


def _legacy_identity(g: GeometryArrays) -> str:
    """The pre-option hash: the 17 schema arrays in order, then the schema string (independent reimplementation)."""
    digest = hashlib.sha256()
    for name in _ARRAY_FIELDS:
        a = np.asarray(getattr(g, name))
        for chunk in (name.encode(), repr(a.dtype).encode(), repr(a.shape).encode(), np.ascontiguousarray(a).tobytes()):
            digest.update(chunk)
    digest.update(SCHEMA.encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. GeometryArrays
# ---------------------------------------------------------------------------
def test_geometry_arrays_q3_default_is_unchanged():
    faces, raw_keys, face_keys = _inputs()
    provider = _LegacyProvider()                              # no ``order`` argument: must still work at q3
    g = GeometryArrays.build(provider, faces=faces, raw_keys=raw_keys, face_keys=face_keys)
    explicit = GeometryArrays.build(_Provider(), faces=faces, raw_keys=raw_keys, face_keys=face_keys, face_order=3)
    assert g.face_order == 3 and g.p07_face_points is None and g.p07_face_weight is None
    assert g.face_points.shape == (5, 9, 3) and g.p07_face_tensor.shape == (5, 9, 3, 3)
    assert g.p07_points is g.face_points and g.p07_weight is g.p06_face_weight
    assert g.identity == _legacy_identity(g) == explicit.identity
    for name in _ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(g, name), getattr(explicit, name))


def test_geometry_arrays_q3_provider_is_called_without_the_order_argument():
    seen = []

    class Strict(_Provider):
        def face_node_weight(self, faces, keys, *args, **kwargs):
            seen.append((args, kwargs))
            return _Provider.face_node_weight(self, faces, keys)

    faces, raw_keys, face_keys = _inputs()
    GeometryArrays.build(Strict(), faces=faces, raw_keys=raw_keys, face_keys=face_keys)
    assert seen == [((), {})]


def test_geometry_arrays_q2_shapes_and_the_p07_q3_set():
    faces, raw_keys, face_keys = _inputs()
    provider = _Provider()
    q3 = GeometryArrays.build(_Provider(), faces=faces, raw_keys=raw_keys, face_keys=face_keys)
    q2 = GeometryArrays.build(provider, faces=faces, raw_keys=raw_keys, face_keys=face_keys, face_order=2)
    assert provider.calls == [2, 3]
    F = len(face_keys)
    assert q2.face_order == 2
    assert q2.face_points.shape == (F, 4, 3) and q2.p05_face_h.shape == (F, 4, 3)
    assert q2.p05_face_jacobian.shape == q2.p06_face_J.shape == q2.p06_face_B.shape == q2.p06_face_weight.shape == (F, 4)
    assert q2.p06_face_K.shape == (F, 4, 3)
    assert q2.p07_face_tensor.shape == (F, 9, 3, 3)
    assert q2.p07_face_points.shape == (F, 9, 3) and q2.p07_face_weight.shape == (F, 9)
    # the P07 set is exactly the q3 set, and p07_face_tensor is evaluated at it
    np.testing.assert_array_equal(q2.p07_points, q3.face_points)
    np.testing.assert_array_equal(q2.p07_weight, q3.p06_face_weight)
    np.testing.assert_array_equal(q2.p07_face_tensor, q3.p07_face_tensor)
    # everything raw is untouched
    for name in _ARRAY_FIELDS:
        if "raw" in name:
            np.testing.assert_array_equal(getattr(q2, name), getattr(q3, name))
    assert q2.identity != q3.identity
    q2.verify()


def test_geometry_arrays_q2_identity_save_load_and_validation(tmp_path):
    faces, raw_keys, face_keys = _inputs()
    q2 = GeometryArrays.build(_Provider(), faces=faces, raw_keys=raw_keys, face_keys=face_keys, face_order=2)
    q3 = GeometryArrays.build(_Provider(), faces=faces, raw_keys=raw_keys, face_keys=face_keys)
    # the q3 file has exactly the historic members; the q2 file adds the P07 set and ``face_order``
    q3.save(tmp_path / "q3.npz")
    with np.load(tmp_path / "q3.npz") as data:
        assert set(data.files) == set(_ARRAY_FIELDS) | {"schema", "identity"}
    q2.save(tmp_path / "q2.npz")
    with np.load(tmp_path / "q2.npz") as data:
        assert set(data.files) == set(_ARRAY_FIELDS) | {"schema", "identity", "p07_face_points", "p07_face_weight", "face_order"}
    back = GeometryArrays.load(tmp_path / "q2.npz", expected_identity=q2.identity)
    assert back.face_order == 2
    for name in _ARRAY_FIELDS + ("p07_face_points", "p07_face_weight"):
        np.testing.assert_array_equal(getattr(back, name), getattr(q2, name))
    back3 = GeometryArrays.load(tmp_path / "q3.npz")
    assert back3.face_order == 3 and back3.p07_face_points is None and back3.identity == q3.identity
    # the P07 set is part of the identity
    arrays2 = {name: getattr(q2, name) for name in _ARRAY_FIELDS}
    arrays3 = {name: getattr(q3, name) for name in _ARRAY_FIELDS}
    tampered = GeometryArrays(schema=SCHEMA, identity=q2.identity, **arrays2, face_order=2,
                              p07_face_points=q2.p07_face_points, p07_face_weight=q2.p07_face_weight * 1.0000001)
    with pytest.raises(ValueError, match="identity"):
        tampered.verify()
    # presence of the P07 set must match the order (and the node axis the order)
    with pytest.raises(ValueError, match="face_order"):
        GeometryArrays(schema=SCHEMA, identity="x", **arrays3, face_order=2)
    with pytest.raises(ValueError, match="face_order"):
        GeometryArrays(schema=SCHEMA, identity="x", **arrays2, face_order=2)
    with pytest.raises(ValueError, match="four"):
        GeometryArrays(schema=SCHEMA, identity="x", **arrays2)
    with pytest.raises(ValueError, match="face_order"):
        GeometryArrays.build(_Provider(), faces=faces, raw_keys=raw_keys, face_keys=face_keys, face_order=4)


def test_face_geometry_batches_concatenate_to_the_one_shot_arrays():
    from p_shared import owner_closure as oc

    faces, raw_keys, face_keys = _inputs(n_faces=7)
    for order in (3, 2):
        one = GeometryArrays.build(_Provider(), faces=faces, raw_keys=raw_keys, face_keys=face_keys, face_order=order)
        batched = oc._owner_geometry_arrays(_Provider(), faces, raw_keys, face_keys, raw_chunk=4, face_chunk=3,
                                            face_order=order)
        assert batched.identity == one.identity and batched.face_order == order
    assert inspect.signature(oc._owner_geometry_arrays).parameters["face_order"].default == 3
    assert inspect.signature(build_face_geometry_arrays).parameters["face_order"].default == 3
    # an empty batch keeps the node axis
    empty = build_face_geometry_arrays(_Provider(), faces, face_keys[:0], 2)
    assert empty["face_points"].shape == (0, 4, 3) and empty["p07_face_points"].shape == (0, 9, 3)


# ---------------------------------------------------------------------------
# 2. Provider.face_node_weight(order)
# ---------------------------------------------------------------------------
def test_provider_face_node_weight_order_matches_the_frozen_quadrature_bitwise():
    from p07_diffusion_global.numerics import quadrature
    from p_shared.provider import ScriptsGeometryProvider

    rng = np.random.default_rng(3)
    faces = tuple(np.sort(rng.uniform(0.0, 1.0, size=8)) for _ in range(3))
    keys = np.column_stack([rng.integers(0, 3, 12)] + [rng.integers(0, 7, 12) for _ in range(3)])
    provider = ScriptsGeometryProvider(object(), curvature="fd")
    for order in (2, 3):
        points, weight = provider.face_node_weight(faces, keys, order=order)
        ref_points, ref_weight = quadrature(faces, keys, order, face=True)
        assert points.shape == (12, order * order, 3)
        np.testing.assert_array_equal(points, ref_points)
        np.testing.assert_array_equal(weight, ref_weight)
    points, weight = provider.face_node_weight(faces, keys)        # the method's own default order is 3, not the option
    np.testing.assert_array_equal(points, quadrature(faces, keys, 3, face=True)[0])
    assert provider.face_quadrature == "q2"                        # the option default (DEFAULT_FACE_QUADRATURE)
    for name in ("q3", "q2"):
        assert ScriptsGeometryProvider(object(), curvature="fd", face_quadrature=name).face_quadrature == name
    with pytest.raises(ValueError):
        ScriptsGeometryProvider(object(), curvature="fd", face_quadrature="q4")


def test_face_quadrature_module():
    from p_shared import face_quadrature as fq

    assert fq.DEFAULT_FACE_QUADRATURE == "q2" and fq.FACE_QUADRATURE_CHOICES == ("q3", "q2")
    assert fq.face_order("q3") == 3 and fq.face_order("q2") == 2 and fq.face_order() == 2
    assert fq.check_face_quadrature("q2") == "q2"
    with pytest.raises(ValueError):
        fq.check_face_quadrature("q1")


# ---------------------------------------------------------------------------
# 3. operator_plan: a q2 world derived from the q3 synthetic world (same R1/R4 rows, first four face nodes)
# ---------------------------------------------------------------------------
def _with(world, **changes):
    return SimpleNamespace(**{**vars(world), **changes})


def _q2_world(world, *, keep_faces=None):
    """The synthetic q3 world with its R2/R3 rows, their Neumann rows and the face arrays cut to the first four nodes;
    the q3 face set moves to ``p07_face_points`` / ``p07_face_weight`` (as ``GeometryArrays.build(face_order=2)``).
    ``keep_faces`` (census rows) stay at nine nodes, geometry and Neumann rows stay q3: a mixed-count row set."""
    keep_faces = set(int(r) for r in (keep_faces or ()))
    row_index = dict(world.row_index)
    for key, row in world.row_index.items():
        if isinstance(key, tuple) and key[0] in ("R2", "R3"):
            ridx = key[1] if key[0] == "R2" else key[1] // 2
            if ridx in keep_faces:
                continue
            row_index[key] = dataclasses.replace(row, value=row.value[:4], gradient=row.gradient[:4],
                                                 trace_target_points=row.trace_target_points[:4])
    neumann_index = (dict(world.neumann_index) if keep_faces else
                     {k: v for k, v in world.neumann_index.items() if not (k[0] in ("R2", "R3") and k[2] >= 4)})
    g = world.geometry
    arrays = {name: getattr(g, name) for name in _ARRAY_FIELDS}
    if not keep_faces:
        for name in ("face_points", "p05_face_h", "p05_face_jacobian", "p06_face_J", "p06_face_B", "p06_face_K",
                     "p06_face_weight"):
            arrays[name] = arrays[name][:, :4]
        arrays.update(face_order=2, p07_face_points=g.face_points, p07_face_weight=g.p06_face_weight)
    arrays_all = dict(arrays)
    geometry = GeometryArrays(schema=SCHEMA, identity=GeometryArrays._compute_identity(arrays_all), **arrays_all)
    return _with(world, row_index=row_index, neumann_index=neumann_index, geometry=geometry)


@pytest.fixture(scope="module")
def world3():
    return make_world(owners=OWNERS)


@pytest.fixture(scope="module")
def world2(world3):
    return _q2_world(world3)


def test_plan_faces_qf_inferred_and_exposed(world3, world2):
    plan3, plan2 = lower_world(world3), lower_world(world2)
    Fc = len(world3.face_rows)
    assert FACE_NODE_COUNTS == (9, 4)
    assert plan3.faces.nodes == 9 and plan2.faces.nodes == 4
    for name in ("common_value_slot", "common_gradient_slot", "lower_slot", "upper_slot", "fallback_query", "jac",
                 "weight", "J", "B"):
        assert getattr(plan2.faces, name).shape == (Fc, 4), name
        assert getattr(plan3.faces, name).shape == (Fc, 9), name
    assert plan2.faces.h.shape == (Fc, 4, 3) and plan2.faces.K.shape == (Fc, 4, 3)
    # the Neumann targets are flat (face, node) indices with Qf nodes per face
    for plan, qf in ((plan2, 4), (plan3, 9)):
        for target in (plan.faces.common_neumann_target, plan.faces.side_neumann_target):
            assert target.size == 0 or int(target.max()) < Fc * qf
    # P07 keeps its nine nodes and the q3 integrand (its weight is the q3 weight, not the q2 one)
    assert plan2.p07.integrand.shape == plan3.p07.integrand.shape
    np.testing.assert_array_equal(plan2.p07.integrand, plan3.p07.integrand)
    np.testing.assert_array_equal(plan2.p07.p07_id, plan3.p07.p07_id)
    # the R1 cells are identical
    np.testing.assert_array_equal(plan2.cells.evolution_weight, plan3.cells.evolution_weight)


def test_plan_rejects_mixed_node_counts_and_a_geometry_that_disagrees(world3):
    conditioned_or_any = int(world3.face_rows[3])
    mixed = _q2_world(world3, keep_faces=[conditioned_or_any])       # one face keeps nine nodes, the rest four
    # mixed rows against nine-node geometry -> a q2 face lacks R2 targets
    with pytest.raises(ValueError, match="every face needs"):
        lower_world(mixed)
    # q2 rows against nine-node geometry
    q2_rows_q3_geometry = _with(_q2_world(world3), geometry=world3.geometry, neumann_index=world3.neumann_index)
    with pytest.raises(ValueError, match="face nodes"):
        lower_world(q2_rows_q3_geometry)
    # q3 rows against q2 geometry
    q3_rows_q2_geometry = _with(world3, geometry=_q2_world(world3).geometry)
    with pytest.raises(ValueError, match="face nodes"):
        lower_world(q3_rows_q2_geometry)


def test_pack_owner_rows_packs_neumann_rows_for_face_nodes_only(world3, world2):
    from drbx.stencils.operator_plan import pack_owner_rows

    def neumann_nodes(world, face_nodes):
        packed = pack_owner_rows(world.row_index, world.neumann_index, raw_ids=world.raw_ids, face_rows=world.face_rows,
                                 p07_ids=world.census.p07_id[world.p07_rows], face_nodes=face_nodes)
        return sorted({int(q) for chunk in packed.neumann for q, r in zip(chunk.quad_node, chunk.request)
                       if str(r) in ("R2", "R3")})

    assert neumann_nodes(world3, 9) == list(range(9))
    assert neumann_nodes(world2, 4) == list(range(4))


# ---------------------------------------------------------------------------
# 3b. The JAX operators on the synthetic q2 world: shape-generic, P07/centered/q1 untouched
# ---------------------------------------------------------------------------
class _PointwiseBoundary(Boundary):
    """The synthetic boundary data evaluated one point at a time: the value at a point does not depend on which other
    points share the call (so plans with different point tables see bitwise equal data)."""

    def __init__(self, n_fields: int = 5, seed: int = 11):
        super().__init__(n_fields, seed)
        self.A = 0.3 * self.A                     # smooth over the real logical box (0 < u < 1, angles up to 2 pi)

    def dirichlet(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        parts = [Boundary.dirichlet(self, p[None]) for p in points]
        v = np.concatenate([a for a, _ in parts]) if parts else np.empty((0, self.n_fields))
        g = np.concatenate([b for _, b in parts]) if parts else np.empty((0, 3, self.n_fields))
        return 1.0 + 0.1 * v, 0.1 * g

    def normal(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, g = self.dirichlet(points)
        return np.einsum("qa,qaf->qf", self.direction(points), g)


def _admissible(world):
    for key, row in list(world.row_index.items()):
        if hasattr(row, "value") and getattr(row, "value").ndim == 2 and hasattr(row, "gradient"):
            a = np.abs(row.value)
            world.row_index[key] = dataclasses.replace(row, value=a / a.sum(axis=-1, keepdims=True))
    for key, row in list(world.neumann_index.items()):
        a = np.abs(row.value)
        world.neumann_index[key] = dataclasses.replace(row, value=a / a.sum(), boundary_value=0.02 * row.boundary_value)
    return world


def _operator_outputs(plan, fields, kinds=("neumann", "neumann", "dirichlet", "dirichlet", "dirichlet"), owners=None):
    """P05 (centered / jump), P06 (q1 material+remainder / correction) and P07 outputs on a five-column state."""
    from drbx.native.fci_perpendicular_p05_operator import p05_terms
    from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action
    from drbx.native.fci_perpendicular_p07_operator import p07_action
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables

    boundary = _PointwiseBoundary(5)
    bc = boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal)
    fields = jnp.asarray(fields)
    t = p05_terms(plan, fields, bc, kinds, [(4, i) for i in range(4)])
    a = p06_action(plan, fields, bc, kinds)
    p7 = p07_action(plan, fields[:, :4], bc_columns(bc, np.arange(4)), kinds[:4])
    out = {k: np.asarray(v) for k, v in dict(
        p05_centered=t.centered_owner, p05_jump=t.jump_owner, p05_face_jump=t.face_jump,
        p06_material=a.material, p06_remainder=a.remainder, p06_total=a.total, p06_correction=a.correction,
        p07=p7).items()}
    if owners is not None:    # jump / correction at the closure owners only: the others carry a partial q1 evolution volume
        out = {k: (v[np.asarray(owners)] if k in ("p05_jump", "p06_correction") else v) for k, v in out.items()}
    return out


def test_operators_run_at_q2_with_p07_centered_and_q1_untouched():
    w3 = _admissible(make_world(owners=OWNERS))
    w2 = _q2_world(w3)
    fields = 1.0 + 0.1 * np.random.default_rng(5).uniform(size=(w3.n_owners, 5))
    out3 = _operator_outputs(lower_world(w3), fields)
    out2 = _operator_outputs(lower_world(w2), fields)
    for name in ("p07", "p05_centered", "p06_material", "p06_remainder"):
        np.testing.assert_array_equal(out2[name], out3[name], err_msg=name)
    for name in ("p05_jump", "p06_correction"):
        assert np.all(np.isfinite(out2[name])) and np.any(out2[name] != 0.0), name
        assert np.max(np.abs(out2[name] - out3[name])) > 0.0, name
    assert out2["p05_face_jump"].shape == (len(w2.face_rows), 4)


# ---------------------------------------------------------------------------
# 4. Policy and identity
# ---------------------------------------------------------------------------
def test_build_policy_and_identity_distinguish_face_quadrature(monkeypatch):
    from p_shared import build_artifact as ba

    assert ba.build_policy("fd", "q3", "profile7") == ba.POLICY           # the frozen identities, exactly
    assert ba.build_policy("fd", "q3")["quadrature"] == {"raw": "q1", "face": "q3"}
    q2 = ba.build_policy("fd", "q2")
    assert q2["quadrature"] == {"raw": "q1", "face": "q2", "p07_face": "q3"}
    assert ba.build_policy("fd") == q2                                    # q2 is the option default
    assert {k: v for k, v in q2.items() if k not in ("quadrature", "inner_support")} == {k: v for k, v in ba.POLICY.items() if k != "quadrature"}
    assert ba.POLICY["quadrature"] == {"raw": "q1", "face": "q3"}            # the module constant is untouched
    assert ba.build_policy("autodiff", "q2") == {**q2, "curvature": "autodiff"}
    assert ba.build_policy("autodiff", "q3", "profile7") == {**ba.POLICY, "curvature": "autodiff"}
    with pytest.raises(ValueError):
        ba.build_policy("fd", "q4")

    monkeypatch.setattr(ba, "_geometry_component_hashes", lambda root, n: {"geometry": "g"})
    monkeypatch.setattr(ba, "_sidecar_component_hashes", lambda path: {"sidecar": "s"})
    kw = dict(n=32, input_root=Path("."), sidecar_path=Path("."))
    q3_id = ba.build_identity(**kw, curvature="fd", face_quadrature="q3", inner_support="profile7")
    assert q3_id["policy"] == ba.POLICY and set(q3_id["source_hashes"]) == set(ba.SOURCE_FILES)
    q2_id = ba.build_identity(**kw, curvature="fd", face_quadrature="q2", inner_support="profile7")
    assert q2_id == ba.build_identity(**kw, curvature="fd", inner_support="profile7")   # q2 is the default
    assert q2_id != q3_id and q2_id["policy"]["quadrature"]["face"] == "q2"
    assert set(q2_id["source_hashes"]) == set(ba.SOURCE_FILES) | set(ba.Q2_SOURCE_FILES)
    for rel in ba.Q2_SOURCE_FILES:
        assert (REPO / rel).is_file()


def test_build_artifact_cli_flag():
    from p_shared import build_artifact as ba

    argv = ["--n", "32", "--input-root", ".", "--sidecar", "s", "--output", "o", "--workers", "1"]
    assert ba.parse_args(argv).face_quadrature == "q2"
    for name in ("q3", "q2"):
        assert ba.parse_args(argv + ["--face-quadrature", name]).face_quadrature == name


# ---------------------------------------------------------------------------
# 5. Option threading
# ---------------------------------------------------------------------------
def test_face_quadrature_option_is_threaded_with_the_default():
    from p_shared import build_artifact, jax_replay, owner_closure, replay_support, step3_gates
    from p_shared import provider as ps_provider

    targets = [replay_support.build_environment, owner_closure.load_provider_for_env,
               owner_closure.run_owner_closure_check, jax_replay.run_jax_owner_closure_check,
               step3_gates.build_setup, step3_gates.run_g32, step3_gates.run_g33,
               build_artifact.build_policy, build_artifact.build_identity, build_artifact.build_geometry_only,
               build_artifact.run_full_build, build_artifact._init_geometry_worker,
               ps_provider.ScriptsGeometryProvider.from_sidecar, ps_provider.ScriptsGeometryProvider.__init__]
    for fn in targets:
        parameter = inspect.signature(fn).parameters["face_quadrature"]
        assert parameter.default == DEFAULT_FACE_QUADRATURE == "q2", fn
        assert parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD), fn
    assert replay_support.Environment.__dataclass_fields__["face_quadrature"].default == DEFAULT_FACE_QUADRATURE
    assert step3_gates.Step3Setup.__dataclass_fields__["face_quadrature"].default == DEFAULT_FACE_QUADRATURE
    # the package-level face order stays 3 (the frozen q3 geometry); only the harness option defaults to q2
    assert inspect.signature(GeometryArrays.build).parameters["face_order"].default == 3
    assert GeometryArrays.face_order == 3 and GeometryArrays.p07_face_points is None
    assert set(GeometryArrays.__dataclass_fields__) == {"schema", "identity", *_ARRAY_FIELDS}     # the schema fields
    from drbx.stencils import builder
    assert inspect.signature(builder.build_geometry_arrays).parameters["face_order"].default == 3


def test_build_owner_rows_rejects_a_provider_that_disagrees_on_face_quadrature():
    from p_shared import owner_closure

    with pytest.raises(ValueError, match="face_quadrature"):
        owner_closure.build_owner_rows(SimpleNamespace(curvature="fd", face_quadrature="q3"), [0],
                                       provider=SimpleNamespace(curvature="fd", face_quadrature="q2"))
    with pytest.raises(ValueError, match="face_quadrature"):
        owner_closure.build_owner_rows(SimpleNamespace(curvature="fd", face_quadrature="q2"), [0],
                                       provider=SimpleNamespace(curvature="fd", face_quadrature="q3"))
    with pytest.raises(ValueError, match="face_quadrature"):        # an env without the attribute takes the default (q2)
        owner_closure.build_owner_rows(SimpleNamespace(curvature="fd"), [0],
                                       provider=SimpleNamespace(curvature="fd", face_quadrature="q3"))


# ---------------------------------------------------------------------------
# 6. Real, bounded N32 owner-closure smoke (three owners; skipped without the HSX inputs)
# ---------------------------------------------------------------------------
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
_have_inputs = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file()
needs_inputs = pytest.mark.skipif(not _have_inputs, reason="HSX N32 geometry/sidecar inputs are unavailable")
REAL: dict = {}


@pytest.fixture(scope="module")
def real():
    from p_shared import owner_closure as oc
    from p_shared.replay_support import build_environment

    out = {}
    for name in ("q3", "q2"):
        env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="fd", face_quadrature=name)
        if not out:
            picked = oc.select_owners(env.t, env.census)
            owners = sorted({picked["interior"], picked["wall_full"], picked["wall_partial"]})
        provider = oc.load_provider_for_env(SIDECAR, curvature="fd", face_quadrature=name)
        built = oc.build_owner_rows(env, owners, provider=provider)
        plan = lower_perpendicular_plan_from_rows(
            built["row_index"], built["neumann_index"], grid=_loader_grid(env), census=env.census,
            geometry=built["geometry"], raw_volume=env.t.rv, owner_volume=env.t.vol, raw_ids=built["raw_ids"],
            face_rows=built["face_row_indices"], p07_rows=built["p07_row_indices"])
        out[name] = dict(env=env, built=built, plan=plan)
    out["owners"] = owners
    xy = out["q3"]["env"].t.g.owner_centroid_xy                  # (n_owners, 2): smooth positive five-column state
    phase = 0.3 * np.array([1.0, 0.7, -0.5, 0.9, 0.4])[None, :]
    out["fields"] = 1.0 + 0.1 * np.sin(phase * (2.0 * xy[:, :1] + xy[:, 1:2]) + np.arange(5)[None, :])
    return out


def _loader_grid(env):
    from drbx.stencils.loader import LoaderGrid

    t = env.t
    return LoaderGrid.from_arrays(n=env.n, raw_to_owner=t.ro, eta_centers=t.centers[2])


@needs_inputs
def test_real_owner_closure_rows_and_plans_at_q3_and_q2(real):
    from p_shared import owner_closure as oc

    q3, q2 = real["q3"], real["q2"]
    g3, g2 = q3["built"]["geometry"], q2["built"]["geometry"]
    F = len(q3["built"]["face_row_indices"])
    assert F > 0 and q3["plan"].faces.nodes == 9 and q2["plan"].faces.nodes == 4
    assert q3["built"]["face_quadrature"] == "q3" and q2["built"]["face_quadrature"] == "q2"
    assert g3.face_order == 3 and g3.p07_face_points is None and g2.face_order == 2
    assert g2.face_points.shape == (F, 4, 3) and g2.p06_face_weight.shape == (F, 4)
    # q3 is the historic geometry (the batched legacy call), q2's P07 set is exactly the q3 set
    provider = oc.load_provider_for_env(SIDECAR, curvature="fd", face_quadrature="q3")
    census = q3["env"].census
    from drbx.stencils import builder
    keys = builder.census_face_keys(census, q3["built"]["face_row_indices"])
    legacy = build_face_geometry_arrays(provider, q3["env"].ctx.faces, keys)
    for name, value in legacy.items():
        np.testing.assert_array_equal(getattr(g3, name), value, err_msg=name)
    np.testing.assert_array_equal(g2.p07_points, g3.face_points)
    np.testing.assert_array_equal(g2.p07_weight, g3.p06_face_weight)
    np.testing.assert_array_equal(g2.p07_face_tensor, g3.p07_face_tensor)
    # the total face weight of a face is the same to roundoff: both rules integrate the constant exactly
    np.testing.assert_allclose(g2.p06_face_weight.sum(axis=1), g3.p06_face_weight.sum(axis=1), rtol=1e-12)
    # the P07 rows (R4) are the same rows
    assert len(q3["plan"].p07.p07_id) == len(q2["plan"].p07.p07_id)
    np.testing.assert_array_equal(q3["plan"].p07.integrand, q2["plan"].p07.integrand)
    # the owner closure rows: R1 and R4 identical, R2/R3 four-node
    k3, k2 = q3["built"]["row_index"], q2["built"]["row_index"]
    assert set(k3) == set(k2)
    for key in k3:
        if isinstance(key, tuple) and key[0] in ("R2", "R3"):
            assert len(k2[key].trace_target_points) == 4 and len(k3[key].trace_target_points) == 9
    REAL["rows"] = dict(faces=F, plan_bytes=None)


@needs_inputs
def test_real_owner_closure_operators_q2_versus_q3(real):
    kinds = ("neumann", "neumann", "dirichlet", "dirichlet", "dirichlet")
    out3 = _operator_outputs(real["q3"]["plan"], real["fields"], kinds, real["owners"])
    out2 = _operator_outputs(real["q2"]["plan"], real["fields"], kinds, real["owners"])
    assert real["q3"]["plan"].faces.wall.any()
    # untouched by the face rule: P07 (q3 rows), the P05 centered bracket and the P06 q1 material + remainder
    for name in ("p07", "p05_centered", "p06_material", "p06_remainder"):
        np.testing.assert_array_equal(out2[name], out3[name], err_msg=name)
        assert np.all(np.isfinite(out3[name]))
    # the face terms change (and stay finite)
    report = {}
    for name in ("p05_jump", "p06_correction"):
        assert np.all(np.isfinite(out2[name])) and np.all(np.isfinite(out3[name])), name
        delta = float(np.max(np.abs(out2[name] - out3[name])))
        assert delta > 0.0, name
        report[name] = (delta, float(np.max(np.abs(out3[name]))))
    REAL["max_abs_diff"] = report
    print("q2 - q3 (max |diff|, max |q3|):", report)
