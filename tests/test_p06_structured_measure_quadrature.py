"""Regression controls for the research P06 evolution-measure reference."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def module(name):
    path = ROOT / "scripts/p06_structured_global" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"p06_measure_{name}", path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj


def test_reference_saves_independent_j_and_j_over_b_pairs(monkeypatch):
    numeric = module("numerics")
    import perpendicular_structured.reference_geometry as geometry

    monkeypatch.setattr(geometry, "curvature_geometry", lambda _ref, p: SimpleNamespace(
        J=1+p[:, 0], B=1+2*p[:, 0]))
    monkeypatch.setattr(numeric, "_evaluate_fields", lambda _name, _ref, p, _t:
        (np.zeros((5, len(p))), np.zeros((5, len(p), 3))))
    source = {"varying": True}
    def terms(_values, _gradients, geo):
        x = (geo.J-1) if source["varying"] else np.full(len(geo.J), 2.)
        material = np.repeat(x[:, None], 4, axis=1)
        zero = np.zeros_like(material)
        return material, zero, material
    monkeypatch.setattr(numeric, "_continuum_terms", terms)
    context = SimpleNamespace(resolution=1, x_faces=np.array([0., 1.]),
        y_faces=np.array([0., 1.]), z_faces=np.array([0., 1.]))
    pairs = numeric._reference_on_raw_cells(context, None, np.array([0]), 7, 0.)
    p = pairs["numerator_physical"][0, 0, 0, 0]/pairs["volume_physical"][0]
    e = pairs["numerator_evolution"][0, 0, 0, 0]/pairs["volume_evolution"][0]
    np.testing.assert_allclose(p, 5/9, atol=1e-13)
    np.testing.assert_allclose(e, (0.5-np.log(3)/8)/(0.5+np.log(3)/4), atol=1e-7)
    assert abs(p-e) > 0.05
    np.testing.assert_array_equal(pairs["numerator_evolution"][:, 1], 0.)
    np.testing.assert_allclose(pairs["numerator_evolution"][:, 0], pairs["numerator_evolution"][:, 2])
    source["varying"] = False
    constant = numeric._reference_on_raw_cells(context, None, np.array([0]), 5, 0.)
    for measure in ("physical", "evolution"):
        np.testing.assert_allclose(constant[f"numerator_{measure}"][0, 0, 0, 0]
            /constant[f"volume_{measure}"][0], 2., atol=1e-14)


def test_config_rejects_old_reference_and_unsupported_face_rule(tmp_path):
    runner = module("parallel_runner")
    config = json.loads((ROOT / "scripts/p06_structured_global/configuration.json").read_text())
    path = tmp_path / "configuration.json"
    for change in ({"schema": "drbx.p06-structured-global-config-v1"},
                   {"primary_reference_measure": "physical_J"},
                   {"candidate_face_order": 5}):
        path.write_text(json.dumps({**config, **change}))
        with pytest.raises(ValueError):
            runner._load_portable(path)


def test_candidate_rules_and_physical_norm_are_separate(tmp_path):
    numeric = module("numerics")
    runner = module("parallel_runner")
    context = SimpleNamespace(x_faces=np.array([0., 1.]), y_faces=np.array([0., 1.]), z_faces=np.array([0., 1.]))
    keys = np.array([[0, 0, 0]])
    q3, w3 = numeric._cell_quadrature(context, keys, 3)
    p3, v3 = numeric.base._cell_quadrature(context, keys)
    np.testing.assert_array_equal(q3, p3)
    np.testing.assert_array_equal(w3, v3)
    for order in (5, 7):
        points, weights = numeric._cell_quadrature(context, keys, order)
        assert points.shape == (1, order**3, 3)
        np.testing.assert_allclose(np.sum(weights*points[..., 0]**4), 1/5, atol=1e-14)
    data = SimpleNamespace(owner_volume=np.array([1., 9.]), masks={"all":np.array([True, True])})
    stats = numeric._compact_statistics(np.array([1., 0.]), np.zeros(2), data)
    np.testing.assert_allclose(stats["absolute_l2"], np.sqrt(1/10))
    config = json.loads((ROOT / "scripts/p06_structured_global/configuration.json").read_text())
    path = tmp_path / "config.json"
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"content_identity":{"frozen":"bytes"}}))
    prepare = tmp_path / "N32.prepare.npz"
    prepare.write_bytes(b"same preparation bytes")
    identities = []
    for step in (2e-4, 1e-4, 5e-5):
        path.write_text(json.dumps({**config, "curl_step":step}))
        identities.append(runner._manifest_identity(path, inputs, prepare, numeric)["sha256"])
    assert len(set(identities)) == 3
    for change in ({"reference_order":7},{"candidate_cell_order":3}):
        path.write_text(json.dumps({**config,**change}))
        with pytest.raises(ValueError):runner._load_portable(path)


def test_old_physical_target_chunk_schema_rejected(tmp_path):
    numeric = module("numerics")
    path = tmp_path / "chunk.npz"
    numeric._write_npz(path, {"indices":np.array([0])},
        {"schema":"drbx.p06-structured-global-chunk-v1","array_sha256":numeric._array_hash(np.array([0]))})
    with pytest.raises(ValueError, match="schema mismatch"):
        numeric._load_npz(path, numeric.CHUNK_SCHEMA)


def test_face_numerator_must_be_divided_by_new_cell_mass():
    face_numerator = np.array([0.2, -0.1])
    centered_numerator = np.array([2., 3.])
    old_mass, new_mass = 1., 1.25
    old_u = (centered_numerator+face_numerator)/old_mass
    new_u = (centered_numerator+face_numerator)/new_mass
    np.testing.assert_allclose(new_u, centered_numerator/new_mass+face_numerator/new_mass)
    assert not np.allclose(new_u, centered_numerator/new_mass+(old_u-centered_numerator/old_mass))


def test_case_reduction_uses_evolution_target_and_keeps_physical_diagnostic(tmp_path, monkeypatch):
    numeric = module("numerics")
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"schema":"drbx.p06-structured-global-runtime-v3",
        "candidate":{},"scope":{},"candidate_cell_order":1,"candidate_face_order":3,
        "reference_order":1,"paths":{"output":str(tmp_path),"geometry":str(tmp_path),"baseline":str(tmp_path)}}))
    monkeypatch.setattr(numeric, "_source_identity", lambda _config: {})
    monkeypatch.setattr(numeric.integrated, "_load_resolution", lambda *_args:
        SimpleNamespace(owner_keys=np.array([[0,0,0]]),owner_volume=np.array([7.]),
                        masks={"all":np.array([True])}))
    def save(path, arrays, schema, details=None):
        meta={"schema":schema,"status":"complete","details":details or {},
              "array_sha256":numeric._array_hash(*(arrays[k] for k in sorted(arrays)))}
        numeric._write_npz(path,arrays,meta)
    prepare={"raw_owner":np.array([0]),"owner_keys":np.array([[0,0,0]]),"owner_volume":np.array([7.])}
    save(tmp_path/"N1.prepare.npz",prepare,numeric.PREPARE_SCHEMA)
    chunks=tmp_path/"N1.chunks"; chunks.mkdir()
    cell={"indices":np.array([0]),"evolution_volume":np.array([2.]),"physical_volume":np.array([2.])}
    for field in numeric.FIELD_NAMES:
        for term in numeric.TERMS:
            value=0. if term=="remainder" else 4.
            cell[f"candidate:{field}:{term}"]=np.full((1,4),value)
            cell[f"candidate_directional:{field}:{term}"]=np.zeros((1,3,4))
            cell[f"reference_physical:{field}:{term}"]=np.full((1,4),0. if term=="remainder" else 9.)
            cell[f"reference_evolution:{field}:{term}"]=np.full((1,4),value)
    save(chunks/"cell_0000000_0000001.npz",cell,numeric.CHUNK_SCHEMA)
    face={"indices":np.array([0]),"keys":np.array([[0,0,0,0]]),
          "lower_valid":np.array([False]),"upper_valid":np.array([False]),
          "lower_raw":np.array([-1]),"upper_raw":np.array([-1]),
          "correction":np.zeros((4,1,2,4))}
    save(chunks/"face_0000000_0000001.npz",face,numeric.CHUNK_SCHEMA,
         {"dirichlet_trace_error_max":0.})
    reference={"indices":np.array([0]),"volume_physical_q1":np.array([2.]),
               "volume_evolution_q1":np.array([3.]),
               "numerator_physical_q1":np.zeros((4,3,1,4)),
               "numerator_evolution_q1":np.zeros((4,3,1,4))}
    reference["numerator_physical_q1"][:,(0,2)]=10.
    reference["numerator_evolution_q1"][:,(0,2)]=6.
    save(chunks/"reference_0000000_0000001.npz",reference,numeric.CHUNK_SCHEMA)
    numeric._case(SimpleNamespace(config=config,resolution=1))
    arrays,_=numeric._load_npz(tmp_path/"N1.npz",numeric.SCHEMA)
    np.testing.assert_array_equal(arrays["target:variable_dirichlet:material"],2.)
    np.testing.assert_array_equal(arrays["target_physical_q1:variable_dirichlet:material"],5.)
    np.testing.assert_array_equal(arrays["candidate:U:variable_dirichlet:remainder"],0.)
    np.testing.assert_array_equal(arrays["owner_volume"],7.)
