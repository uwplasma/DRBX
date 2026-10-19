"""The cached FCI maps are reused only for the magnetic field they were traced in."""

from types import SimpleNamespace

import numpy as np
import pytest

from drbx.fci_braginskii.geometry_build import hsx_fci_builder as hsx

SENTINEL = object()


def _payload(identity):
    payload = {
        "fci_maps_trace_substeps": np.asarray(4),
        "fci_maps_source_fingerprint": np.asarray(hsx._hsx_fci_map_source_fingerprint()),
        "fci_maps_trace_backend": np.asarray("numpy"),
    }
    if identity is not None:
        payload["fci_maps_field_identity"] = np.asarray(hsx._field_identity_json(identity))
    return payload


def _load(monkeypatch, payload, identity):
    monkeypatch.setattr(hsx, "fci_maps_from_metric_cache_payload", lambda *a, **k: SENTINEL)
    monkeypatch.setattr(hsx, "validate_hsx_fci_maps", lambda *a, **k: None)
    return hsx._build_or_load_hsx_fci_maps(
        grid=SimpleNamespace(shape=(2, 2, 2)),
        topology="toroidal",
        construct_fci_maps=True,
        fci_trace_substeps=4,
        cache_payload=payload,
        cache_path=None,
        metric_evaluator=None,  # reaching the tracer without one raises
        bfield=None,
        makegrid_path=None,
        field_identity=identity,
    )


def _identity(sha, currents):
    return {"makegrid": {"sha256": sha, "provenance": {}}, "makegrid_currents": currents}


def test_same_field_reuses_cached_maps(monkeypatch):
    field_a = _identity("a", [1.0, 2.0])
    assert _load(monkeypatch, _payload(field_a), field_a)[0] is SENTINEL


@pytest.mark.parametrize(
    "other", [_identity("b", [1.0, 2.0]), _identity("a", [1.0, 2.5])], ids=["coils", "currents"]
)
def test_different_field_retraces(monkeypatch, capsys, other):
    with pytest.raises(RuntimeError, match="MetricEvaluator is required"):
        _load(monkeypatch, _payload(_identity("a", [1.0, 2.0])), other)
    assert "different magnetic field" in capsys.readouterr().out


def test_cache_without_identity_retraces(monkeypatch):
    with pytest.raises(RuntimeError, match="MetricEvaluator is required"):
        _load(monkeypatch, _payload(None), _identity("a", [1.0]))
