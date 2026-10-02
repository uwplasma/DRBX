"""The P-path ``bfield_toroidal`` operator option (``scripts/p_shared/bfield.py``): ``"spline"`` (default, frozen)
or ``"compact_c3"`` toroidal interpolation of the B evaluator.

Fast: choices/validation, the build policy and identity sources (the spline policy is exactly the one built without
the option), and ``check_artifact_options`` against spline / compact_c3 policies.  Slow, skip-gated on the local N32
inputs: a real provider with ``bfield_toroidal="compact_c3"`` swaps the evaluator, records it in the provenance, moves
the perpendicular tensor by a small nonzero amount and evaluates the autodiff curvature finitely.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import build_artifact as ba  # noqa: E402
from p_shared import bfield  # noqa: E402
from p_shared.provider import ScriptsGeometryProvider  # noqa: E402
from p08_step4_global import smoke  # noqa: E402

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
THREE = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}


def test_choices_and_check():
    assert bfield.BFIELD_TOROIDAL_CHOICES == ("spline", "compact_c3")
    assert bfield.DEFAULT_BFIELD_TOROIDAL == "spline"
    for name in bfield.BFIELD_TOROIDAL_CHOICES:
        assert bfield.check_bfield_toroidal(name) == name
    with pytest.raises(ValueError, match="bfield_toroidal"):
        bfield.check_bfield_toroidal("sideways")
    with pytest.raises(ValueError, match="bfield_toroidal"):
        ba.build_policy("fd", "q3", "profile7", "sideways")


def test_build_policy_spline_is_unchanged_and_compact_is_recorded():
    for args in (("fd", "q3", "profile7"), ("autodiff", "q2", "fixed_radius")):
        assert ba.build_policy(*args, "spline") == ba.build_policy(*args)
        assert "bfield_toroidal" not in ba.build_policy(*args)
        compact = ba.build_policy(*args, "compact_c3")
        assert compact["bfield_toroidal"] == "compact_c3"
        assert {k: v for k, v in compact.items() if k != "bfield_toroidal"} == ba.build_policy(*args)
    assert "bfield_toroidal" not in ba.POLICY


def test_bfield_source_files_exist_and_are_pinned_only_for_compact():
    for rel in ba.BFIELD_SOURCE_FILES:
        assert (ba.REPO / rel).is_file(), rel


def test_provider_rejects_mismatched_evaluator():
    class Ref:
        class bfield_evaluator:                      # noqa: N801
            toroidal_method = "compact_c3"

    with pytest.raises(ValueError, match="does not match"):
        ScriptsGeometryProvider(Ref(), curvature="fd")
    assert ScriptsGeometryProvider(Ref(), curvature="fd", bfield_toroidal="compact_c3").bfield_toroidal == "compact_c3"
    assert ScriptsGeometryProvider(object(), curvature="fd").bfield_toroidal == "spline"


def test_check_artifact_options_spline_and_compact():
    spline = ba.build_policy("autodiff", "q2", "fixed_radius")
    compact = ba.build_policy("autodiff", "q2", "fixed_radius", "compact_c3")
    smoke.check_artifact_options({"policy": spline}, THREE)                              # three-key config, spline
    smoke.check_artifact_options({"policy": spline}, {**THREE, "bfield_toroidal": "spline"})
    smoke.check_artifact_options({"policy": compact}, {**THREE, "bfield_toroidal": "compact_c3"})
    with pytest.raises(ValueError, match="not the pinned"):
        smoke.check_artifact_options({"policy": compact}, THREE)
    with pytest.raises(ValueError, match="not the pinned"):
        smoke.check_artifact_options({"policy": compact}, {**THREE, "bfield_toroidal": "spline"})
    with pytest.raises(ValueError, match="not the pinned"):
        smoke.check_artifact_options({"policy": spline}, {**THREE, "bfield_toroidal": "compact_c3"})


@pytest.mark.slow
def test_real_compact_c3_provider():
    directory = GEOMETRY / f"{N}x{N}x{N}"
    if not ((directory / "base_geometry.npz").is_file() and (directory / "rlp_topology.npz").is_file()
            and SIDECAR.is_file()):
        pytest.skip(f"HSX N{N} geometry/sidecar inputs are unavailable")
    try:
        json.loads(SIDECAR.read_text())
        from perpendicular_structured.reconstruction import load_context

        t = load_context(N, str(WORKSPACE))
        spline = ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False, curvature="autodiff",
                                                      bfield_toroidal="spline")
        compact = ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False, curvature="autodiff",
                                                       bfield_toroidal="compact_c3")
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"HSX inputs unavailable: {error}")
    assert spline.bfield_toroidal == "spline" and compact.bfield_toroidal == "compact_c3"
    assert getattr(spline.reference.bfield_evaluator, "toroidal_method", "spline") == "spline"
    assert compact.reference.bfield_evaluator.toroidal_method == "compact_c3"
    assert compact.reference.provenance["bfield_toroidal"] == "compact_c3"
    assert spline.reference.provenance["bfield_toroidal"] == "spline"

    pts = np.asarray(t.pts)
    interior = pts[(pts[:, 0] > 0.2) & (pts[:, 0] < 0.8)]
    points = interior[np.random.default_rng(0).choice(len(interior), 8, replace=False)]
    a, b = spline.p07_perpendicular_tensor(points), compact.p07_perpendicular_tensor(points)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    assert 0.0 < rel < 1.0e-2, rel
    J, B, K = compact.p06_curvature(points)
    assert np.all(np.isfinite(J)) and np.all(np.isfinite(B)) and np.all(np.isfinite(K))
