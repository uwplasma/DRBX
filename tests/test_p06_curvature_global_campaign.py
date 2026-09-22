from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from drbx.geometry.fci_boundary_functional_reconstruction import (
    half_open_periodic_patch_index,
)


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "scripts/p06_curvature_global"


def test_half_open_wall_patch_ties_and_periodic_seam_are_deterministic():
    faces = np.linspace(0.0, 2.0 * np.pi, 9)
    assert half_open_periodic_patch_index(faces, 0.0, 2.0 * np.pi) == 0
    assert half_open_periodic_patch_index(faces, 2.0 * np.pi, 2.0 * np.pi) == 0
    assert half_open_periodic_patch_index(faces, faces[3], 2.0 * np.pi) == 3
    assert half_open_periodic_patch_index(
        faces, -np.finfo(float).eps, 2.0 * np.pi
    ) == 0
    assert half_open_periodic_patch_index(faces, -1.0e-8, 2.0 * np.pi) == 7


def test_campaign_freezes_point_policy_and_primary_actions_without_pooled_gate():
    config = json.loads((CAMPAIGN / "configuration.json").read_text())
    candidate = config["candidate"]
    assert "wall-center point" in candidate["cell_boundary_relation"]
    assert "wall-center point" in candidate["face_boundary_relation"]
    assert config["scope"]["primary_action"] == {
        "density": "U",
        "Te": "U",
        "Ti": "U",
        "vorticity": "centered",
    }
    assert "no pooled gate" in config["scope"]["acceptance"]


def test_portable_manifests_use_relative_paths_and_exclude_archived_answers():
    inputs = json.loads((CAMPAIGN / "input_manifest.json").read_text())
    sources = json.loads((CAMPAIGN / "source_manifest.json").read_text())
    paths = [record["path"] for record in inputs["files"]]
    assert paths
    assert all(not Path(path).is_absolute() for path in paths)
    assert not any("perpendicular_bracket_reference_requalification" in path for path in paths)
    source_paths = {record["path"] for record in sources["files"]}
    assert "scripts/p06_curvature_global/numerics.py" in source_paths
    assert "src/drbx/geometry/fci_boundary_functional_reconstruction.py" in source_paths
    assert "src/drbx/geometry/fci_perpendicular_bracket.py" in source_paths


def test_remote_handoff_is_publication_pending_and_computation_only():
    handoff = (CAMPAIGN / "REMOTE_HANDOFF.md").read_text()
    assert "publication-pending draft" in handoff
    assert "<PUBLISHED_P06_COMMIT>" in handoff
    assert "run drbx on perlmutter" in handoff
    assert "/Users/" not in handoff
    assert handoff.rstrip().endswith(
        "Run the specified computation and return its artifacts and operational receipt. "
        "Do not analyze the scientific results; we will do that locally."
    )
