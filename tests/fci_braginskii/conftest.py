"""Shared fixtures for the fci_braginskii backend's self-contained tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fake_geometry_artifact(tmp_path, monkeypatch):
    """Install a minimal geometry artifact on a loaded ``simulate_hsx_blob`` module."""

    def install(driver, *, topology="toroidal", shape=(4, 8, 12), metadata=None):
        artifact_dir = tmp_path / "geometry"
        artifact_dir.mkdir(exist_ok=True)
        (artifact_dir / "manifest.json").write_text("{}")
        owner = None
        if topology == "toroidal":
            owner = SimpleNamespace(
                angular_group_size=np.asarray((8, 4, 2, 1)),
                topology=SimpleNamespace(
                    is_active_owner=np.ones(shape, dtype=bool),
                    is_merge_source=np.zeros(shape, dtype=bool),
                ),
            )
        artifact = SimpleNamespace(
            geometry=SimpleNamespace(shape=shape),
            cell_positions=np.zeros(shape + (3,)),
            topology_name=topology,
            nfp=2,
            polar_angular_geometry=owner,
            metadata={
                "trace_substeps": 7,
                "angular_profile_safety_ratio": 0.5,
                **(metadata or {}),
            },
        )
        monkeypatch.setattr(driver, "load_fci_simulation_geometry", lambda _path: artifact)
        return artifact_dir, artifact

    return install
