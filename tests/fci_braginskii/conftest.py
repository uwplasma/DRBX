"""Shared fixtures for the fci_braginskii backend's self-contained tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# An 8x8x8 HSX bundle (QHS coils, shipped vessel; 0.4 MB) so the compiled time step runs in CI.  Point
# DRBX_TEST_GEOMETRY_BUNDLE at a larger bundle to run the same tests on it.
os.environ.setdefault("DRBX_TEST_GEOMETRY_BUNDLE", str(Path(__file__).resolve().parents[1] / "data" / "hsx_8x8x8"))


@pytest.fixture(scope="session")
def hsx_run(tmp_path_factory):
    """Run the HSX deck on the test bundle once per distinct configuration and return its history.

    ``hsx_run(extra, final_time=..., num_steps=..., fresh=False)`` appends ``extra`` TOML lines to the
    ``[fci_braginskii]`` table.  The result is a dict of arrays plus ``path`` (the history file) and ``columns``
    (the wall-ledger columns as a dict of rows).  A compile costs 15 to 30 s, so equal requests share a run;
    ``fresh=True`` always runs, for tests that patch the model.
    """

    from drbx.cli import main

    cache: dict = {}

    def run(extra="", *, final_time=0.003, num_steps=4, save_every=1, fresh=False):
        key = (extra, final_time, num_steps, save_every)
        if not fresh and key in cache:
            return cache[key]
        root = tmp_path_factory.mktemp("hsx")
        deck = root / "hsx.toml"
        deck.write_text(
            '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
            f'geometry = "{os.environ["DRBX_TEST_GEOMETRY_BUNDLE"]}"\noutput = "{root / "hsx"}"\n'
            f"final_time = {final_time}\nnum_steps = {num_steps}\nsave_every = {save_every}\n" + extra
        )
        assert main(["run", str(deck)]) in (0, None)
        with np.load(root / "hsx.npz") as history:
            data = {name: history[name] for name in history.files}
        data["path"] = root / "hsx.npz"
        data["columns"] = {str(name): data["wall_ledger"][:, k] for k, name in enumerate(data["wall_ledger_columns"])}
        if not fresh:
            cache[key] = data
        return data

    return run


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
