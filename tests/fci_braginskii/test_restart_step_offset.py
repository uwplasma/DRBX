"""Restarted runs keep global step labels on their checkpoints.

Before the fix a run restarted from ``history.checkpoint_step000010.npz``
numbered its own steps from 1, so its tenth step (global step 20) was saved as
``history.checkpoint_step000010.npz`` with ``step = 10``.  With the same
``--output`` that silently overwrote the checkpoint the run restarted from.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import os
import sys

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPOSITORY / "src" / "drbx" / "fci_braginskii" / "run.py"
DRIVER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "drbx" / "fci_braginskii" / "run.py"


def _driver_module():
    spec = importlib.util.spec_from_file_location("hsx_driver_restart_offset", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_restart_step_offset_reads_checkpoint_step_and_defaults_to_zero(tmp_path):
    hsx = _driver_module()
    checkpoint = tmp_path / "run.checkpoint_step000010.npz"
    np.savez(checkpoint, density=np.ones((2, 2, 2)), step=np.asarray(10, dtype=np.int64))
    assert hsx._restart_step_offset(checkpoint) == 10
    history = tmp_path / "history.npz"
    np.savez(history, density=np.ones((3, 2, 2, 2)), times=np.arange(3.0))
    assert hsx._restart_step_offset(history) == 0


def test_snapshot_paths_use_the_global_step(tmp_path):
    hsx = _driver_module()
    history = tmp_path / "out" / "history.npz"
    root = tmp_path / "snap"
    kwargs = dict(step=20, requested_time=0.5)
    assert hsx._snapshot_path(history, root, failure=False, periodic=True, **kwargs).name == (
        "history.checkpoint_step000020.npz"
    )
    assert hsx._snapshot_path(history, root, failure=True, periodic=False, **kwargs) == (
        history.parent / "history.failure_step000020.npz"
    )
    assert hsx._snapshot_path(history, root, failure=False, periodic=False, **kwargs).parent == root


def test_run_applies_the_start_offset_before_saving_and_checkpointing():
    source = DRIVER_SOURCE.read_text()
    tree = ast.parse(source)
    run = _function(tree, "run_full_eb")
    assert "start_step" in [arg.arg for arg in run.args.kwonlyargs]
    save = ast.get_source_segment(source, _function(run, "save_snapshot"))
    assert save.index("step = int(step) + int(start_step)") < save.index("_snapshot_payload(")
    assert "(step + int(start_step)) % int(checkpoint_every) == 0" in ast.get_source_segment(source, run)
    main = ast.get_source_segment(source, _function(tree, "run"))
    assert "_restart_step_offset(args.restart_from) if restart_used else 0" in main


@pytest.mark.slow
def test_restart_from_a_checkpoint_reproduces_the_continuous_run(tmp_path):
    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to run the HSX restart check")
    from drbx.cli import main

    def run(name, steps, extra):
        deck = tmp_path / f"{name}.toml"
        deck.write_text(
            '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
            f'geometry = "{bundle}"\noutput = "{tmp_path / name}"\nfinal_time = 0.003\nnum_steps = {steps}\n{extra}\n'
        )
        assert main(["run", str(deck)]) in (0, None)
        return np.load(tmp_path / f"{name}.npz")

    continuous = run("continuous", 4, "checkpoint_every = 2")
    restarted = run("restarted", 2, f'restart_from = "{tmp_path / "continuous.checkpoint_step000002.npz"}"')
    for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"):
        np.testing.assert_allclose(restarted[name][-1], continuous[name][-1], rtol=0, atol=1.0e-12)
