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
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPOSITORY / "simulate_hsx_blob.py"


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


def test_saved_step_and_periodic_checkpoint_use_the_global_step():
    source = DRIVER_PATH.read_text()
    tree = ast.parse(source)
    run = _function(tree, "run_full_eb")
    assert "start_step" in [arg.arg for arg in run.args.kwonlyargs]
    save = ast.get_source_segment(source, _function(run, "save_snapshot"))
    assert save is not None
    body = save.split('"""')[-1] if '"""' in save else save
    # The offset is applied before the step reaches the payload or file name.
    offset_at = body.index("step = int(step) + int(start_step)")
    assert offset_at < body.index('payload["step"]')
    assert offset_at < body.index("checkpoint_step{int(step):06d}")
    run_source = ast.get_source_segment(source, run)
    assert "(step + int(start_step)) % int(checkpoint_every) == 0" in run_source
    main = ast.get_source_segment(source, _function(tree, "main"))
    assert "_restart_step_offset(args.restart_from) if restart_used else 0" in main
