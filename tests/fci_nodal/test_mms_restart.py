from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from drbx.fci_nodal.case import case as driver
from drbx.fci_nodal.case._receipts import code_files, stage
from drbx.fci_nodal.mms.restart import Checkpoint, Paused, SEGMENT_KEYS
from drbx.fci_nodal.options import RuntimeOptions


@pytest.fixture
def snapshot():
    state = np.array([[1., -0., np.nextafter(1., 2.), 0.5, -4., 7.]], dtype=np.float64)
    progress = dict(time=.005, cache_time=np.nextafter(.005, 0.).item(), steps=1,
                    bounds=dict(dt=.001, positivity_bound=dict(minimum=.002, components=[-0., .1])),
                    stages=[[dict(ampere=dict(iterations=7, residual=-0.))]],
                    targets=[dict(time=.005, steps=1, partials=[dict(all=[1, .125, [0.], [0.]])])],
                    seconds=[.456789123456789])
    return state, progress


def test_checkpoint_exact_round_trip(tmp_path, snapshot):
    state, progress = snapshot
    identity = dict(options={"wall_law": "floating_wall"}, threads="2")
    Checkpoint(tmp_path, identity).save(state, progress)
    restored, saved = Checkpoint(tmp_path, identity).load()
    assert restored.dtype == state.dtype and restored.shape == state.shape
    assert restored.tobytes() == state.tobytes()
    assert json.dumps(saved, sort_keys=True) == json.dumps(progress, sort_keys=True)


def test_torn_generation_keeps_previous_commit(tmp_path, snapshot, monkeypatch):
    state, progress = snapshot
    checkpoint = Checkpoint(tmp_path, {})
    checkpoint.save(state, progress)
    replace = Path.replace

    def interrupt(path, destination):
        if path.name == "manifest.pending.json":
            raise InterruptedError("interrupted before checkpoint publication")
        return replace(path, destination)

    # Reuse the first slot so its old manifest must have been invalidated.
    checkpoint.save(state+1, dict(progress, steps=2))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt)
        with pytest.raises(InterruptedError):
            checkpoint.save(state+2, dict(progress, steps=3))
    saved, meta = Checkpoint(tmp_path, {}).load()
    assert meta["steps"] == 2
    assert saved.tobytes() == (state+1).tobytes()


def test_checkpoint_mismatch_and_corruption(tmp_path, snapshot, capsys):
    Checkpoint(tmp_path, dict(threads="2")).save(*snapshot)
    assert Checkpoint(tmp_path, dict(threads="3")).load() is None
    assert capsys.readouterr().out.splitlines() == [
        "MMS checkpoint identity differs: threads; starting from time zero"]
    (tmp_path/"slot-0/owner0000.npy").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="file hash differs"):
        Checkpoint(tmp_path, dict(threads="3")).load()


def test_segment_controls_stay_out_of_stage_identity(tmp_path, monkeypatch):
    initial = (dict(identity=dict(options=dict(owner_count=1)), products={}, output_files={}),
               dict(stage="owner_inputs", sha256="inputs"))
    monkeypatch.setattr(driver, "verify_chain", lambda *_: {"owner_inputs": initial})
    controls = dict(segment_steps=6, segment_wall_seconds=120., checkpoint_interval_seconds=30.)

    def fake_stage(case, name, *, options, action, **kwargs):
        assert set(SEGMENT_KEYS).isdisjoint(options)
        if name == "mms_run":
            return action(case)
        return dict(products={}, output_files={}), dict(stage=name, sha256=name)

    def execute(name, payload, *args):
        assert name == "mms_run"
        invocation = payload["checkpoint"]
        assert invocation["controls"] == controls
        assert invocation["options"] == dict(runtime=asdict(RuntimeOptions()),
                                              field_profile="wall_flat_temperature")
        assert [stamp["stage"] for stamp in invocation["input_receipts"]] == [
            "owner_inputs", "wall_data", "mms_prepare"]
        return Paused(.001, 6), None

    monkeypatch.setattr(driver, "stage", fake_stage)
    monkeypatch.setattr(driver, "_execute", execute)
    options = dict(runtime=RuntimeOptions(), field_profile="wall_flat_temperature", **controls)
    assert driver.run_mms(tmp_path, options) == dict(paused=True, steps=6, time=.001)
    with pytest.raises(ValueError, match="require local launch mode"):
        driver.run_mms(tmp_path, dict(options, launch_mode="per_owner"))


def test_pause_leaves_previous_completed_receipt(tmp_path):
    settings = dict(NPROC="2")
    options = dict(runtime=asdict(RuntimeOptions()), max_steps=8)

    def publish(attempt):
        product = attempt/"result.json"
        product.write_text("{}\n")
        return {"summary": product}, 0, 0, code_files("mms_run")

    stage(tmp_path, "mms_run", settings=settings, options=options,
          inputs=(), expected=("summary",), action=publish)
    receipt = tmp_path/"mms_run/receipt.json"
    before = receipt.read_bytes()

    paused, stamp = stage(tmp_path, "mms_run", settings=settings,
                          options=dict(options, max_steps=9), inputs=(), expected=("summary",),
                          action=lambda _: (Paused(.005, 6), 0, 0, code_files("mms_run")))
    assert paused == Paused(.005, 6) and stamp is None
    assert receipt.read_bytes() == before
