"""Fast tests for ``scripts/p_shared/runner.py``'s per-unit peak-RSS
measurement (``/proc`` VmHWM reset with an ``ru_maxrss`` fallback) and the
stage summary's controller-vs-worker RSS split. ``/proc/self`` is replaced by
a temp directory, so nothing here depends on the host OS."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from p_shared import runner  # noqa: E402

UNIT = {"stage": "cells", "n": 4, "start": 0, "stop": 8}
IDENTITY = {"id": "runner-rss-test"}


def _fake_proc(tmp_path, vmhwm_kib=2 * 2 ** 20):
    proc = tmp_path / "proc_self"
    proc.mkdir()
    (proc / "status").write_text(f"Name:\tpython\nVmPeak:\t 9999999 kB\nVmHWM:\t {vmhwm_kib} kB\nVmRSS:\t 1 kB\n")
    (proc / "clear_refs").write_text("")
    return proc


def _write_unit(output):
    return runner.write_unit(output, UNIT, IDENTITY, chunks={"chunk": {"x": np.arange(3)}}, started=time.time())


def test_reset_peak_rss_uses_proc_vmhwm_and_receipt_records_the_method(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_PROC_SELF", _fake_proc(tmp_path))
    monkeypatch.setattr(runner, "_RSS_METHOD", "ru_maxrss")
    assert runner.reset_peak_rss() == "vmhwm_reset"
    assert (runner._PROC_SELF / "clear_refs").read_text() == "5"
    assert runner.peak_rss_gib() == pytest.approx(2.0)

    output = tmp_path / "out"
    receipt = _write_unit(output)
    assert receipt["rss_method"] == "vmhwm_reset"
    assert receipt["peak_rss_gib"] == pytest.approx(2.0)
    on_disk = json.loads(runner.receipt_path(output, UNIT).read_text())
    assert on_disk["rss_method"] == "vmhwm_reset" and on_disk["peak_rss_gib"] == pytest.approx(2.0)
    # The extra receipt keys do not disturb resume validation.
    assert runner.valid_unit(output, UNIT, IDENTITY)


def test_run_unit_resets_the_window_before_computing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_PROC_SELF", _fake_proc(tmp_path))
    monkeypatch.setattr(runner, "_RSS_METHOD", "ru_maxrss")
    seen = {}

    def compute(unit):
        seen["clear_refs"] = (runner._PROC_SELF / "clear_refs").read_text()
        seen["method"] = runner._RSS_METHOD
        return unit["start"]

    assert runner._run_unit(compute, UNIT) == 0
    assert seen == {"clear_refs": "5", "method": "vmhwm_reset"}


def test_peak_rss_falls_back_to_ru_maxrss_without_proc(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_PROC_SELF", tmp_path / "no_such_proc")
    monkeypatch.setattr(runner, "_RSS_METHOD", "vmhwm_reset")  # a stale success must not stick
    assert runner.reset_peak_rss() == "ru_maxrss"
    assert runner.peak_rss_gib() == pytest.approx(runner._ru_maxrss_gib())
    assert runner.peak_rss_gib() > 0.0

    output = tmp_path / "out"
    receipt = _write_unit(output)
    assert receipt["rss_method"] == "ru_maxrss"
    assert receipt["peak_rss_gib"] > 0.0
    assert runner.valid_unit(output, UNIT, IDENTITY)


def test_peak_rss_falls_back_when_the_reset_write_fails(tmp_path, monkeypatch):
    proc = _fake_proc(tmp_path)
    (proc / "clear_refs").unlink()
    (proc / "clear_refs").mkdir()  # writing a directory raises OSError
    monkeypatch.setattr(runner, "_PROC_SELF", proc)
    monkeypatch.setattr(runner, "_RSS_METHOD", "ru_maxrss")
    assert runner.reset_peak_rss() == "ru_maxrss"


def test_stage_summary_reports_controller_and_worker_peaks_separately(tmp_path):
    summary = runner.run_stage(tmp_path, "cells", [], IDENTITY, compute=None, initializer=None, initargs=(), workers=1)
    assert summary["executed_units"] == 0
    assert summary["peak_worker_rss_gib"] == 0.0
    assert summary["controller_peak_rss_gib"] > 0.0
    written = list((tmp_path / "executions").glob("*_cells.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["controller_peak_rss_gib"] > 0.0
