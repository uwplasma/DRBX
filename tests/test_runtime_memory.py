from __future__ import annotations

import os

from jax_drb.runtime.memory import (
    bytes_to_mebibytes,
    measure_peak_rss,
    process_tree_pids,
    process_tree_rss_bytes,
)


def test_bytes_to_mebibytes_handles_none_and_values() -> None:
    assert bytes_to_mebibytes(None) is None
    assert bytes_to_mebibytes(2 * 1024 * 1024) == 2.0


def test_process_tree_helpers_include_current_process() -> None:
    pids = process_tree_pids(os.getpid())

    assert os.getpid() in pids
    assert process_tree_rss_bytes(os.getpid()) is not None


def test_measure_peak_rss_returns_result_and_measurement() -> None:
    result, measurement = measure_peak_rss(lambda: "done", sampling_interval_seconds=0.01)

    assert result == "done"
    assert measurement.status in {
        "sampled_process_tree_rss",
        "sampled_process_tree_rss_with_partial_failures",
    }
    assert measurement.sample_count >= 1
    assert measurement.peak_rss_bytes is None or measurement.peak_rss_bytes > 0
