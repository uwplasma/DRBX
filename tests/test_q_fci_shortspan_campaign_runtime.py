"""Portable process startup and worker-memory reporting for the Q campaign."""
from types import SimpleNamespace

import pytest

from scripts.q_fci_shortspan_midpoint_global import numerical_runner as nr


@pytest.mark.parametrize('platform,reported', [('darwin', 3 * 1024**3), ('linux', 3 * 1024**2)])
def test_peak_rss_units(monkeypatch, platform, reported):
    monkeypatch.setattr(nr, 'sys', SimpleNamespace(platform=platform))
    monkeypatch.setattr(nr.resource, 'getrusage', lambda _: SimpleNamespace(ru_maxrss=reported))
    assert nr.peak_rss_gib() == 3.


def test_parallel_dispatch_uses_spawn_and_keeps_complete_groups(monkeypatch, tmp_path):
    seen = {}

    class Pool:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, fn, jobs):
            seen['function'] = fn
            seen['jobs'] = list(jobs)
            return [dict(skipped=False, cpu_s=1.) for _ in seen['jobs']]

    monkeypatch.setattr(nr, 'validate', lambda *args: {'method': {'owner_batch_max': 64}})
    monkeypatch.setattr(nr, 'sha', lambda *args: 'frozen-design')
    monkeypatch.setattr(nr, 'worker_policy', lambda *args: {'effective_workers': 2})
    monkeypatch.setattr(nr, 'write', lambda *args: None)
    monkeypatch.setattr(nr.concurrent.futures, 'ProcessPoolExecutor', Pool)
    result = nr.run_jobs(32, [[0, 1], [2]], 'inputs', tmp_path, 'exact', 2, 6., 3.)
    assert seen['mp_context'].get_start_method() == 'spawn'
    assert seen['initializer'] is nr.init_worker
    assert seen['jobs'] == [([0, 1], 'frozen-design'), ([2], 'frozen-design')]
    assert result['owners'] == 3 and result['chunks'] == 2
    assert result['computed'] == 2 and result['chunk_cpu_s'] == 2.
