"""Bounded actual-HSX spawn-pool replay against the serial RK4 smoke files."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import os
import sys
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.q_fci_return_campaign import campaign as c


def work(job):
    ids, path = job
    c.do_trace((ids, path, 'bounded-rk4-pool-check'))
    return {'pid': os.getpid(), 'path': path, 'peak_rss_gib': c.rss()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    jobs = []
    controls = {}
    for i in (0, 1):
        with np.load(a.output / f'smoke_64_{i}.npz') as z:
            control = {k: z[k][:4] for k in ('ids', 'valid', 'endpoint', 'ell', 'numerical')}
        path = str(a.output / f'pool_{i}.npz')
        controls[path] = control
        jobs.append((control['ids'].tolist(), path))
    args = SimpleNamespace(input_root=a.workspace, output=a.output, workers=2)
    results = list(c.pool_jobs(args, 32, jobs, work, trace_capacity=16))
    assert len({r['pid'] for r in results}) == 2
    for result in results:
        with np.load(result['path']) as z:
            control = controls[result['path']]
            assert c.completed(Path(result['path']), 'bounded-rk4-pool-check', control['ids'].tolist())
            np.testing.assert_array_equal(z['valid'], control['valid'])
            for key in ('endpoint', 'ell', 'numerical'):
                np.testing.assert_allclose(z[key], control[key], rtol=1e-10, atol=1e-12, equal_nan=True)
    c.write(a.output / 'pool_validation.json', {'passed': True, 'workers': results, 'source': c.source_identity(),
            'scope': 'Two spawned macOS workers, four real N32 observations each; Linux affinity requires remote verification.'})
    print('Two-worker actual-HSX RK4 replay passed', flush=True)


if __name__ == '__main__':
    main()
