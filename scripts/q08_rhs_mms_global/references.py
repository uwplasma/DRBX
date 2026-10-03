"""Spawn-parallel independent CPU references, computed once for every BC."""
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from campaign import load, read, write, sha, BASE_ID

ENV = None


def initialize(source, run, old, canonical, identity, n):
    global ENV
    # Pin before importing/initializing JAX, as in the original CPU campaign.
    # Otherwise every spawned reference process may create an allocation-wide
    # thread pool despite the BLAS thread settings.
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        slot = multiprocessing.current_process()._identity[-1]-1
        os.sched_setaffinity(0, {allowed[slot % len(allowed)]})
    load(source, run, gpu=False)
    from scripts.q08_extraction_global import common as c
    model = c.load_geometry_model(Path(old)/'inputs')
    ctx, topology = model.context(n, canonical)
    ENV = dict(run=Path(run), old=Path(old), identity=identity, n=n, t=topology,
               geom=lambda p: model.geom(ctx, p))


def valid(path, identity, input_receipt):
    receipt = path.with_suffix('.json')
    if not receipt.exists():
        return False
    r = read(receipt)
    if (r.get('identity') != identity or r.get('input_receipt') != input_receipt or
            r.get('passed') is not True or sha(path) != r.get('sha256')):
        raise ValueError('reference checkpoint corrupted or incompatible: '+str(path))
    return r


def one(index):
    from drbx.stencils.q_artifact import load_q_bank
    from scripts.q08_extraction_global.common import atomic_npz
    from scripts.q08_extraction_global.gpu import peak_rss_gib
    from science import oracle, continuum, TERMS
    e = ENV; n = e['n']; old = e['old']/'cpu'/f'N{n}'/f'chunk_{index:06d}'
    dest = e['run']/'references'/f'N{n}'/f'chunk_{index:06d}.npz'
    old_receipt = sha(old/'stats.json')
    prior = valid(dest, e['identity'], old_receipt)
    if prior:
        return prior
    tick = time.perf_counter()
    stats = read(old/'stats.json')
    if not stats['passed'] or stats['campaign_identity'] != BASE_ID:
        raise ValueError('invalid original chunk receipt')
    for name, h in stats['files'].items():
        if sha(old/name) != h:
            raise ValueError('original input changed')
    bank = load_q_bank(old/'bank.npz', expected_identity=stats['bank_identity'])
    with np.load(old/'geometry.npz') as z:
        geometry = {k: z[k].copy() for k in z.files}
    O = oracle(bank, geometry)
    R, diagnostic = continuum(bank, geometry, e['geom'])
    owners = bank.owners; t = e['t']
    if O.shape != (2, 22, len(owners), len(TERMS)) or R.shape != O.shape[1:] or not np.isfinite(O).all():
        raise ValueError('reference shape/finite gate')
    radial = (t.pts[t.order[t.starts[owners]], 0]*n).astype(int)
    atomic_npz(dest, O=O, R=R, owners=owners, volume=t.vol[owners], radial=radial)
    r = dict(passed=True, identity=e['identity'], n=n, index=index, owners=owners.tolist(),
             input_receipt=old_receipt, sha256=sha(dest), seconds=time.perf_counter()-tick,
             peak_rss_gib=peak_rss_gib(), diagnostics=diagnostic,
             cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
    if r['peak_rss_gib'] > 4:
        raise MemoryError('reference worker exceeded declared 4 GiB; payload preserved without pass receipt')
    write(dest.with_suffix('.json'), r)
    return r


def validate(run, old, identity, n):
    from science import TERMS
    plan = read(old/'inputs/plan.json')[str(n)]
    records = []; files = {}; owners = []
    for i, oo in enumerate(plan['chunks']):
        path = run/'references'/f'N{n}'/f'chunk_{i:06d}.npz'
        r = valid(path, identity, sha(old/'cpu'/f'N{n}'/f'chunk_{i:06d}'/'stats.json'))
        if not r or r['owners'] != oo or r['index'] != i or r['n'] != n:
            raise ValueError('reference coverage/receipt')
        with np.load(path) as z:
            if (set(z.files) != {'O', 'R', 'owners', 'volume', 'radial'} or
                z['O'].shape != (2, 22, len(oo), len(TERMS)) or z['R'].shape != (22, len(oo), len(TERMS)) or
                not np.array_equal(z['owners'], oo) or np.any(z['volume'] <= 0) or
                not all(np.isfinite(z[k]).all() for k in z.files)):
                raise ValueError('reference arrays invalid')
        records.append(r); owners.extend(oo)
        files[str(path.relative_to(run))] = r['sha256']
        files[str(path.with_suffix('.json').relative_to(run))] = sha(path.with_suffix('.json'))
    if not np.array_equal(owners, np.arange(plan['n_owner'])):
        raise ValueError('references do not cover every owner once in order')
    return dict(passed=True, identity=identity, n=n, owners=len(owners), chunks=len(records), files=files,
        reference_step_max_by_term=np.max([r['diagnostics']['reference_step_max_by_term'] for r in records], axis=0).tolist(),
        reference_cpu_work_seconds=sum(r['seconds'] for r in records),
        max_worker_rss_gib=max(r['peak_rss_gib'] for r in records))


def run_references(run, old, source, identity, canonical, n, workers, host_gib, *, pilot=False):
    if isinstance(workers, bool) or workers < 1 or not np.isfinite(host_gib) or host_gib < 8+4*workers:
        raise ValueError('reference resource budget requires 8 + 4*workers GiB')
    effective = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    if workers > effective:
        raise ValueError('reference workers exceed actual CPU affinity')
    plan = read(old/'inputs/plan.json')[str(n)]
    indices = list(range(len(plan['chunks'])))
    if pilot:
        indices = sorted(set([0, len(indices)//2, len(indices)-1]))
    tick = time.perf_counter(); results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(str(source), str(run), str(old), canonical, identity, n)) as pool:
        for r in pool.map(one, indices, chunksize=1):
            results.append(r)
            if len(results) % 50 == 0 or len(results) == len(indices):
                print(f'references N{n}: {len(results)}/{len(indices)}', flush=True)
    if pilot:
        result = dict(passed=True, identity=identity, n=n, workers=workers, host_gib=host_gib,
            indices=indices, wall_seconds=time.perf_counter()-tick,
            projected_reference_work_seconds=sum(r['seconds'] for r in results)/len(results)*sum(
                len(v['chunks']) for v in read(old/'inputs/plan.json').values())/workers,
            warning='three stratified N32 chunks only; excludes GPU stage and startup imbalance', records=results)
        write(run/'pilot.json', result)
    else:
        result = validate(run, old, identity, n)
        result.update(workers=workers, host_gib=host_gib, wall_seconds=time.perf_counter()-tick)
        write(run/f'references_N{n}.json', result)
    return result
