"""Portable, bounded-memory direct-flux setup/action runner.

Run as ``python -m q03_direct_campaign.campaign`` with scripts on PYTHONPATH.
No geometry evaluator, JAX import, or historical research module is needed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
from .common import SCHEMA, FIELDS, atomic_json, load_inputs, sha, source_identity
from .numerics import POLICY, OwnerIndex, owner_matrix, fit_batch

STATE = None


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'campaign already running: {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def initialize(input_path, identity, batch_size, save_coefficients):
    global STATE
    # Parent validates all hashes once. Worker mappings are read-only.
    d, meta = load_inputs(input_path, verify=False)
    index = OwnerIndex(d['owner.centers'], d['owner.planes'], meta['resolution'])
    STATE = d, meta, index, identity, batch_size, save_coefficients


def valid_chunk(path, identity, start, stop):
    path = Path(path)
    receipt = path.with_suffix('.json')
    if not receipt.exists():
        return False  # interrupted write before receipt: recompute safely
    r = json.loads(receipt.read_text())
    if r['identity'] != identity or r['range'] != [start, stop]:
        raise RuntimeError(f'incompatible chunk identity: {path}')
    if not path.exists() or sha(path) != r['sha256']:
        raise RuntimeError(f'corrupt chunk: {path}')
    return True


def compute(job):
    start, stop, path = job
    path = Path(path)
    d, meta, index, identity, batch_size, save_coefficients = STATE
    if valid_chunk(path, identity, start, stop):
        return {'start': start, 'stop': stop, 'reused': True}
    began = time.monotonic()
    count = stop-start
    # Boundaries use the same high-order prescribed flux as the reference.
    flux = np.array(d['reference.high'][start:stop])
    donors_all = np.full((count, 120), -1, dtype=np.int64) if save_coefficients else None
    coefficients_all = np.zeros((count, 120)) if save_coefficients else None
    diagnostics = {name: np.full(count, np.nan) for name in
                   ('rank', 'condition', 'cubic_defect', 'constant_flux', 'candidate_count')}
    interior = np.flatnonzero(~d['face.boundary'][start:stop]) + start
    stage_seconds = {'selection': 0., 'moments': 0., 'fit_action': 0.}
    for lo in range(0, len(interior), batch_size):
        ids = interior[lo:lo+batch_size]
        t = time.monotonic()
        selected = [index.select(d['face.center'][f], d['face.plane'][f],
                                 d['face.minus'][f], d['face.plus'][f]) for f in ids]
        donors = np.stack([v[0] for v in selected])
        distance2 = np.stack([v[1] for v in selected])
        stage_seconds['selection'] += time.monotonic()-t
        t = time.monotonic()
        P = owner_matrix(donors, d['owner.centers'], d['owner.moments'],
                         d['face.center'][ids], d['face.scale'][ids], meta['resolution'])
        stage_seconds['moments'] += time.monotonic()-t
        t = time.monotonic()
        coefficient, diag = fit_batch(P, distance2, d['target'][ids])
        flux[ids-start] = np.einsum('fi,fij->fj', coefficient, d['owner.states'][donors])
        stage_seconds['fit_action'] += time.monotonic()-t
        for name, value in diag.items():
            diagnostics[name][ids-start] = value
        diagnostics['candidate_count'][ids-start] = [sum(v[2]) for v in selected]
        if save_coefficients:
            donors_all[ids-start] = donors
            coefficients_all[ids-start] = coefficient
    if not np.all(np.isfinite(flux)):
        raise RuntimeError('nonfinite flux')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    payload = dict(face_ids=d['face.ids'][start:stop], flux=flux, **diagnostics)
    if save_coefficients:
        payload.update(donors=donors_all, coefficients=coefficients_all)
    with tmp.open('wb') as stream:
        np.savez(stream, **payload)  # bounded, uncompressed; no giant cache archive
    os.replace(tmp, path)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {'identity': identity, 'range': [start, stop], 'sha256': sha(path),
               'seconds': time.monotonic()-began, 'stage_seconds': stage_seconds,
               'peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss*1024),
               'interior_faces': len(interior), 'reused': False}
    atomic_json(path.with_suffix('.json'), receipt)
    return {k: receipt[k] for k in ('seconds', 'stage_seconds', 'peak_rss_bytes', 'reused')} | {'start': start, 'stop': stop}


def jobs_for(output, faces, chunk_size):
    for lo in range(0, faces, chunk_size):
        hi = min(lo+chunk_size, faces)
        yield lo, hi, Path(output)/'chunks'/f'faces_{lo:08d}_{hi:08d}.npz'


def scatter(total, d, lo, hi, flux):
    np.add.at(total, d['face.minus'][lo:hi], flux)
    good = d['face.plus'][lo:hi] >= 0
    np.add.at(total, d['face.plus'][lo:hi][good], -flux[good])


def assemble(d, meta, output, identity, chunk_size):
    output = Path(output)
    totals = {name: np.zeros((len(d['volume']), len(FIELDS))) for name in ('direct', 'high', 'low')}
    max_defect = 0.
    for lo, hi, path in jobs_for(output, len(d['face.ids']), chunk_size):
        if not valid_chunk(path, identity, lo, hi):
            raise RuntimeError(f'missing chunk: {path}')
        with np.load(path) as z:
            if not np.array_equal(z['face_ids'], d['face.ids'][lo:hi]):
                raise RuntimeError('chunk face coverage mismatch')
            scatter(totals['direct'], d, lo, hi, z['flux'])
            finite = np.isfinite(z['cubic_defect'])
            if finite.any():
                max_defect = max(max_defect, float(z['cubic_defect'][finite].max()))
        for key in ('high', 'low'):
            scatter(totals[key], d, lo, hi, d['reference.'+key][lo:hi])
    owners = d['complete_owners']
    volume = d['volume'][owners]
    action = {key: value[owners]/volume[:, None] for key, value in totals.items()}
    # Compare low/high INTERNAL reference only: the action and each reference
    # must share the same prescribed high-order boundary flux.
    metrics = {}
    for j, field in enumerate(FIELDS):
        error = action['direct'][:, j]-action['high'][:, j]
        reference_delta = action['high'][:, j]-action['low'][:, j]
        sse = float(np.dot(volume, error**2))
        rms = float(np.sqrt(sse/volume.sum()))
        ref_rms = float(np.sqrt(np.dot(volume, reference_delta**2)/volume.sum()))
        regions = {}
        for name in sorted(k for k in d if k.startswith('region.')):
            mask = d[name][owners]
            regional_sse = float(np.dot(volume[mask], error[mask]**2))
            regions[name[7:]] = {'owners': int(mask.sum()), 'squared_error': regional_sse,
                                  'global_SSE_fraction': regional_sse/sse if sse else 0.}
        metrics[field] = {'L2': rms, 'Linf': float(np.max(abs(error))),
                          'reference_delta_L2': ref_rms,
                          'reference_fraction': ref_rms/rms if rms else None,
                          'reference_qualified': bool(meta['reference']['qualification_available'] and (ref_rms < .1*rms or field == 'constant')),
                          'regions': regions}
    tmp = output/'actions.npz.tmp'
    with tmp.open('wb') as stream:
        np.savez(stream, owners=owners, volume=volume, **action)
    os.replace(tmp, output/'actions.npz')
    field_contract = {f: {
        'field': manifest['field'], 'sampling': manifest['sampling_convention'],
        'continuous_geometry': {key: manifest['provenance'].get(key) for key in
             ('metric_cache_sha256', 'makegrid_sha256', 'reference_magnetic_field')},
    } for f, manifest in meta['identity'].get('field_manifests', {}).items()}
    result = {'schema': SCHEMA, 'resolution': meta['resolution'], 'identity': identity,
              'field_contract': field_contract, 'diffusivity': 1.0,
              'scope': meta['scope'], 'completed': True, 'faces': len(d['face.ids']),
              'complete_owners': len(owners), 'metrics': metrics,
              'max_cubic_defect': max_defect,
              'algebra_checks_passed': bool(max_defect <= 1e-10),
              'structural_properties_certified': False, 'convergence_passed': False,
              'actions_sha256': sha(output/'actions.npz')}
    atomic_json(output/'summary.json', result)
    return result


def run(input_path, output, workers=1, chunk_size=512, batch_size=32, save_coefficients=False):
    if min(workers, chunk_size, batch_size) < 1:
        raise ValueError('worker/chunk/batch counts must be positive')
    output = Path(output)
    with exclusive(output/'campaign.lock'):
        began = time.monotonic()
        d, meta = load_inputs(input_path)
        identity = {'schema': SCHEMA, 'input': sha(Path(input_path)/'manifest.json'),
                    'source': source_identity(), 'policy': POLICY,
                    'chunk_size': chunk_size, 'batch_size': batch_size,
                    'save_coefficients': save_coefficients}
        run_manifest = output/'run.json'
        if run_manifest.exists() and json.loads(run_manifest.read_text())['identity'] != identity:
            raise RuntimeError('incompatible run: choose a new output directory')
        atomic_json(run_manifest, {'identity': identity, 'input_path': str(Path(input_path).resolve()),
                                   'workers': workers, 'pid': os.getpid()})
        jobs = iter(jobs_for(output, len(d['face.ids']), chunk_size))
        finished = 0

        def record(r):
            nonlocal finished
            finished += r['stop']-r['start']
            atomic_json(output/'status.json', {'state': 'running', 'completed_faces': finished,
                        'total_faces': len(d['face.ids']), 'seconds': time.monotonic()-began, 'last_chunk': r})

        try:
            if workers == 1:
                initialize(input_path, identity, batch_size, save_coefficients)
                for job in jobs:
                    record(compute(job))
            else:
                # Spawn works on macOS/Linux. Limit queued jobs to 2 per worker;
                # chunks return small receipts, not coefficients through IPC.
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                         initializer=initialize, initargs=(input_path, identity, batch_size, save_coefficients)) as pool:
                    pending = set()
                    for _ in range(2*workers):
                        job = next(jobs, None)
                        if job is not None:
                            pending.add(pool.submit(compute, job))
                    while pending:
                        ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in ready:
                            record(future.result())
                            job = next(jobs, None)
                            if job is not None:
                                pending.add(pool.submit(compute, job))
            result = assemble(d, meta, output, identity, chunk_size)
            atomic_json(output/'status.json', {'state': 'completed', 'exit_code': 0,
                         'seconds': time.monotonic()-began, 'summary_sha256': sha(output/'summary.json')})
            return result
        except BaseException as exc:
            atomic_json(output/'status.json', {'state': 'failed', 'error': repr(exc),
                         'seconds': time.monotonic()-began, 'completed_faces': finished})
            raise


def orders(outputs, destination):
    summaries = sorted([json.loads((Path(p)/'summary.json').read_text()) for p in outputs], key=lambda s: s['resolution'])
    if [s['resolution'] for s in summaries] != [32, 48, 64]:
        raise ValueError('require exactly N32/N48/N64')
    if any(not s['completed'] or s['scope'] != 'global' for s in summaries):
        raise ValueError('only complete global runs can qualify convergence')
    first = summaries[0]['identity']
    if any(s['identity']['source'] != first['source'] or s['identity']['policy'] != first['policy'] for s in summaries):
        raise ValueError('incompatible source or numerical policy across resolutions')
    if not summaries[0].get('field_contract') or any(s.get('field_contract') != summaries[0]['field_contract'] or s.get('diffusivity') != 1.0 for s in summaries):
        raise ValueError('incompatible or absent field/continuous-reference contract')
    fields = {}
    for f in FIELDS[:-1]:
        e = [s['metrics'][f]['L2'] for s in summaries]
        p = [float(np.log(e[i]/e[i+1])/np.log(summaries[i+1]['resolution']/summaries[i]['resolution'])) for i in (0, 1)]
        qualified = all(s['metrics'][f]['reference_qualified'] for s in summaries)
        fields[f] = {'L2': e, 'orders': p, 'reference_qualified': qualified,
                     'passed': bool(qualified and all(v >= 1.8 for v in p))}
    result = {'fields': fields, 'global_operator_convergence_passed': all(v['passed'] for v in fields.values()),
              'Q04_certified': False, 'solution_and_structural_certification': 'pending',
              'summary_hashes': {str(Path(p).resolve()): sha(Path(p)/'summary.json') for p in outputs}}
    atomic_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=1); p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--save-coefficients', action='store_true')
    p = sub.add_parser('orders'); p.add_argument('outputs', nargs=3); p.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.command == 'run':
        result = run(a.input, a.output, a.workers, a.chunk_size, a.batch_size, a.save_coefficients)
        print(json.dumps({k: result[k] for k in ('completed', 'resolution', 'scope', 'faces', 'max_cubic_defect')}))
    else:
        print(json.dumps(orders(a.outputs, a.output)))


if __name__ == '__main__':
    main()
