"""Local producer adapter: freeze scalar-owner and continuous-reference inputs.

Only this preparation step uses the historical HSX workspace loaders. The
resulting directory is self-contained for campaign.py. No numerical solver
source is modified, and no global execution is implied by preparation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import importlib.util
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time
import numpy as np
from .common import SCHEMA, FIELDS, atomic_json, sha, source_identity, load_inputs
from .numerics import POLICY, centered_owner_moments
from .campaign import exclusive

REFERENCE_STATE = None


def prototype(workspace):
    path = Path(workspace)/'work/parallel_q03_direct_owner_flux_20260921/compare.py'
    spec = importlib.util.spec_from_file_location('q03_direct_frozen_prototype', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analytic_flux(compact, ctx, faces, ids, q):
    points, weights, J, B, Bmag = compact.face_quadrature_batch(ctx, faces, ids, q)
    unit = B/Bmag[..., None]
    normal = unit[np.arange(len(ids)), :, faces['axis'][ids]]
    factor = weights*J*normal
    flux = np.empty((len(ids), len(FIELDS)))
    for j, name in enumerate(FIELDS):
        field = ctx['fields'][name]['field']
        gradient = field.value_gradient_hessian(points.reshape(-1, 3))[1].reshape(points.shape)*float(field.amplitude)
        flux[:, j] = np.sum(factor*np.sum(unit*gradient, axis=2), axis=1)
    return flux


def initialize_references(workspace, N):
    global REFERENCE_STATE
    c = prototype(workspace)
    ctx = c.compact.load_context(N)
    faces = dict(np.load(Path(workspace)/f'work/parallel_q03_compact_corrected_20260920/N{N}/canonical_faces.npz'))
    REFERENCE_STATE = c.compact, ctx, faces


def reference_chunk(job):
    ids, path = job
    path = Path(path)
    record = path.with_suffix('.json')
    if record.exists():
        r = json.loads(record.read_text())
        if not path.exists() or sha(path) != r['sha256']:
            raise RuntimeError(f'corrupt reference chunk: {path}')
        with np.load(path) as z:
            if not np.array_equal(z['ids'], ids):
                raise RuntimeError('reference chunk face mismatch')
        return path
    compact, ctx, faces = REFERENCE_STATE
    began = time.monotonic()
    low, high = np.empty((len(ids),4)), np.empty((len(ids),4))
    for lo in range(0, len(ids), 24):
        hi = min(lo+24, len(ids))
        low[lo:hi] = analytic_flux(compact, ctx, faces, ids[lo:hi], 9)
        high[lo:hi] = analytic_flux(compact, ctx, faces, ids[lo:hi], 11)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise RuntimeError('nonfinite continuum reference flux')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, ids=ids, q9=low, q11=high)
    temp.replace(path)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_json(record, {'sha256':sha(path), 'seconds':time.monotonic()-began,
                         'peak_rss_bytes':int(rss if sys.platform=='darwin' else rss*1024)})
    return path


def prepare(workspace, N, output, sample=None, reference_mode='continuous', fresh_references=False, reference_workers=1):
    global REFERENCE_STATE
    if reference_workers < 1:
        raise ValueError('reference worker count must be positive')
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    began = time.monotonic()
    with exclusive(output/'prepare.lock'):
        c = prototype(workspace)
        ctx = c.compact.load_context(N)
        xyz, labels, rv, volume, centers, planes, order, ptr, states = c.owner_data(ctx)
        d = c.data(N)
        old_manifest = json.loads((c.INPUT/f'N{N}/manifest.json').read_text())
        if sha(ctx['manifest']) != old_manifest['source_inputs']['reference_identity']['reference']['geometry_manifest_sha256']:
            raise RuntimeError('geometry differs from frozen face-target geometry')
        if not np.array_equal(volume, d['volume']):
            raise RuntimeError('owner-volume identity mismatch')
        # Verify only the inputs this producer actually consumes, not the
        # multi-GB directional candidate pools which this formulation avoids.
        names = ['volume', 'target'] + ['face.'+k for k in ('center','scale','minus','plus','boundary','plane')]
        names += sorted(k for k in d if k.startswith('region.'))
        if reference_mode == 'frozen':
            names += ['ref_flux.'+f for f in FIELDS]
        for name in names:
            if sha(c.INPUT/f'N{N}/{name}.npy') != old_manifest['files'][name+'.npy']['sha256']:
                raise RuntimeError(f'frozen input changed: {name}')
        ids = np.arange(len(d['face.minus']))
        owners = np.arange(len(volume))
        sample_identity = None
        if sample:
            sample = Path(sample).resolve()
            with np.load(sample/'comparison.npz') as z:
                ids, owners = z['faces'].copy(), z['owners'].copy()
            required = np.flatnonzero(np.isin(d['face.minus'], owners) | np.isin(d['face.plus'], owners))
            if not np.array_equal(ids, required):
                raise RuntimeError('sample is not a complete-owner face union')
            sample_identity = {name: sha(sample/name) for name in
                               ('comparison.npz', 'qualified_reference.npz', 'reference_qualification.json')}
            qmeta = json.loads((sample/'reference_qualification.json').read_text())
            if qmeta['N'] != N or qmeta['identity']['geometry'] != sha(ctx['manifest']) or qmeta['identity']['comparison'] != sample_identity['comparison.npz'] or qmeta['identity']['artifact'] != sample_identity['qualified_reference.npz']:
                raise RuntimeError('sample reference identity mismatch')
        # Include every imported research/package source file already loaded;
        # changing a producer helper invalidates resumable reference chunks.
        imported = {}
        for module in tuple(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file:
                p = Path(file).resolve()
                if p.suffix == '.py' and p.is_relative_to(workspace) and p.exists():
                    imported[str(p.relative_to(workspace))] = sha(p)
        identity = {'source': source_identity(), 'producer_imports': imported,
                    'geometry': sha(ctx['manifest']), 'frozen_manifest': sha(c.INPUT/f'N{N}/manifest.json'),
                    'field_manifests': {f: ctx['fields'][f]['manifest'] for f in FIELDS},
                    'resolution': N, 'policy': POLICY, 'sample': sample_identity, 'reference_mode': reference_mode,
                    'fresh_references': fresh_references}
        # Normalize numpy values before comparing serialized identities.
        identity = json.loads(json.dumps(identity, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))
        receipt = output/'preparation.json'
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise RuntimeError('incompatible prepared directory; select a new output')
        atomic_json(receipt, identity)
        if (output/'manifest.json').exists():
            load_inputs(output)
            return {'reused': True, 'output': str(output)}
        payload = {
            'owner.centers': centers, 'owner.planes': planes, 'owner.states': states,
            'owner.moments': centered_owner_moments(xyz, labels, rv, volume, centers, N),
            'volume': volume, 'complete_owners': owners, 'face.ids': ids,
            'target': d['target'][ids, :20],
        }
        for k in ('center','scale','minus','plus','boundary','plane'):
            payload['face.'+k] = d['face.'+k][ids]
        payload.update({k: d[k] for k in names if k.startswith('region.')})
        qualification_available = reference_mode == 'continuous'
        if sample and reference_mode == 'continuous' and not fresh_references:
            with np.load(sample/'qualified_reference.npz') as z:
                high, low = z['flux_q11'].copy(), z['flux_q9'].copy()
        elif reference_mode == 'frozen':
            high = np.column_stack([d['ref_flux.'+f][ids] for f in FIELDS])
            low = high.copy()
        else:
            faces = dict(np.load(workspace/f'work/parallel_q03_compact_corrected_20260920/N{N}/canonical_faces.npz'))
            for key in ('minus','plus','boundary'):
                if not np.array_equal(faces[key], d['face.'+key]):
                    raise RuntimeError('canonical face identity mismatch')
            high, low = np.empty((len(ids),4)), np.empty((len(ids),4))
            # Short checkpoint units; fixed quadrature batch bounds evaluator
            # memory regardless of total resolution. Resume validates hashes.
            jobs = iter((ids[start:min(start+240,len(ids))],
                         output/'reference_chunks'/f'{start:08d}_{min(start+240,len(ids)):08d}.npz')
                         for start in range(0,len(ids),240))
            completed = 0
            def collect(path):
                nonlocal completed
                with np.load(path) as z:
                    positions = np.searchsorted(ids, z['ids'])
                    low[positions], high[positions] = z['q9'], z['q11']
                    completed += len(positions)
                atomic_json(output/'preparation_status.json', {'state':'references', 'faces':completed,
                            'total':len(ids), 'seconds':time.monotonic()-began})
            if reference_workers == 1:
                REFERENCE_STATE = c.compact, ctx, faces
                for job in jobs:
                    collect(reference_chunk(job))
            else:
                with ProcessPoolExecutor(max_workers=reference_workers,
                         mp_context=multiprocessing.get_context('spawn'),
                         initializer=initialize_references, initargs=(str(workspace),N)) as pool:
                    pending = set()
                    for _ in range(2*reference_workers):
                        job = next(jobs,None)
                        if job is not None:
                            pending.add(pool.submit(reference_chunk,job))
                    while pending:
                        ready,pending = wait(pending,return_when=FIRST_COMPLETED)
                        for future in ready:
                            collect(future.result())
                            job = next(jobs,None)
                            if job is not None:
                                pending.add(pool.submit(reference_chunk,job))
        # Keep boundary data fixed while qualifying the interior operator.
        low[payload['face.boundary']] = high[payload['face.boundary']]
        payload['reference.high'], payload['reference.low'] = high, low
        for name, array in payload.items():
            with (output/(name+'.npy.tmp')).open('wb') as stream:
                np.save(stream, array, allow_pickle=False)
            (output/(name+'.npy.tmp')).replace(output/(name+'.npy'))
        manifest = {'schema': SCHEMA, 'resolution': N, 'scope':'sample' if sample else 'global',
                    'identity':identity, 'fields': FIELDS,
                    'reference': {'orders':[9,11] if qualification_available else None,
                                  'qualification_available': qualification_available,
                                  'boundary':'identical prescribed high-order flux'},
                    'files':{name+'.npy':sha(output/(name+'.npy')) for name in payload}}
        atomic_json(output/'manifest.json', manifest)
        result = {'state':'prepared', 'faces':len(ids), 'owners':len(owners), 'scope':manifest['scope'],
                  'seconds':time.monotonic()-began, 'manifest_sha256':sha(output/'manifest.json')}
        atomic_json(output/'preparation_status.json', result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', required=True); p.add_argument('--N', type=int, choices=(32,48,64), required=True)
    p.add_argument('--output', required=True); p.add_argument('--sample', help='existing bounded comparison directory')
    p.add_argument('--reference-mode', choices=('continuous','frozen'), default='continuous')
    p.add_argument('--fresh-references', action='store_true', help='recompute even if sample has qualified saved references')
    p.add_argument('--reference-workers', type=int, default=1, help='independent continuous-reference producers; each loads geometry once')
    a = p.parse_args()
    print(json.dumps(prepare(a.workspace, a.N, a.output, a.sample, a.reference_mode, a.fresh_references, a.reference_workers)))
