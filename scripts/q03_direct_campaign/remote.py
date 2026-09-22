"""Frozen, computation-only direct-cubic campaign; no historical workspace imports."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_PLATFORMS','cpu')

import numpy as np
import scipy
from . import campaign, continuous, prepare as producer
from .common import SCHEMA, FIELDS, atomic_json, sha, source_identity, load_inputs
from .numerics import POLICY

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def seed_data(path):
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in Path(path).glob('*.npy')}


def verify(args):
    pinned = read(HERE/'seed_manifest.json')
    if sha(args.seeds/'manifest.json') != pinned['manifest_sha256']:
        raise RuntimeError('wrong seed bundle')
    manifest = read(args.seeds/'manifest.json')
    if manifest['schema'] != 'q03-direct-seeds-v1' or manifest['policy'] != POLICY:
        raise RuntimeError('incompatible seed policy')
    for name,digest in manifest['files'].items():
        if sha(args.seeds/name) != digest:
            raise RuntimeError(f'changed/missing seed: {name}')
    for key in ('metric_cache','makegrid'):
        if sha(getattr(args,key)) != manifest['external'][key]['sha256']:
            raise RuntimeError(f'changed {key}')
    geometry_source = read(HERE/'geometry_source_manifest.json')
    for name,digest in geometry_source.items():
        if sha(REPO/name) != digest:
            raise RuntimeError(f'changed geometry source: {name}')
    identity = {'source':source_identity(),'seed_manifest':pinned['manifest_sha256'],
                'geometry_source':geometry_source,'external':manifest['external'],
                'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),
                'machine':platform.machine(),'policy':POLICY}
    path = args.output/'campaign.json'
    if path.exists() and read(path) != identity:
        raise RuntimeError('incompatible campaign identity; do not mix checkpoints')
    atomic_json(path,identity)
    atomic_json(args.output/'input_locations.json',{'seeds':str(args.seeds),'metric_cache':str(args.metric_cache),
                                                  'makegrid':str(args.makegrid)})
    return identity


def initialize_reference(seeds, metric_cache, makegrid):
    path = Path(seeds)
    d = seed_data(path); meta = read(path/'metadata.json')
    ctx = continuous.context(d,meta,metric_cache,makegrid)
    faces = {'axis':d['canonical.axis'],'storage':d['canonical.storage']}
    producer.REFERENCE_STATE = continuous, ctx, faces


def prepare_resolution(args, N, identity, sample=False):
    seed = args.seeds/f'N{N}'
    d = seed_data(seed); meta = read(seed/'metadata.json')
    output = args.output/('preflight' if sample else 'inputs')/f'N{N}'
    with campaign.exclusive(output/'prepare.lock'):
        began = time.monotonic()
        rid = {'campaign':identity,'resolution':N,'sample':sample,'orders':[9,11]}
        receipt = output/'preparation.json'
        if receipt.exists() and read(receipt) != rid:
            raise RuntimeError('incompatible reference preparation')
        atomic_json(receipt,rid)
        if (output/'manifest.json').exists():
            load_inputs(output)
            return output
        if sample:
            with np.load(seed/'preflight.npz') as z:
                ids,owners = z['faces'].copy(),z['owners'].copy()
            required = np.flatnonzero(np.isin(d['face.minus'],owners)|np.isin(d['face.plus'],owners))
            np.testing.assert_array_equal(ids,required)
        else:
            ids,owners = d['face.ids'],d['complete_owners']
        high,low = np.empty((len(ids),len(FIELDS))),np.empty((len(ids),len(FIELDS)))
        jobs = iter((ids[start:min(start+240,len(ids))],output/'reference_chunks'/f'{start:08d}_{min(start+240,len(ids)):08d}.npz')
                    for start in range(0,len(ids),240))
        completed = 0
        def collect(path):
            nonlocal completed
            with np.load(path) as z:
                pos = np.searchsorted(ids,z['ids']);low[pos],high[pos] = z['q9'],z['q11']
                completed += len(pos)
            atomic_json(output/'preparation_status.json',{'state':'references','faces':completed,
                'total':len(ids),'seconds':time.monotonic()-began})
        init = (str(seed),str(args.metric_cache),str(args.makegrid))
        if args.reference_workers == 1:
            initialize_reference(*init)
            for job in jobs:
                collect(producer.reference_chunk(job))
        else:
            with ProcessPoolExecutor(max_workers=args.reference_workers,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize_reference,initargs=init) as pool:
                pending=set()
                for _ in range(2*args.reference_workers):
                    job=next(jobs,None)
                    if job is not None:pending.add(pool.submit(producer.reference_chunk,job))
                while pending:
                    ready,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for f in ready:
                        collect(f.result())
                        job=next(jobs,None)
                        if job is not None:pending.add(pool.submit(producer.reference_chunk,job))
        payload={k:a for k,a in d.items() if k.startswith(('owner.','region.')) or k=='volume'}
        payload.update({k:d[k][ids] for k in ('target','face.center','face.scale','face.minus','face.plus','face.boundary','face.plane')})
        low[payload['face.boundary']]=high[payload['face.boundary']]
        payload.update({'complete_owners':owners,'face.ids':ids,'reference.high':high,'reference.low':low})
        for name,a in payload.items():
            temp=output/(name+'.npy.tmp')
            with temp.open('wb') as stream:np.save(stream,a,allow_pickle=False)
            temp.replace(output/(name+'.npy'))
        manifest={'schema':SCHEMA,'resolution':N,'scope':'sample' if sample else 'global','fields':FIELDS,
                  'identity':{'policy':POLICY,'campaign':identity,'field_manifests':meta['field_manifests'],
                              'geometry':meta['geometry_sha256']},
                  'reference':{'orders':[9,11],'qualification_available':True,'boundary':'identical prescribed high-order flux'},
                  'files':{name+'.npy':sha(output/(name+'.npy')) for name in payload}}
        atomic_json(output/'manifest.json',manifest)
        atomic_json(output/'preparation_status.json',{'state':'prepared','faces':len(ids),'seconds':time.monotonic()-began})
        return output


def preflight(args,identity):
    records=[]
    for N in (32,48,64):
        path=prepare_resolution(args,N,identity,sample=True)
        d,_=load_inputs(path)
        with np.load(args.seeds/f'N{N}/preflight.npz') as z:
            np.testing.assert_allclose(d['reference.high'],z['q11'],rtol=2e-9,atol=2e-13)
            q9=z['q9'].copy();q9[d['face.boundary']]=z['q11'][d['face.boundary']]
            np.testing.assert_allclose(d['reference.low'],q9,rtol=2e-9,atol=2e-13)
            outputs=[args.output/'preflight'/f'N{N}_{mode}' for mode in ('serial','parallel')]
            campaign.run(path,outputs[0],workers=1)
            campaign.run(path,outputs[1],workers=args.workers)
            with np.load(outputs[0]/'actions.npz') as a,np.load(outputs[1]/'actions.npz') as b:
                np.testing.assert_allclose(a['direct'],b['direct'],rtol=1e-11,atol=1e-11)
                # Saved prototype boundary flux is negligible; allow roundoff
                # amplified by tiny owner volumes, far below spatial errors.
                np.testing.assert_allclose(a['direct'],z['direct'],rtol=1e-8,atol=1e-8)
                diff=float(np.max(abs(a['direct']-z['direct'])))
            records.append({'N':N,'faces':len(z['faces']),'prototype_action_max_abs':diff,'passed':True})
    result={'identity':identity,'passed':True,'resolutions':records}
    atomic_json(args.output/'preflight.json',result)
    return result


def validate(args,identity):
    results=[]
    for N in (32,48,64):
        path=args.output/'inputs'/f'N{N}';out=args.output/'results'/f'N{N}'
        d,meta=load_inputs(path);run=read(out/'run.json');status=read(out/'status.json')
        if status['state']!='completed' or status['summary_sha256']!=sha(out/'summary.json'):
            raise RuntimeError(f'incomplete/changed result N{N}')
        if meta['identity']['campaign']!=identity or run['identity']['source']!=source_identity() or run['identity']['input']!=sha(path/'manifest.json'):
            raise RuntimeError(f'changed run identity N{N}')
        summary=read(out/'summary.json')
        if sha(out/'actions.npz')!=summary['actions_sha256']:
            raise RuntimeError('changed action artifact')
        campaign.assemble(d,meta,out,run['identity'],run['identity']['chunk_size'])
        if sha(out/'summary.json')!=status['summary_sha256']:
            raise RuntimeError('result reassembly differs')
        results.append(out)
    orders=campaign.orders(results,args.output/'operator_orders.json')
    receipt={'completed':True,'identity':identity,'global_operator_convergence_passed':orders['global_operator_convergence_passed'],
             'Q04_certified':False,'orders_sha256':sha(args.output/'operator_orders.json')}
    atomic_json(args.output/'validation.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify','preflight','run','validate'))
    p.add_argument('--seeds',type=Path,required=True);p.add_argument('--metric-cache',type=Path,required=True)
    p.add_argument('--makegrid',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int);p.add_argument('--reference-workers',type=int)
    args=p.parse_args()
    for k in ('seeds','metric_cache','makegrid','output'):setattr(args,k,getattr(args,k).resolve())
    if args.command in ('preflight','run') and (not args.workers or args.workers<1 or not args.reference_workers or args.reference_workers<1):
        p.error('positive --workers and --reference-workers must be chosen explicitly')
    began=time.monotonic()
    with campaign.exclusive(args.output/'controller.lock'):
        try:
            identity=verify(args)
            if args.command=='verify':result={'verified':True}
            elif args.command=='preflight':result=preflight(args,identity)
            elif args.command=='run':
                checked=read(args.output/'preflight.json')
                if checked['identity']!=identity or not checked['passed']:
                    raise RuntimeError('matching preflight required')
                for N in (32,48,64):
                    path=prepare_resolution(args,N,identity)
                    campaign.run(path,args.output/'results'/f'N{N}',workers=args.workers)
                result={'completed':True}
            else:result=validate(args,identity)
            atomic_json(args.output/'operations'/f'{args.command}.json',{'exit_status':0,'seconds':time.monotonic()-began,
                'workers':args.workers,'reference_workers':args.reference_workers,'argv':sys.argv,'result':result})
            print(json.dumps(result))
        except BaseException as exc:
            atomic_json(args.output/'operations'/f'{args.command}.json',{'exit_status':1,'seconds':time.monotonic()-began,
                'argv':sys.argv,'error':repr(exc)})
            raise


if __name__=='__main__':
    main()
