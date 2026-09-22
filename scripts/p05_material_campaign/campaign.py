#!/usr/bin/env python3
"""Computation-only P05 material accuracy campaign, reusing a pinned centered run."""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from p05_material_campaign import kernel
sys.path.insert(0,str(REPO/'scripts/hsx_remote_qualification'))
import parallel_runner as runner

SCHEMA='drbx.p05-material-global-v1'
CHUNK=256
STATE={}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return runner._sha256(Path(path))
def write(path,value): runner._atomic_json(Path(path),value)
def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files if k!='metadata_json'}


def verify(args):
    manifests={name:read(HERE/(name+'_manifest.json')) for name in ('source','input','centered')}
    for name,root in (('source',REPO),('input',args.input_root),('centered',args.centered_root)):
        for record in manifests[name]['files']:
            p=root/record['path']
            if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
                raise RuntimeError(f'changed/missing {name} input: {p}')
    payload={'schema':SCHEMA,'policy':kernel.POLICY,'manifests':manifests,'chunk':CHUNK}
    identity=runner._digest(payload)
    p=args.output/'campaign_manifest.json'
    if p.exists() and read(p)['identity']!=identity:
        raise RuntimeError('incompatible campaign; use a new directory')
    write(p,{'identity':identity,'content':payload})
    write(args.output/'input_locations.json',{'repo':str(REPO),'input_root':str(args.input_root),
          'centered_root':str(args.centered_root)})
    return identity


def runtime(args):
    return runner._materialize(REPO/'scripts/hsx_remote_qualification/configuration.json',
                              REPO,args.input_root,args.output)


def initialize(settings):
    global STATE
    numeric=runner._bootstrap(REPO,REPO,Path(settings['output']))
    config=read(settings['runtime'])
    n=settings['N']
    context=numeric.cubic._load_context(Path(config['paths']['geometry']),Path(config['paths']['baseline']),n)
    reference=numeric.integrated._reference(Path(config['paths']['reference_sidecar']),verify_hashes=False)
    prepare=arrays(Path(settings['centered'])/f'N{n}.prepare.npz')
    STATE=dict(settings,numeric=numeric,context=context,reference=reference,prepare=prepare)


def expected_faces(n,indices,root):
    flux=np.empty((3,len(indices)));digests=np.empty(len(indices),dtype='U64')
    for first in np.unique(indices//512*512):
        last=min(int(first)+512,3*n**3+3*n**2)
        z=arrays(Path(root)/f'N{n}.chunks/face_{first:07d}_{last:07d}.npz')
        selected=np.flatnonzero((indices>=first)&(indices<last));local=indices[selected]-first
        if not np.array_equal(z['indices'],np.arange(first,last)):
            raise RuntimeError('centered chunk coverage mismatch')
        flux[:,selected]=z['product'][:3,0,local]
        digests[selected]=z['donor_sha256'][local]
    return flux,digests


def chunk_path(output,n,lo,hi): return Path(output)/f'N{n}.chunks'/f'faces_{lo:07d}_{hi:07d}.npz'


def validate_chunk(path,identity,indices):
    p=Path(path);receipt=p.with_suffix('.json')
    if not receipt.exists(): return False
    r=read(receipt)
    if r['identity']!=identity or r['indices_sha256']!=hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest():
        raise RuntimeError(f'incompatible checkpoint: {path}')
    if not p.exists() or sha(p)!=r['sha256']: raise RuntimeError(f'corrupt checkpoint: {path}')
    with np.load(p) as z:
        if not np.array_equal(z['indices'],indices): raise RuntimeError('checkpoint coverage mismatch')
    return True


def compute(job):
    indices,path=job; s=STATE
    if validate_chunk(path,s['identity'],indices): return str(path)
    started=time.monotonic()
    z=kernel.compute(s['numeric'],s['context'],s['reference'],s['prepare'],indices)
    expected,digests=expected_faces(s['N'],indices,s['centered'])
    if not np.array_equal(z['donor_hash'],digests):
        bad=indices[z['donor_hash']!=digests]
        raise RuntimeError(f'changed frozen donor graph N{s["N"]}: faces {bad[:20].tolist()}')
    difference=float(np.max(abs(z['central_flux']-expected)))
    if not np.allclose(z['central_flux'],expected,rtol=2e-9,atol=2e-13):
        raise RuntimeError(f'central flux failed frozen replay: {difference}')
    if z['reproduction'].max(initial=0)>1e-10:
        raise RuntimeError('side-fit polynomial reproduction failed')
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**z)
    temp.replace(p)
    write(p.with_suffix('.json'),{'identity':s['identity'],'indices_sha256':hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest(),
        'sha256':sha(p),'seconds':time.monotonic()-started,'peak_rss_gib':s['numeric']._max_rss_gib(),
        'central_flux_replay_max':difference,'maximum_reproduction':float(z['reproduction'].max(initial=0))})
    return str(p)


def settings(args,n,identity,runtime_path):
    return {'N':n,'identity':identity,'runtime':str(runtime_path),'output':str(args.output),
            'centered':str(args.centered_root)}


def preflight(args,identity):
    _,runtime_path,numeric=runtime(args)
    for n in args.resolutions:
        fixture=HERE/f'preflight_N{n}.npz'
        f=arrays(fixture);indices=f['indices']
        initialize(settings(args,n,identity,runtime_path))
        path=args.output/f'N{n}.preflight.npz'
        compute((indices,path));z=arrays(path)
        if n in (48,64):
            np.testing.assert_allclose(z['delta_flux'],f['delta_flux'],rtol=2e-8,atol=2e-13)
            total=np.zeros((3,len(STATE['prepare']['owner_volume'])))
            kernel.scatter(numeric,n,STATE['prepare']['raw_owner'],indices,z['delta_flux'],total)
            rho=float(numeric.mms.PHYSICAL_PARAMETERS['rho_star'])
            replay=-total[:,f['owners']]/(rho*f['volume'][None,:])
            np.testing.assert_allclose(replay,f['action_delta'],rtol=2e-8,atol=5e-10)
            action_error=float(np.max(abs(replay-f['action_delta'])))
        else: action_error=None
        write(args.output/f'N{n}.preflight.status.json',{'identity':identity,'passed':True,'faces':len(indices),
              'frozen_bounded_jump_replay':n in (48,64),'action_max_abs_difference':action_error,
              'chunk_sha256':sha(path)})
        STATE.clear()
    return {'preflight':'passed'}


def jobs(args,n):
    total=3*n**3+3*n**2
    for lo in range(0,total,CHUNK):
        hi=min(lo+CHUNK,total)
        yield np.arange(lo,hi,dtype=np.int64),chunk_path(args.output,n,lo,hi)


def execute(args,identity):
    _,runtime_path,_=runtime(args)
    for n in args.resolutions:
        pre=read(args.output/f'N{n}.preflight.status.json')
        if pre['identity']!=identity or not pre['passed'] or sha(args.output/f'N{n}.preflight.npz')!=pre['chunk_sha256']:
            raise RuntimeError('missing/incompatible preflight')
        # Check receipts before initializing expensive geometry workers.
        pending_jobs=[];completed=0
        for indices,path in jobs(args,n):
            if validate_chunk(path,identity,indices):completed+=len(indices)
            else:pending_jobs.append((indices,path))
        started=time.monotonic()
        if pending_jobs:
            with multiprocessing.get_context('spawn').Pool(processes=args.workers,
                    initializer=initialize,initargs=(settings(args,n,identity,runtime_path),),
                    maxtasksperchild=16) as pool:
                for path in pool.imap_unordered(compute,pending_jobs,chunksize=1):
                    with np.load(path) as z:completed+=len(z['indices'])
                    write(args.output/'status.json',{'state':'running','N':n,'completed_faces':completed,
                        'total_faces':3*n**3+3*n**2,'seconds':time.monotonic()-started})
        assemble(args,n,identity)
    return merge(args,identity)


def reference_budget(q,field):
    e=q['fields'][field]
    if field=='actual_vorticity':
        return sum(e[k] for k in ('direct_q3_minus_ibp_q3_representative_rms','quadrature_q4_minus_q3_rms','finite_difference_step_sensitivity_rms'))
    rule=e['rules'][str(e['selected_order'])]
    return rule['reference_difference_rms']+rule['high_reference_q7_minus_q5_rms']


def assemble(args,n,identity):
    config,_,numeric=runtime(args)
    base=arrays(args.centered_root/f'N{n}.npz');prepare=arrays(args.centered_root/f'N{n}.prepare.npz')
    if not np.array_equal(base['owner_volume'],prepare['owner_volume']) or not np.array_equal(base['owner_keys'],prepare['owner_keys']):
        raise RuntimeError('centered owner identity mismatch')
    volume=base['owner_volume'];total=np.zeros((3,len(volume)))
    maximum_reproduction=0.;wall=np.zeros_like(total)
    for indices,path in jobs(args,n):
        if not validate_chunk(path,identity,indices):raise RuntimeError(f'missing completed chunk: {path}')
        z=arrays(path);kernel.scatter(numeric,n,prepare['raw_owner'],indices,z['delta_flux'],total)
        keys=numeric._face_keys(n,indices);w=(keys[:,0]==0)&(keys[:,1]==n)
        kernel.scatter(numeric,n,prepare['raw_owner'],indices[w],z['delta_flux'][:,w],wall)
        maximum_reproduction=max(maximum_reproduction,float(z['reproduction'].max(initial=0)))
    rho=float(numeric.mms.PHYSICAL_PARAMETERS['rho_star'])
    delta=-total/(rho*volume[None,:]);candidate=base['action:matched:A'][:3]+delta
    reference=np.stack([base['reference:'+f] for f in kernel.FIELDS])
    q=read(Path(config['paths']['reference_root'])/f'N{n}.qualification.json')
    # Reuse the frozen regional classification from the completed case.
    owner_keys=base['owner_keys']
    # Actual topology masks are loaded once at assembly, not inferred from N.
    data=numeric.integrated._load_resolution(Path(config['paths']['geometry']),Path(config['paths']['baseline']),n)
    masks={k:np.asarray(v,dtype=bool) for k,v in data.masks.items()}
    stats={}
    for j,f in enumerate(kernel.FIELDS):
        err=candidate[j]-reference[j];err0=base['action:matched:A'][j]-reference[j]
        sse=float(np.dot(volume,err*err));error=float(np.sqrt(sse/volume.sum()));budget=reference_budget(q,f)
        stats[f]={'L2':error,'Linf':float(np.max(abs(err))),
            'centered_A_L2':float(np.sqrt(np.dot(volume,err0*err0)/volume.sum())),
            'correction_L2':float(np.sqrt(np.dot(volume,delta[j]*delta[j])/volume.sum())),
            'signed_linear_SSE':float(2*np.dot(volume,err0*delta[j])),
            'correction_SSE':float(np.dot(volume,delta[j]*delta[j])),
            'reference_absolute_budget':budget,'reference_fraction':budget/error,
            'reference_qualified':bool(budget<.1*error),'regions':{
                k:{'squared_error':float(np.dot(volume[m],err[m]**2)),
                   'global_SSE_fraction':float(np.dot(volume[m],err[m]**2)/sse) if sse else 0.,
                   'owners':int(m.sum())} for k,m in masks.items()}}
    out=args.output/f'N{n}.npz';temp=out.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,owner_keys=owner_keys,owner_volume=volume,reference=reference,
            centered_A=base['action:matched:A'][:3],centered_C=base['action:matched:C'][:3],
            material_A=candidate,delta=delta,wall_delta=-wall/(rho*volume[None,:]),**{'region:'+k:v for k,v in masks.items()})
    temp.replace(out)
    result={'schema':SCHEMA,'identity':identity,'N':n,'completed':True,'statistics':stats,
            'owners':len(volume),'faces':3*n**3+3*n**2,'maximum_reproduction':maximum_reproduction,
            'output_sha256':sha(out),'structural_certification':False,'solution_certification':False,
            'centered_A_unchanged':True,'reference_qualification':q,
            'centered_case_sha256':sha(args.centered_root/f'N{n}.json')}
    write(args.output/f'N{n}.json',result)
    return result


def merge(args,identity):
    if not all((args.output/f'N{n}.json').exists() for n in (32,48,64)):
        return {'completed_resolutions':args.resolutions,'global_study_complete':False}
    cases=[read(args.output/f'N{n}.json') for n in (32,48,64)]
    for n,c in zip((32,48,64),cases):
        if c['identity']!=identity or not c['completed'] or sha(args.output/f'N{n}.npz')!=c['output_sha256']:
            raise RuntimeError('incompatible/corrupt assembled case')
    result={}
    for f in kernel.FIELDS:
        errors=[c['statistics'][f]['L2'] for c in cases]
        orders=[float(np.log(errors[j]/errors[j+1])/np.log((48,64)[j]/(32,48)[j])) for j in (0,1)]
        result[f]={'errors':errors,'orders':orders,'both_orders_ge_1_8':bool(all(p>=1.8 for p in orders)),
                   'references_qualified':all(c['statistics'][f]['reference_qualified'] for c in cases),
                   'required_material_field':f in kernel.FIELDS[1:]}
    summary={'schema':SCHEMA,'identity':identity,'status':'computation completed','results':result,
         'material_operator_convergence_passed':all(v['both_orders_ge_1_8'] and v['references_qualified'] for f,v in result.items() if f in kernel.FIELDS[1:]),
         'positivity_dissipation_certified':False,'solution_certified':False,'production_promotion':False}
    write(args.output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify-inputs','preflight','run','validate'))
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--centered-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int)
    p.add_argument('--resolutions',type=int,nargs='+',choices=(32,48,64),default=[32,48,64])
    a=p.parse_args()
    a.input_root=a.input_root.resolve();a.centered_root=a.centered_root.resolve();a.output=a.output.resolve()
    if a.command=='run' and (a.workers is None or a.workers<1):p.error('run requires allocation-selected positive --workers')
    started=time.monotonic()
    with runner._exclusive_output(a.output/'controller'):
        try:
            identity=verify(a)
            if a.command=='preflight':result=preflight(a,identity)
            elif a.command=='run':result=execute(a,identity)
            elif a.command=='validate':
                for n in a.resolutions:assemble(a,n,identity)
                result=merge(a,identity)
            else:result={'inputs_verified':True,'identity':identity}
            write(a.output/(a.command+'_receipt.json'),{'command':a.command,'exit_status':0,'identity':identity,
                   'seconds':time.monotonic()-started,'workers':a.workers,'resolutions':a.resolutions})
            write(a.output/'status.json',{'state':'stage completed','stage':a.command,'exit_status':0})
            print(json.dumps(result),flush=True)
        except BaseException as exc:
            write(a.output/'status.json',{'state':'failed','stage':a.command,'error':repr(exc),'seconds':time.monotonic()-started})
            raise


if __name__=='__main__':main()
