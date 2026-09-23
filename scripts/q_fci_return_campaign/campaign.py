#!/usr/bin/env python3
"""Portable, node-local CPU campaign with verified resumable numerical stages."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('JAX_ENABLE_X64','true');os.environ.setdefault('JAX_PLATFORMS','cpu')
sys.path.insert(0,str(REPO))


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def default(x):return x.tolist() if hasattr(x,'tolist') else x.item()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x,indent=2,sort_keys=True,default=default)+'\n');os.replace(tmp,p)
def save(p,**x):
    import numpy as np
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**x)
    os.replace(tmp,p)
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if sys.platform=='darwin' else 2**20)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('campaign already has a writer')
        yield


def source_identity():
    result={p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
    for name in ('configuration.json','input_manifest.json','geometry_source.tar.gz','geometry_source_manifest.json'):
        result[name]=sha(HERE/name)
    result['frozen_mms.py']=sha(HERE.parent/'q03_direct_campaign/frozen_mms.py')
    return result


def unpack(output):
    target=output/'software';manifest=read(HERE/'geometry_source_manifest.json')
    if sha(HERE/'geometry_source.tar.gz')!=manifest['archive_sha256']:raise RuntimeError('source archive mismatch')
    marker=target/'receipt.json'
    if not marker.exists():
        target.mkdir(parents=True,exist_ok=True)
        with tarfile.open(HERE/'geometry_source.tar.gz') as tar:
            for member in tar.getmembers():
                rel=Path(member.name)
                if rel.is_absolute() or '..' in rel.parts or not member.isfile():raise RuntimeError('unsafe source member')
                dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tar.extractfile(member).read())
        write(marker,{'archive_sha256':manifest['archive_sha256']})
    if read(marker)['archive_sha256']!=manifest['archive_sha256']:raise RuntimeError('wrong extracted source')
    for rel,sig in manifest['files'].items():
        if sha(target/rel)!=sig:raise RuntimeError(f'changed frozen source {rel}')
    return target


def verify(args):
    expected=read(HERE/'input_manifest.json')
    root=args.input_root
    for record in expected['files']:
        p=root/record['path']
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
            raise RuntimeError(f'wrong or missing immutable input: {p}')
    unpack(args.output)
    identity={'schema':'q-fci-return-campaign-v1','source':source_identity(),'inputs':digest(expected),
              'configuration':read(HERE/'configuration.json')}
    target=args.output/'campaign.json'
    if target.exists() and read(target)!=identity:raise RuntimeError('incompatible campaign; use a new folder')
    write(target,identity);write(args.output/'input_locations.json',{'input_root':str(root)})
    import numpy,scipy
    write(args.output/'environment.json',{'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'platform':platform.platform()})
    return identity


CTX=None;NUM=None;ROWS=None
def initialize(n,input_root,output,row_dir=None,trace_capacity=None):
    global CTX,NUM,ROWS
    # Pin each Linux child before JAX initializes to avoid one full-node thread pool per worker.
    identity=multiprocessing.current_process()._identity
    if identity and hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[(identity[0]-1)%len(cpus)]})
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(Path(output)/'cache/jax'))
    sys.path.insert(0,str(Path(output)/'software/src'))
    os.environ['DRBX_CACHE_DIR']=str(Path(output)/'cache/jax')
    from scripts.q_fci_return_campaign import numerics
    NUM=numerics;CTX=NUM.context(n,input_root,read(HERE/'configuration.json'))
    if trace_capacity:CTX['trace_capacity']=trace_capacity
    ROWS=load_catalogue(Path(row_dir)) if row_dir else None


def receipt(path,identity,unit):
    write(path.with_suffix('.json'),{'identity':identity,'unit':unit,'sha256':sha(path),'bytes':path.stat().st_size})


def completed(path,identity,unit):
    p=path.with_suffix('.json')
    if not p.exists():return False
    r=read(p)
    if r['identity']!=identity or r['unit']!=unit or not path.exists() or sha(path)!=r['sha256']:
        raise RuntimeError(f'corrupt/incompatible completed unit: {path}')
    return True


def do_trace(job):
    ids,path,identity=job
    data=NUM.trace_rows(CTX,ids);data['peak_rss_gib']=rss();save(path,**data);receipt(Path(path),identity,ids)
    return str(path)


def load_catalogue(path):
    import numpy as np
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in path.glob('*.npy')}


def pool_jobs(args,n,jobs,worker,row_dir=None,trace_capacity=None):
    init=(n,str(args.input_root),str(args.output),str(row_dir) if row_dir else None,trace_capacity)
    if args.workers==1:
        initialize(*init)
        for job in jobs:yield worker(job)
        return
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=init) as pool:
        pending=set();jobs=iter(jobs)
        for _ in range(2*args.workers):
            job=next(jobs,None)
            if job is not None:pending.add(pool.submit(worker,job))
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                job=next(jobs,None)
                if job is not None:pending.add(pool.submit(worker,job))


def trace_plan(ctx,ids,chunk=24):
    import numpy as np
    # Bounded subsets stay bounded; each saved batch has its own content identity.
    keys=NUM.row_keys(ctx['N'],np.asarray(ids));units=[]
    for k,d in sorted(set(map(tuple,keys[:,2:]))):
        local=np.asarray(ids)[(keys[:,2]==k)&(keys[:,3]==d)]
        units.extend(local[i:i+chunk].tolist() for i in range(0,len(local),chunk))
    return units


def trace_path(root,unit):return root/'trace_chunks'/(digest(unit)+'.npz')


def trace_stage(args,n,root,ids,campaign_id,chunk=24):
    import numpy as np
    plan=root/'trace_plan.json'
    previous=read(plan)['units'] if plan.exists() else []
    done_ids=[r for u in previous for r in u]
    units=previous+trace_plan(CTX,np.setdiff1d(ids,done_ids),chunk) if len(np.setdiff1d(ids,done_ids)) else previous
    ident=digest({'campaign':campaign_id,'N':n,'stage':'trace'})
    plan=root/'trace_plan.json'
    data={'identity':ident,'units':units}
    if plan.exists():
        old=read(plan)
        if old['identity']!=ident or not {digest(u) for u in old['units']}.issubset({digest(u) for u in units}):raise RuntimeError('trace plan is not a compatible extension')
    write(plan,data);jobs=[]
    for k,unit in enumerate(units):
        p=trace_path(root,unit)
        if not completed(p,ident,unit):jobs.append((unit,str(p),ident))
    started=time.monotonic();count=len(units)-len(jobs)
    for result in pool_jobs(args,n,jobs,do_trace,trace_capacity=4*chunk):
        count+=1;write(root/'progress.json',{'stage':'trace','completed':count,'total':len(units),'seconds':time.monotonic()-started})
    cat=root/'rows';cat.mkdir(exist_ok=True);size=2*n**3
    shapes={'valid':(size,),'available':(size,),'source':(size,4,3),'endpoint':(size,4,3),'ell':(size,4),'F':(size,4),'Z':(size,),'g_sec':(size,4),'numerical':(size,4)}
    arrays={k:np.lib.format.open_memmap(cat/f'{k}.npy',mode='w+',dtype=bool if k in ('valid','available') else float,shape=v) for k,v in shapes.items()}
    for key,a in arrays.items():a[:]=False if a.dtype==bool else np.nan
    timings=[]
    for k,unit in enumerate(units):
        p=trace_path(root,unit);assert completed(p,ident,unit)
        with np.load(p) as z:
            assert np.array_equal(z['ids'],unit)
            valid=z['valid'];assert valid.shape==(len(unit),)
            for key,a in arrays.items():
                if key=='available':a[unit]=True
                else:
                    a[unit]=z[key]
                    if z[key].dtype.kind=='f' and not np.all(np.isfinite(z[key][valid])):raise ValueError('nonfinite valid row')
            timings.append({'seconds':float(z['seconds']),'fit_seconds':float(z['fit_seconds']),'peak_rss_gib':float(z['peak_rss_gib']),'rows':len(unit),'invalid':int((~valid).sum())})
    for a in arrays.values():a.flush()
    write(cat/'manifest.json',{'identity':ident,'ids':sum(map(len,units)),'files':{k+'.npy':sha(cat/(k+'.npy')) for k in arrays},'timings':timings})
    return cat


def do_faces(job):
    import numpy as np
    ids,path,identity=job;t=time.monotonic();keys=CTX['keys'][ids];outputs=[];metas=[];rowids=[];coeffs=[]
    for fid,key in zip(ids,keys):
        if key[0]==0 and key[1]==CTX['N']:
            f=NUM.prescribed_boundary_flux(key[None],np.zeros((1,4)))[0]
            outputs.append((f,f));metas.append({'boundary':True});rowids.append(np.empty(0,int));coeffs.append(np.empty(0));continue
        failures=[]
        for halo in CTX['config']['policy']['halos']:
            required=NUM.candidate_ids(CTX['N'],key,halo)
            if not np.all(ROWS['available'][required]):
                return {'missing_rows':required[~ROWS['available'][required]].tolist()}
            try:
                fn,fe,rr,weights,meta=NUM.return_map(CTX,key,ROWS,halo)
                break
            except ValueError as exc:failures.append(str(exc))
        else:raise RuntimeError(f'no observable face target {key}: {failures}')
        np.testing.assert_allclose(weights@(ROWS['numerical'][rr]@np.array([.3,-.7,.2,1.1])),fn@np.array([.3,-.7,.2,1.1]),rtol=1e-10,atol=1e-13)
        outputs.append((fn,fe));metas.append(meta);rowids.append(rr);coeffs.append(weights)
    maximum=max([len(a) for a in rowids]+[1]);ri=np.full((len(ids),maximum),-1,int);ww=np.zeros(ri.shape)
    for k,(r,w) in enumerate(zip(rowids,coeffs)):ri[k,:len(r)]=r;ww[k,:len(r)]=w
    # Batched geometry evaluation shares work across all physical fields.
    refs={f'q{q}':NUM.exact_flux(CTX,keys,q) for q in (5,9,11)}
    boundary=(keys[:,0]==0)&(keys[:,1]==CTX['N'])
    for value in refs.values():
        if np.max(np.abs(value[boundary]),initial=0)>1e-20:raise ValueError('frozen MMS boundary flux is not zero')
        value[boundary]=0
    output=np.asarray(outputs)
    save(path,ids=ids,keys=keys,numerical=output[:,0],g_sec=output[:,1],row_ids=ri,weights=ww,
         diagnostics=np.array(json.dumps(metas)),seconds=np.array(time.monotonic()-t),peak_rss_gib=np.array(rss()),**refs)
    receipt(Path(path),identity,ids);return str(path)


def do_volume(job):
    import numpy as np
    ids,path,identity=job;t=time.monotonic();ijk=np.array(np.unravel_index(ids,(CTX['N'],)*3)).T
    out={}
    for order in (5,7):
        p,w=NUM.quadrature(CTX,ijk,order,face=False);J=np.abs(np.linalg.det(CTX['evaluator'].jacobian_matrix(p.reshape(-1,3))))
        out[f'q{order}']=np.sum(w*J.reshape(w.shape),axis=1)
    save(path,ids=ids,seconds=np.array(time.monotonic()-t),peak_rss_gib=np.array(rss()),**out)
    receipt(Path(path),identity,ids);return str(path)


def do_strong(job):
    import numpy as np
    name,owner,reference,numerical=job
    members=np.flatnonzero(CTX['topology']['compact_raw_owner'].ravel()==owner)
    t=time.monotonic();value,volume=NUM.strong_reference(CTX,members,9)
    return {'track':name,'volume_q9':volume,'reference_q9_volume':value,
            'face_q11_minus_strong':np.asarray(reference)-value,
            'numerical_minus_strong':np.asarray(numerical)-value,
            'seconds':time.monotonic()-t,'worker_pid':os.getpid(),'peak_rss_gib':rss()}


def bounded_selection(ctx):
    import numpy as np
    n=ctx['N'];grid=ctx['artifact'].geometry.grid;labels=ctx['topology']['compact_raw_owner'];owners=[];tracks=[]
    for name,seed in [('ordinary',(18,16,13)),('agglomerated',(8,16,21)),('transition',(10,16,29))]:
        target=(np.array(seed)+.5)*np.array([1/32,2*np.pi/32,2*np.pi/32])
        cell=[]
        for axis,centers in enumerate((grid.x.centers,grid.y.centers,grid.z.centers)):
            delta=centers-target[axis]
            if axis:delta=(delta+np.pi)%(2*np.pi)-np.pi
            cell.append(int(np.argmin(abs(delta))))
        tracks.append({'name':name,'raw':cell,'owner':int(labels[tuple(cell)])})
    for name,cell in [('axis',(0,0,0)),('wall',(n-1,0,0)),('inward',(n-2,0,0)),('theta_seam',(n//2,0,n//2))]:
        tracks.append({'name':name,'raw':list(cell),'owner':int(labels[cell])})
    owners=np.unique([x['owner'] for x in tracks])
    ids=np.flatnonzero(np.isin(ctx['lower'],owners)|np.isin(ctx['upper'],owners))
    raw=np.flatnonzero(np.isin(labels.ravel(),owners))
    return owners,ids,raw,tracks


def stage_units(args,n,root,ids,name,chunk,worker,campaign_id,row_dir=None):
    units=[ids[i:i+chunk].tolist() for i in range(0,len(ids),chunk)]
    ident=digest({'campaign':campaign_id,'N':n,'stage':name,'units':units})
    plan={'identity':ident,'units':units};p=root/f'{name}_plan.json'
    if p.exists() and read(p)!=plan:raise RuntimeError('unit plan mismatch')
    write(p,plan);jobs=[]
    for i,unit in enumerate(units):
        path=root/f'{name}_chunks'/f'{i:06d}.npz'
        if not completed(path,ident,unit):jobs.append((unit,str(path),ident))
    count=len(units)-len(jobs);t=time.monotonic();missing=[]
    for result in pool_jobs(args,n,jobs,worker,row_dir):
        if isinstance(result,dict):missing.extend(result['missing_rows']);continue
        count+=1;write(root/'progress.json',{'stage':name,'completed':count,'total':len(units),'seconds':time.monotonic()-t})
    return sorted(set(missing))


def assemble(root,ctx,owners,face_ids,raw_ids,campaign_id):
    import numpy as np
    face_plan=read(root/'face_plan.json');vol_plan=read(root/'volume_plan.json')
    flux={kind:np.empty((len(face_ids),4)) for kind in ('numerical','g_sec','q5','q9','q11')};diags=[];timing=[]
    for i,unit in enumerate(face_plan['units']):
        path=root/'face_chunks'/f'{i:06d}.npz';assert completed(path,face_plan['identity'],unit)
        with np.load(path) as z:
            assert np.array_equal(z['ids'],unit);pos=np.searchsorted(face_ids,unit)
            for kind in flux:
                assert np.all(np.isfinite(z[kind]));flux[kind][pos]=z[kind]
            diags.extend(json.loads(str(z['diagnostics'])));timing.append({'seconds':float(z['seconds']),'peak_rss_gib':float(z['peak_rss_gib']),'faces':len(unit)})
    cont={q:np.zeros(len(ctx['volume'])) for q in ('q5','q7')};vol_timing=[];labels=ctx['topology']['compact_raw_owner'].ravel()
    for i,unit in enumerate(vol_plan['units']):
        path=root/'volume_chunks'/f'{i:06d}.npz';assert completed(path,vol_plan['identity'],unit)
        with np.load(path) as z:
            assert np.array_equal(z['ids'],unit)
            for q in cont:
                assert np.all(np.isfinite(z[q])) and np.all(z[q]>0)
                np.add.at(cont[q],labels[unit],z[q])
            vol_timing.append({'seconds':float(z['seconds']),'peak_rss_gib':float(z['peak_rss_gib']),'cells':len(unit)})
    vol=ctx['volume'][owners];vc=cont['q7'][owners]
    if np.any(vc<=0):raise ValueError('missing continuous volume')
    actions={k:NUM.assemble(ctx,face_ids,f)[owners] for k,f in flux.items()}
    refhigh=actions['q11']*vol[:,None]/vc[:,None]
    reflow=actions['q9']*vol[:,None]/cont['q5'][owners,None]
    weighted=lambda e:np.sqrt(np.sum(vol[:,None]*e*e,axis=0)/vol.sum())
    rms=weighted(actions['numerical']-refhigh);budget=weighted(refhigh-reflow)
    arrays={'owners':owners,'face_ids':face_ids,'volume':vol,'continuous_volume_q5':cont['q5'][owners],'continuous_volume_q7':vc,
            'reference':refhigh,'reference_low':reflow,**{k+'_action':v for k,v in actions.items()}}
    for kind,f in flux.items():
        for axis in range(3):
            mask=ctx['keys'][face_ids,0]==axis
            arrays[f'{kind}_axis{axis}_action']=NUM.assemble(ctx,face_ids[mask],f[mask])[owners]
        np.testing.assert_allclose(sum(arrays[f'{kind}_axis{axis}_action'] for axis in range(3)),actions[kind],rtol=1e-11,atol=1e-12)
    save(root/'actions.npz',**arrays)
    # Verify the complete-owner exterior ledger separately from owner reduction.
    balance={}
    for k,f in flux.items():
        exterior=np.isin(ctx['lower'][face_ids],owners).astype(int)-np.isin(ctx['upper'][face_ids],owners).astype(int)
        balance[k]=float(np.max(abs(vol@actions[k]-exterior@f)))
    nonwall=[x for x in diags if not x.get('boundary')]
    summary={'identity':campaign_id,'N':ctx['N'],'scope':'global' if len(owners)==len(ctx['volume']) else 'bounded',
             'owners':len(owners),'faces':len(face_ids),'fields':NUM.FIELDS,'operator_rms':rms,
             'exact_secant_rms':weighted(actions['g_sec']-refhigh),'endpoint_contribution_rms':weighted(actions['numerical']-actions['g_sec']),
             'reference_budget_rms':budget,'reference_fraction':budget[:3]/np.maximum(rms[:3],1e-300),
             'constant_max':float(abs(actions['numerical'][:,-1]).max()),'balance':balance,
             'fit':{'rank_min':min(x['rank'] for x in nonwall),'condition_max':max(x['condition'] for x in nonwall),'target_defect_max':max(x['target_relative_defect'] for x in nonwall),'max_rows':max(x['rows'] for x in nonwall),'halo_counts':{str(h):sum(x['halo']==h for x in nonwall) for h in (2,3,4,6,8)}},
             'face_timings':timing,'volume_timings':vol_timing,'actions_sha256':sha(root/'actions.npz')}
    # Regional sums are diagnostics; the global denominator is never cropped.
    radial=ctx['owner_ids'][owners]//ctx['N']**2;members=np.bincount(labels,minlength=len(ctx['volume']))[owners]
    masks={'axis':radial==0,'wall':radial==ctx['N']-1,'inward':radial==ctx['N']-2,'agglomerated':members>1,'ordinary':members==1}
    error=actions['numerical']-refhigh;total=np.sum(vol[:,None]*error**2,axis=0)
    summary['regional_squared_error_share']={name:np.sum(vol[mask,None]*error[mask]**2,axis=0)/np.maximum(total,1e-300) for name,mask in masks.items()}
    write(root/'summary.json',summary);return summary


def resolution(args,n,campaign_id,sample):
    import numpy as np
    root=args.output/('preflight' if sample else 'global')/f'N{n}';root.mkdir(parents=True,exist_ok=True)
    initialize(n,str(args.input_root),str(args.output));ctx=CTX
    if sample:owners,faces,raw,tracks=bounded_selection(ctx)
    else:owners=np.arange(len(ctx['volume']));faces=np.arange(len(ctx['keys']));raw=np.arange(n**3);tracks=[]
    write(root/'selection.json',{'owners':owners,'face_ids':faces,'raw_ids':raw,'tracks':tracks})
    if sample:
        ids=np.unique(np.concatenate([NUM.candidate_ids(n,ctx['keys'][f],2) for f in faces if ctx['keys'][f,1]!=n]))
    else:ids=np.arange(2*n**3)
    if (root/'trace_plan.json').exists():ids=np.union1d(ids,[r for u in read(root/'trace_plan.json')['units'] for r in u])
    while True:
        cat=trace_stage(args,n,root,ids,campaign_id,4 if sample else 24)
        missing=stage_units(args,n,root,faces,'face',8 if sample else 64,do_faces,campaign_id,cat)
        if not missing:break
        ids=np.union1d(ids,missing)
    stage_units(args,n,root,raw,'volume',8 if sample else 32,do_volume,campaign_id)
    # Worker initialization may have changed globals, restore the matching context only.
    summary=assemble(root,ctx,owners,faces,raw,campaign_id)
    if sample:
        jobs=[]
        # Independent strong-volume controls use the same CPU pool as other stages.
        with np.load(root/'actions.npz') as z:
            for name in ('ordinary','wall'):
                track=next(x for x in tracks if x['name']==name);owner=track['owner']
                pos=np.flatnonzero(z['owners']==owner)[0]
                jobs.append((name,owner,z['reference'][pos],z['numerical_action'][pos]))
        strong=list(pool_jobs(args,n,jobs,do_strong))
        strong.sort(key=lambda x:x['track'])
        summary['strong_volume_checks']=strong;write(root/'summary.json',summary)
    write(root/'completion.json',{'identity':campaign_id,'summary_sha256':sha(root/'summary.json'),'complete':True})
    return summary


def validate(args,campaign_id,sample=False):
    import numpy as np
    records=[]
    for n in (32,48,64):
        root=args.output/('preflight' if sample else 'global')/f'N{n}';r=read(root/'completion.json')
        if r['identity']!=campaign_id or r['summary_sha256']!=sha(root/'summary.json'):raise RuntimeError('completion identity mismatch')
        s=read(root/'summary.json');sel=read(root/'selection.json')
        plan=read(root/'trace_plan.json')
        for unit in plan['units']:
            if not completed(trace_path(root,unit),plan['identity'],unit):raise RuntimeError('missing trace checkpoint')
        for rel,sig in read(root/'rows/manifest.json')['files'].items():
            if sha(root/'rows'/rel)!=sig:raise RuntimeError('changed trace catalogue')
        if s['constant_max']>1e-8 or max(s['balance'].values())>1e-11:raise RuntimeError('constant or incidence check failed')
        if sha(root/'actions.npz')!=s['actions_sha256']:raise RuntimeError('changed action arrays')
        for stage,key in (('face','face_ids'),('volume','raw_ids')):
            plan=read(root/f'{stage}_plan.json');units=plan['units']
            if [j for unit in units for j in unit]!=sel[key]:raise RuntimeError('incomplete planned coverage')
            for i,unit in enumerate(units):
                if not completed(root/f'{stage}_chunks'/f'{i:06d}.npz',plan['identity'],unit):raise RuntimeError('missing completed unit')
        with np.load(root/'actions.npz') as z:
            for k in z.files:
                if not np.all(np.isfinite(z[k])):raise RuntimeError(f'nonfinite action {k}')
            e=np.sqrt(np.sum(z['volume'][:,None]*(z['numerical_action']-z['reference'])**2,axis=0)/z['volume'].sum())
            np.testing.assert_allclose(e,s['operator_rms'],rtol=2e-14,atol=1e-16)
        records.append(s)
    errors=np.array([r['operator_rms'][:3] for r in records]);orders=np.log(errors[:-1]/errors[1:])/np.log(np.array([1.5,4/3]))[:,None]
    passed=not sample and bool(np.all(orders>=1.8) and all(max(r['reference_fraction'])<.1 for r in records))
    out={'computation_complete':True,'scope':'bounded' if sample else 'global','fields':records[0]['fields'][:3],'errors':errors,'orders':orders,
         'global_static_accuracy_passed':passed,'structural_evolved_production_certified':False,'resolutions':records}
    write(args.output/('preflight_validation.json' if sample else 'validation.json'),out)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify','preflight','run','validate','status'))
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--memory-budget-gib',type=float,default=0)
    p.add_argument('--worker-memory-gib',type=float,default=2.5);p.add_argument('--memory-reserve-gib',type=float,default=2.)
    args=p.parse_args();args.output=args.output.resolve();args.input_root=args.input_root.resolve()
    if args.workers<1:raise ValueError('workers must be positive')
    if args.memory_budget_gib:
        cap=int((args.memory_budget_gib-args.memory_reserve_gib)//args.worker_memory_gib)
        if cap<1:raise ValueError('insufficient memory budget')
        args.workers=min(args.workers,cap)
    if args.command=='status':
        print(json.dumps(read(args.output/'status.json'),indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    with lock(args.output/'.lock'):
        identity=verify(args);cid=digest(identity)
        t=time.monotonic();write(args.output/'status.json',{'state':'running','command':args.command,'workers':args.workers,'started_unix':time.time()})
        try:
            if args.command=='preflight':
                for n in (32,48,64):resolution(args,n,cid,True)
                validate(args,cid,True)
            elif args.command=='run':
                validate(args,cid,True)
                for n in (32,48,64):resolution(args,n,cid,False)
                validate(args,cid,False)
            elif args.command=='validate':validate(args,cid,False)
            write(args.output/'status.json',{'state':'complete','command':args.command,'seconds':time.monotonic()-t,'workers':args.workers,'completed_unix':time.time()})
        except BaseException as exc:
            write(args.output/'status.json',{'state':'failed','command':args.command,'seconds':time.monotonic()-t,'error':str(exc),'updated_unix':time.time()});raise


if __name__=='__main__':main()
