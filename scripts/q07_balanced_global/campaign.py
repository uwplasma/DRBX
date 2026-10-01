"""Portable, checkpointed paired Q07 rescore. Immutable inputs; CPU processes."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ.update(JAX_ENABLE_X64='true',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import sys,json,time,argparse,hashlib,fcntl,multiprocessing as mp,shutil,resource,platform
HERE=Path(__file__).resolve().parent
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(Path(os.environ.get('Q07_OUTPUT',str(HERE/'local'))).resolve()/'jax_cache'))
sys.path[:0]=[str(HERE),str(HERE/'source/src'),str(HERE/'source/scripts'),str(HERE/'source/frozen_reference')]
import numpy as np
import jax
import fields as m
import kernel
from frozen import model
from drbx.stencils.q_parallel import raw_members,sha256_array
REGIONS=('global','core','first_ring','inner','last_two_aggregate','transition','bulk','wall','outermost_wall')
METRICS=('N-O','O-R','N-R');MAX_RAW=128;ENV=None;INIT_ERROR=None
OUT=Path(os.environ.get('Q07_OUTPUT',str(HERE/'local'))).resolve()
INPUT_ROOT=Path(os.environ.get('Q07_INPUT_ROOT',str(HERE.parents[1]))).resolve()
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(OUT/'jax_cache'))
MEMORY=float(os.environ.get('Q07_WORKER_GIB','4'))
REPLAY_COMPONENT_ATOL=1e-8
REPLAY_COMBINED_ATOL=2*REPLAY_COMPONENT_ATOL
REPLAY_SUM_EPS_FACTOR=128

def bounded_action_replay(actual,expected):
    """Check each constituent and their sum without changing scientific gates."""
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=expected.shape or actual.shape[-1]!=30:raise ValueError('bounded replay shape')
    finite(actual,expected)
    # Final axes are [span, centered/correction/combined, material field].
    a=actual.reshape(actual.shape[:-1]+(2,3,5));b=expected.reshape(expected.shape[:-1]+(2,3,5))
    limits=(REPLAY_COMPONENT_ATOL,REPLAY_COMPONENT_ATOL,REPLAY_COMBINED_ATOL)
    maxima={}
    for i,name in enumerate(('centered','correction','combined')):
        err=float(abs(a[...,i,:]-b[...,i,:]).max());maxima[name]=err
        if err>limits[i]:raise ValueError(f'bounded replay {name} {err} > {limits[i]}')
    # Check actual and saved-reference algebra independently. The relative
    # roundoff budget allows owner reductions; it is not a spatial-error gate.
    residual=0.
    for label,x in (('actual',a),('expected',b)):
        error=abs(x[...,2,:]-x[...,0,:]-x[...,1,:])
        scale=1.+abs(x).sum(axis=-2)
        bound=REPLAY_SUM_EPS_FACTOR*np.finfo(np.float64).eps*scale
        if np.any(error>bound):raise ValueError('bounded replay sum identity '+label)
        residual=max(residual,float(error.max()))
    return dict(**maxima,sum_identity_residual=residual)

def sha(p):
    """Hash current bytes, never infer content identity from filesystem times.

    Some cluster filesystems preserve every observed stat field across rapid
    same-size rewrites. Metadata is only a best-effort concurrent-write guard,
    not a valid cache key. Large canonical inputs are checked at verification
    and worker initialization, not inside the numerical chunk loop.
    """
    p=Path(p);s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    ss=p.stat()
    if (ss.st_dev,ss.st_ino,ss.st_size,ss.st_mtime_ns,ss.st_ctime_ns)!=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns):raise ValueError('file changed during hash')
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
def finite(*a):
    if not all(np.isfinite(x).all() for x in a):raise ValueError('nonfinite arrays')
def resources():
    if rss()>MEMORY:raise RuntimeError(f'worker RSS {rss()} exceeds {MEMORY} GiB')
    if shutil.disk_usage(OUT).free<3*1024**3:raise RuntimeError('free disk below 3 GiB')
def manifest():return json.loads((HERE/'inputs_manifest.json').read_text())
def plans(n):
    p=HERE/'inputs/traces/plan.json'
    if sha(p)!=manifest()['files']['inputs/traces/plan.json']:raise ValueError('plan changed')
    return json.loads(p.read_text())[str(n)]
def sources():
    paths=[*HERE.glob('*.py'),*(HERE/'source').rglob('*.py'),*(HERE/'source').rglob('*.json')]
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(paths)}
def design():
    d=json.loads((HERE/'design.json').read_text())
    if sources()!=d['sources'] or sha(HERE/'inputs_manifest.json')!=d['manifest_sha256']:raise ValueError('frozen source/input manifest changed')
    return d

def freeze():
    if (HERE/'design.json').exists():design();return
    d=dict(schema='q07-balanced-shortspan-paired-v2',replay_policy='constituent-sum-budget-v1',sources=sources(),manifest_sha256=sha(HERE/'inputs_manifest.json'),terms=kernel.TERMS,cases=m.DESIGNS,kinds=m.KINDS,regions=REGIONS,metrics=METRICS,max_raw=MAX_RAW,case_batch=m.NF,
      inner_spans=[1/32,1/128],outer_spans=[1/16,1/64],trace_steps=64,trace_backend='cpu',material_span=1/128,tau=m.TAU,mu=m.MU,owner_counts={str(n):sum(map(len,plans(n)['global_'])) for n in (32,48,64)},chunk_counts={str(n):len(plans(n)['global_']) for n in (32,48,64)},gates=dict(replay=1e-8,bounded_replay_components=REPLAY_COMPONENT_ATOL,bounded_replay_combined=REPLAY_COMBINED_ATOL,bounded_replay_sum_eps_factor=REPLAY_SUM_EPS_FACTOR,constant_N_O_by_span=[1e-7,4e-7],center_b=1e-10),
      scientific_gate='Paired balanced h/32 versus h/128, centered/correction/combined all five material fields, 18 original plus four held-out states. Same continuum target, donors and BC treatment. No diffusion changes, tuning, or promotion. Regional scientific regressions are outcomes.')
    write_json(HERE/'design.json',d)
def verify():
    OUT.mkdir(parents=True,exist_ok=True);d=design();mf=manifest()
    for rel,h in mf['files'].items():
        if sha(HERE/rel)!=h:raise ValueError('input checksum '+rel)
    for rel,h in mf['canonical'].items():
        if sha(INPUT_ROOT/rel)!=h:raise ValueError('canonical checksum '+rel)
    if jax.default_backend()!='cpu' or not jax.config.jax_enable_x64:raise ValueError('CPU x64 required')
    receipt=dict(passed=True,identity=digest(d),input_root=str(INPUT_ROOT),canonical=mf['canonical'],backend=jax.default_backend(),python=sys.version,platform=platform.platform(),versions=dict(numpy=np.__version__,jax=jax.__version__),verified_files=len(mf['files']))
    write_json(OUT/'verification.json',receipt);return receipt

def init(n):
    global ENV
    d=design();v=json.loads((OUT/'verification.json').read_text())
    if not v['passed'] or v['identity']!=digest(d):raise ValueError('verification required')
    for rel,h in manifest()['canonical'].items():
        if (f'/{n}x{n}x{n}/' in rel or '/' not in rel) and sha(INPUT_ROOT/rel)!=h:raise ValueError('canonical changed '+rel)
    ctx,t=model.context(n,INPUT_ROOT)
    choicepath=HERE/f'inputs/choices_N{n}.npz'
    if sha(choicepath)!=manifest()['files'][str(choicepath.relative_to(HERE))]:raise ValueError('choices changed')
    with np.load(choicepath) as z:choices=z['choice'].copy()
    state=np.zeros((m.NF,5,len(t.vol)));phi=np.zeros((m.NF,len(t.vol)))
    for start in range(0,len(t.pts),4096):
        sl=slice(start,start+4096);val,_=m.fields(t.pts[sl]);p,_=m.phi_fields(t.pts[sl])
        for ci in range(m.NF):
            for fi in range(5):np.add.at(state[ci,fi],t.ro[sl],t.rv[sl]*val[:,ci,fi])
            np.add.at(phi[ci],t.ro[sl],t.rv[sl]*p[:,ci])
    state/=t.vol;phi/=t.vol;finite(state,phi)
    ENV=dict(n=n,d=d,identity=digest(d),ctx=ctx,t=t,state=state,phi=phi,choices=choices,choice_source=sha(choicepath));resources()
def worker_init(n):
    global INIT_ERROR
    try:init(n)
    except Exception as e:INIT_ERROR=repr(e)
def groups(t,owners):
    group=[];count=0
    for o in owners:
        size=t.starts[o+1]-t.starts[o]
        if group and count+size>MAX_RAW:yield np.array(group);group=[];count=0
        group.append(int(o));count+=size
    if group:yield np.array(group)
def masks(r,n,last):
    return (np.ones(len(r),bool),r==0,r==1,r<=last,(r>=last-1)&(r<=last),(r>=last-1)&(r<=last+2),(r>last)&(r<n-2),r>=n-2,r==n-1)
def empty_stats():
    shape=(len(REGIONS),3,4,m.NF,len(kernel.TERMS))
    return dict(sum2=np.zeros(shape),maximum=np.zeros(shape),max_owner=np.full(shape,-1,dtype=np.int64),signed_sum=np.zeros(shape),reference_sum2=np.zeros((len(REGIONS),m.NF,len(kernel.TERMS))),volume=np.zeros(len(REGIONS)),count=np.zeros(len(REGIONS),dtype=np.int64))
def accumulate(s,owners,volume,radial,n,N,O,R):
    errors=(N-O[:,None],np.broadcast_to(O[:,None]-R[:,None],N.shape),N-R[:,None])
    for ri,mask in enumerate(masks(radial,n,plans(n)['last_aggregate'])):
        if not mask.any():continue
        w=volume[mask];s['volume'][ri]+=w.sum();s['count'][ri]+=mask.sum();s['reference_sum2'][ri]+=np.einsum('o,oct->ct',w,R[mask]**2)
        for ei,err in enumerate(errors):
            x=err[mask];s['sum2'][ri,ei]+=np.einsum('o,obct->bct',w,x*x);s['signed_sum'][ri,ei]+=np.einsum('o,obct->bct',w,x)
            pos=abs(x).argmax(axis=0);mx=np.take_along_axis(abs(x),pos[None],axis=0)[0];improve=(mx>s['maximum'][ri,ei])|(s['max_owner'][ri,ei]<0)
            s['maximum'][ri,ei]=np.maximum(mx,s['maximum'][ri,ei]);s['max_owner'][ri,ei]=np.where(improve,owners[mask][pos],s['max_owner'][ri,ei])

def short_traces(n,block,raw,points):
    from frozen.trace import trace
    import jax.numpy as jnp
    key=sha256_array(raw);path=OUT/f'traces/N{n}/block_{block:06d}_{key[:12]}.npz';receipt=path.with_suffix('.json');identity=digest(design())
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['identity']!=identity or r['raw_hash']!=key or r['sha256']!=sha(path) or not r['passed'] or r['steps']!=64 or r['backend']!='cpu' or r['crossings'] or r['reentries']:raise ValueError('trace cache identity/hash')
        with np.load(path) as z:
            if not np.array_equal(z['raw'],raw) or not np.array_equal(z['points'],points):raise ValueError('trace cache coverage')
            ends=z['ends'].copy()
        if ends.shape!=(len(raw),4,3) or not np.isfinite(ends).all() or (ends[:,:,0]>1).any():raise ValueError('trace cache endpoint')
        return ends,r
    start=time.perf_counter();seeds=np.repeat(points,4,axis=0);dt=np.tile(np.array([-2,2,-1,1])*2*np.pi/n/256,len(raw));parts=[]
    for st in range(0,len(seeds),256):
        pp=seeds[st:st+256];dd=dt[st:st+256];count=len(pp);pp=np.pad(pp,((0,256-count),(0,0)),mode='edge');dd=np.pad(dd,(0,256-count),mode='edge')
        out=trace(ENV['ctx']['tracer'],jnp.asarray(pp),jnp.asarray(dd),steps=64)
        e,good,cross,reentry=[np.asarray(x)[:count] for x in out[:4]]
        if not good.all() or cross.any() or reentry.any() or (e[:,0]>1).any():raise ValueError('unqualified short trace')
        parts.append(e)
    ends=np.concatenate(parts).reshape(-1,4,3);finite(ends);save_arrays(path,dict(raw=raw,points=points,ends=ends))
    r=dict(identity=identity,raw_hash=key,sha256=sha(path),passed=True,steps=64,backend=jax.default_backend(),crossings=0,reentries=0,seconds=time.perf_counter()-start,relative_path=str(path.relative_to(OUT)))
    write_json(receipt,r);return ends,r

def compute(block,selected=None,return_actions=False):
    e=ENV;n=e['n'];t=e['t'];resources();start=time.perf_counter();owners=np.array(plans(n)['global_'][block])
    if selected is not None:
        if not np.isin(selected,owners).all():raise ValueError('owner selection')
        owners=np.asarray(selected)
    path=HERE/f'inputs/traces/N{n}/trace_{block:06d}.npz'
    if sha(path)!=manifest()['files'][str(path.relative_to(HERE))]:raise ValueError('trace hash')
    with np.load(path) as z:
        take=np.isin(t.ro[z['raw']],owners);rawall=z['raw'][take];endsall=z['ends'][take]
        if not z['valid'].reshape(-1,4)[take].all() or z['crossed'].reshape(-1,4)[take].any() or z['reentry'].reshape(-1,4)[take].any() or (endsall[:,:,0]>1).any():raise ValueError('unqualified wall crossing')
    if not np.array_equal(rawall,raw_members(t,owners)):raise ValueError('raw coverage')
    short,trace_receipt=short_traces(n,block,rawall,t.pts[rawall])
    stats=empty_stats();details=[];actions=[];prep=score=0.
    for batch in groups(t,owners):
        raw=raw_members(t,batch);take=np.isin(rawall,raw);ends=endsall[take];newends=short[take];cache={}
        def geom(p):
            key=sha256_array(p)
            if key not in cache:cache[key]=model.geom(e['ctx'],p)
            return cache[key]
        st=time.perf_counter()
        N,O,R,meta=kernel.evaluate(t,batch,raw,ends,newends,e['choices'][raw],e['state'],e['phi'],geom,lambda p:e['ctx']['evaluator']._position_and_jacobian(p)[1]);score+=time.perf_counter()-st
        radial=np.array([t.pts[raw_members(t,[o])[0],0]*n-.5 for o in batch]).round().astype(int)
        accumulate(stats,batch,t.vol[batch],radial,n,N,O,R);details.append(meta)
        if return_actions:actions.append((batch,N,O,R))
        resources()
    stats['owners']=owners
    meta=dict(identity=e['identity'],n=n,block=block,owners=owners.tolist(),raw_count=len(rawall),total_seconds=time.perf_counter()-start,prepare_seconds=prep,score_seconds=score,peak_rss_gib=rss(),trace_sha256=sha(path),choice_sha256=sha256_array(e['choices'][rawall]),finite=True,complete_owner_coverage=True,short_trace=trace_receipt,owner_gradient_l1=np.max([x['owner_gradient_l1'] for x in details],axis=0).tolist(),
      constant_error=max(x['constant_error'] for x in details),constant_error_by_span=np.max([x['constant_error_by_span'] for x in details],axis=0).tolist(),minimum_thermodynamic_slot=min(x['minimum_thermodynamic_slot'] for x in details),fallback_rows=sum(x['fallback_rows'] for x in details),oracle_fallback_rows=sum(x['oracle_fallback_rows'] for x in details),reference_sensitivity=max(x['reference_sensitivity'] for x in details))
    finite(*stats.values());return stats,meta,actions

def save_arrays(path,stats):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**stats)
    os.replace(tmp,path)
def checked_saved(n,block,owners):
    path=OUT/f'chunks/N{n}/block_{block:06d}.npz';rp=path.with_suffix('.json')
    if not rp.exists():return None
    r=json.loads(rp.read_text())
    if r['identity']!=digest(design()) or sha(path)!=r['sha256'] or r['owners']!=list(map(int,owners)):raise ValueError('chunk identity/coverage')
    if 'short_trace' in r and sha(OUT/r['short_trace']['relative_path'])!=r['short_trace']['sha256']:raise ValueError('short trace checksum')
    if not r['finite'] or not r['complete_owner_coverage'] or np.any(np.asarray(r.get('constant_error_by_span',[r['constant_error']]*2))>np.array([1e-7,4e-7])) or r['minimum_thermodynamic_slot']<=0:raise ValueError('chunk gate')
    with np.load(path) as z:
        expected=empty_stats()
        if set(z.files)!=set(expected)|{'owners'} or not np.array_equal(z['owners'],owners):raise ValueError('chunk schema')
        for k,v in expected.items():
            if z[k].shape!=v.shape:raise ValueError('chunk shape '+k)
        finite(*(z[k] for k in z.files))
        if int(z['count'][0])!=len(owners) or z['volume'][0]<=0 or (z['sum2']<0).any():raise ValueError('chunk invariant')
    return r

def task(block):
    if INIT_ERROR:raise RuntimeError(INIT_ERROR)
    n=ENV['n'];owners=plans(n)['global_'][block];prior=checked_saved(n,block,owners)
    if prior:return prior
    stats,meta,_=compute(block)
    if design()!=ENV['d']:raise ValueError('source changed mid-chunk')
    p=OUT/f'chunks/N{n}/block_{block:06d}.npz';save_arrays(p,stats);meta.update(sha256=sha(p),bytes=p.stat().st_size);write_json(p.with_suffix('.json'),meta);return meta

def validate(n):
    d=design();seen=[];receipts=[]
    for b,owners in enumerate(plans(n)['global_']):
        r=checked_saved(n,b,owners)
        if r is None:raise ValueError(f'missing chunk {n}/{b}')
        seen+=r['owners'];receipts.append(r['sha256'])
    if sorted(seen)!=list(range(d['owner_counts'][str(n)])):raise ValueError('owner coverage')
    write_json(OUT/f'validation_N{n}.json',dict(identity=digest(d),n=n,owners=len(seen),chunks=receipts,passed=True))
def preflight_task(job):
    if INIT_ERROR:raise RuntimeError(INIT_ERROR)
    block,selected=job;n=ENV['n'];_,meta,actions=compute(block,selected,True)
    sites=json.loads((HERE/'inputs/evidence/q07_h128_bounded/sites.json').read_text())
    with np.load(HERE/'inputs/evidence/q07_h128_bounded/arrays.npz') as old:
        worst=0.;details=[]
        for owners,N,O,R in actions:
            for i,owner in enumerate(owners):
                site=next(x for x in sites if x['n']==n and x['owner']==owner);prefix=f'N{n}_{site["name"]}'
                bn=old[prefix+'_N'];bo=old[prefix+'_O'];br=old[prefix+'_R']
                expect=np.concatenate([bn[si].transpose(1,2,0,3).reshape(4,m.NF,15) for si in (0,2)],axis=-1)
                eo=np.concatenate([bo[si].transpose(1,0,2).reshape(m.NF,15) for si in (0,2)],axis=-1)
                er=np.tile(br.transpose(1,0,2).reshape(m.NF,15),(1,2))
                for label,actual,expected in (('N',N[i],expect),('O',O[i],eo),('R',R[i],er)):
                    try:check=bounded_action_replay(actual,expected)
                    except ValueError as error:raise ValueError(f'N{n} owner={owner} action={label}: {error}') from error
                    worst=max(worst,check['centered'],check['correction'],check['combined'])
                    details.append(dict(owner=int(owner),action=label,**check))
    meta['bounded_replay']=worst;meta['bounded_replay_details']=details;return meta

def check_workers(workers):
    if workers<1:raise ValueError('positive workers required')
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    if workers>available:raise ValueError('workers exceed CPU affinity')
    budget=float(os.environ.get('Q07_TOTAL_MEMORY_GIB',str(workers*MEMORY+2)))
    if workers*MEMORY+2>budget:raise ValueError('workers exceed declared host-memory budget')

def preflight(workers):
    check_workers(workers);verify();checks=[]
    sites=json.loads((HERE/'inputs/evidence/q07_h128_bounded/sites.json').read_text())
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for n in (32,48,64):
            mapping={o:b for b,oo in enumerate(plans(n)['global_']) for o in oo};jobs={}
            for site in sites:
                if site['n']==n:jobs.setdefault(mapping[site['owner']],[]).append(site['owner'])
            jobs=[(b,sorted(owners)) for b,owners in jobs.items()]
            with mp.get_context('spawn').Pool(min(workers,len(jobs)),initializer=worker_init,initargs=(n,),maxtasksperchild=24) as pool:
                checks.extend(pool.map(preflight_task,jobs))
            print('PREFLIGHT',n,'passed',flush=True)
        write_json(OUT/'preflight.json',dict(identity=digest(design()),passed=True,sites=len(sites),workers=workers,checks=checks,max_bounded_replay=max(x['bounded_replay'] for x in checks)))

def pilot_task(block):
    if INIT_ERROR:raise RuntimeError(INIT_ERROR)
    owners=next(groups(ENV['t'],plans(64)['global_'][block]));s,meta,_=compute(block,owners);path=OUT/f'pilot/N64_{block}.npz';save_arrays(path,s);meta['bytes']=path.stat().st_size;return meta

def pilot(workers):
    gate=json.loads((OUT/'preflight.json').read_text());d=design()
    if not gate['passed'] or gate['identity']!=digest(d):raise ValueError('preflight required')
    check_workers(workers)
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with mp.get_context('spawn').Pool(min(workers,4),initializer=worker_init,initargs=(64,)) as pool:samples=pool.map(pilot_task,(0,120,256,511))
    rates={name:samples[i]['total_seconds']/samples[i]['raw_count'] for name,i in [('core',0),('inner',1),('outer',2),('wall',3)]};serial=0
    for n in (32,48,64):
        last=plans(n)['last_aggregate'];serial+=(rates['core']+rates['inner']*last+rates['outer']*(n-last-3)+rates['wall']*2)*n*n
    write_json(OUT/'pilot.json',dict(identity=digest(d),passed=True,samples=samples,regional_seconds_per_raw=rates,estimated_serial_seconds=serial,optimistic_two_worker_minutes=serial/120,conservative_two_worker_minutes=serial/120*1.5+5,estimated_output_gib=max(x['bytes'] for x in samples)*793*1.5/1024**3,max_worker_rss_gib=max(x['peak_rss_gib'] for x in samples)))

def run(n,workers):
    d=design()
    check_workers(workers)
    for name in ('preflight','pilot'):
        g=json.loads((OUT/f'{name}.json').read_text())
        if not g['passed'] or g['identity']!=digest(d):raise ValueError(name+' required')
    with (OUT/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resources();start=time.perf_counter();total=len(plans(n)['global_'])
        with mp.get_context('spawn').Pool(workers,initializer=worker_init,initargs=(n,),maxtasksperchild=24) as pool:
            for done,r in enumerate(pool.imap_unordered(task,range(total),chunksize=1),1):
                write_json(OUT/f'progress_N{n}.json',dict(identity=digest(d),completed=done,total=total,elapsed_seconds=time.perf_counter()-start,workers=workers));print('DONE',n,done,total,flush=True)
        validate(n)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','verify','preflight','pilot','run','validate','analyze','validate-completion'));p.add_argument('--n',type=int,choices=(32,48,64));p.add_argument('--workers',type=int);a=p.parse_args()
    if a.command in ('run','validate') and a.n is None:p.error('--n required')
    if a.command in ('run','preflight','pilot') and a.workers is None:p.error('--workers required')
    OUT.mkdir(parents=True,exist_ok=True)
    if a.command=='run':run(a.n,a.workers)
    elif a.command in ('preflight','pilot'):globals()[a.command](a.workers)
    elif a.command=='validate':validate(a.n)
    elif a.command in ('analyze','validate-completion'):
        import analyze;analyze.main(validate_only=a.command=='validate-completion')
    else:globals()[a.command]()
if __name__=='__main__':
    sys.modules['campaign']=sys.modules[__name__]
    main()
