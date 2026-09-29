"""Portable node-local GPU-trace / CPU-reconstruction Q campaign."""
import argparse,os,json,time,subprocess,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import multiprocessing as mp
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['JAX_ENABLE_X64']='true';os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import storage as io
HERE=Path(__file__).resolve().parent
NS=(32,48,64)

def verify_inputs(root,input_root):
    entries=io.read(HERE/'input_manifest.json')['files'];audit=[]
    for z in entries:
        p=Path(input_root)/z['path'];ok=p.is_file() and p.stat().st_size==z['bytes'] and io.sha(p)==z['sha256'];audit.append(dict(**z,absolute_path=str(p.resolve()),mtime_ns=p.stat().st_mtime_ns if p.exists() else None,verified=ok))
    io.write(Path(root)/'provenance/input_audit.json',audit)
    if not all(x['verified'] for x in audit):raise RuntimeError('input verification failed; see provenance/input_audit.json')

def input_guard(root,input_root):
    audit=io.read(Path(root)/'provenance/input_audit.json')
    for z in audit:
        p=Path(input_root)/z['path']
        if not z['verified'] or str(p.resolve())!=z['absolute_path'] or not p.exists() or p.stat().st_size!=z['bytes'] or p.stat().st_mtime_ns!=z['mtime_ns']:
            raise RuntimeError('input path/stat changed; rerun prescribed verify before resuming')

def groups(t,owners,capacity):
    out=[];current=[];count=0
    for o in owners:
        c=int(t.starts[o+1]-t.starts[o])
        if c>capacity:raise ValueError('raw chunk must fit the complete largest owner')
        if count+c>capacity:out.append(current);current=[];count=0
        current.append(int(o));count+=c
    if current:out.append(current)
    return out

def initialize(a):
    from .model import context
    from .fields import FIELDS, BASELINE_FIELDS
    root=Path(a.campaign);root.mkdir(parents=True,exist_ok=True)
    if (root/'design.json').exists():
        _,d=io.identity(root)
        if d['raw_chunk']!=a.raw_chunk or d['test_mode']!=a.test_mode:raise RuntimeError('existing design differs')
        verify_inputs(root,a.input_root);return
    verify_inputs(root,a.input_root)
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    except subprocess.CalledProcessError:commit='unavailable'
    design=dict(schema='q-layered-balanced28-dn-v2',commit=commit,sources=io.sources(),inputs=io.read(HERE/'input_manifest.json'),raw_chunk=a.raw_chunk,rk4_steps=64,spans=[1/16,1/32],BC=['D','N'],test_mode=a.test_mode,input_root=str(Path(a.input_root).resolve()),inner_policy='balanced28',fields=list(FIELDS),baseline_fields=list(BASELINE_FIELDS),method='always balanced28 inner / ringwise outer / two-layer quartic D and physical-normal N / fixed five-plane eta / divB')
    plans={}
    for n in NS:
        _,t=context(n,a.input_root,physical=False,magnetic=False)
        profile=np.array([len(np.unique(t.ro.reshape((n,)*3)[i,:,0])) for i in range(n)]);last=int(np.flatnonzero(profile==n)[0])-1;sample=set()
        for th,et in ((1.83,2.41),(4.37,5.12),(0,0),(.73,1.19),(2.91,4.07)):
            j,k=[int(np.argmin(abs((t.centers[x+1]-v+np.pi)%(2*np.pi)-np.pi))) for x,v in enumerate((th,et))]
            for i in sorted({0,1,2,last-1,last,last+1,last+2,int(.24*n),int(.26*n),{32:7,48:12,64:18}[n],{32:8,48:12,64:16}[n],n//2,3*n//4,n-4,n-3,n-2,n-1}):sample.add(int(t.ro[(i*n+j)*n+k]))
        full=groups(t,range(len(t.vol)),a.raw_chunk);pre=groups(t,sorted(sample),a.raw_chunk);pick=np.unique(np.linspace(0,len(full)-1,12,dtype=int));pilot=[full[i] for i in pick]
        plans[str(n)]=dict(geometry=full,preflight=pre,pilot=pilot,global_=full,owner_count=len(t.vol),raw_count=n**3,last_aggregate=last)
    io.write(root/'plan.json',plans);design['plan_sha256']=io.sha(root/'plan.json');io.write(root/'design.json',design)

def plan(root,stage,n):
    _,d=io.identity(root)
    if io.sha(Path(root)/'plan.json')!=d['plan_sha256']:raise RuntimeError('plan changed')
    p=io.read(Path(root)/'plan.json')[str(n)];return p['global_' if stage=='global' else stage]

def require(root,relative):
    p=Path(root)/relative;ident,_=io.identity(root)
    if not p.exists() or io.read(p).get('identity')!=ident or not io.read(p).get('operational_pass',False):raise RuntimeError(f'missing valid gate: {p}')

def workers(a):
    if a.workers<1 or a.memory_gib<=0 or a.per_worker_gib<=0:raise ValueError('positive worker and memory limits required')
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count();cap=int(a.memory_gib//a.per_worker_gib)
    if cap<1:raise ValueError('memory budget below one worker')
    return min(a.workers,affinity,cap)

def run_cpu(a,mode):
    os.environ['JAX_PLATFORMS']='cpu'
    from .worker import init,task
    root=Path(a.campaign);ident,d=io.identity(root);stage='geometry' if mode=='geometry' else a.stage
    if mode=='score':
        require(root,f'{stage}/trace_N{a.N}.json')
        if stage in ('pilot','global'):require(root,f'preflight/check_N{a.N}.json');require(root,f'preflight/validate_N{a.N}.json')
        if stage=='global':require(root,f'pilot/validate_N{a.N}.json')
    jobs=[(stage,b,o) for b,o in enumerate(plan(root,stage,a.N))];num=min(workers(a),len(jobs));start=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=num,mp_context=mp.get_context('spawn'),initializer=init,initargs=(root,a.input_root,a.N,mode)) as pool:
        for i,res in enumerate(pool.map(task,jobs,chunksize=1)):
            results.append(res)
            if (i+1)%25==0:print(json.dumps(dict(stage=stage,N=a.N,complete=i+1,total=len(jobs))),flush=True)
    peak=max(io.read((root/stage/f'N{a.N}'/f'{mode}_{b:06d}.npz').with_suffix('.json'))['peak_rss_gib'] for b in range(len(jobs)))
    io.write(root/stage/f'run_N{a.N}.json',dict(identity=ident,requested_workers=a.workers,effective_workers=num,memory_gib=a.memory_gib,per_worker_gib=a.per_worker_gib,peak_worker_gib=peak,wall_s=time.monotonic()-start,backend='cpu',chunks=len(jobs),operational_pass=True))
    if mode=='geometry':validate(a,stage='geometry',prefix='geometry')

def trace_stage(a):
    root=Path(a.campaign);ident,d=io.identity(root)
    if a.backend=='cpu-test' and not d['test_mode']:raise RuntimeError('CPU test tracing requires a test-only campaign')
    if a.backend=='cpu-test' and a.stage!='preflight':raise RuntimeError('CPU test tracing limited to preflight')
    if a.stage=='global' and d['test_mode']:raise RuntimeError('test campaign cannot run global')
    if a.stage in ('pilot','global'):
        require(root,f'geometry/validate_N{a.N}.json');require(root,f'preflight/check_N{a.N}.json');require(root,f'preflight/validate_N{a.N}.json')
    if a.stage=='global':require(root,f'pilot/validate_N{a.N}.json')
    os.environ['JAX_PLATFORMS']='cpu' if a.backend=='cpu-test' else 'cuda,cpu'
    if a.backend=='gpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='':raise RuntimeError('GPU visibility hidden')
    from .model import context,ALPHAS
    from .trace import trace_fixed_batch
    import jax
    devices=jax.local_devices(backend='cpu' if a.backend=='cpu-test' else 'gpu')
    selected=[devices[i] for i in a.devices] if a.backend=='gpu' else devices[:1]
    if len(set(a.devices))!=len(a.devices):raise ValueError('duplicate GPU indices')
    with jax.default_device(jax.local_devices(backend='cpu')[0]):ctx,t=context(a.N,a.input_root)
    host=ctx['tracer'];resident=[(dev,jax.device_put(host,dev)) for dev in selected];jobs=list(enumerate(plan(root,a.stage,a.N)));start=time.monotonic()
    import queue
    todo=queue.Queue()
    for job in jobs:todo.put(job)
    def consume(dev,field):
        while True:
            try:block,owners=todo.get_nowait()
            except queue.Empty:return
            path=root/a.stage/f'N{a.N}'/f'trace_{block:06d}.npz'
            if io.complete(path,ident):continue
            raw=np.concatenate([t.order[t.starts[o]:t.starts[o+1]] for o in owners]);seeds=np.repeat(t.pts[raw],4,axis=0);delta=np.tile(np.ravel(np.column_stack((-ALPHAS*t.g.deta/2,ALPHAS*t.g.deta/2))),len(raw));st=time.monotonic()
            z=trace_fixed_batch(field,seeds,delta,64,4*d['raw_chunk'],dev)
            if not z[1].all() or z[3].any() or max(np.maximum(z[4]-1,0))*a.N>2:raise RuntimeError(('trace validity/reentry/reach',a.N,block))
            io.arrays(path,raw=raw,ends=z[0].reshape(-1,4,3),valid=z[1],crossed=z[2],reentry=z[3],max_u=z[4],min_J=z[5],first_cross=z[6],max_point=z[7],cross_point=z[8])
            io.record(path,ident,backend=dev.platform,device=str(dev),rk4_steps=64,wall_s=time.monotonic()-st,raw=len(raw))
    with ThreadPoolExecutor(max_workers=len(resident)) as pool:
        fs=[pool.submit(consume,*x) for x in resident]
        for f in fs:f.result()
    for block,_ in jobs:
        path=root/a.stage/f'N{a.N}'/f'trace_{block:06d}.npz'
        if not io.complete(path,ident):raise RuntimeError('incomplete trace stage')
        if io.read(path.with_suffix('.json'))['backend']!=('gpu' if a.backend=='gpu' else 'cpu'):raise RuntimeError('mixed trace backend')
    io.write(root/a.stage/f'trace_N{a.N}.json',dict(identity=ident,backend=a.backend,devices=[str(x) for x in selected],jax_version=jax.__version__,rk4_steps=64,chunks=len(jobs),wall_s=time.monotonic()-start,operational_pass=True))

def check_trace(a):
    os.environ['JAX_PLATFORMS']='cpu'
    from .model import context,ALPHAS
    from .trace import trace_fixed_batch
    import jax
    root=Path(a.campaign);ident,d=io.identity(root);require(root,f'preflight/trace_N{a.N}.json');ctx,t=context(a.N,a.input_root);dev=jax.devices('cpu')[0];errs=[0.,0.];count=0
    jobs=plan(root,'preflight',a.N)
    for b in np.unique(np.linspace(0,len(jobs)-1,min(8,len(jobs)),dtype=int)):
        p=root/'preflight'/f'N{a.N}'/f'trace_{b:06d}.npz';io.complete(p,ident);z=np.load(p)
        radial=z['raw']//(a.N*a.N)
        first_each_ring=np.unique(radial,return_index=True)[1]
        ix=np.unique(np.r_[first_each_ring,np.linspace(0,len(z['raw'])-1,min(16,len(z['raw'])),dtype=int)])
        raw=z['raw'][ix];seeds=np.repeat(t.pts[raw],4,axis=0);delta=np.tile(np.ravel(np.column_stack((-ALPHAS*t.g.deta/2,ALPHAS*t.g.deta/2))),len(raw))
        capacity=64*math.ceil(len(seeds)/64)
        results=[trace_fixed_batch(ctx['tracer'],seeds,delta,steps,capacity,dev) for steps in (64,256)];saved=z['ends'][ix].reshape(-1,3)
        for j,other in enumerate((saved,results[1][0])):
            diff=results[0][0]-other;diff[:,1:]=(diff[:,1:]+np.pi)%(2*np.pi)-np.pi;errs[j]=max(errs[j],float(abs(diff*np.array([a.N,a.N/(2*np.pi),a.N/(2*np.pi)])).max()))
        if not all(v[1].all() and not v[3].any() for v in results):raise RuntimeError('CPU step trace validity')
        count+=len(raw)
    if errs[0]>1e-8 or errs[1]>1e-7:raise RuntimeError(('GPU/CPU or RK4 step comparison',errs))
    backend=io.read(root/f'preflight/trace_N{a.N}.json')['backend']
    io.write(root/f'preflight/check_N{a.N}.json',dict(identity=ident,operational_pass=True,backend=backend,raw_checked=count,cpu_gpu_scaled_error=errs[0],rk64_256_scaled_error=errs[1]))

def validate(a,stage=None,prefix='score'):
    root=Path(a.campaign);stage=stage or a.stage;ident,d=io.identity(root);jobs=plan(root,stage,a.N);expected=np.array([o for job in jobs for o in job]);seen=[];maxconst=0.;meta=[]
    for b,owners in enumerate(jobs):
        path=root/stage/f'N{a.N}'/f'{prefix}_{b:06d}.npz'
        if not io.complete(path,ident):raise RuntimeError(f'incomplete chunk {path}')
        z=np.load(path);assert np.array_equal(z['owners'],owners);seen.extend(z['owners'].tolist());m=io.read(path.with_suffix('.json'));maxconst=max(maxconst,m['constant_max']);meta.append(m)
        if prefix=='score' and not all(np.all(np.isfinite(z[k])) for k in ('N','O','R','R_half')):raise RuntimeError('nonfinite checkpoint')
    if len(set(seen))!=len(seen) or not np.array_equal(expected,seen):raise RuntimeError('owner coverage failure')
    io.write(root/stage/f'validate_N{a.N}.json',dict(identity=ident,operational_pass=True,chunks=len(jobs),owners=len(seen),raw=sum(z['raw'] for z in meta),max_constant=maxconst,max_condition=max(z['max_condition'] for z in meta),max_reproduction=max(z['max_reproduction'] for z in meta),max_donors=max(z['max_donors'] for z in meta),repair_planes=sum(z['repair_planes'] for z in meta),balanced_planes=sum(z['balanced_planes'] for z in meta),inner_plane_donor_counts=sorted({c for z in meta for c in z['inner_plane_donor_counts']}),inner_max_pool=max(z['inner_max_pool'] for z in meta),inner_max_scaled_radius=max(z['inner_max_scaled_radius'] for z in meta),inner_expanded_planes=sum(z['inner_expanded_planes'] for z in meta),inner_policy='balanced28'))

def reduce(a):
    from .fields import FIELDS
    root=Path(a.campaign);ident,_=io.identity(root);result=[]
    for n in NS:
        require(root,f'global/validate_N{n}.json');jobs=plan(root,'global',n);zs=[]
        for block in range(len(jobs)):
            path=root/'global'/f'N{n}'/f'score_{block:06d}.npz'
            if not io.complete(path,ident):raise RuntimeError('missing reduction input')
            with np.load(path) as z:zs.append({k:z[k] for k in ('owners','N','O','R','R_half','radial','volume')})
        D={k:np.concatenate([z[k] for z in zs]) for k in ('owners','N','O','R','R_half','radial','volume')};p=io.read(root/'plan.json')[str(n)];assert len(D['owners'])==p['owner_count']
        io.arrays(root/f'results_N{n}.npz',**D,fields=np.array(FIELDS))
        last=p['last_aggregate'];rad=D['radial'];regions={'global':np.ones(len(rad),bool),'core':rad==0,'first_ring':rad==1,'inner':rad<=last,'inner_join':(rad>=last-1)&(rad<=last),'aggregate_join':(rad>=last-1)&(rad<=last+2),'outer':(rad>last)&(rad<n-2),'wall':rad>=n-2,'wall_join':(rad>=n-4)&(rad<=n-2)}
        for bc in range(2):
         for ai,alpha in enumerate((1/16,1/32)):
          for field,name in enumerate(FIELDS):
           for region,mask in regions.items():
            wt=D['volume'][mask];wt=wt/wt.sum();N=D['N'][mask,bc,ai,field];O=D['O'][mask,ai,field];R=D['R'][mask,field];Rh=D['R_half'][mask,field]
            row=dict(Ngrid=n,BC=('D','N')[bc],alpha=alpha,field=name,region=region)
            for label,v in dict(NO=N-O,OR=O-R,NR=N-R,reference_step=R-Rh).items():
                row[label]=float(np.sqrt(wt@abs(v)**2));row[label+'_max']=float(abs(v).max());row[label+'_max_owner']=int(D['owners'][mask][np.argmax(abs(v))])
            row['reference_step_fraction_NR']=row['reference_step']/max(row['NR'],1e-300)
            result.append(row)
    orders=[]
    for bc in ('D','N'):
     for alpha in (1/16,1/32):
      for field in FIELDS:
       for region in regions:
        series=[next(x for x in result if x['BC']==bc and x['alpha']==alpha and x['field']==field and x['region']==region and x['Ngrid']==n) for n in NS]
        for channel in ('NO','OR','NR','NO_max','OR_max','NR_max'):
            v=[x[channel] for x in series];slopes=[math.log(max(v[i],1e-300)/max(v[i+1],1e-300))/math.log(NS[i+1]/NS[i]) for i in range(2)]
            orders.append(dict(BC=bc,alpha=alpha,field=field,region=region,channel=channel,orders=slopes,order_ge_1p8=all(s>=1.8 for s in slopes) if field!='constant' else None))
    io.write(root/'summary.json',dict(identity=ident,fields=list(FIELDS),inner_policy='balanced28',norms=result,orders=orders,interpretation='machine output only; N-O and O-R separate; no automatic promotion'))
    io.write(root/'completion.json',dict(identity=ident,operational_pass=True,resolutions=list(NS),complete=True,scientific_pass_not_required_for_completion=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('init','verify','geometry','trace','check','score','validate','reduce'));p.add_argument('--campaign',required=True);p.add_argument('--input-root',required=True);p.add_argument('--N',type=int,choices=NS);p.add_argument('--stage',choices=('preflight','pilot','global'),default='preflight');p.add_argument('--raw-chunk',type=int,default=128);p.add_argument('--test-mode',action='store_true');p.add_argument('--backend',choices=('gpu','cpu-test'),default='gpu');p.add_argument('--devices',nargs='+',type=int,default=[0,1,2,3]);p.add_argument('--workers',type=int);p.add_argument('--memory-gib',type=float);p.add_argument('--per-worker-gib',type=float,default=2.5);a=p.parse_args()
    if a.raw_chunk<64:p.error('raw chunk must be at least64')
    if a.command not in ('init','verify','reduce') and a.N is None:p.error('--N required')
    if a.command in ('geometry','score') and (a.workers is None or a.memory_gib is None):p.error('explicit --workers and --memory-gib required')
    root=Path(a.campaign);(root/'cache').mkdir(parents=True,exist_ok=True);os.environ['JAX_COMPILATION_CACHE_DIR']=str((root/'cache'/('gpu' if a.command=='trace' and a.backend=='gpu' else 'cpu')).resolve());os.environ['DRBX_CACHE_DIR']=os.environ['JAX_COMPILATION_CACHE_DIR']
    with io.lock(root):
        if a.command=='init':initialize(a)
        else:
            io.identity(root)
            if a.command!='verify':input_guard(root,a.input_root)
            if a.command=='verify':verify_inputs(root,a.input_root)
            elif a.command=='geometry':run_cpu(a,'geometry')
            elif a.command=='trace':trace_stage(a)
            elif a.command=='score':run_cpu(a,'score')
            elif a.command=='check':check_trace(a)
            elif a.command=='validate':validate(a)
            elif a.command=='reduce':reduce(a)
    print(json.dumps(dict(command=a.command,completed=True)),flush=True)
if __name__=='__main__':main()
