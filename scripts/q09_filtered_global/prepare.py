"""Checkpointed GPU traces and spawn-parallel host preparation for filtered Q09."""
from pathlib import Path
import concurrent.futures
import multiprocessing
import time
import numpy as np
from . import common as c
from .geometry import setup

ENV=None


class SelectedEndpoints:
    """Bounded worker test only: reject every untraced raw index explicitly."""
    def __init__(self,raw,ends):
        self.lookup={int(r):i for i,r in enumerate(raw)};self.ends=ends
    def __getitem__(self,raw):
        return self.ends[[self.lookup[int(r)] for r in raw]]


def bounded_worker_probe(run,ident):
    """Fresh spawned-process initialization, actual bank I/O and exact resume."""
    global ENV
    from scripts.q09_evolved_mms.bootstrap import configure
    configure(run,gpu=False)
    from drbx.stencils.q_artifact import load_q_bank
    run=Path(run);n=32;e=setup(run,n,tracing=False)
    bank=load_q_bank(run/'bounded/N32/bank.npz')
    with np.load(run/'bounded/N32/data.npz') as z:ends=z['ends'].copy()
    dest=run/'worker_preflight';dest.mkdir(parents=True,exist_ok=True)
    c.write(dest/'traces_N32.json',dict(identity=ident,scope='bounded process/I/O probe only',raw=bank.raw.tolist(),source=c.sha(run/'bounded/N32/data.npz')))
    ENV=(dest,n,ident,4.,e,SelectedEndpoints(bank.raw,ends))
    task=(0,bank.owners.tolist());first=worker(task);again=worker(task)
    if first!=again:raise ValueError('bounded worker checkpoint was recomputed')
    actual=load_q_bank(dest/'cpu/N32/chunk_000000/bank.npz')
    for key in bank.arrays:
        # Same CPU shape/backend; provenance differs but numerical rows do not.
        np.testing.assert_array_equal(actual.arrays[key],bank.arrays[key])
    return first

def trace_points(e,points,*,capacity=16384,steps=64):
    from .trace_kernel import trace
    import jax.numpy as jnp
    points=np.asarray(points); seeds=np.repeat(points,4,axis=0)
    delta=2*np.pi/e['t'].n/32
    # prepare_paired_chunks expects outer-/outer+/inner-/inner+, not monotone order.
    ds=np.tile([-1.,1.,-.5,.5],len(points))*delta
    out=[[] for _ in range(6)]
    for start in range(0,len(seeds),capacity):
        p=seeds[start:start+capacity]; d=ds[start:start+capacity]; k=len(p)
        padded=np.concatenate((p,np.tile([[.5,0.,0.]],(capacity-k,1))))
        result=trace(e['ctx']['tracer'],e['jtable'],jnp.asarray(padded),jnp.asarray(np.r_[d,np.zeros(capacity-k)]),steps=steps)
        for j in range(6): out[j].append(np.asarray(result[j])[:k])
    ends,valid,cross,reentry,maxu,minj=map(np.concatenate,out)
    record=dict(trajectories=len(seeds),valid=int(valid.sum()),crossings=int(cross.sum()),reentries=int(reentry.sum()),exterior=int((ends[:,0]>1).sum()),max_u=float(maxu.max()),min_j=float(minj.min()),steps=steps)
    if not valid.all() or cross.any() or reentry.any() or np.any(ends[:,0]>1) or not np.isfinite(ends).all():
        raise ValueError(('unqualified trace reach/crossing',record))
    np.testing.assert_allclose(ends[:,2],seeds[:,2]+ds,atol=0,rtol=0)
    return ends.reshape(-1,4,3),record


def trace_grid(run,n,ident,*,batch_raw=4096):
    import jax
    from scripts.q09_evolved_mms.campaign import hardware
    from scripts.q08_extraction_global.common import atomic_npz
    if jax.default_backend()!='gpu': raise ValueError('full tracing requires GPU')
    inventory=hardware()
    if (Path(run)/f'traces_N{n}.json').exists():
        return c.require(run,f'traces_N{n}',ident)
    e=setup(run,n);folder=Path(run)/'traces'/f'N{n}';folder.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter();files={}; count=0
    for start in range(0,n**3,batch_raw):
        stop=min(start+batch_raw,n**3);path=folder/f'{start:07d}.npz';receipt=path.with_suffix('.json')
        if receipt.exists():
            r=c.read(receipt)
            if r.get('identity')!=ident or (r['start'],r['stop'])!=(start,stop) or r['sha256']!=c.sha(path): raise ValueError('trace checkpoint changed')
        else:
            ends,r=trace_points(e,e['t'].pts[start:stop],capacity=4*batch_raw)
            atomic_npz(path,raw=np.arange(start,stop),points=e['t'].pts[start:stop],ends=ends)
            r.update(passed=True,identity=ident,start=start,stop=stop,sha256=c.sha(path));c.write(receipt,r)
        files[str(path.relative_to(run))]=c.sha(path);files[str(receipt.relative_to(run))]=c.sha(receipt);count+=stop-start
        print(f'trace N{n} {count}/{n**3}',flush=True)
    c.write(Path(run)/f'traces_N{n}.json',dict(passed=True,identity=ident,n=n,raw=count,seconds=time.perf_counter()-tick,hardware=inventory,files=files))


def load_traces(run,n,ident):
    r=c.require(run,f'traces_N{n}',ident);ends=[];raw=[];points=[]
    for name in sorted(r['files']):
        if not name.endswith('.npz'): continue
        with np.load(Path(run)/name) as z: raw.append(z['raw']);points.append(z['points']);ends.append(z['ends'])
    if not np.array_equal(np.concatenate(raw),np.arange(n**3)): raise ValueError('trace coverage')
    return np.concatenate(ends),np.concatenate(points)


def build(e,owners,ends,ident,trace_identity):
    from drbx.stencils.q_parallel import prepare_paired_chunks,raw_members
    from drbx.stencils.q_parallel_rhs import prepare_six_field_rhs
    from drbx.stencils.q_bank import build_q_bank
    from .reference import prepare
    owners=np.asarray(owners,dtype=np.int64);raw=raw_members(e['t'],owners)
    if len(raw)>128: raise ValueError('whole-owner reference capacity')
    outer,inner=prepare_paired_chunks(e['t'],owners,ends,e['geom'],e['jac'],raw=raw,source_identity=trace_identity,geometry_identity=c.GEOMETRY_ID)
    bank=build_q_bank(outer,inner,identity=dict(filtered_campaign=ident,arm=c.ARM))
    rt,_=prepare_six_field_rhs(inner,outer,e['geom'],diffusion_span=1/32,center_b_atol=1e-10)
    geom=dict(magnetic_L=rt.material.magnetic_L,b_eta=rt.material.b_eta,eta_step=rt.material.eta_step,bmag=rt.bmag)
    ref=prepare(bank,geom,e['geom'])
    return bank,geom,ref


def chunk_valid(folder,ident,owners):
    p=folder/'stats.json'
    if not p.exists(): return None
    r=c.read(p)
    if r.get('campaign_identity')!=ident or r.get('passed') is not True or r.get('owners')!=list(owners): raise ValueError('chunk checkpoint lineage')
    if set(r['files'])!={'bank.npz','geometry.npz','reference.npz','reference.json'}: raise ValueError('chunk payload coverage')
    for name,h in r['files'].items():
        if c.sha(folder/name)!=h: raise ValueError('chunk content changed')
    return r


def init_worker(run,n,ident,worker_gib):
    global ENV
    from scripts.q09_evolved_mms.bootstrap import configure
    configure(run,gpu=False)
    ends,points=load_traces(run,n,ident);e=setup(run,n,tracing=False)
    np.testing.assert_array_equal(points,e['t'].pts)
    ENV=(Path(run),n,ident,worker_gib,e,ends)


def worker(task):
    from scripts.q08_extraction_global.common import atomic_npz
    from scripts.q09_evolved_mms.campaign import resource_measurement
    from drbx.stencils.q_parallel import raw_members
    from drbx.stencils.q_artifact import save_q_bank,load_q_bank
    from .reference import FIELDS
    run,n,ident,limit,e,ends=ENV;index,owners=task;folder=run/'cpu'/f'N{n}'/f'chunk_{index:06d}'
    old=chunk_valid(folder,ident,owners)
    if old is not None: return old
    folder.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();raw=raw_members(e['t'],np.asarray(owners))
    bank,geom,ref=build(e,owners,ends[raw],ident,c.sha(run/f'traces_N{n}.json'))
    save_q_bank(bank,folder/'bank.npz');load_q_bank(folder/'bank.npz',expected_identity=bank.identity)
    atomic_npz(folder/'geometry.npz',**geom);atomic_npz(folder/'reference.npz',**{k:np.asarray(getattr(ref,k)) for k in FIELDS})
    c.write(folder/'reference.json',ref.diagnostics)
    resources=resource_measurement();rss=resources['peak_host_rss_bytes']/2**30
    if rss>limit: raise MemoryError(f'worker peak {rss} GiB exceeds {limit}')
    r=dict(passed=True,campaign_identity=ident,n=n,index=index,owners=list(owners),raw=raw.tolist(),bank_identity=bank.identity,
        seconds=time.perf_counter()-tick,resources=resources,footprint=bank.footprint(),
        files={k:c.sha(folder/k) for k in ('bank.npz','geometry.npz','reference.npz','reference.json')})
    c.write(folder/'stats.json',r);return r


def run_cpu(run,n,ident,workers,worker_gib,host_gib,*,pilot=False):
    import os
    plan=c.read(Path(run)/'inputs/plan.json')[str(n)]['chunks']
    selected=sorted(set([0,len(plan)//3,2*len(plan)//3,len(plan)-1])) if pilot else range(len(plan))
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    if workers<1 or workers>affinity or workers*worker_gib+4>host_gib: raise MemoryError('worker/affinity/host admission refused')
    c.require(run,f'traces_N{n}',ident);tick=time.perf_counter()
    completed=Path(run)/f'{"pilot" if pilot else "cpu"}_N{n}.json'
    if completed.exists():
        receipt=c.require(run,completed.stem,ident)
        for i in selected:
            if chunk_valid(Path(run)/'cpu'/f'N{n}'/f'chunk_{i:06d}',ident,plan[i]) is None:
                raise ValueError('completed CPU stage lost a chunk')
        return receipt
    pending=[(i,plan[i]) for i in selected if chunk_valid(Path(run)/'cpu'/f'N{n}'/f'chunk_{i:06d}',ident,plan[i]) is None]
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(str(run),n,ident,worker_gib)) as pool:
            # At most 2*workers submitted; no unbounded futures/arrays on the controller.
            it=iter(pending);live={};done=0
            def submit():
                try: item=next(it)
                except StopIteration: return
                live[pool.submit(worker,item)]=item[0]
            for _ in range(min(len(pending),2*workers)): submit()
            while live:
                ready,_=concurrent.futures.wait(live,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in ready:
                    f.result();del live[f];done+=1;submit()
                    print(f'prepare N{n} {done}/{len(pending)} new chunks',flush=True)
    receipts=[chunk_valid(Path(run)/'cpu'/f'N{n}'/f'chunk_{i:06d}',ident,plan[i]) for i in selected]
    seconds=[r['seconds'] for r in receipts];raw=sum(len(r['raw']) for r in receipts)
    r=dict(passed=True,identity=ident,n=n,chunks=len(receipts),raw=raw,workers=workers,seconds=time.perf_counter()-tick,
        per_chunk_seconds=seconds,predicted_cpu_seconds=float(np.mean(seconds)*len(plan)/workers),
        max_worker_rss_gib=max(r['resources']['peak_host_rss_bytes']/2**30 for r in receipts),
        files={f'cpu/N{n}/chunk_{i:06d}/stats.json':c.sha(Path(run)/'cpu'/f'N{n}'/f'chunk_{i:06d}'/'stats.json') for i in selected})
    if not pilot:
        if raw!=n**3 or [o for x in receipts for o in x['owners']]!=list(range(c.read(Path(run)/'inputs/plan.json')[str(n)]['n_owner'])): raise ValueError('complete owner coverage')
    c.write(Path(run)/f'{"pilot" if pilot else "cpu"}_N{n}.json',r);return r
