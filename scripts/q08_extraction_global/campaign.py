"""Resumable Q08 implementation replay; CPU preparation then real GPU RHS audit."""
from __future__ import annotations
import argparse, concurrent.futures as cf, fcntl, hashlib, json, multiprocessing as mp
import os, platform, resource, shutil, sys, tarfile, time, math
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from scripts.q08_extraction_global.bootstrap import configure
NS=(32,48,64)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
def source_files():
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*.py'))
            if not any(part in ('inputs','__pycache__') for part in p.relative_to(HERE).parts)}
def design():
    d=read(HERE/'design.json')
    if source_files()!=d['sources'] or sha(HERE/'input_manifest.json')!=d['input_manifest_sha256']:
        raise ValueError('frozen source/input manifest changed')
    if sha(HERE/'inputs.tar.gz')!=d['input_archive_sha256']:raise ValueError('input archive changed')
    return d

def freeze():
    """Local maintainer operation, never part of the remote execution contract."""
    repo=HERE.parents[1];runtime=HERE/'runtime/drbx'
    for group in ('','native','stencils','geometry'):
        p=runtime/group;p.mkdir(parents=True,exist_ok=True);(p/'__init__.py').write_text('')
    paths=[*repo.glob('src/drbx/native/q*.py'),*repo.glob('src/drbx/stencils/q*.py')]
    paths += [repo/'src/drbx'/s for s in ('_host_guards.py','geometry/_reconstruction_primitives.py','stencils/query_tables.py','native/owner_plane_layout.py','native/fci_parallel_production_flux.py','native/characteristic_wall_residual.py')]
    source={}
    for p in paths:
        rel=p.relative_to(repo/'src/drbx');shutil.copy2(p,runtime/rel);source[str(p.relative_to(repo))]=sha(p)
    d=dict(schema='q08-extraction-global-v1',sources=source_files(),package_source=source,
           input_manifest_sha256=sha(HERE/'input_manifest.json'),input_archive_sha256=sha(HERE/'inputs.tar.gz'),
           n=list(NS),material_inner_span=1/32,material_outer_span=1/16,diffusion_spans=[1/32,1/16],
           magnetic_evaluator='compact_c3',saved_trace_steps=64,new_tracing=False,new_continuum_references=False,
           cases=22,boundary_patterns=4,fields=6,devices=[1,4],max_raw_per_chunk=128,
           row_replay='shape/dtype/bytes exact on same CPU',action_atol=1e-8,action_rtol=1e-11,
           center_b_atol=1e-10,cross_platform_prepared_array_comparison=False,
           scope='Full-grid extraction and GPU execution replay; no new physical qualification or promotion')
    write(HERE/'design.json',d);print(digest(d))

def require(run,stage,identity):
    r=read(run/f'{stage}.json')
    if not r.get('passed') or r.get('campaign_identity')!=identity:raise ValueError(stage+' identity/gate')
    return r

def check_inputs(run,root):
    """Content checks on every stage, not only the initial extraction."""
    mf=read(HERE/'input_manifest.json')
    for base,mapping in ((run/'inputs',mf['files']),(root,mf['canonical'])):
        for rel,h in mapping.items():
            if sha(base/rel)!=h:raise ValueError('frozen input changed '+str(base/rel))

def verify(run,root,d):
    if (run/'verification.json').exists():
        require(run,'verification',digest(d))
        if read(run/'verification.json')['canonical_root']!=str(root):raise ValueError('existing RUN canonical root mismatch')
    mf=read(HERE/'input_manifest.json');inp=run/'inputs';inp.mkdir(parents=True,exist_ok=True)
    # Only regular, listed relative files may be extracted. No links or paths
    # from the archive control destinations outside RUN.
    with tarfile.open(HERE/'inputs.tar.gz','r:gz') as tar:
        members=tar.getmembers();expected={'inputs/'+k for k in mf['files']}
        if {m.name for m in members}!=expected or len(members)!=len(expected):raise ValueError('archive membership')
        for member in members:
            rel=Path(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe archive')
            target=run/rel
            if target.exists():
                if sha(target)!=mf['files'][str(rel.relative_to('inputs'))]:raise ValueError('existing extracted input changed '+str(target))
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.name+'.extracting')
            with tar.extractfile(member) as src,tmp.open('wb') as out:shutil.copyfileobj(src,out)
            os.replace(tmp,target)
    for rel,h in mf['files'].items():
        if sha(inp/rel)!=h:raise ValueError('input checksum '+rel)
    for rel,h in mf['canonical'].items():
        if sha(root/rel)!=h:raise ValueError('canonical checksum '+rel)
    import jax,numpy as np
    if jax.default_backend()!='cpu' or not jax.config.jax_enable_x64:raise ValueError('CPU x64 required')
    result=dict(passed=True,campaign_identity=digest(d),canonical_root=str(root),canonical=mf['canonical'],
                archive_sha256=d['input_archive_sha256'],python=sys.version,platform=platform.platform(),
                versions=dict(jax=jax.__version__,numpy=np.__version__),input_count=len(mf['files']))
    write(run/'verification.json',result);write(run/'provenance/design.json',d);return result

def check_resources(run,workers,worker_gib,host_gib):
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    if workers<1 or workers>affinity:raise ValueError('workers exceed effective CPU affinity')
    if not math.isfinite(worker_gib) or not math.isfinite(host_gib) or worker_gib<3 or workers*worker_gib+4>host_gib:raise ValueError('workers * worker GiB + 4 must fit host memory budget')
    if shutil.disk_usage(run).free<3*1024**3:raise RuntimeError('free disk below 3 GiB')
    return dict(workers=workers,effective_affinity=affinity,worker_gib=worker_gib,host_gib=host_gib)

def chunk_valid(folder,identity,owners=None):
    p=folder/'stats.json'
    if not p.exists():return False
    r=read(p)
    if not r.get('passed') or r.get('campaign_identity')!=identity:return False
    if owners is not None and r.get('owners')!=list(owners):raise ValueError('checkpoint owner mismatch')
    if set(r.get('files',{}))!={'bank.npz','geometry.npz'}:raise ValueError('checkpoint manifest')
    for rel,h in r['files'].items():
        if sha(folder/rel)!=h:raise ValueError('corrupt checkpoint '+str(folder/rel))
    return True

_INIT_FAILURE=None

def worker_init(n,root,inp,run,identity,worker_gib):
    global _INIT_FAILURE
    try:
        if hasattr(os,'sched_getaffinity'):
            allowed=sorted(os.sched_getaffinity(0));slot=mp.current_process()._identity[-1]-1
            os.sched_setaffinity(0,{allowed[slot%len(allowed)]})
        configure(output=run)
        from scripts.q08_extraction_global import compute
        compute.initialize(n,root,inp,run)
        compute.ENV['campaign_identity']=identity
        compute.ENV['worker_gib']=worker_gib
    except Exception as exc:_INIT_FAILURE=repr(exc)

def worker_chunk(task):
    if _INIT_FAILURE:raise RuntimeError('worker initialization: '+_INIT_FAILURE)
    from scripts.q08_extraction_global import compute
    index,owners=task;r=compute.chunk(index,owners)
    if rss()>compute.ENV['worker_gib']:raise RuntimeError(f'worker RSS {rss():.3f} GiB exceeds declared limit')
    folder=Path(compute.ENV['rundir'])/'cpu'/f'N{compute.ENV["n"]}'/f'chunk_{index:06d}'
    r.update(campaign_identity=compute.ENV['campaign_identity'],peak_rss_gib=rss(),cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)
    write(folder/'stats.json',r)
    return r

def setup_data(run,root,n,identity):
    from scripts.q08_extraction_global import compute
    compute.initialize(n,root,run/'inputs',run)
    files={name:sha(run/'data'/f'N{n}'/name) for name in ('state.npy','phi.npy','raw_to_owner.npy')}
    result=dict(passed=True,campaign_identity=identity,n=n,files=files)
    write(run/f'data_N{n}.json',result);return result

def preflight(run,root,identity,worker_gib):
    from scripts.q08_extraction_global import compute
    rec=[]
    for selection in read(run/'inputs/bounded/selection.json'):
        n=selection['n'];sub=run/'preflight_work';sub.mkdir(exist_ok=True)
        # Dedicated folder; bounded rows cannot satisfy full-grid receipts.
        compute.initialize(n,root,run/'inputs',sub)
        compute.ENV['campaign_identity']=identity
        result=compute.chunk(0,selection['owners'])
        if rss()>worker_gib:raise RuntimeError('preflight exceeds worker memory budget')
        # Independent previously accepted actions, not only two implementations
        # recomputing the same field catalogue. compute provides this adapter.
        if hasattr(compute,'bounded_replay'):
            result['persisted_replay']=compute.bounded_replay(run/'inputs/bounded',n)
        else:raise RuntimeError('independent bounded replay adapter required')
        rec.append(result)
    out=dict(passed=True,campaign_identity=identity,grids=rec,peak_rss_gib=rss())
    write(run/'preflight.json',out);return out

def run_cpu(run,root,n,identity,workers,worker_gib,host_gib,pilot=False):
    resources=check_resources(run,workers,worker_gib,host_gib)
    require(run,f'data_N{n}',identity)
    for rel,h in read(run/f'data_N{n}.json')['files'].items():
        if sha(run/'data'/f'N{n}'/rel)!=h:raise ValueError('owner data changed')
    plan=read(run/'inputs/plan.json')[str(n)]['chunks'];todo=list(enumerate(plan))
    if pilot:
        # Fixed geometry sampling includes core, intermediate bulk and wall.
        ids=sorted(set([0,len(plan)//3,2*len(plan)//3,len(plan)-1]));todo=[todo[i] for i in ids]
    todo=[(i,o) for i,o in todo if not chunk_valid(run/'cpu'/f'N{n}'/f'chunk_{i:06d}',identity,o)]
    start=time.perf_counter();records=[]
    context=mp.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=context,max_tasks_per_child=24,
             initializer=worker_init,initargs=(n,str(root),str(run/'inputs'),str(run),identity,worker_gib)) as pool:
        # Bound submitted futures so a large plan does not create a large
        # parent queue or let completed arrays accumulate in memory.
        it=iter(todo);pending={}
        def submit():
            try:task=next(it)
            except StopIteration:return False
            pending[pool.submit(worker_chunk,task)]=task[0];return True
        for _ in range(min(len(todo),2*workers)):submit()
        while pending:
            done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for f in done:
                index=pending.pop(f);r=f.result();records.append(r);submit()
                progress=dict(stage='pilot' if pilot else 'cpu',n=n,completed_this_invocation=len(records),remaining_this_invocation=len(todo)-len(records),elapsed_seconds=time.perf_counter()-start,last_chunk=index)
                write(run/'progress.json',progress);print(json.dumps(progress),flush=True)
                check_resources(run,workers,worker_gib,host_gib)
    if pilot:
        ids=sorted(set([0,len(plan)//3,2*len(plan)//3,len(plan)-1]));records=[read(run/'cpu'/f'N{n}'/f'chunk_{i:06d}'/'stats.json') for i in ids]
        result=dict(passed=True,campaign_identity=identity,n=n,resources=resources,records=records,elapsed_seconds=time.perf_counter()-start,
                    extrapolation='Representative preparation/replay CPU work; GPU measured separately. No runtime guarantee.')
        write(run/f'pilot_N{n}.json',result);return result
    return validate_cpu(run,n,identity)

def validate_cpu(run,n,identity):
    import numpy as np
    plan=read(run/'inputs/plan.json')[str(n)];owners=[];raw_ids=[];max_scaled=0.;files={};total_bytes=0
    data=require(run,f'data_N{n}',identity)
    for rel,h in data['files'].items():
        if sha(run/'data'/f'N{n}'/rel)!=h:raise ValueError('owner data changed')
    ro=np.load(run/'data'/f'N{n}'/'raw_to_owner.npy',allow_pickle=False)
    order=np.argsort(ro,kind='stable');starts=np.r_[0,np.cumsum(np.bincount(ro))]
    components={'centered','correction','diffusion','combined'}
    leaves=components|{'raw_current.'+name for name in ('divergence_homogeneous','divergence_lift','divergence_physical','vorticity_homogeneous','vorticity_lift','vorticity_current','electron_phi','electron_ti_compensation','electron_generalized_force','inputs_valid')}|{'raw_electron_material','inputs_valid','eigensystem_admissible'}
    for i,oo in enumerate(plan['chunks']):
        folder=run/'cpu'/f'N{n}'/f'chunk_{i:06d}'
        if not chunk_valid(folder,identity,oo):raise ValueError('missing or invalid chunk '+str(folder))
        r=read(folder/'stats.json');owners+=oo
        raw=np.concatenate([order[starts[o]:starts[o+1]] for o in oo]);raw_ids.extend(r['raw'])
        if not np.array_equal(r['raw'],raw):raise ValueError('raw owner membership/order')
        records=r['components'];keys=[(m['span'],m['kind']) for m in records]
        if len(keys)!=8 or set(keys)!={(span,k) for span in (1/16,1/32) for k in range(4)}:raise ValueError('CPU case coverage')
        checked_max=0.
        for record in records:
            if set(record['leaves'])!=leaves:raise ValueError('CPU leaf coverage')
            for name,metric in record['leaves'].items():
                expected=[22,len(oo),6] if name in components else [22,len(raw)]
                if metric['shape']!=expected:raise ValueError('CPU state/leaf shape coverage')
                boolean=name.endswith('inputs_valid') or name=='eigensystem_admissible'
                if metric['dtype']!=('bool' if boolean else 'float64'):raise ValueError('CPU leaf precision')
                scaled=metric['max_scaled_error'];absolute=metric['max_abs_error']
                if not np.isfinite([scaled,absolute]).all() or not 0<=scaled<=1 or absolute<0:raise ValueError('CPU nonfinite/failed replay')
                checked_max=max(checked_max,scaled)
        if not np.isfinite(r['max_scaled']) or r['max_scaled']!=checked_max or r['all_arrays_bitwise'] is not True:raise ValueError('replay summary gate')
        max_scaled=max(max_scaled,checked_max)
        files[str(i)]=sha(folder/'stats.json');total_bytes+=sum((folder/f).stat().st_size for f in r['files'])
    if not np.array_equal(np.sort(owners),np.arange(plan['n_owner'])):raise ValueError('complete owner coverage')
    if not np.array_equal(np.sort(raw_ids),np.arange(plan['n_raw'])):raise ValueError('complete raw coverage')
    result=dict(passed=True,campaign_identity=identity,n=n,owners=len(owners),raw=plan['n_raw'],chunks=len(files),max_scaled=max_scaled,bytes=total_bytes,chunk_receipts=files)
    write(run/f'cpu_N{n}.json',result);return result

def analyze(run,identity):
    cpu=[require(run,f'cpu_N{n}',identity) for n in NS]
    gpu=[require(run,f'gpu_N{n}',identity) for n in NS]
    result=dict(passed=True,campaign_identity=identity,cpu=cpu,gpu=gpu,scientific_qualification=False,production_promoted=False)
    write(run/'analysis.json',result)
    lines=['# Q08 extraction/global GPU replay','',f'Campaign identity: `{identity}`.','',
           'Implementation replay only. No new MMS convergence gate, production promotion or span selection.','',
           '| N | owners | CPU chunks | largest CPU scaled discrepancy | compact files (GiB) |','|---|---:|---:|---:|---:|']
    for r in cpu:lines.append(f'| {r["n"]} | {r["owners"]} | {r["chunks"]} | {r["max_scaled"]:.6g} | {r["bytes"]/2**30:.4f} |')
    lines+=['','Full GPU correctness/performance records are in `gpu_N*.json` and `gpu/`. Timings separate compilation, transfers and synchronized warm calls.','Both one- and four-device paths must pass before completion. GPU/CPU float64 differences use the frozen mixed absolute/relative replay budget.','Full Q08 physical/evolution gates remain outside this campaign.']
    (run/'report.md').write_text('\n'.join(lines)+'\n');return result

def completion(run,identity):
    require(run,'preflight',identity)
    for n in NS:
        validate_cpu(run,n,identity);require(run,f'gpu_N{n}',identity)
    r=read(run/'analysis.json')
    if r['campaign_identity']!=identity or not r['passed']:raise ValueError('analysis identity')
    # GPU module independently validates all expected per-case receipts.
    from scripts.q08_extraction_global.gpu import validate_resolution
    for n in NS:validate_resolution(run,n,identity)
    files={f:sha(run/f) for f in ['analysis.json','report.md',*[f'cpu_N{n}.json' for n in NS],*[f'gpu_N{n}.json' for n in NS]]}
    result=dict(passed=True,campaign_identity=identity,files=files,scientific_qualification=False)
    write(run/'completion.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','verify','preflight','data','pilot','cpu','validate-cpu','gpu','analyze','validate-completion','status'))
    p.add_argument('--run',type=Path,default=os.environ.get('Q08_RUN'));p.add_argument('--input-root',type=Path,default=os.environ.get('Q08_INPUT_ROOT'))
    p.add_argument('--n',type=int,choices=NS);p.add_argument('--workers',type=int,default=1);p.add_argument('--worker-gib',type=float,default=6);p.add_argument('--host-gib',type=float,default=12)
    a=p.parse_args()
    if a.stage=='freeze':freeze();return
    if a.run is None: p.error('--run or Q08_RUN required')
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(run).free<3*1024**3:raise RuntimeError('free disk below 3 GiB')
    if a.stage=='status':print(json.dumps(read(run/'completion.json') if (run/'completion.json').exists() else read(run/'progress.json')));return
    configure(gpu=a.stage=='gpu',output=run)
    d=design();identity=digest(d)
    lock=(run/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=a.input_root.resolve() if a.input_root else None
    if a.stage not in ('verify',):
        v=require(run,'verification',identity)
        if root is None:root=Path(v['canonical_root'])
        if root!=Path(v['canonical_root']):raise ValueError('canonical root changed; use a separate RUN')
        check_inputs(run,root)
    if a.stage in ('data','pilot','cpu','validate-cpu','gpu') and a.n is None:p.error('--n required')
    start=time.perf_counter()
    if a.stage=='verify':
        if root is None:p.error('--input-root required for verify')
        result=verify(run,root,d)
    elif a.stage=='preflight':result=preflight(run,root,identity,a.worker_gib)
    elif a.stage=='data':require(run,'preflight',identity);result=setup_data(run,root,a.n,identity)
    elif a.stage in ('pilot','cpu'):
        require(run,'preflight',identity)
        if a.stage=='cpu':require(run,f'pilot_N{a.n}',identity)
        result=run_cpu(run,root,a.n,identity,a.workers,a.worker_gib,a.host_gib,a.stage=='pilot')
    elif a.stage=='validate-cpu':result=validate_cpu(run,a.n,identity)
    elif a.stage=='gpu':
        validate_cpu(run,a.n,identity)
        from scripts.q08_extraction_global.gpu import run_resolution
        result=run_resolution(run,a.n,identity,host_memory_gib=a.host_gib)
    elif a.stage=='analyze':result=analyze(run,identity)
    else:result=completion(run,identity)
    write(run/'provenance'/f'{a.stage}_{a.n or "all"}.json',dict(campaign_identity=identity,argv=sys.argv,elapsed_seconds=time.perf_counter()-start,peak_rss_gib=rss(),passed=True))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
