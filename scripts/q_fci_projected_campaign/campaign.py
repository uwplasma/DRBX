#!/usr/bin/env python3
"""Node-local, checkpointed CPU qualification of the projected FCI candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('JAX_ENABLE_X64','true');os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
from scripts.q_fci_return_campaign import campaign as old
read,write,save,sha,digest,receipt,completed,rss=old.read,old.write,old.save,old.sha,old.digest,old.receipt,old.completed,old.rss
CHANNELS=('full_numerical','half_numerical','fourth_numerical','full_exact','half_exact','fourth_exact','projection','q5','q9','q11')
CTX=None;K=None


def config():return read(HERE/'configuration.json')


def source_identity():
    paths=list(HERE.glob('*.py'))+[HERE/'configuration.json']
    paths+=list((HERE.parent/'q_fci_return_campaign').glob('*.py'))
    paths+=[HERE.parent/'q_fci_return_campaign'/n for n in ('input_manifest.json','geometry_source.tar.gz','geometry_source_manifest.json')]
    paths+=[HERE.parent/'q03_direct_campaign/frozen_mms.py']
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(paths)}


def setup_paths(output):
    output.mkdir(parents=True,exist_ok=True)
    for d in ('tmp','logs','cache/jax','provenance'):(output/d).mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(output/'tmp');os.environ['DRBX_CACHE_DIR']=str(output/'cache/jax')
    os.environ['JAX_COMPILATION_CACHE_DIR']=str(output/'cache/jax')
    os.environ['XDG_CACHE_HOME']=str(output/'cache/xdg')
    # Prevent incidental Python output in the immutable input/source trees.
    sys.dont_write_bytecode=True


def verify(args):
    expected=read(HERE.parent/'q_fci_return_campaign/input_manifest.json')
    for record in expected['files']:
        p=args.input_root/record['path']
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
            raise RuntimeError(f'wrong immutable input: {p}')
    old.unpack(args.output)
    identity={'schema':'q-fci-projected-fourth-global-v1','source':source_identity(),'inputs':expected,'configuration':config()}
    p=args.output/'campaign.json'
    if p.exists() and read(p)!=identity:raise RuntimeError('incompatible campaign identity; use a fresh folder')
    write(p,identity);write(args.output/'input_locations.json',{'input_root':str(args.input_root)})
    import numpy,scipy
    write(args.output/'environment.json',{'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'platform':platform.platform()})
    return digest(identity)


def initialize(n,input_root,output):
    global CTX,K
    identity=multiprocessing.current_process()._identity
    if identity and hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[(identity[0]-1)%len(cpus)]})
    setup_paths(Path(output));sys.path.insert(0,str(Path(output)/'software/src'))
    from scripts.q_fci_projected_campaign import kernel
    K=kernel;CTX=K.q.context(n,input_root,config())
    from scripts.q_fci_projected_campaign.endpoints import CachedOwnerMoments
    CTX['model']=CachedOwnerMoments.from_model(CTX['model'])
    from drbx.geometry.hsx_jax_field import JaxHsxMagneticField
    CTX['rk_field']=JaxHsxMagneticField.from_evaluators(CTX['evaluator'],CTX['bfield'])


def do_job(job):
    import numpy as np
    stage,ids,path,identity=job;path=Path(path);t=time.monotonic();cpu=time.process_time()
    try:
        if stage.startswith('face'):
            order=7 if stage=='face_q7' else 5
            prepared=path.parent.parent/(stage+'_maps')/path.name
            if completed(prepared,identity,ids):
                with np.load(prepared) as z:data={k:z[k] for k in z.files}
            else:
                # Trace checkpoint precedes interpolation and survives later map/reference failures.
                trace_index=[0]
                def cached_trace(seeds,delta,steps):
                    tracepath=path.parent.parent/(stage+'_traces')/path.stem/f'{trace_index[0]:03d}.npz'
                    trace_index[0]+=1
                    if completed(tracepath,identity,ids):
                        with np.load(tracepath) as z:
                            if not np.array_equal(seeds,z['seeds']) or not np.array_equal(delta,z['delta']) or int(z['steps'])!=steps:
                                raise RuntimeError('trace seed/interval identity mismatch')
                            return [z[k] for k in ('endpoints','lengths','valid','bad','beta_min','beta_max')],int(z['executed'])
                    result,executed=K.trace_batch(CTX,seeds,delta,steps)
                    save(tracepath,seeds=seeds,delta=delta,steps=steps,executed=executed,**dict(zip(('endpoints','lengths','valid','bad','beta_min','beta_max'),result)))
                    receipt(tracepath,identity,ids);return result,executed
                data=K.prepare_faces(CTX,np.array(ids),order,trace_function=cached_trace,max_halvings=CTX['config']['policy']['max_wall_span_halvings'])
                save(prepared,**data);receipt(prepared,identity,ids)
            values=K.evaluate_faces(CTX,data)
            save(path,ids=ids,keys=data['keys'],**values,map_sha256=np.array(sha(prepared)),map_digest=data['map_digest'],trace_real_legs=data['trace_real_legs'],trace_executed_legs=data['trace_executed_legs'],span_level_counts=np.bincount(data['span_levels'],minlength=9),initial_invalid_legs=data['initial_invalid_legs'],seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_rss_gib=rss(),worker_pid=os.getpid())
        elif stage=='volume':
            ijk=np.array(np.unravel_index(ids,(CTX['N'],)*3)).T;v={}
            for order in (5,7):
                p,w=K.q.quadrature(CTX,ijk,order,face=False)
                J=np.abs(np.linalg.det(CTX['evaluator'].jacobian_matrix(p.reshape(-1,3))))
                v[f'q{order}']=np.sum(w*J.reshape(w.shape),axis=1)
            save(path,ids=ids,**v,seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_rss_gib=rss(),worker_pid=os.getpid())
        elif stage=='strong':
            owner=int(ids[0]);raw=np.flatnonzero(CTX['topology']['compact_raw_owner'].ravel()==owner)
            value,volume=K.q.strong_reference(CTX,raw,9)
            save(path,ids=ids,value=value,volume=volume,seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_rss_gib=rss(),worker_pid=os.getpid())
        else:raise ValueError(stage)
        receipt(path,identity,ids);return stage,str(path)
    except BaseException as exc:
        write(path.with_suffix('.failure.json'),{'identity':identity,'stage':stage,'unit':ids,'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-t,'cpu_seconds':time.process_time()-cpu,'peak_rss_gib':rss()});raise


def pool_jobs(args,n,jobs):
    if args.workers==1:
        for job in jobs:yield do_job(job)
        return
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(n,str(args.input_root),str(args.output))) as pool:
        pending=set();jobs=iter(jobs)
        for _ in range(2*args.workers):
            j=next(jobs,None)
            if j is not None:pending.add(pool.submit(do_job,j))
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for f in done:
                yield f.result()
                j=next(jobs,None)
                if j is not None:pending.add(pool.submit(do_job,j))


def select(ctx,scope):
    import numpy as np
    if scope=='global':return np.arange(len(ctx['volume'])),np.arange(len(ctx['keys'])),np.arange(ctx['N']**3),[]
    owners,faces,raw,tracks=old.bounded_selection(ctx)
    if scope=='smoke':
        # One complete ordinary owner. This cannot substitute for preflight.
        tracks=tracks[:1];owners=np.array([tracks[0]['owner']]);faces=np.flatnonzero(np.isin(ctx['lower'],owners)|np.isin(ctx['upper'],owners));raw=np.flatnonzero(np.isin(ctx['topology']['compact_raw_owner'].ravel(),owners))
    return owners,faces,raw,tracks


def make_plans(root,selection,cid,n,scope):
    policy=config()['policy'];spec=[('face',selection['face_ids'],policy['face_chunk'] if scope=='global' else policy['preflight_face_chunk']),('volume',selection['raw_ids'],policy['volume_chunk'] if scope=='global' else policy['preflight_volume_chunk'])]
    if scope!='global':
        spec.append(('face_q7',selection['face_ids'],policy['preflight_face_chunk']))
        spec.append(('strong',[t['owner'] for t in selection['tracks'] if t['name'] in ('ordinary','wall')],1))
    plans={}
    for name,ids,chunk in spec:
        ids=list(map(int,ids));units=[ids[i:i+chunk] for i in range(0,len(ids),chunk)]
        plan={'identity':digest({'campaign':cid,'N':n,'scope':scope,'stage':name,'units':units}),'units':units}
        path=root/(name+'_plan.json')
        if path.exists() and read(path)!=plan:raise RuntimeError('changed stage plan')
        write(path,plan);plans[name]=plan
    return plans


def pending_jobs(root,plans):
    for stage,plan in plans.items():
        for i,unit in enumerate(plan['units']):
            path=root/(stage+'_chunks')/f'{i:06d}.npz'
            if not completed(path,plan['identity'],unit):yield stage,unit,str(path),plan['identity']


def accumulate(ctx,ids,flux,out):
    import numpy as np
    for sign,side in ((1,ctx['lower']),(-1,ctx['upper'])):
        a=side[ids];ok=a>=0;np.add.at(out,a[ok],sign*flux[ok])


def reduction(root,ctx,selection,plans,cid,scope):
    import numpy as np
    owners=np.array(selection['owners']);faces=np.array(selection['face_ids']);v=ctx['volume'][owners];nc=len(ctx['volume'])
    sums={k:np.zeros((nc,4)) for k in CHANNELS};axis=np.zeros((3,nc,4));ext={k:np.zeros(4) for k in CHANNELS};timings=[];volume_timings=[];levels=np.zeros(9,dtype=int);initial_invalid=0
    # Deterministic canonical-unit reduction, independent of worker completion order.
    for i,unit in enumerate(plans['face']['units']):
        path=root/'face_chunks'/f'{i:06d}.npz'
        if not completed(path,plans['face']['identity'],unit):raise RuntimeError('incomplete face stage')
        with np.load(path) as z:
            ids=np.array(unit);assert np.array_equal(z['ids'],ids)
            prepared=root/'face_maps'/path.name
            if not completed(prepared,plans['face']['identity'],unit) or str(z['map_sha256'])!=sha(prepared):raise RuntimeError('map checkpoint mismatch')
            with np.load(prepared) as m:
                full=K.unpack_csr('full',m);half=K.unpack_csr('half',m);y=ctx['state'][m['donors']]
                for name,F in (('full',full),('half',half)):
                    np.testing.assert_allclose(F@y,z[name+'_numerical'],rtol=1e-10,atol=1e-13)
                np.testing.assert_allclose((4*(half@y)-full@y)/3,z['fourth_numerical'],rtol=1e-10,atol=1e-13)
            exterior=np.isin(ctx['lower'][ids],owners).astype(int)-np.isin(ctx['upper'][ids],owners).astype(int)
            for k in CHANNELS:
                f=z[k];assert np.all(np.isfinite(f));accumulate(ctx,ids,f,sums[k]);ext[k]+=exterior@f
            for a in range(3):
                mask=z['keys'][:,0]==a;accumulate(ctx,ids[mask],z['fourth_numerical'][mask],axis[a])
            timings.append({k:float(z[k]) for k in ('seconds','cpu_seconds','peak_rss_gib','trace_real_legs','trace_executed_legs')})
            levels+=z['span_level_counts'];initial_invalid+=int(z['initial_invalid_legs'])
    cont={k:np.zeros(nc) for k in ('q5','q7')};labels=ctx['topology']['compact_raw_owner'].ravel()
    for i,unit in enumerate(plans['volume']['units']):
        path=root/'volume_chunks'/f'{i:06d}.npz'
        if not completed(path,plans['volume']['identity'],unit):raise RuntimeError('incomplete volume stage')
        with np.load(path) as z:
            assert np.array_equal(z['ids'],unit)
            for k in cont:
                assert np.all(np.isfinite(z[k])) and np.all(z[k]>0);np.add.at(cont[k],labels[unit],z[k])
            volume_timings.append({k:float(z[k]) for k in ('seconds','cpu_seconds','peak_rss_gib')})
    for k in cont:
        if np.any(cont[k][owners]<=0):raise RuntimeError('missing continuous owner volume')
    actions={k:sums[k][owners]/v[:,None] for k in CHANNELS};ref=sums['q11'][owners]/cont['q7'][owners,None];low=sums['q9'][owners]/cont['q5'][owners,None]
    weighted=lambda e:np.sqrt(np.sum(v[:,None]*e*e,axis=0)/sum(v))
    errors=actions['fourth_numerical']-ref;norm=weighted(errors);budget=weighted(ref-low)
    balance={k:float(np.max(abs(np.sum(sums[k][owners],axis=0)-ext[k]))) for k in sums}
    constant=float(np.max(abs(actions['fourth_numerical'][:,-1])))
    if constant>1e-8 or max(balance.values())>1e-10:raise RuntimeError('constant/incidence check failed')
    q7action=None
    if scope!='global':
        total=np.zeros((nc,4))
        for i,unit in enumerate(plans['face_q7']['units']):
            path=root/'face_q7_chunks'/f'{i:06d}.npz'
            if not completed(path,plans['face_q7']['identity'],unit):raise RuntimeError('missing q7 construction')
            with np.load(path) as z:accumulate(ctx,np.array(unit),z['fourth_numerical'],total)
        q7action=total[owners]/v[:,None]
    arrays={'owners':owners,'face_ids':faces,'volume':v,'continuous_volume_q5':cont['q5'][owners],'continuous_volume_q7':cont['q7'][owners],'reference':ref,'reference_low':low,'fourth_axis_action':axis[:,owners]/v[None,:,None],**{k+'_action':a for k,a in actions.items()}}
    if q7action is not None:arrays['q7_fourth_numerical_action']=q7action
    save(root/'actions.npz',**arrays)
    radial=ctx['owner_ids'][owners]//ctx['N']**2;members=np.bincount(labels,minlength=nc)[owners];raw=ctx['owner_ids'][owners];ij=np.array(np.unravel_index(raw,(ctx['N'],)*3)).T;grid=ctx['artifact'].geometry.grid;coords=np.column_stack([getattr(grid,a).centers[ij[:,j]] for j,a in enumerate('xyz')])
    # Fixed logical regions supplement owner/topology strata without dropping any owner.
    masks={'axis':radial==0,'wall':radial==ctx['N']-1,'inward':radial==ctx['N']-2,'aggregate':members>1,'ordinary':members==1,'inner_half':coords[:,0]<.5,'outer_half':coords[:,0]>=.5,'eta_first_quadrant':coords[:,2]<np.pi/2}
    regions={}
    for name,mask in masks.items():
        regions[name]={'owners':int(sum(mask)),'volume':float(sum(v[mask])),'SSE':np.sum(v[mask,None]*errors[mask]**2,axis=0),'max_error':np.max(abs(errors[mask]),axis=0) if np.any(mask) else np.zeros(4)}
    summary={'identity':cid,'N':ctx['N'],'scope':scope,'owners':len(owners),'faces':len(faces),'fields':K.q.FIELDS,'operator_rms':norm,'max_error':np.max(abs(errors),axis=0),'reference_budget_rms':budget,'reference_fraction':budget[:3]/np.maximum(norm[:3],1e-300),'constant_max':constant,'balance':balance,'channel_rms':{k:weighted(a-ref) for k,a in actions.items()},'transfer_rms':weighted(actions['fourth_numerical']-actions['fourth_exact']),'projection_rms_same_q5':weighted(actions['projection']-actions['q5']),'finite_line_rms':weighted(actions['fourth_exact']-actions['projection']),'regions':regions,'face_timings':timings,'actions_sha256':sha(root/'actions.npz')}
    summary.update(volume_timings=volume_timings,span_level_counts=levels,initial_invalid_legs_resolved=initial_invalid)
    if q7action is not None:summary['complete_q7_candidate_change_rms']=weighted(q7action-actions['fourth_numerical'])
    if scope!='global':
        summary['strong_volume_controls']=[]
        for i,unit in enumerate(plans['strong']['units']):
            path=root/'strong_chunks'/f'{i:06d}.npz'
            if not completed(path,plans['strong']['identity'],unit):raise RuntimeError('missing strong reference')
            with np.load(path) as z:
                pos=int(np.flatnonzero(owners==unit[0])[0]);summary['strong_volume_controls'].append({'owner':unit[0],'reference':z['value'],'face_reference_minus_strong':ref[pos]-z['value'],'volume_q9':z['volume']})
    if scope!='global':
        nonwall=int(np.sum(~((ctx['keys'][:,0]==0)&(ctx['keys'][:,1]==ctx['N']))))
        summary['global_cost_projection']={'minimum_primary_scalar_legs':nonwall*100,'sample_primary_face_cpu_seconds':sum(x['cpu_seconds'] for x in timings),'sample_volume_cpu_seconds':sum(x['cpu_seconds'] for x in volume_timings),'sample_face_count':len(faces),'global_face_count':len(ctx['keys']),'estimated_face_CPU_seconds':sum(x['cpu_seconds'] for x in timings)*len(ctx['keys'])/len(faces),'estimated_volume_CPU_seconds':sum(x['cpu_seconds'] for x in volume_timings)*ctx['N']**3/len(selection['raw_ids']),'note':'CPU work projection only; sample includes JIT and atypical strata. Worker count, filesystem contention and memory determine elapsed time. No scaling run implied.'}
    write(root/'summary.json',summary);write(root/'completion.json',{'identity':cid,'summary_sha256':sha(root/'summary.json'),'complete':True})
    return summary


def resolution(args,n,cid,scope):
    initialize(n,str(args.input_root),str(args.output));ctx=CTX
    owners,faces,raw,tracks=select(ctx,scope);selection={'owners':owners.tolist(),'face_ids':faces.tolist(),'raw_ids':raw.tolist(),'tracks':tracks}
    root=args.output/scope/f'N{n}';root.mkdir(parents=True,exist_ok=True);p=root/'selection.json'
    if p.exists() and read(p)!=selection:raise RuntimeError('selection mismatch')
    write(p,selection);plans=make_plans(root,selection,cid,n,scope)
    total=sum(len(p['units']) for p in plans.values());done=total-sum(1 for _ in pending_jobs(root,plans));t=time.monotonic()
    for stage,path in pool_jobs(args,n,pending_jobs(root,plans)):
        done+=1;write(root/'progress.json',{'stage':stage,'completed':done,'total':total,'seconds':time.monotonic()-t,'effective_workers':args.workers})
    return reduction(root,ctx,selection,plans,cid,scope)


def validate(args,cid,scope,resolutions):
    import numpy as np
    records=[]
    for n in resolutions:
        root=args.output/scope/f'N{n}';completion=read(root/'completion.json')
        if completion['identity']!=cid or completion['summary_sha256']!=sha(root/'summary.json'):raise RuntimeError('completion mismatch')
        s=read(root/'summary.json');sel=read(root/'selection.json')
        stages=[('face','face_ids'),('volume','raw_ids')]
        if scope!='global':stages+=[('face_q7','face_ids'),('strong',None)]
        for stage,key in stages:
            plan=read(root/(stage+'_plan.json'))
            expected=sel[key] if key else [t['owner'] for t in sel['tracks'] if t['name'] in ('ordinary','wall')]
            if [v for u in plan['units'] for v in u]!=expected:raise RuntimeError('incomplete coverage')
            for i,unit in enumerate(plan['units']):
                path=root/(stage+'_chunks')/f'{i:06d}.npz'
                if not completed(path,plan['identity'],unit):raise RuntimeError('missing checkpoint')
                if stage.startswith('face'):
                    p=root/(stage+'_maps')/path.name
                    if not completed(p,plan['identity'],unit):raise RuntimeError('missing map')
                    with np.load(path) as z:
                        if str(z['map_sha256'])!=sha(p):raise RuntimeError('changed prepared map')
                    for trace in sorted((root/(stage+'_traces')/path.stem).glob('*.npz')):
                        if not completed(trace,plan['identity'],unit):raise RuntimeError('missing trace receipt')
        if sha(root/'actions.npz')!=s['actions_sha256']:raise RuntimeError('action hash mismatch')
        if s['constant_max']>1e-8 or max(s['balance'].values())>1e-10:raise RuntimeError('constant/incidence validation failed')
        with np.load(root/'actions.npz') as z:
            for k in z.files:
                if not np.all(np.isfinite(z[k])):raise RuntimeError(f'nonfinite {k}')
            rms=np.sqrt(np.sum(z['volume'][:,None]*(z['fourth_numerical_action']-z['reference'])**2,axis=0)/sum(z['volume']))
            np.testing.assert_allclose(rms,s['operator_rms'],rtol=3e-14,atol=1e-15)
        records.append(s)
    err=np.array([r['operator_rms'][:3] for r in records]);orders=np.log(err[:-1]/err[1:])/np.log(np.array(resolutions[1:])/np.array(resolutions[:-1]))[:,None] if len(records)>1 else np.empty((0,3))
    full=scope=='global' and list(resolutions)==[32,48,64]
    result={'computation_complete':True,'scope':scope,'resolutions_requested':resolutions,'errors':err,'orders':orders,'global_static_accuracy_passed':bool(full and np.all(orders>=1.8) and all(max(r['reference_fraction'])<.1 for r in records)),'reference_qualification':'empirical q9/q11 faces and q5/q7 volumes; bounded strong-volume controls','structural_evolved_production_certified':False,'records':records}
    write(args.output/(scope+'_validation.json'),result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify','smoke','preflight','run','validate','status'))
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resolutions',type=int,nargs='+',choices=(32,48,64),default=[32,48,64])
    p.add_argument('--workers',type=int,default=1);p.add_argument('--memory-budget-gib',type=float,default=0)
    p.add_argument('--worker-memory-gib',type=float,default=2.5);p.add_argument('--memory-reserve-gib',type=float,default=2.)
    args=p.parse_args();args.output=args.output.resolve();args.input_root=args.input_root.resolve();args.resolutions=sorted(set(args.resolutions));requested=args.workers
    if args.workers<1 or args.worker_memory_gib<=0 or args.memory_reserve_gib<0:raise ValueError('invalid worker/memory request')
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count();args.workers=min(args.workers,cpus)
    if args.memory_budget_gib:
        cap=int((args.memory_budget_gib-args.memory_reserve_gib)//args.worker_memory_gib)
        if cap<1:raise ValueError('insufficient memory allowance')
        args.workers=min(args.workers,cap)
    if args.command=='status':print(json.dumps(read(args.output/'status.json'),indent=2));return
    setup_paths(args.output)
    with old.lock(args.output/'.lock'):
        t=time.monotonic();status={'command':args.command,'requested_workers':requested,'effective_workers':args.workers,'memory_budget_gib':args.memory_budget_gib,'worker_memory_gib':args.worker_memory_gib,'memory_reserve_gib':args.memory_reserve_gib,'started_unix':time.time()}
        write(args.output/'status.json',dict(status,state='running'))
        try:
            cid=verify(args)
            if args.command in ('smoke','preflight'):
                for n in args.resolutions:resolution(args,n,cid,args.command)
                validate(args,cid,args.command,args.resolutions)
            elif args.command=='run':
                validate(args,cid,'preflight',[32,48,64])
                for n in args.resolutions:resolution(args,n,cid,'global')
                validate(args,cid,'global',args.resolutions)
            elif args.command=='validate':validate(args,cid,'global',args.resolutions)
            write(args.output/'status.json',dict(status,state='complete',seconds=time.monotonic()-t,completed_unix=time.time()))
        except BaseException as exc:
            write(args.output/'status.json',dict(status,state='failed',seconds=time.monotonic()-t,error=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
