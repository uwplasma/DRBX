"""Fresh CPU process per numerical job; all caches and payloads under RUN."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,time,os,signal
from . import common as c

def op_args(a,bounded=False):
    return SimpleNamespace(**dict(operator=a.operator,bc=a.bc,
        bank=c.HERE/'inputs/bounded/bank.npz' if bounded else a.run/'prepared/bank.npz',
        geometry=c.HERE/'inputs/bounded/geometry.npz' if bounded else a.run/'prepared/geometry.npz',
        topology=a.canonical_root/f'geometry_artifacts/rlp_convergence_32_48_64_20260917/{a.n}x{a.n}x{a.n}',
        owners=None,scope='tooling' if bounded else 'full',command='verify' if bounded else 'solve',dense_limit=128,
        base_time=0.,diffusion_coefficient=1.,last_aggregate=None,regional_plan=c.HERE/'inputs/plan.json',
        seeds=a.seeds,k=a.k,ncv=a.ncv,krylov_seconds=a.krylov_seconds,krylov_tol=a.krylov_tol,
        residual_tol=a.residual_tol,lanczos_tol=a.lanczos_tol,maxiter=a.maxiter,seed_json=None,seed_index=0,seed_vector=None))
def save_spectrum(o,op,path,result):
    np=o.np;vectors=np.array([np.array(p.pop('vector_real'))+1j*np.array(p.pop('vector_imag')) for p in result['pairs']])
    # rightmost may reference the same pair; store its index, not an extra buffer.
    result['rightmost_verified']=next((i for i,p in enumerate(result['pairs']) if p['accepted']),None)
    arrays=dict(vectors=vectors,lam=np.array([complex(*p['lam']) for p in result['pairs']]),owners=op.owners,H=op.H,coordinates=op.coordinates)
    with path.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
    path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
    c.write(path,result)
def rho(o,op,a):
    linear=o.sla.LinearOperator((op.size,)*2,matvec=op.matvec,dtype=o.np.float64)
    values,vectors=o.sla.eigs(linear,k=2,which='LM',ncv=min(a.ncv,op.size),tol=1e-3,maxiter=a.maxiter,
        v0=o.np.random.default_rng(7).normal(size=op.size),return_eigenvectors=True)
    residuals=[float(o.np.linalg.norm(op.matvec(vectors[:,i])-lam*vectors[:,i])/max(abs(lam)*o.np.linalg.norm(vectors[:,i]),1e-300)) for i,lam in enumerate(values)]
    return dict(value=float(max(abs(values))),original_L_relative_residuals=residuals,values=[[float(x.real),float(x.imag)] for x in values],qualified='iterative estimate, not upper bound')
def dispatch(a,o):
    np=o.np;action=a.action
    if action=='source':
        plan=c.read(c.HERE/'inputs/plan.json')[str(a.n)]
        with np.load(c.HERE/'inputs'/f'N{a.n}_traces.npz') as z:ends=z['ends']
        with np.load(c.HERE/'inputs'/f'choices_N{a.n}.npz') as z:choices=z['choice']
        owners=[v for chunk in plan['chunks'] for v in chunk]
        if owners!=list(range(plan['n_owner'])) or ends.shape!=(a.n**3,4,3) or not np.isfinite(ends).all() or choices.shape!=(a.n**3,):raise ValueError('retained inputs incomplete')
        revision=os.environ.get('DRBX_SOURCE_REVISION')
        return dict(passed=True,trace_shape=list(ends.shape),owners=plan['n_owner'],raw=plan['n_raw'],source_revision=revision)
    if action=='verify-restarts':return o.verify_restarts(a)
    if action in {'merge','merge-bounded'}:
        from .prepare import merge
        return merge(a,bounded=action=='merge-bounded')
    args=op_args(a,action=='verify' or getattr(a,'bounded_test',False));op=o.Operator(args)
    if action=='verify':
        result=o.verification(op,args)
        path=a.run/f'results/verification_modes_{a.operator}_{a.bc}.json'
        save_spectrum(o,op,path,result['spectrum'])
        with np.load(path.with_suffix('.npz')) as z:
            np.testing.assert_array_equal(z['owners'],op.owners);np.testing.assert_array_equal(z['H'],op.H)
            for v,lam in zip(z['vectors'],z['lam']):
                if np.linalg.norm(op.matvec(v)-lam*v)>a.residual_tol*max(abs(lam)*np.linalg.norm(v),1e-10):raise ValueError('NPZ roundtrip original-L residual failed')
        result['npz_roundtrip_passed']=True
        return result
    folder=a.run/'checkpoints'/f'{a.operator}_{a.bc}';folder.mkdir(parents=True,exist_ok=True)
    if action=='pilot':
        rng=np.random.default_rng(7);u=rng.normal(size=op.size);timing={}
        for name,fn in [('first_matvec',op.matvec),('warm_matvec',op.matvec),('first_vjp',op.rmatvec),('warm_vjp',op.rmatvec)]:
            tick=time.perf_counter();fn(u);timing[name]=time.perf_counter()-tick
        scale=rho(o,op,args)
        dt=.3/scale['value'];warm=timing['warm_matvec']
        growth=[dict(target_real_lambda=g,dt_times_real_lambda=dt*g,mu=float(1+dt*g+(dt*g)**2/2+(dt*g)**3/6+(dt*g)**4/24),mu_minus_one=float(dt*g+(dt*g)**2/2+(dt*g)**3/6+(dt*g)**4/24)) for g in (1.,10.)]
        stall=dict(diagnostic_only=True,rho=scale['value'],dt=dt,seconds_per_matvec=warm,k=a.k,ncv=a.ncv,
            first_sweep_seconds=4*min(a.ncv,op.size)*warm,restart_seconds=4*(min(a.ncv,op.size)-min(a.ncv,op.size)//2)*warm,
            target_separations=growth,assumption='RK4 real growth; matvec-only lower estimate, excludes orthogonalization and Schur costs')
        return dict(metadata=op.metadata(),timing=timing,spectral_radius=scale,stall_diagnostic=stall,energy_probe=op.energy(u))
    if action=='compare-derivatives':return o.compare_derivatives(op,args)
    if action=='spectrum':
        checkpoint=folder/f'seed_{a.seed}.json';old=c.checked(a.run,checkpoint,a.identity)
        if old is not None and old['converged']:return dict(resumed=True,checkpoint=str(checkpoint.relative_to(a.run)))
        pilot=c.read(a.run/f'results/pilot_{a.operator}_{a.bc}.json');scale=pilot['spectral_radius']['value']
        args.seeds=[a.seed];attempts=[];invocation_attempts=0
        search_deadline=getattr(a,'search_deadline',time.monotonic()+a.stage_seconds*.8)
        while old is None or not old['converged']:
            if getattr(a,'max_attempts',None) is not None and invocation_attempts>=a.max_attempts:break
            remaining=search_deadline-time.monotonic()
            warm=pilot['timing']['warm_matvec']
            reserve=max(.1,(4*min(a.ncv,op.size)+4*a.k+4)*warm)
            if remaining<=reserve:break
            attempt=0 if old is None else old['attempt']+1
            args.seed_vector=None
            if old is not None:
                with np.load(a.run/old['vectors_file']) as z:
                    vectors=z['vectors'];args.seed_vector=np.real(vectors[0])+np.imag(vectors[0])
            args.krylov_seconds=min(a.krylov_seconds,remaining-reserve)
            if invocation_attempts==0 and getattr(a,'test_first_krylov_seconds',None) is not None:
                args.krylov_seconds=min(args.krylov_seconds,a.test_first_krylov_seconds)
            result=o.spectrum(op,args,rho=scale)
            result.update(metadata=op.metadata(),attempt=attempt,
                converged=len(result['pairs'])>=a.k and all(p['accepted'] for p in result['pairs']) and all(r['converged_tol'] for r in result['krylov_runs']),
                resume_contract='warm Ritz-vector restart; original eigcore does not serialize its basis')
            archive=folder/f'seed_{a.seed}_attempt_{attempt}.json';save_spectrum(o,op,archive,result)
            result.update(identity=a.identity,vectors_file=str(archive.with_suffix('.npz').relative_to(a.run)),
                files={str(p.relative_to(a.run)):c.sha(p) for p in (archive,archive.with_suffix('.npz'))})
            c.write(checkpoint,result);old=result;invocation_attempts+=1
            attempts.append(dict(attempt=attempt,converged=result['converged'],krylov_seconds=args.krylov_seconds))
        return dict(converged=old is not None and old['converged'],attempts=attempts,
            budget_exhausted=time.monotonic()>=search_deadline,missing_checkpoint=old is None,
            max_attempts_reached=getattr(a,'max_attempts',None) is not None and invocation_attempts>=a.max_attempts)
    if action=='analyses':
        path=folder/'analyses.json';old=c.checked(a.run,path,a.identity)
        if old is None or not old['abscissa']['converged']:
            result=dict(abscissa=o.abscissa(op,args),energy_probes=op.energy_probes(a.seeds),metadata=op.metadata())
            files=[]
            if result['abscissa']['converged']:
                vector=np.asarray(result['abscissa'].pop('vector_y'));artifact=folder/'abscissa.npz'
                with artifact.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,vector_y=vector,owners=op.owners,H=op.H)
                artifact.with_suffix('.npz.tmp').replace(artifact);files.append(artifact)
                result['abscissa']['vectors_file']=str(artifact.relative_to(a.run))
            c.receipt(a,path,result,files)
        else:result=old
        seeds=[c.checked(a.run,folder/f'seed_{s}.json',a.identity) for s in a.seeds]
        missing=[seed for seed,saved in zip(a.seeds,seeds) if saved is None]
        unconverged=[seed for seed,saved in zip(a.seeds,seeds) if saved is None or not saved['converged']]
        candidates=[];kept=[]
        for seed,saved in zip(a.seeds,seeds):
            if saved is None:continue
            with np.load(a.run/saved['vectors_file']) as z:vectors=z['vectors']
            for i,(pair,v) in enumerate(zip(saved['pairs'],vectors)):
                lam=complex(*pair['lam'])
                duplicate=next((j for j,(p,x) in enumerate(zip(candidates,kept)) if
                    abs(complex(*p['lam'])-lam)<1e-7*max(abs(lam),1) and abs(np.vdot(x,v))/(np.linalg.norm(x)*np.linalg.norm(v))>1-1e-5),None)
                entry=dict(pair,seed=seed,mode_index=i,vectors_file=saved['vectors_file'])
                if duplicate is None:candidates.append(entry);kept.append(v)
                elif pair['absolute_residual']<candidates[duplicate]['absolute_residual']:candidates[duplicate]=entry;kept[duplicate]=v
        candidates.sort(key=lambda p:-p['lam'][0]);verified=[p for p in candidates if p['accepted']]
        maximum=verified[0]['lam'][0] if verified else None
        c.write(folder/'summary.json',dict(converged=not unconverged and result['abscissa']['converged'],missing_seeds=missing,unconverged_seeds=unconverged,
            numerical_abscissa=result['abscissa'],top_candidates=candidates[:a.k],top_verified_modes=verified[:a.k],
            modal_vs_transient=dict(rightmost_verified_real_part=maximum,numerical_abscissa=result['abscissa']['value'],
                positive_verified_candidate=maximum is not None and maximum>0),
            limitation='negative candidates cannot certify an exhaustive search; read original-L residuals and Lanczos residual'))
        return dict(passed=True)
    if action=='validate':
        records=[];missing=[];unconverged=[]
        for seed in a.seeds:
            r=c.checked(a.run,folder/f'seed_{seed}.json',a.identity)
            if r is None:missing.append(seed);unconverged.append(seed);continue
            if not r['converged']:unconverged.append(seed)
            if r['metadata']['bank_identity']!=op.bank.identity:raise ValueError('bank identity changed')
            with np.load(a.run/r['vectors_file']) as z:
                np.testing.assert_array_equal(z['owners'],op.owners);np.testing.assert_array_equal(z['H'],op.H)
                vectors=z['vectors'];values=z['lam']
            if len(vectors)!=len(r['pairs']):raise ValueError('mode/metadata count mismatch')
            for i,(v,lam) in enumerate(zip(vectors,values)):
                av=op.matvec(v);absolute=float(np.linalg.norm(av-lam*v)/np.linalg.norm(v))
                relative=absolute/abs(lam) if abs(lam)>r['rho_estimate']*1e-13 else None
                accepted=relative<=a.residual_tol if relative is not None else absolute<=a.residual_tol*r['rho_estimate']*1e-8
                if r['pairs'][i]['accepted'] and not accepted:raise ValueError('accepted original-L residual failed reload')
                records.append(dict(seed=seed,index=i,lambda_value=[float(lam.real),float(lam.imag)],relative_residual=relative,
                    absolute_residual=absolute,accepted=bool(accepted),energy_rate=op.energy(v),localization=op.localization(v)))
        analysis=c.checked(a.run,folder/'analyses.json',a.identity)
        if analysis is None:raise ValueError('missing abscissa/energy checkpoint')
        abc=analysis['abscissa'];verified_abc=dict(converged=False)
        if abc['converged']:
            with np.load(a.run/abc['vectors_file']) as z:
                np.testing.assert_array_equal(z['owners'],op.owners);np.testing.assert_array_equal(z['H'],op.H);y=z['vector_y']
            residual=float(np.linalg.norm(op.normalized_symmetric(y)-abc['value']*y)/np.linalg.norm(y))
            if abs(residual-abc['residual'])>1e-7*max(abs(abc['value']),1):raise ValueError('reloaded abscissa residual changed')
            verified_abc=dict(converged=True,value=abc['value'],original_symmetric_residual=residual,energy_rate=op.energy(y/op.sqrtH))
        return dict(passed=True,complete=not unconverged and abc['converged'],missing_seeds=missing,unconverged_seeds=unconverged,metadata=op.metadata(),modes=records,abscissa=verified_abc)
    raise ValueError(action)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);job=parser.parse_args().job
    a=SimpleNamespace(**c.read(job))
    for k in ('run','repo_root','canonical_root'):setattr(a,k,Path(getattr(a,k)))
    if a.action not in {'source','verify','verify-restarts','merge-bounded'} and not a.remote_ok and not getattr(a,'bounded_test',False):raise ValueError('numerical full-domain child requires --remote-ok')
    c.bind(a,verify_canonical=False);c.guard(a.stage_seconds,a.worker_gib)
    backend=dict(initialized=False,configured_platform='cpu',core_index=getattr(a,'core_index',None))
    tick=time.perf_counter()
    result_path=a.run/f'results/{a.action}_{a.operator}_{a.bc}.json' if a.operator else a.run/f'results/{a.action}.json'
    if a.action=='spectrum':result_path=a.run/f'results/spectrum_{a.operator}_{a.bc}_{a.seed}.json'
    c.write(a.run/'receipts'/f'job_ack_{job.stem}.json',dict(pid=os.getpid(),backend=backend,resumed=False))
    try:
        from . import operators as o
        backend=o.initialize(a.run/'caches'/job.stem,a.repo_root,core_index=getattr(a,'core_index',None))
        if a.action in {'source','verify','verify-restarts','pilot','compare-derivatives'}:
            old=c.checked(a.run,result_path,a.identity)
            if old is not None:
                c.write(a.run/'receipts'/f'job_ack_{job.stem}.json',dict(pid=os.getpid(),backend=backend,resumed=True));return
        c.write(a.run/'receipts'/f'job_ack_{job.stem}.json',dict(pid=os.getpid(),backend=backend,resumed=False))
        result=dispatch(a,o)
    except TimeoutError:
        if a.action not in {'spectrum','analyses'}:raise
        signal.setitimer(signal.ITIMER_REAL,0)
        if a.action=='analyses':
            from .campaign import partial_summary
            partial_summary(a,a.operator,a.bc,'stage time guard reached during analyses')
        result=dict(converged=False,budget_exhausted=True,reason='per-child time guard reached; committed checkpoints retained')
    except Exception as error:
        c.write(a.run/'receipts'/f'failure_{job.stem}.json',dict(error=repr(error),backend=backend,process_seconds=time.perf_counter()-tick,peak_rss_gib=c.peak_gib()))
        raise
    signal.setitimer(signal.ITIMER_REAL,0)
    result.update(identity=a.identity,backend=backend,process_seconds=time.perf_counter()-tick,peak_rss_gib=c.peak_gib())
    path=a.run/('prepared-bounded/merge.json' if a.action=='merge-bounded' else 'prepared/merge.json') if a.action in {'merge','merge-bounded'} else result_path
    if a.action=='verify':
        result['files']={str(p.relative_to(a.run)):c.sha(p) for p in (a.run/f'results/verification_modes_{a.operator}_{a.bc}.json',a.run/f'results/verification_modes_{a.operator}_{a.bc}.npz')}
    c.write(path,result)
if __name__=='__main__':main()
