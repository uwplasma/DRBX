"""Checked concurrent stages; controllers never import JAX."""
from pathlib import Path
import time
from . import common as c
from . import scheduler

def job(action,operator=None,bc=None,seed=None):return dict(action=action,operator=operator,bc=bc,seed=seed)
def child(args,action,operator=None,bc=None,seed=None):
    return scheduler.run(args,[job(action,operator,bc,seed)],deadline=getattr(args,'deadline',None))
def source(args):
    record=child(args,'source')
    return c.receipt(args,'receipts/source.json',dict(passed=True,frozen_sources_checked=True,eigcore_unchanged=True,scheduling=record),[args.run/'results/source.json'])
def preflight(args):
    c.checked(args.run,'receipts/source.json',args.identity) or (_ for _ in ()).throw(ValueError('source required'))
    record=scheduler.run(args,[job('verify',op,bc) for op,bc in c.CASES]+[job('verify-restarts')],deadline=args.deadline)
    cases=[c.read(args.run/f'results/verify_{op}_{bc}.json') for op,bc in c.CASES]
    control=c.read(args.run/'results/verify-restarts.json');passed=all(r['passed'] for r in cases) and control['passed']
    c.write(args.run/'results/verification.json',dict(passed=passed,scope='bounded artificial closure; tooling only',cases=cases,restarts=control))
    if not passed:raise ValueError('bounded verification failed')
    return c.receipt(args,'receipts/preflight.json',dict(passed=True,backend='cpu',scheduling=record),[args.run/'results/verification.json',*sorted((args.run/'results').glob('verification_modes_*'))])
def partial_summary(args,op,bc,reason):
    folder=args.run/'checkpoints'/f'{op}_{bc}';folder.mkdir(parents=True,exist_ok=True)
    analysis=c.checked(args.run,folder/'analyses.json',args.identity)
    if analysis is None:analysis=c.receipt(args,folder/'analyses.json',dict(abscissa=dict(converged=False,value=None,residual=None,reason=reason),energy_probes=[],metadata=None))
    rows=[];missing=[];unconverged=[]
    for seed in args.seeds:
        saved=c.checked(args.run,folder/f'seed_{seed}.json',args.identity)
        if saved is None:missing.append(seed);unconverged.append(seed);continue
        if not saved['converged']:unconverged.append(seed)
        rows.extend(dict(p,seed=seed,mode_index=i,vectors_file=saved['vectors_file']) for i,p in enumerate(saved['pairs']))
    rows.sort(key=lambda p:-p['lam'][0]);verified=[p for p in rows if p['accepted']]
    c.write(folder/'summary.json',dict(converged=False,unconverged_seeds=unconverged,missing_seeds=missing,
        numerical_abscissa=analysis['abscissa'],top_candidates=rows[:args.k],top_verified_modes=verified[:args.k],
        reason=reason,limitation='partial budget summary; duplicates across seeds may remain; no exhaustive spectral conclusion'))
    return missing,unconverged

def solve(args):
    tick=time.perf_counter();args.search_deadline=args.deadline-.2*args.stage_seconds
    done={case:set() for case in c.CASES};analyses_queued=set()
    def finished(spec,records):
        if spec['action']!='spectrum':return []
        case=(spec['operator'],spec['bc']);done[case].add(spec['seed'])
        if done[case]==set(args.seeds) and case not in analyses_queued:
            analyses_queued.add(case);return [job('analyses',*case)]
        return []
    record=scheduler.run(args,[job('spectrum',op,bc,seed) for op,bc in c.CASES for seed in args.seeds],finished=finished,deadline=args.deadline)
    statuses=[];unconverged=[];missing=[]
    for op,bc in c.CASES:
        folder=args.run/'checkpoints'/f'{op}_{bc}';path=folder/'summary.json'
        ran=any(r['job']['action']=='analyses' and (r['job']['operator'],r['job']['bc'])==(op,bc) and r['exit_code']==0 for r in record['jobs'])
        if not ran:partial_summary(args,op,bc,'stage budget exhausted before analyses completed')
        summary=c.read(path);statuses.append(summary['converged'])
        for seed in args.seeds:
            saved=c.checked(args.run,folder/f'seed_{seed}.json',args.identity)
            if saved is None:missing.append([op,bc,seed])
            if saved is None or not saved['converged']:unconverged.append([op,bc,seed])
    files=sorted((args.run/'checkpoints').glob('*/*.json'))+sorted((args.run/'checkpoints').glob('*/*.npz'))
    return c.receipt(args,'receipts/solve.json',dict(passed=True,complete=all(statuses),seconds=time.perf_counter()-tick,
        unconverged_seeds=unconverged,missing_seeds=missing,scheduling=record,max_attempts=args.max_attempts,
        qualification='budget-limited partial results continue to derivative comparison and validation'),files)

def stage(args):
    if args.stage=='prepare':
        from .prepare import selection
        selection(args)
    full=args.stage in {'pilot','solve','compare-derivatives','validate'}
    if full and not args.remote_ok and not args.bounded_test:raise ValueError(f'{args.stage} requires --remote-ok')
    c.bind(args);args.deadline=time.monotonic()+args.stage_seconds
    if args.stage!='prepare':
        path=args.run/f'receipts/{args.stage}.json';old=c.read(path) if path.exists() else None
        if old is not None and old['identity']!=args.identity:raise ValueError('stage receipt identity mismatch')
        if old is not None and args.stage=='validate':
            current={str(p.relative_to(args.run)):c.sha(p) for p in sorted((args.run/'checkpoints').glob('*/*.json'))}
            if old.get('checkpoint_hashes')!=current:old=None
        if old is not None and old.get('complete',True):
            c.checked(args.run,path,args.identity);print(f'{args.stage}: checked resume',flush=True);return old
    if args.stage=='source':return source(args)
    if args.stage=='preflight':return preflight(args)
    c.checked(args.run,'receipts/preflight.json',args.identity) or (_ for _ in ()).throw(ValueError('preflight required'))
    if args.stage=='prepare':
        from .prepare import run
        return run(args)
    if not args.bounded_test:c.checked(args.run,'receipts/prepare.json',args.identity) or (_ for _ in ()).throw(ValueError('complete prepare required'))
    if args.stage=='solve':
        c.checked(args.run,'receipts/pilot.json',args.identity) or (_ for _ in ()).throw(ValueError('pilot required'))
        return solve(args)
    tick=time.perf_counter()
    specs=[job(args.stage,op,bc) for op,bc in c.CASES if args.stage!='compare-derivatives' or op=='complete']
    if args.stage=='compare-derivatives':
        records=[child(args,**spec) for spec in specs];record=dict(sequential=True,batches=records)
    else:record=scheduler.run(args,specs,deadline=args.deadline)
    if record.get('skipped'):raise TimeoutError('non-solve stage budget exhausted')
    files=[args.run/f'results/{args.stage}_{s["operator"]}_{s["bc"]}.json' for s in specs]
    result=c.receipt(args,f'receipts/{args.stage}.json',dict(passed=True,complete=True,seconds=time.perf_counter()-tick,scheduling=record),files)
    if args.stage=='validate':
        result['checkpoint_hashes']={str(p.relative_to(args.run)):c.sha(p) for p in sorted((args.run/'checkpoints').glob('*/*.json'))}
        result['all_modes_converged']=all(c.read(p).get('complete',False) for p in files)
        c.write(args.run/'receipts/validate.json',result)
    return result

def main():
    args=c.parser().parse_args();c.cpu_env()
    with c.lock(args.run,'stage'):stage(args)
if __name__=='__main__':main()
