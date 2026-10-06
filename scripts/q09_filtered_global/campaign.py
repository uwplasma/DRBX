"""Frozen filtered-field 24-run Q09 campaign stages. No production changes."""
from pathlib import Path
import argparse
import fcntl
import os
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from scripts.q09_filtered_global import common as c


def tests(run,ident):
    from scripts.q09_evolved_mms.campaign import fixtures
    fixtures()
    import pytest
    code=pytest.main(['-q','--noconftest',*[str(c.REPO/p) for p in c.TESTS]])
    if code: raise RuntimeError('portable filtered campaign tests failed')
    c.write(run/'tests.json',dict(passed=True,identity=ident))


def validate(run,ident):
    from scripts.q09_evolved_global.campaign import validate_grid
    from scripts.q09_evolved_global.reduce import reduce_all
    c.verified(run,ident);c.require(run,'tests',ident)
    names=['verification.json','tests.json','preflight_cpu.json','preflight_gpu.json','run_config.json']
    for gate in ('preflight_cpu','preflight_gpu'):
        c.require(run,gate,ident);names.extend(c.read(run/(gate+'.json')).get('files',{}))
    for n in c.GRIDS:
        for gate in (f'traces_N{n}',f'cpu_N{n}'):
            r=c.require(run,gate,ident);names.append(gate+'.json');names.extend(r['files'])
        plan=c.read(run/'inputs/plan.json')[str(n)]
        from .prepare import chunk_valid
        for i,owners in enumerate(plan['chunks']):
            path=run/'cpu'/f'N{n}'/f'chunk_{i:06d}';receipt=chunk_valid(path,ident,owners)
            if receipt is None: raise ValueError('missing filtered prepared chunk')
            names.extend(str((path/k).relative_to(run)) for k in receipt['files'])
        validate_grid(run,ident,n)
        names.extend(f'N{n}/'+k for k in ('completion.json','summary.json','time_history.json'))
        names.extend(f'N{n}/'+k for k in c.read(run/f'N{n}/completion.json')['files'])
    reduce_all(run,ident)
    analysis=c.read(run/'analysis.json');analysis['filtered_arm']=c.ARM;analysis['configuration']=c.configuration();c.write(run/'analysis.json',analysis)
    report=run/'report.md';report.write_text(report.read_text().replace('# Q09 full-domain','# Q09 filtered m<=3 full-domain',1)+'\nMagnetic arm: '+c.ARM+'. J B^i filtered at m<=3 per period; unchanged map/owner volumes. All traces, choices, magnetic coefficients and continuum forcing regenerated. Compare with the raw campaign locally.\n')
    names+=['analysis.json','orders.csv','report.md']
    c.write(run/'completion.json',dict(passed=True,identity=ident,filtered_arm=c.ARM,cases=24,accepted_steps=5800,snapshots=120,
        files={name:c.sha(run/name) for name in sorted(set(names))},scientific_qualification=False,production_promoted=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','source','verify','tests','preflight-cpu','preflight-gpu','trace','pilot','prepare','evolve','validate'))
    p.add_argument('--run',type=Path);p.add_argument('--canonical-root',type=Path)
    p.add_argument('--n',type=int,choices=c.GRIDS);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--worker-gib',type=float,default=4.);p.add_argument('--host-gib',type=float,default=200.)
    a=p.parse_args()
    if a.command=='freeze': print(c.freeze());return
    ident=c.identity()
    if a.command=='source':print(ident);return
    if a.run is None:p.error('--run required')
    if a.command=='verify' and a.canonical_root is None:p.error('--canonical-root required')
    if a.command in ('trace','pilot','prepare','evolve') and a.n is None:p.error('--n required')
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    # Spawn workers initialize their own CPU-only JAX; the controller does not.
    if a.command not in ('verify','pilot','prepare'):
        from scripts.q09_evolved_mms.bootstrap import configure
        configure(run,gpu=a.command in ('preflight-gpu','trace','evolve'))
    with (run/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tick=time.perf_counter()
        try:
            if a.command=='verify':c.verify(run,a.canonical_root,ident)
            else:
                c.verified(run,ident)
                if a.command=='tests':tests(run,ident)
                else:
                    c.require(run,'tests',ident)
                    if a.command.startswith('preflight-'):
                        from . import preflight
                        (preflight.cpu if a.command=='preflight-cpu' else preflight.gpu)(run,ident)
                    elif a.command=='trace':
                        c.require(run,'preflight_gpu',ident)
                        from .prepare import trace_grid
                        trace_grid(run,a.n,ident)
                    elif a.command in ('prepare','pilot'):
                        c.require(run,'preflight_gpu',ident)
                        from .prepare import run_cpu
                        run_cpu(run,a.n,ident,a.workers,a.worker_gib,a.host_gib,pilot=a.command=='pilot')
                    elif a.command=='evolve':
                        from .provider import configure_grid,load
                        from scripts.q09_evolved_mms.campaign import run_pilot
                        from scripts.q09_evolved_global.campaign import validate_grid
                        configure_grid(run,a.n,ident,a.host_gib)
                        run_pilot(run/f'N{a.n}',ident,dt0=c.END/c.STEPS[a.n],end=c.END,checkpoint_every=25,snapshots=True,levels=1,provider_loader=load)
                        validate_grid(run,ident,a.n)
                    elif a.command=='validate':validate(run,ident)
        except BaseException as exc:
            c.write(run/'status.json',dict(identity=ident,stage=a.command,n=a.n,status='failed',error=repr(exc),seconds=time.perf_counter()-tick));raise
        c.write(run/'status.json',dict(identity=ident,stage=a.command,n=a.n,status='complete',seconds=time.perf_counter()-tick))

if __name__=='__main__':main()
