"""Detached-friendly fail-fast stages with chunk/case resume; no polling loop."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from campaign import HERE, check_source, bind, write


def main():
    p = argparse.ArgumentParser()
    for name in ('run','baseline-run','baseline-source','implementation-run'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, required=True, help='CPU reference workers; never candidate RHS workers')
    p.add_argument('--host-gib', type=float, required=True)
    p.add_argument('--through', choices=('pilot','N32','N48','complete'), default='complete')
    a = p.parse_args(); identity, _ = check_source()
    run, old, source, implementation = bind(a, identity)
    with (run/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common = ['--run',str(run),'--baseline-run',str(old),'--baseline-source',str(source),
                  '--implementation-run',str(implementation)]
        command = [sys.executable,str(HERE/'campaign.py')]
        stages = [('tests',[sys.executable,str(HERE/'verification/test_campaign.py'),
                           '--baseline-source',str(source),'--run',str(run)]),
                  ('verify',command+['verify',*common]),
                  ('preflight',command+['preflight',*common]),
                  ('preflight_gpu',command+['preflight-gpu',*common]),
                  ('pilot',command+['pilot',*common,'--workers',str(a.workers),'--host-gib',str(a.host_gib)])]
        if a.through != 'pilot':
            for n in (32,48,64):
                stages += [(f'references_N{n}',command+['references',*common,'--n',str(n),
                    '--workers',str(a.workers),'--host-gib',str(a.host_gib)]),
                    (f'gpu_N{n}',command+['gpu',*common,'--n',str(n),'--host-gib',str(a.host_gib)])]
                if a.through == f'N{n}': break
        if a.through == 'complete':
            stages += [('analyze',command+['analyze',*common]),
                       ('validate_completion',command+['validate-completion',*common])]
        env = os.environ.copy()
        env.update(TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        # Each subprocess independently selects its correct backend. Leave
        # scheduler CUDA_VISIBLE_DEVICES intact for the GPU children.
        history=[]
        for name, cmd in stages:
            tick=time.perf_counter(); log=run/'logs'/f'{name}_{time.time_ns()}.log'
            write(run/'status.json',dict(identity=identity,stage=name,status='running',log=str(log),command=cmd))
            with log.open('w') as stream:
                result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
            record=dict(stage=name,command=cmd,exit_code=result.returncode,seconds=time.perf_counter()-tick,log=str(log))
            history.append(record);print(record,flush=True)
            write(run/'controller_receipt.json',dict(identity=identity,history=history))
            if result.returncode:
                write(run/'status.json',dict(identity=identity,status='failed',**record))
                raise SystemExit(result.returncode)
        write(run/'status.json',dict(identity=identity,status='completed',through=a.through))


if __name__ == '__main__':
    main()
