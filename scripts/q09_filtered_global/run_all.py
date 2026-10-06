"""Sequential fail-fast/resumable controller; numerical CPU stages are parallel."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from scripts.q09_filtered_global import common as c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--canonical-root',type=Path,required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--host-gib',type=float,required=True)
    p.add_argument('--worker-gib',type=float,default=4.);p.add_argument('--through',choices=('pilot','N32','N48','all'),default='all')
    a=p.parse_args();run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);ident=c.identity()
    with (run/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (run/'logs').mkdir(exist_ok=True);(run/'provenance').mkdir(exist_ok=True)
        c.write(run/'provenance/manifest.json',c.read(c.HERE/'manifest.json'))
        stages=[('verify',None),('tests',None),('preflight-cpu',None),('preflight-gpu',None)]
        for n in c.GRIDS:
            stages += [('trace',n),('pilot',n)]
            if a.through=='pilot':break
            stages += [('prepare',n),('evolve',n)]
            if a.through==f'N{n}':break
        if a.through=='all':stages.append(('validate',None))
        env=dict(os.environ)
        for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
        env['PYTHONUNBUFFERED']='1'
        for stage,n in stages:
            if shutil.disk_usage(run).free<5*2**30:raise OSError('less than 5 GiB free: preserve checkpoints and free space')
            key=stage+(f'_N{n}' if n else '');receipt=run/'provenance'/f'{key}.json'
            command=[sys.executable,'-m','scripts.q09_filtered_global.campaign',stage,'--run',str(run),'--canonical-root',str(a.canonical_root.resolve()),'--workers',str(a.workers),'--host-gib',str(a.host_gib),'--worker-gib',str(a.worker_gib)]
            if n:command+=['--n',str(n)]
            # Source/input verification is cheap; always repeat. Science stages verify/resume their own chunks.
            if stage in ('tests','preflight-cpu','preflight-gpu') and receipt.exists():
                r=c.read(receipt)
                if r.get('identity')!=ident:raise ValueError('controller identity changed')
                if r.get('exit')==0:
                    gate=stage.replace('-','_');c.require(run,gate,ident);continue
            tick=time.time();log=run/'logs'/f'{key}.log'
            print(f'{key}: {" ".join(command)}',flush=True)
            with log.open('ab') as stream:
                stream.write(('\nRESUME '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n').encode())
                result=subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
            c.write(receipt,dict(identity=ident,command=command,start_unix=tick,end_unix=time.time(),seconds=time.time()-tick,exit=result.returncode,log=str(log.relative_to(run))))
            if result.returncode:raise RuntimeError(f'{key} failed; see {log}')
        print('Completed requested stages; deterministic receipts saved. No scientific qualification implied.',flush=True)

if __name__=='__main__':main()
