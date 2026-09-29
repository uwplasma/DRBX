"""Run the prescribed campaign stages with logs in one download directory."""
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
from . import storage as io

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True);p.add_argument('--input-root',required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--memory-gib',type=float,required=True)
    p.add_argument('--per-worker-gib',type=float,default=2.5);p.add_argument('--devices',nargs='+',type=int,required=True)
    a=p.parse_args();root=Path(a.campaign).resolve();root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';calls=[]
    base=[sys.executable,'-m','scripts.q_fci_layered_global.campaign'];paths=['--campaign',str(root),'--input-root',str(Path(a.input_root).resolve())]
    resources=['--workers',str(a.workers),'--memory-gib',str(a.memory_gib),'--per-worker-gib',str(a.per_worker_gib)]
    def run(label,args):
        cmd=base+args+paths;start=time.time();path=root/'logs'/f'{label}.log';print('Running',label,flush=True)
        with path.open('a') as out:code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT,env=env)
        calls.append(dict(label=label,argv=cmd,start_unix=start,elapsed_s=time.time()-start,exit_status=code,log=str(path)))
        io.write(root/'provenance/launch_receipt.json',dict(calls=calls,latest_run_complete=False))
        if code:raise SystemExit(f'{label} failed ({code}); see {path}')
    with (root/'driver.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('another campaign driver is running')
        run('input_verify',['init','--raw-chunk','512'])
        for n in (32,48,64):run(f'geometry_N{n}',['geometry','--N',str(n)]+resources)
        for stage in ('preflight','pilot','global'):
            for n in (32,48,64):
                opts=['--stage',stage,'--N',str(n)]
                run(f'{stage}_trace_N{n}',['trace',*opts,'--backend','gpu','--devices',*map(str,a.devices)])
                if stage=='preflight':run(f'{stage}_check_N{n}',['check','--N',str(n)])
                run(f'{stage}_score_N{n}',['score',*opts,*resources]);run(f'{stage}_validate_N{n}',['validate',*opts])
        run('reduce',['reduce'])
        io.write(root/'provenance/launch_receipt.json',dict(calls=calls,latest_run_complete=True))
if __name__=='__main__':main()
