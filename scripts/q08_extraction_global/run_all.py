"""Single-controller stage chaining with durable logs and exact exit receipts."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--worker-gib',type=float,required=True);p.add_argument('--host-gib',type=float,required=True)
    p.add_argument('--through',choices=('pilot','cpu','complete'),default='complete')
    a=p.parse_args();run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);logs=run/'logs';logs.mkdir(exist_ok=True)
    lock=(run/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,Q08_RUN=str(run),Q08_INPUT_ROOT=str(a.input_root.resolve()),TMPDIR=str(run/'tmp'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    base=[sys.executable,str(HERE/'campaign.py')]
    stages=[('verify',base+['verify']),('tests',[sys.executable,str(HERE/'verification/test_campaign.py')]),('preflight',base+['preflight','--worker-gib',str(a.worker_gib)])]
    resource_args=['--workers',str(a.workers),'--worker-gib',str(a.worker_gib),'--host-gib',str(a.host_gib)]
    for n in (32,48,64):
        stages += [(f'data_N{n}',base+['data','--n',str(n)]),(f'pilot_N{n}',base+['pilot','--n',str(n),*resource_args])]
        if a.through!='pilot':stages += [(f'cpu_N{n}',base+['cpu','--n',str(n),*resource_args])]
    if a.through=='complete':
        for n in (32,48,64):stages.append((f'gpu_N{n}',base+['gpu','--n',str(n),'--host-gib',str(a.host_gib)]))
        stages += [('analyze',base+['analyze']),('validate-completion',base+['validate-completion'])]
    from campaign import write
    for name,cmd in stages:
        attempt=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
        log=logs/f'{name}_{attempt}.log';start=time.perf_counter();stage_env=dict(env)
        if name.startswith('gpu_'):
            stage_env['JAX_PLATFORMS']='cuda,cpu';stage_env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        else:stage_env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',JAX_ENABLE_X64='true')
        print(json.dumps(dict(stage=name,command=cmd,log=str(log))),flush=True)
        with log.open('w') as stream:
            child=subprocess.run(cmd,cwd=HERE,env=stage_env,stdout=stream,stderr=subprocess.STDOUT)
        record=dict(stage=name,command=cmd,log=str(log.relative_to(run)),exit_code=child.returncode,elapsed_seconds=time.perf_counter()-start,python=sys.executable)
        write(run/'provenance'/f'exit_{name}_{attempt}.json',record)
        print(json.dumps(record),flush=True)
        if child.returncode:raise SystemExit(child.returncode)
    print(json.dumps(dict(completed_through=a.through,run=str(run))),flush=True)
if __name__=='__main__':main()
