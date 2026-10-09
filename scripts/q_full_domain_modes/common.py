"""CPU-only provenance, atomic checkpoints and process admission."""
from pathlib import Path
from contextlib import contextmanager
import argparse, fcntl, hashlib, json, os, resource, signal, sys, threading, time
HERE=Path(__file__).resolve().parent
STAGES=('source','preflight','prepare','pilot','solve','compare-derivatives','validate')
CASES=(('diffusion','D'),('diffusion','N'),('complete','D'),('complete','N'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n');q.replace(p)
def peak_gib(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)/2**30
def guard(seconds,gib):
    if seconds<=0 or gib<=0: raise ValueError('positive time and memory caps required')
    def alarm(*a): raise TimeoutError(f'process exceeded {seconds}s')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,seconds)
    def monitor():
        while True:
            if peak_gib()>gib:
                print(f'RSS {peak_gib():.6f} GiB exceeds {gib} GiB',file=sys.stderr,flush=True);os._exit(75)
            time.sleep(.05)
    threading.Thread(target=monitor,daemon=True).start()
def cpu_env():
    os.environ.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1',JAX_NUM_THREADS='1',
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[key]='1'
    sys.dont_write_bytecode=True
    return os.environ.copy()
def bootstrap(run,repo,core_index=None):
    cpu_env()
    if hasattr(os,'sched_setaffinity'):
        import multiprocessing
        allowed=sorted(os.sched_getaffinity(0));identity=multiprocessing.current_process()._identity
        index=core_index if core_index is not None else (((identity[0]-1) if identity else 0)%len(allowed))
        if not 0<=index<len(allowed):raise ValueError('core index outside inherited affinity')
        os.sched_setaffinity(0,{allowed[index]})
    sys.path.insert(0,str(repo))
    from scripts.q09_evolved_mms.bootstrap import configure
    configure(run,gpu=False)
    import jax
    if jax.default_backend()!='cpu' or not jax.config.jax_enable_x64: raise ValueError('CPU float64 required')
    import importlib
    root=(Path(repo)/'scripts/q09_evolved_mms/runtime/drbx').resolve()
    for name in ('drbx.native.q_plan','drbx.stencils.q_artifact'):
        if not Path(importlib.import_module(name).__file__).resolve().is_relative_to(root): raise ValueError('frozen imports escaped')
    return dict(backend=jax.default_backend(),devices=list(map(str,jax.devices())),x64=bool(jax.config.jax_enable_x64),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        core_index=core_index,affinity_supported=hasattr(os,'sched_setaffinity'),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        XLA_FLAGS=os.environ['XLA_FLAGS'],blas_threads=os.environ['OPENBLAS_NUM_THREADS'])
@contextmanager
def lock(run,name):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with (run/(name+'.lock')).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
        fcntl.flock(f,fcntl.LOCK_UN)
def parser(stages=True):
    p=argparse.ArgumentParser(description='Resumable CPU-only full-domain Q campaign')
    if stages:p.add_argument('stage',choices=STAGES)
    else:p.add_argument('--through',choices=STAGES,default='validate');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--run',type=Path,required=True);p.add_argument('--canonical-root',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,default=HERE.parents[1],help='checkout DRBX root; override only for package staging tests')
    p.add_argument('--remote-ok',action='store_true');p.add_argument('--workers',type=int,default=1)
    p.add_argument('--worker-gib',type=float,default=4);p.add_argument('--host-gib',type=float,default=16)
    p.add_argument('--stage-seconds',type=float,default=7200);p.add_argument('--krylov-seconds',type=float,default=1500)
    p.add_argument('--max-attempts',type=int,default=None,help='unlimited within stage budget by default')
    p.add_argument('--bounded-test',action='store_true',help='seven-owner fixture only; no full-domain operator')
    p.add_argument('--test-first-krylov-seconds',type=float,help='bounded-only first-attempt time budget control')
    p.add_argument('--test-kill-job',help='bounded-only once-per-run scheduler kill control, action_operator_bc_seed')
    p.add_argument('--n',type=int,default=32);p.add_argument('--chunks',type=int,nargs='+');p.add_argument('--max-chunks',type=int)
    p.add_argument('--interrupt-after-payload',type=int,help='bounded recovery test; once per run/chunk')
    p.add_argument('--seeds',type=int,nargs='+',default=[7,8]);p.add_argument('--k',type=int,default=6)
    p.add_argument('--ncv',type=int,default=48);p.add_argument('--maxiter',type=int,default=2000)
    p.add_argument('--residual-tol',type=float,default=1e-6);p.add_argument('--krylov-tol',type=float,default=1e-10)
    p.add_argument('--lanczos-tol',type=float,default=1e-9)
    return p
def bind(args,verify_canonical=True):
    args.run=args.run.resolve();args.repo_root=args.repo_root.resolve();args.canonical_root=args.canonical_root.resolve()
    if args.workers<1 or args.worker_gib<=0 or args.host_gib<args.worker_gib+1: raise ValueError('invalid host/worker budget')
    if getattr(args,'max_attempts',None) is not None and args.max_attempts<1:raise ValueError('max-attempts positive')
    if (getattr(args,'test_first_krylov_seconds',None) is not None or getattr(args,'test_kill_job',None)) and not getattr(args,'bounded_test',False):raise ValueError('test controls require bounded-test')
    if args.ncv<=args.k+1 or args.k<1: raise ValueError('ncv must exceed k+1')
    manifest=read(HERE/'input_manifest.json');sources=read(HERE/'source_manifest.json')
    for name,h in sources['package'].items():
        if sha(HERE/name)!=h:raise ValueError(f'package changed: {name}')
    for name,h in sources['dependencies'].items():
        if sha(args.repo_root/name)!=h:raise ValueError(f'frozen dependency changed: {name}')
    for name,h in manifest['files'].items():
        if sha(HERE/name)!=h:raise ValueError(f'input changed: {name}')
    canonical_receipt=args.run/'canonical_verified.json'
    canonical_record=dict(root=str(args.canonical_root),sha256=manifest['canonical'],input_manifest=sha(HERE/'input_manifest.json'))
    if verify_canonical:
        for name,h in manifest['canonical'].items():
            if sha(args.canonical_root/name)!=h:raise ValueError(f'canonical input changed: {name}')
        write(canonical_receipt,canonical_record)
    elif not canonical_receipt.exists() or read(canonical_receipt)!=canonical_record:
        raise ValueError('worker requires controller SHA verification of immutable canonical inputs')
    if sha(HERE/'eigcore.py')!=manifest['origins']['eigcore.py']['sha256']:raise ValueError('eigcore differs from original')
    config=dict(schema='q-full-campaign-v1',n=args.n,canonical_root=str(args.canonical_root),repo_root=str(args.repo_root),
        source_manifest=sha(HERE/'source_manifest.json'),input_manifest=sha(HERE/'input_manifest.json'),
        science=dict(bounded_tooling=getattr(args,'bounded_test',False),base_time=0,phi='fixed',boundary='homogeneous perturbations',source='excluded',derivative='Q09 frozen-projector JVP',H='owner volume'),
        solver=dict(k=args.k,ncv=args.ncv,seeds=args.seeds,maxiter=args.maxiter,residual_tol=args.residual_tol,krylov_tol=args.krylov_tol,lanczos_tol=args.lanczos_tol))
    ident=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['identity']=ident
    path=args.run/'run_config.json'
    if path.exists() and read(path)!=config:raise ValueError('run folder bound to different source/input/science/solver')
    if verify_canonical:
        write(path,config);write(args.run/'input_manifest.json',manifest);write(args.run/'source_manifest.json',sources)
    elif not path.exists():raise ValueError('worker requires controller run binding')
    args.identity=ident
    return config
def checked(run,path,identity):
    p=Path(run)/path
    if not p.exists():return None
    r=read(p)
    if r['identity']!=identity:raise ValueError(f'checkpoint identity mismatch: {p}')
    for name,h in r.get('files',{}).items():
        if sha(Path(run)/name)!=h:raise ValueError(f'checkpoint payload changed: {name}')
    return r
def receipt(args,path,value,files=()):
    value.update(identity=args.identity,files={str(Path(p).relative_to(args.run)):sha(p) for p in files})
    write(args.run/path,value);return value


def admission(args):
    reserve=max(1.,.05*args.host_gib)
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    effective=min(args.workers,affinity,int((args.host_gib-reserve)//args.worker_gib))
    if effective<1:raise MemoryError('host cannot admit worker plus reserve')
    return dict(requested_concurrency=args.workers,effective_concurrency=effective,affinity=affinity,reserve_gib=reserve,worker_gib=args.worker_gib,host_gib=args.host_gib)
