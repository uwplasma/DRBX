#!/usr/bin/env python3
"""Resumable node-local CPU campaign for static P07N field-derived Neumann qualification."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true';os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,fcntl,hashlib,json,platform,resource,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,FIRST_COMPLETED,wait
from multiprocessing import get_context
from contextlib import contextmanager
sys.dont_write_bytecode=True
import numpy as np
import core
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
STAGES=('preflight-face','preflight-reference','preflight-q5','observations','faces','reference')
STATE={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=encode,allow_nan=False)+'\n');tmp.replace(path)
def save(path,**data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(path)
def rss():
    val=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val/(2**30 if sys.platform=='darwin' else 2**20)
@contextmanager
def lock(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'.campaign.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def config():
    cfg=json.loads((HERE/'configuration.json').read_text())
    if cfg['schema']!='drbx.p07n-field-derived-static-global-v1' or cfg['resolutions']!=[32,48,64] or cfg['face_quadrature']!=3 or cfg['reference']!='stored_physical_raw_volume_midpoint':
        raise ValueError('unsupported frozen P07N field-derived contract')
    if cfg['fields']!=list(core.NAMES):raise ValueError('field catalogue changed')
    return cfg

def source_hashes():
    rel=['scripts/p07n_field_derived_global/campaign.py','scripts/p07n_field_derived_global/core.py',
         'scripts/p07n_field_derived_global/fields.py','scripts/p07n_field_derived_global/configuration.json',
         'scripts/p07n_field_derived_global/README.md',
         'scripts/p07n_field_derived_global/input_manifest.json',
         'scripts/p07_combined_global/kernels.py','scripts/p07_combined_global/topology.py',
         'scripts/p07_diffusion_global/numerics.py','hsx_mms_continuum_reference.py',
         'src/drbx/geometry/fci_perpendicular_neumann_trace.py',
         'src/drbx/geometry/fci_perpendicular_integrated_rows.py',
         'src/drbx/geometry/_fci_perpendicular_point_primitives.py',
         'src/drbx/geometry/fci_perpendicular_reconstruction.py']
    return {p:sha(REPO/p) for p in rel}

def verify(input_root,output):
    input_root=Path(input_root);output=Path(output);cfg=config();im=json.loads((HERE/'input_manifest.json').read_text())
    for rec in im['files']:
        path=input_root/rec['path']
        if not path.is_file() or path.stat().st_size!=rec['bytes'] or sha(path)!=rec['sha256']:
            raise ValueError(f'missing or changed immutable input: {path}')
    sources=source_hashes();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    identity=digest({'configuration':cfg,'input_manifest':im,'sources':sources,'commit':commit})
    manifest_path=output/'campaign_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity']!=identity:
        raise ValueError('campaign identity changed; use a new output folder')
    side=json.loads((input_root/'DRBX/work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json').read_text())
    side['metric_cache']['path']=str((input_root/'hsx_metric_d58d392545fd3917efeb83b6.npz').resolve())
    side['makegrid']['path']=str((input_root/'mgrid_res2p5cm_180pln.nc').resolve())
    side['artifact']['path']=str((input_root/'prototype_runs/geometry/hsx_fci_64x64x64').resolve())
    side['metric_query_batch_size']=4096
    local=output/'reference_sidecar.json'
    if local.exists():
        if json.loads(local.read_text())!=side:raise ValueError('localized reference sidecar changed')
    else:write(local,side)
    if not manifest_path.exists():
        import jax
        if jax.default_backend()!='cpu':raise ValueError('JAX CPU backend required')
        write(manifest_path,{'identity':identity,'configuration':cfg,'inputs':im,'source_hashes':sources,
             'commit':commit,'input_root':str(input_root.resolve()),'localized_sidecar_sha256':sha(local),
             'python':sys.version,'platform':platform.platform(),'jax_backend':jax.default_backend()})
    else:
        saved=json.loads(manifest_path.read_text())
        if saved['localized_sidecar_sha256']!=sha(local):raise ValueError('localized sidecar hash changed')
    for name in ('chunks','scratch','cache','logs','invocations'):(output/name).mkdir(exist_ok=True)
    return identity

def load_topology(output,n):
    with np.load(Path(output)/f'N{n}.topology.npz',allow_pickle=False) as z:
        return z['face_ids'].copy(),z['family'].copy(),z['endpoints'].copy()

def choose_preflight(t,face_ids,family,endpoints):
    n=t.n
    old=[(n-1,0,0),(n-1,n//2,n//2),(n-1,n-1,n-1),(n-4,0,0),(n-5,n//2,n//2)]
    anchors=old+[(int(np.argmin(abs(t.centers[0]-u))),j,k) for u in (.25,.5,.75) for j,k in ((0,0),(n//2,n//2))]
    selected=[int(t.ro[np.ravel_multi_index(a,(n,n,n))]) for a in anchors]
    extra,labels=core.numerics.select_owners(t.g,t.centers)
    selected=np.unique(np.r_[selected,extra])
    positions=np.flatnonzero(np.isin(endpoints,selected).any(axis=1))
    family_extra=[int(np.flatnonzero(family==code)[0]) for code in range(8) if not np.any(family[positions]==code)]
    positions=np.unique(np.r_[positions,np.asarray(family_extra,dtype=np.int64)])
    raw=np.flatnonzero(np.isin(t.ro,selected))
    # A fixed geometry-stratified face subset; q5 does not alter the candidate.
    q5=[]
    for code in range(1,8):
        choices=positions[family[positions]==code]
        if len(choices):q5.append(int(choices[len(choices)//2]))
    for o in selected:
        members=core.k.members(t,int(o))
        if len(members)>1:
            choices=positions[np.any(endpoints[positions]==o,axis=1)&(family[positions]!=0)]
            if len(choices):q5.append(int(choices[len(choices)//2]))
    for anchor in anchors[5:]:
        owner=int(t.ro[np.ravel_multi_index(anchor,(n,n,n))])
        choices=positions[np.any(endpoints[positions]==owner,axis=1)&(family[positions]!=0)]
        if len(choices):q5.append(int(choices[len(choices)//2]))
    for name in ('axis','agglomerated'):
        mask=core.numerics.masks(t.g)[name]
        choices_owner=selected[mask[selected]]
        if len(choices_owner):
            choices=positions[np.any(endpoints[positions]==choices_owner[0],axis=1)&(family[positions]!=0)]
            if len(choices):q5.append(int(choices[len(choices)//2]))
    q5=np.unique(q5)
    return {'owner_ids':selected,'raw_ids':raw,'face_ids':face_ids[positions],
            'q5_face_ids':face_ids[q5],'anchors':anchors,'additional_labels':labels}

def topo_one(input_root,output,n):
    core.topology.census(n,input_root,output)
    t,_=core.load(input_root,Path(output)/'reference_sidecar.json',n)
    ids,fam,ep=load_topology(output,n);sel=choose_preflight(t,ids,fam,ep)
    save(Path(output)/f'N{n}.preflight_selection.npz',owner_ids=sel['owner_ids'],raw_ids=sel['raw_ids'],face_ids=sel['face_ids'],q5_face_ids=sel['q5_face_ids'])
    write(Path(output)/f'N{n}.preflight_selection.json',{'owner_ids':sel['owner_ids'],'raw_count':len(sel['raw_ids']),
       'face_count':len(sel['face_ids']),'q5_face_count':len(sel['q5_face_ids']),
       'anchors':sel['anchors'],'additional_labels':sel['additional_labels']})
    return {'n':n,'faces':len(ids),'owners':len(t.vol),'preflight_faces':len(sel['face_ids']),
            'preflight_owners':len(sel['owner_ids']),'rss_gib':rss()}

def effective(args):
    if args.workers is None or args.workers<1:raise ValueError('--workers must be positive for computational stages')
    if args.worker_memory_gib is None or args.worker_memory_gib<=0 or args.memory_budget_gib is None or args.memory_budget_gib<=0:
        raise ValueError('--worker-memory-gib and --memory-budget-gib required')
    count=min(args.workers,int((args.memory_budget_gib-args.memory_reserve_gib)//args.worker_memory_gib))
    if count<1:raise ValueError('memory budget cannot fit one worker plus reserve')
    return count

def topology_stage(args,identity):
    output=args.output;path=output/'topology_summary.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:raise ValueError('topology identity changed')
        for n in args.resolutions:
            if sha(output/f'N{n}.topology.npz')!=saved['hashes'][str(n)]:raise ValueError('topology hash changed')
            if sha(output/f'N{n}.preflight_selection.npz')!=saved['selection_hashes'][str(n)]:
                raise ValueError('preflight selection hash changed')
        return
    count=effective(args)
    with ProcessPoolExecutor(max_workers=count,mp_context=get_context('spawn')) as pool:
        items=list(pool.map(topo_one,[str(args.input_root)]*len(args.resolutions),[str(output)]*len(args.resolutions),args.resolutions))
    if any(item['rss_gib']>args.worker_memory_gib for item in items):raise MemoryError('topology worker exceeded declared memory cap')
    write(path,{'identity':identity,'results':items,'hashes':{str(n):sha(output/f'N{n}.topology.npz') for n in args.resolutions},
       'selection_hashes':{str(n):sha(output/f'N{n}.preflight_selection.npz') for n in args.resolutions},
       'requested_workers':args.workers,'effective_workers':count})

def ids_for(output,n,stage):
    if stage in ('faces','preflight-face'):
        if stage=='faces':return load_topology(output,n)[0]
        with np.load(Path(output)/f'N{n}.preflight_selection.npz') as z:return z['face_ids'].copy()
    if stage=='preflight-q5':
        with np.load(Path(output)/f'N{n}.preflight_selection.npz') as z:return z['q5_face_ids'].copy()
    if stage=='preflight-reference':
        with np.load(Path(output)/f'N{n}.preflight_selection.npz') as z:return z['raw_ids'].copy()
    return np.arange(n**3,dtype=np.int64)

def plan(output,identity,resolutions):
    cfg=config();chunks={'preflight-face':cfg['preflight_face_chunk'],'preflight-reference':cfg['preflight_reference_chunk'],
       'preflight-q5':cfg['preflight_face_chunk'],'observations':cfg['observation_chunk'],
       'faces':cfg['face_chunk'],'reference':cfg['reference_chunk']}
    data={'identity':identity,'resolutions':resolutions,'stages':{}}
    for stage in STAGES:
        data['stages'][stage]={}
        for n in resolutions:
            ids=ids_for(output,n,stage);size=chunks[stage]
            data['stages'][stage][str(n)]=[{'stage':stage,'n':n,'start':j,'stop':min(j+size,len(ids))} for j in range(0,len(ids),size)]
    path=Path(output)/'plan.json'
    if path.exists() and json.loads(path.read_text())!=data:raise ValueError('plan/configuration changed')
    write(path,data);return data

def unit_path(output,unit):
    return Path(output)/'chunks'/f"N{unit['n']}"/unit['stage']/f"{unit['start']:07d}-{unit['stop']-1:07d}.npz"
def unit_ids(output,unit):return ids_for(output,unit['n'],unit['stage'])[unit['start']:unit['stop']]
def valid_unit(output,unit,identity):
    path=unit_path(output,unit);receipt=path.with_suffix('.json')
    if not path.exists() and not receipt.exists():return False
    if not path.exists() or not receipt.exists():return False
    item=json.loads(receipt.read_text())
    if item['identity']!=identity or item['unit']!=unit:raise ValueError(f'stale checkpoint {path}')
    if sha(path)!=item['sha256']:raise ValueError(f'corrupt checkpoint {path}')
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z['ids'],unit_ids(output,unit)):raise ValueError(f'wrong chunk IDs {path}')
        for name in z.files:
            if z[name].dtype.kind in 'fc' and not np.isfinite(z[name]).all():raise ValueError(f'nonfinite {path}:{name}')
        expected={'preflight-face':('N','D','O_q3','endpoints','N_zero_data','N_zero_owner'),'faces':('N','D','O_q3','endpoints'),
          'preflight-reference':('numerator','control_numerator','owner_ids'),
          'reference':('numerator','owner_ids'),'observations':('numerator','owner_ids'),
          'preflight-q5':('O_q5',)}[unit['stage']]
        if any(key not in z for key in expected):raise ValueError(f'incomplete chunk {path}')
    return True

def initialize(input_root,output,n,stage,identity,worker_memory_gib):
    global STATE
    output=Path(output);os.environ['XDG_CACHE_HOME']=str(output/'cache');os.environ['DRBX_CACHE_DIR']=str(output/'cache/jax');os.environ['TMPDIR']=str(output/'scratch')
    import jax
    if jax.default_backend()!='cpu':raise ValueError('JAX CPU backend required in worker')
    t,ref=core.load(input_root,output/'reference_sidecar.json',n)
    if abs(ref.finite_difference_step-config()['metric_derivative_step'])>1e-15:
        raise ValueError('continuum metric derivative step differs from frozen contract')
    obs=None
    if stage=='faces':
        path=output/f'N{n}.owner_values.npz';record=json.loads((output/f'N{n}.observations.reduction.json').read_text())
        if sha(path)!=record['owner_values_sha256']:raise ValueError('owner values hash changed')
        with np.load(path,allow_pickle=False) as z:obs=z['values'].copy()
    STATE={'t':t,'ref':ref,'obs':obs,'stage':stage,'output':output,'identity':identity,'n':n,
           'worker_memory_gib':worker_memory_gib}

def compute(unit):
    s=STATE;ids=unit_ids(s['output'],unit);stage=s['stage'];started=time.monotonic()
    if stage in ('faces','preflight-face'):
        all_ids,fam,ep=load_topology(s['output'],s['n']);pos=np.searchsorted(all_ids,ids)
        data=core.face_chunk(s['t'],s['ref'],ids,fam[pos],ep[pos],s['obs'],patch_limit=config()['patch_cache_limit'],
                              linearity=(stage=='preflight-face'))
    elif stage=='preflight-q5':data=core.oracle_faces(s['t'],s['ref'],ids,5)
    elif stage=='observations':data=core.observation_chunk(s['t'],s['ref'],ids)
    else:
        data=core.reference_chunk(s['t'],s['ref'],ids)
        if stage=='preflight-reference':
            original=s['ref'].finite_difference_step
            try:
                s['ref'].finite_difference_step=config()['metric_derivative_control_step']
                data['control_numerator']=core.reference_chunk(s['t'],s['ref'],ids,step=config()['wall_metric_tangent_control_step'])['numerator']
            finally:s['ref'].finite_difference_step=original
    if rss()>s['worker_memory_gib']:
        raise MemoryError(f"{stage} worker exceeded declared memory cap: {rss():.3f} GiB")
    path=unit_path(s['output'],unit);save(path,**data)
    record={'identity':s['identity'],'unit':unit,'sha256':sha(path),'seconds':time.monotonic()-started,
            'peak_rss_gib':rss(),'pid':os.getpid(),'jax_backend':'cpu'}
    write(path.with_suffix('.json'),record)
    if not valid_unit(s['output'],unit,s['identity']):raise ValueError('new chunk did not validate')
    return record

def run_stage(args,stage,stage_units,identity):
    count=effective(args);output=args.output;records=[];resumed=0;run=0
    for n in args.resolutions:
        units=stage_units[str(n)];todo=[]
        for u in units:
            if valid_unit(output,u,identity):resumed+=1
            else:todo.append(u)
        if args.max_units is not None:
            remain=max(0,args.max_units-run);todo=todo[:remain]
        if not todo:continue
        with ProcessPoolExecutor(max_workers=count,mp_context=get_context('spawn'),initializer=initialize,
            initargs=(str(args.input_root),str(output),n,stage,identity,args.worker_memory_gib),max_tasks_per_child=args.max_tasks_per_worker) as pool:
            pending={};iterator=iter(todo)
            def fill():
                for _ in range(max(0,2*count-len(pending))):
                    u=next(iterator,None)
                    if u is None:break
                    pending[pool.submit(compute,u)]=u
            fill()
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    u=pending.pop(future)
                    try:records.append(future.result());run+=1
                    except Exception as exc:
                        write(output/'failure.json',{'stage':stage,'unit':u,'error':repr(exc),'time':time.time()});raise
                write(output/'progress.json',{'stage':stage,'n':n,'completed_this_invocation':run,'time':time.time()})
                fill()
    write(output/'executions'/f'{time.time_ns()}_{stage}.json',{'stage':stage,'requested_workers':args.workers,
       'effective_workers':count,'memory_budget_gib':args.memory_budget_gib,'worker_memory_gib':args.worker_memory_gib,
       'memory_reserve_gib':args.memory_reserve_gib,'executed_units':run,'resumed_units':resumed,
       'peak_worker_rss_gib':max((r['peak_rss_gib'] for r in records),default=0),
       'worker_seconds':sum(r['seconds'] for r in records)})
    return run

def complete_chunks(output,units,identity):
    cursor=0;expected=ids_for(output,units[0]['n'],units[0]['stage']) if units else np.empty(0,dtype=np.int64)
    for u in units:
        if u['start']!=cursor or u['stop']<=cursor or not valid_unit(output,u,identity):raise ValueError(f'incomplete/overlapping stage at {cursor}: {u}')
        cursor=u['stop']
    if cursor!=len(expected):raise ValueError(f'missing tail stage IDs {cursor}/{len(expected)}')
    return len(expected)

def reduce_observations(output,n,units,identity):
    t,_=core.load(json.loads((output/'campaign_manifest.json').read_text())['input_root'],output/'reference_sidecar.json',n)
    values=np.zeros((len(t.vol),len(core.NAMES)));volume=np.zeros(len(t.vol))
    complete_chunks(output,units,identity)
    for u in units:
        with np.load(unit_path(output,u)) as z:
            ids=z['ids'];oid=z['owner_ids']
            if not np.array_equal(oid,t.ro[ids]):raise ValueError('observation owner mismatch')
            np.add.at(values,oid,z['numerator']);np.add.at(volume,oid,t.rv[ids])
    if not np.allclose(volume,t.vol,rtol=1e-12,atol=1e-12):raise ValueError('incomplete raw observation members')
    values/=t.vol[:,None]
    path=output/f'N{n}.owner_values.npz';save(path,values=values)
    return {'owner_values_sha256':sha(path),'owners':len(t.vol),'raw_members':n**3}

def reduce_reference(t,output,units,identity,control=False):
    complete_chunks(output,units,identity)
    numerator=np.zeros((len(t.vol),len(core.NAMES)));control_num=np.zeros_like(numerator);volume=np.zeros(len(t.vol))
    for u in units:
        with np.load(unit_path(output,u)) as z:
            ids=z['ids'];oid=z['owner_ids']
            if not np.array_equal(oid,t.ro[ids]):raise ValueError('reference owner mismatch')
            np.add.at(numerator,oid,z['numerator']);np.add.at(volume,oid,t.rv[ids])
            if control:np.add.at(control_num,oid,z['control_numerator'])
    return numerator,control_num,volume

def reduce_faces(t,output,units,identity):
    complete_chunks(output,units,identity)
    all_ids,all_fam,all_ep=load_topology(output,t.n)
    arrays={key:np.zeros((len(t.vol),len(core.NAMES))) for key in ('N','D','O_q3')}
    wall={key:np.zeros(len(core.NAMES)) for key in ('area','signed','square','grad_square','max','normalized_max')}
    balance=np.zeros(len(core.NAMES));count=0
    for u in units:
        with np.load(unit_path(output,u)) as z:
            ids=z['ids'];fam=z['family'];ep=z['endpoints'];count+=len(ids)
            pos=np.searchsorted(all_ids,ids)
            if not np.array_equal(all_ids[pos],ids) or not np.array_equal(ep,all_ep[pos]) or not np.array_equal(fam,all_fam[pos]):
                raise ValueError('face topology changed')
            if np.any(fam==0):
                axis=fam==0
                if any(np.any(z[key][axis]!=0) for key in ('N','D','O_q3')):raise ValueError('collapsed axis nonzero')
            for col,sign in ((0,-1),(1,1)):
                owner=ep[:,col];valid=owner>=0
                for key in arrays:np.add.at(arrays[key],owner[valid],sign*z[key][valid])
            balance+=np.sum(np.where(ep[:,0,None]<0,z['N'],0)-np.where(ep[:,1,None]<0,z['N'],0),axis=0)
            wall['area']+=sum(z['wall_area'])
            for name,key in (('signed','wall_signed'),('square','wall_square'),('grad_square','wall_grad_square')):wall[name]+=np.sum(z[key],axis=0)
            for name,key in (('max','wall_max'),('normalized_max','wall_normalized_max')):wall[name]=np.maximum(wall[name],np.max(z[key],axis=0))
    return arrays,wall,balance,count

def reduce_face_linearity(output,units,identity):
    """max|N-(N_zero_data+N_zero_owner)| and max|N| over boundary-family (1,2,4) faces."""
    complete_chunks(output,units,identity)
    max_abs=0.0;max_n=0.0
    for u in units:
        with np.load(unit_path(output,u)) as z:
            mask=np.isin(z['family'],(1,2,4))
            if not np.any(mask):continue
            diff=z['N'][mask]-(z['N_zero_data'][mask]+z['N_zero_owner'][mask])
            max_abs=max(max_abs,float(np.max(np.abs(diff))))
            max_n=max(max_n,float(np.max(np.abs(z['N'][mask]))))
    return max_abs,max_n

def masks(t,ep,fam):
    result=core.numerics.masks(t.g);n=t.n;radius=t.g.owner_flat_ids//(n*n)
    result['axis_core']=radius==0;result['first_ring']=radius==1
    coupled=np.zeros(len(t.vol),bool);ringwise=np.zeros(len(t.vol),bool)
    for code,target in ((7,coupled),(6,ringwise)):
        pairs=ep[fam==code];target[pairs[pairs>=0]]=True
    result['coupled_region']=coupled;result['ringwise_region']=ringwise;result['coupled_ringwise_join']=coupled&ringwise
    result['aggregate_interface']=result['transition'];result['physical_wall']=result['boundary']
    th=np.zeros((n,n,n),bool);th[:,(0,n-1),:]=True
    eta=np.zeros((n,n,n),bool);eta[:,:,(0,n-1)]=True
    result['theta_seam']=np.bincount(t.ro,weights=th.ravel(),minlength=len(t.vol))>0
    result['eta_seam']=np.bincount(t.ro,weights=eta.ravel(),minlength=len(t.vol))>0
    correction=(t.g.owner_flat_ids//(n*n))
    radial=t.centers[0][correction]
    result['correction_collar']=(radial>=.25)&(radial<=.75)
    result['correction_transition']=np.isin(correction,(int(.25*n),int(.5*n),int(.75*n)))
    return result

def stats(error,volume,regions):
    sq=volume*error**2;total=float(np.sum(sq));V=float(np.sum(volume))
    out={'l2':float(np.sqrt(total/V)),'max_abs':float(np.max(abs(error))),'squared_error_integral':total,'regions':{}}
    for label,mask in regions.items():
        v=float(np.sum(volume[mask]));out['regions'][label]={'owners':int(np.sum(mask)),
          'l2':float(np.sqrt(np.sum(sq[mask])/v)) if v>0 else None,
          'max_abs':float(np.max(abs(error[mask]))) if np.any(mask) else None}
    return out

def reduce_preflight(args,plan,identity):
    output=args.output;cases={}
    for n in args.resolutions:
        t,_=core.load(args.input_root,output/'reference_sidecar.json',n)
        units=plan['stages'];faces,wall,_,_=reduce_faces(t,output,units['preflight-face'][str(n)],identity)
        lin_max_abs,lin_max_n=reduce_face_linearity(output,units['preflight-face'][str(n)],identity)
        row_linearity_pass=bool(lin_max_abs<=1e-12*max(1.0,lin_max_n))
        r,control,vol=reduce_reference(t,output,units['preflight-reference'][str(n)],identity,True)
        with np.load(output/f'N{n}.preflight_selection.npz') as z:owners=z['owner_ids'];q5_ids=z['q5_face_ids']
        if not np.allclose(vol[owners],t.vol[owners],rtol=1e-12,atol=1e-12):raise ValueError('preflight missing owner members')
        q5units=units['preflight-q5'][str(n)];complete_chunks(output,q5units,identity)
        q5={};q3={}
        for u in q5units:
            with np.load(unit_path(output,u)) as z:
                for fid,val in zip(z['ids'],z['O_q5']):q5[int(fid)]=val.tolist()
        for u in units['preflight-face'][str(n)]:
            with np.load(unit_path(output,u)) as z:
                for fid,val in zip(z['ids'],z['O_q3']):
                    if int(fid) in q5:q3[int(fid)]=val.tolist()
        if set(q5)!=set(map(int,q5_ids)):raise ValueError('q5 sample coverage mismatch')
        if set(q3)!=set(q5):raise ValueError('q3 control face coverage mismatch')
        R=r[owners]/t.vol[owners,None];Rc=control[owners]/t.vol[owners,None]
        action={key:value[owners]/t.vol[owners,None] for key,value in faces.items()}
        max_uncertainty=np.max(abs(Rc-R),axis=0)
        weight=t.vol[owners];denom=np.sum(weight)
        reference_checks={}
        for col,name in enumerate(core.NAMES):
            uncertainty=float(np.sqrt(np.sum(weight*(Rc[:,col]-R[:,col])**2)/denom))
            spatial=float(np.sqrt(np.sum(weight*(action['N'][:,col]-R[:,col])**2)/denom))
            reference_checks[name]={'uncertainty_l2':uncertainty,'sample_spatial_l2':spatial,
              'pass':uncertainty<=config()['bounded_reference_fraction_maximum']*spatial+config()['bounded_reference_absolute_floor']}
        cases[str(n)]={'owners':owners.tolist(),'face_count':len(ids_for(output,n,'preflight-face')),
           'q5_face_count':len(q5),'max_reference_control_delta':max_uncertainty.tolist(),
           'row_linearity_max_abs':lin_max_abs,'row_linearity_pass':row_linearity_pass,
           'reference_checks':reference_checks,
           'q5_sample':{str(fid):{'O_q3':q3[fid], 'O_q5':q5[fid],
             'O_q3_minus_O_q5':(np.asarray(q3[fid])-np.asarray(q5[fid])).tolist()} for fid in sorted(q5)},
           'fields':{name:{'N':action['N'][:,j].tolist(),'D':action['D'][:,j].tolist(),
             'O_q3':action['O_q3'][:,j].tolist(),'R':R[:,j].tolist(),
             'N_minus_D':(action['N'][:,j]-action['D'][:,j]).tolist(),
             'N_minus_O':(action['N'][:,j]-action['O_q3'][:,j]).tolist(),
             'O_minus_R':(action['O_q3'][:,j]-R[:,j]).tolist(),
             'N_minus_R':(action['N'][:,j]-R[:,j]).tolist()} for j,name in enumerate(core.NAMES)}}
        save(output/f'N{n}.preflight.npz',owner_ids=owners,N=action['N'],D=action['D'],O_q3=action['O_q3'],R=R,R_control=Rc,
             q5_face_ids=q5_ids,O_q5=np.array([q5[int(fid)] for fid in q5_ids]))
    write(output/'preflight.json',{'identity':identity,'cases':cases,'status':'complete',
       'scientific_policy':'reference numerical uncertainty <= 10% of sampled N-R L2; O-R is not uncertainty'})

def reduce_global(args,plan,identity):
    output=args.output;cfg=config();cases={}
    for n in args.resolutions:
        t,_=core.load(args.input_root,output/'reference_sidecar.json',n)
        obs_record=reduce_observations(output,n,plan['stages']['observations'][str(n)],identity)
        write(output/f'N{n}.observations.reduction.json',{'identity':identity,**obs_record})
        faces,wall,balance,count=reduce_faces(t,output,plan['stages']['faces'][str(n)],identity)
        r,_,vol=reduce_reference(t,output,plan['stages']['reference'][str(n)],identity)
        if not np.allclose(vol,t.vol,rtol=1e-12,atol=1e-12):raise ValueError('global missing raw reference members')
        N,D,O=(faces[key]/t.vol[:,None] for key in ('N','D','O_q3'));R=r/t.vol[:,None]
        fid,fam,ep=load_topology(output,n)
        if count!=len(fid):raise ValueError('global face count mismatch')
        regional=masks(t,ep,fam)
        results={}
        for col,name in enumerate(core.NAMES):
            results[name]={'N_minus_R':stats(N[:,col]-R[:,col],t.vol,regional),
              'N_minus_O':stats(N[:,col]-O[:,col],t.vol,regional),
              'O_minus_R':stats(O[:,col]-R[:,col],t.vol,regional),
              'N_minus_D':stats(N[:,col]-D[:,col],t.vol,regional)}
        if np.max(abs(np.sum(faces['N'],axis=0)-balance))>1e-6:raise ValueError('shared-face balance failed')
        wall_stats={}
        for j,name in enumerate(core.NAMES):
            wall_stats[name]={'area':float(wall['area'][j]),'signed_bias':float(wall['signed'][j]/wall['area'][j]),
              'rms':float(np.sqrt(wall['square'][j]/wall['area'][j])),
              'max_abs':float(wall['max'][j]),
              'physical_gradient_normalized_rms':float(np.sqrt(wall['square'][j]/max(wall['grad_square'][j],1e-24))),
              'physical_gradient_normalized_max':float(wall['normalized_max'][j]),
              'normalization':'physical |grad_x f|=sqrt(g_contra^ij f_i f_j); pointwise floor 1e-12 field/m'}
        save(output/f'N{n}.global.npz',owner_ids=np.arange(len(t.vol)),owner_flat_ids=t.g.owner_flat_ids,
            volume=t.vol,N=N,D=D,O_q3=O,R=R,N_minus_O=N-O,O_minus_R=O-R,N_minus_R=N-R,N_minus_D=N-D,
            **{f'region_{name}':mask for name,mask in regional.items()})
        cases[str(n)]={'owners':len(t.vol),'faces':count,'fields':results,'wall':wall_stats,
                        'arrays_sha256':sha(output/f'N{n}.global.npz'),'balance':(np.sum(faces['N'],axis=0)-balance).tolist()}
    orders={}
    for name in core.NAMES[:-1]:
        e=np.array([cases[str(n)]['fields'][name]['N_minus_R']['l2'] for n in args.resolutions])
        if len(e)==3 and np.all(e>0):
            p=np.log(e[:-1]/e[1:])/np.log(np.array([48/32,64/48]));passed=bool(np.all(p>=cfg['global_l2_order_minimum']))
            orders[name]={'errors':e.tolist(),'orders':p.tolist(),'order_pass':passed}
    def ungated_orders(key):
        # Not gated: reported next to the N-R `orders` above, with the per-grid
        # regional L2 (and max_abs) that stats() already computed for this split,
        # so the N-O/O-R decomposition of the N-R gate reads straight from summary.json.
        out={}
        for name in core.NAMES[:-1]:
            per_grid={str(n):{'l2':cases[str(n)]['fields'][name][key]['l2'],
                'max_abs':cases[str(n)]['fields'][name][key]['max_abs'],
                'regions':cases[str(n)]['fields'][name][key]['regions']} for n in args.resolutions}
            e=np.array([per_grid[str(n)]['l2'] for n in args.resolutions])
            entry={'per_grid':per_grid}
            if len(e)==3 and np.all(e>0):
                p=np.log(e[:-1]/e[1:])/np.log(np.array([48/32,64/48]))
                entry['errors']=e.tolist();entry['orders']=p.tolist()
            out[name]=entry
        return out
    O_minus_R_orders=ungated_orders('O_minus_R');N_minus_O_orders=ungated_orders('N_minus_O')
    constant=max(cases[str(n)]['fields']['constant']['N_minus_R']['max_abs'] for n in args.resolutions)
    wall_trend={name:bool(all(cases[str(a)]['wall'][name]['rms']>=cases[str(b)]['wall'][name]['rms'] for a,b in ((32,48),(48,64)))) for name in core.NAMES[:-1]} if len(args.resolutions)==3 else {}
    preflight=json.loads((output/'preflight.json').read_text())
    summary={'identity':identity,'computation_completed':len(args.resolutions)==3,'cases':cases,'orders':orders,
      'global_order_pass':len(orders)==4 and all(x['order_pass'] for x in orders.values()),
      'constant_pass':constant<=cfg['constant_action_absolute_maximum'],
      'wall_rms_trend_pass':bool(wall_trend) and all(wall_trend.values()),'wall_rms_trend':wall_trend,
      'reference_qualified_by_bounded_checks':all(x['pass'] for case in preflight['cases'].values()
          for x in case['reference_checks'].values()),
      'O_minus_R_orders':O_minus_R_orders,'N_minus_O_orders':N_minus_O_orders,
      'row_linearity_pass':all(case['row_linearity_pass'] for case in preflight['cases'].values()),
      'accuracy_record_note':'O_minus_R_orders and N_minus_O_orders are reported, not gated; the scientific gate is global_order_pass on N_minus_R alone',
      'reference_convention':cfg['reference'],'production_promoted':False}
    write(output/'summary.json',summary)
    return summary

def validate(args,plan,identity):
    if plan['identity']!=identity:raise ValueError('plan identity changed')
    output=args.output
    topology_record=json.loads((output/'topology_summary.json').read_text())
    if topology_record['identity']!=identity:raise ValueError('topology summary identity changed')
    for n in args.resolutions:
        if sha(output/f'N{n}.topology.npz')!=topology_record['hashes'][str(n)]:raise ValueError('topology changed')
        if sha(output/f'N{n}.preflight_selection.npz')!=topology_record['selection_hashes'][str(n)]:
            raise ValueError('preflight selection changed')
    for stage in STAGES:
        for n in args.resolutions:complete_chunks(output,plan['stages'][stage][str(n)],identity)
    pf=json.loads((output/'preflight.json').read_text());summary=json.loads((output/'summary.json').read_text())
    if pf['identity']!=identity or summary['identity']!=identity or not summary['computation_completed']:raise ValueError('incomplete results')
    for n in args.resolutions:
        if sha(output/f'N{n}.global.npz')!=summary['cases'][str(n)]['arrays_sha256']:raise ValueError('global result hash changed')
        t,_=core.load(args.input_root,output/'reference_sidecar.json',n)
        face_totals,_,_,face_count=reduce_faces(t,output,plan['stages']['faces'][str(n)],identity)
        reference,_,covered=reduce_reference(t,output,plan['stages']['reference'][str(n)],identity)
        if face_count!=len(ids_for(output,n,'faces')) or not np.allclose(covered,t.vol,rtol=1e-12,atol=1e-12):
            raise ValueError('validation coverage mismatch')
        with np.load(output/f'N{n}.global.npz') as z:
            if not np.array_equal(z['owner_ids'],np.arange(len(z['volume']))):raise ValueError('owner coverage mismatch')
            for name in ('N','D','O_q3','R'):
                if z[name].shape!=(len(z['volume']),len(core.NAMES)):raise ValueError('action shape mismatch')
            for name in ('N','D','O_q3'):
                if not np.array_equal(z[name],face_totals[name]/t.vol[:,None]):raise ValueError(f'{name} reduction changed')
            if not np.array_equal(z['R'],reference/t.vol[:,None]):raise ValueError('reference reduction changed')
            for key,left,right in (('N_minus_O','N','O_q3'),('O_minus_R','O_q3','R'),
                                   ('N_minus_R','N','R'),('N_minus_D','N','D')):
                if not np.array_equal(z[key],z[left]-z[right]):raise ValueError(f'{key} split changed')
    write(output/'validation.json',{'identity':identity,'operational_complete':True,
       'scientific_global_order_pass':summary['global_order_pass'],
       'constant_pass':summary['constant_pass'],'wall_rms_trend_pass':summary['wall_rms_trend_pass']})

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify-inputs','topology','plan','preflight','run-stage','reduce-stage','run','validate'))
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resolutions',type=int,nargs='+',choices=(32,48,64),default=[32,48,64])
    p.add_argument('--stage',choices=STAGES);p.add_argument('--workers',type=int)
    p.add_argument('--memory-budget-gib',type=float);p.add_argument('--worker-memory-gib',type=float)
    p.add_argument('--memory-reserve-gib',type=float,default=1.0);p.add_argument('--max-tasks-per-worker',type=int,default=32)
    p.add_argument('--max-units',type=int)
    return p.parse_args()

def main():
    args=parse();args.input_root=args.input_root.resolve();args.output=args.output.resolve()
    if args.resolutions!=sorted(set(args.resolutions)):
        raise ValueError('resolutions must be unique and ascending')
    if args.max_tasks_per_worker<1:raise ValueError('max tasks per worker must be positive')
    if args.command in ('topology','preflight','run-stage','run'):effective(args)
    with lock(args.output):
        for name in ('scratch','cache','logs'):(args.output/name).mkdir(exist_ok=True)
        os.environ['XDG_CACHE_HOME']=str(args.output/'cache');os.environ['DRBX_CACHE_DIR']=str(args.output/'cache/jax');os.environ['TMPDIR']=str(args.output/'scratch')
        identity=verify(args.input_root,args.output)
        write(args.output/'invocations'/f'{time.time_ns()}_{args.command}.json',{'argv':sys.argv,'identity':identity,'time':time.time()})
        if args.command=='verify-inputs':return
        if args.command in ('topology','plan','preflight','run'):topology_stage(args,identity)
        if args.command=='topology':return
        if args.command in ('plan','preflight','run'):planned=plan(args.output,identity,args.resolutions)
        else:
            planned=json.loads((args.output/'plan.json').read_text())
            if planned['identity']!=identity or planned['resolutions']!=args.resolutions:raise ValueError('plan identity/resolutions mismatch')
        if args.command=='plan':return
        if args.command=='preflight':
            for stage in STAGES[:3]:run_stage(args,stage,planned['stages'][stage],identity)
            reduce_preflight(args,planned,identity)
        elif args.command=='run-stage':
            if args.stage is None:raise ValueError('--stage required')
            run_stage(args,args.stage,planned['stages'][args.stage],identity)
        elif args.command=='reduce-stage':
            if args.stage=='preflight-reference' or args.stage=='preflight-face' or args.stage=='preflight-q5':reduce_preflight(args,planned,identity)
            elif args.stage=='observations':
                for n in args.resolutions:
                    result=reduce_observations(args.output,n,planned['stages']['observations'][str(n)],identity)
                    write(args.output/f'N{n}.observations.reduction.json',{'identity':identity,**result})
            elif args.stage in ('faces','reference'):reduce_global(args,planned,identity)
            else:raise ValueError('--stage required')
        elif args.command=='run':
            if not (args.output/'preflight.json').exists() or json.loads((args.output/'preflight.json').read_text())['identity']!=identity:
                raise ValueError('complete preflight required before main run')
            run_stage(args,'observations',planned['stages']['observations'],identity)
            for n in args.resolutions:
                result=reduce_observations(args.output,n,planned['stages']['observations'][str(n)],identity)
                write(args.output/f'N{n}.observations.reduction.json',{'identity':identity,**result})
            for stage in ('faces','reference'):run_stage(args,stage,planned['stages'][stage],identity)
            reduce_global(args,planned,identity)
            if args.resolutions==[32,48,64]:validate(args,planned,identity)
        elif args.command=='validate':validate(args,planned,identity)
    print(json.dumps({'command':args.command,'status':'complete','output':str(args.output)}))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        a=parse();write(a.output/'last_exit.json',{'command':a.command,'status':'failed','error':repr(exc),'time':time.time()});raise
