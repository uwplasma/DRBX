#!/usr/bin/env python3
"""Node-local parallel CPU P07 global operator qualification (computation only)."""
from pathlib import Path
import os,sys,json,hashlib,time,argparse,fcntl,subprocess,platform,multiprocessing as mp
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from contextlib import contextmanager
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];STATE={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def encode(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,default=encode)+'\n');tmp.replace(p)
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**data)
 tmp.replace(p)
@contextmanager
def locked(output):
 output.mkdir(parents=True,exist_ok=True)
 with (output/'.campaign.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield

def verify(args):
 cfg=json.loads((HERE/'configuration.json').read_text());im=json.loads((HERE/'input_manifest.json').read_text());sm=json.loads((HERE/'source_manifest.json').read_text())
 for root,manifest in ((args.input_root,im),(REPO,sm)):
  for rec in manifest['files']:
   p=root/rec['path']
   if not p.is_file() or p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:raise ValueError(f'missing/changed input or source: {p}')
 content={'configuration':cfg,'inputs':im,'sources':sm};ident=digest(content);p=args.output/'campaign_manifest.json'
 if p.exists() and json.loads(p.read_text())['identity']!=ident:raise ValueError('incompatible campaign identity; use a new output folder')
 side=json.loads((args.input_root/cfg['inputs']['reference_sidecar']).read_text())
 for name in ('metric_cache','makegrid'):side[name]['path']=str((args.input_root/cfg['inputs'][name]).resolve())
 side['metric_query_batch_size']=cfg['metric_query_batch_size']
 write(args.output/'reference_sidecar.json',side)
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
 write(p,{'identity':ident,'content':content,'commit':commit,'input_root':str(args.input_root),'source_root':str(REPO),'environment':{'python':sys.version,'platform':platform.platform(),'packages':{p:version(p) for p in ('numpy','scipy','jax','jaxlib')}}})
 return cfg,ident

def initialize(input_root,output,n,cfg,ident):
 global STATE
 setup_start=time.monotonic()
 output=Path(output);os.environ['DRBX_CACHE_DIR']=str(output/'cache/jax');os.environ['XDG_CACHE_HOME']=str(output/'cache');os.environ['TMPDIR']=str(output/'scratch');(output/'scratch').mkdir(exist_ok=True)
 import numerics as num
 faces,centers,raw_owner,raw_volume,volume,geom=num.build_context(n,Path(input_root)/cfg['inputs']['geometry'])
 ref=num.reference(output/'reference_sidecar.json',verify_hashes=False);vals=num.owner_values(centers,raw_owner,raw_volume,volume,ref)
 STATE={'faces':faces,'centers':centers,'raw_owner':raw_owner,'raw_volume':raw_volume,'volume':volume,'geometry':geom,'reference':ref,'values':vals,'output':output,'identity':ident,'numerics':num,'n':n,'setup_seconds':time.monotonic()-setup_start}

def path_for(output,n,unit):return Path(output)/f'N{n}'/unit['scope']/f"{unit['id']}.npz"
def valid_unit(output,n,unit,ident):
 p=path_for(output,n,unit);receipt=p.with_suffix('.json')
 if not p.exists() and not receipt.exists():return False
 if not receipt.exists():return False # interrupted before atomic receipt
 r=json.loads(receipt.read_text())
 if r['identity']!=ident or r['unit']!=unit:raise ValueError(f'checkpoint identity mismatch: {receipt}')
 if not p.exists() or sha(p)!=r['sha256']:raise ValueError(f'checkpoint checksum mismatch: {p}')
 with np.load(p,allow_pickle=False) as z:
  if not np.array_equal(z['indices'],np.array(unit['indices'])):raise ValueError(f'checkpoint indices mismatch: {p}')
  for key in z.files:
   if z[key].dtype.kind in 'fc' and not np.all(np.isfinite(z[key])):raise ValueError(f'nonfinite checkpoint: {p}:{key}')
  required={'flux','oracle_flux','constant_flux','lower_raw','upper_raw','donors','donor_count'} if unit['kind']=='face' else {'numerator','continuous_volume'}
  if not required.issubset(z.files):raise ValueError(f'incomplete checkpoint: {p}')
 return True

def worker(unit):
 s=STATE;num=s['numerics'];ids=np.array(unit['indices'],np.int64);started=time.monotonic()
 result=num.face_chunk(s,ids) if unit['kind']=='face' else num.cell_chunk(s,ids,unit['order'])
 p=path_for(s['output'],s['n'],unit);save(p,result)
 receipt={'identity':s['identity'],'unit':unit,'sha256':sha(p),'pid':os.getpid(),'seconds':time.monotonic()-started,'peak_rss_gib':num.rss(),'setup_seconds':s['setup_seconds'],'exit_status':0}
 write(p.with_suffix('.json'),receipt)
 if not valid_unit(s['output'],s['n'],unit,s['identity']):raise ValueError('unit failed validation')
 return {k:receipt[k] for k in ('pid','seconds','peak_rss_gib','setup_seconds')}

def units_for(scope,faceids,rawids,cfg,orders=(3,)):
 units=[]
 for kind,ids,chunk in [('face',faceids,cfg['face_chunk']),('cell',rawids,cfg['cell_chunk'])]:
  for order in ([3] if kind=='face' else orders):
   for lo in range(0,len(ids),chunk):
    units.append({'scope':scope,'id':f'{kind}_q{order}_{lo:08d}','kind':kind,'order':order,'indices':np.asarray(ids[lo:lo+chunk]).tolist()})
 return units

def effective_workers(args):
 if args.workers is None or args.workers<1:raise ValueError('--workers must be chosen by the allocation setup')
 n=args.workers
 if args.memory_budget_gib is not None:
  if args.worker_memory_gib is None or args.worker_memory_gib<=0:raise ValueError('memory budget requires positive --worker-memory-gib')
  n=min(n,int((args.memory_budget_gib-args.memory_reserve_gib)//args.worker_memory_gib))
  if n<1:raise ValueError('memory budget does not fit one worker plus reserve')
 return n

def execute(args,n,units,cfg,ident):
 effective=effective_workers(args);todo=[u for u in units if not valid_unit(args.output,n,u,ident)];start=time.monotonic();done=len(units)-len(todo);records=[]
 write(args.output/f'N{n}'/f'{units[0]["scope"]}_plan.json',{'identity':ident,'units':units})
 with ProcessPoolExecutor(max_workers=effective,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(args.input_root),str(args.output),n,cfg,ident),max_tasks_per_child=args.max_tasks_per_worker) as pool:
  iterator=iter(todo);pending={}
  def submit():
   for _ in range(max(0,2*effective-len(pending))):
    u=next(iterator,None)
    if u is None:break
    pending[pool.submit(worker,u)]=u
  submit()
  while pending:
   finished,_=wait(pending,return_when=FIRST_COMPLETED)
   for fut in finished:
    unit=pending.pop(fut)
    try:records.append(fut.result())
    except Exception as exc:
     write(args.output/'failure.json',{'resolution':n,'unit':unit,'error':repr(exc),'updated_unix':time.time()});raise
    done+=1
   write(args.output/'progress.json',{'resolution':n,'scope':units[0]['scope'],'completed':done,'total':len(units),'updated_unix':time.time(),'elapsed_seconds':time.monotonic()-start})
   submit()
 execution_record={'identity':ident,'requested_workers':args.workers,'effective_workers':effective,'memory_budget_gib':args.memory_budget_gib,'worker_memory_gib':args.worker_memory_gib,'memory_reserve_gib':args.memory_reserve_gib,'resumed_units':len(units)-len(todo),'executed_units':len(todo),'seconds':time.monotonic()-start,'peak_worker_rss_gib':max([x['peak_rss_gib'] for x in records],default=0),'workers':records}
 write(args.output/f'N{n}'/f'{units[0]["scope"]}_execution.json',execution_record)
 write(args.output/'executions'/f'{time.time_ns()}_N{n}_{units[0]["scope"]}.json',execution_record)

def aggregate(output,n,units,ident,ctx,owners,rawids):
 num=ctx['numerics'];nowner=len(ctx['volume']);a=np.zeros((nowner,4));oracle=a.copy();constant=np.zeros((nowner,2));numerators={};volumes={};seen={};net=np.zeros(4);cnet=np.zeros(2)
 for unit in units:
  if not valid_unit(output,n,unit,ident):raise ValueError('missing validated unit')
  key=(unit['kind'],unit['order']);seen.setdefault(key,set());idx=unit['indices']
  if seen[key].intersection(idx):raise ValueError('overlapping unit coverage')
  seen[key].update(idx)
  with np.load(path_for(output,n,unit)) as z:
   if unit['kind']=='face':
    for which,sign in [('lower_raw',-1),('upper_raw',1)]:
     raw=z[which];ok=raw>=0;oi=ctx['raw_owner'][raw[ok]]
     np.add.at(a,oi,sign*z['flux'][ok]);np.add.at(oracle,oi,sign*z['oracle_flux'][ok]);np.add.at(constant,oi,sign*z['constant_flux'][ok])
    external=(z['lower_raw']<0)|(z['upper_raw']<0)
    sign=(z['lower_raw']<0).astype(float)-(z['upper_raw']<0).astype(float)
    net+=np.sum(sign[:,None]*z['flux'],axis=0);cnet+=np.sum(sign[:,None]*z['constant_flux'],axis=0)
   else:
    order=unit['order'];numerators.setdefault(order,np.zeros((nowner,4)));volumes.setdefault(order,np.zeros(nowner));oi=ctx['raw_owner'][z['indices']]
    np.add.at(numerators[order],oi,z['numerator']);np.add.at(volumes[order],oi,z['continuous_volume'])
 for order in numerators:
  if seen[('cell',order)]!=set(map(int,rawids)):raise ValueError('incomplete cell coverage')
 expected_faces=set(map(int,num.closure(ctx['geometry'],owners)[1])) if len(owners)<nowner else set(range(num.face_count(n)))
 if seen[('face',3)]!=expected_faces:raise ValueError('incomplete face coverage')
 V=ctx['volume'][owners];out={'owner_ids':owners,'owner_flat_ids':ctx['geometry'].owner_flat_ids[owners],'volume':V,'action':a[owners]/V[:,None],'oracle':oracle[owners]/V[:,None],'constant_action':constant[owners]/V[:,None]}
 for order in numerators:
  out[f'reference_q{order}']=numerators[order][owners]/volumes[order][owners,None]
  out[f'stored_volume_reference_q{order}']=numerators[order][owners]/V[:,None]
  out[f'continuous_average_q{order}']=numerators[order][owners]/volumes[order][owners,None]
  out[f'continuous_volume_q{order}']=volumes[order][owners]
 out['balance_residual']=np.sum(a,axis=0)-net;out['constant_balance_residual']=np.sum(constant,axis=0)-cnet
 # Regional primary masks are disjoint; seam flags remain independent diagnostics.
 for name,mask in num.masks(ctx['geometry']).items():out['region_'+name]=mask[owners]
 return out

def summarize(arrays):
 V=arrays['volume'];out={};orders=sorted(int(k[11:]) for k in arrays if k.startswith('reference_q'))
 for i,name in enumerate(('phi_mms','Ti_mms','regular_neumann','mixed_eta_neumann')):
  e=arrays['action'][:,i]-arrays['reference_q3'][:,i];e2=V*e*e
  stats={'l2':float(np.sqrt(sum(e2)/sum(V))),'max_abs':float(max(abs(e))),'squared_error_integral':float(sum(e2)),'regions':{}}
  for key in arrays:
   if key.startswith('region_'):
    mask=arrays[key];stats['regions'][key[7:]]={'owners':int(sum(mask)),'squared_error_integral':float(sum(e2[mask])),'global_squared_error_fraction':float(sum(e2[mask])/max(sum(e2),1e-300)),'l2':float(np.sqrt(sum(e2[mask])/sum(V[mask]))) if mask.any() else None}
  for order in orders:
   delta=arrays[f'reference_q{order}'][:,i]-arrays['reference_q3'][:,i]
   stats[f'q{order}_minus_q3_l2']=float(np.sqrt(V@(delta*delta)/sum(V)))
  out[name]=stats
 return out

def preflight(args,cfg,ident):
 for n in args.resolutions:
  initialize(args.input_root,args.output,n,cfg,ident);ctx=STATE;num=ctx['numerics'];owners,labels=num.select_owners(ctx['geometry'],ctx['centers']);rawids,faceids=num.closure(ctx['geometry'],owners)
  write(args.output/f'N{n}/selection.json',{'owners':owners,'labels':labels,'raw_ids':rawids,'face_ids':faceids})
  units=units_for('preflight',faceids,rawids,cfg,(3,5,7));execute(args,n,units,cfg,ident)
  arrays=aggregate(args.output,n,units,ident,ctx,owners,rawids);save(args.output/f'N{n}/preflight.npz',arrays)
  stats=summarize(arrays);write(args.output/f'N{n}/preflight.json',{'identity':ident,'stats':stats,'constant_max':float(np.max(abs(arrays['constant_action']))),'status':'complete','arrays_sha256':sha(args.output/f'N{n}/preflight.npz'),'qualification':'bounded actual-HSX reference comparison; scientific flags do not stop computation'})

def run(args,cfg,ident):
 for n in args.resolutions:
  pf=args.output/f'N{n}/preflight.json'
  if not pf.exists() or json.loads(pf.read_text())['identity']!=ident:raise ValueError(f'preflight required: {pf}')
  initialize(args.input_root,args.output,n,cfg,ident);ctx=STATE;num=ctx['numerics'];owners=np.arange(len(ctx['volume']));rawids=np.arange(n**3);faceids=np.arange(num.face_count(n));units=units_for('global',faceids,rawids,cfg)
  execute(args,n,units,cfg,ident);arrays=aggregate(args.output,n,units,ident,ctx,owners,rawids);save(args.output/f'N{n}/result.npz',arrays)
  write(args.output/f'N{n}/result.json',{'identity':ident,'status':'complete','stats':summarize(arrays),'constant_max':float(np.max(abs(arrays['constant_action']))),'balance_max':float(np.max(abs(arrays['balance_residual']))),'arrays_sha256':sha(args.output/f'N{n}/result.npz')})
 merge(args,cfg,ident)

def merge(args,cfg,ident):
 if not all((args.output/f'N{n}/result.json').exists() for n in (32,48,64)):return
 cases={str(n):json.loads((args.output/f'N{n}/result.json').read_text()) for n in (32,48,64)}
 if any(x['identity']!=ident for x in cases.values()):raise ValueError('case identity mismatch')
 fields={};refbudget={}
 for f in cases['32']['stats']:
  e=np.array([cases[str(n)]['stats'][f]['l2'] for n in (32,48,64)]);order=np.log(e[:-1]/e[1:])/np.log(np.array([48/32,64/48]));fields[f]={'errors':e,'orders':order,'order_pass':bool(np.all(order>=1.8))}
  refbudget[f]={}
  for n in (32,48,64):
   pf=json.loads((args.output/f'N{n}/preflight.json').read_text());s=pf['stats'][f]
   refbudget[f][str(n)]={'sample_q7_minus_q3_l2':s['q7_minus_q3_l2'],'fraction_of_sample_spatial_l2':s['q7_minus_q3_l2']/max(s['l2'],1e-300),'bounded_check_pass':bool(s['q7_minus_q3_l2']<=.1*s['l2'])}
 write(args.output/'summary.json',{'identity':ident,'computation_completed':True,'fields':fields,'global_order_pass':all(v['order_pass'] for v in fields.values()),'reference_bounded_checks':refbudget,'reference_qualified_by_bounded_checks':all(v['bounded_check_pass'] for d in refbudget.values() for v in d.values()),'production_promoted':False,'structural_elliptic_evolved_qualification':'separate milestones'})

def validate(args,cfg,ident):
 for n in args.resolutions:
  initialize(args.input_root,args.output,n,cfg,ident);ctx=STATE;num=ctx['numerics']
  for scope,file in [('preflight','preflight'),('global','result')]:
   plan=json.loads((args.output/f'N{n}/{scope}_plan.json').read_text())
   if plan['identity']!=ident:raise ValueError('plan identity mismatch')
   result=json.loads((args.output/f'N{n}/{file}.json').read_text())
   if result['identity']!=ident or result['arrays_sha256']!=sha(args.output/f'N{n}/{file}.npz'):raise ValueError('result identity/hash mismatch')
   if scope=='preflight':
    owners,_=num.select_owners(ctx['geometry'],ctx['centers']);rawids,_=num.closure(ctx['geometry'],owners)
   else:owners=np.arange(len(ctx['volume']));rawids=np.arange(n**3)
   # Reassemble both stages from independently checked chunks in fixed order.
   fresh=aggregate(args.output,n,plan['units'],ident,ctx,owners,rawids)
   with np.load(args.output/f'N{n}/{file}.npz') as z:
    if set(z.files)!=set(fresh):raise ValueError('assembled array catalogue mismatch')
    for k,v in fresh.items():
     if not np.array_equal(v,z[k]):raise ValueError(f'assembled result mismatch: {k}')
   if digest(summarize(fresh))!=digest(result['stats']):raise ValueError('summary statistics mismatch')
 merge(args,cfg,ident);write(args.output/'validation.json',{'identity':ident,'operational_validation':'passed','resolutions':args.resolutions,'scientific_results':'see summary.json; no automatic retuning'})

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['verify-inputs','preflight','run','validate']);ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resolutions',type=int,nargs='+',choices=[32,48,64],default=[32,48,64]);ap.add_argument('--workers',type=int);ap.add_argument('--memory-budget-gib',type=float);ap.add_argument('--worker-memory-gib',type=float);ap.add_argument('--memory-reserve-gib',type=float,default=1.);ap.add_argument('--max-tasks-per-worker',type=int,default=128)
 args=ap.parse_args();args.input_root=args.input_root.resolve();args.output=args.output.resolve()
 if args.command in ('preflight','run'):effective_workers(args)
 if args.max_tasks_per_worker<1:ap.error('--max-tasks-per-worker must be positive')
 with locked(args.output):
  cfg,ident=verify(args);write(args.output/'invocations'/f'{time.time_ns()}_{args.command}.json',{'command':sys.argv,'started_unix':time.time(),'pid':os.getpid(),'identity':ident})
  try:
   if args.command=='preflight':preflight(args,cfg,ident)
   elif args.command=='run':run(args,cfg,ident)
   elif args.command=='validate':validate(args,cfg,ident)
  except BaseException as exc:
   write(args.output/'last_exit.json',{'command':args.command,'status':'failed','error':repr(exc),'finished_unix':time.time()});raise
  write(args.output/'last_exit.json',{'command':args.command,'status':'completed','exit_code':0,'finished_unix':time.time()})
 print(json.dumps({'command':args.command,'status':'completed','output':str(args.output)}))
if __name__=='__main__':main()
