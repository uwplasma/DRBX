#!/usr/bin/env python3
"""Node-local CPU, resumable structured bracket campaign with bounded preflight."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'
import argparse,fcntl,hashlib,json,platform,subprocess,sys,time
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from p05_structured_global import numerics as k
from p07_combined_global import topology
from p07_combined_global.kernels import num,rss
STATE={}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def encode(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save_json(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,default=encode,allow_nan=False)+'\n');tmp.replace(path)
def save_npz(path,**x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**x)
 tmp.replace(path)
def config():return json.loads((HERE/'configuration.json').read_text())
def sources():
 files=[HERE/n for n in ('campaign.py','numerics.py','configuration.json','input_manifest.json')]
 files+=list((REPO/'scripts/perpendicular_structured').glob('*.py'))
 files += [REPO/n for n in ('scripts/p07_combined_global/kernels.py','scripts/p07_combined_global/topology.py','scripts/p07_diffusion_global/numerics.py','hsx_mms_continuum_reference.py','src/drbx/geometry/fci_perpendicular_bracket.py','src/drbx/geometry/fci_boundary_functional_reconstruction.py','src/drbx/geometry/MetricEvaluator.py','src/drbx/geometry/Bfield_evaluator.py')]
 return {str(p.relative_to(REPO)):sha(p) for p in files}
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
@contextmanager
def lock(output):
 output.mkdir(parents=True,exist_ok=True)
 with (output/'.campaign.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def verify(args):
 manifest=json.loads((HERE/'input_manifest.json').read_text())
 for item in manifest['files']:
  p=args.input_root/item['path']
  if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError(f'input identity mismatch: {p}')
 content=dict(commit=commit(),sources=sources(),config=config(),inputs=manifest)
 record=dict(identity=digest(content),content=content,input_root=str(args.input_root),python=sys.version,platform=platform.platform())
 dest=args.output/'manifest.json'
 if dest.exists() and json.loads(dest.read_text())['identity']!=record['identity']:raise ValueError('incompatible campaign; use a fresh folder')
 side=json.loads((args.input_root/'DRBX/work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json').read_text())
 for key,path in [('metric_cache','hsx_metric_d58d392545fd3917efeb83b6.npz'),('makegrid','mgrid_res2p5cm_180pln.nc'),('artifact','prototype_runs/geometry/hsx_fci_64x64x64')]:side[key]['path']=str(args.input_root/path)
 side['metric_query_batch_size']=4096
 if (args.output/'reference_sidecar.json').exists() and json.loads((args.output/'reference_sidecar.json').read_text())!=side:raise ValueError('sidecar changed')
 save_json(args.output/'reference_sidecar.json',side);record['sidecar_sha256']=sha(args.output/'reference_sidecar.json');save_json(dest,record)
 for sub in ('logs','cache','scratch','chunks','executions'):(args.output/sub).mkdir(exist_ok=True)
 for n in config()['resolutions']:
  topology.census(n,args.input_root,args.output)
  t=k.load_context(n,args.input_root);owners,labels=num.select_owners(t.g,t.centers)
  # Additional complete near-wall and radial/aggregate transition owners.
  extra=[int(t.ro[np.ravel_multi_index((i,n//3,n//4),(n,)*3)]) for i in (1,n-2,n-3)]
  owners=np.unique(np.r_[owners,extra]);raw=np.flatnonzero(np.isin(t.ro,owners))
  with np.load(args.output/f'N{n}.topology.npz') as z:faces=z['face_ids'][np.isin(z['endpoints'],owners).any(axis=1)]
  save_npz(args.output/f'N{n}.selection.npz',owners=owners,raw=raw,faces=faces)
  save_json(args.output/f'N{n}.selection.json',dict(owners=owners,labels=labels,extra_owners=extra))
 record['topology_hashes']={str(p.name):sha(p) for p in sorted(args.output.glob('N*.topology.npz'))+sorted(args.output.glob('N*.selection.npz'))}
 save_json(dest,record)
 print(json.dumps({'verified':True,'identity':record['identity']}),flush=True)
 return record

def current(args):
 m=json.loads((args.output/'manifest.json').read_text())
 if m['content']['sources']!=sources() or m['content']['config']!=config() or m['content']['commit']!=commit():raise ValueError('source/config/commit changed')
 if sha(args.output/'reference_sidecar.json')!=m['sidecar_sha256']:raise ValueError('sidecar changed')
 if str(args.input_root)!=m['input_root']:raise ValueError('input root changed')
 for name,expected in m['topology_hashes'].items():
  if sha(args.output/name)!=expected:raise ValueError('topology/selection checkpoint changed')
 return m['identity']

def initialize(input_root,output,n):
 output=Path(output);os.environ['DRBX_CACHE_DIR']=str(output/'cache');os.environ['JAX_COMPILATION_CACHE_DIR']=str(output/'cache');os.environ['TMPDIR']=str(output/'scratch')
 t=k.load_context(n,Path(input_root));ref=num.reference(output/'reference_sidecar.json',verify_hashes=False)
 STATE.clear();STATE.update(t=t,ref=ref,S=k.StructuredReconstruction(t),output=output)
 p=output/f'N{n}.observations.npz'
 STATE['values']=np.load(p)['values'] if p.exists() else None

def chunkpath(output,unit):return output/'chunks'/f"N{unit['n']}_{unit['stage']}_{unit['start']:07d}_{unit['stop']:07d}.npz"
def valid(output,unit,identity):
 p=chunkpath(output,unit);receipt=p.with_suffix('.json')
 if not p.exists() or not receipt.exists():return False
 try:
  r=json.loads(receipt.read_text());return r['identity']==identity and r['unit']==unit and r['sha256']==sha(p)
 except (KeyError,ValueError,OSError):return False

def work(unit,identity):
 st=time.monotonic();t=STATE['t'];ref=STATE['ref'];S=STATE['S'];values=STATE['values'];out=STATE['output'];stage=unit['stage'];n=t.n
 if stage=='observations':ids=np.arange(unit['start'],unit['stop']);data=k.observations(t,ref,ids)
 else:
  if stage.startswith('face'):
   with np.load(out/f'N{n}.topology.npz') as z:ids=z['face_ids'][unit['start']:unit['stop']]
   order=3 if stage=='faces' else 5;data=k.face_chunk(t,ref,S,values,ids,order=order,candidate=stage=='faces')
  elif stage.startswith('cell'):
   ids=np.arange(unit['start'],unit['stop']);order=3 if stage=='cells' else 5;data=k.cell_chunk(t,ref,S,values,ids,order=order,candidate=stage=='cells')
  elif stage=='preflight' or stage=='controls':
   with np.load(out/f'N{n}.selection.npz') as z:owner=int(z['owners'][unit['start']])
   raw=np.flatnonzero(t.ro==owner)
   with np.load(out/f'N{n}.topology.npz') as z:
    sel=np.any(z['endpoints']==owner,axis=1);ids=z['face_ids'][sel];ep=z['endpoints'][sel]
   if stage=='preflight':
    # Only the local donor closure is observed; never use zeros for participating owners.
    donors=set();keys=topology.decode(n,ids);p,_=num.quadrature(t.faces,keys,3,face=True)
    for key,points in zip(keys,p):
     row=S.rows(key,points);donors.update(map(int,row.donor_ids))
     if not (key[0]==0 and key[1]==0):
      for side in S.side_rows(key,points):
       if side is not None:donors.update(map(int,side.donor_ids))
    keys=np.array(np.unravel_index(raw,(n,)*3)).T;p,_=num.quadrature(t.faces,keys,3,face=False)
    for key,points in zip(keys,p):donors.update(map(int,S.rows(key,points,'cell').donor_ids))
    donors.add(owner);rawdon=np.flatnonzero(np.isin(t.ro,list(donors)));values=np.zeros((len(t.vol),len(k.FIELDS)))
    for start in range(0,len(rawdon),2048):
     z=k.observations(t,ref,rawdon[start:start+2048]);np.add.at(values,z['owners'],z['numerator'])
    values/=t.vol[:,None]
    f=k.face_chunk(t,ref,S,values,ids);c=k.cell_chunk(t,ref,S,values,raw)
    error=max(f['constant_error'].max(),c['constant_error'].max());residual=max(f['support_residual'].max(),c['support_residual'].max())
    if error>1e-8 or residual>1e-9:raise ValueError(('preflight failed',owner,error,residual))
    data={'owner':np.array(owner),'raw':raw,'faces':ids,'endpoints':ep,'constant_error':np.array(error),'support_residual':np.array(residual),'local_action':assemble_subset(t,values,owner,ep,f,c)[0],'upwind_norm':np.array(np.linalg.norm(f['upwind']))}
   else:
    data={'owner':np.array(owner)}
    for q,step,label in [(5,2e-4,'q5'),(7,2e-4,'q7'),(9,2e-4,'q9'),(7,1e-4,'q7_halfstep')]:
     ref.finite_difference_step=step
     f=k.face_chunk(t,ref,S,values,ids,order=q,candidate=False);c=k.cell_chunk(t,ref,S,values,raw,order=q,candidate=False,step=step)
     _,refvals=assemble_subset(t,values,owner,ep,f,c)
     data[label]=refvals
    ref.finite_difference_step=2e-4
  else:raise ValueError(stage)
 for name,a in data.items():
  if np.issubdtype(np.asarray(a).dtype,np.number) and not np.isfinite(a).all():raise ValueError(('nonfinite',stage,name))
 p=chunkpath(out,unit);save_npz(p,**data)
 receipt=dict(identity=identity,unit=unit,sha256=sha(p),seconds=time.monotonic()-st,peak_rss_gib=rss());save_json(p.with_suffix('.json'),receipt)
 return receipt

def assemble_subset(t,values,owner,ep,f,c):
 sign=(ep[:,0]==owner).astype(int)-(ep[:,1]==owner).astype(int)
 forms=[];refs=[]
 for j,(a,b) in enumerate(k.PAIRS):
  primitive=[];oracle=[]
  for kk,(gen,scalar) in enumerate(((a,b),(b,a))):
   primitive.append(np.dot(sign,f['product'][:,j,kk]-values[owner,scalar]*f['generator'][:,gen])-c['correction'][:,j,kk].sum())
   oracle.append(np.dot(sign,f['oracle_product'][:,j,kk]-values[owner,scalar]*f['oracle_generator'][:,gen])-c['oracle_correction'][:,j,kk].sum())
  A,B=primitive[0]/t.vol[owner],-primitive[1]/t.vol[owner];forms.append([A,B,.5*(A+B),A+np.dot(sign,f['upwind'][:,j])/t.vol[owner]])
  refs.append([oracle[0]/c['volume'].sum(),c['direct'][:,j].sum()/c['volume'].sum()])
 return np.array(forms),np.array(refs)

def units(args,stage,n):
 cfg=config()
 if stage=='observations':total=n**3;chunk=cfg['observation_chunk']
 elif stage in ('faces','face_reference'):
  with np.load(args.output/f'N{n}.topology.npz') as z:total=len(z['face_ids'])
  chunk=cfg['face_chunk' if stage=='faces' else 'reference_face_chunk']
 elif stage in ('cells','cell_reference'):total=n**3;chunk=cfg['cell_chunk' if stage=='cells' else 'reference_cell_chunk']
 else:
  with np.load(args.output/f'N{n}.selection.npz') as z:total=len(z['owners'])
  chunk=1
 return [dict(n=n,stage=stage,start=i,stop=min(i+chunk,total)) for i in range(0,total,chunk)]

def workers(args):
 if args.workers is None or args.workers<1:raise ValueError('--workers must be selected by allocation setup')
 count=args.workers
 if args.memory_budget_gib is not None:
  if args.worker_memory_gib is None or args.worker_memory_gib<=0:raise ValueError('memory budget requires positive worker estimate')
  count=min(count,int((args.memory_budget_gib-args.memory_reserve_gib)//args.worker_memory_gib))
 if count<1:raise ValueError('insufficient worker memory budget')
 return count

def execute(args,stage):
 identity=current(args);nw=workers(args)
 for n in config()['resolutions']:
  allunits=units(args,stage,n);todo=[u for u in allunits if not valid(args.output,u,identity)];records=[];start=time.monotonic()
  if todo:
   with ProcessPoolExecutor(max_workers=nw,mp_context=get_context('spawn'),initializer=initialize,initargs=(str(args.input_root),str(args.output),n),max_tasks_per_child=args.max_tasks_per_worker) as pool:
    it=iter(todo);pending={}
    def submit():
     while len(pending)<2*nw:
      unit=next(it,None)
      if unit is None:break
      pending[pool.submit(work,unit,identity)]=unit
    submit()
    while pending:
     done,_=wait(pending,return_when=FIRST_COMPLETED)
     for fut in done:
      unit=pending.pop(fut)
      try:records.append(fut.result())
      except Exception as e:
       save_json(args.output/'failure.json',dict(unit=unit,error=repr(e)));raise
     save_json(args.output/'progress.json',dict(n=n,stage=stage,completed=len(allunits)-len(todo)+len(records),total=len(allunits),unix=time.time()));submit()
  record=dict(n=n,stage=stage,requested_workers=args.workers,effective_workers=nw,memory_budget_gib=args.memory_budget_gib,worker_memory_gib=args.worker_memory_gib,memory_reserve_gib=args.memory_reserve_gib,seconds=time.monotonic()-start,resumed=len(allunits)-len(todo),executed=len(todo),peak_worker_rss_gib=max([r['peak_rss_gib'] for r in records],default=0))
  save_json(args.output/'executions'/f'{time.time_ns()}_{stage}_N{n}.json',record)
  if stage=='observations':
   t=k.load_context(n,args.input_root);v=np.zeros((len(t.vol),len(k.FIELDS)))
   for u in allunits:
    with np.load(chunkpath(args.output,u)) as z:np.add.at(v,z['owners'],z['numerator'])
   save_npz(args.output/f'N{n}.observations.npz',values=v/t.vol[:,None])
  print(json.dumps(record),flush=True)
 save_json(args.output/f'{stage}.complete.json',dict(identity=identity,stage=stage,completed=True))


def reduce(args):
 identity=current(args);summary={}
 for n in config()['resolutions']:
  t=k.load_context(n,args.input_root);O=len(t.vol);P=len(k.PAIRS);F=len(k.FIELDS)
  values=np.load(args.output/f'N{n}.observations.npz')['values']
  flux=np.zeros((O,P,2));oracle=np.zeros_like(flux);gen=np.zeros((O,F));ogen=np.zeros_like(gen);up=np.zeros((O,P));cor=np.zeros_like(flux);ocor=np.zeros_like(flux);direct=np.zeros((O,P));vol=np.zeros(O)
  with np.load(args.output/f'N{n}.topology.npz') as z:faceids=z['face_ids'];endpoints=z['endpoints']
  maxconstant=0.;maxres=0.
  for stage in ('faces','cells','face_reference','cell_reference'):
   for unit in units(args,stage,n):
    if not valid(args.output,unit,identity):raise ValueError(('missing/incompatible unit',unit))
    with np.load(chunkpath(args.output,unit)) as z:
     expected=faceids[unit['start']:unit['stop']] if stage.startswith('face') else np.arange(unit['start'],unit['stop'])
     if not np.array_equal(z['ids'],expected):raise ValueError('coverage mismatch')
     if stage.startswith('face'):
      ep=endpoints[unit['start']:unit['stop']]
      for side,sign in ((0,1),(1,-1)):
       active=ep[:,side]>=0;oid=ep[active,side]
       if stage=='faces':
        np.add.at(flux,oid,sign*z['product'][active]);np.add.at(gen,oid,sign*z['generator'][active]);np.add.at(up,oid,sign*z['upwind'][active])
       else:np.add.at(oracle,oid,sign*z['oracle_product'][active]);np.add.at(ogen,oid,sign*z['oracle_generator'][active])
     elif stage=='cells':np.add.at(cor,z['owners'],z['correction'])
     else:
      np.add.at(ocor,z['owners'],z['oracle_correction']);np.add.at(direct,z['owners'],z['direct']);np.add.at(vol,z['owners'],z['volume'])
     if stage in ('faces','cells'):maxconstant=max(maxconstant,float(z['constant_error'].max()));maxres=max(maxres,float(z['support_residual'].max()))
  if np.any(vol<=0) or maxconstant>1e-8 or maxres>1e-9:raise ValueError('invalid implementation or coverage')
  forms=np.zeros((O,P,4));ref=np.zeros((O,P));ibp=np.zeros((O,P))
  for j,(a,b) in enumerate(k.PAIRS):
   A=(flux[:,j,0]-values[:,b]*gen[:,a]-cor[:,j,0])/t.vol
   B=-(flux[:,j,1]-values[:,a]*gen[:,b]-cor[:,j,1])/t.vol
   forms[:,j]=np.stack((A,B,.5*(A+B),A+up[:,j]/t.vol),axis=1)
   ibp[:,j]=(oracle[:,j,0]-values[:,b]*ogen[:,a]-ocor[:,j,0])/vol
   ref[:,j]=ibp[:,j] if b==1 else direct[:,j]/vol
  err=forms-ref[:,:,None];rms=np.sqrt(np.einsum('o,opf->pf',t.vol,err**2)/t.vol.sum())
  # Independent assembly using face traversal per control owner (no global scatter reuse).
  independent=0.
  with np.load(args.output/f'N{n}.selection.npz') as z:controls=z['owners']
  budgets=[]
  for owner in controls:
   fparts=[];cparts=[]
   for stage,parts in (('faces',fparts),('cells',cparts)):
    for unit in units(args,stage,n):
     if stage=='faces':pick=np.any(endpoints[unit['start']:unit['stop']]==owner,axis=1)
     else:pick=t.ro[unit['start']:unit['stop']]==owner
     if not pick.any():continue
     with np.load(chunkpath(args.output,unit)) as z:parts.append({key:z[key][pick] for key in z.files})
   f={key:np.concatenate([z[key] for z in fparts]) for key in fparts[0]};c={key:np.concatenate([z[key] for z in cparts]) for key in cparts[0]}
   ep=endpoints[np.searchsorted(faceids,f['ids'])];got,_=assemble_subset(t,values,int(owner),ep,f,c);independent=max(independent,float(abs(got-forms[owner]).max()))
   unit=units(args,'controls',n)[int(np.flatnonzero(controls==owner)[0])]
   if not valid(args.output,unit,identity):raise ValueError('missing reference control')
   with np.load(chunkpath(args.output,unit)) as z:budgets.append({key:z[key] for key in z.files})
  if independent>1e-10:raise ValueError(('independent assembly mismatch',independent))
  save_npz(args.output/f'N{n}.result.npz',forms=forms,reference=ref,reference_ibp=ibp,owner_volume=t.vol,continuous_volume=vol,error=err,owner_raw_representative=t.order[t.starts[:-1]])
  save_json(args.output/f'N{n}.reference_controls.json',dict(records=budgets))
  budget_records=[]
  for z in budgets:
   select=np.array([0 if b==1 else 1 for a,b in k.PAIRS]);jj=np.arange(P)
   low=z['q5'][jj,select];middle=z['q7'][jj,select];high=z['q9'][jj,select];half=z['q7_halfstep'][jj,select]
   budget_records.append(abs(middle-low)+abs(high-middle)+abs(half-middle))
  bounded_budget=np.sqrt(np.mean(np.asarray(budget_records)**2,axis=0))
  fractions=bounded_budget[:,None]/np.maximum(rms,1e-300)
  radial=t.pts[t.order[t.starts[:-1]],0];regions={}
  for name,mask in [('axis',radial<1/n),('first_ring',(radial>=1/n)&(radial<2/n)),('wall',radial>1-1/n),('nearwall',radial>1-3/n),('aggregate',t.starts[1:]-t.starts[:-1]>1),('ordinary',t.starts[1:]-t.starts[:-1]==1)]:
   regions[name]=dict(rms=np.sqrt(np.einsum('o,opf->pf',t.vol[mask],err[mask]**2)/t.vol[mask].sum()),maximum=abs(err[mask]).max(axis=0))
  summary[str(n)]=dict(rms=rms,maximum=abs(err).max(axis=0),regions=regions,constant_reconstruction_error=maxconstant,max_support_residual=maxres,independent_assembly_max=independent,constant_action_max=float(abs(forms[:,5]).max()),bounded_reference_budget=bounded_budget,reference_fraction=fractions)
 orders=[]
 for na,nb in ((32,48),(48,64)):orders.append(np.log(np.maximum(np.asarray(summary[str(na)]['rms']),1e-300)/np.maximum(summary[str(nb)]['rms'],1e-300))/np.log(nb/na))
 primary=np.ones((P,4),bool);primary[5]=False;primary[0,3]=False
 orderpass=all(np.all(o[primary]>=config()['global_order_minimum']) for o in orders)
 implementationpass=all(summary[str(n)]['constant_action_max']<=config()['constant_action_tolerance'] for n in (32,48,64))
 referencepass=all(np.all(np.asarray(summary[str(n)]['reference_fraction'])[primary]<=config()['reference_fraction_maximum']) for n in (32,48,64))
 save_json(args.output/'summary.json',dict(identity=identity,status='computation complete',cases=k.CASES,forms=['A','B','C','U'],results=summary,orders=orders,primary_mask=primary,order_pass=orderpass,reference_pass=referencepass,implementation_pass=implementationpass,qualification_pass=orderpass and referencepass and implementationpass,reference_status='bounded q5/q7/q9 and half-step artifacts returned for local assessment',production_qualified=False))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('verify-inputs','preflight','run','validate','smoke'))
 p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int);p.add_argument('--memory-budget-gib',type=float);p.add_argument('--worker-memory-gib',type=float);p.add_argument('--memory-reserve-gib',type=float,default=1);p.add_argument('--max-tasks-per-worker',type=int,default=64);p.add_argument('--resolution',type=int,default=32);p.add_argument('--sample-index',type=int,default=0)
 a=p.parse_args();a.input_root=a.input_root.resolve();a.output=a.output.resolve()
 with lock(a.output):
  if a.command=='verify-inputs':verify(a)
  elif a.command=='preflight':execute(a,'preflight')
  elif a.command=='run':
   ident=current(a)
   for n in config()['resolutions']:
    for u in units(a,'preflight',n):
     if not valid(a.output,u,ident):raise ValueError('complete prescribed preflight required')
   for stage in ('observations','faces','cells','face_reference','cell_reference','controls'):execute(a,stage)
   reduce(a)
  elif a.command=='validate':reduce(a)
  else:
   ident=current(a);initialize(str(a.input_root),str(a.output),a.resolution);u=units(a,'preflight',a.resolution)[a.sample_index];print(work(u,ident),flush=True)
if __name__=='__main__':main()
