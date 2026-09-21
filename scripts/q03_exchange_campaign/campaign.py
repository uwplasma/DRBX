#!/usr/bin/env python3
"""Computation-only, restartable Q03 whole-support HSX campaign."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,concurrent.futures as cf,fcntl,json,platform,sys,time,traceback
from pathlib import Path
import numpy as np
from common import SCHEMA,POLICY,sha,atomic_json,source_identity,load,args_for
from numerics import FIELDS,OBS,fit
from optimized_exchange import selector

STATE=None
def identity(inputs):
 return {'schema':SCHEMA,'policy':POLICY,'input_manifest_sha256':sha(Path(inputs)/'manifest.json'),'source':source_identity(),'numpy':np.__version__,'python':platform.python_version(),'machine':platform.machine()}

def verify(inputs,out):
 inputs=Path(inputs);out=Path(out);manifest=json.loads((inputs/'manifest.json').read_text())
 if manifest['schema']!=SCHEMA or manifest['policy']!=POLICY:raise RuntimeError('input schema/policy mismatch')
 declared=json.loads((Path(__file__).parent/'source_manifest.json').read_text())
 if source_identity()!=declared:raise RuntimeError('source manifest mismatch')
 for n,entry in manifest['resolutions'].items():
  root=inputs/f'N{n}';p=root/'manifest.json'
  if sha(p)!=entry['manifest_sha256']:raise RuntimeError(f'resolution manifest mismatch: {n}')
  m=json.loads(p.read_text())
  for name,meta in m['files'].items():
   file=root/name
   if file.stat().st_size!=meta['bytes'] or sha(file)!=meta['sha256']:raise RuntimeError(f'input mismatch: {file}')
 ident=identity(inputs);atomic_json(out/'input_verification.json',{'status':'passed','identity':ident,'input_root':str(inputs.resolve()),'time':time.time()});return ident

def initialize(inputs,N,ident):
 global STATE
 STATE=(load(inputs,N),N,ident)

def receipt_valid(path,ident,ids):
 path=Path(path);rp=path.with_suffix('.receipt.json')
 if not path.exists() or not rp.exists():return False
 r=json.loads(rp.read_text())
 if r['identity']!=ident or r['faces']!=list(map(int,ids)):raise RuntimeError(f'incompatible checkpoint: {path}')
 if sha(path)!=r['sha256']:raise RuntimeError(f'corrupt checkpoint: {path}')
 with np.load(path) as a:
  if not np.array_equal(a['faces'],ids) or a['indptr'][-1]!=len(a['support']) or len(a['support'])!=len(a['coefficient']):raise RuntimeError(f'incomplete checkpoint: {path}')
 return True

def compute(job):
 ids,path=job;path=Path(path);d,N,ident=STATE;started=time.perf_counter();ids=np.asarray(ids,int)
 if receipt_valid(path,ident,ids):return {'path':str(path),'reused':True,'faces':len(ids)}
 ptr=[0];ss=[];cc=[];diagnostics=[];flux=np.empty((len(ids),len(FIELDS),len(OBS)))
 for i,face in enumerate(ids):
  if d['face.boundary'][face]:
   support=np.empty(0,dtype=np.int32);coef=np.empty(0);diag=[0,0.,0.,0.,0.,0.,0,0.,0.,0]
   for j,f in enumerate(FIELDS):flux[i,j,:]=d['ref_flux.'+f][face]
  else:
   args=args_for(d,face,N);support,meta=selector(*args,record_swaps=False);coef,fitdiag=fit(args[0],support,args[4],args[5],args[6])
   if fitdiag['rank']!=19 or not fitdiag['usable']:raise RuntimeError(f'unusable fit N{N} face {face}: {fitdiag}')
   diag=[fitdiag['rank'],fitdiag['condition'],fitdiag['residual'],fitdiag['weighted_norm'],meta['ratio'],meta['accepted_exchanges'],meta['stop']=='cap',fitdiag['compatibility'],fitdiag['roundoff_indicator'],meta['scalar_ranking_fallbacks']]
   for j,f in enumerate(FIELDS):
    for k,o in enumerate(OBS):flux[i,j,k]=coef@d[f'row.{f}.{o}'][support]
  ss.append(np.asarray(support,np.int32));cc.append(coef);ptr.append(ptr[-1]+len(support));diagnostics.append(diag)
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,faces=ids,indptr=np.asarray(ptr,np.int64),support=np.concatenate(ss),coefficient=np.concatenate(cc),flux=flux,diagnostics=np.asarray(diagnostics))
 os.replace(tmp,path)
 atomic_json(path.with_suffix('.receipt.json'),{'identity':ident,'faces':ids.tolist(),'sha256':sha(path),'elapsed_seconds':time.perf_counter()-started,'pid':os.getpid()})
 return {'path':str(path),'reused':False,'faces':len(ids),'seconds':time.perf_counter()-started}

def execute(inputs,N,ident,jobs,workers,out):
 completed=0;reused_faces=0;computed_faces=0;total_faces=sum(len(ids) for ids,path in jobs);start=time.perf_counter()
 with cf.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(inputs),N,ident)) as pool:
  source=iter(jobs);pending=set()
  for _ in range(2*workers):
   try:pending.add(pool.submit(compute,next(source)))
   except StopIteration:break
  while pending:
   done,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
   for future in done:
    r=future.result();completed+=r['faces'];reused_faces+=r['faces'] if r['reused'] else 0;computed_faces+=0 if r['reused'] else r['faces'];elapsed=time.perf_counter()-start
    atomic_json(Path(out)/'progress.json',{'N':N,'completed_faces':completed,'total_faces':total_faces,'reused_faces':reused_faces,'computed_faces':computed_faces,'elapsed_seconds':elapsed,'estimated_remaining_seconds':(total_faces-completed)*elapsed/computed_faces if computed_faces else None,'last_chunk':r})
    try:pending.add(pool.submit(compute,next(source)))
    except StopIteration:pass

def preflight(inputs,out,workers,ident):
 from oracle import selector as original
 report={'status':'passed','resolutions':{}}
 for N in (32,48,64):
  d=load(inputs,N);m=json.loads((Path(inputs)/f'N{N}/manifest.json').read_text());ids=np.asarray(m['preflight_faces'],int)
  checks={'faces':len(ids),'support_mismatches':0,'coefficient_max_abs':0.,'flux_max_abs':0.}
  for face in ids:
   args=args_for(d,face,N);a,am=selector(*args);b,bm=original(*args)
   if not np.array_equal(a,b) or [(s['out'],s['in']) for s in am['swaps']]!=[(s['out'],s['in']) for s in bm['swaps']]:raise RuntimeError(f'optimized/oracle trajectory mismatch N{N} face {face}')
   ca,da=fit(args[0],a,args[4],args[5],args[6]);cb,db=fit(args[0],b,args[4],args[5],args[6]);checks['coefficient_max_abs']=max(checks['coefficient_max_abs'],float(np.max(abs(ca-cb))))
   if da['rank']!=19 or not da['usable']:raise RuntimeError('preflight unusable fit')
   for f in FIELDS:
    for o in OBS:checks['flux_max_abs']=max(checks['flux_max_abs'],abs(float(ca@d[f'row.{f}.{o}'][a]-cb@d[f'row.{f}.{o}'][b])))
  # Compare complete compact chunk serialization serial versus the worker pool.
  initialize(inputs,N,ident);serial=Path(out)/f'preflight/N{N}/serial.npz';compute((ids,serial))
  jobs=[(part,Path(out)/f'preflight/N{N}/parallel_{i}.npz') for i,part in enumerate(np.array_split(ids,min(workers,len(ids)))) if len(part)]
  execute(inputs,N,ident,jobs,workers,Path(out)/f'preflight/N{N}')
  with np.load(serial) as s:
   cursor=0
   for part,path in jobs:
    with np.load(path) as p:
     for key in ('faces','flux','diagnostics'):
      if not np.array_equal(p[key],s[key][cursor:cursor+len(part)]):raise RuntimeError(f'parallel disagreement: {key}')
     first=int(s['indptr'][cursor]);last=int(s['indptr'][cursor+len(part)])
     if not np.array_equal(p['support'],s['support'][first:last]) or not np.array_equal(p['coefficient'],s['coefficient'][first:last]):raise RuntimeError('parallel coefficient disagreement')
    cursor+=len(part)
  # Reuse one valid temporary chunk; remove and rebuild another only here.
  if not compute((ids,serial))['reused']:raise RuntimeError('resume failed to reuse valid chunk')
  part,path=jobs[-1];before=sha(path);Path(path).unlink();Path(path).with_suffix('.receipt.json').unlink();compute((part,path))
  # NPZ bytes include timestamps; compare payload arrays, not container hashes.
  with np.load(serial) as s,np.load(path) as p:
   if not np.array_equal(p['flux'],s['flux'][-len(part):]):raise RuntimeError('resume rebuild disagreement')
  checks.update({'parallel_equal':True,'resume_reuse_passed':True,'resume_rebuild_passed':True});report['resolutions'][str(N)]=checks
 report['identity']=ident;atomic_json(Path(out)/'preflight.json',report)

def jobs_for(N,d,out,chunk_size):
 nf=len(d['face.minus']);return [(np.arange(start,min(start+chunk_size,nf)),Path(out)/f'N{N}/chunks/chunk_{start:07d}_{min(start+chunk_size,nf)-1:07d}.npz') for start in range(0,nf,chunk_size)]

def assemble(inputs,N,out,ident,chunk_size):
 d=load(inputs,N);nf=len(d['face.minus']);volume=d['volume'];integrated=np.zeros((len(volume),len(FIELDS),len(OBS)));seen=np.zeros(nf,bool);diagnostic=np.empty((nf,10));fluxfile=Path(out)/f'N{N}/face_flux.npy';fluxfile.parent.mkdir(parents=True,exist_ok=True)
 allflux=np.lib.format.open_memmap(fluxfile,mode='w+',dtype=np.float64,shape=(nf,len(FIELDS),len(OBS)))
 for ids,path in jobs_for(N,d,out,chunk_size):
  if not receipt_valid(path,ident,ids):raise RuntimeError(f'missing chunk: {path}')
  with np.load(path) as a:
   ids=a['faces'];f=a['flux']
   if np.any(seen[ids]) or not np.all(np.isfinite(f)):raise RuntimeError('duplicate/nonfinite face data')
   seen[ids]=True;allflux[ids]=f;diagnostic[ids]=a['diagnostics'];minus=d['face.minus'][ids];plus=d['face.plus'][ids]
   np.add.at(integrated,minus,f);valid=plus>=0;np.add.at(integrated,plus[valid],-f[valid])
 if not np.all(seen) or np.any(volume<=0):raise RuntimeError('incomplete faces or invalid volume')
 allflux.flush();del allflux;actions=integrated/volume[:,None,None];arrays={'volume':np.asarray(volume),'diagnostics':diagnostic};metrics={}
 masks={k.split('.',1)[1]:v for k,v in d.items() if k.startswith('region.')}
 if not np.all(sum(np.asarray(v,int) for v in masks.values())==1):raise RuntimeError('region masks do not partition owners')
 for name,mask in masks.items():arrays['region:'+name]=np.asarray(mask)
 for j,f in enumerate(FIELDS):
  arrays['reference:'+f]=np.asarray(d['reference.'+f])
  for k,o in enumerate(OBS):
   key=f'{f}:{o}';value=actions[:,j,k];error=value-d['reference.'+f];arrays['action:'+key]=value;arrays['baseline:'+key]=np.asarray(d[f'baseline.{f}.{o}'])
   sse=float(np.sum(volume*error**2));metrics[key]={'L2':float(np.sqrt(sse/np.sum(volume))),'SSE':sse,'max_abs':float(np.max(abs(error))),'regions':{name:{'volume':float(sum(volume[mask])),'SSE':float(np.sum(volume[mask]*error[mask]**2))} for name,mask in masks.items()}}
 p=Path(out)/f'N{N}/actions.npz';tmp=p.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 os.replace(tmp,p)
 result={'N':N,'identity':ident,'completed':True,'faces':nf,'owners':len(volume),'metrics':metrics,'rank19_interior':bool(np.all(diagnostic[~d['face.boundary'],0]==19)),'max_cubic_residual':float(np.max(diagnostic[:,2])),'diagnostic_columns':['rank','condition','residual','weighted_norm','objective_ratio','exchanges','cap_hit','compatibility','roundoff_indicator','scalar_ranking_fallbacks'],'actions_sha256':sha(p),'face_flux_sha256':sha(fluxfile),'structural_certification':False}
 atomic_json(Path(out)/f'N{N}/summary.json',result);return result

def final_summary(out,ident):
 results={N:json.loads((Path(out)/f'N{N}/summary.json').read_text()) for N in (32,48,64)}
 if any(r['identity']!=ident for r in results.values()):raise RuntimeError('mixed resolution identity')
 orders={}
 for f in FIELDS[:-1]:
  for o in OBS:
   key=f'{f}:{o}';errors=[results[N]['metrics'][key]['L2'] for N in (32,48,64)];values=[float(np.log(errors[i]/errors[i+1])/np.log(b/a)) if errors[i]>0 and errors[i+1]>0 else None for i,(a,b) in enumerate(((32,48),(48,64)))];orders[key]={'errors':errors,'orders':values,'order_gate':all(x is not None and x>=1.8 for x in values)}
 result={'computation_completed':True,'identity':ident,'orders':orders,'G3_operator_order_gate':all(orders[f+':G3']['order_gate'] for f in FIELDS[:-1]),'full_certification':False,'reference_note':'Inherited qualified integrated reference. Compare its uncertainty with new errors during local analysis; order flag alone is not full certification.'}
 atomic_json(Path(out)/'global_summary.json',result)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('verify','preflight','run','validate'));p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int);p.add_argument('--chunk-size',type=int,default=512);a=p.parse_args()
 if a.command in ('preflight','run') and (a.workers is None or a.workers<1):p.error('--workers must be explicitly positive')
 if a.chunk_size<1:p.error('invalid chunk size')
 inputs=Path(a.inputs).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 with (out/'.writer.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();status='failed';error=None
  try:
   ident=verify(inputs,out);contract={'identity':ident,'chunk_size':a.chunk_size}
   if (out/'campaign.json').exists() and json.loads((out/'campaign.json').read_text())!=contract:raise RuntimeError('campaign identity/chunk layout mismatch')
   atomic_json(out/'campaign.json',contract)
   if a.command=='preflight':preflight(inputs,out,a.workers,ident)
   elif a.command=='run':
    receipt=json.loads((out/'preflight.json').read_text())
    if receipt['status']!='passed' or receipt['identity']!=ident:raise RuntimeError('matching preflight required')
    for N in (32,48,64):
     d=load(inputs,N);execute(inputs,N,ident,jobs_for(N,d,out,a.chunk_size),a.workers,out);assemble(inputs,N,out,ident,a.chunk_size)
    final_summary(out,ident)
   elif a.command=='validate':
    for N in (32,48,64):assemble(inputs,N,out,ident,a.chunk_size)
    final_summary(out,ident)
   status='completed'
  except BaseException as e:error=repr(e);raise
  finally:
   atomic_json(out/'operations'/f'{a.command}_{int(start)}.json',{'command':sys.argv,'input_root':str(inputs),'output_root':str(out),'status':status,'exit_code':0 if status=='completed' else 1,'error':error,'workers':a.workers,'start_time':start,'elapsed_seconds':time.time()-start,'pid':os.getpid(),'job_id':os.environ.get('SLURM_JOB_ID')})
 print(json.dumps({'command':a.command,'status':status,'output':str(out)}),flush=True)

if __name__=='__main__':main()
