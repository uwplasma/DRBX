import os,sys,json,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'scripts/hsx_remote_qualification'))
import parallel_runner as runner
import numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--input-root',type=Path,default=repo.parent);parser.add_argument('--output',type=Path,default=repo/'work/hsx_v3_selection_check');args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
runtime,_,numeric=runner._materialize(repo/'scripts/hsx_remote_qualification/configuration.json',repo,args.input_root,out)
results={}
for n in (32,48,64):
 t=time.monotonic();c=numeric.cubic._load_context(Path(runtime['paths']['geometry']),Path(runtime['paths']['baseline']),n)
 count=0;maxdiff=0.;held=[]
 for axis in range(3):
  for k in (0,n//2,n-1):
   keys,points=numeric.cubic._planar_faces(c,axis,k)
   mask=np.flatnonzero((keys[:,0]==24)&(keys[:,1]==23))
   ix=np.unique(np.r_[np.linspace(0,len(points)-1,8,dtype=int),mask]);ix=ix[~((axis==0)&(keys[ix,0]==0))]
   for row in ix:
    point=points[row:row+1]
    a=numeric.cubic._row_batch(c,axis,k,point,exact_query=False)
    b=numeric.cubic._row_batch(c,axis,k,point,exact_query=True)
    if not np.array_equal(a[0],b[0]):raise RuntimeError((n,axis,k,keys[row].tolist(),'indexed mismatch'))
    maxdiff=max(maxdiff,float(np.max(np.abs(a[1]-b[1]))));count+=1
    held.append({'axis':axis,'eta':k,'key':keys[row].tolist(),'donors':a[0].tolist()})
 gradient,donors,counts=numeric._boundary_derivatives(c)
 if not np.all(np.isfinite(gradient)) or not np.all(counts>0):raise RuntimeError('boundary')
 result={'n':n,'rows_checked':count,'weight_max_difference':maxdiff,'boundary_rows':n*n,'boundary_gradient_finite':True,'elapsed_seconds':time.monotonic()-t,'donor_records':held}
 results[str(n)]=result; (out/f'N{n}.selection.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='donor_records'}),flush=True)
(out/'summary.json').write_text(json.dumps({'passed':True,'resolutions':{n:{k:v for k,v in r.items() if k!='donor_records'} for n,r in results.items()}},indent=2))
