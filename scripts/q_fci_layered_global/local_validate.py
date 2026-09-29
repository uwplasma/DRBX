"""Local extraction replay against workspace research evidence; not used remotely."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'
import argparse,time
from pathlib import Path
import numpy as np
from . import model,storage as io
from .support import Hybrid
from .wall import Wall
from .fields import FIELDS,all_fields
from .worker import boundary_grad

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.workspace);start=time.monotonic();checks=dict(interior_action=0.,wall_action=0.,oracle=0.,reference=0.);count=0
 inner=root/'work/q_fci_rank_outer_comparison_20260929';wall=root/'work/q_fci_layered_dn_qualification_20260929';cases=io.read(inner/'sample.json');walls=io.read(wall/'sample.json');wa=io.read(wall/'actions.json')
 for n in (32,48,64):
  ctx,t=model.context(n,root);state=model.states(t);h=Hybrid(t);old=io.read(inner/f'actions_N{n}.json');geo=dict(np.load(inner/f'geometry_N{n}.npz'));lk={int(x):j for j,x in enumerate(geo['raw'])}
  for z in [x for x in cases if x['N']==n]:
   nn=[]
   for rr in z['raw_members']:
    j=lk[rr];ids,V,D,meta=h.rows(np.array(np.unravel_index(rr,(n,)*3)),geo['slots'][j]);nn.append(geo['L'][j]@np.einsum('si,sid->sd',geo['b'][j],D)@state[ids])
   val=np.einsum('r,raf->af',z['weights'],nn)
   for row in [x for x in old if x['owner']==z['owner'] and x['method']=='ringwise']:
    ai=list(model.ALPHAS).index(row['alpha']);fi=FIELDS.index(row['field']);checks['interior_action']=max(checks['interior_action'],abs(val[ai,fi]-complex(*row['N'])));count+=1
  geo=dict(np.load(wall/f'geometry_N{n}.npz'));lk={int(x):j for j,x in enumerate(geo['raw'])}
  for z in [x for x in walls if x['N']==n]:
   pos=lk[z['raw_members'][0]];w=Wall(ctx,t,z['radial'],z['j'],z['k']);grad=boundary_grad(w,geo['slots'][pos],state[w.ids]);N=np.einsum('as,si,bsif->baf',geo['L'][pos],geo['b'][pos],grad)
   for row in [x for x in wa if x['Ngrid']==n and x['owner']==z['owner'] and x['kind'] in ('D','N')]:
    bi=('D','N').index(row['kind']);ai=list(model.ALPHAS).index(row['alpha']);fi=FIELDS.index(row['field']);checks['wall_action']=max(checks['wall_action'],abs(N[bi,ai,fi]-complex(*row['N'])));count+=1
  (R,db),(Rh,dbh)=model.references(ctx,t.pts[geo['raw']]);checks['reference']=max(checks['reference'],float(abs(R[:,:geo['R'].shape[-1]]-geo['R']).max()),float(abs(Rh[:,:geo['R_half'].shape[-1]]-geo['R_half']).max()))
  exact=all_fields(geo['slots'].reshape(-1,3))[1].reshape(len(geo['raw']),6,3,len(FIELDS));O=np.einsum('ras,rsi,rsif->raf',geo['L'],geo['b'],exact)
  for z in [x for x in wa if x['Ngrid']==n]:
   pos=lk[next(v['raw_members'][0] for v in walls if v['N']==n and v['owner']==z['owner'])];ai=list(model.ALPHAS).index(z['alpha']);fi=FIELDS.index(z['field']);checks['oracle']=max(checks['oracle'],abs(O[pos,ai,fi]-complex(*z['O'])))
  print('replayed',n,checks,flush=True)
 assert max(checks.values())<1e-8,checks
 io.write(a.output,dict(checks=checks,comparisons=count,wall_s=time.monotonic()-start,pass_=True,gpu_execution_tested=False))
if __name__=='__main__':main()
