"""Frozen eight-family interior candidate construction.

Minimal function extraction from the 2026-09-26 common-patch, balanced-donor,
and combined-support research policies. Frozen gates and direct moments remain.
"""
import hashlib,math
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scripts.q_fci_return_campaign import numerics as qnum
from . import qcommon as cm
from .selection import local_effective_rows,FAMILIES
run=SimpleNamespace(cm=cm,base=qnum.base,CONF=(('h4',.25),('h8',.125)))
def plane_participation(ctx,index,ee):
 key=(np.asarray(ee).shape,np.asarray(ee).tobytes())
 cached=index._plane_cache.get(key)
 if cached is not None:return cached
 n=ctx['N'];ns=ee.shape[1];byplane={}
 for block in range(4):
  for sid in range(ns):
   planes,weights=run.cm.eta_planes(ctx,ee[block,sid])
   for plane,w in zip(planes,weights):
    if w!=0:byplane.setdefault(int(plane%n),[]).append((block,sid,float(w),ee[block,sid]))
 index._plane_cache[key]=byplane
 return byplane
def fit_plane(run,ctx,index,q,scale,plane,targets,donors):
 cm=run.cm;donors=np.asarray(donors,int);targets=np.asarray(targets,float)
 if len(donors)<15:return None,'fewer_than_15_owners'
 if len(donors)>81:return None,'owner_cap'
 if np.any(index.plane[donors]!=plane):return None,'cross_plane_owner'
 xy=np.asarray([q[0]*np.cos(q[1]),q[0]*np.sin(q[1])]);sc=np.asarray([scale,scale]);exps=cm.EXPS4
 P=index.local(donors,xy,sc,exps)
 centroid=np.asarray(ctx['model'].centroid)[donors]
 d=np.linalg.norm((centroid-xy)/sc,axis=1);W=1/(1+d*d)**2;B=P.T*W[None,:]
 U,s,Vt=np.linalg.svd(B,full_matrices=False);tol=max(np.finfo(float).eps*max(B.shape),1e-10)*s[0]
 rank=int(np.count_nonzero(s>tol));cond=float(s[0]/s[-1]) if s[-1]>0 else float('inf')
 if rank!=15:return None,f'rank_{rank}'
 if not np.isfinite(cond) or cond>1e8:return None,f'condition_{cond}'
 targetxy=np.column_stack((targets[:,0]*np.cos(targets[:,1]),targets[:,0]*np.sin(targets[:,1])))
 z=(targetxy-xy)/sc;phi=np.column_stack([z[:,0]**a*z[:,1]**b for a,b in exps])
 coeff=(W[:,None]*(Vt[:rank].T@((U[:,:rank].T@phi.T)/s[:rank,None]))).T
 repro=np.max(abs(P.T@coeff.T-phi.T),axis=0);constant=abs(np.sum(coeff,axis=1)-1)
 try:clear,*_=cm.hull_metrics(index,donors,q,targets)
 except Exception as exc:return None,'hull_'+type(exc).__name__
 if np.max(repro)>3e-10:return None,f'reproduction_{np.max(repro)}'
 if np.max(constant)>3e-10:return None,f'constant_{np.max(constant)}'
 if np.min(clear)<-1e-12:return None,f'hull_clearance_{np.min(clear)}'
 if not np.all(np.isfinite(coeff)):return None,'nonfinite_coefficients'
 cells=np.concatenate([index.members(int(x)) for x in donors])
 member_xy=np.column_stack((index.x[cells],index.y[cells]));radius=float(np.max(np.linalg.norm((member_xy-xy)/sc,axis=1)))
 return {'donors':donors,'coeff':coeff,'rank':rank,'condition':cond,'reproduction':float(np.max(repro)),
  'constant':float(np.max(constant)),'hull_clearance':float(np.min(clear)),
  'radius':radius,'row_l1':float(np.max(np.sum(abs(coeff),axis=1))),
  'row_l2':float(np.max(np.linalg.norm(coeff,axis=1))),
  'moment_matrix_sha256':hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest(),
  'weight_sha256':hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()},'admitted'


def effective_rows(run,ctx,p,M,owner):
 return local_effective_rows(ctx,p,M,owner)
class common_policy:
 fit_plane=staticmethod(fit_plane)
 effective_rows=staticmethod(effective_rows)
def rectangle(index,i,j,plane,hr,ht):
 n=index.shape[0];rr=np.arange(max(0,i-hr),min(n-1,i+hr)+1);tt=(j+np.arange(-ht,ht+1))%n
 cells=index.labels.reshape(index.shape)[np.ix_(rr,tt,[plane])]
 return np.unique(cells.ravel())
def donors(index,q,ijk,plane):
 key=(tuple(map(int,ijk)),int(plane))
 cached=index._donor_sets.get(key)
 if cached is not None:return cached
 i,j=ijk[:2];rect={name:rectangle(index,i,j,plane,hr,ht) for name,hr,ht in
                   (('rect_2_3',2,3),('rect_3_3',3,3),('rect_4_4',4,4))}
 pool=rectangle(index,i,j,plane,6,6);k=len(rect['rect_4_4']);assert set(rect['rect_4_4']).issubset(set(pool)) and len(pool)>=k
 xy=np.asarray([q[0]*np.cos(q[1]),q[0]*np.sin(q[1])]);cent=np.asarray(index.ctx['model'].centroid)[pool]
 dist=np.linalg.norm(cent-xy,axis=1);order=np.lexsort((pool,dist));rect['nearest_count_4_4']=np.sort(pool[order[:k]])
 index._donor_sets[key]=(rect,pool)
 return rect,pool
def direct_moment(index,donors,xy,scale):
 out=np.empty((len(donors),len(run.cm.EXPS4)))
 for row,owner in enumerate(donors):
  key=(int(owner),np.asarray(xy,float).tobytes(),float(scale))
  cached=index._direct_cache.get(key)
  if cached is None:
   members=index.members(int(owner));v=index.volume[members];x=(index.x[members]-xy[0])/scale;y=(index.y[members]-xy[1])/scale
   cached=np.asarray([math.fsum(float(vv)*float(xx)**a*float(yy)**b for vv,xx,yy in zip(v,x,y))/math.fsum(map(float,v)) for a,b in run.cm.EXPS4])
   index._direct_cache[key]=cached
  out[row]=cached
 return out
def candidate(ctx,index,q,p,ee,owner,ijk,family):
 n=ctx['N'];ns=len(p);grid=ctx['artifact'].geometry.grid;du=float(grid.x.centers[1]-grid.x.centers[0]);scale=max(du,float(q[0])*2*np.pi/n)
 byplane=plane_participation(ctx,index,ee)
 trip=[[],[],[]];planes=[];fail=[];audit=[]
 for plane,used in sorted(byplane.items()):
  sets,pool=donors(index,q,ijk,plane);ds=sets[family];targets=np.asarray([x[3] for x in used]);fit,reason=common_policy.fit_plane(run,ctx,index,q,scale,plane,targets,ds)
  xy=np.asarray([q[0]*np.cos(q[1]),q[0]*np.sin(q[1])]);Pcached=index.local(ds,xy,np.asarray([scale,scale]),run.cm.EXPS4);Pdirect=direct_moment(index,ds,xy,scale)
  momerr=float(np.max(abs(Pcached-Pdirect)));assert momerr<1e-9
  footprint=index.footprint(ds);rcols=footprint['radial_layers'];tcols=footprint['angular_columns']
  row={'plane':plane,'family':family,'donor_count':len(ds),'donor_ids':ds.tolist(),'pool_count':len(pool),'raw_rectangle_ijk':ijk,
       'radial_layers':rcols,'angular_columns':tcols,'min_radial_layer':min(rcols),'max_radial_layer':max(rcols),
       'angular_column_count':len(tcols),'aggregate_owner_count':footprint['aggregate_owners'],
       'max_direct_moment_defect':momerr,'admitted':fit is not None,'rejection':None if fit is not None else reason}
  if fit is None:fail.append({'plane':plane,'reason':reason,'donors':len(ds)});planes.append(row);continue
  row.update(rank=fit['rank'],condition=fit['condition'],reproduction=fit['reproduction'],constant=fit['constant'],hull_clearance=fit['hull_clearance'],
             member_radius_scaled=fit['radius'],endpoint_row_l1=fit['row_l1'],endpoint_row_l2=fit['row_l2'],
             moment_matrix_sha256=fit['moment_matrix_sha256'],weight_sha256=fit['weight_sha256'])
  # Independent degree 0..4 monomial endpoint recovery from direct true moments.
  z=(np.column_stack((targets[:,0]*np.cos(targets[:,1]),targets[:,0]*np.sin(targets[:,1])))-xy)/scale
  phi=np.column_stack([z[:,0]**a*z[:,1]**b for a,b in run.cm.EXPS4]);err=float(np.max(abs(Pdirect.T@fit['coeff'].T-phi.T)))
  row['direct_monomial_reproduction']=err
  row['direct_reproduction_exceeds_frozen_gate']=bool(err>3e-10)
  planes.append(row);audit.append((momerr,err))
  for k,(block,sid,w,_) in enumerate(used):
   trip[0].extend([block*ns+sid]*len(ds));trip[1].extend(ds.tolist());trip[2].extend((w*fit['coeff'][k]).tolist())
 if fail:return None,planes,fail,None
 M=sparse.csr_matrix((trip[2],(trip[0],trip[1])),shape=(4*ns,len(ctx['volume'])));M.sum_duplicates();M.sort_indices()
 defect=float(np.max(abs(M@np.ones(M.shape[1])-1)))
 if defect>3e-10:return None,planes,[{'reason':'map_constant_defect','value':defect}],None
 rows=common_policy.effective_rows(run,ctx,p,M,owner)
 A=max(sum(x['l1'] for x in rows if x['config']==c) for c in (0,1))
 score={'A':A,'max_condition':max(x['condition'] for x in planes),'max_reproduction':max(x['reproduction'] for x in planes),
        'max_moment_defect':max(x['max_direct_moment_defect'] for x in planes),
        'max_direct_monomial_reproduction':max(x['direct_monomial_reproduction'] for x in planes),
        'max_endpoint_row_l1':max(x['endpoint_row_l1'] for x in planes),'max_endpoint_row_l2':max(x['endpoint_row_l2'] for x in planes),
        'max_member_radius_scaled':max(x['member_radius_scaled'] for x in planes),'summed_donor_count':sum(x['donor_count'] for x in planes),
        'min_hull_clearance':min(x['hull_clearance'] for x in planes),'max_map_constant_defect':defect,'effective_rows':rows}
 return M,planes,[],score

class balance:
 candidate=staticmethod(candidate)
 direct_moment=staticmethod(direct_moment)
def full_candidate(ctx,index,q,p,ee,owner,family):
 n=ctx['N'];ns=len(p);grid=ctx['artifact'].geometry.grid;du=float(grid.x.centers[1]-grid.x.centers[0]);scale=max(du,float(q[0])*2*np.pi/n)
 byplane=plane_participation(ctx,index,ee)
 trip=[[],[],[]];planes=[];fail=[]
 K=int(family[1:]);labels=index.labels.reshape(index.shape);xy=np.asarray([q[0]*np.cos(q[1]),q[0]*np.sin(q[1])])
 for plane,used in sorted(byplane.items()):
  donors=np.unique(labels[:K,:,plane].ravel());targets=np.asarray([x[3] for x in used]);fit,reason=common_policy.fit_plane(run,ctx,index,q,scale,plane,targets,donors)
  footprint=index.footprint(donors)
  row={'plane':plane,'family':family,'donor_count':len(donors),'donor_ids':donors.tolist(),
       'radial_layers':footprint['radial_layers'],'angular_columns':footprint['angular_columns'],
       'aggregate_owner_count':footprint['aggregate_owners'],'cached_admitted':fit is not None,'cached_rejection':None if fit is not None else reason}
  if fit is None:fail.append({'plane':plane,'reason':reason,'donors':len(donors)});planes.append(row);continue
  Pcached=index.local(donors,xy,np.asarray([scale,scale]),run.cm.EXPS4);Pdirect=balance.direct_moment(index,donors,xy,scale);moment=float(np.max(abs(Pcached-Pdirect)))
  z=(np.column_stack((targets[:,0]*np.cos(targets[:,1]),targets[:,0]*np.sin(targets[:,1])))-xy)/scale
  phi=np.column_stack([z[:,0]**a*z[:,1]**b for a,b in run.cm.EXPS4]);direct=float(np.max(abs(Pdirect.T@fit['coeff'].T-phi.T)))
  row.update(condition=fit['condition'],cached_reproduction=fit['reproduction'],direct_reproduction=direct,direct_moment_defect=moment,
   constant=fit['constant'],hull_clearance=fit['hull_clearance'],radius=fit['radius'],endpoint_row_l1=fit['row_l1'],endpoint_row_l2=fit['row_l2'],
   moment_matrix_sha256=fit['moment_matrix_sha256'])
  if direct>3e-10:
   fail.append({'plane':plane,'reason':f'direct_reproduction_{direct}','cached_reproduction':fit['reproduction'],'donors':len(donors)})
   row['direct_admitted']=False;planes.append(row);continue
  row['direct_admitted']=True;planes.append(row)
  for k,(block,sid,w,_) in enumerate(used):
   trip[0].extend([block*ns+sid]*len(donors));trip[1].extend(donors.tolist());trip[2].extend((w*fit['coeff'][k]).tolist())
 if fail:return None,planes,fail,None
 M=sparse.csr_matrix((trip[2],(trip[0],trip[1])),shape=(4*ns,len(ctx['volume'])));M.sum_duplicates();M.sort_indices()
 defect=float(np.max(abs(M@np.ones(M.shape[1])-1)))
 if defect>3e-10:return None,planes,[{'reason':'map_constant_defect','value':defect}],None
 rows=common_policy.effective_rows(run,ctx,p,M,owner);A=max(sum(x['l1'] for x in rows if x['config']==c) for c in (0,1))
 score={'A':A,'max_endpoint_row_l1':max(x['endpoint_row_l1'] for x in planes),
  'max_endpoint_row_l2':max(x['endpoint_row_l2'] for x in planes),'max_member_radius_scaled':max(x['radius'] for x in planes),
  'summed_donor_count':sum(x['donor_count'] for x in planes),'max_condition':max(x['condition'] for x in planes),
  'max_cached_reproduction':max(x['cached_reproduction'] for x in planes),'max_direct_reproduction':max(x['direct_reproduction'] for x in planes),
  'max_direct_moment_defect':max(x['direct_moment_defect'] for x in planes),'max_map_constant_defect':defect,'effective_rows':rows}
 return M,planes,[],score
