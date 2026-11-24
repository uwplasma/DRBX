"""Deterministic expanded PointCubic, variable supports and owner bank."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
POWERS=((1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3))
def basis(xy):return np.stack([xy[...,0]**a*xy[...,1]**b for a,b in POWERS],axis=-1)

EXPANDED_FIT_SUPPORT = 'expand_fit_support'


class PointCubic:
 def __init__(self,layout,allowed,widths,expand_widths):
  self.layout=layout;self.N=layout.N;self.P=layout.P;self.xy=np.c_[layout.node_u*np.cos(layout.node_theta),layout.node_u*np.sin(layout.node_theta)];self.allowed=np.asarray(allowed,bool);self.widths=tuple(widths);self.expand_widths=tuple(expand_widths);self.trees={};self.fits={};self.changed={};self.conditions=[];self.poly_error=0.;self.radii,self.level=np.unique(layout.node_u,return_inverse=True)
 def nearest(self,plane,i,width):
  if plane not in self.trees:
   pool=np.flatnonzero(self.allowed[plane]);self.trees[plane]=(pool,cKDTree(self.xy[pool]))
  pool,tree=self.trees[plane];distance,_=tree.query(self.xy[i],k=min(width,len(pool)));cand=np.asarray(tree.query_ball_point(self.xy[i],float(np.max(distance))+1e-11/self.N),int);dd=np.linalg.norm(self.xy[pool[cand]]-self.xy[i],axis=1)*self.N;ids=pool[cand[np.lexsort((pool[cand],np.round(dd,11)))[:min(width,len(pool))]]];return np.r_[i,ids[ids!=i]]
 def coefficients(self,ids,i):
  delta=(self.xy[ids]-self.xy[i])*self.N;V=basis(delta[1:]);wt=1/np.maximum(np.linalg.norm(delta[1:],axis=1),1.);sv=np.linalg.svd(V*wt[:,None],compute_uv=False)
  if len(sv)<len(POWERS) or sv[-1]<=sv[0]*1e-12:return None
  C=np.linalg.pinv(V*wt[:,None],rcond=1e-12)*wt[None,:]
  if np.max(abs(C@V-np.eye(len(POWERS)) ))>1e-12:return None
  return C,float(sv[0]/sv[-1])
 def fit(self,owner):
  plane=int(self.layout.plane(owner))%self.layout.symmetry_period_planes;i=int(owner)-int(self.layout.plane_offsets[int(self.layout.plane(owner))]);key=(plane,i)
  if key in self.fits:return self.fits[key]
  if not self.allowed[plane,i]:raise ValueError('Owner outside allowed support')
  old=None
  for width in self.widths:
   ids=self.nearest(plane,i,width);res=self.coefficients(ids,i)
   if res is not None:old=(ids,res[0],res[1]);break
  if old is None:raise ValueError(('No full-rank cubic support',key))
  ids,C,cond=old
  if getattr(self,'rule',EXPANDED_FIT_SUPPORT)==EXPANDED_FIT_SUPPORT and len(np.unique(self.level[ids]))<3:
   for width in self.expand_widths:
    ids=self.nearest(plane,i,width)
    if len(np.unique(self.level[ids]))<3:continue
    res=self.coefficients(ids,i)
    if res is not None:C,cond=res;self.changed[key]=dict(old_support=len(old[0]),support=len(ids),condition=cond);break
   else:raise ValueError(('No radial-diverse cubic support',key))
  self.conditions.append(cond);self.fits[key]=(ids,C);return ids,C

class FitTable:
 def __init__(self,rec,owners):
  self.layout=rec.layout;self.N=rec.N;self.P=rec.P;self.keys=np.unique(np.asarray(owners,int)%self.layout.period_nodes);fits=[rec.fit(int(k)) for k in self.keys];self.width=max(len(i) for i,c in fits);nk=len(fits)
  self.index=np.full(self.layout.period_nodes,-1,np.int64);self.index[self.keys]=np.arange(nk);self.ids=np.zeros((nk,self.width),np.int64);self.ncol=np.array([len(i) for i,c in fits]);self.C=np.zeros((nk,len(POWERS),self.width-1));self.Vt=np.zeros((nk,self.width-1,len(POWERS)));self.center=rec.xy[self.keys%self.P];self.poly_error=0.
  for k,(ids,C) in enumerate(fits):self.ids[k,:len(ids)]=ids;self.C[k,:,:len(ids)-1]=C;self.Vt[k,:len(ids)-1]=basis((rec.xy[ids[1:]]-self.center[k])*self.N)
 def evaluate(self,owners,feet):
  owners=np.asarray(owners,np.int64);k=self.index[owners%self.layout.period_nodes]
  if np.any(k<0):raise ValueError('Owner without fit')
  d=basis((feet-self.center[k])*self.N);v=np.matmul(d[:,None,:],self.C[k])[:,0,:];chk=np.matmul(v[:,None,:],self.Vt[k])[:,0,:];err=np.abs(chk-d).max(axis=1)/np.maximum(np.abs(d).max(axis=1),1.);self.poly_error=max(self.poly_error,float(err.max(initial=0)))
  if self.poly_error>1e-12:raise ArithmeticError(('Cubic polynomial exactness',self.poly_error))
  val=np.c_[1-v.sum(1),v];valid=np.arange(self.width)[None,:]<self.ncol[k,None];val[~valid]=0;plane=self.layout.plane(owners);col=self.layout.plane_offsets[plane,None]+self.ids[k];col[~valid]=0
  return k,val,col,valid
 def rows(self,owners,feet,nstates):
  k,val,col,valid=self.evaluate(owners,feet);n=len(owners);out=sparse.csr_matrix((val.ravel(),col.ravel().astype(np.int32 if nstates<2**31 else np.int64),np.arange(n+1,dtype=np.int64)*self.width),shape=(n,nstates));out.eliminate_zeros();return out
 def bank(self):
  ptr=np.r_[0,np.cumsum(self.ncol)];ids=np.concatenate([self.ids[k,:nc] for k,nc in enumerate(self.ncol)]);return dict(keys=self.keys,index=self.index,indptr=ptr,indices=ids,plane_offsets=self.layout.plane_offsets,period_nodes=self.layout.period_nodes)
