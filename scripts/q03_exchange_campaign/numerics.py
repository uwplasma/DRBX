"""Frozen HSX observation functionals; NumPy-only remote consumer.

Arithmetic/order copied from the qualified research helpers. Geometry and
candidate pools are immutable exported inputs, not rebuilt on the remote host.
"""
import itertools
from types import SimpleNamespace as NS
import numpy as np

EXPS5=tuple(e for d in range(6) for e in itertools.product(range(d+1),repeat=3) if sum(e)==d)
F_M3=np.asarray([i for i,e in enumerate(EXPS5) if 0<sum(e)<=3],int)
FIELDS=('radial_eta','angular_x','mixed_y_eta','constant')
OBS=('exact','G3')

def regular_coordinates(points,center,scale):
 points=np.asarray(points)
 return np.column_stack(((points[:,0]*np.cos(points[:,1])-center[0])/scale[0],(points[:,0]*np.sin(points[:,1])-center[1])/scale[1],((points[:,2]-center[2]+np.pi)%(2*np.pi)-np.pi)/scale[2]))

def row_moments(rows,indices,center,scale,max_degree):
 exps=[e for e in EXPS5 if sum(e)<=max_degree]
 source=regular_coordinates(rows['source'][indices],center,scale);target=regular_coordinates(rows['target'][indices],center,scale)
 a=np.column_stack([np.prod(target**np.asarray(e)[None,:],axis=1) for e in exps]);b=np.column_stack([np.prod(source**np.asarray(e)[None,:],axis=1) for e in exps])
 return (a-b)/rows['length'][indices,None],np.linalg.norm(.5*(source+target),axis=1)

def distance_weight(distance):return 1./(1.+np.asarray(distance)**2)**1.5
def points(rows,ids,center,scale):return regular_coordinates(rows['source'][ids],center,scale),regular_coordinates(rows['target'][ids],center,scale)
def sector(z):return np.minimum((np.mod(np.arctan2(z[:,1],z[:,0]),2*np.pi)/(np.pi/4)).astype(int),7)

ex=NS(run=NS(compact=NS(row_moments=row_moments,distance_weight=distance_weight),prior=NS(F_M3=F_M3)),an=NS(points=points),directional=NS(sector=sector))

def fit(rows,support,center,scale,target):
 A,distance=row_moments(rows,support,center,scale,5);weight=distance_weight(distance)
 A=A[:,F_M3];target=target[F_M3];B=A.T*weight[None,:]
 U,s,Vt=np.linalg.svd(B,full_matrices=False);tol=np.finfo(float).eps*max(B.shape)*s[0];rank=int(np.count_nonzero(s>tol))
 coefficient=weight*(Vt[:rank].T@((U[:,:rank].T@target)/s[:rank]))
 residual=float(np.linalg.norm(A.T@coefficient-target)/max(np.linalg.norm(target),1e-300))
 projection=U[:,:rank]@(U[:,:rank].T@target);compatibility=float(np.linalg.norm(target-projection)/max(np.linalg.norm(target),1e-300));condition=float(s[0]/s[rank-1]);l1=float(np.sum(abs(coefficient)));roundoff=float(np.finfo(float).eps*condition*max(1.,l1))
 usable=bool(np.all(np.isfinite(coefficient)) and compatibility<=1e-10 and residual<=1e-10 and roundoff<=1e-8)
 return coefficient,{'rank':rank,'condition':condition,'residual':residual,'weighted_norm':float(np.linalg.norm(coefficient/weight)),'compatibility':compatibility,'roundoff_indicator':roundoff,'usable':usable}
