import numpy as np

from ...geometry_build.clipping_primitives import cross
from ._geometry import TAU

def point_in_segments(x,edges):
 a=edges[:,0];b=edges[:,1];take=(a[:,1]>x[1])!=(b[:,1]>x[1]);e=a[take];f=b[take]
 return bool(np.count_nonzero(x[0]<(f[:,0]-e[:,0])*(x[1]-e[:,1])/(f[:,1]-e[:,1])+e[:,0])%2)

def segment_in_cell(edge,c):
 a,b=edge;d=b-a;r0,r1,t0,t1=c;ts=[0.,1.]
 aa=np.dot(d,d)
 for r in [r0,r1]:
  if aa:
   bb=2*np.dot(a,d);cc=np.dot(a,a)-r*r;disc=bb*bb-4*aa*cc
   if disc>=0:ts.extend([(-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)])
 for t in [t0,t1]:
  v=np.array([np.cos(t),np.sin(t)]);den=cross(d,v)
  if abs(den)>1e-20:ts.append(-cross(a,v)/den)
 ts=np.unique(np.clip(ts,0,1));ts=np.r_[ts,(ts[:-1]+ts[1:])/2]
 q=a+ts[:,None]*d;rho=np.linalg.norm(q,axis=1);theta=(np.arctan2(q[:,1],q[:,0])-t0)%TAU
 return bool(np.any((rho>=r0-2e-12)&(rho<=r1+2e-12)&((theta<=t1-t0+2e-12)|(theta>=TAU-2e-12))))
