"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
import numpy as np
from functools import lru_cache


def bind(context):
    N = context.N
    TAU = context.TAU
    SUB = context.SUB
    RINGS = context.RINGS
    DISK_ANGLES = context.disk_angles
    gu = context.gu

    @lru_cache(maxsize=1)
    def midpoint_mesh():
     radii=np.r_[np.arange(N-RINGS,N)/N,1-.16/N,1-.016/N,1.];th=np.arange(DISK_ANGLES)*TAU/(DISK_ANGLES);xy=(radii[:,None,None]*np.stack([np.cos(th),np.sin(th)],1)[None]).reshape(-1,2);tri=[];nt=len(th)
     for i in range(len(radii)-1):
      for j in range(nt):
       a=i*nt+j;b=(i+1)*nt+j;c=(i+1)*nt+(j+1)%nt;d=i*nt+(j+1)%nt;tri.extend([[a,b,c],[a,c,d]])
     return dict(xy=xy,tri=np.array(tri),radii=radii)


    @lru_cache(maxsize=1)
    def plane_mesh():
     quads,owners,*_=gu.tiles(SUB);cells=gu.build()[1];tri=[];ow=[]
     for q,j in zip(quads,owners):
      if cells[j,0]<(N-RINGS)/N-1e-12:continue
      if cells[j,1]==1:
       a,b=q[1],q[2];rr=[q[[0,3]],np.array([a,b])*(1-.16/N),np.array([a,b])*(1-.016/N),np.array([a,b])]
       for i in range(3):
        aa,dd=rr[i];bb,cc=rr[i+1];tri.extend([[aa,bb,cc],[aa,cc,dd]]);ow.extend([j,j])
      else:
       for i in range(1,len(q)-1):tri.append(q[[0,i,i+1]]);ow.append(j)
     tri=np.array(tri);xy,inv=np.unique(tri.reshape(-1,2),axis=0,return_inverse=True)
     return dict(xy=xy,tri=inv.reshape(-1,3),owner=np.array(ow))


    return {'midpoint_mesh': midpoint_mesh, 'plane_mesh': plane_mesh}
