"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
import numpy as np


def bind(context):
    gu = context.gu

    def clip(poly,a,b,positive=True):
     if not len(poly):return np.empty((0,2))
     v=gu.cross(b-a,poly-a)*(1 if positive else -1);out=[]
     for x,y,fx,fy in zip(poly,np.roll(poly,-1,axis=0),v,np.roll(v,-1)):
      if fx>=-1e-15:out.append(x)
      if (fx>1e-15 and fy<-1e-15) or (fy>1e-15 and fx<-1e-15):out.append(x+fx/(fx-fy)*(y-x))
     if len(out)<3:return np.empty((0,2))
     out=np.asarray(out);keep=np.linalg.norm(out-np.roll(out,1,axis=0),axis=1)>1e-14;out=out[keep]
     return out if len(out)>=3 else np.empty((0,2))


    def area(p):return abs(np.sum(gu.cross(p[1:-1]-p[0],p[2:]-p[0])))/2 if len(p)>=3 else 0.


    def remainder(child,g):
     if not len(g):return [child]
     inside=child;parts=[]
     for a,b in zip(g,np.roll(g,-1,axis=0)):
      out=clip(inside,a,b,False)
      if area(out)>1e-18:parts.append(out)
      inside=clip(inside,a,b,True)
      if not len(inside):break
     assert abs(sum(area(p) for p in parts)+area(inside)-area(child))<1e-12*max(area(child),1e-12)
     return parts


    return {'clip': clip, 'area': area, 'remainder': remainder}
