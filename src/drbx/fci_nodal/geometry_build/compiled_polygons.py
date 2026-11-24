"""Compiled JAX half-plane clipping and centroids in float64 and canonical order.

Roundoff differences accepted by the user. Dynamic counts avoid per-vertex-count
compilations. Capacity is checked, never truncated. The caller supplies the JAX backend and precision.
"""
import numpy as np
CAP=16
BATCH=256

def make_kernel():
 import jax,jax.numpy as j
 def single(poly,count,land,quad,qn,xy):
  overflow=j.array(False)
  def edge(e,state):
   p,n,over=state
   def actual(state):
    p,n,over=state;a=quad[e];b=quad[j.where(e+1<qn,e+1,0)];d=b-a;v=d[0]*(land[:,1]-a[1])-d[1]*(land[:,0]-a[0]);f=p@v;out=j.zeros_like(p)
    def vertex(i,carry):
     out,k,over=carry
     def emit(carry):
      out,k,over=carry;prev=j.where(i==0,n-1,i-1);pa=p[prev];pb=p[i];fa=f[prev];fb=f[i]
      def intersection(state):
       z,k,o=state;left=jax.lax.optimization_barrier(fb*pa);right=jax.lax.optimization_barrier(fa*pb);value=jax.lax.optimization_barrier(left-right)/(fb-fa);return z.at[j.minimum(k,CAP-1)].set(value),k+1,o|(k>=CAP)
      out,k,over=jax.lax.cond((fa>=0)!=(fb>=0),intersection,lambda x:x,(out,k,over))
      out,k,over=jax.lax.cond(fb>=0,lambda x:(x[0].at[j.minimum(x[1],CAP-1)].set(pb),x[1]+1,x[2]|(x[1]>=CAP)),lambda x:x,(out,k,over));return out,k,over
     return jax.lax.cond(i<n,emit,lambda x:x,(out,k,over))
    out,nnew,over=jax.lax.fori_loop(0,CAP,vertex,(out,j.array(0,j.int64),over));nnew=j.where(nnew>=3,nnew,0);return out,nnew,over
   return jax.lax.cond((e<qn)&(n>=3),actual,lambda x:x,(p,n,over))
  poly,count,overflow=jax.lax.fori_loop(0,4,edge,(poly,count,overflow));points=poly@xy;aa=points[1:-1]-points[0];bb=points[2:]-points[0];left=jax.lax.optimization_barrier(aa[:,0]*bb[:,1]);right=jax.lax.optimization_barrier(aa[:,1]*bb[:,0]);w=jax.lax.optimization_barrier(left-right)/2;w=j.where(j.arange(1,CAP-1)<count-1,w,0.);area=w.sum();bary=((poly[0]+poly[1:-1]+poly[2:])*w[:,None]).sum(0)/(3*area);area=j.where((count<3)|(area<=1e-18),0.,area)
  return poly,count,area,bary,overflow
 return jax.jit(jax.vmap(single))

class Polygons:
 def __init__(self,executable=None):
  if executable is not None:
   self.executable=executable;return
  dummy=(np.zeros((BATCH,CAP,3)),np.zeros(BATCH,np.int64),np.zeros((BATCH,3,2)),np.zeros((BATCH,4,2)),np.full(BATCH,4,np.int64),np.zeros((BATCH,3,2)))
  self.executable=make_kernel().lower(*dummy).compile()
 def run(self,polys,counts,lands,quads,qcounts,xys):
  import jax
  n=len(counts);out=[]
  for start in range(0,n,BATCH):
   stop=min(start+BATCH,n);length=stop-start;args=[]
   for a in [polys,counts,lands,quads,qcounts,xys]:
    z=np.zeros((BATCH,)+a.shape[1:],a.dtype);z[:length]=a[start:stop];args.append(z)
   result=jax.device_get(self.executable(*args))
   if result[-1][:length].any():raise ArithmeticError('Compiled polygon capacity exceeded; no vertices discarded')
   out.append(tuple(a[:length] for a in result[:-1]))
  return tuple(np.concatenate([a[k] for a in out]) for k in range(4))
