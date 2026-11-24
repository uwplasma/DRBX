"""Compiled float64 dense RK4 tracer for supplemental polygon tracing's fixed-field JAX evaluator.

No production-cache access. Same output contract as tracer_dense.trace.
Set single-thread environment before importing JAX (see README).
"""
import numpy as np
import jax
import jax.numpy as jnp
import inspect
# Match the NumPy polynomial companion convention in the reference environment.
_REVERSED="[::-1,::-1]" in "".join(inspect.getsource(np.polynomial.polynomial.polyroots).split())

def _mul(a,b):
 p=a*b
 return jnp.where(jnp.isfinite(p),p,jnp.nan)
def _logical(xy,eta):
 p=xy*xy
 norm=jnp.sqrt(jnp.where(jnp.all(jnp.isfinite(p),axis=-1),jnp.sum(p,axis=-1),jnp.nan))
 return jnp.stack((norm,jnp.arctan2(xy[:,1],xy[:,0])%(2*jnp.pi),eta%(2*jnp.pi)),axis=1)
def _polyval(c,t):
 v=jnp.zeros_like(t)
 for k in range(c.shape[1]-1,-1,-1):v=v*t+c[:,k]
 return v
def _radius_poly(c):
 # Materialize rounded products before summing. The finiteness dependency
 # prevents multiply/add contraction that moves grazing wall roots.
 prod=c[:,:,None,:2]*c[:,None,:,:2];finite=jnp.all(jnp.isfinite(prod),axis=(1,2,3));terms=[]
 for k in range(7):
  dims=[]
  for d in (0,1):
   val=jnp.zeros(c.shape[0],dtype=c.dtype)
   for i in range(max(0,k-3),min(3,k)+1):val=val+prod[:,i,k-i,d]
   dims.append(val)
  terms.append(dims[0]+dims[1])
 return jnp.where(finite[:,None],jnp.stack(terms,axis=1),jnp.nan)
def _roots(c):
 """Vectorized companion roots grouped by exact polynomial degree.

 Match NumPy's trimmed companion size: padding the eigensystem itself
 perturbs multiple/tangent roots, so only returned root arrays are padded.
 """
 n=c.shape[1]-1;degree=jnp.max(jnp.where(c!=0,jnp.arange(n+1)[None,:],0),axis=1);out=jnp.full((len(c),n),jnp.nan+0j,jnp.complex128)
 for d in range(1,n+1):
  use=degree==d
  def solve(old):
   cc=jnp.where(use[:,None],c[:,:d+1],jnp.asarray([-2.]+[0.]*(d-1)+[1.]))
   if d==1:rr=(-cc[:,0]/cc[:,1])[:,None].astype(jnp.complex128)
   else:
    mat=jnp.broadcast_to(jnp.eye(d,k=-1),(len(c),d,d));mat=mat.at[:,:,-1].add(-cc[:,:d]/cc[:,d,None]);rr=jnp.linalg.eigvals(mat[:,::-1,::-1] if _REVERSED else mat)
   padded=jnp.pad(rr,((0,0),(0,n-d)),constant_values=jnp.nan);return jnp.where(use[:,None],padded,old)
  out=jax.lax.cond(jnp.any(use),solve,lambda old:old,out)
 return out
def _first(coeff):
 a,d,c,b=jnp.moveaxis(coeff[:,:,:2],1,0);end=a+d+c+b;controls=jnp.stack((a,a+d/3,a+2*d/3+c/3,end),axis=1);candidate=jnp.max(jnp.linalg.norm(controls,axis=2),axis=1)>=1
 poly=_radius_poly(coeff).at[:,0].add(-1.);deriv=poly[:,1:]*jnp.arange(1,7)
 def solve(_):
  safe=jnp.where(candidate[:,None],poly,jnp.array([-2.,1.,0.,0.,0.,0.,0.]));rr=_roots(safe);real=rr.real;valid=(abs(rr.imag)<1e-9)&(real>0)&(real<=1+1e-12)
  slope=jnp.zeros_like(real)
  for k in range(5,-1,-1):slope=slope*real+deriv[:,k,None]
  t=jnp.min(jnp.where(valid&(slope>0),real,jnp.inf),axis=1);valid=candidate&jnp.isfinite(t);t=jnp.minimum(1.,t)
  def refine(i,v):
   residual=_polyval(poly,v);dv=_polyval(deriv,v);return jnp.where(valid&(abs(residual)>5e-15),jnp.clip(v-residual/dv,0.,1.),v)
  t=jax.lax.fori_loop(0,8,refine,t);return jnp.where(valid,t,jnp.nan)
 return jax.lax.cond(jnp.any(candidate),solve,lambda _:jnp.full(coeff.shape[0],jnp.nan),None),poly

def _make(evaluator,steps):
 evaluate=evaluator._evaluate
 def program(seeds,delta,beta_min0,beta_max0,max_u0,six):
  sign=jnp.sign(delta);h=delta/steps;state=jnp.stack((seeds[:,0]*jnp.cos(seeds[:,1]),seeds[:,0]*jnp.sin(seeds[:,1]),jnp.zeros(len(seeds))),axis=1)
  def rhs(state,eta):
   q=_logical(state[:,:2],eta);pos,A,B,mag=evaluate(q);u,theta=q[:,:2].T;c,s=jnp.cos(theta),jnp.sin(theta)
   v=jnp.stack(((_mul(c,B[:,0])-_mul(_mul(u,s),B[:,1]))/B[:,2],(_mul(s,B[:,0])+_mul(_mul(u,c),B[:,1]))/B[:,2],sign*mag/abs(B[:,2])),axis=1)
   finite=jnp.all(jnp.isfinite(pos))&jnp.all(jnp.isfinite(A))&jnp.all(jnp.isfinite(B))&jnp.all(jnp.isfinite(mag))&jnp.all(mag>0)
   code=jnp.where(~finite,1,jnp.where(jnp.any(jnp.linalg.det(A)<=0),2,jnp.where(jnp.any(B[:,2]==0),3,0)))
   return v,(jnp.min(B[:,2]),jnp.max(B[:,2]),jnp.max(q[:,0]),code)
  f,diag=rhs(state,seeds[:,2]);lo=jnp.minimum(beta_min0,diag[0]);hi=jnp.maximum(beta_max0,diag[1]);umax=jnp.maximum(max_u0,diag[2]);error=jnp.where(diag[3]!=0,diag[3],jnp.where(lo*hi<=0,3,0))
  n=len(seeds);hit=jnp.zeros(n,bool);re=jnp.zeros(n,bool);wall=jnp.full((n,3),jnp.nan);ws=wall.copy();first=jnp.full(n,-1,jnp.int64);res=jnp.full(n,jnp.nan);maxu=seeds[:,0]
  initial=(state,h,f,hit,re,wall,ws,first,res,maxu,lo,hi,umax,error)
  def body(carry,step):
   state,h,k1,hit,re,wall,ws,first,res,maxu,lo,hi,umax,error=carry;eta=seeds[:,2]+step*h
   s2=state+_mul(h[:,None],k1)/2;k2,d2=rhs(s2,eta+h/2)
   s3=state+_mul(h[:,None],k2)/2;k3,d3=rhs(s3,eta+h/2)
   s4=state+_mul(h[:,None],k3);k4,d4=rhs(s4,eta+h)
   end=state+_mul(h[:,None],(k1+2*k2+2*k3+k4))/six;fn,dn=rhs(end,eta+h)
   for dd in (d2,d3,d4,dn):
    lo=jnp.minimum(lo,dd[0]);hi=jnp.maximum(hi,dd[1]);umax=jnp.maximum(umax,dd[2]);code=jnp.where(dd[3]!=0,dd[3],jnp.where(lo*hi<=0,3,0));error=jnp.where(error!=0,error,code)
   d=_mul(h[:,None],k1);e=_mul(h[:,None],fn);coeff=jnp.stack((state,d,_mul(3.,(end-state))-2*d-e,2*(state-end)+d+e),axis=1)
   # Invalid evaluator states must raise the reference error, not enter LAPACK with NaNs.
   safe=jnp.where(jnp.isfinite(coeff),coeff,0.);times,poly=_first(safe);fresh=(~hit)&jnp.isfinite(times);tt=jnp.where(fresh,times,0.)
   powers=jnp.stack((jnp.ones_like(tt),tt,tt*tt,tt*tt*tt),axis=1);vals=jnp.einsum('pc,pcd->pd',powers,coeff);wp=_logical(vals[:,:2],eta+tt*h)
   ws=jnp.where(fresh[:,None],vals,ws);wall=jnp.where(fresh[:,None],wp,wall);res=jnp.where(fresh,abs(wp[:,0]-1),res);first=jnp.where(fresh,step,first);error=jnp.where((error==0)&jnp.any(fresh&(res>1e-12)),4,error)
   possible=(hit|fresh)&(~re);start=jnp.where(fresh,times,0.);deriv=poly[:,1:]*jnp.arange(1,7)
   def extrema(_):
    cc=jnp.where(possible[:,None],deriv,jnp.array([1.,0.,0.,0.,0.,0.]));rr=_roots(cc);tt=rr.real;valid=(abs(rr.imag)<1e-9)&(tt>start[:,None]+1e-10)&(tt<1)
    pp=poly.at[:,0].add(1.);vv=jnp.zeros_like(tt)
    for k in range(6,-1,-1):vv=vv*tt+pp[:,k,None]
    endrad=_polyval(pp,jnp.ones(n));return possible&((endrad<1-1e-13)|jnp.any(valid&(vv<1-1e-13),axis=1))
   re=re|jax.lax.cond(jnp.any(possible),extrema,lambda _:jnp.zeros(n,bool),None);hit=hit|fresh
   maxu=jnp.maximum(maxu,jnp.max(jnp.linalg.norm(jnp.stack((state[:,:2],s2[:,:2],s3[:,:2],s4[:,:2],end[:,:2]),axis=1),axis=2),axis=1))
   end=jnp.where(hit[:,None],ws,end);h=jnp.where(hit,0.,h);fn=jnp.where(hit[:,None],0.,fn)
   return (end,h,fn,hit,re,wall,ws,first,res,maxu,lo,hi,umax,error),coeff
  (state,h,f,hit,re,wall,ws,first,res,maxu,lo,hi,umax,error),hist=jax.lax.scan(body,initial,jnp.arange(steps,dtype=jnp.int64))
  endpoint=_logical(state[:,:2],seeds[:,2]+delta);endpoint=jnp.where(hit[:,None],jnp.nan,endpoint);length=jnp.where(hit,jnp.nan,state[:,2])
  def incidence(_):
   q=jnp.where(hit[:,None],wall,seeds);pos,A,B,mag=evaluate(q);inverse=jnp.linalg.inv(A);guu=jnp.sum(inverse[:,0,:]**2,axis=1);bn=B[:,0]/mag/jnp.sqrt(guu);bn=jnp.where(hit,bn,jnp.nan)
   finite=jnp.all(jnp.isfinite(pos))&jnp.all(jnp.isfinite(A))&jnp.all(jnp.isfinite(B))&jnp.all(jnp.isfinite(mag))&jnp.all(mag>0)
   code=jnp.where(~finite,1,jnp.where(jnp.any(jnp.linalg.det(A)<=0),2,jnp.where(jnp.any(B[:,2]==0),3,jnp.where(jnp.any(abs(bn)>1+1e-12),5,0))))
   return bn,jnp.min(B[:,2]),jnp.max(B[:,2]),jnp.max(q[:,0]),code
  bn,blo,bhi,bum,code=jax.lax.cond(jnp.any(hit),incidence,lambda _:(jnp.full(n,jnp.nan),lo,hi,umax,jnp.array(0)),None);lo=jnp.minimum(lo,blo);hi=jnp.maximum(hi,bhi);umax=jnp.maximum(umax,bum);error=jnp.where(error!=0,error,jnp.where(code!=0,code,jnp.where(lo*hi<=0,3,jnp.where(jnp.any((~hit)&(endpoint[:,0]>1)),6,0))))
  return dict(history=jnp.swapaxes(hist,0,1),hit=hit,endpoint=endpoint,length=length,wall=wall,wall_length=ws[:,2],bn=bn,angle=jnp.arcsin(jnp.clip(abs(bn),0.,1.)),reentry=re,first_step=first,wall_residual=res,continuation_max_u=maxu),jnp.stack((lo,hi,umax,error))
 return jax.jit(program)

def compiled(evaluator,batch_size=128,steps=64):
 programs=evaluator.__dict__.setdefault('_trace_programs',{})
 key=int(batch_size),int(steps)
 if key not in programs:
  fn=_make(evaluator,int(steps));args=(jax.ShapeDtypeStruct((batch_size,3),jnp.float64),jax.ShapeDtypeStruct((batch_size,),jnp.float64),*[jax.ShapeDtypeStruct((),jnp.float64)]*4);programs[key]=fn.lower(*args).compile()
 return programs[key]
def trace(evaluator,seeds,delta,steps=64):
 seeds=np.asarray(seeds,float);delta=np.broadcast_to(np.asarray(delta,float),(len(seeds),)).copy();batch=int(getattr(evaluator,'trace_batch_size',128));parts=[]
 for start in range(0,len(seeds),batch):
  q=seeds[start:start+batch];dd=delta[start:start+batch];n=len(q)
  if n<batch:q=np.concatenate((q,np.repeat(q[-1:],batch-n,axis=0)));dd=np.r_[dd,np.repeat(dd[-1],batch-n)]
  result,diag=compiled(evaluator,batch,steps)(q,dd,float(evaluator.beta_min),float(evaluator.beta_max),float(evaluator.max_u),6.);result={k:np.asarray(v)[:n] for k,v in result.items()};lo,hi,um,error=np.asarray(diag);evaluator.beta_min=float(lo);evaluator.beta_max=float(hi);evaluator.max_u=float(um)
  messages={1:'Invalid continuous geometry/field',2:'Nonpositive chart Jacobian',3:'B^eta zero or sign change',4:'Wall root residual',5:'Invalid wall incidence',6:'Non-hit endpoint outside wall'}
  if error:raise ValueError(messages[int(error)])
  parts.append(result)
 if not parts:raise ValueError('need at least one array to concatenate')
 return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
