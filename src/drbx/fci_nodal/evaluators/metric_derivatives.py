"""Whole-domain actual-point frozen metric/field jacfwd, fixed512 point batches."""
import numpy as np
METHOD='jax_jacfwd'
CAPACITY=512

def make_function(table,volume_only=False):
 import jax
 import jax.numpy as jnp
 from .eta_filtered_field import JaxEtaFilterTable
 spec=(table.max_harmonic,table.nfp,table.nu,table.ntheta,table.u_max)
 def single(q,metric,table):
  pos,A=metric.position_and_jacobian(q[None,:]);pos=pos[0];A=A[0]
  F=table.flux_density(q[None,:])[0];J=jnp.linalg.det(A);B=F/J
  mag=jnp.linalg.norm(A@B);b=B/mag
  return (F if volume_only else jnp.concatenate([b,mag[None],F])),(pos,A,B,mag,J)
 # has_aux shares the actual-point values with the differentiated computation.
 tangent=jax.jacfwd(single,argnums=0,has_aux=True)
 def batch(metric,coefficients,q):
  table=JaxEtaFilterTable(coefficients,*spec)
  # Bounded internal tiles keep second-map derivatives below local RSS caps.
  # One jitted call still owns and returns all512 points, no shifted probes.
  jac,aux=jax.lax.map(lambda point:tangent(point,metric,table),q,batch_size=32)
  pos,A,B,M,J=aux;b=B/M[:,None]
  if volume_only:return dict(pos=pos,A=A,B=B,M=M,J=J,b=b,dF=jac,divB=jnp.trace(jac,axis1=1,axis2=2)/J)
  db=jnp.swapaxes(jac[:,:3,:],1,2);dmag=jac[:,3,:]
  return dict(pos=pos,A=A,B=B,M=M,J=J,b=b,db=db,dmag=dmag,dF=jac[:,4:7,:],along_db=jnp.einsum('pa,pai->pi',b,db),along_logmag=jnp.einsum('pa,pa->p',b,dmag)/M,divB=jnp.trace(jac[:,4:7,:],axis1=1,axis2=2)/J)
 return jax.jit(batch)

def evaluate(ev,points,volume_only=False,*,cache=None):
 if cache is None:cache=ev.__dict__.setdefault('_derivative_functions',{})
 if volume_only not in cache:cache[volume_only]=make_function(ev.table,volume_only)
 fn=cache[volume_only]
 q=np.asarray(points,float).reshape(-1,3);out={}
 for lo in range(0,len(q),CAPACITY):
  x=q[lo:lo+CAPACITY].copy();x[:,1:]%=2*np.pi;n=len(x)
  x=np.r_[x,np.repeat(x[-1:],CAPACITY-n,axis=0)]
  values={k:np.asarray(v)[:n] for k,v in fn(ev.metric,ev.table.coefficients,x).items()}
  invalid=~np.logical_and.reduce([np.isfinite(v).reshape(n,-1).all(1) for v in values.values()])|(values['J']<=0)|(values['M']<=0)
  if invalid.any():
   bad=np.flatnonzero(invalid)
   raise ValueError(dict(error='Invalid frozen field/metric or jacfwd at actual points',metric_derivatives=METHOD,indices=(lo+bad).tolist(),points=[[float(v) if np.isfinite(v) else None for v in point] for point in q[lo+bad[:16]]]))
  for k,v in values.items():out.setdefault(k,[]).append(v)
 if not len(q):return {}
 return {k:np.concatenate(v) for k,v in out.items()}

def volume_metric(ev,q,derivatives=True):
 if not derivatives:
  B,M,A,_=ev(q);return dict(B=B,M=M,J=np.linalg.det(A),b=B/M[:,None])
 values=evaluate(ev,q,True)
 return {k:values[k] for k in ['B','M','J','b','divB']}
