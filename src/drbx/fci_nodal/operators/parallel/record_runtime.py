"""One cached JAX path; shared products, wall work and records use one CPU mesh."""

import jax,jax.numpy as jnp
from . import linear_ops as op

def reduce_status(status):
 count,ident,code,state=status;idx=jax.lax.axis_index('records');owner=jax.lax.pmin(jnp.where(count>0,idx,jnp.int32(2147483647)),'records');take=(idx==owner);total=jax.lax.psum(count,'records')
 return total,jnp.where(total>0,jax.lax.psum(jnp.where(take,ident,0),'records'),-1),jax.lax.psum(jnp.where(take,code,0),'records'),jax.lax.psum(jnp.where(take,state,jnp.zeros_like(state)),'records')


@jax.jit
def bw_action(p,B,x):
 return B@x-(p['RT']@((p['mu']*p['sg'])[:,None]*(p['R']@op.gradient(p,x))))/p['H'][:,None]
@jax.jit
def bw_transpose(p,BT,y):
 z=y/p['H'][:,None]
 return BT@y-p['CT']@op.solve(p,p['RT']@((p['mu']*p['sg'])[:,None]*(p['R']@z)))

@jax.jit
def phi_action(p,phi):return jnp.zeros((len(phi),6)).at[:,4].set(p['mu_model']*op.gradient(p,phi[:,None])[:,0])


