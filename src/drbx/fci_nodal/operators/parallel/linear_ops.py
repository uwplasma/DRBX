"""Shared JAX linear, M-solve and wall primitives; no application backend."""

import numpy as np
import jax,jax.numpy as jnp

def solve(p,rhs):return p['blocks'](p['H'],rhs)

def gradient(p,x):return solve(p,p['C']@x)

def trace(p,x):return p['R']@x+p['leg'][:,None]*(p['R']@gradient(p,x))

def transpose(p,y):return p['RT']@y+p['CT']@solve(p,p['RT']@(p['leg'][:,None]*y))

def lift(p,y):return (p['RT']@((p['mu']*p['sg'])[:,None]*y))/p['H'][:,None]


@jax.jit
def apply_lift(p,x):
 mu,H=p['mu'],p['H']
 w=mu.reshape((-1,)+(1,)*(np.ndim(x)-1))
 return transpose(p,w*x)/H.reshape((-1,)+(1,)*(np.ndim(x)-1))

@jax.jit
def corrected_center_wall_band(p,U,tw=None,jw=None):
 tau=p['tau'];mu=p['mu_model'];B2=p['B2'];g=gradient(p,U);tg=trace(p,U);tf=tg if tw is None else tw;n,t,i,v,w,o=U.T
 gn,gt,gi,gv,gw,go=tg.T;fn,ft,fi,fv,fw,fo=tf.T
 D=lambda a,aw:(p['D']@a[:,None]+lift(p,aw[:,None]))[:,0]
 G=lambda a,aw:(p['G']@a[:,None]+lift(p,aw[:,None]-p['R']@a[:,None]))[:,0]
 gp=G(n*(t+tau*i),gn*(gt+tau*gi));gpe=G(n*t,gn*gt)
 dj=D(n*(v-w),fn*(fv-fw) if jw is None else jw);dv=D(v,fv);dw=D(w,fw)
 y=[-D(n*w,fn*fw),-w*g[:,1]+2*t/(3*n)*(.71*dj-n*dw),-v*g[:,2]+2*i/(3*n)*(dj-n*dv),-v*g[:,3]-gp/n,-w*g[:,4]-mu*gpe/n-.71*mu*g[:,1]-mu*tau*gradient(p,(n*i)[:,None])[:,0],-v*g[:,5]+B2*dj/n]
 for a,b in ((1,4),(2,3),(3,3),(4,4),(5,3)):y[a]+=.5*(U[:,b]*g[:,a]-D(U[:,b]*U[:,a],tf[:,b]*tf[:,a])+U[:,a]*(dv if b==3 else dw))
 return jnp.stack(y,1)


