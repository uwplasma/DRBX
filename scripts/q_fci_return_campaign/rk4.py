"""Compiled CPU RK4 in regular x/y, with RK4 connection length and seed masks."""
from functools import partial
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=('steps',))
def advance(field, seeds, delta, *, steps):
    eta0=seeds[0,2]
    state=jnp.stack((seeds[:,0]*jnp.cos(seeds[:,1]),seeds[:,0]*jnp.sin(seeds[:,1]),jnp.zeros(len(seeds))),axis=1)
    alive=jnp.ones(len(seeds),dtype=bool)
    bad=jnp.zeros(len(seeds),dtype=bool)
    h=delta/steps
    def rhs(a,eta,active):
        r=jnp.linalg.norm(a[:,:2],axis=1)
        inside=jnp.isfinite(a).all(axis=1)&(r>0)&(r<1)
        use=active&inside
        # Inactive/outside rows are evaluated at an interior dummy point only.
        p=jnp.stack((jnp.where(use,r,.5),jnp.where(use,jnp.mod(jnp.arctan2(a[:,1],a[:,0]),2*jnp.pi),0.),jnp.full(len(seeds),jnp.mod(eta,2*jnp.pi))),axis=1)
        B,Bmag=field(p);beta=B[:,2]/Bmag
        bad_field=use&((beta<=1e-10)|(~jnp.isfinite(B).all(axis=1))|(~jnp.isfinite(Bmag)))
        safe_beta=jnp.where(bad_field|(~use),1.,beta)
        b=B/Bmag[:,None];c=jnp.cos(p[:,1]);s=jnp.sin(p[:,1])
        out=jnp.stack(((c*b[:,0]-r*s*b[:,1])/safe_beta,(s*b[:,0]+r*c*b[:,1])/safe_beta,1/safe_beta),axis=1)
        return jnp.where((use&(~bad_field))[:,None],out,0.),use&(~bad_field),bad_field
    def step(i,carry):
        a,live,error=carry;t=eta0+i*h
        k1,v1,e1=rhs(a,t,live)
        k2,v2,e2=rhs(a+.5*h*k1,t+.5*h,v1)
        k3,v3,e3=rhs(a+.5*h*k2,t+.5*h,v2)
        k4,v4,e4=rhs(a+h*k3,t+h,v3)
        candidate=a+h*(k1+2*k2+2*k3+k4)/6
        radius2=jnp.sum(candidate[:,:2]**2,axis=1)
        valid=v4&jnp.isfinite(candidate).all(axis=1)&(radius2>0)&(radius2<1)
        return jnp.where(valid[:,None],candidate,a),valid,error|e1|e2|e3|e4
    state,alive,bad=jax.lax.fori_loop(0,steps,step,(state,alive,bad))
    end=jnp.stack((jnp.linalg.norm(state[:,:2],axis=1),jnp.mod(jnp.arctan2(state[:,1],state[:,0]),2*jnp.pi),jnp.full(len(seeds),jnp.mod(eta0+delta,2*jnp.pi))),axis=1)
    return end,jnp.abs(state[:,2]),alive,bad
