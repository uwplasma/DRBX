"""Research filtered-field RK4 tracer derived from frozen Q tracer.

Only magnetic evaluation and table domain limit differ; all crossing guards retained.

Adapted from the 2026-09-27 Q optimization audit prototype. The same
source-box, Jacobian, beta, first-crossing, reentry, and reach checks remain.
"""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64",True)

@partial(jax.jit, static_argnames=("steps","column",))
def trace(field, table, seeds, deltas, steps, column=False):
    xy = jnp.stack((seeds[:, 0]*jnp.cos(seeds[:, 1]), seeds[:, 0]*jnp.sin(seeds[:, 1])), axis=1)
    h = deltas / steps
    Rlo, Rhi = field.bfield.R[0], field.bfield.R[-1]
    Zlo, Zhi = field.bfield.Z[0], field.bfield.Z[-1]

    fallback_pos, fallback_A = field.metric.position_and_jacobian(jnp.array([[.5, 0., 0.]]))

    def rhs(xy, eta):
        u = jnp.linalg.norm(xy, axis=1)
        theta = jnp.mod(jnp.arctan2(xy[:, 1], xy[:, 0]), 2*jnp.pi)
        q = jnp.stack((u, theta, jnp.mod(eta, 2*jnp.pi)), axis=1)
        pos, A = field.metric.position_and_jacobian(q)
        R = jnp.linalg.norm(pos[:, :2], axis=1)
        Z = pos[:, 2]
        J = jnp.linalg.det(A)
        good = (u>0)&(u<table.u_max)&(R>=Rlo)&(R<=Rhi)&(Z>=Zlo)&(Z<=Zhi)&(J>0)&jnp.isfinite(J)
        safe_pos = jnp.where(good[:, None], pos, fallback_pos)
        safe_A = jnp.where(good[:, None, None], A, fallback_A)
        safe_q = jnp.where(good[:, None], q, jnp.array([.5,0.,0.]))
        F = column_flux(field,safe_q) if column else table.flux_density(safe_q)
        B = F/jnp.linalg.det(safe_A)[:,None]
        cartesian = jnp.einsum('pij,pj->pi',safe_A,B)
        Bmag = jnp.linalg.norm(cartesian, axis=-1)
        b = B/Bmag[:, None]
        beta = b[:, 2]
        good = good & jnp.isfinite(B).all(axis=1) & (beta>1e-10)
        bx = (jnp.cos(theta)*b[:, 0]-u*jnp.sin(theta)*b[:, 1])/beta
        by = (jnp.sin(theta)*b[:, 0]+u*jnp.cos(theta)*b[:, 1])/beta
        v = jnp.stack((bx, by), axis=1)
        return jnp.where(good[:, None], v, 0.), good, u, J

    def step(carry, i):
        xy, valid, crossed, reentered, maxu, minj, firstcross, maxxy, maxeta, firstpoint = carry
        eta = seeds[:, 2]+i*h
        k1, v1, u1, j1 = rhs(xy, eta)
        k2, v2, u2, j2 = rhs(xy+.5*h[:, None]*k1, eta+.5*h)
        k3, v3, u3, j3 = rhs(xy+.5*h[:, None]*k2, eta+.5*h)
        k4, v4, u4, j4 = rhs(xy+h[:, None]*k3, eta+h)
        nextxy = xy+h[:, None]*(k1+2*k2+2*k3+k4)/6
        unext = jnp.linalg.norm(nextxy, axis=1)
        nowcross = (u1>1)|(u2>1)|(u3>1)|(u4>1)|(unext>1)
        newly_crossed=(~crossed)&nowcross
        firstcross = jnp.where((~crossed)&nowcross, i, firstcross)
        reentered = reentered | (crossed & (unext<=1))
        crossed = crossed | nowcross
        stage_xy=jnp.stack((xy,xy+.5*h[:,None]*k1,xy+.5*h[:,None]*k2,xy+h[:,None]*k3,nextxy),axis=1)
        stage_eta=jnp.stack((eta,eta+.5*h,eta+.5*h,eta+h,eta+h),axis=1)
        stage_u=jnp.stack((u1,u2,u3,u4,unext),axis=1)
        first_above=jnp.argmax(stage_u>1,axis=1)
        before=jnp.maximum(first_above-1,0)
        ub=jnp.take_along_axis(stage_u,before[:,None],axis=1)[:,0]
        ua=jnp.take_along_axis(stage_u,first_above[:,None],axis=1)[:,0]
        frac=jnp.clip((1-ub)/jnp.where(ua>ub,ua-ub,1),0,1)
        xyb=jnp.take_along_axis(stage_xy,before[:,None,None],axis=1)[:,0,:]
        xya=jnp.take_along_axis(stage_xy,first_above[:,None,None],axis=1)[:,0,:]
        eb=jnp.take_along_axis(stage_eta,before[:,None],axis=1)[:,0]
        ea=jnp.take_along_axis(stage_eta,first_above[:,None],axis=1)[:,0]
        xyc=xyb+frac[:,None]*(xya-xyb)
        interpolated=jnp.stack((jnp.ones(len(seeds)),jnp.mod(jnp.arctan2(xyc[:,1],xyc[:,0]),2*jnp.pi),eb+frac*(ea-eb)),axis=1)
        firstpoint=jnp.where(newly_crossed[:,None],interpolated,firstpoint)
        argmax=jnp.argmax(stage_u,axis=1)
        stage_max=jnp.take_along_axis(stage_u,argmax[:,None],axis=1)[:,0]
        better=stage_max>maxu
        maxxy=jnp.where(better[:,None],jnp.take_along_axis(stage_xy,argmax[:,None,None],axis=1)[:,0,:],maxxy)
        maxeta=jnp.where(better,jnp.take_along_axis(stage_eta,argmax[:,None],axis=1)[:,0],maxeta)
        maxu=jnp.maximum(maxu,stage_max)
        minj = jnp.minimum(minj, jnp.minimum(jnp.minimum(j1,j2),jnp.minimum(j3,j4)))
        valid = valid & v1 & v2 & v3 & v4 & jnp.isfinite(nextxy).all(axis=1)
        return (nextxy, valid, crossed, reentered, maxu, minj, firstcross, maxxy, maxeta, firstpoint), None

    init = (xy, jnp.ones(len(seeds), bool), jnp.zeros(len(seeds), bool), jnp.zeros(len(seeds), bool), seeds[:, 0], jnp.full(len(seeds), jnp.inf), jnp.full(len(seeds), -1), xy, seeds[:,2],jnp.full((len(seeds),3),jnp.nan))
    (xy, valid, crossed, reentered, maxu, minj, firstcross, maxxy, maxeta, firstpoint), _ = jax.lax.scan(step, init, jnp.arange(steps))
    end = jnp.stack((jnp.linalg.norm(xy,axis=1), jnp.mod(jnp.arctan2(xy[:,1],xy[:,0]),2*jnp.pi), seeds[:,2]+deltas),axis=1)
    maxpoint=jnp.stack((maxu,jnp.mod(jnp.arctan2(maxxy[:,1],maxxy[:,0]),2*jnp.pi),maxeta),axis=1)
    return end, valid, crossed, reentered, maxu, minj, firstcross, maxpoint, firstpoint





def column_flux(field,q):
    """Exact 64-sample m<=3 column filter; no transverse table interpolation."""
    eta=jnp.arange(64)*(2*jnp.pi/4/64)
    sample=jnp.stack((jnp.broadcast_to(q[:,0,None],(len(q),64)),jnp.broadcast_to(q[:,1,None],(len(q),64)),jnp.broadcast_to(eta,(len(q),64))),axis=-1)
    pos,A=field.metric.position_and_jacobian(sample.reshape(-1,3))
    cart=field.bfield.evaluate_cartesian(pos)
    F=(jnp.linalg.det(A)[:,None]*jnp.linalg.solve(A,cart[...,None])[...,0]).reshape(len(q),64,3)
    weights=1+2*jnp.sum(jnp.cos((q[:,2,None,None]-eta[None,:,None])*4*jnp.arange(1,4)[None,None,:]),axis=-1)
    return jnp.einsum('ps,psc->pc',weights,F)/64

def trace_padded(field,table,seeds,deltas,steps=64,column=False):
    seeds=np.asarray(seeds,float);deltas=np.asarray(deltas,float)
    if seeds.ndim!=2 or seeds.shape[1]!=3 or deltas.shape!=(len(seeds),):raise ValueError('shape')
    out=[[] for _ in range(9)];capacity=32 if column else 256
    for start in range(0,len(seeds),capacity):
        q=seeds[start:start+capacity];d=deltas[start:start+capacity];n=len(q)
        q=np.concatenate((q,np.tile([[.5,0.,0.]],(capacity-n,1))))
        d=np.r_[d,np.zeros(capacity-n)]
        ans=trace(field,table,jnp.asarray(q),jnp.asarray(d),steps,column)
        for i,x in enumerate(ans):out[i].append(np.asarray(x)[:n])
    return tuple(np.concatenate(v) for v in out)
