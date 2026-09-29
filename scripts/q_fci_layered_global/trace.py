"""Frozen single-metric RK4-256 research tracer.

Adapted from the 2026-09-27 Q optimization audit prototype. The same
source-box, Jacobian, beta, first-crossing, reentry, and reach checks remain.
"""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64",True)

@partial(jax.jit, static_argnames=("steps",))
def trace(field, seeds, deltas, steps):
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
        good = (u>0)&(u<1.1)&(R>=Rlo)&(R<=Rhi)&(Z>=Zlo)&(Z<=Zhi)&(J>0)&jnp.isfinite(J)
        safe_pos = jnp.where(good[:, None], pos, fallback_pos)
        safe_A = jnp.where(good[:, None, None], A, fallback_A)
        cartesian = field.bfield.evaluate_cartesian(safe_pos)
        B = jnp.linalg.solve(safe_A, cartesian[..., None])[..., 0]
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




CAPACITIES=(64,128,256)
def trace_fixed_batch(field, seeds, deltas, steps, capacity, device):
    """Trace one large, fixed-shape batch on an explicitly selected device.

    ``field`` must already reside on that device. Only seeds/deltas are sent
    per call; all RK stages stay on device. Fetch the nine outputs together,
    synchronizing execution before returning host arrays to the CPU fitter.
    """
    seeds = np.asarray(seeds, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    count = len(seeds)
    if seeds.shape != (count, 3) or deltas.shape != (count,):
        raise ValueError('seed/delta shape')
    if not 0 < count <= capacity or steps < 1:
        raise ValueError('positive steps and 0 < count <= capacity required')
    padded = np.tile(np.array([[.5, 0., 0.]]), (capacity, 1))
    padded[:count] = seeds
    delta = np.zeros(capacity, dtype=np.float64)
    delta[:count] = deltas
    with jax.default_device(device):
        result = trace(field, jax.device_put(padded, device),
                       jax.device_put(delta, device), steps)
        host = jax.device_get(result)
    return tuple(np.asarray(value)[:count].copy() for value in host)


def trace_padded(field,seeds,deltas,steps=256):
    """Fixed shapes, harmless valid padding, all nine real-row diagnostics."""
    seeds=np.asarray(seeds,float);deltas=np.asarray(deltas,float)
    if seeds.ndim!=2 or seeds.shape[1]!=3 or deltas.shape!=(len(seeds),):raise ValueError("seed/delta shape")
    if len(seeds)==0:return tuple(np.empty((0,3)) if i in (0,7,8) else np.empty(0) for i in range(9))
    out=[[] for _ in range(9)]
    for lo in range(0,len(seeds),CAPACITIES[-1]):
        q=seeds[lo:lo+CAPACITIES[-1]];d=deltas[lo:lo+CAPACITIES[-1]];size=len(q)
        cap=next(c for c in CAPACITIES if size<=c)
        if cap>size:
            q=np.vstack((q,np.tile(np.array([[.5,0.,0.]]),(cap-size,1))))
            d=np.r_[d,np.zeros(cap-size)]
        result=trace(field,jnp.asarray(q),jnp.asarray(d),steps)
        for i,value in enumerate(result):out[i].append(np.asarray(value)[:size])
    return tuple(np.concatenate(x,axis=0) for x in out)
