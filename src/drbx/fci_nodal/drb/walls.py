"""Strong P rows and physical wall relations for the conducting and floating wall laws.

All geometry/sparse maps are plan arguments. Residuals and z_ext are independent
data, explicitly stop-gradient.
"""
from ..options import CONDUCTING_WALL, FLOATING_WALL
from dataclasses import dataclass
from typing import NamedTuple
from functools import lru_cache
import jax
import jax.numpy as jnp
from ..operators.perp.sbp_curvature import _material
from ..operators.perp.sbp_laplacian_apply import _node_derivative
from ..operators.perp.sbp_ops import d_eta,extend_periodic

from ..operators.parallel import linear_ops as qo

@dataclass(frozen=True)
class WallOps:
    T:object; TT:object; Tc:object; TcT:object; F:object; K:object
    H:object; Om:object; Js:object; curvature_flux:object
    hw:object; bn:object; trace_gram:object; mass_trace_gram:object
    Y:object; Khit:object; Bdiv:object; Wm:object; A0plus:object; gi:object; wall_foot_to_hit_area_ratio:object
jax.tree_util.register_dataclass(WallOps,data_fields=list(WallOps.__dataclass_fields__),meta_fields=[])

class Residual(NamedTuple):
    q_flux:object; q_vi:object; q_current:object; q_law:object
    p_psi:object; p_omega:object; p_te:object; p_flux:object; p_velocity:object

@dataclass(frozen=True)
class Closure:
    residual:Residual; z_ext:object
    wall_law:str='floating_wall'
jax.tree_util.register_dataclass(Closure,data_fields=['residual','z_ext'],meta_fields=['wall_law'])

def make_wall_data(plan,base,*,wall_law,z_ext,residual=None):
    if plan.wall_ops is None:raise ValueError('Call with_wall_plan first')
    w=plan.wall_ops;nh=len(plan.q['sg']);nw=len(w.Js)
    if residual is None:
        residual=Residual(jnp.zeros((nh,3)),jnp.zeros(nh),jnp.zeros(nh),jnp.zeros((nh,2)),
            jnp.zeros(nw),jnp.zeros(nw),jnp.zeros(nw),jnp.zeros((nw,3)),jnp.zeros((nw,2)))
    closure=Closure(residual,jnp.asarray(z_ext).reshape(nw,3),wall_law)
    return base._replace(closure=closure)

def project_mass(w,y,fields):
    result=y
    for f,mask in fields:
        row=w.T@result[:,f:f+1]
        reaction=mask[:,None]*row/w.mass_trace_gram[:,None]
        result=result.at[:,f].add(-(w.TT@reaction)[:,0]/w.H)
    return result


@jax.custom_jvp
def symmetric_positive(A):
    lam,V=jnp.linalg.eigh(A)
    return (V*jnp.maximum(lam,0.)[...,None,:])@jnp.swapaxes(V,-1,-2)
@symmetric_positive.defjvp
def _positive_jvp(primals,tangents):
    A,=primals;dA,=tangents;lam,V=jnp.linalg.eigh(A)
    gap=lam[..., :,None]-lam[...,None,:]
    num=jnp.maximum(lam,0.)[..., :,None]-jnp.maximum(lam,0.)[...,None,:]
    divided=jnp.where(gap!=0,num/jnp.where(gap!=0,gap,1.),(lam[..., :,None]>0).astype(A.dtype))
    vt=jnp.swapaxes(V,-1,-2)
    return symmetric_positive(A),V@(divided*(vt@dA@V))@vt

def material_matrix(z,tau):
    n,t,i=z.T
    rows=[jnp.stack([2*t+2*tau*n*i,2*n,2*tau*n*n],1),
          jnp.stack([4*t/3*(t/n+tau*i),14*t/3,4*tau*n*t/3],1),
          jnp.stack([4*i/3*(t/n+tau*i),4*i/3,4*tau*i/3*(n-2.5)],1)]
    return jnp.stack(rows,1),jnp.stack([t/n+tau*i,3*n/(2*t),3*tau*n*n/(2*i)],1)

def perpendicular_wall_relations(U,plan,closure,*,owner_grid=None):
    w=plan.wall_ops;res=jax.tree.map(jax.lax.stop_gradient,closure.residual)
    zext=jax.lax.stop_gradient(closure.z_ext);tw=w.T@U;z=tw[:,:3];n,t,i=z.T;tau=plan.params.tau
    sound=jnp.sqrt(t+w.gi*tau*i)
    phi=(.5*jnp.log(plan.qparams[1]*t/(2*jnp.pi*(t+w.gi*tau*i)))*t if closure.wall_law==FLOATING_WALL else jnp.zeros_like(t))+res.p_psi
    datum=phi+tau*n*i
    if owner_grid is None:
        E,N=plan.ctx.plan.structure.n_eta,plan.ctx.plan.structure.n
        phi2=phi.reshape(E,N)
        dt=_node_derivative(phi2,1,N).reshape(-1)
        de=d_eta(extend_periodic(phi2,2),plan.ctx.plan.structure.deta,2).reshape(-1)
    else:
        dt,de=_owner_wall_derivatives(w.T.mesh,owner_grid)(phi)
    ven=plan.params.rho_star*(w.hw[:,1]*de-w.hw[:,2]*dt)/w.Js
    A,W=material_matrix(z,tau)
    cn=w.curvature_flux/w.Js
    An=cn[:,None,None]*A-(ven/plan.params.rho_star)[:,None,None]*jnp.eye(3)
    sw=jnp.sqrt(W)
    positive=symmetric_positive(sw[:,:,None]*An/sw[:,None,:])/sw[:,:,None]*sw[:,None,:]
    if closure.wall_law==CONDUCTING_WALL:
        positive=w.A0plus/w.Js[:,None,None]
        mismatch=z-jnp.ones_like(z)
    else:mismatch=z-zext
    flux=res.p_flux-plan.params.rho_star*jnp.einsum('hij,hj->hi',positive,mismatch)
    if closure.wall_law==CONDUCTING_WALL:flux=flux.at[:,1].set(res.p_flux[:,1])
    inward=(ven<0)&(w.bn!=0)
    velocity=jnp.sign(w.bn)[:,None]*sound[:,None]*jnp.ones((1,2))+res.p_velocity
    if closure.wall_law==CONDUCTING_WALL:velocity=res.p_velocity
    return dict(trace=tw,psi=datum,phi=phi,flux=flux,An=An,An_plus=positive,
        sound=sound,ven=ven,inflow=inward,velocity=velocity,z_ext=zext)

def parallel_wall_relations(U,psi,plan,closure,*,hit_active=None):
    w=plan.wall_ops;p=plan.q;res=jax.tree.map(jax.lax.stop_gradient,closure.residual)
    r=p['R']@U;arr=qo.trace(p,U);sg=p['sg'];ell=jnp.abs(p['leg'])
    if hit_active is not None:
        neutral=jnp.array([1.,1.,1.,0.,0.,0.],dtype=U.dtype)
        r=jnp.where(hit_active[:,None],r,neutral)
        arr=jnp.where(hit_active[:,None],arr,neutral)
        ell=jnp.where(hit_active,ell,1.)
    thermal=r[:,:3]+(sg*ell)[:,None]*res.q_flux/plan.chi[:3]
    sound=jnp.sqrt(thermal[:,1]+w.gi*plan.params.tau*thermal[:,2])
    fast=sg*arr[:,3]>sound
    target=jnp.where(fast,sg*arr[:,3],sound)  # unsmoothed, derivative on selected side
    if closure.wall_law==FLOATING_WALL:
        vi=sg*(target+res.q_vi);ve=vi-res.q_current/thermal[:,0]
        g=jnp.column_stack([thermal,vi,ve,arr[:,5]])
    else:
        dz=thermal-1.;pressure=dz@jnp.array([2.,1.,1.]);effort=dz@jnp.array([2.,1.71,1.])
        psil=qo.trace(p,psi[:,None])[:,0]-1.
        L=jnp.array([[1.,0.],[1.,-1.]])
        physicalL=jnp.broadcast_to(L,(len(r),2,2)).at[:,1,1].set(-1/thermal[:,0])
        wd=plan.chi[3:5]*jnp.array([1.,1/plan.qparams[1]])/ell[:,None]
        wd=wd*read_wall_foot_to_hit_area_ratio(plan)[:,None]
        stress=L.T[None]@(wd[:,:,None]*physicalL)
        load=jnp.einsum('ij,hj->hi',L.T,wd*r[:,3:5])
        force=sg[:,None]*jnp.stack([pressure,-effort+psil],1)+load
        v=jnp.linalg.solve(jnp.eye(2)[None]+w.Y@stress,
            (jnp.einsum('hij,hj->hi',w.Y,force)+res.q_law)[...,None])[...,0]
        vi=v[:,0];ve=vi-v[:,1]/thermal[:,0]
        g=jnp.column_stack([thermal,vi,ve,arr[:,5]])
    return dict(g=g,foot=r,arriving=arr,sound=sound,fast=fast,target=target)


def read_wall_foot_to_hit_area_ratio(plan):return plan.wall_ops.wall_foot_to_hit_area_ratio

def compatible_state(U,plan,wall):
    """Project initial strong-value rows to supplied constant residual data."""
    w=plan.wall_ops;cl=wall.closure
    result=U
    for f,target,mask in [(5,cl.residual.p_omega,jnp.ones(len(w.Js))),
                          (1,1+cl.residual.p_te,(w.curvature_flux!=0).astype(U.dtype))] if cl.wall_law==CONDUCTING_WALL else [(5,cl.residual.p_omega,jnp.ones(len(w.Js)))]:
        error=(w.T@result[:,f:f+1])[:,0]-jax.lax.stop_gradient(target)
        result=result.at[:,f].add(-(w.TT@((mask*error/w.mass_trace_gram)[:,None]))[:,0]/w.H)
    return result

def moving_rate(y,exact_dt,plan,wall):
    """Hp reaction imposes d(T U)/dt=T dU*/dt on time-dependent strong rows."""
    w=plan.wall_ops;fields=[(5,jnp.ones(len(w.Js)))]
    if wall.closure.wall_law==CONDUCTING_WALL:fields.append((1,(w.curvature_flux!=0).astype(y.dtype)))
    return exact_dt+project_mass(w,y-exact_dt,fields)


@lru_cache(None)
def _owner_wall_derivatives(mesh,grid):
    from jax.sharding import PartitionSpec as P
    from ..operators.perp.owner_plane_layout import exchange_plane_halo

    def local(phi):
        x=phi.reshape(-1,grid.wall_per_plane)
        if min(grid.plane_counts)==max(grid.plane_counts):
            ext=exchange_plane_halo(x,2,'records',mesh.size)
        else:
            count=jnp.asarray(grid.plane_counts)[jax.lax.axis_index('records')]
            last=jax.lax.dynamic_slice_in_dim(x,count-2,2)
            left=jax.lax.ppermute(last,'records',[(p,(p+1)%mesh.size) for p in range(mesh.size)])
            right=jax.lax.ppermute(x[:2],'records',[(p,(p-1)%mesh.size) for p in range(mesh.size)])
            ext=jnp.concatenate((left,x,jnp.zeros_like(right)),axis=0)
            ext=jax.lax.dynamic_update_slice_in_dim(ext,right,count+2,axis=0)
        return (_node_derivative(x,1,grid.wall_per_plane).reshape(-1),
                d_eta(ext,grid.deta,2).reshape(-1))
    return jax.jit(jax.shard_map(local,mesh=mesh,in_specs=P('records'),
                                out_specs=(P('records'),P('records')),check_vma=False))


def owner_wall_rows(U,psi,plan,closure,p_wall,q_wall,bracket,curvature,B,*,hit_active):
    """Strong P rows and native-mass Q loads; P supplies uncorrected bracket and curvature."""
    w=plan.wall_ops;pd=p_wall;qd=q_wall
    own=w.Tc@U
    velocity=jnp.where(pd['inflow'][:,None],pd['velocity'],own[:,3:5])
    penalty=jnp.maximum(-w.Js*pd['ven'],0.)*(w.bn!=0)
    bracket=bracket.at[:,3:5].add(-w.Om*(w.TcT@(penalty[:,None]*(own[:,3:5]-velocity)))/w.H[:,None])
    q=jnp.column_stack([U[:,0:3],U[:,5],psi-plan.params.tau*U[:,0]*U[:,2]])
    wall_jump=w.curvature_flux[:,None]*(w.T@q-w.Tc@q)
    correction=.5*(w.TT@wall_jump+w.TcT@wall_jump)
    C=curvature+w.Om*correction/w.H[:,None]
    n,t,i=U[:,:3].T
    material=_material(n,t,i,B,plan.params.tau,plan.opts.psi,C[:,:4])
    cpsi=C[:,4]+plan.params.tau*(i*C[:,0]+n*C[:,2])
    out=material+jnp.column_stack([-2*n,-4*t/3,-4*i/3,jnp.zeros_like(n)])*cpsi[:,None]
    curv=jnp.zeros_like(U).at[:,[0,1,2,5]].set(plan.params.rho_star*out)
    p=bracket+curv-(w.K@U)*plan.params.D/w.H[:,None]
    p=p.at[:,:3].add(w.Om*(w.TT@(w.Js[:,None]*pd['flux']))/w.H[:,None])
    g=qd['g'];p0=plan.q;ell=jnp.where(hit_active,jnp.abs(p0['leg']),1.)
    bd=(p0['RT']@((p0['mu']*read_wall_foot_to_hit_area_ratio(plan)/ell)[:,None]*g))/p0['H'][:,None]/plan.dirichlet_mass_correction[:,None]
    bn=jnp.zeros_like(U).at[:,:3].set((p0['RT']@((p0['mu']*p0['sg'])[:,None]*jax.lax.stop_gradient(closure.residual.q_flux)/plan.chi[None,:3]))/p0['H'][:,None])
    return p,bn,bd
