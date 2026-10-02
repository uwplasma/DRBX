"""Compact pure-JAX Q reconstruction and six-field algebra.

Prescribed data remain live in the legacy QBoundaryData layout. Wall response
maps are compact; donor observations are gathered once for six fields and phi.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .q_parallel import QBoundaryData
from .q_parallel_rhs import SixFieldAction
from .q_parallel_material import material_from_slots
from .q_parallel_current_phi import current_phi_from_raw
from .q_parallel_vorticity import vorticity_from_slots


class QReconstruction(NamedTuple):
    value: object                # (..., fields, raw, five slots)
    homogeneous: object          # same, no second zero-BC reconstruction


def stage_q_plan(plan, *, device=None):
    out = jax.tree.map(lambda a: jax.device_put(a, device), plan)
    for a in jax.tree.leaves(out):
        a.block_until_ready()
    return out


def _check(plan, x, kinds):
    if x.ndim < 2 or x.shape[-1] != plan.n_owner:
        raise ValueError('expected (...,fields,n_owner) state')
    if x.dtype.kind not in ('f','c'):
        raise TypeError('floating scalar state required')
    if len(kinds) != x.shape[-2] or any(k not in ('D','N') for k in kinds):
        raise ValueError('one D/N kind per field required')


def _boundary_shape(b, x, nr):
    shapes = ((nr,35),(nr,3),(nr,3,2),(nr,35))
    for a, tail in zip(b, shapes, strict=True):
        if jnp.shape(a) != x.shape[:-1]+tail:
            raise ValueError('boundary shape mismatch')


def _scalar(plan, x, ib, ob, kinds, gathered, slots=(0,1,2,3,4)):
    nr=plan.donor.shape[0]; _boundary_shape(ib,x,nr); _boundary_shape(ob,x,nr)
    wall=plan.wall_index
    d=jnp.asarray(plan.row_value_D,dtype=x.dtype)[:,jnp.array(slots)]
    n=d.at[wall].set(jnp.asarray(plan.row_value_N_wall,dtype=x.dtype)[:,jnp.array(slots)])
    select=jnp.array([k=='D' for k in kinds])[:,None,None,None]
    rows=jnp.where(select,d[None],n[None])
    homogeneous=jnp.sum(gathered[..., :, :,None,:]*rows,axis=-1)
    value=homogeneous
    # Pairs use separate legacy prescribed arrays even when their node geometry
    # agrees. Center follows inner data, as in the original material assembly.
    for si, slot in enumerate(slots):
        bc=ob if slot in (0,4) else ib
        old_slot={0:0,1:0,2:2,3:1,4:1}[slot]
        trace=jnp.take(jnp.asarray(bc.dirichlet_trace,dtype=x.dtype),wall,axis=-2)
        normal=jnp.take(jnp.asarray(bc.neumann_normal,dtype=x.dtype),wall,axis=-2)
        query=jnp.take(jnp.asarray(bc.dirichlet_query_value,dtype=x.dtype),wall,axis=-2)[...,old_slot]
        dnode=jnp.sum(trace*jnp.asarray(plan.boundary_value_D_trace,dtype=x.dtype)[:,slot,:],axis=-1)
        nnode=jnp.sum(normal*jnp.asarray(plan.boundary_value_N_normal,dtype=x.dtype)[:,slot,:],axis=-1)
        is_d=jnp.array([k=='D' for k in kinds])[:,None]
        node=jnp.where(is_d,dnode,nnode)
        v=value[...,si].at[...,wall].add(node)
        v=v.at[...,wall].add(jnp.where(is_d,query,jnp.zeros_like(query)))
        value=value.at[...,si].set(v)
    return QReconstruction(value,homogeneous)


def reconstruct_q_state(plan, state, inner_boundary, outer_boundary, *, kinds):
    """Reconstruct five scalar positions; complex scalar fields are supported."""
    x=jnp.asarray(state); _check(plan,x,kinds)
    return _scalar(plan,x,inner_boundary,outer_boundary,kinds,jnp.take(x,plan.donor,axis=-1))


def _project_scalar(plan, raw):
    return jnp.sum(jnp.take(raw,plan.owner_raw,axis=-1)*jnp.asarray(plan.owner_weight,dtype=raw.dtype),axis=-1)


def apply_traced_gradient(plan, slots):
    """Inner h/32 traced G, from (...,raw,5) reconstructed scalar slots."""
    v=jnp.asarray(slots)
    if v.shape[-2:] != (plan.donor.shape[0],5):
        raise ValueError('expected (...,raw,5) scalar slots')
    return _project_scalar(plan,jnp.asarray(plan.b_eta/(2*plan.eta_step),dtype=v.dtype)*(v[...,3]-v[...,1]))


def apply_tube_divergence(plan, slots):
    """Balanced material tube D, with center weight fixed to div(b).

    This adapter uses Q07/Q08 balanced material weights. The old Q06
    prepare_tube_divergence action uses the original unbalanced magnetic_L;
    callers requiring that diagnostic must keep its legacy runtime.
    """
    v=jnp.asarray(slots)
    if v.shape[-2:] != (plan.donor.shape[0],5):
        raise ValueError('expected (...,raw,5) scalar slots')
    return _project_scalar(plan,jnp.sum(v[...,jnp.array([1,3,2])]*jnp.asarray(plan.magnetic_L,dtype=v.dtype),axis=-1))


def _diffusion(plan,x,bc,kinds,gathered,coefficients):
    a=jnp.asarray(coefficients,dtype=x.dtype)
    if a.ndim<1 or a.shape[-1]!=6 or jnp.broadcast_shapes(a.shape,x.shape[:-1])!=x.shape[:-1]:
        raise ValueError('six coefficients must broadcast within state batch')
    d=jnp.asarray(plan.diffusion_D,dtype=x.dtype)
    n=d.at[plan.wall_index].set(jnp.asarray(plan.diffusion_N_wall,dtype=x.dtype))
    is_d=jnp.array([k=='D' for k in kinds])[:,None,None]
    raw=jnp.sum(gathered*jnp.where(is_d,d[None],n[None]),axis=-1)
    wall=plan.wall_index
    trace=jnp.take(jnp.asarray(bc.dirichlet_trace,dtype=x.dtype),wall,axis=-2)
    normal=jnp.take(jnp.asarray(bc.neumann_normal,dtype=x.dtype),wall,axis=-2)
    tangent=jnp.take(jnp.asarray(bc.dirichlet_tangent,dtype=x.dtype),wall,axis=-3)
    node=jnp.where(is_d[...,0],jnp.sum(trace*jnp.asarray(plan.boundary_D_node,dtype=x.dtype),axis=-1),
                   jnp.sum(normal*jnp.asarray(plan.boundary_N_normal,dtype=x.dtype),axis=-1))
    raw=raw.at[...,wall].add(node)
    t=jnp.sum(tangent*jnp.asarray(plan.boundary_D_tangent,dtype=x.dtype),axis=(-1,-2))
    raw=raw.at[...,wall].add(jnp.where(is_d[...,0],t,jnp.zeros_like(t)))
    action=a[..., :,None]*_project_scalar(plan,raw)
    valid=jnp.all(jnp.isfinite(a)&(a>=0),axis=-1)
    finite=jnp.all(jnp.isfinite(x),axis=(-2,-1))&jnp.all(jnp.isfinite(action),axis=(-2,-1))
    return jnp.swapaxes(action,-1,-2),valid,finite


def apply_q_plan(plan,state,inner_boundary,outer_boundary,phi,phi_boundary,
                 coefficients,*,kinds,phi_kind,tau,mu):
    """Preserve SixFieldAction outputs; six-field material state is real only."""
    x=jnp.asarray(state)
    if x.ndim<2 or x.shape[-2]!=6 or x.dtype.kind!='f':
        raise ValueError('real (...,6,n_owner) state required')
    _check(plan,x,kinds)
    powner=jnp.asarray(phi)
    if powner.shape!=x.shape[:-2]+(x.shape[-1],):
        raise ValueError('phi and state batch/owner shapes must match')
    if phi_kind not in ('D','N'):
        raise ValueError('phi_kind must be D or N')
    if powner.dtype.kind!='f':
        raise ValueError('real phi required')
    # One gather serves six scalar fields, homogeneous current, diffusion and phi.
    fetched=jnp.take(jnp.concatenate((x,powner[...,None,:]),axis=-2),plan.donor,axis=-1)
    slots=_scalar(plan,x,inner_boundary,outer_boundary,kinds,fetched[...,:6,:,:])
    pb=QBoundaryData(*(jnp.asarray(a)[...,None,:,:,:] if i==2 else jnp.asarray(a)[...,None,:,:]
                      for i,a in enumerate(phi_boundary)))
    ps=_scalar(plan,powner[...,None,:],pb,pb,(phi_kind,),fetched[...,6:,:,:],slots=(1,3,2)).value[...,0,:,:]
    # value is (...,field,raw,slot); material consumes (...,raw,slot,field).
    stencil=jnp.moveaxis(slots.value[...,:5,:,:],-3,-1)
    material=material_from_slots(stencil,ps,plan.magnetic_L,plan.b_eta,plan.eta_step,tau=tau,mu=mu)
    homogeneous=jnp.moveaxis(slots.homogeneous[...,:5,:,jnp.array([1,3,2])],-3,-1)
    inner=jnp.moveaxis(slots.value[...,:5,:,jnp.array([1,3,2])],-3,-1)
    def current_div(v):
        return jnp.sum(plan.magnetic_L*v[...,0]*(v[...,3]-v[...,4]),axis=-1)
    physical,d0=current_div(inner),current_div(homogeneous)
    scale=plan.b_eta/(2*plan.eta_step)
    cp=current_phi_from_raw(d0,physical-d0,scale*(ps[...,1]-ps[...,0]),
        scale*(inner[...,1,2]-inner[...,0,2]),inner[...,2,0],
        jnp.broadcast_to(plan.bmag,d0.shape),tau=tau,mu=mu)
    centered5=material.material.at[...,4].add(cp.electron_generalized_force)
    omega=vorticity_from_slots(slots.value[...,5,:,:],inner[...,2,3],plan.b_eta,plan.eta_step)
    centered_raw=jnp.concatenate((centered5,(omega.centered+cp.vorticity_current)[...,None]),axis=-1)
    correction_raw=jnp.concatenate((material.correction,omega.correction[...,None]),axis=-1)
    def project(a):
        return jnp.sum(jnp.take(a,plan.owner_raw,axis=-2)*plan.owner_weight[...,None],axis=-2)
    centered,correction=project(centered_raw),project(correction_raw)
    bc=inner_boundary if plan.diffusion_span==1/32 else outer_boundary
    diff,cv,df=_diffusion(plan,x,bc,kinds,fetched[...,:6,:,:],coefficients)
    combined=centered+correction+diff
    finite_batch=jnp.all(jnp.isfinite(combined),axis=(-2,-1))
    valid=(material.inputs_finite&material.thermodynamic_states_positive&cp.inputs_valid&omega.inputs_finite&
           cv[...,None]&df[...,None]&finite_batch[...,None])
    return SixFieldAction(centered,correction,diff,combined,cp,material.material[...,4],valid,material.eigensystem_admissible)
