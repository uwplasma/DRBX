"""Pure-JAX application of fixed Q direct/tube divergence and scalar diagnostics."""
from __future__ import annotations

from .q_parallel import stage_q


def stage_parallel_divergence(runtime, *, device=None):
    """Stage the lean selected runtime actually passed to JAX."""
    return stage_q(runtime,device=device)


def _raw(runtime, owner_values, boundary, kind):
    import jax.numpy as jnp
    if kind not in ('D','N'):raise ValueError('kind must be D or N')
    x=jnp.asarray(owner_values)
    if x.ndim<1 or x.shape[-1]!=runtime.metadata['n_owner']:
        raise ValueError('owner state length mismatch')
    nr=len(runtime.raw)
    coeff=runtime.coefficient_D if kind=='D' else runtime.coefficient_N
    slots=len(runtime.metadata['slots'])
    gathered=jnp.take(x,runtime.donor,axis=-1)
    if slots==1:
        raw=jnp.sum(gathered*jnp.asarray(coeff,dtype=x.dtype),axis=-1)
    else:
        raw=jnp.sum(gathered[..., :,None,:]*jnp.asarray(coeff,dtype=x.dtype),axis=-1)
    if kind=='D':
        trace=jnp.asarray(boundary.dirichlet_trace,dtype=x.dtype)
        query=jnp.asarray(boundary.dirichlet_query_value,dtype=x.dtype)
        tangent=jnp.asarray(boundary.dirichlet_tangent,dtype=x.dtype)
        if (trace.shape!=(*x.shape[:-1],nr,35) or
            query.shape!=(*x.shape[:-1],nr,3) or
            tangent.shape!=(*x.shape[:-1],nr,3,2)):
            raise ValueError('Dirichlet boundary shape mismatch')
        if slots==1:
            raw=raw+jnp.sum(trace*jnp.asarray(runtime.lift_D_node,dtype=x.dtype),axis=-1)
            raw=raw+jnp.sum(query*jnp.asarray(runtime.lift_D_query,dtype=x.dtype),axis=-1)
            raw=raw+jnp.sum(tangent[...,2,:]*jnp.asarray(runtime.lift_D_tangent,dtype=x.dtype),axis=-1)
        else:
            raw=raw+jnp.sum(trace[..., :,None,:]*jnp.asarray(runtime.lift_D_node,dtype=x.dtype),axis=-1)
            raw=raw+query*jnp.asarray(runtime.lift_D_query,dtype=x.dtype)
            raw=raw+jnp.sum(tangent*jnp.asarray(runtime.lift_D_tangent,dtype=x.dtype),axis=-1)
    else:
        normal=jnp.asarray(boundary.neumann_normal,dtype=x.dtype)
        if normal.shape!=(*x.shape[:-1],nr,35):raise ValueError('Neumann boundary shape mismatch')
        if slots==1:
            raw=raw+jnp.sum(normal*jnp.asarray(runtime.lift_N_normal,dtype=x.dtype),axis=-1)
        else:
            raw=raw+jnp.sum(normal[..., :,None,:]*jnp.asarray(runtime.lift_N_normal,dtype=x.dtype),axis=-1)
    return raw


def apply_raw_parallel_divergence(runtime, owner_values, boundary, *, kind):
    """Return each raw midpoint candidate action before owner projection."""
    if runtime.metadata['candidate'] not in ('direct','tube'):
        raise ValueError('direct or tube runtime required')
    return _raw(runtime,owner_values,boundary,kind)


def apply_parallel_divergence(runtime, owner_values, boundary, *, kind):
    """Return ``(..., n_chunk_owner)`` after complete-member projection."""
    import jax.numpy as jnp
    raw=apply_raw_parallel_divergence(runtime,owner_values,boundary,kind=kind)
    projected=jnp.take(raw,runtime.owner_raw,axis=-1)
    return jnp.sum(projected*jnp.asarray(runtime.owner_weight,dtype=raw.dtype),axis=-1)


def apply_raw_scalar_slots(runtime, owner_values, boundary, *, kind):
    """Diagnostic reconstructed scalar ``(..., n_raw, 3)`` at cap/center slots."""
    if runtime.metadata['candidate']!='scalar_slots':
        raise ValueError('scalar-slot diagnostic runtime required')
    return _raw(runtime,owner_values,boundary,kind)
