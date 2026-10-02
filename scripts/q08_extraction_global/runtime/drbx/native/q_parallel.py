"""Pure-JAX application of fixed frozen traced-Q diffusion rows.

Fields have shape ``(..., n_owner)``. Boundary arrays use the same leading
batch shape, followed by raw row and prescribed-data axes. A prepared chunk
contains both BC variants; ``kind`` is static when JIT compiling.
"""
from __future__ import annotations
from typing import NamedTuple


class QBoundaryData(NamedTuple):
    dirichlet_trace: object       # (..., n_raw, 35) wall lattice values
    dirichlet_query_value: object # (..., n_raw, 3) wall-projected query values
    dirichlet_tangent: object     # (..., n_raw, 3, 2) theta/eta derivatives
    neumann_normal: object        # (..., n_raw, 35) physical-normal derivative


def stage_q(prepared, *, device=None):
    """Stage the lean runtime payload used by :func:`apply_q` once.

    Host-only helper. The returned fixed view may be captured by a JIT closure;
    no directional diagnostics or wall geometry are transferred.
    """
    from dataclasses import fields, replace
    import jax
    arrays = {f.name: jax.device_put(getattr(prepared, f.name), device)
              for f in fields(prepared) if f.name != 'metadata'}
    for array in arrays.values():
        array.block_until_ready()
    return replace(prepared, **arrays)


def homogeneous_boundary(prepared, *, batch_shape=(), dtype=None):
    import jax.numpy as jnp
    nr=len(prepared.raw)
    if dtype is None: dtype=jnp.float64
    return QBoundaryData(jnp.zeros((*batch_shape,nr,35),dtype),
                         jnp.zeros((*batch_shape,nr,3),dtype),
                         jnp.zeros((*batch_shape,nr,3,2),dtype),
                         jnp.zeros((*batch_shape,nr,35),dtype))


def apply_q(prepared, owner_values, boundary: QBoundaryData, *, kind: str):
    """Apply frozen D or physical-normal N action and project complete raw owners.

    The D query *value* is retained for future value operators. This diffusion
    contracts only gradients; it uses wall-node traces and query tangential
    derivatives. No wall solve, geometry evaluation, search, or host callback
    occurs here. ``prepared`` is fixed across state/boundary changes.
    """
    import jax.numpy as jnp
    if kind not in ('D','N'): raise ValueError('kind must be D or N')
    x=jnp.asarray(owner_values)
    if x.shape[-1]!=prepared.metadata['n_owner']: raise ValueError('owner state length mismatch')
    ids=jnp.asarray(prepared.donor)
    coeff=jnp.asarray(prepared.diffusion_D if kind=='D' else prepared.diffusion_N,dtype=x.dtype)
    gathered=jnp.take(x,ids,axis=-1)
    raw=jnp.sum(gathered*coeff,axis=-1)
    if kind=='D':
        trace=jnp.asarray(boundary.dirichlet_trace,dtype=x.dtype)
        query_value=jnp.asarray(boundary.dirichlet_query_value,dtype=x.dtype)
        tangent=jnp.asarray(boundary.dirichlet_tangent,dtype=x.dtype)
        if (trace.shape[-2:]!=(len(prepared.raw),35) or
            query_value.shape[-2:]!=(len(prepared.raw),3) or
            tangent.shape[-3:]!=(len(prepared.raw),3,2) or
            trace.shape[:-2]!=x.shape[:-1] or query_value.shape[:-2]!=x.shape[:-1] or
            tangent.shape[:-3]!=x.shape[:-1]):
            raise ValueError('Dirichlet boundary shape mismatch')
        raw=raw+jnp.sum(trace*jnp.asarray(prepared.boundary_D_node,dtype=x.dtype),axis=-1)
        raw=raw+jnp.sum(tangent*jnp.asarray(prepared.boundary_D_tangent,dtype=x.dtype),axis=(-1,-2))
    else:
        normal=jnp.asarray(boundary.neumann_normal,dtype=x.dtype)
        if normal.shape[-2:]!=(len(prepared.raw),35) or normal.shape[:-2]!=x.shape[:-1]:
            raise ValueError('Neumann boundary shape mismatch')
        raw=raw+jnp.sum(normal*jnp.asarray(prepared.boundary_N_normal,dtype=x.dtype),axis=-1)
    projected=jnp.take(raw,jnp.asarray(prepared.owner_raw),axis=-1)
    return jnp.sum(projected*jnp.asarray(prepared.owner_weight,dtype=x.dtype),axis=-1)
