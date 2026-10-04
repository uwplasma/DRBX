"""Q07 existing constant channels and an explicit variable-coefficient audit.

The six existing parameters multiply unit-coefficient diffusion, including Te
and Ti. Do not insert 2/(3*n), temperature powers, mu, or additional derivatives:
those are not the semantics of these existing phenomenological coefficients.
"""
from dataclasses import replace
from typing import NamedTuple
from .q_parallel import QBoundaryData, apply_q
from .q_parallel_gradient import stage_parallel_gradient, apply_raw_slot_gradients
from ..stencils.q_parallel_channels import CHANNELS


class DiffusionChannels(NamedTuple):
    action: object
    coefficients_valid: object
    inputs_finite: object


def apply_diffusion_channels(runtime, state, boundary, coefficients, *, kinds):
    """Return (...,6,n_chunk_owner); state/BC axes are (...,6,n_owner/raw,...).

    runtime is the unchanged staged Q05 diffusion runtime. coefficients has
    shape (6,), or leading axes broadcasting within the state batch. These
    remain live differentiable scalar parameters, constant in space. Select
    either previously accepted diffusion span explicitly through runtime.
    """
    import jax.numpy as jnp
    x = jnp.asarray(state)
    if x.ndim < 2 or x.shape[-2] != len(CHANNELS):
        raise ValueError('expected (...,6,n_owner) channel state')
    if x.dtype.kind != 'f':
        raise TypeError('real floating channel state required')
    a = jnp.asarray(coefficients, dtype=x.dtype)
    if a.ndim < 1 or a.shape[-1] != len(CHANNELS):
        raise ValueError('six constant-in-space coefficients required')
    if jnp.broadcast_shapes(a.shape, x.shape[:-1]) != x.shape[:-1]:
        raise ValueError('coefficients must broadcast within state batch')
    if len(kinds) != 6 or any(k not in ('D','N') for k in kinds):
        raise ValueError('six D/N channel kinds required')
    actions = []
    for f, kind in enumerate(kinds):
        bc = QBoundaryData(boundary.dirichlet_trace[...,f,:,:],
            boundary.dirichlet_query_value[...,f,:,:],
            boundary.dirichlet_tangent[...,f,:,:,:],boundary.neumann_normal[...,f,:,:])
        unit = apply_q(runtime, x[...,f,:], bc, kind=kind)
        actions.append(a[...,f,None]*unit)
    action = jnp.stack(actions,axis=-2)
    valid = jnp.all(jnp.isfinite(a) & (a >= 0),axis=-1)
    finite = jnp.all(jnp.isfinite(x),axis=(-2,-1)) & jnp.all(jnp.isfinite(action),axis=(-2,-1))
    return DiffusionChannels(action, valid, finite)


def stage_coefficient_diffusion(runtime, *, device=None):
    import jax
    weights = jax.device_put(runtime.magnetic_L, device)
    weights.block_until_ready()
    return replace(runtime,gradient=stage_parallel_gradient(runtime.gradient,device=device),magnetic_L=weights)


def apply_coefficient_diffusion(runtime, owner_values, boundary, coefficient_slots, *, kind):
    """Return D(chi*G_cap f); coefficient slots are explicit live audit data.

    Shapes (...,n_raw,3), in minus/plus/center order. The coefficient producer
    owns its BC and value semantics. No chi reconstruction or positivity repair
    is inferred. The separate validity flag must be inspected by callers.
    """
    import jax.numpy as jnp
    if runtime.metadata.get('schema') != 'drbx.q-coefficient-diffusion.v1':
        raise ValueError('coefficient diffusion runtime required')
    x = jnp.asarray(owner_values)
    if x.dtype.kind != 'f':
        raise TypeError('real floating diffusion state required')
    grad = apply_raw_slot_gradients(runtime.gradient,x,boundary,kind=kind)
    chi = jnp.asarray(coefficient_slots,dtype=grad.dtype)
    if chi.shape != grad.shape:
        raise ValueError('coefficient slots must match field batch, raw and slot axes')
    raw = jnp.sum(chi*grad*jnp.asarray(runtime.magnetic_L,dtype=grad.dtype),axis=-1)
    action = jnp.sum(jnp.take(raw,runtime.gradient.owner_raw,axis=-1)*
                     jnp.asarray(runtime.gradient.owner_weight,dtype=grad.dtype),axis=-1)
    valid = jnp.all(jnp.isfinite(chi) & (chi >= 0),axis=(-2,-1))
    finite = jnp.all(jnp.isfinite(x),axis=-1) & jnp.all(jnp.isfinite(action),axis=-1)
    return DiffusionChannels(action,valid,finite)
