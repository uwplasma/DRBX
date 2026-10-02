"""Pure-JAX FCI parallel gradient from precontracted traced scalar values."""
from __future__ import annotations
from .q_parallel import stage_q


def stage_traced_gradient(runtime, *, device=None):
    return stage_q(runtime, device=device)


def apply_raw_traced_gradient(runtime, owner_values, boundary, *, kind):
    """Return (..., n_raw); BC arrays share the state's leading batch axes.

    D uses wall-node and wall-projected query VALUES. D tangential derivatives
    are accepted by the shared boundary interface but are not read. N uses
    prescribed physical-normal data through the existing scalar extension.
    """
    import jax.numpy as jnp
    if runtime.metadata.get('schema') != 'drbx.q-traced-gradient.v1':
        raise ValueError('traced-gradient runtime required')
    if kind not in ('D', 'N'):
        raise ValueError('kind must be D or N')
    x = jnp.asarray(owner_values)
    if x.ndim < 1 or x.shape[-1] != runtime.metadata['n_owner']:
        raise ValueError('owner state length mismatch')
    if x.dtype.kind not in 'fc':
        raise TypeError('floating or complex owner state required')
    coeff = runtime.coefficient_D if kind == 'D' else runtime.coefficient_N
    raw = jnp.sum(jnp.take(x, runtime.donor, axis=-1)*jnp.asarray(coeff, dtype=x.dtype), axis=-1)
    nr = len(runtime.raw)
    if kind == 'D':
        trace = jnp.asarray(boundary.dirichlet_trace, dtype=x.dtype)
        query = jnp.asarray(boundary.dirichlet_query_value, dtype=x.dtype)
        if (trace.shape != (*x.shape[:-1], nr, 35) or
                query.shape != (*x.shape[:-1], nr, 3)):
            raise ValueError('Dirichlet boundary shape mismatch')
        raw = raw + jnp.sum(trace*jnp.asarray(runtime.lift_D_node, dtype=x.dtype), axis=-1)
        raw = raw + jnp.sum(query*jnp.asarray(runtime.lift_D_query, dtype=x.dtype), axis=-1)
    else:
        normal = jnp.asarray(boundary.neumann_normal, dtype=x.dtype)
        if normal.shape != (*x.shape[:-1], nr, 35):
            raise ValueError('Neumann boundary shape mismatch')
        raw = raw + jnp.sum(normal*jnp.asarray(runtime.lift_N_normal, dtype=x.dtype), axis=-1)
    return raw


def apply_traced_gradient(runtime, owner_values, boundary, *, kind):
    """Return (..., n_chunk_owner) with complete physical-volume projection."""
    import jax.numpy as jnp
    raw = apply_raw_traced_gradient(runtime, owner_values, boundary, kind=kind)
    return jnp.sum(jnp.take(raw, runtime.owner_raw, axis=-1)*
                   jnp.asarray(runtime.owner_weight, dtype=raw.dtype), axis=-1)
