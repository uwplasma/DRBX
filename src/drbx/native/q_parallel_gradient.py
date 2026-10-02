"""Pure-JAX direct midpoint parallel gradient and raw three-slot diagnostic."""
from __future__ import annotations

from .q_parallel import QBoundaryData, stage_q


def stage_parallel_gradient(runtime, *, device=None):
    """Stage the lean center or diagnostic runtime actually used by JAX."""
    return stage_q(runtime, device=device)


def _raw_slots(runtime, owner_values, boundary: QBoundaryData, kind: str):
    import jax.numpy as jnp
    if kind not in ('D', 'N'):
        raise ValueError('kind must be D or N')
    x = jnp.asarray(owner_values)
    if x.ndim < 1 or x.shape[-1] != runtime.metadata['n_owner']:
        raise ValueError('owner state length mismatch')
    nr = len(runtime.raw)
    slots = runtime.coefficient_D.shape[1]
    coefficients = runtime.coefficient_D if kind == 'D' else runtime.coefficient_N
    gathered = jnp.take(x, runtime.donor, axis=-1)
    result = jnp.sum(gathered[..., :, None, :] * jnp.asarray(coefficients, dtype=x.dtype), axis=-1)
    if kind == 'D':
        trace = jnp.asarray(boundary.dirichlet_trace, dtype=x.dtype)
        query = jnp.asarray(boundary.dirichlet_query_value, dtype=x.dtype)
        tangent = jnp.asarray(boundary.dirichlet_tangent, dtype=x.dtype)
        if (trace.shape != (*x.shape[:-1], nr, 35) or
            query.shape != (*x.shape[:-1], nr, 3) or
            tangent.shape != (*x.shape[:-1], nr, 3, 2)):
            raise ValueError('Dirichlet boundary shape mismatch')
        # Query value remains required by the shared boundary contract but G
        # differentiates the reconstructed scalar and does not use that value.
        result = result + jnp.sum(trace[..., :, None, :] *
                                  jnp.asarray(runtime.lift_D_node, dtype=x.dtype), axis=-1)
        tangent_slots = tangent if slots == 3 else tangent[..., :, 2:3, :]
        result = result + jnp.sum(tangent_slots *
                                  jnp.asarray(runtime.lift_D_tangent, dtype=x.dtype), axis=-1)
    else:
        normal = jnp.asarray(boundary.neumann_normal, dtype=x.dtype)
        if normal.shape != (*x.shape[:-1], nr, 35):
            raise ValueError('Neumann boundary shape mismatch')
        result = result + jnp.sum(normal[..., :, None, :] *
                                  jnp.asarray(runtime.lift_N_normal, dtype=x.dtype), axis=-1)
    return result


def apply_parallel_gradient(runtime, owner_values, boundary: QBoundaryData, *, kind: str):
    """Return direct G at raw midpoints projected to complete owners.

    ``owner_values`` is ``(..., n_owner)`` and boundary batch axes must agree.
    Prepared coefficients are fixed; state and prescribed boundary values may
    change under one JIT. The boundary contribution is affine.
    """
    import jax.numpy as jnp
    if runtime.metadata['slot_order'] != ('raw_midpoint',):
        raise ValueError('center-only runtime required')
    raw = _raw_slots(runtime, owner_values, boundary, kind)[..., 0]
    projected = jnp.take(raw, runtime.owner_raw, axis=-1)
    return jnp.sum(projected * jnp.asarray(runtime.owner_weight, dtype=raw.dtype), axis=-1)


def apply_raw_slot_gradients(runtime, owner_values, boundary: QBoundaryData, *, kind: str):
    """Return ``(..., n_raw, 3)`` in minus-cap, plus-cap, midpoint order."""
    if runtime.metadata['slot_order'] != ('minus_cap', 'plus_cap', 'raw_midpoint'):
        raise ValueError('three-slot diagnostic runtime required')
    return _raw_slots(runtime, owner_values, boundary, kind)
