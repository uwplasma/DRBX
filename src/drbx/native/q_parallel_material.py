"""Bounded Q07 centered material terms and optional characteristic correction.

This is a research consumer, not production current/phi/SAT wiring. Products
are taken at reconstructed slots, coefficients at raw centers, projection last.
No clipping, collision, diffusion, vorticity or time integration is performed.
"""
from dataclasses import replace
from typing import NamedTuple
from .q_parallel import QBoundaryData
from .q_parallel_divergence import apply_raw_scalar_slots, stage_parallel_divergence
from .q_parallel_characteristic import eta_characteristic_correction


class MaterialAction(NamedTuple):
    material: object
    generalized_force: object
    centered: object
    correction: object
    combined: object
    eigensystem_admissible: object
    thermodynamic_states_positive: object
    inputs_finite: object


def stage_material_transport(runtime, *, device=None):
    import jax
    arrays = {k:jax.device_put(getattr(runtime, k), device)
              for k in ('magnetic_L', 'b_eta', 'eta_step')}
    for x in arrays.values():
        x.block_until_ready()
    return replace(runtime, inner=stage_parallel_divergence(runtime.inner, device=device),
                   outer=stage_parallel_divergence(runtime.outer, device=device), **arrays)


def material_from_slots(stencil, phi_slots, magnetic_L, b_eta, eta_step, *, tau, mu):
    """Raw action from (...,5 eta slots,5 fields) and (...,3) phi slots.

    Material slots are (-2,-1,0,1,2)*eta_step. Phi slots are (-1,1,0).
    Te/Ti compression uses tube D(j), D(V); pressure uses G of slot products.
    The DAE material -mu*tau*G(Ti) and +mu*G(phi+tau*Ti) share one G(Ti).
    The correction has neither another geometry source nor another phi force.
    Returned flags describe raw rows; callers must inspect them before use.
    """
    import jax.numpy as jnp
    q = jnp.asarray(stencil, dtype=jnp.float64)
    phi = jnp.asarray(phi_slots, dtype=q.dtype)
    if q.ndim < 2 or q.shape[-2:] != (5, 5):
        raise ValueError('expected (...,5 slots,5 fields)')
    if phi.shape != q.shape[:-2] + (3,):
        raise ValueError('phi slots must match material batch with 3 slots')
    L = jnp.asarray(magnetic_L, dtype=q.dtype)
    if jnp.broadcast_shapes(L.shape, phi.shape) != phi.shape:
        raise ValueError('tube weights must broadcast within scalar slots')
    # Selected scalar slots have the accepted minus/plus/center ordering.
    v = q[..., jnp.array([1, 3, 2]), :]
    n, te, ti, vi, ve = (v[..., i] for i in range(5))
    n0, te0, ti0, vi0, ve0 = (q[..., 2, i] for i in range(5))
    scale = jnp.asarray(b_eta)/(2*jnp.asarray(eta_step))
    G = lambda x: scale*(x[..., 1]-x[..., 0])
    D = lambda x: jnp.sum(L*x, axis=-1)
    current = n*(vi-ve)
    dj = D(current)
    gti = G(ti)
    material = jnp.stack((
        -D(n*ve),
        -ve0*G(te) + 2*te0/(3*n0)*(.71*dj-n0*D(ve)),
        -vi0*gti + 2*ti0/(3*n0)*(dj-n0*D(vi)),
        -vi0*G(vi) - G(n*(te+tau*ti))/n0,
        -ve0*G(ve) - mu*G(n*te)/n0 - .71*mu*G(te) - mu*tau*gti,
    ), axis=-1)
    force = jnp.zeros_like(material).at[..., 4].set(mu*(G(phi)+tau*gti))
    centered = material + force
    result = eta_characteristic_correction(q, b_eta, eta_step, tau=tau, mu=mu)
    finite = result.inputs_finite & jnp.all(jnp.isfinite(phi), axis=-1)
    finite &= jnp.all(jnp.isfinite(material+force+result.correction), axis=-1)
    return MaterialAction(material, force, centered, result.correction,
                          centered+result.correction, result.eigensystem_admissible,
                          result.thermodynamic_states_positive, finite)


def reconstruct_material_slots(scalar, state, boundary, *, kinds):
    """State (...,5,n_owner); BC arrays (...,5,n_raw,...); five static BC kinds."""
    import jax.numpy as jnp
    x = jnp.asarray(state)
    if x.ndim < 2 or x.shape[-2] != 5:
        raise ValueError('expected (...,5 fields,n_owner) state')
    if len(kinds) != 5 or any(k not in ('D', 'N') for k in kinds):
        raise ValueError('five D/N kinds required')
    values = []
    for f, kind in enumerate(kinds):
        bc = QBoundaryData(boundary.dirichlet_trace[..., f, :, :],
            boundary.dirichlet_query_value[..., f, :, :],
            boundary.dirichlet_tangent[..., f, :, :, :],
            boundary.neumann_normal[..., f, :, :])
        values.append(apply_raw_scalar_slots(scalar, x[..., f, :], bc, kind=kind))
    return jnp.stack(values, axis=-1)


def apply_raw_material_transport(runtime, state, inner_boundary, outer_boundary,
                                 phi, phi_boundary, *, kinds, phi_kind, tau, mu):
    import jax.numpy as jnp
    if runtime.metadata.get('schema') != 'drbx.q-material.v1':
        raise ValueError('prepared material runtime required')
    if jnp.shape(phi) != jnp.shape(state)[:-2] + (jnp.shape(state)[-1],):
        raise ValueError('phi and material state batch/owner shapes must match')
    inner = reconstruct_material_slots(runtime.inner, state, inner_boundary, kinds=kinds)
    outer = reconstruct_material_slots(runtime.outer, state, outer_boundary, kinds=kinds)
    stencil = jnp.stack((outer[..., 0, :], inner[..., 0, :], inner[..., 2, :],
                         inner[..., 1, :], outer[..., 1, :]), axis=-2)
    p = apply_raw_scalar_slots(runtime.inner, phi, phi_boundary, kind=phi_kind)
    return material_from_slots(stencil, p, runtime.magnetic_L, runtime.b_eta,
                               runtime.eta_step, tau=tau, mu=mu)


def project_material_action(runtime, action):
    """Project the five action arrays; validity flags remain per raw row."""
    import jax.numpy as jnp
    def project(x):
        return jnp.sum(jnp.take(x, runtime.inner.owner_raw, axis=-2)*
                       jnp.asarray(runtime.inner.owner_weight)[..., None], axis=-2)
    return MaterialAction(*(project(x) for x in action[:5]), *action[5:])


def apply_material_transport(runtime, *args, **kwargs):
    return project_material_action(runtime, apply_raw_material_transport(runtime, *args, **kwargs))
