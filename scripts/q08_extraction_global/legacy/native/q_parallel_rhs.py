"""Six-field Q07 research RHS with prescribed primitive boundary data.

One owner-state entry point composes the existing material, scalar vorticity,
current/phi and constant diffusion channels. No sheath law, endpoint SAT,
collision/source model, polarization solve or production selector is added.
"""
from dataclasses import replace
from typing import NamedTuple

from .q_parallel import QBoundaryData, stage_q
from .q_parallel_material import (reconstruct_material_slots, material_from_slots,
                                  stage_material_transport)
from .q_parallel_divergence import apply_raw_scalar_slots
from .q_parallel_current_phi import current_phi_from_raw
from .q_parallel_vorticity import vorticity_from_slots
from .q_parallel_channels import apply_diffusion_channels


class SixFieldAction(NamedTuple):
    centered: object             # (..., n_chunk_owner, 6), includes current/phi
    correction: object           # same layout, characteristic plus scalar omega
    diffusion: object            # same layout, six constant channels
    combined: object             # sum of preceding three arrays
    raw_current: object          # CurrentPhiAction diagnostics, before projection
    raw_electron_material: object
    inputs_valid: object         # (..., n_raw)
    eigensystem_admissible: object


def stage_six_field_rhs(runtime, *, device=None):
    import jax
    bmag = jax.device_put(runtime.bmag, device)
    bmag.block_until_ready()
    return replace(runtime, material=stage_material_transport(runtime.material, device=device),
                   diffusion=stage_q(runtime.diffusion, device=device), bmag=bmag)


def field_boundary(boundary, selection):
    """Select fields while preserving arbitrary leading batch dimensions."""
    return QBoundaryData(boundary.dirichlet_trace[..., selection, :, :],
        boundary.dirichlet_query_value[..., selection, :, :],
        boundary.dirichlet_tangent[..., selection, :, :, :],
        boundary.neumann_normal[..., selection, :, :])


def apply_six_field_rhs(runtime, state, inner_boundary, outer_boundary,
                       phi, phi_boundary, coefficients, *, kinds, phi_kind,
                       tau, mu):
    """Evaluate the selected six-field parallel block; callers inspect flags.

    State layout is (...,6,n_owner), output (...,n_chunk_owner,6). Boundary
    fields use the same six-field ordering. Current is formed from reconstructed
    primitive slots, never independently reconstructed or given another BC.
    Its boundary contribution is physical minus zero-BC current at fixed owners;
    this contains product cross terms and is NOT an affine primitive-state map.
    The split is diagnostic, and the contribution enters the current action once.

    The material electron row already includes -mu*tau*G(Ti). Its matched
    generalized force is supplied once by the current/phi adapter. Diffusion
    coefficients are constant in space, with the existing channel normalization.
    """
    import jax.numpy as jnp
    if runtime.metadata.get('schema') != 'drbx.q-six-field-prescribed.v1':
        raise ValueError('prepared six-field runtime required')
    x = jnp.asarray(state)
    if x.ndim < 2 or x.shape[-2] != 6 or x.dtype.kind != 'f':
        raise ValueError('real (...,6,n_owner) state required')
    if len(kinds) != 6 or any(k not in ('D', 'N') for k in kinds):
        raise ValueError('six D/N kinds required')
    if jnp.shape(phi) != x.shape[:-2] + (x.shape[-1],):
        raise ValueError('phi and state batch/owner shapes must match')
    rt = runtime.material
    if 'balanced_tube' not in rt.metadata:
        raise ValueError('geometry-consistent material weights required')
    ib = field_boundary(inner_boundary, slice(0, 5))
    ob = field_boundary(outer_boundary, slice(0, 5))
    inner = reconstruct_material_slots(rt.inner, x[..., :5, :], ib, kinds=kinds[:5])
    outer = reconstruct_material_slots(rt.outer, x[..., :5, :], ob, kinds=kinds[:5])
    stencil = jnp.stack((outer[..., 0, :], inner[..., 0, :], inner[..., 2, :],
                         inner[..., 1, :], outer[..., 1, :]), axis=-2)
    p = apply_raw_scalar_slots(rt.inner, phi, phi_boundary, kind=phi_kind)
    material = material_from_slots(stencil, p, rt.magnetic_L, rt.b_eta,
                                    rt.eta_step, tau=tau, mu=mu)
    zero_bc = QBoundaryData(*(jnp.zeros_like(a) for a in ib))
    homogeneous = reconstruct_material_slots(rt.inner, x[..., :5, :], zero_bc,
                                              kinds=kinds[:5])
    def current_div(v):
        return jnp.sum(rt.magnetic_L*v[..., 0]*(v[..., 3]-v[..., 4]), axis=-1)
    physical, d0 = current_div(inner), current_div(homogeneous)
    scale = rt.b_eta/(2*rt.eta_step)
    gphi = scale*(p[..., 1]-p[..., 0])
    gti = scale*(inner[..., 1, 2]-inner[..., 0, 2])
    cp = current_phi_from_raw(d0, physical-d0, gphi, gti, inner[..., 2, 0],
        jnp.broadcast_to(runtime.bmag, d0.shape), tau=tau, mu=mu)
    # Replace the already-tested generalized force, rather than appending it.
    centered5 = material.material.at[..., 4].add(cp.electron_generalized_force)
    wi = apply_raw_scalar_slots(rt.inner, x[..., 5, :],
        field_boundary(inner_boundary, 5), kind=kinds[5])
    wo = apply_raw_scalar_slots(rt.outer, x[..., 5, :],
        field_boundary(outer_boundary, 5), kind=kinds[5])
    wstencil = jnp.stack((wo[..., 0], wi[..., 0], wi[..., 2], wi[..., 1], wo[..., 1]), axis=-1)
    omega = vorticity_from_slots(wstencil, inner[..., 2, 3], rt.b_eta, rt.eta_step)
    centered_raw = jnp.concatenate((centered5,
        (omega.centered+cp.vorticity_current)[..., None]), axis=-1)
    correction_raw = jnp.concatenate((material.correction, omega.correction[..., None]), axis=-1)
    def project(a):
        return jnp.sum(jnp.take(a, rt.inner.owner_raw, axis=-2)*
                       jnp.asarray(rt.inner.owner_weight)[..., None], axis=-2)
    centered, correction = project(centered_raw), project(correction_raw)
    db = inner_boundary if runtime.metadata['diffusion_span'] == 1/32 else outer_boundary
    diffusion = apply_diffusion_channels(runtime.diffusion, x, db, coefficients, kinds=kinds)
    diff = jnp.swapaxes(diffusion.action, -1, -2)
    combined = centered+correction+diff
    finite_batch = jnp.all(jnp.isfinite(combined), axis=(-2, -1))
    valid = (material.inputs_finite & material.thermodynamic_states_positive &
             cp.inputs_valid & omega.inputs_finite &
             diffusion.coefficients_valid[..., None] & diffusion.inputs_finite[..., None] &
             finite_batch[..., None])
    return SixFieldAction(centered, correction, diff, combined, cp,
                           material.material[..., 4], valid, material.eigensystem_admissible)
