"""Bounded traced-Q vorticity advection, separate from the current/SAT term."""
from typing import NamedTuple
from .q_parallel_divergence import apply_raw_scalar_slots


class VorticityAdvection(NamedTuple):
    centered: object
    correction: object
    combined: object
    inputs_finite: object


def vorticity_from_slots(omega_stencil, ion_velocity_center, b_eta, eta_step):
    """Point advection -Vi*G(omega) plus a separate scalar upwind correction.

    omega_stencil is (...,5), eta offsets (-2,-1,0,1,2)*eta_step. Velocity
    is frozen at the raw center for directional splitting. This is advection,
    not -D(Vi*omega); no geometric compression or current source belongs here.
    The same second-order point derivatives as Q07 material correction are
    used, without the five-field eigensystem. No positivity clipping.
    """
    import jax.numpy as jnp
    w = jnp.asarray(omega_stencil)
    if w.ndim < 1 or w.shape[-1] != 5:
        raise ValueError('expected (...,5) vorticity stencil')
    if w.dtype.kind != 'f':
        raise TypeError('real floating vorticity required')
    vi, beta, step = (jnp.asarray(x, dtype=w.dtype) for x in
                      (ion_velocity_center, b_eta, eta_step))
    if jnp.broadcast_shapes(w.shape[:-1], vi.shape, beta.shape, step.shape) != w.shape[:-1]:
        raise ValueError('speed and geometry must broadcast within raw batch')
    mm, m, c, p, pp = (w[..., i] for i in range(5))
    center = (p-m)/(2*step)
    backward = (4*(c-m)-(c-mm))/(2*step)
    forward = (4*(p-c)-(pp-c))/(2*step)
    speed = beta*vi
    centered = -speed*center
    correction = -jnp.maximum(speed, 0)*(backward-center)
    correction -= jnp.minimum(speed, 0)*(forward-center)
    finite = jnp.all(jnp.isfinite(w), axis=-1)
    finite &= jnp.isfinite(speed) & jnp.isfinite(step) & (step > 0)
    return VorticityAdvection(centered, correction, centered+correction, finite)


def apply_raw_vorticity_advection(runtime, omega, ion_velocity, inner_boundary,
                                  outer_boundary, velocity_boundary, *,
                                  omega_kind, velocity_kind):
    """Use a checked paired MaterialRuntime, without computing material rows."""
    import jax.numpy as jnp
    if runtime.metadata.get('schema') != 'drbx.q-material.v1':
        raise ValueError('checked paired material runtime required')
    if jnp.shape(omega) != jnp.shape(ion_velocity):
        raise ValueError('omega and ion velocity shapes must match')
    inner = apply_raw_scalar_slots(runtime.inner, omega, inner_boundary, kind=omega_kind)
    outer = apply_raw_scalar_slots(runtime.outer, omega, outer_boundary, kind=omega_kind)
    vi = apply_raw_scalar_slots(runtime.inner, ion_velocity, velocity_boundary, kind=velocity_kind)[..., 2]
    stencil = jnp.stack((outer[...,0], inner[...,0], inner[...,2], inner[...,1], outer[...,1]), axis=-1)
    return vorticity_from_slots(stencil, vi, runtime.b_eta, runtime.eta_step)


def apply_vorticity_advection(runtime, *args, **kwargs):
    """Project raw advection/correction independently to complete owners."""
    import jax.numpy as jnp
    raw = apply_raw_vorticity_advection(runtime, *args, **kwargs)
    def project(x):
        return jnp.sum(jnp.take(x, runtime.inner.owner_raw, axis=-1)*
                       jnp.asarray(runtime.inner.owner_weight), axis=-1)
    return VorticityAdvection(*(project(x) for x in raw[:3]), raw.inputs_finite)
