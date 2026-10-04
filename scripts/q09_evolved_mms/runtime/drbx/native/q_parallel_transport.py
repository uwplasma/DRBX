"""Pure-JAX centered Q07 density transport with live primitive boundary data."""
from dataclasses import replace

from .q_parallel_divergence import apply_raw_scalar_slots, stage_parallel_divergence


def stage_density_transport(runtime, *, device=None):
    """Stage shared scalar rows once and only the additional tube weights."""
    import jax
    scalar = stage_parallel_divergence(runtime.scalar, device=device)
    weights = jax.device_put(runtime.magnetic_L, device)
    weights.block_until_ready()
    return replace(runtime, scalar=scalar, magnetic_L=weights)


def apply_raw_density_flux(runtime, density, electron_velocity, density_boundary,
                           velocity_boundary, *, density_kind, velocity_kind):
    """Return n*Ve at minus cap, plus cap and midpoint, before differentiation.

    Both primitives have the same (..., n_owner) shape. Each may independently
    use Dirichlet or physical-normal Neumann data. This smooth centered block
    does not impose positivity floors or add characteristic dissipation.
    """
    import jax.numpy as jnp
    n, v = jnp.asarray(density), jnp.asarray(electron_velocity)
    if n.shape != v.shape:
        raise ValueError('density and electron velocity shapes must match')
    dtype = jnp.result_type(n, v, 1.0)
    n, v = n.astype(dtype), v.astype(dtype)
    ns = apply_raw_scalar_slots(runtime.scalar, n, density_boundary, kind=density_kind)
    vs = apply_raw_scalar_slots(runtime.scalar, v, velocity_boundary, kind=velocity_kind)
    return ns * vs


def apply_raw_density_transport(runtime, density, electron_velocity,
                                density_boundary, velocity_boundary, *,
                                density_kind, velocity_kind):
    """Return -div(b*n*Ve) per raw midpoint, using unchanged tube weights."""
    import jax.numpy as jnp
    flux = apply_raw_density_flux(runtime, density, electron_velocity,
        density_boundary, velocity_boundary, density_kind=density_kind,
        velocity_kind=velocity_kind)
    return -jnp.sum(flux*jnp.asarray(runtime.magnetic_L, dtype=flux.dtype), axis=-1)


def apply_density_transport(runtime, density, electron_velocity, density_boundary,
                             velocity_boundary, *, density_kind, velocity_kind):
    """Apply the centered continuity contribution then project complete owners."""
    import jax.numpy as jnp
    raw = apply_raw_density_transport(runtime, density, electron_velocity,
        density_boundary, velocity_boundary, density_kind=density_kind,
        velocity_kind=velocity_kind)
    return jnp.sum(jnp.take(raw, runtime.scalar.owner_raw, axis=-1) *
                   jnp.asarray(runtime.scalar.owner_weight, dtype=raw.dtype), axis=-1)
