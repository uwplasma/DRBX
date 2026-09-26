"""Matrix-free weighted negative adjoints for support-operator pairs.

The helper here deliberately knows nothing about FCI tracing, boundary data,
or a particular support layout.  It turns a supplied *homogeneous linear*
gradient ``G`` into its compatible divergence

``D = -M^-1 G^T W``.

``jax.linear_transpose`` forms the transpose without assembling ``G``.  Any
affine boundary contribution must be split from ``gradient`` before using this
module.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray
Gradient = Callable[[Array], Array]
Divergence = Callable[[Array], Array]


def _validated_mass_and_mask(
    name: str,
    mass: Array,
    active: Array | None,
) -> tuple[Array, Array]:
    """Validate one diagonal measure and return JAX mass/mask arrays."""

    mass_value = jnp.asarray(mass)
    if np.dtype(mass_value.dtype).kind != "f":
        raise TypeError(f"{name}_mass must have a real floating-point dtype")
    if active is None:
        active_value = jnp.ones(mass_value.shape, dtype=bool)
    else:
        active_value = jnp.asarray(active)
        if active_value.shape != mass_value.shape:
            raise ValueError(f"{name}_active must have shape {mass_value.shape}")
        if active_value.dtype != jnp.dtype(bool):
            raise TypeError(f"{name}_active must have boolean dtype")
    # Geometry fields can be tracers when an RHS is built below ``jit`` or
    # ``shard_map``.  Shape and dtype remain available there, but attempting
    # to inspect values on the host is illegal.  Retain full eager validation
    # whenever the inputs are concrete.
    try:
        host_mass = np.asarray(mass_value)
        host_active = np.asarray(active_value)
    except jax.errors.TracerArrayConversionError:
        pass
    else:
        active_mass = host_mass[host_active]
        if not np.all(np.isfinite(active_mass)) or np.any(active_mass <= 0.0):
            raise ValueError(
                f"{name}_mass must be finite and positive on active entries"
            )
    return mass_value, active_value


def build_weighted_negative_adjoint(
    gradient: Gradient,
    primal_mass: Array,
    dual_mass: Array,
    *,
    primal_active: Array | None = None,
    dual_active: Array | None = None,
) -> Divergence:
    """Build ``D(q) = -M^-1 G^T W q`` without materializing ``G``.

    ``gradient`` must be a pure homogeneous linear map from the primal array
    shape to the dual array shape.  The mass arrays define those two shapes,
    respectively.  Inactive entries need not have positive masses; they are
    excluded from the weighted pairing and are always returned as zero.

    The returned callable accepts either one dual field with ``dual_mass``'s
    shape or any number of leading batch dimensions followed by that shape.
    It is safe to call from :func:`jax.jit`.
    """

    if not callable(gradient):
        raise TypeError("gradient must be callable")
    primal_mass_value, primal_active_value = _validated_mass_and_mask(
        "primal", primal_mass, primal_active
    )
    dual_mass_value, dual_active_value = _validated_mass_and_mask(
        "dual", dual_mass, dual_active
    )
    if primal_mass_value.dtype != dual_mass_value.dtype:
        raise TypeError("primal_mass and dual_mass must have the same dtype")

    primal_shape = primal_mass_value.shape
    dual_shape = dual_mass_value.shape
    zero_primal = jnp.zeros(primal_shape, dtype=primal_mass_value.dtype)
    dual_template = jnp.asarray(gradient(zero_primal))
    if dual_template.shape != dual_shape:
        raise ValueError(
            "gradient(zero primal) must have the same shape as dual_mass; "
            f"got {dual_template.shape}, expected {dual_shape}"
        )
    if dual_template.dtype != dual_mass_value.dtype:
        raise TypeError(
            "gradient output dtype must match primal_mass and dual_mass dtype"
        )

    # Keeping inactive denominators benign prevents undefined inactive values
    # (such as zero aggregate volume) from participating in the computation.
    safe_primal_mass = jnp.where(primal_active_value, primal_mass_value, 1.0)
    active_dual_mass = jnp.where(dual_active_value, dual_mass_value, 0.0)
    transpose = jax.linear_transpose(gradient, zero_primal)
    dual_ndim = len(dual_shape)

    def apply_one(dual_values: Array) -> Array:
        masked_values = jnp.where(dual_active_value, dual_values, 0.0)
        (adjoint,) = transpose(active_dual_mass * masked_values)
        result = -adjoint / safe_primal_mass
        return jnp.where(primal_active_value, result, 0.0)

    def divergence(dual_values: Array) -> Array:
        """Apply the weighted negative adjoint to one or more dual fields."""

        values = jnp.asarray(dual_values)
        if values.dtype != dual_mass_value.dtype:
            raise TypeError(
                f"dual_values must have dtype {dual_mass_value.dtype}, got {values.dtype}"
            )
        if dual_ndim:
            if values.ndim < dual_ndim or values.shape[-dual_ndim:] != dual_shape:
                raise ValueError(
                    "dual_values must have trailing shape "
                    f"{dual_shape}; got {values.shape}"
                )
            leading_shape = values.shape[:-dual_ndim]
        else:
            leading_shape = values.shape
        if not leading_shape:
            return apply_one(values)
        flattened = values.reshape((-1,) + dual_shape)
        return jax.vmap(apply_one)(flattened).reshape(leading_shape + primal_shape)

    return divergence


__all__ = [
    "Gradient",
    "Divergence",
    "build_weighted_negative_adjoint",
]
