"""Absolute flux action with the analytic absolute-flux JVP analytic JVP."""
from functools import partial

import jax
import jax.numpy as jnp

from . import checked_characteristics as cf


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def absolute_action(u, z, tau, mu):
    return cf.action(u, z, jnp, tau, mu)[0]


@absolute_action.defjvp
def _absolute_jvp(tau, mu, primals, tangents):
    u, z = primals
    du, dz = tangents
    ab, E, _, _ = cf.absolute_data(u, z, jnp, tau, mu)
    value = absolute_action(u, z, tau, mu)
    tangent = (jnp.einsum('pij,pj->pi', ab, dz[:, :5])
               + jnp.einsum('pij,pj->pi', E, du[:, :5]))
    return value, tangent
