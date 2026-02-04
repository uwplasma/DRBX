from __future__ import annotations

import jax.numpy as jnp


def poisson_bracket_fourier(kx: float, ky: float, phi: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
    """Poisson bracket [phi, f] for a *single* Fourier mode.

    For perturbations with the same (kx, ky), the self-nonlinearity vanishes:
      [phi, f] = (∂x phi)(∂y f) - (∂y phi)(∂x f) = 0

    This is provided as a placeholder for future multi-mode (spectral) extensions.
    """

    _ = (kx, ky, phi, f)
    return jnp.zeros_like(f)


def _ddx_periodic_centered(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)


def _ddy_periodic_centered(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)


def poisson_bracket_centered(phi: jnp.ndarray, f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Centered-difference Poisson bracket on a 2D periodic grid.

    This is simple but does *not* guarantee discrete conservation properties.
    Prefer `poisson_bracket_arakawa` for a conservative discrete bracket.
    """

    dphi_dx = _ddx_periodic_centered(phi, dx)
    dphi_dy = _ddy_periodic_centered(phi, dy)
    df_dx = _ddx_periodic_centered(f, dx)
    df_dy = _ddy_periodic_centered(f, dy)
    return dphi_dx * df_dy - dphi_dy * df_dx


def poisson_bracket_arakawa(phi: jnp.ndarray, f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Arakawa (1966) conservative Poisson bracket on a 2D periodic grid.

    Implements the classic 2nd-order Arakawa Jacobian:

      J = (J1 + J2 + J3)/3

    which is antisymmetric and discretely conserves quadratic invariants for
    2D incompressible flows (e.g. energy/enstrophy for Euler/vorticity).
    """

    # Shifts in x (axis=0) and y (axis=1)
    p_ip = jnp.roll(phi, -1, axis=0)
    p_im = jnp.roll(phi, 1, axis=0)
    p_jp = jnp.roll(phi, -1, axis=1)
    p_jm = jnp.roll(phi, 1, axis=1)

    p_ip_jp = jnp.roll(p_ip, -1, axis=1)
    p_ip_jm = jnp.roll(p_ip, 1, axis=1)
    p_im_jp = jnp.roll(p_im, -1, axis=1)
    p_im_jm = jnp.roll(p_im, 1, axis=1)

    f_ip = jnp.roll(f, -1, axis=0)
    f_im = jnp.roll(f, 1, axis=0)
    f_jp = jnp.roll(f, -1, axis=1)
    f_jm = jnp.roll(f, 1, axis=1)

    f_ip_jp = jnp.roll(f_ip, -1, axis=1)
    f_ip_jm = jnp.roll(f_ip, 1, axis=1)
    f_im_jp = jnp.roll(f_im, -1, axis=1)
    f_im_jm = jnp.roll(f_im, 1, axis=1)

    inv4 = 1.0 / (4.0 * dx * dy)

    j1 = (p_ip - p_im) * (f_jp - f_jm) - (p_jp - p_jm) * (f_ip - f_im)

    j2 = (
        p_ip * (f_ip_jp - f_ip_jm)
        - p_im * (f_im_jp - f_im_jm)
        - p_jp * (f_ip_jp - f_im_jp)
        + p_jm * (f_ip_jm - f_im_jm)
    )

    j3 = (
        p_ip_jp * (f_jp - f_ip)
        - p_im_jm * (f_im - f_jm)
        - p_im_jp * (f_jp - f_im)
        + p_ip_jm * (f_ip - f_jm)
    )

    return (j1 + j2 + j3) * (inv4 / 3.0)
