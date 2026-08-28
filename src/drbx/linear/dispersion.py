"""Linear dispersion operators for two reduced edge-plasma models.

Each function assembles the linear operator ``A`` of a single Fourier mode
(``delta ~ exp(i k.x)``) directly from the model equations, so that the
eigenfrequencies returned by :func:`drbx.linear.eigenmodes` reproduce the
analytic dispersion relation without that relation being wired in by hand.

Shear-Alfven wave (electromagnetic, electron inertia), reduced two-field model
in ``(phi, psi)`` (potential, parallel flux):

    d/dt phi = i * v_A^2 * k_par * psi
    d/dt psi = i * k_par * phi / (1 + k_perp^2 * d_e^2)

whose eigenvalues are ``+/- i * omega`` with
``omega = k_par * v_A / sqrt(1 + k_perp^2 d_e^2)`` (Stegmeir et al.,
Phys. Plasmas 26, 052517 (2019)).

Resistive drift wave, Hasegawa-Wakatani two-field model in ``(phi, n)``
(potential, density), Fourier-reduced with ``kperp2 = k_perp^2``:

    d/dt phi = (-alpha/kperp2) phi + (alpha/kperp2) n
    d/dt n   = (alpha - i kappa k_y) phi - alpha n

with adiabaticity ``alpha`` and density-gradient drive ``kappa``. In the
adiabatic limit ``alpha -> inf`` the drift mode frequency tends to
``omega_star = kappa k_y / (1 + kperp2)``; at finite ``alpha`` the resistive
coupling makes it grow (Dudson et al., Comput. Phys. Commun. 180, 1467 (2009);
Hasegawa & Wakatani, Phys. Rev. Lett. 50, 682 (1983)).
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "shear_alfven_operator",
    "shear_alfven_frequency",
    "resistive_drift_wave_operator",
    "drift_wave_adiabatic_frequency",
    "interchange_operator",
    "interchange_growth_rate",
    "full_drb_resistive_drift_wave_operator",
]


def shear_alfven_operator(k_par, k_perp, alfven_speed, electron_skin_depth):
    """Two-field ``(phi, psi)`` shear-Alfven operator (see module docstring)."""

    k_par = jnp.asarray(k_par, dtype=jnp.float64)
    k_perp = jnp.asarray(k_perp, dtype=jnp.float64)
    v_a = jnp.asarray(alfven_speed, dtype=jnp.float64)
    d_e = jnp.asarray(electron_skin_depth, dtype=jnp.float64)
    inertia = 1.0 + (k_perp * d_e) ** 2
    return jnp.array(
        [
            [0.0, 1j * v_a**2 * k_par],
            [1j * k_par / inertia, 0.0],
        ],
        dtype=jnp.complex128,
    )


def shear_alfven_frequency(k_par, k_perp, alfven_speed, electron_skin_depth):
    """Analytic shear-Alfven angular frequency ``omega(k)`` with electron inertia."""

    k_par = jnp.asarray(k_par, dtype=jnp.float64)
    k_perp = jnp.asarray(k_perp, dtype=jnp.float64)
    v_a = jnp.asarray(alfven_speed, dtype=jnp.float64)
    d_e = jnp.asarray(electron_skin_depth, dtype=jnp.float64)
    return k_par * v_a / jnp.sqrt(1.0 + (k_perp * d_e) ** 2)


def resistive_drift_wave_operator(k_y, kperp2, adiabaticity, gradient):
    """Two-field ``(phi, n)`` Hasegawa-Wakatani operator (see module docstring)."""

    k_y = jnp.asarray(k_y, dtype=jnp.float64)
    kperp2 = jnp.asarray(kperp2, dtype=jnp.float64)
    alpha = jnp.asarray(adiabaticity, dtype=jnp.float64)
    kappa = jnp.asarray(gradient, dtype=jnp.float64)
    return jnp.array(
        [
            [-alpha / kperp2, alpha / kperp2],
            [alpha - 1j * kappa * k_y, -alpha],
        ],
        dtype=jnp.complex128,
    )


def drift_wave_adiabatic_frequency(k_y, kperp2, gradient):
    """Adiabatic-limit drift-wave frequency ``omega_star = kappa k_y/(1+kperp2)``."""

    k_y = jnp.asarray(k_y, dtype=jnp.float64)
    kperp2 = jnp.asarray(kperp2, dtype=jnp.float64)
    kappa = jnp.asarray(gradient, dtype=jnp.float64)
    return kappa * k_y / (1.0 + kperp2)


def interchange_operator(k_y, kperp2, gravity, gradient):
    """Curvature-driven interchange (Rayleigh-Taylor) operator in ``(phi, n)``.

    The ideal two-field interchange model in the drift plane,

        d/dt lap(phi) = -g d/dy n,   d/dt n = -kappa d/dy phi,

    reduces for a single Fourier mode to
    ``A = [[0, i g k_y / kperp2], [-i kappa k_y, 0]]``, whose eigenvalues are
    ``lambda = +/- sqrt(g kappa) |k_y| / sqrt(kperp2)``. With bad curvature
    (``g kappa > 0``) one eigenvalue is real and positive -- the interchange
    instability; with good curvature it is a stable oscillation. ``g`` is the
    effective gravity (curvature/grad-B drive) and ``kappa`` the background
    density gradient. This is the linear physics behind SOL blob propagation.
    """

    k_y = jnp.asarray(k_y, dtype=jnp.float64)
    kperp2 = jnp.asarray(kperp2, dtype=jnp.float64)
    g = jnp.asarray(gravity, dtype=jnp.float64)
    kappa = jnp.asarray(gradient, dtype=jnp.float64)
    return jnp.array(
        [
            [0.0, 1j * g * k_y / kperp2],
            [-1j * kappa * k_y, 0.0],
        ],
        dtype=jnp.complex128,
    )


def interchange_growth_rate(k_y, kperp2, gravity, gradient):
    """Analytic interchange growth rate ``sqrt(g kappa) |k_y| / sqrt(kperp2)``.

    Real and positive for bad curvature (``g kappa > 0``); returns 0 for good
    curvature, where the mode is a stable oscillation instead.
    """

    k_y = jnp.asarray(k_y, dtype=jnp.float64)
    kperp2 = jnp.asarray(kperp2, dtype=jnp.float64)
    g = jnp.asarray(gravity, dtype=jnp.float64)
    kappa = jnp.asarray(gradient, dtype=jnp.float64)
    drive = g * kappa
    return jnp.where(drive > 0, jnp.sqrt(jnp.abs(drive)) * jnp.abs(k_y) / jnp.sqrt(kperp2), 0.0)


def full_drb_resistive_drift_wave_operator(
    k_y,
    kperp2,
    k_par,
    rho_star,
    B0,
    gradient,
    tau,
    mu,
    nu,
):
    """Continuous six-field electrostatic DRB slab operator.

    The perturbation convention is
    ``q ~ exp(lambda*t + i*k_y*y + i*k_par*z)`` with state order
    ``q = (delta_n, delta_Te, delta_Ti, delta_Vi, delta_Ve, delta_omega)``.
    It is derived directly from the continuous legacy EB equations on a flat,
    constant-``B0`` slab about ``n0(x)=1+gradient*x``, ``Te0=Ti0=1`` and zero
    flow.  The polarization closure is eliminated analytically using
    ``delta_phi = -delta_omega/kperp2 - tau*delta_Ti``.  This sign follows the
    Boussinesq balance ``L_perp(phi) = omega - tau*L_perp(Ti)`` and matches the
    production residual ``-L_perp(phi) - tau*L_perp(Ti) + omega = 0``.

    This routine intentionally has no dependency on the production spatial
    discretization or RHS implementation so it can serve as an independent
    linear-physics reference.
    """

    k_y = jnp.asarray(k_y, dtype=jnp.float64)
    kperp2 = jnp.asarray(kperp2, dtype=jnp.float64)
    k_par = jnp.asarray(k_par, dtype=jnp.float64)
    rho_star = jnp.asarray(rho_star, dtype=jnp.float64)
    b0 = jnp.asarray(B0, dtype=jnp.float64)
    gradient = jnp.asarray(gradient, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    D = 1j * k_par
    d = k_y * gradient / (rho_star * b0)
    two_thirds = jnp.asarray(2.0 / 3.0, dtype=jnp.float64)
    return jnp.array(
        [
            [0, 0, -1j * d * tau, 0, -D, -1j * d / kperp2],
            [0, 0, 0, two_thirds * 0.71 * D, -two_thirds * 1.71 * D, 0],
            [0, 0, 0, 0, -two_thirds * D, 0],
            [-(1 + tau) * D, -D, -tau * D, 0, 0, 0],
            [
                -mu * D,
                -mu * 1.71 * D,
                -mu * tau * D,
                mu * nu,
                -mu * nu,
                -mu * D / kperp2,
            ],
            [0, 0, 0, b0**2 * D, -b0**2 * D, 0],
        ],
        dtype=jnp.complex128,
    )
