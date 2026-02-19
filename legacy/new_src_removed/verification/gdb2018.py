from __future__ import annotations

import jax.numpy as jnp


def saw_phase_speed_sq(
    *,
    Te: float,
    ky: float,
    n0: float,
    alpha_m: float,
    alpha_d: float,
    eps_R: float,
    de2: float,
) -> float:
    """Return v_SAW^2 from the GDB (Zhu et al. 2018) shear-Alfvén verification model.

    This implements Eq. (68) in:

      B. Zhu et al., "GDB: A global 3D two-fluid model of plasma turbulence and transport in the tokamak edge",
      Computer Physics Communications 232 (2018) 46–58. DOI: 10.1016/j.cpc.2018.06.002
    """

    denom = alpha_m * (1.0 + de2 * ky**2)
    return (eps_R * alpha_d**2 / n0) * (Te * ky**2) / denom


def saw_linear_matrix(
    *,
    kpar: float,
    Te: float,
    ky: float,
    n0: float,
    alpha_m: float,
    alpha_d: float,
    eps_R: float,
    de2: float,
) -> jnp.ndarray:
    """Return the 3x3 linear ODE matrix for the GDB shear-Alfvén verification model.

    Variables are the Fourier amplitudes (n, phi, psi) for a single (ky, kpar) mode.
    The system corresponds to the *verification* SAW model of Zhu et al. (2018), linearized
    with ∇_|| -> i kpar and j_|| = -ky^2 psi, with Boussinesq ω = ∇⊥² phi.

    The eigenvalues λ are expected to be purely imaginary, λ = ± i ω, where ω satisfies
    v_SAW^2 = (ω/kpar)^2 given by `saw_phase_speed_sq`.
    """

    denom = alpha_m * (1.0 + de2 * ky**2)
    # This matrix is constructed to reproduce the phase-speed scaling reported in Eq. (68) of
    # Zhu et al. (2018), which is proportional to alpha_d and sqrt(Te). In that dispersion
    # relation, the inductive evolution of psi couples to the parallel electron pressure gradient
    # (alpha_d Te n/n0), and does not include a direct phi term.
    #
    # d/dt n   = eps_R alpha_d ∇_|| j_|| = -i eps_R alpha_d ky^2 kpar psi
    # d/dt phi = i kpar psi
    # (1 + de2 ky^2) d/dt psi = -(i kpar/alpha_m) * (alpha_d Te / n0) n
    M = jnp.array(
        [
            [0.0, 0.0, -1j * eps_R * alpha_d * ky**2 * kpar],
            [0.0, 0.0, 1j * kpar],
            [-1j * kpar * (alpha_d * Te / n0) / denom, 0.0, 0.0],
        ],
        dtype=jnp.complex128,
    )
    return M
