from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.models.params import DRBParams


class State(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray

    @classmethod
    def zeros(cls, nl: int, dtype=jnp.complex128) -> "State":
        z = jnp.zeros((nl,), dtype=dtype)
        return cls(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z)

    @classmethod
    def random(
        cls,
        key: jax.Array,
        nl: int,
        *,
        amplitude: float = 1e-3,
        dtype=jnp.complex128,
    ) -> "State":
        keys = jr.split(key, 10)

        def cplx(kre, kim):
            re = jr.normal(kre, (nl,), dtype=jnp.float64)
            im = jr.normal(kim, (nl,), dtype=jnp.float64)
            z = re + 1j * im
            return (amplitude * z).astype(dtype)

        return cls(
            n=cplx(keys[0], keys[1]),
            omega=cplx(keys[2], keys[3]),
            vpar_e=cplx(keys[4], keys[5]),
            vpar_i=cplx(keys[6], keys[7]),
            Te=cplx(keys[8], keys[9]),
        )


def phi_from_omega(omega: jnp.ndarray, kperp2: jnp.ndarray, kperp2_min: float) -> jnp.ndarray:
    k2 = jnp.maximum(kperp2, kperp2_min)
    return -omega / k2


def rhs_nonlinear(
    t: float,
    y: State,
    params: DRBParams,
    geom,
    *,
    kx: float,
    ky: float,
) -> State:
    """Cold-ion drift-reduced Braginskii-like RHS in flux-tube (single-(kx,ky)) form.

    For a single Fourier mode, the nonlinear Poisson bracket self-interaction vanishes, so this
    v1 implementation is linear in `y` but kept in this form for future extension.
    """

    k2 = geom.kperp2(kx, ky)
    phi = phi_from_omega(y.omega, k2, params.kperp2_min)

    dpar = geom.dpar
    C = geom.curvature

    # Parallel current (n0 = 1 normalization)
    jpar = y.vpar_i - y.vpar_e

    # Drives from background gradients: -[phi, n0] -> -i ky omega_n phi
    drive_n = -1j * ky * params.omega_n * phi
    drive_Te = -1j * ky * params.omega_Te * phi

    # Curvature operators
    if params.curvature_on:
        C_phi = C(kx, ky, phi)
        C_p = C(kx, ky, y.n + y.Te)
    else:
        C_phi = jnp.zeros_like(phi)
        C_p = jnp.zeros_like(phi)

    # Perp diffusion in Fourier space: D * ∇_⊥^2 f -> -D k_⊥^2 f
    lap_n = -k2 * y.n
    lap_omega = -k2 * y.omega
    lap_Te = -k2 * y.Te

    # --- Model ---
    # Continuity
    dn = drive_n - dpar(y.vpar_e) + C_phi - C_p + params.Dn * lap_n

    # Vorticity
    domega = -dpar(jpar) + C_p + params.DOmega * lap_omega

    # Electron parallel momentum (Ohm's law + inertia)
    # me_hat d/dt v_e = -∇_||(phi - n - Te) - eta (v_e - v_i)
    grad_par_pe = dpar(phi - y.n - y.Te)
    dvpar_e = (-grad_par_pe - params.eta * (y.vpar_e - y.vpar_i)) / params.me_hat

    # Ion parallel momentum (cold ions)
    dvpar_i = -dpar(phi)

    # Electron temperature
    dTe = drive_Te - (2.0 / 3.0) * dpar(y.vpar_e) + params.DTe * lap_Te

    return State(n=dn, omega=domega, vpar_e=dvpar_e, vpar_i=dvpar_i, Te=dTe)


def equilibrium(nl: int, dtype=jnp.complex128) -> State:
    return State.zeros(nl, dtype=dtype)
