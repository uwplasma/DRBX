from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.models.params import DRBParams


class Equilibrium(eqx.Module):
    """Background profiles along the field line used by the RHS.

    The evolving `State` is interpreted as a perturbation about this equilibrium.
    """

    n0: jnp.ndarray
    Te0: jnp.ndarray

    @classmethod
    def constant(
        cls,
        nl: int,
        *,
        n0: float = 1.0,
        Te0: float = 1.0,
        dtype=jnp.float64,
    ) -> "Equilibrium":
        return cls(
            n0=jnp.full((nl,), float(n0), dtype=dtype),
            Te0=jnp.full((nl,), float(Te0), dtype=dtype),
        )


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


def phi_from_omega(
    omega: jnp.ndarray,
    kperp2: jnp.ndarray,
    *,
    kperp2_min: float,
    boussinesq: bool,
    n0: jnp.ndarray | None = None,
    n0_min: float = 1e-6,
) -> jnp.ndarray:
    k2 = jnp.maximum(kperp2, kperp2_min)
    if boussinesq:
        return -omega / k2
    if n0 is None:
        raise ValueError("Non-Boussinesq polarization requires an equilibrium density n0.")
    return -omega / (k2 * jnp.maximum(n0, n0_min))


def rhs_nonlinear(
    t: float,
    y: State,
    params: DRBParams,
    geom,
    *,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> State:
    """Cold-ion drift-reduced Braginskii-like RHS in flux-tube (single-(kx,ky)) form.

    For a single Fourier mode, the nonlinear Poisson bracket self-interaction vanishes, so this
    implementation is linear in `y` but kept in this form for future extension.
    """

    k2 = geom.kperp2(kx, ky)
    if eq is None:
        eq = Equilibrium.constant(int(y.n.size), n0=1.0, Te0=1.0)
    phi = phi_from_omega(
        y.omega,
        k2,
        kperp2_min=params.kperp2_min,
        boussinesq=params.boussinesq,
        n0=eq.n0,
        n0_min=params.n0_min,
    )

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
        C_T = (2.0 / 3.0) * C(kx, ky, (7.0 / 2.0) * y.Te + y.n - phi)
    else:
        C_phi = jnp.zeros_like(phi)
        C_p = jnp.zeros_like(phi)
        C_T = jnp.zeros_like(phi)

    # Perp diffusion in Fourier space: D * ∇_⊥^2 f -> -D k_⊥^2 f
    lap_n = -k2 * y.n
    lap_omega = -k2 * y.omega
    lap_Te = -k2 * y.Te

    # --- Model ---
    # Continuity
    # Many SOL/edge DRB conventions use a curvature-compressibility term C(p) - C(phi),
    # consistent with e.g. Mosetto et al. (2012) in the cold-ion limit.
    dn = drive_n - dpar(y.vpar_e) + (C_p - C_phi) + params.Dn * lap_n

    # Vorticity
    domega = dpar(jpar) + C_p + params.DOmega * lap_omega

    # Electron parallel momentum (Ohm's law + inertia)
    # me_hat d/dt v_e = -∇_||(phi - n - Te) - eta (v_e - v_i)
    grad_par_phi_pe = dpar(phi - y.n - float(params.alpha_Te_ohm) * y.Te)
    dvpar_e = (grad_par_phi_pe - params.eta * (y.vpar_e - y.vpar_i)) / params.me_hat

    # Ion parallel momentum (cold ions)
    dvpar_i = -dpar(phi)

    # Electron temperature
    dTe = drive_Te + C_T - (2.0 / 3.0) * dpar(y.vpar_e) + params.DTe * lap_Te

    # Optional SOL/sheath closure (open-field-line geometries only).
    #
    # For linear stability studies, a common reduced approach is to represent end-plate
    # losses with a volumetric loss rate nu_sh ~ 2 c_s / L_parallel. In our normalization
    # c_s is O(1), so we use nu_sh ~ 2/L_parallel with an optional multiplier.
    if getattr(params, "sheath_on", False) and hasattr(geom, "sheath_mask"):
        Lpar = jnp.abs(jnp.asarray(geom.l[-1] - geom.l[0], dtype=jnp.float64)) + 1e-30
        nu = float(getattr(params, "sheath_nu_factor", 1.0)) * (2.0 / Lpar)

        dn = dn - nu * y.n
        dTe = dTe - nu * y.Te
        domega = domega - nu * y.omega
        dvpar_e = dvpar_e - nu * y.vpar_e
        dvpar_i = dvpar_i - nu * y.vpar_i

    return State(n=dn, omega=domega, vpar_e=dvpar_e, vpar_i=dvpar_i, Te=dTe)


def equilibrium(nl: int, dtype=jnp.complex128) -> State:
    return State.zeros(nl, dtype=dtype)


def default_equilibrium(nl: int, *, n0: float = 1.0) -> Equilibrium:
    return Equilibrium.constant(nl, n0=n0, Te0=1.0)
