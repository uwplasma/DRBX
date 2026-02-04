from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.models.params import DRBParams
from jaxdrb.models.bcs import bc_relaxation_1d
from jaxdrb.models.sheath import (
    apply_loizu_mpse_boundary_conditions,
    apply_loizu2012_mpse_full_linear_bc,
    sheath_bc_rate,
    sheath_loss_rate,
)


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

    def d2par(f: jnp.ndarray) -> jnp.ndarray:
        return dpar(dpar(f))

    # Electron inertia handling:
    # - For me_hat > 0: evolve vpar_e with an inertial Ohm's law.
    # - For me_hat = 0: treat Ohm's law as an algebraic constraint and use a fast relaxation
    #   toward the constrained value (avoids division-by-zero and keeps the 5-field state shape).
    use_algebraic_ohm = params.me_hat == 0.0

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
    # vpar_e used in compressibility is the constrained value in the me_hat=0 limit.
    grad_par_phi_pe = dpar(phi - y.n - float(params.alpha_Te_ohm) * y.Te)
    eta_eff = jnp.maximum(params.eta, 1e-12)
    vpar_e_eff = jnp.where(use_algebraic_ohm, y.vpar_i + grad_par_phi_pe / eta_eff, y.vpar_e)

    # Parallel current (n0 = 1 normalization)
    jpar = y.vpar_i - vpar_e_eff

    dn = drive_n - dpar(vpar_e_eff) + (C_p - C_phi) + params.Dn * lap_n
    dn = dn - float(getattr(params, "nu_sink_n", 0.0)) * y.n

    # Vorticity
    domega = dpar(jpar) + C_p + params.DOmega * lap_omega

    # Electron parallel momentum (Ohm's law + inertia)
    # me_hat d/dt v_e = -∇_||(phi - n - Te) - eta (v_e - v_i)
    if use_algebraic_ohm:
        # Relax vpar_e toward the algebraic Ohm's-law constraint value.
        dvpar_e = -eta_eff * (y.vpar_e - vpar_e_eff)
    else:
        dvpar_e = (grad_par_phi_pe - params.eta * (y.vpar_e - y.vpar_i)) / params.me_hat
    dvpar_e = dvpar_e + float(getattr(params, "nu_par_e", 0.0)) * d2par(y.vpar_e)
    dvpar_e = dvpar_e - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_e

    # Ion parallel momentum (cold ions)
    dvpar_i = -dpar(phi)
    dvpar_i = dvpar_i + float(getattr(params, "nu_par_i", 0.0)) * d2par(y.vpar_i)
    dvpar_i = dvpar_i - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_i

    # Electron temperature
    dTe = drive_Te + C_T - (2.0 / 3.0) * dpar(vpar_e_eff) + params.DTe * lap_Te
    dTe = dTe + float(getattr(params, "chi_par_Te", 0.0)) * d2par(y.Te)
    dTe = dTe - float(getattr(params, "nu_sink_Te", 0.0)) * y.Te

    # Optional MPSE (sheath) boundary conditions for open field lines.
    # Model 0: velocity-only (legacy). Model 1: Loizu 2012 "full set" (linearized, model-aligned).
    if int(getattr(params, "sheath_bc_model", 0)) == 1:
        dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc = apply_loizu2012_mpse_full_linear_bc(
            params=params,
            geom=geom,
            eq=eq,
            kperp2=k2,
            phi=phi,
            n=y.n,
            omega=y.omega,
            vpar_e=vpar_e_eff,
            vpar_i=y.vpar_i,
            Te=y.Te,
            dpar=dpar,
            d2par=d2par,
        )
        dn = dn + dn_bc
        domega = domega + domega_bc
        dvpar_e = dvpar_e + dvpar_e_bc
        dvpar_i = dvpar_i + dvpar_i_bc
        dTe = dTe + dTe_bc
    else:
        dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
            params=params, geom=geom, eq=eq, phi=phi, vpar_e=vpar_e_eff, vpar_i=y.vpar_i, Te=y.Te
        )
        dvpar_e = dvpar_e + dvpar_e_sh
        dvpar_i = dvpar_i + dvpar_i_sh

    # Additional MPSE (sheath) sinks on open field lines: represent end-plate losses and current closure.
    bc = sheath_bc_rate(params, geom)
    if bc is not None:
        nu_bc, mask = bc
        dn = dn - nu_bc * mask * y.n
        dTe = dTe - nu_bc * mask * y.Te
        domega = domega - nu_bc * mask * y.omega

    # Optional volumetric sheath-loss proxy.
    nu_loss = sheath_loss_rate(params, geom)
    dn = dn - nu_loss * y.n
    domega = domega - nu_loss * y.omega
    dvpar_e = dvpar_e - nu_loss * y.vpar_e
    dvpar_i = dvpar_i - nu_loss * y.vpar_i
    dTe = dTe - nu_loss * y.Te

    # Optional user-defined boundary conditions along l (weak relaxation).
    if getattr(params, "line_bcs", None) is not None and params.line_bcs.enabled:
        dl = float(geom.dl)
        dn = dn + bc_relaxation_1d(y.n, bc=params.line_bcs.n, dl=dl)
        domega = domega + bc_relaxation_1d(y.omega, bc=params.line_bcs.omega, dl=dl)
        dvpar_e = dvpar_e + bc_relaxation_1d(y.vpar_e, bc=params.line_bcs.vpar_e, dl=dl)
        dvpar_i = dvpar_i + bc_relaxation_1d(y.vpar_i, bc=params.line_bcs.vpar_i, dl=dl)
        dTe = dTe + bc_relaxation_1d(y.Te, bc=params.line_bcs.Te, dl=dl)

    return State(n=dn, omega=domega, vpar_e=dvpar_e, vpar_i=dvpar_i, Te=dTe)


def equilibrium(nl: int, dtype=jnp.complex128) -> State:
    return State.zeros(nl, dtype=dtype)


def default_equilibrium(nl: int, *, n0: float = 1.0) -> Equilibrium:
    return Equilibrium.constant(nl, n0=n0, Te0=1.0)
