from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.models.cold_ion_drb import Equilibrium, phi_from_omega
from jaxdrb.models.params import DRBParams
from jaxdrb.models.sheath import (
    apply_loizu_mpse_boundary_conditions,
    sheath_bc_rate,
    sheath_loss_rate,
)


class State(eqx.Module):
    """Hot-ion electrostatic extension state (adds an ion-temperature field Ti)."""

    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray

    @classmethod
    def zeros(cls, nl: int, dtype=jnp.complex128) -> "State":
        z = jnp.zeros((nl,), dtype=dtype)
        return cls(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, Ti=z)

    @classmethod
    def random(
        cls,
        key: jax.Array,
        nl: int,
        *,
        amplitude: float = 1e-3,
        dtype=jnp.complex128,
    ) -> "State":
        keys = jr.split(key, 12)

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
            Ti=cplx(keys[10], keys[11]),
        )


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
    """Hot-ion drift-reduced Braginskii-like RHS (electrostatic, flux-tube, single-(kx,ky)).

    This extends the cold-ion model by:

    - adding an ion temperature field `Ti`,
    - including an ion pressure contribution in the ion parallel momentum,
    - including ion pressure in the curvature drive through the total pressure perturbation.

    The implementation is intentionally minimal and primarily intended for qualitative trend studies.
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

    use_algebraic_ohm = params.me_hat == 0.0

    drive_n = -1j * ky * params.omega_n * phi
    drive_Te = -1j * ky * params.omega_Te * phi
    drive_Ti = -1j * ky * getattr(params, "omega_Ti", 0.0) * phi

    tau_i = getattr(params, "tau_i", 0.0)

    # Total pressure perturbation for curvature forcing.
    # For tau_i=0 this reduces to (n + Te), matching the cold-ion model.
    p_tot = (1.0 + tau_i) * y.n + y.Te + tau_i * y.Ti

    if params.curvature_on:
        C_phi = C(kx, ky, phi)
        C_p = C(kx, ky, p_tot)
        C_T = (2.0 / 3.0) * C(kx, ky, (7.0 / 2.0) * y.Te + y.n - phi)
    else:
        C_phi = jnp.zeros_like(phi)
        C_p = jnp.zeros_like(phi)
        C_T = jnp.zeros_like(phi)

    lap_n = -k2 * y.n
    lap_omega = -k2 * y.omega
    lap_Te = -k2 * y.Te
    lap_Ti = -k2 * y.Ti

    # Continuity: use constrained vpar_e in the me_hat=0 limit.
    grad_par_phi_pe = dpar(phi - y.n - float(params.alpha_Te_ohm) * y.Te)
    eta_eff = jnp.maximum(params.eta, 1e-12)
    vpar_e_eff = jnp.where(use_algebraic_ohm, y.vpar_i + grad_par_phi_pe / eta_eff, y.vpar_e)

    jpar = y.vpar_i - vpar_e_eff
    dn = drive_n - dpar(vpar_e_eff) + (C_p - C_phi) + params.Dn * lap_n
    dn = dn - float(getattr(params, "nu_sink_n", 0.0)) * y.n

    # Vorticity
    domega = dpar(jpar) + C_p + params.DOmega * lap_omega

    # Electron parallel momentum
    if use_algebraic_ohm:
        dvpar_e = -eta_eff * (y.vpar_e - vpar_e_eff)
    else:
        dvpar_e = (grad_par_phi_pe - params.eta * (y.vpar_e - y.vpar_i)) / params.me_hat
    dvpar_e = dvpar_e + float(getattr(params, "nu_par_e", 0.0)) * d2par(y.vpar_e)
    dvpar_e = dvpar_e - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_e

    # Ion parallel momentum with ion pressure (tau_i=0 recovers cold ions).
    dvpar_i = -dpar(phi + tau_i * (y.n + y.Ti))
    dvpar_i = dvpar_i + float(getattr(params, "nu_par_i", 0.0)) * d2par(y.vpar_i)
    dvpar_i = dvpar_i - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_i

    # Electron temperature
    dTe = drive_Te + C_T - (2.0 / 3.0) * dpar(vpar_e_eff) + params.DTe * lap_Te
    dTe = dTe + float(getattr(params, "chi_par_Te", 0.0)) * d2par(y.Te)
    dTe = dTe - float(getattr(params, "nu_sink_Te", 0.0)) * y.Te

    # Ion temperature (minimal)
    DTi = getattr(params, "DTi", params.DTe)
    dTi = drive_Ti - (2.0 / 3.0) * dpar(y.vpar_i) + DTi * lap_Ti

    # Loizu-style MPSE sheath BCs (applied in the cold-ion sound-speed normalization).
    dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
        params=params, geom=geom, eq=eq, phi=phi, vpar_e=vpar_e_eff, vpar_i=y.vpar_i, Te=y.Te
    )
    dvpar_e = dvpar_e + dvpar_e_sh
    dvpar_i = dvpar_i + dvpar_i_sh

    bc = sheath_bc_rate(params, geom)
    if bc is not None:
        nu_bc, mask = bc
        dn = dn - nu_bc * mask * y.n
        dTe = dTe - nu_bc * mask * y.Te
        dTi = dTi - nu_bc * mask * y.Ti
        domega = domega - nu_bc * mask * y.omega

    # Optional volumetric loss proxy.
    nu_loss = sheath_loss_rate(params, geom)
    dn = dn - nu_loss * y.n
    domega = domega - nu_loss * y.omega
    dvpar_e = dvpar_e - nu_loss * y.vpar_e
    dvpar_i = dvpar_i - nu_loss * y.vpar_i
    dTe = dTe - nu_loss * y.Te
    dTi = dTi - nu_loss * y.Ti

    return State(n=dn, omega=domega, vpar_e=dvpar_e, vpar_i=dvpar_i, Te=dTe, Ti=dTi)


def equilibrium(nl: int, dtype=jnp.complex128) -> State:
    return State.zeros(nl, dtype=dtype)
