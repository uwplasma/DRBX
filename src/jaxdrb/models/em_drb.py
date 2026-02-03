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
    """Electromagnetic extension state (reduced, Ampère-closed).

    This model eliminates `vpar_e` in favor of an inductive variable `psi ~ -A_parallel`
    together with an Ampère closure for the parallel current.
    """

    n: jnp.ndarray
    omega: jnp.ndarray
    psi: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray

    @classmethod
    def zeros(cls, nl: int, dtype=jnp.complex128) -> "State":
        z = jnp.zeros((nl,), dtype=dtype)
        return cls(n=z, omega=z, psi=z, vpar_i=z, Te=z)

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
            psi=cplx(keys[4], keys[5]),
            vpar_i=cplx(keys[6], keys[7]),
            Te=cplx(keys[8], keys[9]),
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
    """Electromagnetic drift-reduced Braginskii-like RHS (flux-tube, single-(kx,ky)).

    This is a minimal inductive extension intended to expose qualitative finite-beta trends while
    preserving the matrix-free solver workflow. It uses:

    - Ampère closure: j_parallel = -∇_⊥^2 psi = +k_⊥^2 psi
    - Induction/Ohm-like equation: (β/2 + m̂_e k_⊥^2) ∂_t psi = -∇_||(phi - n - Te) - η j_||

    The electron parallel velocity is eliminated using j_|| = v_||i - v_||e.
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

    # Ampère closure: j_|| = -∇_⊥^2 psi -> +k_⊥^2 psi
    jpar = k2 * y.psi
    vpar_e = y.vpar_i - jpar

    drive_n = -1j * ky * params.omega_n * phi
    drive_Te = -1j * ky * params.omega_Te * phi

    if params.curvature_on:
        C_phi = C(kx, ky, phi)
        C_p = C(kx, ky, y.n + y.Te)
        C_T = (2.0 / 3.0) * C(kx, ky, (7.0 / 2.0) * y.Te + y.n - phi)
    else:
        C_phi = jnp.zeros_like(phi)
        C_p = jnp.zeros_like(phi)
        C_T = jnp.zeros_like(phi)

    lap_n = -k2 * y.n
    lap_omega = -k2 * y.omega
    lap_Te = -k2 * y.Te
    lap_psi = -k2 * y.psi

    # Continuity
    dn = drive_n - dpar(vpar_e) + (C_p - C_phi) + params.Dn * lap_n
    dn = dn - float(getattr(params, "nu_sink_n", 0.0)) * y.n

    # Vorticity (current coupling via Ampère)
    domega = dpar(jpar) + C_p + params.DOmega * lap_omega

    # Induction / generalized Ohm
    grad_par_phi_pe = dpar(phi - y.n - float(params.alpha_Te_ohm) * y.Te)
    coef = 0.5 * getattr(params, "beta", 0.0) + params.me_hat * jnp.maximum(k2, params.kperp2_min)
    coef = jnp.maximum(coef, 1e-12)
    dpsi = (-grad_par_phi_pe - params.eta * jpar + getattr(params, "Dpsi", 0.0) * lap_psi) / coef

    # Ion parallel momentum (cold ions)
    dvpar_i = -dpar(phi)
    dvpar_i = dvpar_i + float(getattr(params, "nu_par_i", 0.0)) * d2par(y.vpar_i)
    dvpar_i = dvpar_i - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_i

    # Electron temperature (using vpar_e reconstructed from vpar_i and jpar)
    dTe = drive_Te + C_T - (2.0 / 3.0) * dpar(vpar_e) + params.DTe * lap_Te
    dTe = dTe + float(getattr(params, "chi_par_Te", 0.0)) * d2par(y.Te)
    dTe = dTe - float(getattr(params, "nu_sink_Te", 0.0)) * y.Te

    # Loizu-style MPSE sheath BCs: use reconstructed vpar_e = vpar_i - jpar and enforce the BCs
    # through equivalent relaxation on (vpar_i, psi).
    dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
        params=params, geom=geom, eq=eq, phi=phi, vpar_e=vpar_e, vpar_i=y.vpar_i, Te=y.Te
    )
    dvpar_i = dvpar_i + dvpar_i_sh
    # Convert dvpar_e contribution into a psi relaxation via jpar = k2 * psi and vpar_e = vpar_i - jpar:
    # dvpar_e = dvpar_i - d(jpar)  ->  d(jpar) = dvpar_i - dvpar_e.
    djpar = dvpar_i_sh - dvpar_e_sh
    k2_safe = jnp.maximum(k2, float(getattr(params, "kperp2_min", 1e-6)))
    dpsi = dpsi + (djpar / k2_safe)

    bc = sheath_bc_rate(params, geom)
    if bc is not None:
        nu_bc, mask = bc
        dn = dn - nu_bc * mask * y.n
        dTe = dTe - nu_bc * mask * y.Te
        domega = domega - nu_bc * mask * y.omega
        dpsi = dpsi - nu_bc * mask * y.psi

    # Optional volumetric loss proxy.
    nu_loss = sheath_loss_rate(params, geom)
    dn = dn - nu_loss * y.n
    domega = domega - nu_loss * y.omega
    dpsi = dpsi - nu_loss * y.psi
    dvpar_i = dvpar_i - nu_loss * y.vpar_i
    dTe = dTe - nu_loss * y.Te

    return State(n=dn, omega=domega, psi=dpsi, vpar_i=dvpar_i, Te=dTe)


def equilibrium(nl: int, dtype=jnp.complex128) -> State:
    return State.zeros(nl, dtype=dtype)
