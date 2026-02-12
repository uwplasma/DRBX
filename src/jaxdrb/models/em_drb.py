from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.models.cold_ion_drb import Equilibrium, phi_from_omega
from jaxdrb.models.params import DRBParams
from jaxdrb.models.bcs import bc_relaxation_1d
from jaxdrb.models.braginskii import chi_par_Te as chi_par_Te_eff
from jaxdrb.models.braginskii import eta_parallel as eta_parallel_eff
from jaxdrb.models.braginskii import nu_par_i as nu_par_i_eff
from jaxdrb.models.sheath import (
    apply_loizu_mpse_boundary_conditions,
    apply_loizu2012_mpse_full_linear_bc,
    sheath_energy_losses,
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


class RHSDecomposition(eqx.Module):
    """Split of the EM RHS into conservative/source/dissipative parts."""

    conservative: State
    source: State
    dissipative: State

    def total(self) -> State:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: State, b: State) -> State:
    return State(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        psi=a.psi + b.psi,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_zeros_like(y: State) -> State:
    z = jnp.zeros_like(y.n)
    return State(n=z, omega=z, psi=z, vpar_i=z, Te=z)


def rhs_nonlinear_decomposed(
    t: float,
    y: State,
    params: DRBParams,
    geom,
    *,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> RHSDecomposition:
    """Return split RHS terms for EM DRB in flux-tube form."""

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
        n=y.n,
        non_boussinesq_perturbed_density_on=bool(
            getattr(params, "non_boussinesq_perturbed_density_on", False)
        ),
    )

    dpar = geom.dpar
    C = geom.curvature

    def d2par(f: jnp.ndarray) -> jnp.ndarray:
        return dpar(dpar(f))

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

    grad_par_phi_pe = dpar(phi - y.n - float(params.alpha_Te_ohm) * y.Te)
    coef = 0.5 * getattr(params, "beta", 0.0) + params.me_hat * jnp.maximum(k2, params.kperp2_min)
    coef = jnp.maximum(coef, 1e-12)
    eta_eff = jnp.maximum(eta_parallel_eff(params, eq, Te_state=y.Te), 1e-12)

    conservative = State(
        n=-dpar(vpar_e),
        omega=dpar(jpar),
        psi=-grad_par_phi_pe / coef,
        vpar_i=-dpar(phi),
        Te=-(2.0 / 3.0) * dpar(vpar_e),
    )

    source = State(
        n=drive_n + (C_p - C_phi),
        omega=C_p,
        psi=jnp.zeros_like(y.psi),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=drive_Te + C_T,
    )

    dissipative = State(
        n=params.Dn * lap_n - float(getattr(params, "nu_sink_n", 0.0)) * y.n,
        omega=params.DOmega * lap_omega,
        psi=(-eta_eff * jpar + getattr(params, "Dpsi", 0.0) * lap_psi) / coef,
        vpar_i=nu_par_i_eff(params, eq, Te_state=y.Te) * d2par(y.vpar_i)
        - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_i,
        Te=params.DTe * lap_Te
        + chi_par_Te_eff(params, eq, Te_state=y.Te) * d2par(y.Te)
        - float(getattr(params, "nu_sink_Te", 0.0)) * y.Te,
    )

    if int(getattr(params, "sheath_bc_model", 0)) == 1:
        dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc = apply_loizu2012_mpse_full_linear_bc(
            params=params,
            geom=geom,
            eq=eq,
            kperp2=k2,
            phi=phi,
            n=y.n,
            omega=y.omega,
            vpar_e=vpar_e,
            vpar_i=y.vpar_i,
            Te=y.Te,
            dpar=dpar,
            d2par=d2par,
        )
        dissipative = _state_add(
            dissipative,
            State(
                n=dn_bc,
                omega=domega_bc,
                psi=jnp.zeros_like(y.psi),
                vpar_i=dvpar_i_bc,
                Te=dTe_bc,
            ),
        )
        djpar = dvpar_i_bc - dvpar_e_bc
        k2_safe = jnp.maximum(k2, float(getattr(params, "kperp2_min", 1e-6)))
        dissipative = _state_add(
            dissipative,
            State(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                psi=djpar / k2_safe,
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=jnp.zeros_like(y.Te),
            ),
        )
    else:
        dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
            params=params, geom=geom, eq=eq, phi=phi, vpar_e=vpar_e, vpar_i=y.vpar_i, Te=y.Te
        )
        dissipative = _state_add(
            dissipative,
            State(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                psi=jnp.zeros_like(y.psi),
                vpar_i=dvpar_i_sh,
                Te=jnp.zeros_like(y.Te),
            ),
        )
        djpar = dvpar_i_sh - dvpar_e_sh
        k2_safe = jnp.maximum(k2, float(getattr(params, "kperp2_min", 1e-6)))
        dissipative = _state_add(
            dissipative,
            State(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                psi=djpar / k2_safe,
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=jnp.zeros_like(y.Te),
            ),
        )

    dTe_sh, _ = sheath_energy_losses(params=params, geom=geom, Te=y.Te)
    dissipative = _state_add(
        dissipative,
        State(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            psi=jnp.zeros_like(y.psi),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=dTe_sh,
        ),
    )

    if bool(getattr(params, "sheath_end_damp_on", False)):
        bc = sheath_bc_rate(params, geom)
        if bc is not None:
            nu_bc, mask = bc
            dissipative = _state_add(
                dissipative,
                State(
                    n=-nu_bc * mask * y.n,
                    omega=-nu_bc * mask * y.omega,
                    psi=-nu_bc * mask * y.psi,
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=-nu_bc * mask * y.Te,
                ),
            )

    nu_loss = sheath_loss_rate(params, geom)
    dissipative = _state_add(
        dissipative,
        State(
            n=-nu_loss * y.n,
            omega=-nu_loss * y.omega,
            psi=-nu_loss * y.psi,
            vpar_i=-nu_loss * y.vpar_i,
            Te=-nu_loss * y.Te,
        ),
    )

    if getattr(params, "line_bcs", None) is not None and params.line_bcs.enabled:
        dl = float(geom.dl)
        dissipative = _state_add(
            dissipative,
            State(
                n=bc_relaxation_1d(y.n, bc=params.line_bcs.n, dl=dl),
                omega=bc_relaxation_1d(y.omega, bc=params.line_bcs.omega, dl=dl),
                psi=bc_relaxation_1d(y.psi, bc=params.line_bcs.psi, dl=dl),
                vpar_i=bc_relaxation_1d(y.vpar_i, bc=params.line_bcs.vpar_i, dl=dl),
                Te=bc_relaxation_1d(y.Te, bc=params.line_bcs.Te, dl=dl),
            ),
        )

    return RHSDecomposition(conservative=conservative, source=source, dissipative=dissipative)


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

    This is a compact inductive extension intended to expose finite-beta effects while preserving
    the matrix-free solver workflow. It uses:

    - Ampere closure: j_parallel = -∇_⊥^2 psi = +k_⊥^2 psi
    - Induction/Ohm-like equation: (beta/2 + m_hat_e k_⊥^2) ∂_t psi = -∇_||(phi - n - Te) - eta j_||

    The electron parallel velocity is eliminated using j_|| = v_||i - v_||e.
    """
    split = rhs_nonlinear_decomposed(t, y, params, geom, kx=kx, ky=ky, eq=eq)
    if not bool(getattr(params, "operator_split_on", False)):
        return split.total()

    out = _state_zeros_like(y)
    if bool(getattr(params, "operator_conservative_on", True)):
        out = _state_add(out, split.conservative)
    if bool(getattr(params, "operator_source_on", True)):
        out = _state_add(out, split.source)
    if bool(getattr(params, "operator_dissipative_on", True)):
        out = _state_add(out, split.dissipative)
    return out


def equilibrium(nl: int, dtype=jnp.complex128) -> State:
    return State.zeros(nl, dtype=dtype)
