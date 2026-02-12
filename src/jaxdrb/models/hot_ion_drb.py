from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from jaxdrb.models.cold_ion_drb import Equilibrium, phi_from_omega
from jaxdrb.models.params import DRBParams
from jaxdrb.models.bcs import bc_relaxation_1d
from jaxdrb.models.braginskii import chi_par_Te as chi_par_Te_eff
from jaxdrb.models.braginskii import chi_par_Ti as chi_par_Ti_eff
from jaxdrb.models.braginskii import eta_parallel as eta_parallel_eff
from jaxdrb.models.braginskii import nu_par_e as nu_par_e_eff
from jaxdrb.models.braginskii import nu_par_i as nu_par_i_eff
from jaxdrb.models.sheath import (
    apply_loizu_mpse_boundary_conditions,
    apply_loizu2012_mpse_full_linear_bc_hot_ion,
    sheath_energy_losses,
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


class RHSDecomposition(eqx.Module):
    """Split of the hot-ion RHS into conservative/source/dissipative parts."""

    conservative: State
    source: State
    dissipative: State

    def total(self) -> State:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: State, b: State) -> State:
    return State(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        Ti=a.Ti + b.Ti,
    )


def _state_zeros_like(y: State) -> State:
    z = jnp.zeros_like(y.n)
    return State(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, Ti=z)


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
    """Return split RHS terms for hot-ion DRB in flux-tube form."""

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

    use_algebraic_ohm = params.me_hat == 0.0

    drive_n = -1j * ky * params.omega_n * phi
    drive_Te = -1j * ky * params.omega_Te * phi
    drive_Ti = -1j * ky * getattr(params, "omega_Ti", 0.0) * phi

    tau_i = getattr(params, "tau_i", 0.0)
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

    grad_par_phi_pe = dpar(phi - y.n - float(params.alpha_Te_ohm) * y.Te)
    eta_eff = jnp.maximum(eta_parallel_eff(params, eq, Te_state=y.Te), 1e-12)
    vpar_e_eff = jnp.where(use_algebraic_ohm, y.vpar_i + grad_par_phi_pe / eta_eff, y.vpar_e)
    jpar = y.vpar_i - vpar_e_eff

    conservative = State(
        n=-dpar(vpar_e_eff),
        omega=dpar(jpar),
        vpar_e=jnp.where(
            use_algebraic_ohm,
            jnp.zeros_like(y.vpar_e),
            grad_par_phi_pe / jnp.maximum(float(params.me_hat), 1e-12),
        ),
        vpar_i=-dpar(phi + tau_i * (y.n + y.Ti)),
        Te=-(2.0 / 3.0) * dpar(vpar_e_eff),
        Ti=-(2.0 / 3.0) * dpar(y.vpar_i),
    )

    source = State(
        n=drive_n + (C_p - C_phi),
        omega=C_p,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=drive_Te + C_T,
        Ti=drive_Ti,
    )

    if use_algebraic_ohm:
        dvpar_e_eta = -eta_eff * (y.vpar_e - vpar_e_eff)
    else:
        dvpar_e_eta = -(eta_eff * (y.vpar_e - y.vpar_i)) / jnp.maximum(float(params.me_hat), 1e-12)

    DTi = getattr(params, "DTi", params.DTe)
    dissipative = State(
        n=params.Dn * lap_n - float(getattr(params, "nu_sink_n", 0.0)) * y.n,
        omega=params.DOmega * lap_omega,
        vpar_e=dvpar_e_eta
        + nu_par_e_eff(params, eq, Te_state=y.Te) * d2par(y.vpar_e)
        - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_e,
        vpar_i=nu_par_i_eff(params, eq, Te_state=y.Te, Ti_state=y.Ti) * d2par(y.vpar_i)
        - float(getattr(params, "nu_sink_vpar", 0.0)) * y.vpar_i,
        Te=params.DTe * lap_Te
        + chi_par_Te_eff(params, eq, Te_state=y.Te) * d2par(y.Te)
        - float(getattr(params, "nu_sink_Te", 0.0)) * y.Te,
        Ti=DTi * lap_Ti + chi_par_Ti_eff(params, eq, Te_state=y.Te, Ti_state=y.Ti) * d2par(y.Ti),
    )

    if int(getattr(params, "sheath_bc_model", 0)) == 1:
        dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc, dTi_bc = (
            apply_loizu2012_mpse_full_linear_bc_hot_ion(
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
                Ti=y.Ti,
                dpar=dpar,
                d2par=d2par,
            )
        )
        dissipative = _state_add(
            dissipative,
            State(
                n=dn_bc,
                omega=domega_bc,
                vpar_e=dvpar_e_bc,
                vpar_i=dvpar_i_bc,
                Te=dTe_bc,
                Ti=dTi_bc,
            ),
        )
    else:
        dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
            params=params,
            geom=geom,
            eq=eq,
            phi=phi,
            vpar_e=vpar_e_eff,
            vpar_i=y.vpar_i,
            Te=y.Te,
            Ti=y.Ti,
        )
        dissipative = _state_add(
            dissipative,
            State(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                vpar_e=dvpar_e_sh,
                vpar_i=dvpar_i_sh,
                Te=jnp.zeros_like(y.Te),
                Ti=jnp.zeros_like(y.Ti),
            ),
        )

    dTe_sh, dTi_sh = sheath_energy_losses(params=params, geom=geom, Te=y.Te, Ti=y.Ti)
    dissipative = _state_add(
        dissipative,
        State(
            n=jnp.zeros_like(y.n),
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=dTe_sh,
            Ti=jnp.zeros_like(y.Ti) if dTi_sh is None else dTi_sh,
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
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=-nu_bc * mask * y.Te,
                    Ti=-nu_bc * mask * y.Ti,
                ),
            )

    nu_loss = sheath_loss_rate(params, geom)
    dissipative = _state_add(
        dissipative,
        State(
            n=-nu_loss * y.n,
            omega=-nu_loss * y.omega,
            vpar_e=-nu_loss * y.vpar_e,
            vpar_i=-nu_loss * y.vpar_i,
            Te=-nu_loss * y.Te,
            Ti=-nu_loss * y.Ti,
        ),
    )

    if getattr(params, "line_bcs", None) is not None and params.line_bcs.enabled:
        dl = float(geom.dl)
        dissipative = _state_add(
            dissipative,
            State(
                n=bc_relaxation_1d(y.n, bc=params.line_bcs.n, dl=dl),
                omega=bc_relaxation_1d(y.omega, bc=params.line_bcs.omega, dl=dl),
                vpar_e=bc_relaxation_1d(y.vpar_e, bc=params.line_bcs.vpar_e, dl=dl),
                vpar_i=bc_relaxation_1d(y.vpar_i, bc=params.line_bcs.vpar_i, dl=dl),
                Te=bc_relaxation_1d(y.Te, bc=params.line_bcs.Te, dl=dl),
                Ti=bc_relaxation_1d(y.Ti, bc=params.line_bcs.Ti, dl=dl),
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
    """Hot-ion drift-reduced Braginskii-like RHS (electrostatic, flux-tube, single-(kx,ky)).

    This extends the cold-ion model by:

    - adding an ion temperature field `Ti`,
    - including an ion pressure contribution in the ion parallel momentum,
    - including ion pressure in the curvature drive through the total pressure perturbation.

    The implementation is intentionally compact (to keep the core solver matrix-free and
    differentiable) while providing a hot-ion branch that supports regression tests and
    literature-aligned trend scans. For scope/roadmap items (e.g. additional closures in 3D),
    see `docs/model/limitations.md`.
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
