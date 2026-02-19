from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.models.cold_ion_drb import Equilibrium, State, phi_from_omega, rhs_nonlinear
from jaxdrb.models.params import DRBParams
from jaxdrb.models.hot_ion_drb import State as HotState
from jaxdrb.models.em_drb import State as EMState


def _mean_abs2(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.real(jnp.conj(z) * z))


def cold_ion_energy_functional(
    y: State,
    *,
    params: DRBParams,
    geom,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> jnp.ndarray:
    """Quadratic cold-ion DRB energy functional for conservative-gate checks.

    This diagnostic is intended for the *periodic, conservative subset* of the cold-ion DRB model:

    - no drives (`omega_n=omega_Te=0`),
    - no curvature,
    - no sinks/diffusion/sheath/source terms,
    - `me_hat > 0` (inertial Ohm's law).

    In that subset, the functional is:

      E = 0.5 < |n|^2 + k_perp^2 |phi|^2 + m_e |v_par,e|^2 + |v_par,i|^2 + c_T |Te|^2 >

    with `c_T = 1.5 * alpha_Te_ohm`, which matches the coupling used in the implemented
    electron-momentum and temperature equations.
    """
    if eq is None:
        eq = Equilibrium.constant(int(y.n.size), n0=1.0, Te0=1.0)

    k2 = geom.kperp2(kx, ky)
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
    c_T = 1.5 * float(getattr(params, "alpha_Te_ohm", 1.71))
    return 0.5 * (
        _mean_abs2(y.n)
        + jnp.mean(k2 * jnp.real(jnp.conj(phi) * phi))
        + float(params.me_hat) * _mean_abs2(y.vpar_e)
        + _mean_abs2(y.vpar_i)
        + c_T * _mean_abs2(y.Te)
    )


def cold_ion_invariants(
    y: State,
    *,
    params: DRBParams,
    geom,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> dict[str, jnp.ndarray]:
    """Return conservative diagnostics for the cold-ion DRB branch.

    Reported quantities:
    - `energy`: quadratic functional from `cold_ion_energy_functional`.
    - `mass`: `<Re[n]>`.
    - `charge`: `<Re[omega]>` (polarization/charge-balance proxy in this reduced model).
    - `current`: `<Re[j_par]>` where `j_par = v_par,i - v_par,e`.
    - `momentum`: `<Re[v_par,i + m_e v_par,e]>` (parallel momentum proxy).
    """
    jpar = y.vpar_i - y.vpar_e
    return {
        "energy": cold_ion_energy_functional(y, params=params, geom=geom, kx=kx, ky=ky, eq=eq),
        "mass": jnp.mean(jnp.real(y.n)),
        "charge": jnp.mean(jnp.real(y.omega)),
        "current": jnp.mean(jnp.real(jpar)),
        "momentum": jnp.mean(jnp.real(y.vpar_i + float(params.me_hat) * y.vpar_e)),
    }


def cold_ion_invariant_rates_from_rhs(
    y: State,
    dy: State,
    *,
    params: DRBParams,
    geom,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> dict[str, jnp.ndarray]:
    """Return instantaneous invariant rates from a state and its RHS.

    For the periodic Boussinesq cold-ion DRB subset this provides an exact discrete
    chain-rule estimate for the quadratic functional:

      dE/dt = Re< n* dn - phi* domega + me v_e* dv_e + v_i* dv_i + c_T Te* dTe >.

    For non-Boussinesq runs, we fall back to a centered finite-difference directional
    derivative of `cold_ion_energy_functional` along `dy`.
    """
    if eq is None:
        eq = Equilibrium.constant(int(y.n.size), n0=1.0, Te0=1.0)

    k2 = geom.kperp2(kx, ky)
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
    c_T = 1.5 * float(getattr(params, "alpha_Te_ohm", 1.71))
    if params.boussinesq:
        denergy = jnp.mean(
            jnp.real(
                jnp.conj(y.n) * dy.n
                - jnp.conj(phi) * dy.omega
                + float(params.me_hat) * jnp.conj(y.vpar_e) * dy.vpar_e
                + jnp.conj(y.vpar_i) * dy.vpar_i
                + c_T * jnp.conj(y.Te) * dy.Te
            )
        )
    else:
        eps = jnp.asarray(1.0e-7, dtype=jnp.float64)
        y_plus = State(
            n=y.n + eps * dy.n,
            omega=y.omega + eps * dy.omega,
            vpar_e=y.vpar_e + eps * dy.vpar_e,
            vpar_i=y.vpar_i + eps * dy.vpar_i,
            Te=y.Te + eps * dy.Te,
        )
        y_minus = State(
            n=y.n - eps * dy.n,
            omega=y.omega - eps * dy.omega,
            vpar_e=y.vpar_e - eps * dy.vpar_e,
            vpar_i=y.vpar_i - eps * dy.vpar_i,
            Te=y.Te - eps * dy.Te,
        )
        E_plus = cold_ion_energy_functional(y_plus, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
        E_minus = cold_ion_energy_functional(y_minus, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
        denergy = (E_plus - E_minus) / (2.0 * eps)

    return {
        "denergy_dt": denergy,
        "dmass_dt": jnp.mean(jnp.real(dy.n)),
        "dcharge_dt": jnp.mean(jnp.real(dy.omega)),
        "dcurrent_dt": jnp.mean(jnp.real(dy.vpar_i - dy.vpar_e)),
        "dmomentum_dt": jnp.mean(jnp.real(dy.vpar_i + float(params.me_hat) * dy.vpar_e)),
    }


def cold_ion_operator_residuals(
    y: State,
    *,
    t: float,
    params: DRBParams,
    geom,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute instantaneous invariant-rate residuals of the cold-ion DRB RHS operator."""
    if eq is None:
        eq = Equilibrium.constant(int(y.n.size), n0=1.0, Te0=1.0)
    dy = rhs_nonlinear(t, y, params, geom, kx=kx, ky=ky, eq=eq)
    return cold_ion_invariant_rates_from_rhs(y, dy, params=params, geom=geom, kx=kx, ky=ky, eq=eq)


def hot_ion_mean_rates_from_rhs(
    y: HotState,
    dy: HotState,
    *,
    params: DRBParams,
    geom,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> dict[str, jnp.ndarray]:
    """Return mean-rate invariants for the hot-ion branch (conservative subset checks)."""
    return {
        "dmass_dt": jnp.mean(jnp.real(dy.n)),
        "dcharge_dt": jnp.mean(jnp.real(dy.omega)),
        "dcurrent_dt": jnp.mean(jnp.real(dy.vpar_i - dy.vpar_e)),
        "dmomentum_dt": jnp.mean(jnp.real(dy.vpar_i + float(params.me_hat) * dy.vpar_e)),
    }


def em_mean_rates_from_rhs(
    y: EMState,
    dy: EMState,
    *,
    params: DRBParams,
    geom,
    kx: float,
    ky: float,
    eq: Equilibrium | None = None,
) -> dict[str, jnp.ndarray]:
    """Return mean-rate invariants for the EM branch (conservative subset checks)."""
    k2 = geom.kperp2(kx, ky)
    djpar = k2 * dy.psi
    dvpar_e = dy.vpar_i - djpar
    return {
        "dmass_dt": jnp.mean(jnp.real(dy.n)),
        "dcharge_dt": jnp.mean(jnp.real(dy.omega)),
        "dcurrent_dt": jnp.mean(jnp.real(djpar)),
        "dmomentum_dt": jnp.mean(jnp.real(dy.vpar_i + float(params.me_hat) * dvpar_e)),
    }
