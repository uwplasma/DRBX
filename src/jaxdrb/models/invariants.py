from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.models.cold_ion_drb import Equilibrium, State, phi_from_omega
from jaxdrb.models.params import DRBParams


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
