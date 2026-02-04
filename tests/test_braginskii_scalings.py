from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jaxdrb.models.braginskii import (
    chi_par_Te,
    chi_par_Ti,
    eta_parallel,
    nu_par_e,
    nu_par_i,
    smooth_floor,
)
from jaxdrb.models.cold_ion_drb import Equilibrium
from jaxdrb.models.params import DRBParams


def test_smooth_floor_is_differentiable() -> None:
    def f(x):
        return jnp.sum(smooth_floor(x, floor=1.0, width=0.1))

    g = jax.grad(f)(jnp.array([-2.0, 0.5, 1.0, 2.0]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_spitzer_eta_scaling_uses_equilibrium_Te0() -> None:
    Te0 = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
    eq = Equilibrium(n0=jnp.ones_like(Te0), Te0=Te0)

    params = DRBParams(
        eta=2.0,
        braginskii_on=True,
        braginskii_eta_on=True,
        braginskii_Tref=1.0,
        braginskii_T_floor=1e-6,
        braginskii_T_smooth=1e-6,
    )

    eta_eff = np.asarray(eta_parallel(params, eq))
    expected = 2.0 * (1.0 / np.asarray(Te0)) ** 1.5
    assert np.allclose(eta_eff, expected, rtol=0.0, atol=1e-12)


def test_spitzer_eta_scaling_is_differentiable_wrt_eta0() -> None:
    Te0 = jnp.array([0.8, 1.2], dtype=jnp.float64)
    eq = Equilibrium(n0=jnp.ones_like(Te0), Te0=Te0)

    def f(eta0):
        p = DRBParams(
            eta=eta0,
            braginskii_on=True,
            braginskii_eta_on=True,
            braginskii_Tref=1.0,
            braginskii_T_floor=1e-6,
            braginskii_T_smooth=1e-6,
        )
        return jnp.sum(eta_parallel(p, eq))

    g = float(jax.grad(f)(2.0))
    expected = float(np.sum((1.0 / np.asarray(Te0)) ** 1.5))
    assert np.isclose(g, expected, rtol=0.0, atol=1e-12)


def test_kappa_and_viscosity_scalings_use_T_pow_5_2() -> None:
    Te0 = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
    eq = Equilibrium(n0=jnp.ones_like(Te0), Te0=Te0)

    params = DRBParams(
        tau_i=1.0,
        chi_par_Te=3.0,
        chi_par_Ti=5.0,
        nu_par_e=7.0,
        nu_par_i=11.0,
        braginskii_on=True,
        braginskii_kappa_e_on=True,
        braginskii_kappa_i_on=True,
        braginskii_visc_e_on=True,
        braginskii_visc_i_on=True,
        braginskii_Tref=1.0,
        braginskii_T_floor=1e-6,
        braginskii_T_smooth=1e-6,
    )

    # Electron conduction/viscosity scale with Te0^{5/2}.
    Te_pow = np.asarray(Te0) ** 2.5
    assert np.allclose(np.asarray(chi_par_Te(params, eq)), 3.0 * Te_pow, rtol=0.0, atol=1e-12)
    assert np.allclose(np.asarray(nu_par_e(params, eq)), 7.0 * Te_pow, rtol=0.0, atol=1e-12)

    # Ion conduction/viscosity scale with Ti0^{5/2} where Ti0 = tau_i * Te0.
    Ti0 = np.asarray(Te0)
    Ti_pow = Ti0**2.5
    assert np.allclose(np.asarray(chi_par_Ti(params, eq)), 5.0 * Ti_pow, rtol=0.0, atol=1e-12)
    assert np.allclose(np.asarray(nu_par_i(params, eq)), 11.0 * Ti_pow, rtol=0.0, atol=1e-12)
