from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.params import DRBParams


def _base_params(**overrides) -> DRBParams:
    base = dict(
        omega_n=0.0,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=False,
        kperp2_min=1e-8,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    return DRBParams(**{**base, **overrides})


def test_non_boussinesq_perturbed_density_toggle_changes_rhs() -> None:
    nl = 32
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    l = np.asarray(geom.l)

    y = State(
        n=(0.3 * np.sin(l)).astype(np.complex128),
        omega=(0.2 * np.cos(2.0 * l)).astype(np.complex128),
        vpar_e=np.zeros(nl, dtype=np.complex128),
        vpar_i=np.zeros(nl, dtype=np.complex128),
        Te=np.zeros(nl, dtype=np.complex128),
    )

    r_eq = rhs_nonlinear(
        0.0,
        y,
        _base_params(omega_n=0.7, non_boussinesq_perturbed_density_on=False),
        geom,
        kx=0.0,
        ky=0.4,
        eq=eq,
    )
    r_state = rhs_nonlinear(
        0.0,
        y,
        _base_params(omega_n=0.7, non_boussinesq_perturbed_density_on=True),
        geom,
        kx=0.0,
        ky=0.4,
        eq=eq,
    )

    rel = np.linalg.norm(np.asarray(r_state.n - r_eq.n)) / (
        np.linalg.norm(np.asarray(r_eq.n)) + 1e-30
    )
    assert rel > 1e-4


def test_state_dependent_braginskii_toggle_changes_rhs() -> None:
    nl = 32
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    l = np.asarray(geom.l)

    y = State(
        n=np.zeros(nl, dtype=np.complex128),
        omega=np.zeros(nl, dtype=np.complex128),
        vpar_e=(0.2 * np.sin(l)).astype(np.complex128),
        vpar_i=(0.1 * np.cos(2.0 * l)).astype(np.complex128),
        Te=(0.25 + 0.1 * np.sin(l)).astype(np.complex128),
    )

    params_eq = _base_params(
        boussinesq=True,
        eta=2.0,
        braginskii_on=True,
        braginskii_eta_on=True,
        braginskii_state_dependent_on=False,
        braginskii_Tref=1.0,
        braginskii_T_floor=1e-6,
        braginskii_T_smooth=1e-6,
    )
    params_state = _base_params(
        boussinesq=True,
        eta=2.0,
        braginskii_on=True,
        braginskii_eta_on=True,
        braginskii_state_dependent_on=True,
        braginskii_Tref=1.0,
        braginskii_T_floor=1e-6,
        braginskii_T_smooth=1e-6,
    )

    r_eq = rhs_nonlinear(0.0, y, params_eq, geom, kx=0.0, ky=0.3, eq=eq)
    r_state = rhs_nonlinear(0.0, y, params_state, geom, kx=0.0, ky=0.3, eq=eq)

    rel = np.linalg.norm(np.asarray(r_state.vpar_e - r_eq.vpar_e)) / (
        np.linalg.norm(np.asarray(r_eq.vpar_e)) + 1e-30
    )
    assert rel > 1e-4


def test_state_dependent_toggles_preserve_finite_rhs_short_step() -> None:
    nl = 32
    geom = SlabGeometry.make(nl=nl, shat=0.1, curvature0=0.2)
    eq = Equilibrium.constant(nl, n0=1.2, Te0=0.8)
    l = np.asarray(geom.l)
    y = State(
        n=(0.2 * np.sin(l)).astype(np.complex128),
        omega=(0.1 * np.cos(2.0 * l)).astype(np.complex128),
        vpar_e=(0.08 * np.sin(3.0 * l)).astype(np.complex128),
        vpar_i=(0.09 * np.cos(l)).astype(np.complex128),
        Te=(0.15 * np.sin(2.0 * l)).astype(np.complex128),
    )

    params = _base_params(
        omega_n=0.4,
        omega_Te=0.3,
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        braginskii_on=True,
        braginskii_state_dependent_on=True,
        braginskii_T_floor=1e-3,
        braginskii_T_smooth=1e-3,
        chi_par_Te=0.2,
        nu_par_e=0.1,
        nu_par_i=0.1,
    )

    r = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.4, eq=eq)
    for field in (r.n, r.omega, r.vpar_e, r.vpar_i, r.Te):
        assert bool(jnp.all(jnp.isfinite(jnp.real(field))))
        assert bool(jnp.all(jnp.isfinite(jnp.imag(field))))
