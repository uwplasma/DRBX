from __future__ import annotations

import jax
import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import (
    Equilibrium,
    State,
    rhs_nonlinear,
    rhs_nonlinear_decomposed,
)
from jaxdrb.models.invariants import cold_ion_invariant_rates_from_rhs
from jaxdrb.models.params import DRBParams


def _allclose_state(a: State, b: State, *, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    return (
        np.allclose(np.asarray(a.n), np.asarray(b.n), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.omega), np.asarray(b.omega), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.vpar_e), np.asarray(b.vpar_e), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.vpar_i), np.asarray(b.vpar_i), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.Te), np.asarray(b.Te), atol=atol, rtol=rtol)
    )


def test_operator_split_reconstructs_legacy_rhs_with_all_toggles_on() -> None:
    nl = 48
    geom = SlabGeometry.make(nl=nl, shat=0.7, curvature0=0.22)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = State.random(jax.random.key(11), nl, amplitude=1e-2)

    base = DRBParams(
        omega_n=1.1,
        omega_Te=0.6,
        eta=0.8,
        me_hat=0.2,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.02,
        DTe=0.03,
        chi_par_Te=0.1,
        nu_par_e=0.04,
        nu_par_i=0.05,
        nu_sink_n=0.01,
        nu_sink_Te=0.02,
        nu_sink_vpar=0.03,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    legacy = rhs_nonlinear(0.0, y, base, geom, kx=0.0, ky=0.4, eq=eq)
    split_on = DRBParams(
        **{
            **base.__dict__,
            "operator_split_on": True,
            "operator_conservative_on": True,
            "operator_source_on": True,
            "operator_dissipative_on": True,
        }
    )
    reconstructed = rhs_nonlinear(0.0, y, split_on, geom, kx=0.0, ky=0.4, eq=eq)
    assert _allclose_state(legacy, reconstructed)


def test_operator_split_all_disabled_returns_zero_rhs() -> None:
    nl = 32
    geom = SlabGeometry.make(nl=nl, shat=0.2, curvature0=0.1)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = State.random(jax.random.key(21), nl, amplitude=1e-2)
    params = DRBParams(
        omega_n=0.9,
        omega_Te=0.2,
        eta=1.0,
        me_hat=0.2,
        operator_split_on=True,
        operator_conservative_on=False,
        operator_source_on=False,
        operator_dissipative_on=False,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.3, eq=eq)
    z = np.zeros_like(np.asarray(y.n))
    assert np.allclose(np.asarray(dy.n), z, atol=0.0, rtol=0.0)
    assert np.allclose(np.asarray(dy.omega), z, atol=0.0, rtol=0.0)
    assert np.allclose(np.asarray(dy.vpar_e), z, atol=0.0, rtol=0.0)
    assert np.allclose(np.asarray(dy.vpar_i), z, atol=0.0, rtol=0.0)
    assert np.allclose(np.asarray(dy.Te), z, atol=0.0, rtol=0.0)


def test_operator_split_conservative_only_keeps_invariant_rates_roundoff_small() -> None:
    nl = 64
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = State.random(jax.random.key(31), nl, amplitude=1e-2)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=False,
        operator_dissipative_on=False,
    )
    dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.35, eq=eq)
    rates = cold_ion_invariant_rates_from_rhs(
        y, dy, params=params, geom=geom, kx=0.0, ky=0.35, eq=eq
    )
    assert abs(float(rates["denergy_dt"])) < 8e-8
    assert abs(float(rates["dmass_dt"])) < 1e-11
    assert abs(float(rates["dcharge_dt"])) < 1e-11
    assert abs(float(rates["dcurrent_dt"])) < 1e-11
    assert abs(float(rates["dmomentum_dt"])) < 1e-11


def test_decomposition_sum_matches_rhs_total() -> None:
    nl = 40
    geom = SlabGeometry.make(nl=nl, shat=0.5, curvature0=0.2)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = State.random(jax.random.key(41), nl, amplitude=1e-2)
    params = DRBParams(
        omega_n=1.0,
        omega_Te=0.3,
        eta=0.7,
        me_hat=0.2,
        curvature_on=True,
        Dn=0.02,
        DOmega=0.01,
        DTe=0.03,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    split = rhs_nonlinear_decomposed(0.0, y, params, geom, kx=0.0, ky=0.25, eq=eq)
    total = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=0.25, eq=eq)
    assert _allclose_state(split.total(), total)
