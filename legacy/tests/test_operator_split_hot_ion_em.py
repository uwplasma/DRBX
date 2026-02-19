from __future__ import annotations

import jax
import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.hot_ion_drb import (
    Equilibrium as HotEquilibrium,
    State as HotState,
    rhs_nonlinear as hot_rhs,
)
from jaxdrb.models.em_drb import Equilibrium as EMEquilibrium
from jaxdrb.models.em_drb import State as EMState
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs
from jaxdrb.models.params import DRBParams


def _allclose_hot(a: HotState, b: HotState, *, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    return (
        np.allclose(np.asarray(a.n), np.asarray(b.n), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.omega), np.asarray(b.omega), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.vpar_e), np.asarray(b.vpar_e), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.vpar_i), np.asarray(b.vpar_i), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.Te), np.asarray(b.Te), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.Ti), np.asarray(b.Ti), atol=atol, rtol=rtol)
    )


def _allclose_em(a: EMState, b: EMState, *, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    return (
        np.allclose(np.asarray(a.n), np.asarray(b.n), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.omega), np.asarray(b.omega), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.psi), np.asarray(b.psi), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.vpar_i), np.asarray(b.vpar_i), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(a.Te), np.asarray(b.Te), atol=atol, rtol=rtol)
    )


def test_hot_ion_split_parity_full_rhs() -> None:
    nl = 48
    geom = SlabGeometry.make(nl=nl, shat=0.6, curvature0=0.25)
    eq = HotEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = HotState.random(jax.random.key(101), nl, amplitude=1e-2)

    base = DRBParams(
        omega_n=0.9,
        omega_Te=0.4,
        omega_Ti=0.2,
        eta=0.6,
        me_hat=0.2,
        tau_i=0.7,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.02,
        DTe=0.03,
        DTi=0.02,
        chi_par_Te=0.05,
        chi_par_Ti=0.04,
        nu_par_e=0.03,
        nu_par_i=0.02,
        nu_sink_n=0.01,
        nu_sink_Te=0.02,
        nu_sink_vpar=0.03,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )

    legacy = hot_rhs(0.0, y, base, geom, kx=0.0, ky=0.35, eq=eq)
    split_on = DRBParams(
        **{
            **base.__dict__,
            "operator_split_on": True,
            "operator_conservative_on": True,
            "operator_source_on": True,
            "operator_dissipative_on": True,
        }
    )
    reconstructed = hot_rhs(0.0, y, split_on, geom, kx=0.0, ky=0.35, eq=eq)
    assert _allclose_hot(legacy, reconstructed)


def test_em_split_parity_full_rhs() -> None:
    nl = 48
    geom = SlabGeometry.make(nl=nl, shat=0.4, curvature0=0.2)
    eq = EMEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = EMState.random(jax.random.key(201), nl, amplitude=1e-2)

    base = DRBParams(
        omega_n=0.8,
        omega_Te=0.5,
        eta=0.7,
        me_hat=0.15,
        beta=0.4,
        Dpsi=0.01,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.02,
        DTe=0.03,
        chi_par_Te=0.05,
        nu_par_i=0.02,
        nu_sink_n=0.01,
        nu_sink_Te=0.02,
        nu_sink_vpar=0.03,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )

    legacy = em_rhs(0.0, y, base, geom, kx=0.0, ky=0.25, eq=eq)
    split_on = DRBParams(
        **{
            **base.__dict__,
            "operator_split_on": True,
            "operator_conservative_on": True,
            "operator_source_on": True,
            "operator_dissipative_on": True,
        }
    )
    reconstructed = em_rhs(0.0, y, split_on, geom, kx=0.0, ky=0.25, eq=eq)
    assert _allclose_em(legacy, reconstructed)
