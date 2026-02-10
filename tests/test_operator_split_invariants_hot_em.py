from __future__ import annotations

import jax

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.em_drb import Equilibrium as EMEquilibrium
from jaxdrb.models.em_drb import State as EMState
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs
from jaxdrb.models.hot_ion_drb import Equilibrium as HotEquilibrium
from jaxdrb.models.hot_ion_drb import State as HotState
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_rhs
from jaxdrb.models.invariants import em_mean_rates_from_rhs, hot_ion_mean_rates_from_rhs
from jaxdrb.models.params import DRBParams


def test_hot_ion_conservative_split_mean_rates_small() -> None:
    nl = 64
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = HotEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = HotState.random(jax.random.key(401), nl, amplitude=1e-2)

    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        eta=0.0,
        me_hat=0.2,
        tau_i=0.7,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        chi_par_Te=0.0,
        chi_par_Ti=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        nu_sink_n=0.0,
        nu_sink_Te=0.0,
        nu_sink_vpar=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=False,
        operator_dissipative_on=False,
    )

    dy = hot_rhs(0.0, y, params, geom, kx=0.0, ky=0.35, eq=eq)
    rates = hot_ion_mean_rates_from_rhs(y, dy, params=params, geom=geom, kx=0.0, ky=0.35, eq=eq)

    assert abs(float(rates["dmass_dt"])) < 1e-11
    assert abs(float(rates["dcharge_dt"])) < 1e-11
    assert abs(float(rates["dcurrent_dt"])) < 1e-11
    assert abs(float(rates["dmomentum_dt"])) < 1e-11


def test_em_conservative_split_mean_rates_small() -> None:
    nl = 64
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = EMEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = EMState.random(jax.random.key(501), nl, amplitude=1e-2)

    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        beta=0.4,
        Dpsi=0.0,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_i=0.0,
        nu_sink_n=0.0,
        nu_sink_Te=0.0,
        nu_sink_vpar=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=False,
        operator_dissipative_on=False,
    )

    dy = em_rhs(0.0, y, params, geom, kx=0.0, ky=0.35, eq=eq)
    rates = em_mean_rates_from_rhs(y, dy, params=params, geom=geom, kx=0.0, ky=0.35, eq=eq)

    assert abs(float(rates["dmass_dt"])) < 1e-11
    assert abs(float(rates["dcharge_dt"])) < 1e-11
    assert abs(float(rates["dcurrent_dt"])) < 1e-11
    assert abs(float(rates["dmomentum_dt"])) < 1e-11
