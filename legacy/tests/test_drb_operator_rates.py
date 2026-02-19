from __future__ import annotations

import jax
import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.invariants import (
    cold_ion_energy_functional,
    cold_ion_invariant_rates_from_rhs,
    cold_ion_operator_residuals,
)
from jaxdrb.models.params import DRBParams


def _conservative_params() -> DRBParams:
    return DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        alpha_Te_ohm=1.71,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=True,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )


def test_energy_rate_matches_centered_fd_for_rhs_direction() -> None:
    nl = 64
    kx = 0.0
    ky = 0.33
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = _conservative_params()
    y = State.random(jax.random.key(7), nl, amplitude=1e-2)
    dy = rhs_nonlinear(0.0, y, params, geom, kx=kx, ky=ky, eq=eq)

    rates = cold_ion_invariant_rates_from_rhs(y, dy, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
    dE = float(rates["denergy_dt"])

    eps = 1e-7
    y_plus = jax.tree_util.tree_map(lambda a, b: a + eps * b, y, dy)
    y_minus = jax.tree_util.tree_map(lambda a, b: a - eps * b, y, dy)
    E_plus = float(
        cold_ion_energy_functional(y_plus, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
    )
    E_minus = float(
        cold_ion_energy_functional(y_minus, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
    )
    dE_fd = (E_plus - E_minus) / (2.0 * eps)
    assert np.isclose(dE, dE_fd, rtol=5e-6, atol=2e-11)


def test_operator_residual_helper_matches_explicit_rhs_path() -> None:
    nl = 32
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = _conservative_params()
    y = State.random(jax.random.key(13), nl, amplitude=1e-2)
    ky = 0.29
    dy = rhs_nonlinear(0.0, y, params, geom, kx=0.0, ky=ky, eq=eq)
    a = cold_ion_invariant_rates_from_rhs(y, dy, params=params, geom=geom, kx=0.0, ky=ky, eq=eq)
    b = cold_ion_operator_residuals(y, t=0.0, params=params, geom=geom, kx=0.0, ky=ky, eq=eq)
    for key in ("denergy_dt", "dmass_dt", "dcharge_dt", "dcurrent_dt", "dmomentum_dt"):
        assert np.isclose(float(a[key]), float(b[key]), atol=1e-13, rtol=1e-13)


def test_operator_rates_stay_roundoff_scale_in_conservative_subset() -> None:
    nl = 48
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = _conservative_params()
    kys = (0.15, 0.35, 0.6)
    seeds = (1, 2, 3)

    max_abs = np.zeros(5, dtype=float)
    for seed in seeds:
        y = State.random(jax.random.key(seed), nl, amplitude=1e-2)
        for ky in kys:
            r = cold_ion_operator_residuals(
                y, t=0.0, params=params, geom=geom, kx=0.0, ky=ky, eq=eq
            )
            vals = np.asarray(
                [
                    abs(float(r["denergy_dt"])),
                    abs(float(r["dmass_dt"])),
                    abs(float(r["dcharge_dt"])),
                    abs(float(r["dcurrent_dt"])),
                    abs(float(r["dmomentum_dt"])),
                ]
            )
            max_abs = np.maximum(max_abs, vals)
    assert max_abs[0] < 7e-8
    assert max_abs[1] < 1e-11
    assert max_abs[2] < 1e-11
    assert max_abs[3] < 1e-11
    assert max_abs[4] < 1e-11
