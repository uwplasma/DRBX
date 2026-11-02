"""Electron sheath heat transmission against Stangeby (25.44-25.46) and the current-mode flux contract."""

import math

import jax.numpy as jnp
import pytest

from drbx.fci_braginskii.native.fci_wall_closure import SheathClosure, current_gamma_e, sheath_lambda, wall_face_fluxes


def stangeby(ti_over_te, mi_over_me, delta=0.0):
    return 2.0 / (1.0 - delta) - 0.5 * math.log(2.0 * math.pi * (1.0 + ti_over_te) / (mi_over_me * (1.0 - delta) ** 2))


@pytest.mark.parametrize("ti_over_te", [0.0, 1.0])
@pytest.mark.parametrize("mi_over_me", [1836.0, 3671.0])
@pytest.mark.parametrize("delta", [0.0, 0.3])
def test_floating_gamma_e_matches_stangeby(ti_over_te, mi_over_me, delta):
    closure = SheathClosure(secondary_electron_emission=delta)
    expected = stangeby(ti_over_te, mi_over_me, delta)
    assert float(closure.gamma_e(ti_over_te, mi_over_me)) == pytest.approx(expected, rel=1e-12)
    # current mode: 2/(1-d) + phi/Te at the floating drop is the same number, so phi is the entrance-to-wall drop only
    lam = float(sheath_lambda(ti_over_te, mi_over_me, delta))
    assert float(current_gamma_e(lam, delta)) == pytest.approx(expected, rel=1e-12)


def test_hydrogen_value_range():
    assert 4.4 < stangeby(1.0, 1836.0) < stangeby(0.0, 1836.0) < 5.3


def test_current_mode_electron_energy_flux_uses_electron_particle_flux():
    state = jnp.array([0.5, 2.0, 2.0, 1.4, 1.4])  # n, Te, Ti, Vi, Ve at the wall
    j = 0.3
    flux = wall_face_fluxes(state, 1.0, 1.0, 4.5, 2.5, wall_current=j)
    gamma_i_flux = 0.5 * 1.4  # ion particle flux n Vi
    assert float(flux.particle) == pytest.approx(gamma_i_flux - j)  # Gamma_e = Gamma_i - j
    assert float(flux.electron_energy) == pytest.approx(4.5 * 2.0 * (gamma_i_flux - j))
