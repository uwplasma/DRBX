from __future__ import annotations

import numpy as np

from jax_drb.native.recycling_atomic import (
    amjuel_energy_loss,
    amjuel_reaction_rate,
    amjuel_reaction_rate_and_energy_loss,
    eval_amjuel_fit,
    eval_openadas_rate,
    hydrogen_cx_sigmav,
    load_amjuel_rate,
    load_openadas_rate,
    openadas_energy_loss,
    openadas_reaction_rate,
)


def test_load_amjuel_rate_tables_are_packaged() -> None:
    coeffs, energy_coeffs, heating = load_amjuel_rate("d", "iz")

    assert coeffs.shape == (9, 9)
    assert energy_coeffs.shape == (9, 9)
    assert np.isfinite(heating)


def test_load_openadas_rate_tables_are_packaged() -> None:
    rate_coeffs, radiation_coeffs, log_temperature, log_density, electron_heating = load_openadas_rate("ne", "iz")

    assert rate_coeffs.shape == (30, 24)
    assert radiation_coeffs.shape == (30, 24)
    assert log_temperature.shape == (30,)
    assert log_density.shape == (24,)
    assert electron_heating < 0.0


def test_eval_amjuel_fit_returns_positive_finite_rates() -> None:
    coeffs, _, _ = load_amjuel_rate("d", "iz")

    values = eval_amjuel_fit(
        np.full((2, 2), 5.0, dtype=np.float64),
        np.full((2, 2), 2.0e18, dtype=np.float64),
        coeffs,
    )

    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_eval_openadas_rate_returns_positive_finite_rates() -> None:
    coeffs, _, log_temperature, log_density, _ = load_openadas_rate("ne", "iz")

    values = eval_openadas_rate(
        np.full((2, 2), 5.0, dtype=np.float64),
        np.full((2, 2), 2.0e18, dtype=np.float64),
        coeffs,
        log_temperature=log_temperature,
        log_density=log_density,
    )

    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_atomic_rate_helpers_return_finite_positive_arrays() -> None:
    dataset_scalars = {"Tnorm": 10.0, "Nnorm": 1.0e19, "Omega_ci": 2.0e6}
    heavy_density = np.full((2, 2), 0.5, dtype=np.float64)
    electron_density = np.full((2, 2), 0.6, dtype=np.float64)
    electron_temperature = np.full((2, 2), 0.7, dtype=np.float64)

    sigma_v_coeffs, sigma_v_E_coeffs, electron_heating = load_amjuel_rate("d", "iz")
    amjuel_rate = amjuel_reaction_rate(
        heavy_density,
        electron_density,
        electron_temperature,
        sigma_v_coeffs,
        dataset_scalars,
    )
    amjuel_loss = amjuel_energy_loss(
        heavy_density,
        electron_density,
        electron_temperature,
        sigma_v_E_coeffs,
        electron_heating,
        amjuel_rate,
        dataset_scalars,
    )
    paired_amjuel_rate, paired_amjuel_loss = amjuel_reaction_rate_and_energy_loss(
        heavy_density,
        electron_density,
        electron_temperature,
        sigma_v_coeffs,
        sigma_v_E_coeffs,
        electron_heating,
        dataset_scalars,
    )

    openadas_rate = openadas_reaction_rate(
        heavy_density,
        electron_density,
        electron_temperature,
        "ne",
        "iz",
        dataset_scalars,
    )
    openadas_loss = openadas_energy_loss(
        heavy_density,
        electron_density,
        electron_temperature,
        "ne",
        "iz",
        reaction_rate=openadas_rate,
        dataset_scalars=dataset_scalars,
    )

    assert np.all(np.isfinite(amjuel_rate))
    assert np.all(np.isfinite(amjuel_loss))
    assert np.all(np.isfinite(openadas_rate))
    assert np.all(np.isfinite(openadas_loss))
    assert np.all(amjuel_rate > 0.0)
    assert np.all(openadas_rate > 0.0)
    np.testing.assert_allclose(paired_amjuel_rate, amjuel_rate, rtol=1.0e-13, atol=0.0)
    np.testing.assert_allclose(paired_amjuel_loss, amjuel_loss, rtol=1.0e-13, atol=0.0)


def test_hydrogen_cx_sigmav_returns_positive_finite_values() -> None:
    dataset_scalars = {"Nnorm": 1.0e19, "Omega_ci": 2.0e6}

    sigmav = hydrogen_cx_sigmav(np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float64), dataset_scalars)

    assert np.all(np.isfinite(sigmav))
    assert np.all(sigmav > 0.0)
