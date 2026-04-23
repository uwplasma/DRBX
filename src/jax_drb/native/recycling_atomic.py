from __future__ import annotations

from functools import lru_cache
import json
from importlib import resources

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver


AMJUEL_FILENAMES = {
    ("d", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("d", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("t", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("t", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("he", "iz"): "iz_AMJUEL_H.x_2.3.9a.json",
    ("he", "rec"): "rec_AMJUEL_H.x_2.3.13a.json",
}

OPENADAS_FILENAMES = {
    ("ne", "iz"): ("scd96_ne.json", "plt96_ne.json", 0, -21.56),
    ("ne", "rec"): ("acd96_ne.json", "prb96_ne.json", 0, 21.56),
}


def charge_exchange_rate_multiplier(config: BoutConfig, *, atom_name: str) -> float:
    """Return the configured charge-exchange multiplier for one neutral species."""

    if not config.has_section(atom_name) or not config.has_option(atom_name, "K_cx_multiplier"):
        return 1.0
    return float(NumericResolver(config).resolve(atom_name, "K_cx_multiplier"))


def amjuel_reaction_rate(
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    sigma_v_coeffs: np.ndarray,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    """Evaluate an AMJUEL reaction rate on code-normalized fields."""

    sigma_v = eval_amjuel_fit(
        np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
        np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
        sigma_v_coeffs,
    )
    return np.asarray(heavy_density, dtype=np.float64) * np.asarray(electron_density, dtype=np.float64) * sigma_v * (
        dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"]
    )


def amjuel_energy_loss(
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    sigma_v_E_coeffs: np.ndarray,
    electron_heating: float,
    reaction_rate: np.ndarray,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    """Evaluate AMJUEL energy loss on code-normalized fields."""

    sigma_v_E = eval_amjuel_fit(
        np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
        np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
        sigma_v_E_coeffs,
    )
    energy_loss = (
        np.asarray(heavy_density, dtype=np.float64)
        * np.asarray(electron_density, dtype=np.float64)
        * sigma_v_E
        * dataset_scalars["Nnorm"]
        / (dataset_scalars["Tnorm"] * dataset_scalars["Omega_ci"])
    )
    return energy_loss - (electron_heating / dataset_scalars["Tnorm"]) * reaction_rate


def openadas_reaction_rate(
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    species_name: str,
    reaction_kind: str,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    """Evaluate an OpenADAS reaction rate on code-normalized fields."""

    rate_coeff, _, log_temperature, log_density, _ = load_openadas_rate(species_name, reaction_kind)
    rate = eval_openadas_rate(
        np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
        np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
        rate_coeff,
        log_temperature=log_temperature,
        log_density=log_density,
    )
    return (
        np.maximum(np.asarray(heavy_density, dtype=np.float64), 0.0)
        * np.maximum(np.asarray(electron_density, dtype=np.float64), 0.0)
        * rate
        * (dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"])
    )


def openadas_energy_loss(
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    species_name: str,
    reaction_kind: str,
    *,
    reaction_rate: np.ndarray,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    """Evaluate an OpenADAS radiation/energy-loss surface on normalized fields."""

    _, radiation_coeff, log_temperature, log_density, electron_heating = load_openadas_rate(species_name, reaction_kind)
    energy_loss_coeff = eval_openadas_rate(
        np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
        np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
        radiation_coeff,
        log_temperature=log_temperature,
        log_density=log_density,
    )
    energy_loss = (
        np.maximum(np.asarray(heavy_density, dtype=np.float64), 0.0)
        * np.maximum(np.asarray(electron_density, dtype=np.float64), 0.0)
        * energy_loss_coeff
        * dataset_scalars["Nnorm"]
        / (dataset_scalars["Tnorm"] * dataset_scalars["Omega_ci"])
    )
    return energy_loss - (electron_heating / dataset_scalars["Tnorm"]) * reaction_rate


@lru_cache(maxsize=None)
def load_amjuel_rate(species_name: str, reaction_kind: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load one packaged AMJUEL rate fit and its associated energy-loss fit."""

    filename = AMJUEL_FILENAMES[(species_name, reaction_kind)]
    payload = json.loads(resources.files("jax_drb.data.atomic_rates").joinpath(filename).read_text(encoding="utf-8"))
    return (
        np.asarray(payload["sigma_v_coeffs"], dtype=np.float64),
        np.asarray(payload["sigma_v_E_coeffs"], dtype=np.float64),
        float(payload["electron_heating"]),
    )


@lru_cache(maxsize=None)
def load_openadas_rate(
    species_name: str,
    reaction_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load one packaged OpenADAS rate/radiation table pair."""

    rate_filename, radiation_filename, level, electron_heating = OPENADAS_FILENAMES[(species_name, reaction_kind)]
    rate_payload = json.loads(resources.files("jax_drb.data.atomic_rates").joinpath(rate_filename).read_text(encoding="utf-8"))
    radiation_payload = json.loads(resources.files("jax_drb.data.atomic_rates").joinpath(radiation_filename).read_text(encoding="utf-8"))
    log_temperature = np.asarray(rate_payload["log_temperature"], dtype=np.float64)
    log_density = np.asarray(rate_payload["log_density"], dtype=np.float64)
    return (
        np.asarray(rate_payload["log_coeff"][level], dtype=np.float64),
        np.asarray(radiation_payload["log_coeff"][level], dtype=np.float64),
        log_temperature,
        log_density,
        float(electron_heating),
    )


def eval_amjuel_fit(temperature_ev: np.ndarray, density_m3: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate the packaged AMJUEL polynomial fit."""

    temperature = np.clip(np.asarray(temperature_ev, dtype=np.float64), 0.1, 1.0e4)
    density = np.clip(np.asarray(density_m3, dtype=np.float64), 1.0e14, 1.0e22)
    logn = np.log(density / 1.0e14)
    logt = np.log(temperature)
    result = np.zeros_like(logt, dtype=np.float64)
    for row in np.asarray(coeffs, dtype=np.float64)[::-1]:
        row_result = np.zeros_like(logn, dtype=np.float64)
        for coefficient in row[::-1]:
            row_result = row_result * logn + coefficient
        result = result * logt + row_result
    return np.exp(result) * 1.0e-6


def eval_openadas_rate(
    temperature_ev: np.ndarray,
    density_m3: np.ndarray,
    coeffs: np.ndarray,
    *,
    log_temperature: np.ndarray,
    log_density: np.ndarray,
) -> np.ndarray:
    """Evaluate the packaged OpenADAS bilinear table fit."""

    temperature = np.asarray(temperature_ev, dtype=np.float64)
    density = np.asarray(density_m3, dtype=np.float64)
    tmin = 10.0 ** float(log_temperature[0])
    tmax = 10.0 ** float(log_temperature[-1])
    nmin = 10.0 ** float(log_density[0])
    nmax = 10.0 ** float(log_density[-1])

    log10_t = np.log10(np.clip(temperature, tmin, tmax))
    log10_n = np.log10(np.clip(density, nmin, nmax))

    high_t = np.searchsorted(log_temperature, log10_t, side="left")
    high_t = np.clip(high_t, 1, log_temperature.size - 1)
    low_t = high_t - 1
    high_n = np.searchsorted(log_density, log10_n, side="left")
    high_n = np.clip(high_n, 1, log_density.size - 1)
    low_n = high_n - 1

    x = (log10_t - log_temperature[low_t]) / (log_temperature[high_t] - log_temperature[low_t])
    y = (log10_n - log_density[low_n]) / (log_density[high_n] - log_density[low_n])

    eval_log_coeff = (
        (coeffs[low_t, low_n] * (1.0 - y) + coeffs[low_t, high_n] * y) * (1.0 - x)
        + (coeffs[high_t, low_n] * (1.0 - y) + coeffs[high_t, high_n] * y) * x
    )
    return np.power(10.0, eval_log_coeff)


def hydrogen_cx_sigmav(teff_ev: np.ndarray, dataset_scalars: dict[str, float]) -> np.ndarray:
    """Evaluate the Janev-style hydrogen charge-exchange fit used by the solver."""

    lnT = np.log(np.asarray(teff_ev, dtype=np.float64))
    ln_sigma_v = np.full_like(lnT, 5.122435e-7, dtype=np.float64)
    for coefficient in (
        -1.514243e-5,
        1.440382e-4,
        -4.096807e-4,
        -4.698969e-4,
        -6.143769e-4,
        7.949876e-3,
        0.3708409,
        -18.5028,
    ):
        ln_sigma_v = ln_sigma_v * lnT + coefficient
    return np.exp(ln_sigma_v) * (1.0e-6 * dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"])
