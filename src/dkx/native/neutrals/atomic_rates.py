"""Hydrogenic atomic reaction-rate coefficients (self-contained, pure JAX).

Ionization and recombination rate coefficients ``<sigma v>(T_e, n_e)`` come from
the packaged AMJUEL double-polynomial fits; the charge-exchange coefficient
``<sigma v>(T_eff)`` comes from the AMJUEL H.2 3.1.8 one-dimensional polynomial
(inlined here, as hermes-3 does, since it carries no tabulated file). Every
routine is pure ``jax.numpy`` and therefore ``jit``/``grad``/``vmap``
transparent; the coefficient tables ship with the package under
``dkx.data.atomic_rates`` (real published AMJUEL fits, verified against the
hermes-3 database provenance), so there is no runtime external dependency.

Temperatures are in eV and densities in m^-3; rate coefficients are returned in
m^3 / s. The AMJUEL fits are clamped to their fitted range
(``T in [0.1, 1e4] eV``, ``n in [1e14, 1e22] m^-3``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import jax.numpy as jnp
import numpy as np

__all__ = [
    "rate_coefficient",
    "energy_loss_coefficient",
    "charge_exchange_rate_coefficient",
    "eval_amjuel_fit",
    "load_amjuel_coefficients",
]

# AMJUEL fit files, keyed by (species, reaction). Deuterium and tritium share the
# hydrogen fits; helium has its own.
_AMJUEL_FILENAMES = {
    ("d", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("d", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("t", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("t", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("he", "iz"): "iz_AMJUEL_H.x_2.3.9a.json",
    ("he", "rec"): "rec_AMJUEL_H.x_2.3.13a.json",
}

# AMJUEL H.2 3.1.8 charge-exchange polynomial: ln(sigma v [cm^3/s]) as a degree-8
# polynomial in ln(T_eff [eV]) (density independent), as inlined in hermes-3's
# hydrogen_charge_exchange.cxx.
_CX_CONSTANT = -18.5028
_CX_COEFFICIENTS = (
    0.3708409,
    7.949876e-3,
    -6.143769e-4,
    -4.698969e-4,
    -4.096807e-4,
    1.440382e-4,
    -1.514243e-5,
    5.122435e-7,
)


@lru_cache(maxsize=None)
def load_amjuel_coefficients(species: str, reaction: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load one AMJUEL fit: ``(sigma_v_coeffs, sigma_v_E_coeffs, electron_heating)``.

    ``sigma_v_coeffs`` and ``sigma_v_E_coeffs`` are 9x9 log-log polynomial
    coefficient matrices; ``electron_heating`` is the potential energy [eV]
    returned to the electrons (e.g. 13.6 eV on recombination).
    """

    filename = _AMJUEL_FILENAMES[(species, reaction)]
    payload = json.loads(
        resources.files("dkx.data.atomic_rates").joinpath(filename).read_text(encoding="utf-8")
    )
    return (
        np.asarray(payload["sigma_v_coeffs"], dtype=np.float64),
        np.asarray(payload["sigma_v_E_coeffs"], dtype=np.float64),
        float(payload["electron_heating"]),
    )


def eval_amjuel_fit(temperature_ev, density_m3, coefficients) -> jnp.ndarray:
    """Evaluate an AMJUEL double-polynomial fit, returning ``<sigma v>`` in m^3/s.

    ``<sigma v> = 1e-6 * exp( sum_ij c_ij (ln T)^i (ln(n/1e14))^j )`` with ``T``
    and ``n`` clamped to the fit's validity range.
    """

    temperature = jnp.clip(jnp.asarray(temperature_ev, dtype=jnp.float64), 0.1, 1.0e4)
    density = jnp.clip(jnp.asarray(density_m3, dtype=jnp.float64), 1.0e14, 1.0e22)
    log_t = jnp.log(temperature)
    log_n = jnp.log(density / 1.0e14)

    coeffs = jnp.asarray(coefficients, dtype=jnp.float64)
    result = jnp.zeros_like(log_t)
    for row in coeffs[::-1]:
        row_result = jnp.zeros_like(log_n)
        for coefficient in row[::-1]:
            row_result = row_result * log_n + coefficient
        result = result * log_t + row_result
    return jnp.exp(result) * 1.0e-6


def rate_coefficient(species: str, reaction: str, electron_temperature, electron_density) -> jnp.ndarray:
    """Ionization (``reaction="iz"``) or recombination (``"rec"``) ``<sigma v>`` [m^3/s]."""

    sigma_v_coeffs, _, _ = load_amjuel_coefficients(species, reaction)
    return eval_amjuel_fit(electron_temperature, electron_density, sigma_v_coeffs)


def energy_loss_coefficient(species: str, reaction: str, electron_temperature, electron_density) -> jnp.ndarray:
    """Electron energy-loss coefficient ``<sigma v E>`` [eV m^3/s] for the reaction.

    The recombination channel returns the ``electron_heating`` potential energy
    (13.6 eV for hydrogen) as a heating term, so the net electron energy sink is
    ``<sigma v E> - electron_heating * <sigma v>``; ionization has no potential
    add-back (``electron_heating = 0``).
    """

    sigma_v_coeffs, sigma_v_E_coeffs, electron_heating = load_amjuel_coefficients(species, reaction)
    energy_loss = eval_amjuel_fit(electron_temperature, electron_density, sigma_v_E_coeffs)
    rate = eval_amjuel_fit(electron_temperature, electron_density, sigma_v_coeffs)
    return energy_loss - float(electron_heating) * rate


def charge_exchange_rate_coefficient(effective_temperature_ev) -> jnp.ndarray:
    """Hydrogenic charge-exchange ``<sigma v>(T_eff)`` [m^3/s] (AMJUEL H.2 3.1.8)."""

    log_t = jnp.log(jnp.clip(jnp.asarray(effective_temperature_ev, dtype=jnp.float64), 0.1, 1.0e4))
    log_sigma_v = jnp.full_like(log_t, _CX_CONSTANT)
    power = jnp.ones_like(log_t)
    for coefficient in _CX_COEFFICIENTS:
        power = power * log_t
        log_sigma_v = log_sigma_v + coefficient * power
    return jnp.exp(log_sigma_v) * 1.0e-6
